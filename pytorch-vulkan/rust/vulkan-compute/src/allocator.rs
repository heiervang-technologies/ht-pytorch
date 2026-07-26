use ash::vk;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

use crate::device::VulkanDevice;
use crate::error::VkcError;

/// A single Vulkan buffer allocation with a host-mapped pointer.
pub struct BufferAlloc {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub mapped_ptr: *mut u8,
    pub size: usize,
}

// Safety: The mapped_ptr points to Vulkan host-mapped memory which is valid
// across threads. Access is externally synchronized.
unsafe impl Send for BufferAlloc {}

#[derive(Clone, Copy)]
struct BufferRecord {
    buffer: vk::Buffer,
    size: usize,
}

#[derive(Clone, Copy)]
pub struct BufferBinding {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
    pub range: vk::DeviceSize,
}

static BUFFER_REGISTRY: std::sync::LazyLock<Mutex<BTreeMap<usize, BufferRecord>>> =
    std::sync::LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// Caching memory pool.
/// Freed buffers are held in a size-bucketed pool for reuse. Sizes are rounded
/// up to the next power of two so that slight size variations still hit cache.
/// Each bucket holds a FIFO queue (oldest freed first = most likely cold-evicted
/// from GPU cache, but simplest strategy). Pool is capped to avoid unbounded
/// memory growth.
static POOL: std::sync::LazyLock<Mutex<BufferPool>> =
    std::sync::LazyLock::new(|| Mutex::new(BufferPool::new()));

const MAX_CACHED_BUFFERS_PER_BUCKET: usize = 1024;
const MAX_TOTAL_CACHED_BYTES: usize = 8192 * 1024 * 1024; // 8 GiB

/// A pooled buffer entry tagged with the flush generation when it was freed.
/// The buffer is only safe to reuse once the flush generation has advanced
/// past freed_at_generation (meaning the GPU work that used it has completed).
struct PooledBuffer {
    alloc: BufferAlloc,
    freed_at_generation: u64,
    immediately_reusable: bool,
}

struct BufferPool {
    buckets: BTreeMap<usize, VecDeque<PooledBuffer>>,
    total_cached_bytes: usize,
}

impl BufferPool {
    fn new() -> Self {
        Self {
            buckets: BTreeMap::new(),
            total_cached_bytes: 0,
        }
    }

    /// Try to get a cached buffer that is safe to reuse at the given generation.
    /// Only returns buffers whose GPU work has completed (freed_at < current).
    fn get(&mut self, size: usize, current_generation: u64) -> Option<BufferAlloc> {
        let bucket = round_up_power_of_two(size)?;
        if let Some(queue) = self.buckets.get_mut(&bucket) {
            // Find the first buffer that is safe to reuse.
            let pos = queue.iter().position(|pb| {
                pb.immediately_reusable
                    || pb.freed_at_generation < current_generation
            });
            if let Some(idx) = pos {
                let pb = queue.remove(idx).unwrap();
                self.total_cached_bytes -= pb.alloc.size;
                return Some(pb.alloc);
            }
        }
        None
    }

    /// Return a buffer to the pool tagged with the current flush generation.
    fn put(
        &mut self,
        alloc: BufferAlloc,
        generation: u64,
        immediately_reusable: bool,
    ) -> Option<BufferAlloc> {
        if alloc.size > MAX_TOTAL_CACHED_BYTES - self.total_cached_bytes {
            return Some(alloc);
        }

        let Some(bucket) = round_up_power_of_two(alloc.size) else {
            return Some(alloc);
        };
        let queue = self.buckets.entry(bucket).or_insert_with(VecDeque::new);

        if queue.len() >= MAX_CACHED_BUFFERS_PER_BUCKET {
            return Some(alloc);
        }

        self.total_cached_bytes += alloc.size;
        queue.push_back(PooledBuffer {
            alloc,
            freed_at_generation: generation,
            immediately_reusable,
        });
        None
    }

    /// Destroy all cached buffers.
    fn drain(&mut self) -> Vec<BufferAlloc> {
        let mut all = Vec::new();
        for (_, queue) in self.buckets.iter_mut() {
            all.extend(queue.drain(..).map(|pb| pb.alloc));
        }
        self.total_cached_bytes = 0;
        self.buckets.clear();
        all
    }
}

pub(crate) fn round_up_power_of_two(size: usize) -> Option<usize> {
    size.max(256).checked_next_power_of_two()
}

pub fn lookup_buffer(mapped_ptr: *const u8) -> Option<BufferBinding> {
    if mapped_ptr.is_null() {
        return None;
    }
    let address = mapped_ptr as usize;
    let registry = BUFFER_REGISTRY.lock().unwrap();
    let (base, record) = registry.range(..=address).next_back()?;
    let offset = address.checked_sub(*base)?;
    if offset >= record.size {
        return None;
    }
    Some(BufferBinding {
        buffer: record.buffer,
        offset: offset as vk::DeviceSize,
        range: (record.size - offset) as vk::DeviceSize,
    })
}

pub fn active_allocations() -> usize {
    BUFFER_REGISTRY.lock().unwrap().len()
}

pub fn active_bytes() -> usize {
    BUFFER_REGISTRY
        .lock()
        .unwrap()
        .values()
        .map(|record| record.size)
        .sum()
}

pub fn cached_allocations() -> usize {
    POOL
        .lock()
        .unwrap()
        .buckets
        .values()
        .map(VecDeque::len)
        .sum()
}

pub fn cached_bytes() -> usize {
    POOL.lock().unwrap().total_cached_bytes
}

pub fn alloc_buffer(dev: &VulkanDevice, size: usize) -> Result<BufferAlloc, VkcError> {
    if size > isize::MAX as usize {
        return Err(VkcError::Allocation(format!(
            "requested buffer size {size} exceeds the host address range"
        )));
    }
    // If size is 0 (e.g. empty tensors), we still allocate the minimum bucket size (256 bytes)
    // so that PyTorch has a valid host-mapped memory pointer to work with.
    let alloc_size = round_up_power_of_two(size).ok_or_else(|| {
        VkcError::Allocation(format!(
            "requested buffer size {size} exceeds the allocator limit"
        ))
    })?;
    if alloc_size > isize::MAX as usize {
        return Err(VkcError::Allocation(format!(
            "rounded buffer size {alloc_size} exceeds the host address range"
        )));
    }

    // Try the pool first. Only reuse buffers from completed flush generations.
    let current_gen = dev.flush_generation();
    {
        let mut pool = POOL.lock().unwrap();
        if let Some(alloc) = pool.get(size, current_gen) {
            // Re-register the mapped pointer (it was deregistered on free).
            let mut registry = BUFFER_REGISTRY.lock().unwrap();
            registry.insert(
                alloc.mapped_ptr as usize,
                BufferRecord {
                    buffer: alloc.buffer,
                    size: alloc.size,
                },
            );
            log::debug!("Pool hit: reusing {}B buffer for {}B request (gen safe)", alloc.size, size);
            return Ok(alloc);
        }
    }

    // Pool miss - allocate fresh with gpu-allocator.
    let buf_info = vk::BufferCreateInfo::default()
        .size(alloc_size as u64)
        .usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        dev.device().create_buffer(&buf_info, None).map_err(VkcError::Vulkan)?
    };

    let mem_reqs = unsafe { dev.device().get_buffer_memory_requirements(buffer) };

    let allocation = match dev
        .allocator
        .lock()
        .unwrap()
        .as_mut()
        .expect("Vulkan allocator is unavailable")
        .allocate(&AllocationCreateDesc {
            name: "Buffer",
            requirements: mem_reqs,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
        Ok(allocation) => allocation,
        Err(error) => {
            unsafe {
                dev.device().destroy_buffer(buffer, None);
            }
            return Err(VkcError::Allocation(format!(
                "gpu-allocator error: {error:?}"
            )));
        }
    };

    if let Err(error) = unsafe {
        dev.device()
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
    } {
        unsafe {
            dev.device().destroy_buffer(buffer, None);
        }
        dev.allocator
            .lock()
            .unwrap()
            .as_mut()
            .expect("Vulkan allocator is unavailable")
            .free(allocation)
            .ok();
        return Err(VkcError::Vulkan(error));
    }

    let Some(mapped_ptr) = allocation.mapped_ptr() else {
        unsafe {
            dev.device().destroy_buffer(buffer, None);
        }
        dev.allocator
            .lock()
            .unwrap()
            .as_mut()
            .expect("Vulkan allocator is unavailable")
            .free(allocation)
            .ok();
        return Err(VkcError::Allocation(
            "failed to map buffer memory".to_string(),
        ));
    };
    let mapped_ptr = mapped_ptr.as_ptr() as *mut u8;

    // Register in global registry.
    {
        let mut registry = BUFFER_REGISTRY.lock().unwrap();
        registry.insert(
            mapped_ptr as usize,
            BufferRecord {
                buffer,
                size: alloc_size,
            },
        );
    }

    log::debug!("Fresh allocation: {}B (requested {}B)", alloc_size, size);

    Ok(BufferAlloc {
        buffer,
        allocation: Some(allocation),
        mapped_ptr,
        size: alloc_size,
    })
}

pub fn free_buffer(dev: &VulkanDevice, alloc: BufferAlloc) {
    // Remove from pointer registry.
    {
        let mut registry = BUFFER_REGISTRY.lock().unwrap();
        registry.remove(&(alloc.mapped_ptr as usize));
    }

    // Tag with current flush generation so pool knows when it's safe to reuse.
    let (generation, pending_work) = dev.queue_snapshot();
    let mut pool = POOL.lock().unwrap();
    if let Some(rejected) = pool.put(alloc, generation, !pending_work) {
        drop(pool);
        if let Err(error) = dev.flush() {
            log::error!("failed to synchronize before destroying a buffer: {error}");
            std::mem::forget(rejected);
            return;
        }
        destroy_buffer(dev, rejected);
    }
}

/// Destroy a buffer without pooling (used for pool eviction and shutdown).
fn destroy_buffer(dev: &VulkanDevice, alloc: BufferAlloc) {
    unsafe {
        dev.device().destroy_buffer(alloc.buffer, None);
    }
    if let Some(a) = alloc.allocation {
        dev.allocator
            .lock()
            .unwrap()
            .as_mut()
            .expect("Vulkan allocator is unavailable")
            .free(a)
            .unwrap();
    }
}

/// Flush the entire pool, destroying all cached buffers.
/// Called on shutdown or when memory pressure is high.
pub fn flush_pool(dev: &VulkanDevice) {
    let mut pool = POOL.lock().unwrap();
    let all = pool.drain();
    for alloc in all {
        destroy_buffer(dev, alloc);
    }
}
