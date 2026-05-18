use ash::vk;
use std::collections::{BTreeMap, HashMap, VecDeque};
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

/// Global registry mapping host-mapped pointers to their VkBuffer handles.
/// Required so that vkc_dispatch can bind the correct VkBuffers to descriptor sets
/// given only the host pointers that PyTorch knows about.
static BUFFER_REGISTRY: std::sync::LazyLock<Mutex<HashMap<usize, vk::Buffer>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

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
        let bucket = round_up_power_of_two(size);
        if let Some(queue) = self.buckets.get_mut(&bucket) {
            // Find the first buffer that is safe to reuse.
            let pos = queue.iter().position(|pb| pb.freed_at_generation < current_generation);
            if let Some(idx) = pos {
                let pb = queue.remove(idx).unwrap();
                self.total_cached_bytes -= pb.alloc.size;
                return Some(pb.alloc);
            }
        }
        None
    }

    /// Return a buffer to the pool tagged with the current flush generation.
    fn put(&mut self, alloc: BufferAlloc, generation: u64) -> Option<BufferAlloc> {
        if self.total_cached_bytes + alloc.size > MAX_TOTAL_CACHED_BYTES {
            return Some(alloc);
        }

        let bucket = round_up_power_of_two(alloc.size);
        let queue = self.buckets.entry(bucket).or_insert_with(VecDeque::new);

        if queue.len() >= MAX_CACHED_BUFFERS_PER_BUCKET {
            return Some(alloc);
        }

        self.total_cached_bytes += alloc.size;
        queue.push_back(PooledBuffer { alloc, freed_at_generation: generation });
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

fn round_up_power_of_two(size: usize) -> usize {
    if size == 0 {
        return 1;
    }
    // Minimum bucket: 256 bytes (one cache line on most GPUs).
    let min_bucket = 256;
    let s = size.max(min_bucket);
    s.next_power_of_two()
}

/// Look up the VkBuffer associated with a host-mapped pointer.
pub fn lookup_buffer(mapped_ptr: *const u8) -> Option<vk::Buffer> {
    let registry = BUFFER_REGISTRY.lock().unwrap();
    registry.get(&(mapped_ptr as usize)).copied()
}

pub fn alloc_buffer(dev: &VulkanDevice, size: usize) -> Result<BufferAlloc, VkcError> {
    // If size is 0 (e.g. empty tensors), we still allocate the minimum bucket size (256 bytes)
    // so that PyTorch has a valid host-mapped memory pointer to work with.
    let alloc_size = round_up_power_of_two(size);

    // Try the pool first. Only reuse buffers from completed flush generations.
    let current_gen = dev.flush_generation();
    {
        let mut pool = POOL.lock().unwrap();
        if let Some(alloc) = pool.get(size, current_gen) {
            // Re-register the mapped pointer (it was deregistered on free).
            let mut registry = BUFFER_REGISTRY.lock().unwrap();
            registry.insert(alloc.mapped_ptr as usize, alloc.buffer);
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

    let allocation = dev.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
        name: "Buffer",
        requirements: mem_reqs,
        location: MemoryLocation::CpuToGpu,
        linear: true,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    }).map_err(|e| VkcError::Allocation(format!("gpu-allocator err: {:?}", e)))?;

    unsafe {
        dev.device()
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            .map_err(VkcError::Vulkan)?;
    }

    let mapped_ptr = allocation.mapped_ptr()
        .ok_or_else(|| VkcError::Allocation("Failed to map memory".to_string()))?
        .as_ptr() as *mut u8;

    // Register in global registry.
    {
        let mut registry = BUFFER_REGISTRY.lock().unwrap();
        registry.insert(mapped_ptr as usize, buffer);
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
    let gen = dev.flush_generation();
    let mut pool = POOL.lock().unwrap();
    if let Some(rejected) = pool.put(alloc, gen) {
        // Pool full - actually destroy.
        destroy_buffer(dev, rejected);
    }
}

/// Destroy a buffer without pooling (used for pool eviction and shutdown).
fn destroy_buffer(dev: &VulkanDevice, alloc: BufferAlloc) {
    unsafe {
        dev.device().destroy_buffer(alloc.buffer, None);
    }
    if let Some(a) = alloc.allocation {
        dev.allocator.lock().unwrap().free(a).unwrap();
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
