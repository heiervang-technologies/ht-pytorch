//! Vulkan compute backend for PyTorch tensors.
//!
//! Exposes a C API consumed by the thin C++ shim that registers ops
//! with PyTorch's PrivateUse1 dispatch key.

mod allocator;
mod device;
mod error;
mod pipeline;

use std::collections::HashSet;
use std::os::raw::c_char;
use std::ptr;
use std::slice;
use std::sync::{Mutex, OnceLock};

pub use error::VkcError;

static PIPELINE_HANDLES: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();
static BUFFER_HANDLES: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();

fn pipeline_handles() -> &'static Mutex<HashSet<usize>> {
    PIPELINE_HANDLES.get_or_init(|| Mutex::new(HashSet::new()))
}

fn buffer_handles() -> &'static Mutex<HashSet<usize>> {
    BUFFER_HANDLES.get_or_init(|| Mutex::new(HashSet::new()))
}

// ---------------------------------------------------------------------------
// C API: Device lifecycle
// ---------------------------------------------------------------------------

/// Initialize the Vulkan device. Must be called before any other vkc_* function.
/// Returns 0 on success, non-zero on error.
#[no_mangle]
pub extern "C" fn vkc_init() -> i32 {
    env_logger::try_init().ok();
    match device::VulkanDevice::init_global() {
        Ok(()) => 0,
        Err(e) => {
            log::error!("vkc_init failed: {e}");
            -1
        }
    }
}

/// Returns 1 if the Vulkan device is initialized and ready.
#[no_mangle]
pub extern "C" fn vkc_is_available() -> i32 {
    device::VulkanDevice::global().map_or(0, |_| 1)
}

#[no_mangle]
pub extern "C" fn vkc_shutdown() -> i32 {
    match device::VulkanDevice::shutdown_global() {
        Ok(()) => 0,
        Err(error) => {
            log::error!("vkc_shutdown failed: {error}");
            -1
        }
    }
}

/// Write the device name into `buf` (up to `buf_len` bytes).
/// Returns the number of bytes written (excluding null terminator), or -1 on error.
#[no_mangle]
pub extern "C" fn vkc_device_name(buf: *mut c_char, buf_len: usize) -> i32 {
    let dev = match device::VulkanDevice::global() {
        Some(d) => d,
        None => return -1,
    };
    let name = dev.device_name();
    let name_bytes = name.as_bytes();
    let copy_len = name_bytes.len().min(buf_len.saturating_sub(1));
    if buf.is_null() || buf_len == 0 {
        return -1;
    }
    unsafe {
        ptr::copy_nonoverlapping(name_bytes.as_ptr(), buf as *mut u8, copy_len);
        *buf.add(copy_len) = 0;
    }
    copy_len as i32
}

#[no_mangle]
pub extern "C" fn vkc_device_extensions(buf: *mut c_char, buf_len: usize) -> i32 {
    let Some(dev) = device::VulkanDevice::global() else {
        return -1;
    };
    let extensions = dev.extension_names().join("\n");
    if buf.is_null() || buf_len == 0 {
        return extensions.len() as i32;
    }
    let copy_len = extensions.len().min(buf_len.saturating_sub(1));
    unsafe {
        ptr::copy_nonoverlapping(extensions.as_ptr(), buf as *mut u8, copy_len);
        *buf.add(copy_len) = 0;
    }
    copy_len as i32
}

// ---------------------------------------------------------------------------
// C API: Buffer allocation
// ---------------------------------------------------------------------------

/// Opaque handle to a Vulkan buffer allocation.
#[repr(C)]
pub struct VkcBuffer {
    _private: [u8; 0],
}

/// Allocate a Vulkan buffer of `size` bytes. Returns a mapped host pointer
/// and an opaque handle via `out_handle`. Returns null on failure.
#[no_mangle]
pub extern "C" fn vkc_alloc(size: usize, out_handle: *mut *mut VkcBuffer) -> *mut u8 {
    if out_handle.is_null() {
        return ptr::null_mut();
    }
    unsafe {
        *out_handle = ptr::null_mut();
    }
    let dev = match device::VulkanDevice::global() {
        Some(d) => d,
        None => return ptr::null_mut(),
    };
    match allocator::alloc_buffer(&dev, size) {
        Ok(alloc) => {
            let mapped = alloc.mapped_ptr;
            let handle = Box::into_raw(Box::new(alloc)) as *mut VkcBuffer;
            buffer_handles().lock().unwrap().insert(handle as usize);
            unsafe {
                *out_handle = handle;
            }
            mapped
        }
        Err(e) => {
            log::error!("vkc_alloc failed: {e}");
            ptr::null_mut()
        }
    }
}

/// Free a buffer previously allocated with `vkc_alloc`.
#[no_mangle]
pub extern "C" fn vkc_free(handle: *mut VkcBuffer) {
    if handle.is_null() {
        return;
    }
    if !buffer_handles().lock().unwrap().remove(&(handle as usize)) {
        log::error!("vkc_free received an unknown or already freed handle");
        return;
    }
    let alloc = unsafe { Box::from_raw(handle as *mut allocator::BufferAlloc) };
    let Some(dev) = device::VulkanDevice::global() else {
        log::error!("vkc_free called after Vulkan device shutdown");
        buffer_handles().lock().unwrap().insert(handle as usize);
        let _ = Box::into_raw(alloc);
        return;
    };
    allocator::free_buffer(&dev, *alloc);
}

/// Flush the buffer memory pool, freeing all cached allocations.
#[no_mangle]
pub extern "C" fn vkc_pool_flush() -> i32 {
    if let Some(dev) = device::VulkanDevice::global() {
        if let Err(error) = dev.flush() {
            log::error!("vkc_pool_flush failed to synchronize: {error}");
            return -1;
        }
        allocator::flush_pool(&dev);
    }
    0
}

/// Flush the asynchronous compute command queue, waiting for all pending kernels.
#[no_mangle]
pub extern "C" fn vkc_flush() -> i32 {
    let dev = match device::VulkanDevice::global() {
        Some(d) => d,
        None => return -1,
    };
    match dev.flush() {
        Ok(()) => 0,
        Err(e) => {
            log::error!("vkc_flush failed: {e}");
            -1
        }
    }
}

/// Asynchronously fill a buffer with a 32-bit integer value.
#[no_mangle]
pub extern "C" fn vkc_fill_buffer(mapped_ptr: *const u8, size: usize, value: u32) -> i32 {
    let dev = match device::VulkanDevice::global() {
        Some(d) => d,
        None => return -1,
    };

    let binding = match allocator::lookup_buffer(mapped_ptr) {
        Some(binding) if size as u64 <= binding.range => binding,
        Some(_) => {
            log::error!("vkc_fill_buffer: range exceeds allocation");
            return -1;
        }
        None => {
            log::error!("vkc_fill_buffer: no VkBuffer registered for pointer");
            return -1;
        }
    };
    if size == 0 || size % 4 != 0 || binding.offset % 4 != 0 {
        log::error!("vkc_fill_buffer requires a non-zero, 4-byte-aligned range");
        return -1;
    }

    match dev.submit_async(|cmd, _| {
        unsafe {
            dev.device()
                .cmd_fill_buffer(cmd, binding.buffer, binding.offset, size as u64, value);

            let memory_barrier = ash::vk::MemoryBarrier::default()
                .src_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    ash::vk::AccessFlags::SHADER_READ | ash::vk::AccessFlags::SHADER_WRITE,
                );

            dev.device().cmd_pipeline_barrier(
                cmd,
                ash::vk::PipelineStageFlags::TRANSFER,
                ash::vk::PipelineStageFlags::COMPUTE_SHADER,
                ash::vk::DependencyFlags::empty(),
                std::slice::from_ref(&memory_barrier),
                &[],
                &[],
            );
        }
        Ok(())
    }) {
        Ok(()) => 0,
        Err(e) => {
            log::error!("vkc_fill_buffer failed: {e}");
            -1
        }
    }
}

/// Asynchronously copy data from one buffer to another.
#[no_mangle]
pub extern "C" fn vkc_copy_buffer(
    src_mapped_ptr: *const u8,
    dst_mapped_ptr: *const u8,
    size: usize,
) -> i32 {
    let dev = match device::VulkanDevice::global() {
        Some(d) => d,
        None => return -1,
    };

    let src = match allocator::lookup_buffer(src_mapped_ptr) {
        Some(binding) if size as u64 <= binding.range => binding,
        Some(_) => {
            log::error!("vkc_copy_buffer: source range exceeds allocation");
            return -1;
        }
        None => {
            log::error!("vkc_copy_buffer: no VkBuffer registered for src pointer");
            return -1;
        }
    };

    let dst = match allocator::lookup_buffer(dst_mapped_ptr) {
        Some(binding) if size as u64 <= binding.range => binding,
        Some(_) => {
            log::error!("vkc_copy_buffer: destination range exceeds allocation");
            return -1;
        }
        None => {
            log::error!("vkc_copy_buffer: no VkBuffer registered for dst pointer");
            return -1;
        }
    };
    if size == 0 || size % 4 != 0 || src.offset % 4 != 0 || dst.offset % 4 != 0 {
        log::error!("vkc_copy_buffer requires a non-zero, 4-byte-aligned range");
        return -1;
    }

    match dev.submit_async(|cmd, _| {
        unsafe {
            let copy_region = ash::vk::BufferCopy::default()
                .src_offset(src.offset)
                .dst_offset(dst.offset)
                .size(size as u64);

            dev.device().cmd_copy_buffer(
                cmd,
                src.buffer,
                dst.buffer,
                std::slice::from_ref(&copy_region),
            );

            let memory_barrier = ash::vk::MemoryBarrier::default()
                .src_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    ash::vk::AccessFlags::SHADER_READ | ash::vk::AccessFlags::SHADER_WRITE,
                );

            dev.device().cmd_pipeline_barrier(
                cmd,
                ash::vk::PipelineStageFlags::TRANSFER,
                ash::vk::PipelineStageFlags::COMPUTE_SHADER | ash::vk::PipelineStageFlags::TRANSFER,
                ash::vk::DependencyFlags::empty(),
                std::slice::from_ref(&memory_barrier),
                &[],
                &[],
            );
        }
        Ok(())
    }) {
        Ok(()) => 0,
        Err(e) => {
            log::error!("vkc_copy_buffer failed: {e}");
            -1
        }
    }
}

/// Return the number of bytes currently held in the memory pool cache.
#[no_mangle]
pub extern "C" fn vkc_pool_cached_bytes() -> usize {
    allocator::cached_bytes()
}

#[no_mangle]
pub extern "C" fn vkc_pool_active_bytes() -> usize {
    allocator::active_bytes()
}

#[no_mangle]
pub extern "C" fn vkc_pool_cached_allocations() -> usize {
    allocator::cached_allocations()
}

#[no_mangle]
pub extern "C" fn vkc_pool_active_allocations() -> usize {
    allocator::active_allocations()
}

#[no_mangle]
pub extern "C" fn vkc_active_pipelines() -> usize {
    pipeline::active_pipelines()
}

#[no_mangle]
pub extern "C" fn vkc_total_dispatches() -> u64 {
    device::VulkanDevice::global()
        .map(|dev| dev.queue_statistics().0)
        .unwrap_or_default()
}

#[no_mangle]
pub extern "C" fn vkc_pending_dispatches() -> u32 {
    device::VulkanDevice::global()
        .map(|dev| dev.queue_statistics().1)
        .unwrap_or_default()
}

#[no_mangle]
pub extern "C" fn vkc_flush_generation() -> u64 {
    device::VulkanDevice::global()
        .map(|dev| dev.queue_statistics().2)
        .unwrap_or_default()
}

#[no_mangle]
pub extern "C" fn vkc_auto_flush_threshold() -> u32 {
    device::auto_flush_threshold()
}

#[no_mangle]
pub extern "C" fn vkc_capabilities() -> u64 {
    let Some(dev) = device::VulkanDevice::global() else {
        return 0;
    };
    let capabilities = dev.capabilities();
    u64::from(capabilities.shader_float16)
        | (u64::from(capabilities.storage_buffer16_bit_access) << 1)
        | (u64::from(capabilities.shader_buffer_float32_atomic_add) << 2)
        | (u64::from(capabilities.shader_shared_float32_atomic_add) << 3)
        | (u64::from(capabilities.cooperative_matrix_nv) << 4)
        | (u64::from(capabilities.push_descriptor) << 5)
}

#[no_mangle]
pub extern "C" fn vkc_max_storage_buffer_bindings() -> u32 {
    device::VulkanDevice::global().map_or(0, |dev| {
        dev.properties()
            .limits
            .max_per_stage_descriptor_storage_buffers
            .min(dev.properties().limits.max_descriptor_set_storage_buffers)
            .min(device::MAX_STORAGE_BUFFER_BINDINGS)
    })
}

#[no_mangle]
pub extern "C" fn vkc_device_vendor_id() -> u32 {
    device::VulkanDevice::global()
        .map(|dev| dev.properties().vendor_id)
        .unwrap_or_default()
}

#[no_mangle]
pub extern "C" fn vkc_device_id() -> u32 {
    device::VulkanDevice::global()
        .map(|dev| dev.properties().device_id)
        .unwrap_or_default()
}

#[no_mangle]
pub extern "C" fn vkc_driver_version() -> u32 {
    device::VulkanDevice::global()
        .map(|dev| dev.properties().driver_version)
        .unwrap_or_default()
}

#[no_mangle]
pub extern "C" fn vkc_api_version() -> u32 {
    device::VulkanDevice::global()
        .map(|dev| dev.properties().api_version)
        .unwrap_or_default()
}

/// Check if a host pointer has a registered VkBuffer. Returns 1 if yes, 0 if no.
#[no_mangle]
pub extern "C" fn vkc_has_buffer(mapped_ptr: *const u8) -> i32 {
    if mapped_ptr.is_null() {
        return 0;
    }
    if allocator::lookup_buffer(mapped_ptr).is_some() {
        1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// C API: Compute dispatch
// ---------------------------------------------------------------------------

/// Load a SPIR-V shader from `spirv_data` (length `spirv_len` bytes) and
/// return an opaque pipeline handle. Returns null on failure.
#[no_mangle]
pub extern "C" fn vkc_load_shader(
    spirv_data: *const u8,
    spirv_len: usize,
    num_bindings: u32,
) -> *mut std::ffi::c_void {
    if spirv_data.is_null()
        || spirv_len == 0
        || spirv_len > isize::MAX as usize
        || num_bindings == 0
        || num_bindings > device::MAX_STORAGE_BUFFER_BINDINGS
    {
        return ptr::null_mut();
    }
    let dev = match device::VulkanDevice::global() {
        Some(d) => d,
        None => return ptr::null_mut(),
    };
    let spirv = unsafe { slice::from_raw_parts(spirv_data, spirv_len) };
    match pipeline::load_compute_pipeline(&dev, spirv, num_bindings) {
        Ok(pipeline) => {
            let handle = Box::into_raw(Box::new(pipeline)) as *mut std::ffi::c_void;
            pipeline_handles().lock().unwrap().insert(handle as usize);
            handle
        }
        Err(e) => {
            log::error!("vkc_load_shader failed: {e}");
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn vkc_destroy_shader(pipeline_handle: *mut std::ffi::c_void) -> i32 {
    if pipeline_handle.is_null() {
        return 0;
    }
    if !pipeline_handles()
        .lock()
        .unwrap()
        .remove(&(pipeline_handle as usize))
    {
        return -1;
    }
    let Some(dev) = device::VulkanDevice::global() else {
        pipeline_handles()
            .lock()
            .unwrap()
            .insert(pipeline_handle as usize);
        return -1;
    };
    if let Err(error) = dev.flush() {
        log::error!("vkc_destroy_shader failed to synchronize: {error}");
        pipeline_handles()
            .lock()
            .unwrap()
            .insert(pipeline_handle as usize);
        return -1;
    }
    let pipeline = unsafe { Box::from_raw(pipeline_handle as *mut pipeline::ComputePipeline) };
    pipeline.destroy(&dev);
    0
}

/// Dispatch a compute shader. `buffers` is an array of `num_buffers` mapped
/// pointers (from vkc_alloc). `push_constants` is optional push constant data.
#[no_mangle]
pub extern "C" fn vkc_dispatch(
    pipeline_handle: *mut std::ffi::c_void,
    buffers: *const *const u8,
    num_buffers: usize,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
    push_constants: *const u8,
    push_constants_len: usize,
) -> i32 {
    if pipeline_handle.is_null()
        || num_buffers > device::MAX_STORAGE_BUFFER_BINDINGS as usize
        || push_constants_len > 128
        || group_count_x == 0
        || group_count_y == 0
        || group_count_z == 0
        || (buffers.is_null() && num_buffers != 0)
        || (push_constants.is_null() && push_constants_len != 0)
    {
        return -1;
    }
    let pipeline_handles_guard = pipeline_handles().lock().unwrap();
    if !pipeline_handles_guard.contains(&(pipeline_handle as usize)) {
        return -1;
    }
    let dev = match device::VulkanDevice::global() {
        Some(d) => d,
        None => return -1,
    };

    let pipeline = unsafe { &*(pipeline_handle as *const pipeline::ComputePipeline) };
    let buffer_ptrs = if buffers.is_null() || num_buffers == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(buffers, num_buffers) }
    };
    let push = if push_constants.is_null() || push_constants_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(push_constants, push_constants_len) }
    };

    let result = pipeline::dispatch(
        &dev,
        pipeline,
        buffer_ptrs,
        [group_count_x, group_count_y, group_count_z],
        push,
    );
    drop(pipeline_handles_guard);
    match result {
        Ok(()) => 0,
        Err(e) => {
            log::error!("vkc_dispatch failed: {e}");
            -1
        }
    }
}
mod tests;
