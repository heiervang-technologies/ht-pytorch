//! Vulkan compute backend for PyTorch tensors.
//!
//! Exposes a C API consumed by the thin C++ shim that registers ops
//! with PyTorch's PrivateUse1 dispatch key.

mod device;
mod allocator;
mod pipeline;
mod error;

use std::os::raw::c_char;
use std::ptr;
use std::slice;

pub use error::VkcError;

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
    let dev = match device::VulkanDevice::global() {
        Some(d) => d,
        None => return ptr::null_mut(),
    };
    match allocator::alloc_buffer(dev, size) {
        Ok(alloc) => {
            let mapped = alloc.mapped_ptr;
            let handle = Box::into_raw(Box::new(alloc)) as *mut VkcBuffer;
            if !out_handle.is_null() {
                unsafe { *out_handle = handle; }
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
    let dev = match device::VulkanDevice::global() {
        Some(d) => d,
        None => return,
    };
    let alloc = unsafe { Box::from_raw(handle as *mut allocator::BufferAlloc) };
    allocator::free_buffer(dev, *alloc);
}

/// Flush the buffer memory pool, freeing all cached allocations.
#[no_mangle]
pub extern "C" fn vkc_pool_flush() {
    if let Some(dev) = device::VulkanDevice::global() {
        allocator::flush_pool(dev);
    }
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
pub extern "C" fn vkc_fill_buffer(
    mapped_ptr: *const u8,
    size: usize,
    value: u32,
) -> i32 {
    let dev = match device::VulkanDevice::global() {
        Some(d) => d,
        None => return -1,
    };

    let vk_buffer = match allocator::lookup_buffer(mapped_ptr) {
        Some(b) => b,
        None => {
            log::error!("vkc_fill_buffer: no VkBuffer registered for pointer");
            return -1;
        }
    };

    match dev.submit_async(|cmd, _| unsafe {
        dev.device().cmd_fill_buffer(cmd, vk_buffer, 0, size as u64, value);
        
        let memory_barrier = ash::vk::MemoryBarrier::default()
            .src_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(ash::vk::AccessFlags::SHADER_READ | ash::vk::AccessFlags::SHADER_WRITE);
            
        dev.device().cmd_pipeline_barrier(
            cmd,
            ash::vk::PipelineStageFlags::TRANSFER,
            ash::vk::PipelineStageFlags::COMPUTE_SHADER,
            ash::vk::DependencyFlags::empty(),
            std::slice::from_ref(&memory_barrier),
            &[],
            &[],
        );
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

    let src_buffer = match allocator::lookup_buffer(src_mapped_ptr) {
        Some(b) => b,
        None => {
            log::error!("vkc_copy_buffer: no VkBuffer registered for src pointer");
            return -1;
        }
    };

    let dst_buffer = match allocator::lookup_buffer(dst_mapped_ptr) {
        Some(b) => b,
        None => {
            log::error!("vkc_copy_buffer: no VkBuffer registered for dst pointer");
            return -1;
        }
    };

    match dev.submit_async(|cmd, _| unsafe {
        let copy_region = ash::vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(size as u64);
            
        dev.device().cmd_copy_buffer(cmd, src_buffer, dst_buffer, std::slice::from_ref(&copy_region));
        
        let memory_barrier = ash::vk::MemoryBarrier::default()
            .src_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(ash::vk::AccessFlags::SHADER_READ | ash::vk::AccessFlags::SHADER_WRITE);
            
        dev.device().cmd_pipeline_barrier(
            cmd,
            ash::vk::PipelineStageFlags::TRANSFER,
            ash::vk::PipelineStageFlags::COMPUTE_SHADER | ash::vk::PipelineStageFlags::TRANSFER,
            ash::vk::DependencyFlags::empty(),
            std::slice::from_ref(&memory_barrier),
            &[],
            &[],
        );
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
    // Access via the pool's total_cached_bytes.
    // We expose this for debugging/monitoring.
    0 // TODO: expose from pool
}

/// Check if a host pointer has a registered VkBuffer. Returns 1 if yes, 0 if no.
#[no_mangle]
pub extern "C" fn vkc_has_buffer(mapped_ptr: *const u8) -> i32 {
    if mapped_ptr.is_null() {
        return 0;
    }
    if allocator::lookup_buffer(mapped_ptr).is_some() { 1 } else { 0 }
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
) -> *mut std::ffi::c_void {
    if spirv_data.is_null() || spirv_len == 0 {
        return ptr::null_mut();
    }
    let dev = match device::VulkanDevice::global() {
        Some(d) => d,
        None => return ptr::null_mut(),
    };
    let spirv = unsafe { slice::from_raw_parts(spirv_data, spirv_len) };
    match pipeline::load_compute_pipeline(dev, spirv) {
        Ok(p) => Box::into_raw(Box::new(p)) as *mut std::ffi::c_void,
        Err(e) => {
            log::error!("vkc_load_shader failed: {e}");
            ptr::null_mut()
        }
    }
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
    if pipeline_handle.is_null() {
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

    match pipeline::dispatch(
        dev, pipeline, buffer_ptrs,
        [group_count_x, group_count_y, group_count_z],
        push,
    ) {
        Ok(()) => 0,
        Err(e) => {
            log::error!("vkc_dispatch failed: {e}");
            -1
        }
    }
}
mod tests;
