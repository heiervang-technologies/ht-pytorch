use ash::vk;
use std::ffi::CStr;
use std::sync::{Mutex, OnceLock};

use crate::error::VkcError;

static GLOBAL_DEVICE: OnceLock<VulkanDevice> = OnceLock::new();

/// Auto-flush after this many dispatches to prevent descriptor pool exhaustion.
const AUTO_FLUSH_THRESHOLD: u32 = 50_000;

use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

pub struct QueueState {
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub descriptor_pool: vk::DescriptorPool,
    pub is_recording: bool,
    pub dispatches_since_flush: u32,
    /// Monotonically increasing generation counter. Incremented on each flush().
    /// Buffers freed during generation N are safe to reuse after generation N+1.
    pub flush_generation: u64,
}

pub struct VulkanDevice {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    push_descriptor: Option<ash::khr::push_descriptor::Device>,
    compute_queue: vk::Queue,
    compute_queue_family: u32,
    command_pool: vk::CommandPool,
    queue_state: Mutex<QueueState>,
    device_name: String,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub allocator: Mutex<Allocator>,
}

// ash::Device and ash::Instance are Send+Sync (they wrap raw pointers to
// Vulkan dispatchable handles, which are thread-safe by spec when externally
// synchronized -- we serialise submissions behind fences).
unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}

impl VulkanDevice {
    pub fn init_global() -> Result<(), VkcError> {
        // If already initialized, succeed silently.
        if GLOBAL_DEVICE.get().is_some() {
            return Ok(());
        }
        let dev = Self::new()?;
        let _ = GLOBAL_DEVICE.set(dev);
        Ok(())
    }

    pub fn global() -> Option<&'static VulkanDevice> {
        GLOBAL_DEVICE.get()
    }

    fn new() -> Result<Self, VkcError> {
        let entry = unsafe { ash::Entry::load().map_err(|_| VkcError::NoGpu)? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"pytorch-vulkan")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_2);

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info);

        let instance = unsafe {
            entry.create_instance(&create_info, None)
                .map_err(VkcError::Vulkan)?
        };

        let physical_device = Self::pick_physical_device(&instance)?;

        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        let compute_queue_family = Self::find_compute_queue(&instance, physical_device)?;

        let queue_priority = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_queue_family)
            .queue_priorities(&queue_priority);

        let available_extensions = unsafe {
            instance.enumerate_device_extension_properties(physical_device)
                .unwrap_or_default()
        };

        let mut extension_names: Vec<*const std::ffi::c_char> = Vec::new();
        let mut has_f16 = false;
        let mut has_atomic_float = false;
        let mut has_coop_mat = false;
        let mut has_push_descriptor = false;

        for ext in &available_extensions {
            let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
            if name.to_bytes() == b"VK_KHR_shader_float16_int8" {
                has_f16 = true;
            } else if name.to_bytes() == b"VK_EXT_shader_atomic_float" {
                has_atomic_float = true;
            } else if name.to_bytes() == b"VK_KHR_cooperative_matrix" {
                has_coop_mat = true;
            } else if name.to_bytes() == b"VK_KHR_push_descriptor" {
                has_push_descriptor = true;
            }
        }

        let mut f16_features = vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(true);
            
        let mut atomic_float_features = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
            .shader_buffer_float32_atomic_add(true)
            .shader_shared_float32_atomic_add(true);

        let mut coop_mat_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(true);

        let mut device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info));

        let mut next_chain: *mut std::ffi::c_void = std::ptr::null_mut();

        if has_coop_mat {
            extension_names.push(c"VK_KHR_cooperative_matrix".as_ptr());
            coop_mat_features.p_next = next_chain;
            next_chain = &mut coop_mat_features as *mut _ as *mut std::ffi::c_void;
        }

        if has_atomic_float {
            extension_names.push(c"VK_EXT_shader_atomic_float".as_ptr());
            atomic_float_features.p_next = next_chain;
            next_chain = &mut atomic_float_features as *mut _ as *mut std::ffi::c_void;
        }

        if has_f16 {
            extension_names.push(c"VK_KHR_shader_float16_int8".as_ptr());
            f16_features.p_next = next_chain;
            next_chain = &mut f16_features as *mut _ as *mut std::ffi::c_void;
        }

        if has_push_descriptor {
            extension_names.push(c"VK_KHR_push_descriptor".as_ptr());
        }

        if !next_chain.is_null() {
            device_create_info.p_next = next_chain;
        }

        let device_create_info = device_create_info.enabled_extension_names(&extension_names);

        let device = unsafe {
            instance.create_device(physical_device, &device_create_info, None)
                .map_err(VkcError::Vulkan)?
        };

        let push_descriptor = if has_push_descriptor {
            Some(ash::khr::push_descriptor::Device::new(&instance, &device))
        } else {
            None
        };

        let compute_queue = unsafe { device.get_device_queue(compute_queue_family, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(compute_queue_family);

        let command_pool = unsafe {
            device.create_command_pool(&pool_info, None)
                .map_err(VkcError::Vulkan)?
        };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe {
            device.allocate_command_buffers(&alloc_info)
                .map_err(VkcError::Vulkan)?[0]
        };

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe {
            device.create_fence(&fence_info, None).map_err(VkcError::Vulkan)?
        };

        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1000000); // 1M bindings

        let desc_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(100000) // 100k sets
            .pool_sizes(std::slice::from_ref(&pool_size));

        let descriptor_pool = unsafe {
            device.create_descriptor_pool(&desc_pool_info, None)
                .map_err(VkcError::Vulkan)?
        };

        let queue_state = Mutex::new(QueueState {
            command_buffer,
            fence,
            descriptor_pool,
            is_recording: false,
            dispatches_since_flush: 0,
            flush_generation: 0,
        });

        let memory_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        }).map_err(|_| VkcError::Allocation("Failed to create gpu-allocator".to_string()))?;

        log::info!("Vulkan device initialized: {device_name}");

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            push_descriptor,
            compute_queue,
            compute_queue_family,
            command_pool,
            queue_state,
            device_name,
            memory_properties,
            allocator: Mutex::new(allocator),
        })
    }

    fn pick_physical_device(instance: &ash::Instance) -> Result<vk::PhysicalDevice, VkcError> {
        let devices = unsafe {
            instance.enumerate_physical_devices().map_err(VkcError::Vulkan)?
        };
        if devices.is_empty() {
            return Err(VkcError::NoGpu);
        }

        // Prefer discrete GPU.
        for &dev in &devices {
            let props = unsafe { instance.get_physical_device_properties(dev) };
            if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                return Ok(dev);
            }
        }
        Ok(devices[0])
    }

    fn find_compute_queue(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<u32, VkcError> {
        let families = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device)
        };
        for (i, family) in families.iter().enumerate() {
            if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                return Ok(i as u32);
            }
        }
        Err(VkcError::NoComputeQueue)
    }

    // --- Accessors ---

    pub fn device(&self) -> &ash::Device { &self.device }
    pub fn physical_device(&self) -> vk::PhysicalDevice { self.physical_device }
    pub fn compute_queue(&self) -> vk::Queue { self.compute_queue }
    pub fn compute_queue_family(&self) -> u32 { self.compute_queue_family }
    pub fn command_pool(&self) -> vk::CommandPool { self.command_pool }
    pub fn device_name(&self) -> &str { &self.device_name }
    pub fn flush_generation(&self) -> u64 {
        self.queue_state.lock().unwrap().flush_generation
    }
    pub fn push_descriptor(&self) -> Option<&ash::khr::push_descriptor::Device> { self.push_descriptor.as_ref() }
    pub fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties { &self.memory_properties }
    pub fn instance(&self) -> &ash::Instance { &self.instance }

    /// Record a command into the active asynchronous command buffer.
    /// Returns the command buffer and a descriptor pool for allocating sets.
    pub fn submit_async<F>(&self, record: F) -> Result<(), VkcError>
    where
        F: FnOnce(vk::CommandBuffer, vk::DescriptorPool),
    {
        let mut state = self.queue_state.lock().unwrap();

        if !state.is_recording {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            unsafe {
                self.device.begin_command_buffer(state.command_buffer, &begin)
                    .map_err(VkcError::Vulkan)?;
            }
            state.is_recording = true;
        }

        record(state.command_buffer, state.descriptor_pool);
        state.dispatches_since_flush += 1;

        Ok(())
    }

    /// Flush the active command buffer to the GPU and wait for it to finish.
    pub fn flush(&self) -> Result<(), VkcError> {
        let mut state = self.queue_state.lock().unwrap();

        if !state.is_recording {
            return Ok(()); // Nothing to flush
        }

        unsafe {
            self.device.end_command_buffer(state.command_buffer).map_err(VkcError::Vulkan)?;
        }

        let submit = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&state.command_buffer));

        unsafe {
            self.device.queue_submit(self.compute_queue, &[submit], state.fence)
                .map_err(VkcError::Vulkan)?;
            self.device.wait_for_fences(&[state.fence], true, u64::MAX)
                .map_err(VkcError::Vulkan)?;
            self.device.reset_fences(&[state.fence])
                .map_err(VkcError::Vulkan)?;
            self.device.reset_command_buffer(state.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(VkcError::Vulkan)?;
            self.device.reset_descriptor_pool(state.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(VkcError::Vulkan)?;
        }

        state.is_recording = false;
        state.dispatches_since_flush = 0;
        state.flush_generation += 1;
        Ok(())
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            
            let state = self.queue_state.lock().unwrap();
            self.device.destroy_fence(state.fence, None);
            self.device.destroy_descriptor_pool(state.descriptor_pool, None);
            self.device.free_command_buffers(self.command_pool, &[state.command_buffer]);
            
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
