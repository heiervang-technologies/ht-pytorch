use ash::vk;
use std::cell::Cell;
use std::ffi::CStr;
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use crate::error::VkcError;

static GLOBAL_DEVICE: LazyLock<RwLock<Option<Arc<VulkanDevice>>>> =
    LazyLock::new(|| RwLock::new(None));

const DEFAULT_AUTO_FLUSH_THRESHOLD: u32 = 2_048;
const MAX_AUTO_FLUSH_THRESHOLD: u32 = 65_536;
pub const MAX_STORAGE_BUFFER_BINDINGS: u32 = 16;
static AUTO_FLUSH_THRESHOLD: LazyLock<u32> = LazyLock::new(|| {
    std::env::var("PYTORCH_VULKAN_MAX_PENDING_DISPATCHES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
        .map(|value| value.min(MAX_AUTO_FLUSH_THRESHOLD))
        .unwrap_or(DEFAULT_AUTO_FLUSH_THRESHOLD)
});

pub fn auto_flush_threshold() -> u32 {
    *AUTO_FLUSH_THRESHOLD
}

use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

#[derive(Clone, Copy)]
pub struct DeviceCapabilities {
    pub shader_float16: bool,
    pub storage_buffer16_bit_access: bool,
    pub shader_buffer_float32_atomic_add: bool,
    pub shader_shared_float32_atomic_add: bool,
    pub cooperative_matrix_nv: bool,
    pub push_descriptor: bool,
}

struct InstanceInitGuard {
    instance: Option<ash::Instance>,
}

impl InstanceInitGuard {
    fn new(instance: ash::Instance) -> Self {
        Self {
            instance: Some(instance),
        }
    }

    fn get(&self) -> &ash::Instance {
        self.instance.as_ref().unwrap()
    }

    fn take(&mut self) -> ash::Instance {
        self.instance.take().unwrap()
    }
}

impl Drop for InstanceInitGuard {
    fn drop(&mut self) {
        if let Some(instance) = self.instance.take() {
            unsafe {
                instance.destroy_instance(None);
            }
        }
    }
}

struct DeviceInitGuard {
    device: Option<ash::Device>,
    command_pool: Cell<vk::CommandPool>,
    command_buffer: Cell<vk::CommandBuffer>,
    fence: Cell<vk::Fence>,
    descriptor_pool: Cell<vk::DescriptorPool>,
}

impl DeviceInitGuard {
    fn new(device: ash::Device) -> Self {
        Self {
            device: Some(device),
            command_pool: Cell::new(vk::CommandPool::null()),
            command_buffer: Cell::new(vk::CommandBuffer::null()),
            fence: Cell::new(vk::Fence::null()),
            descriptor_pool: Cell::new(vk::DescriptorPool::null()),
        }
    }

    fn get(&self) -> &ash::Device {
        self.device.as_ref().unwrap()
    }

    fn take(&mut self) -> ash::Device {
        self.command_pool.set(vk::CommandPool::null());
        self.command_buffer.set(vk::CommandBuffer::null());
        self.fence.set(vk::Fence::null());
        self.descriptor_pool.set(vk::DescriptorPool::null());
        self.device.take().unwrap()
    }
}

impl Drop for DeviceInitGuard {
    fn drop(&mut self) {
        let Some(device) = self.device.take() else {
            return;
        };
        unsafe {
            if self.fence.get() != vk::Fence::null() {
                device.destroy_fence(self.fence.get(), None);
            }
            if self.descriptor_pool.get() != vk::DescriptorPool::null() {
                device.destroy_descriptor_pool(
                    self.descriptor_pool.get(),
                    None,
                );
            }
            if self.command_buffer.get() != vk::CommandBuffer::null()
                && self.command_pool.get() != vk::CommandPool::null()
            {
                device.free_command_buffers(
                    self.command_pool.get(),
                    &[self.command_buffer.get()],
                );
            }
            if self.command_pool.get() != vk::CommandPool::null() {
                device.destroy_command_pool(self.command_pool.get(), None);
            }
            device.destroy_device(None);
        }
    }
}

pub struct QueueState {
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub descriptor_pool: vk::DescriptorPool,
    pub is_recording: bool,
    pub dispatches_since_flush: u32,
    pub total_dispatches: u64,
    /// Monotonically increasing generation counter. Incremented on each flush().
    /// Buffers freed during generation N are safe to reuse after generation N+1.
    pub flush_generation: u64,
}

pub struct VulkanDevice {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    push_descriptor: Option<ash::khr::push_descriptor::Device>,
    compute_queue: vk::Queue,
    command_pool: vk::CommandPool,
    queue_state: Mutex<QueueState>,
    device_name: String,
    extension_names: Vec<String>,
    properties: vk::PhysicalDeviceProperties,
    capabilities: DeviceCapabilities,
    pub allocator: Mutex<Option<Allocator>>,
}

// ash::Device and ash::Instance are Send+Sync (they wrap raw pointers to
// Vulkan dispatchable handles, which are thread-safe by spec when externally
// synchronized -- we serialise submissions behind fences).
unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}

impl VulkanDevice {
    pub fn init_global() -> Result<(), VkcError> {
        let mut global = GLOBAL_DEVICE.write().unwrap();
        if global.is_some() {
            return Ok(());
        }
        *global = Some(Arc::new(Self::new()?));
        Ok(())
    }

    pub fn global() -> Option<Arc<VulkanDevice>> {
        GLOBAL_DEVICE.read().unwrap().clone()
    }

    pub fn shutdown_global() -> Result<(), VkcError> {
        let mut global = GLOBAL_DEVICE.write().unwrap();
        let Some(device) = global.as_ref() else {
            return Ok(());
        };
        device.flush()?;
        if crate::allocator::active_allocations() != 0 {
            return Err(VkcError::Allocation(
                "cannot shut down while Vulkan tensor allocations are live".to_string(),
            ));
        }
        if crate::pipeline::active_pipelines() != 0 {
            return Err(VkcError::Allocation(
                "cannot shut down while Vulkan shader pipelines are live".to_string(),
            ));
        }
        if Arc::strong_count(device) != 1 {
            return Err(VkcError::Allocation(
                "cannot shut down while Vulkan runtime calls are active".to_string(),
            ));
        }
        crate::allocator::flush_pool(device);
        *global = None;
        Ok(())
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
        let mut instance_guard = InstanceInitGuard::new(instance);
        let instance = instance_guard.get();

        let physical_device = Self::pick_physical_device(instance)?;

        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        let compute_queue_family =
            Self::find_compute_queue(instance, physical_device)?;

        let queue_priority = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_queue_family)
            .queue_priorities(&queue_priority);

        let available_extensions = unsafe {
            instance.enumerate_device_extension_properties(physical_device)
                .unwrap_or_default()
        };
        let mut extension_names_available = available_extensions
            .iter()
            .map(|extension| unsafe {
                CStr::from_ptr(extension.extension_name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            })
            .collect::<Vec<_>>();
        extension_names_available.sort();

        let mut extension_names: Vec<*const std::ffi::c_char> = Vec::new();
        let mut has_f16_extension = false;
        let mut has_16bit_storage_extension = false;
        let mut has_atomic_float_extension = false;
        let mut has_coop_mat_nv_extension = false;
        let mut has_push_descriptor_extension = false;

        for ext in &available_extensions {
            let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
            if name.to_bytes() == b"VK_KHR_shader_float16_int8" {
                has_f16_extension = true;
            } else if name.to_bytes() == b"VK_KHR_16bit_storage" {
                has_16bit_storage_extension = true;
            } else if name.to_bytes() == b"VK_EXT_shader_atomic_float" {
                has_atomic_float_extension = true;
            } else if name.to_bytes() == b"VK_NV_cooperative_matrix" {
                has_coop_mat_nv_extension = true;
            } else if name.to_bytes() == b"VK_KHR_push_descriptor" {
                has_push_descriptor_extension = true;
            }
        }
        let mut push_descriptor_properties =
            vk::PhysicalDevicePushDescriptorPropertiesKHR::default();
        let mut subgroup_properties =
            vk::PhysicalDeviceSubgroupProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default();
        properties.p_next = &mut subgroup_properties as *mut _ as *mut _;
        if has_push_descriptor_extension {
            push_descriptor_properties.p_next = properties.p_next;
            properties.p_next =
                &mut push_descriptor_properties as *mut _ as *mut _;
        }
        unsafe {
            instance.get_physical_device_properties2(
                physical_device,
                &mut properties,
            );
        }
        let storage_buffer_binding_limit = props
            .limits
            .max_per_stage_descriptor_storage_buffers
            .min(props.limits.max_descriptor_set_storage_buffers)
            .min(MAX_STORAGE_BUFFER_BINDINGS);
        let supports_push_descriptors = has_push_descriptor_extension
            && push_descriptor_properties.max_push_descriptors
                >= storage_buffer_binding_limit;

        let mut f16_features = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut storage_16bit_features =
            vk::PhysicalDevice16BitStorageFeatures::default();
        let mut atomic_float_features =
            vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default();
        let mut coop_mat_nv_features =
            vk::PhysicalDeviceCooperativeMatrixFeaturesNV::default();
        let api_has_f16 = vk::api_version_major(props.api_version) > 1
            || (vk::api_version_major(props.api_version) == 1
                && vk::api_version_minor(props.api_version) >= 2);
        let api_has_16bit_storage = vk::api_version_major(props.api_version) > 1
            || (vk::api_version_major(props.api_version) == 1
                && vk::api_version_minor(props.api_version) >= 1);
        let mut feature_query = vk::PhysicalDeviceFeatures2::default();
        let mut query_chain: *mut std::ffi::c_void = std::ptr::null_mut();
        if has_coop_mat_nv_extension {
            coop_mat_nv_features.p_next = query_chain;
            query_chain = &mut coop_mat_nv_features as *mut _ as *mut _;
        }
        if has_atomic_float_extension {
            atomic_float_features.p_next = query_chain;
            query_chain = &mut atomic_float_features as *mut _ as *mut _;
        }
        if api_has_f16 || has_f16_extension {
            f16_features.p_next = query_chain;
            query_chain = &mut f16_features as *mut _ as *mut _;
        }
        if api_has_16bit_storage || has_16bit_storage_extension {
            storage_16bit_features.p_next = query_chain;
            query_chain = &mut storage_16bit_features as *mut _ as *mut _;
        }
        feature_query.p_next = query_chain;
        unsafe {
            instance.get_physical_device_features2(physical_device, &mut feature_query);
        }

        let has_f16 =
            (api_has_f16 || has_f16_extension) && f16_features.shader_float16 == vk::TRUE;
        let has_16bit_storage = (api_has_16bit_storage
            || has_16bit_storage_extension)
            && storage_16bit_features.storage_buffer16_bit_access == vk::TRUE;
        let has_buffer_atomic_add = has_atomic_float_extension
            && atomic_float_features.shader_buffer_float32_atomic_add == vk::TRUE;
        let has_shared_atomic_add = has_atomic_float_extension
            && atomic_float_features.shader_shared_float32_atomic_add == vk::TRUE;
        let has_atomic_float = has_buffer_atomic_add || has_shared_atomic_add;
        let has_coop_mat_nv = has_coop_mat_nv_extension
            && coop_mat_nv_features.cooperative_matrix == vk::TRUE
            && subgroup_properties.subgroup_size == 32
            && subgroup_properties
                .supported_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
            && Self::supports_cooperative_matrix_tile(
                &entry,
                instance,
                physical_device,
            );
        let capabilities = DeviceCapabilities {
            shader_float16: has_f16,
            storage_buffer16_bit_access: has_16bit_storage,
            shader_buffer_float32_atomic_add: has_buffer_atomic_add,
            shader_shared_float32_atomic_add: has_shared_atomic_add,
            cooperative_matrix_nv: has_coop_mat_nv,
            push_descriptor: supports_push_descriptors,
        };

        let mut enabled_f16_features =
            vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(has_f16);
        let mut enabled_storage_16bit_features =
            vk::PhysicalDevice16BitStorageFeatures::default()
                .storage_buffer16_bit_access(has_16bit_storage);
        let mut enabled_atomic_float_features =
            vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
                .shader_buffer_float32_atomic_add(has_buffer_atomic_add)
                .shader_shared_float32_atomic_add(has_shared_atomic_add);
        let mut enabled_coop_mat_nv_features =
            vk::PhysicalDeviceCooperativeMatrixFeaturesNV::default()
                .cooperative_matrix(has_coop_mat_nv);

        let mut device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info));

        let mut next_chain: *mut std::ffi::c_void = std::ptr::null_mut();

        if has_coop_mat_nv {
            extension_names.push(c"VK_NV_cooperative_matrix".as_ptr());
            enabled_coop_mat_nv_features.p_next = next_chain;
            next_chain =
                &mut enabled_coop_mat_nv_features as *mut _ as *mut std::ffi::c_void;
        }

        if has_atomic_float {
            extension_names.push(c"VK_EXT_shader_atomic_float".as_ptr());
            enabled_atomic_float_features.p_next = next_chain;
            next_chain =
                &mut enabled_atomic_float_features as *mut _ as *mut std::ffi::c_void;
        }

        if has_f16 {
            if has_f16_extension {
                extension_names.push(c"VK_KHR_shader_float16_int8".as_ptr());
            }
            enabled_f16_features.p_next = next_chain;
            next_chain = &mut enabled_f16_features as *mut _ as *mut std::ffi::c_void;
        }

        if has_16bit_storage {
            if has_16bit_storage_extension {
                extension_names.push(c"VK_KHR_16bit_storage".as_ptr());
            }
            enabled_storage_16bit_features.p_next = next_chain;
            next_chain =
                &mut enabled_storage_16bit_features as *mut _ as *mut std::ffi::c_void;
        }

        if supports_push_descriptors {
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
        let mut device_guard = DeviceInitGuard::new(device);
        let device = device_guard.get();

        let push_descriptor = if supports_push_descriptors {
            Some(ash::khr::push_descriptor::Device::new(instance, device))
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
        device_guard.command_pool.set(command_pool);

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe {
            device.allocate_command_buffers(&alloc_info)
                .map_err(VkcError::Vulkan)?[0]
        };
        device_guard.command_buffer.set(command_buffer);

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe {
            device.create_fence(&fence_info, None).map_err(VkcError::Vulkan)?
        };
        device_guard.fence.set(fence);

        let max_descriptor_sets = if push_descriptor.is_some() {
            1
        } else {
            *AUTO_FLUSH_THRESHOLD
        };
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(
                max_descriptor_sets
                    .saturating_mul(MAX_STORAGE_BUFFER_BINDINGS),
            );

        let desc_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_descriptor_sets)
            .pool_sizes(std::slice::from_ref(&pool_size));

        let descriptor_pool = unsafe {
            device.create_descriptor_pool(&desc_pool_info, None)
                .map_err(VkcError::Vulkan)?
        };
        device_guard.descriptor_pool.set(descriptor_pool);

        let queue_state = Mutex::new(QueueState {
            command_buffer,
            fence,
            descriptor_pool,
            is_recording: false,
            dispatches_since_flush: 0,
            total_dispatches: 0,
            flush_generation: 0,
        });

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        }).map_err(|_| VkcError::Allocation("Failed to create gpu-allocator".to_string()))?;

        log::info!("Vulkan device initialized: {device_name}");
        let device = device_guard.take();
        let instance = instance_guard.take();

        Ok(Self {
            _entry: entry,
            instance,
            device,
            push_descriptor,
            compute_queue,
            command_pool,
            queue_state,
            device_name,
            extension_names: extension_names_available,
            properties: props,
            capabilities,
            allocator: Mutex::new(Some(allocator)),
        })
    }

    fn pick_physical_device(instance: &ash::Instance) -> Result<vk::PhysicalDevice, VkcError> {
        let devices = unsafe {
            instance.enumerate_physical_devices().map_err(VkcError::Vulkan)?
        };
        if devices.is_empty() {
            return Err(VkcError::NoGpu);
        }

        let mut fallback = None;
        let mut found_compute_device = false;
        for &dev in &devices {
            if Self::find_compute_queue(instance, dev).is_err() {
                continue;
            }
            found_compute_device = true;
            let props = unsafe { instance.get_physical_device_properties(dev) };
            let supports_vulkan_1_2 =
                vk::api_version_major(props.api_version) > 1
                    || (vk::api_version_major(props.api_version) == 1
                        && vk::api_version_minor(props.api_version) >= 2);
            if !supports_vulkan_1_2 {
                continue;
            }
            if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                return Ok(dev);
            }
            fallback.get_or_insert(dev);
        }
        fallback.ok_or(if found_compute_device {
            VkcError::UnsupportedVulkanVersion
        } else {
            VkcError::NoComputeQueue
        })
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

    fn supports_cooperative_matrix_tile(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> bool {
        let extension =
            ash::nv::cooperative_matrix::Instance::new(entry, instance);
        let query = extension
            .fp()
            .get_physical_device_cooperative_matrix_properties_nv;
        let mut property_count = 0;
        if unsafe {
            query(
                physical_device,
                &mut property_count,
                std::ptr::null_mut(),
            )
        } != vk::Result::SUCCESS
            || property_count == 0
        {
            return false;
        }
        let mut properties = vec![
            vk::CooperativeMatrixPropertiesNV::default();
            property_count as usize
        ];
        if unsafe {
            query(
                physical_device,
                &mut property_count,
                properties.as_mut_ptr(),
            )
        } != vk::Result::SUCCESS
            || property_count as usize > properties.len()
        {
            return false;
        }
        properties[..property_count as usize].iter().any(|property| {
            property.m_size == 16
                && property.n_size == 16
                && property.k_size == 16
                && property.a_type == vk::ComponentTypeNV::FLOAT16
                && property.b_type == vk::ComponentTypeNV::FLOAT16
                && property.c_type == vk::ComponentTypeNV::FLOAT16
                && property.d_type == vk::ComponentTypeNV::FLOAT16
                && property.scope == vk::ScopeNV::SUBGROUP
        })
    }

    // --- Accessors ---

    pub fn device(&self) -> &ash::Device { &self.device }
    pub fn device_name(&self) -> &str { &self.device_name }
    pub fn extension_names(&self) -> &[String] { &self.extension_names }
    pub fn properties(&self) -> &vk::PhysicalDeviceProperties { &self.properties }
    pub fn capabilities(&self) -> DeviceCapabilities { self.capabilities }
    pub fn flush_generation(&self) -> u64 {
        self.queue_state.lock().unwrap().flush_generation
    }
    pub fn queue_snapshot(&self) -> (u64, bool) {
        let state = self.queue_state.lock().unwrap();
        (state.flush_generation, state.is_recording)
    }
    pub fn queue_statistics(&self) -> (u64, u32, u64) {
        let state = self.queue_state.lock().unwrap();
        (
            state.total_dispatches,
            state.dispatches_since_flush,
            state.flush_generation,
        )
    }
    pub fn push_descriptor(&self) -> Option<&ash::khr::push_descriptor::Device> { self.push_descriptor.as_ref() }

    /// Record a command into the active asynchronous command buffer.
    /// Returns the command buffer and a descriptor pool for allocating sets.
    pub fn submit_async<F>(&self, record: F) -> Result<(), VkcError>
    where
        F: FnOnce(
            vk::CommandBuffer,
            vk::DescriptorPool,
        ) -> Result<(), VkcError>,
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

        if let Err(error) = record(state.command_buffer, state.descriptor_pool) {
            let reset_result = unsafe {
                self.device
                    .reset_command_buffer(
                        state.command_buffer,
                        vk::CommandBufferResetFlags::empty(),
                    )
                    .map_err(VkcError::Vulkan)
            };
            let descriptor_reset_result = unsafe {
                self.device
                    .reset_descriptor_pool(
                        state.descriptor_pool,
                        vk::DescriptorPoolResetFlags::empty(),
                    )
                    .map_err(VkcError::Vulkan)
            };
            state.is_recording = false;
            state.dispatches_since_flush = 0;
            reset_result?;
            descriptor_reset_result?;
            return Err(error);
        }
        state.dispatches_since_flush += 1;
        state.total_dispatches = state.total_dispatches.saturating_add(1);
        let should_flush = state.dispatches_since_flush >= *AUTO_FLUSH_THRESHOLD;
        drop(state);

        if should_flush {
            self.flush()
        } else {
            Ok(())
        }
    }

    /// Flush the active command buffer to the GPU and wait for it to finish.
    pub fn flush(&self) -> Result<(), VkcError> {
        let mut state = self.queue_state.lock().unwrap();

        if !state.is_recording {
            return Ok(()); // Nothing to flush
        }

        let command_buffer = state.command_buffer;
        let fence = state.fence;
        let descriptor_pool = state.descriptor_pool;
        let result = unsafe {
            self.device
                .end_command_buffer(command_buffer)
                .map_err(VkcError::Vulkan)
                .and_then(|()| {
                    let submit = vk::SubmitInfo::default()
                        .command_buffers(std::slice::from_ref(&command_buffer));
                    self.device
                        .queue_submit(self.compute_queue, &[submit], fence)
                        .map_err(VkcError::Vulkan)
                })
                .and_then(|()| {
                    self.device
                        .wait_for_fences(&[fence], true, u64::MAX)
                        .map_err(VkcError::Vulkan)
                })
                .and_then(|()| {
                    self.device
                        .reset_fences(&[fence])
                        .map_err(VkcError::Vulkan)
                })
                .and_then(|()| {
                    self.device
                        .reset_command_buffer(
                            command_buffer,
                            vk::CommandBufferResetFlags::empty(),
                        )
                        .map_err(VkcError::Vulkan)
                })
                .and_then(|()| {
                    self.device
                        .reset_descriptor_pool(
                            descriptor_pool,
                            vk::DescriptorPoolResetFlags::empty(),
                        )
                        .map_err(VkcError::Vulkan)
                })
        };

        state.is_recording = false;
        state.dispatches_since_flush = 0;
        if let Err(error) = result {
            unsafe {
                if let Err(cleanup_error) = self.device.device_wait_idle() {
                    log::error!(
                        "failed to idle Vulkan device after queue error: {cleanup_error:?}"
                    );
                }
                self.device.reset_fences(&[fence]).ok();
                self.device
                    .reset_command_buffer(
                        command_buffer,
                        vk::CommandBufferResetFlags::empty(),
                    )
                    .ok();
                self.device
                    .reset_descriptor_pool(
                        descriptor_pool,
                        vk::DescriptorPoolResetFlags::empty(),
                    )
                    .ok();
            }
            return Err(error);
        }
        state.flush_generation = state.flush_generation.saturating_add(1);
        Ok(())
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
        }
        crate::allocator::flush_pool(self);
        self.allocator.get_mut().unwrap().take();
        unsafe {
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
