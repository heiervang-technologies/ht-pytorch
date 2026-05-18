use ash::vk;

use crate::allocator;
use crate::device::VulkanDevice;
use crate::error::VkcError;

/// A compiled compute pipeline ready for dispatch.
pub struct ComputePipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
}

/// Load SPIR-V bytecode and create a compute pipeline.
///
/// The pipeline expects storage buffers bound as a single descriptor set,
/// with optional push constants.
pub fn load_compute_pipeline(
    dev: &VulkanDevice,
    spirv_bytes: &[u8],
) -> Result<ComputePipeline, VkcError> {
    if spirv_bytes.len() % 4 != 0 {
        return Err(VkcError::InvalidSpirv(spirv_bytes.len()));
    }

    let spirv: Vec<u32> = spirv_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let module_info = vk::ShaderModuleCreateInfo::default().code(&spirv);

    let shader_module = unsafe {
        dev.device()
            .create_shader_module(&module_info, None)
            .map_err(VkcError::Vulkan)?
    };

    // Create descriptor set layout with up to 8 storage buffer bindings.
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..8)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();

    let mut layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    // Use push descriptors if available (avoids descriptor pool allocation overhead).
    if dev.push_descriptor().is_some() {
        layout_info = layout_info.flags(
            vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR
        );
    }

    let descriptor_set_layout = unsafe {
        dev.device()
            .create_descriptor_set_layout(&layout_info, None)
            .map_err(VkcError::Vulkan)?
    };

    // Push constant range for up to 128 bytes.
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(128);

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&descriptor_set_layout))
        .push_constant_ranges(std::slice::from_ref(&push_range));

    let pipeline_layout = unsafe {
        dev.device()
            .create_pipeline_layout(&pipeline_layout_info, None)
            .map_err(VkcError::Vulkan)?
    };

    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(c"main");

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);

    let pipeline = unsafe {
        dev.device()
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_pipelines, err)| VkcError::Vulkan(err))?[0]
    };

    // Shader module can be destroyed after pipeline creation.
    unsafe {
        dev.device().destroy_shader_module(shader_module, None);
    }

    // Create descriptor pool.
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(16);

    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(std::slice::from_ref(&pool_size));

    let descriptor_pool = unsafe {
        dev.device()
            .create_descriptor_pool(&pool_info, None)
            .map_err(VkcError::Vulkan)?
    };

    Ok(ComputePipeline {
        pipeline,
        pipeline_layout,
        descriptor_set_layout,
        descriptor_pool,
    })
}

/// Check if all buffer pointers are registered in the allocator registry.
pub fn all_buffers_registered(buffer_ptrs: &[*const u8]) -> bool {
    buffer_ptrs.iter().all(|ptr| allocator::lookup_buffer(*ptr).is_some())
}

/// Dispatch a compute pipeline with the given buffers and push constants.
pub fn dispatch(
    dev: &VulkanDevice,
    pipeline: &ComputePipeline,
    buffer_ptrs: &[*const u8],
    group_count: [u32; 3],
    push_constants: &[u8],
) -> Result<(), VkcError> {
    // Pre-validate all buffer pointers before recording into the command buffer.
    let mut vk_buffers = Vec::with_capacity(buffer_ptrs.len());
    for ptr in buffer_ptrs {
        match allocator::lookup_buffer(*ptr) {
            Some(b) => vk_buffers.push(b),
            None => return Err(VkcError::Allocation(
                format!("no VkBuffer registered for host pointer {:p}", *ptr)
            )),
        }
    }

    dev.submit_async(|cmd, desc_pool| {
        // Build descriptor buffer infos from pre-validated VkBuffers.
        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        for vk_buffer in &vk_buffers {
            buffer_infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(*vk_buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE),
            );
        }

        let mut descriptor_writes: Vec<vk::WriteDescriptorSet> = Vec::new();
        for (i, info) in buffer_infos.iter().enumerate() {
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info)),
            );
        }

        unsafe {
            dev.device().cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            );

            if let Some(push_desc) = dev.push_descriptor() {
                // Push descriptors: write directly into command buffer, no allocation.
                push_desc.cmd_push_descriptor_set(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.pipeline_layout,
                    0,
                    &descriptor_writes,
                );
            } else {
                // Fallback: traditional descriptor set allocation.
                let alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(desc_pool)
                    .set_layouts(std::slice::from_ref(&pipeline.descriptor_set_layout));
                let descriptor_set = dev.device()
                    .allocate_descriptor_sets(&alloc_info)
                    .expect("Failed to allocate descriptor set")[0];
                // Set dst_set on writes for traditional path.
                let mut trad_writes = descriptor_writes.clone();
                for w in &mut trad_writes {
                    w.dst_set = descriptor_set;
                }
                dev.device().update_descriptor_sets(&trad_writes, &[]);
                dev.device().cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.pipeline_layout,
                    0,
                    std::slice::from_ref(&descriptor_set),
                    &[],
                );
            }

            if !push_constants.is_empty() {
                dev.device().cmd_push_constants(
                    cmd,
                    pipeline.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }

            dev.device().cmd_dispatch(
                cmd,
                group_count[0],
                group_count[1],
                group_count[2],
            );
            
            // Add a barrier so subsequent dispatches see the result of this one.
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                
            dev.device().cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&memory_barrier),
                &[],
                &[],
            );
        }
    })?;

    Ok(())
}

impl ComputePipeline {
    /// Clean up Vulkan resources. Must be called before device destruction.
    pub fn destroy(&self, dev: &VulkanDevice) {
        unsafe {
            dev.device().destroy_pipeline(self.pipeline, None);
            dev.device().destroy_pipeline_layout(self.pipeline_layout, None);
            dev.device().destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            dev.device().destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
