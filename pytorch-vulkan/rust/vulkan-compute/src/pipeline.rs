use ash::vk;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::allocator;
use crate::device::VulkanDevice;
use crate::error::VkcError;

static ACTIVE_PIPELINES: AtomicUsize = AtomicUsize::new(0);

/// A compiled compute pipeline ready for dispatch.
pub struct ComputePipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    binding_count: u32,
}

/// Load SPIR-V bytecode and create a compute pipeline.
///
/// The pipeline expects storage buffers bound as a single descriptor set,
/// with optional push constants.
pub fn load_compute_pipeline(
    dev: &VulkanDevice,
    spirv_bytes: &[u8],
    binding_count: u32,
) -> Result<ComputePipeline, VkcError> {
    if spirv_bytes.len() % 4 != 0 {
        return Err(VkcError::InvalidSpirv(spirv_bytes.len()));
    }
    let limits = &dev.properties().limits;
    let device_binding_limit = limits
        .max_per_stage_descriptor_storage_buffers
        .min(limits.max_descriptor_set_storage_buffers)
        .min(crate::device::MAX_STORAGE_BUFFER_BINDINGS);
    if binding_count == 0 || binding_count > device_binding_limit {
        return Err(VkcError::Shader(format!(
            "shader requires {binding_count} storage buffers, device limit is \
             {device_binding_limit}"
        )));
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

    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..binding_count)
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
        layout_info = layout_info.flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR);
    }

    let descriptor_set_layout = match unsafe {
        dev.device()
            .create_descriptor_set_layout(&layout_info, None)
    } {
        Ok(layout) => layout,
        Err(error) => {
            unsafe {
                dev.device().destroy_shader_module(shader_module, None);
            }
            return Err(VkcError::Vulkan(error));
        }
    };

    // Push constant range for up to 128 bytes.
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(128);

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&descriptor_set_layout))
        .push_constant_ranges(std::slice::from_ref(&push_range));

    let pipeline_layout = match unsafe {
        dev.device()
            .create_pipeline_layout(&pipeline_layout_info, None)
    } {
        Ok(layout) => layout,
        Err(error) => {
            unsafe {
                dev.device()
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
                dev.device().destroy_shader_module(shader_module, None);
            }
            return Err(VkcError::Vulkan(error));
        }
    };

    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(c"main");

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);

    let pipeline = match unsafe {
        dev.device()
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    } {
        Ok(pipelines) => pipelines[0],
        Err((pipelines, error)) => {
            unsafe {
                for pipeline in pipelines {
                    dev.device().destroy_pipeline(pipeline, None);
                }
                dev.device().destroy_pipeline_layout(pipeline_layout, None);
                dev.device()
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
                dev.device().destroy_shader_module(shader_module, None);
            }
            return Err(VkcError::Vulkan(error));
        }
    };

    // Shader module can be destroyed after pipeline creation.
    unsafe {
        dev.device().destroy_shader_module(shader_module, None);
    }

    ACTIVE_PIPELINES.fetch_add(1, Ordering::Relaxed);
    Ok(ComputePipeline {
        pipeline,
        pipeline_layout,
        descriptor_set_layout,
        binding_count,
    })
}

/// Dispatch a compute pipeline with the given buffers and push constants.
pub fn dispatch(
    dev: &VulkanDevice,
    pipeline: &ComputePipeline,
    buffer_ptrs: &[*const u8],
    group_count: [u32; 3],
    push_constants: &[u8],
) -> Result<(), VkcError> {
    if buffer_ptrs.len() != pipeline.binding_count as usize {
        return Err(VkcError::Shader(format!(
            "pipeline requires {} buffers, dispatch supplied {}",
            pipeline.binding_count,
            buffer_ptrs.len()
        )));
    }
    // Pre-validate all buffer pointers before recording into the command buffer.
    let mut bindings = Vec::with_capacity(buffer_ptrs.len());
    for ptr in buffer_ptrs {
        match allocator::lookup_buffer(*ptr) {
            Some(binding) => {
                let alignment = dev
                    .properties()
                    .limits
                    .min_storage_buffer_offset_alignment
                    .max(1);
                if binding.offset % alignment != 0 {
                    return Err(VkcError::Allocation(format!(
                        "buffer offset {} is not aligned to {}",
                        binding.offset, alignment
                    )));
                }
                bindings.push(binding);
            }
            None => {
                return Err(VkcError::Allocation(format!(
                    "no VkBuffer registered for host pointer {:p}",
                    *ptr
                )))
            }
        }
    }

    dev.submit_async(|cmd, desc_pool| {
        // Build descriptor buffer infos from pre-validated VkBuffers.
        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        for binding in &bindings {
            buffer_infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(binding.buffer)
                    .offset(binding.offset)
                    .range(binding.range),
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
            dev.device()
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);

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
                let descriptor_set = dev
                    .device()
                    .allocate_descriptor_sets(&alloc_info)
                    .map_err(VkcError::Vulkan)?[0];
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

            dev.device()
                .cmd_dispatch(cmd, group_count[0], group_count[1], group_count[2]);

            // Add a barrier so subsequent dispatches see the result of this one.
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::TRANSFER_READ
                        | vk::AccessFlags::TRANSFER_WRITE,
                );

            dev.device().cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&memory_barrier),
                &[],
                &[],
            );
        }
        Ok(())
    })?;

    Ok(())
}

impl ComputePipeline {
    /// Clean up Vulkan resources. Must be called before device destruction.
    pub fn destroy(&self, dev: &VulkanDevice) {
        unsafe {
            dev.device().destroy_pipeline(self.pipeline, None);
            dev.device()
                .destroy_pipeline_layout(self.pipeline_layout, None);
            dev.device()
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
        ACTIVE_PIPELINES.fetch_sub(1, Ordering::Relaxed);
    }
}

pub fn active_pipelines() -> usize {
    ACTIVE_PIPELINES.load(Ordering::Relaxed)
}
