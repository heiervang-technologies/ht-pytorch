#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor where_self(
    const Tensor& condition_arg,
    const Tensor& self_arg,
    const Tensor& other_arg) {
  api::Context* const context = api::context();

  // Convert bool condition to float on CPU, then transfer to Vulkan
  Tensor cond_float;
  if (condition_arg.scalar_type() == at::kBool) {
    Tensor cond_cpu = condition_arg.is_vulkan() ? condition_arg.cpu() : condition_arg;
    cond_float = cond_cpu.to(at::kFloat).vulkan();
  } else {
    cond_float = condition_arg.is_vulkan() ? condition_arg : condition_arg.vulkan();
  }
  const Tensor condition = cond_float;
  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();

  const vTensor& v_condition = convert(condition);
  const vTensor& v_self = convert(self);
  const vTensor& v_other = convert(other);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  const struct Block final {
    uvec3 extents;
    int32_t fill0;
  } block{
      v_output.extents(),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(where),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_condition.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::where.self"), TORCH_FN(where_self));
}

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
