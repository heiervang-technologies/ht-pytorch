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

Tensor triangular_op(
    const Tensor& self_arg,
    int64_t diagonal,
    const api::ShaderInfo& shader_descriptor) {
  TORCH_CHECK(
      self_arg.dim() >= 2, "Vulkan triu/tril: input must be at least 2D");

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  const struct Block final {
    ivec4 extents;
    int32_t diagonal;
    int32_t width;
    int32_t height;
    int32_t fill0;
  } block{
      {
          safe_downcast<int32_t>(v_output.extents().data[0u]),
          safe_downcast<int32_t>(v_output.extents().data[1u]),
          safe_downcast<int32_t>(v_output.extents().data[2u]),
          0,
      },
      safe_downcast<int32_t>(diagonal),
      safe_downcast<int32_t>(self_arg.size(-1)),
      safe_downcast<int32_t>(self_arg.size(-2)),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      shader_descriptor,
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor triu(const Tensor& self, int64_t diagonal) {
  return triangular_op(self, diagonal, VK_KERNEL(triu));
}

Tensor tril(const Tensor& self, int64_t diagonal) {
  return triangular_op(self, diagonal, VK_KERNEL(tril));
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::triu"), TORCH_FN(triu));
  m.impl(TORCH_SELECTIVE_NAME("aten::tril"), TORCH_FN(tril));
}

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
