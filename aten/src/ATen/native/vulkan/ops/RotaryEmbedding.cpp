#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor rotary_embedding(
    const Tensor& input_arg,
    const Tensor& cos_arg,
    const Tensor& sin_arg) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const Tensor cos_t = cos_arg.is_vulkan() ? cos_arg : cos_arg.vulkan();
  const Tensor sin_t = sin_arg.is_vulkan() ? sin_arg : sin_arg.vulkan();

  const vTensor& v_input = convert(input);
  const vTensor& v_cos = convert(cos_t);
  const vTensor& v_sin = convert(sin_t);

  vTensor v_output{
      context,
      input_arg.sizes().vec(),
      v_input.dtype(),
  };

  const int64_t head_dim = input_arg.size(-1);
  const int64_t seq_len = input_arg.size(-2);
  // z-extent = ceil(num_heads / 4) for channels-packed batch dim
  const uint32_t z_extent = safe_downcast<uint32_t>(v_input.extents().data[2]);

  const struct {
    ivec4 extents;
  } block{
      {
          safe_downcast<int32_t>(head_dim),
          safe_downcast<int32_t>(seq_len),
          safe_downcast<int32_t>(z_extent),
          safe_downcast<int32_t>(head_dim / 2),
      },
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(rotary_embedding),
      pipeline_barrier,
      {
          safe_downcast<uint32_t>(head_dim),
          safe_downcast<uint32_t>(seq_len),
          z_extent,
      },
      {8, 1, 1},
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_cos.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_sin.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::rotary_embedding"),
      rotary_embedding);
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
