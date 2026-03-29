#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor rms_norm(const Tensor& input_arg, const Tensor& weight_arg) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const Tensor weight = weight_arg.is_vulkan() ? weight_arg : weight_arg.vulkan();

  const vTensor& v_input = convert(input);
  const vTensor& v_weight = convert(weight);

  vTensor v_output{
      context,
      input_arg.sizes().vec(),
      v_input.dtype(),
  };

  const int64_t D = input_arg.size(-1);

  const struct {
    ivec4 extents;
  } block{
      {
          safe_downcast<int32_t>(D),
          safe_downcast<int32_t>(v_input.extents().data[1]),
          safe_downcast<int32_t>(v_input.extents().data[2]),
          0,
      },
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  const uint32_t n_rows_y = safe_downcast<uint32_t>(v_input.extents().data[1]);
  const uint32_t n_rows_z = safe_downcast<uint32_t>(v_input.extents().data[2]);

  context->submit_compute_job(
      VK_KERNEL(rms_norm),
      pipeline_barrier,
      {256 * n_rows_y, n_rows_z, 1},
      {256, 1, 1},
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::rms_norm"), rms_norm);
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
