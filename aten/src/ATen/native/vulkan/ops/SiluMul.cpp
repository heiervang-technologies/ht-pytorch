#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor silu_mul(const Tensor& input_arg) {
  TORCH_CHECK(
      input_arg.dim() >= 1,
      "Vulkan SiLU-mul: input must have at least one dimension");
  TORCH_CHECK(
      input_arg.size(-1) % 2 == 0,
      "Vulkan SiLU-mul: final dimension must be even");

  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  // Input is (B, S, 2*intermediate), output is (B, S, intermediate)
  auto sizes = input_arg.sizes().vec();
  const int64_t half_w = sizes.back() / 2;
  sizes.back() = half_w;

  vTensor v_output{context, sizes, v_input.dtype()};

  const struct {
    ivec4 extents;
  } block{
      {
          safe_downcast<int32_t>(half_w),
          safe_downcast<int32_t>(v_input.extents().data[1]),
          safe_downcast<int32_t>(v_input.extents().data[2]),
          0,
      },
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(silu_mul),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::silu_mul"), silu_mul);
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
