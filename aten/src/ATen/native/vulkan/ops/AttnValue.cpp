#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor attn_value(
    const Tensor& attn_arg,
    const Tensor& v_arg,
    int64_t n_heads,
    int64_t n_kv_heads) {
  api::Context* const context = api::context();

  const Tensor attn = attn_arg.is_vulkan() ? attn_arg : attn_arg.vulkan();
  const Tensor v = v_arg.is_vulkan() ? v_arg : v_arg.vulkan();

  const vTensor& v_attn = convert(attn);
  const vTensor& v_v = convert(v);

  // attn: (n_heads, 1, S_kv), V: (n_kv_heads, S_kv, head_dim)
  // Output: (n_heads, 1, head_dim)
  const int64_t S_kv = v_arg.size(1);
  const int64_t head_dim = v_arg.size(2);

  vTensor v_output{
      context,
      {n_heads, 1, head_dim},
      v_attn.dtype(),
  };

  const struct {
    ivec4 extents;
    int32_t n_heads;
    int32_t n_kv_heads;
  } block{
      {
          safe_downcast<int32_t>(head_dim),
          safe_downcast<int32_t>(S_kv),
          safe_downcast<int32_t>(v_output.extents().data[2]),
          0,
      },
      safe_downcast<int32_t>(n_heads),
      safe_downcast<int32_t>(n_kv_heads),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(attn_value),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_attn.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_v.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::attn_value"), attn_value);
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
