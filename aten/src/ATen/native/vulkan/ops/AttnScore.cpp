#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor attn_score(
    const Tensor& q_arg,
    const Tensor& k_arg,
    double scale,
    int64_t n_heads,
    int64_t n_kv_heads) {
  api::Context* const context = api::context();

  const Tensor q = q_arg.is_vulkan() ? q_arg : q_arg.vulkan();
  const Tensor k = k_arg.is_vulkan() ? k_arg : k_arg.vulkan();

  const vTensor& v_q = convert(q);
  const vTensor& v_k = convert(k);

  // Q: (n_heads, 1, head_dim), K: (n_kv_heads, S_kv, head_dim)
  // Output: (n_heads, 1, S_kv)
  const int64_t S_kv = k_arg.size(1);
  const int64_t head_dim = q_arg.size(2);

  vTensor v_output{
      context,
      {n_heads, 1, S_kv},
      v_q.dtype(),
  };

  const struct {
    ivec4 extents;
    float scale;
    int32_t n_heads;
    int32_t n_kv_heads;
  } block{
      {
          safe_downcast<int32_t>(S_kv),
          safe_downcast<int32_t>(head_dim),
          safe_downcast<int32_t>(v_output.extents().data[2]),
          safe_downcast<int32_t>(v_k.extents().data[2]),
      },
      static_cast<float>(scale),
      safe_downcast<int32_t>(n_heads),
      safe_downcast<int32_t>(n_kv_heads),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(attn_score),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_q.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_k.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::attn_score"), attn_score);
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
