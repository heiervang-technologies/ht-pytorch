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

std::tuple<Tensor, Tensor> topk(
    const Tensor& self_arg,
    const int64_t k,
    const int64_t dim,
    const bool largest,
    const bool sorted) {
  TORCH_CHECK(
      self_arg.dim() >= 1 && self_arg.dim() <= 4,
      "Vulkan topk: input must be 1-4D, got ", self_arg.dim(), "D");

  const int64_t ndim = self_arg.dim();
  const int64_t norm_dim = utils::normalize(dim, ndim);

  // Currently only supports topk along the last (width) dimension
  TORCH_CHECK(
      norm_dim == ndim - 1,
      "Vulkan topk: currently only supports dim=-1 (last dimension)");

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  auto out_sizes = self_arg.sizes().vec();
  out_sizes[norm_dim] = k;

  vTensor v_values{
      context,
      out_sizes,
      v_self.dtype(),
  };

  vTensor v_indices{
      context,
      out_sizes,
      v_self.dtype(),
  };

  const struct Block final {
    ivec4 out_extents;
    int32_t k;
    int32_t dim_size;
    int32_t largest;
    int32_t sorted;
  } block{
      {
          safe_downcast<int32_t>(v_values.extents().data[0u]),
          safe_downcast<int32_t>(v_values.extents().data[1u]),
          safe_downcast<int32_t>(v_values.extents().data[2u]),
          0,
      },
      safe_downcast<int32_t>(k),
      safe_downcast<int32_t>(self_arg.size(norm_dim)),
      largest ? 1 : 0,
      sorted ? 1 : 0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(topk),
      pipeline_barrier,
      v_values.extents(),
      adaptive_work_group_size(v_values.extents()),
      VK_NULL_HANDLE,
      v_values.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_indices.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  // Vulkan has no int64 scalar type, so return the required Long indices on
  // CPU instead of attempting an unsupported Long upload.
  Tensor values_out = convert(v_values);
  Tensor indices_float = convert(v_indices);
  Tensor indices_out = indices_float.cpu().to(at::kLong);

  return std::make_tuple(values_out, indices_out);
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::topk"), TORCH_FN(topk));
}

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
