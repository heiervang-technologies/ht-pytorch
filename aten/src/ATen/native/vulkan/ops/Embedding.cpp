#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor embedding(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  TORCH_CHECK(weight.dim() == 2, "Vulkan embedding: weight must be 2D");

  if (indices.dim() == 1) {
    return weight.index_select(0, indices);
  }

  auto flat_indices = indices.reshape(-1);
  auto embedded = weight.index_select(0, flat_indices);

  auto out_sizes = indices.sizes().vec();
  out_sizes.push_back(weight.size(1));
  return embedded.reshape(out_sizes);
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::embedding"), TORCH_FN(embedding));
}

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
