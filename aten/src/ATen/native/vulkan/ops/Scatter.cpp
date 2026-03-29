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

Tensor scatter_src(
    const Tensor& self_arg,
    int64_t dim,
    const Tensor& index_arg,
    const Tensor& src_arg) {
  TORCH_CHECK(
      self_arg.dim() >= 1 && self_arg.dim() <= 4,
      "Vulkan scatter: input must be 1-4D, got ", self_arg.dim(), "D");

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const Tensor index = index_arg.is_vulkan() ? index_arg : index_arg.vulkan();
  const Tensor src = src_arg.is_vulkan() ? src_arg : src_arg.vulkan();

  const vTensor& v_self = convert(self);
  const vTensor& v_index = convert(index);
  const vTensor& v_src = convert(src);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  const int64_t ndim = self_arg.dim();
  const int64_t norm_dim = utils::normalize(dim, ndim);

  int vk_dim;
  if (ndim <= 2) {
    vk_dim = (norm_dim == ndim - 1) ? 0 : 1;
  } else {
    if (norm_dim == ndim - 1) {
      vk_dim = 0;
    } else if (norm_dim == ndim - 2) {
      vk_dim = 1;
    } else {
      vk_dim = 2;
    }
  }

  int64_t src_dim_size = index_arg.size(norm_dim);

  const struct Block final {
    ivec4 out_extents;
    int32_t dim;
    int32_t src_dim_size;
    int32_t fill0;
    int32_t fill1;
  } block{
      {
          safe_downcast<int32_t>(v_output.extents().data[0u]),
          safe_downcast<int32_t>(v_output.extents().data[1u]),
          safe_downcast<int32_t>(v_output.extents().data[2u]),
          0,
      },
      vk_dim,
      safe_downcast<int32_t>(src_dim_size),
      0,
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(scatter),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_index.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_src.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::scatter.src"), TORCH_FN(scatter_src));
}

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
