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

Tensor gather(
    const Tensor& self_arg,
    const int64_t dim,
    const Tensor& index_arg,
    bool sparse_grad) {
  TORCH_CHECK(
      self_arg.dim() >= 1 && self_arg.dim() <= 4,
      "Vulkan gather: input must be 1-4D, got ", self_arg.dim(), "D");

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const Tensor index = index_arg.is_vulkan() ? index_arg : index_arg.vulkan();

  const vTensor& v_self = convert(self);
  const vTensor& v_index = convert(index);

  vTensor v_output{
      context,
      v_index.sizes(),
      v_self.dtype(),
  };

  const int64_t ndim = self_arg.dim();
  const int64_t norm_dim = utils::normalize(dim, ndim);

  // Map tensor dim to Vulkan image dim (reversed for channels-packed layout)
  // For a 4D NCHW tensor packed as channels: x=W, y=H, z=ceil(C/4)*N
  // For a 3D tensor: x=W, y=H, z=ceil(C/4)
  // For a 2D tensor: x=W, y=H, z=1
  int vk_dim;
  if (ndim <= 2) {
    vk_dim = (norm_dim == ndim - 1) ? 0 : 1;
  } else {
    // For 3D+, width is last dim, height is second-to-last
    if (norm_dim == ndim - 1) {
      vk_dim = 0; // width
    } else if (norm_dim == ndim - 2) {
      vk_dim = 1; // height
    } else {
      vk_dim = 2; // channels/batch
    }
  }

  const struct Block final {
    ivec4 out_extents;
    int32_t dim;
  } block{
      {
          safe_downcast<int32_t>(v_output.extents().data[0u]),
          safe_downcast<int32_t>(v_output.extents().data[1u]),
          safe_downcast<int32_t>(v_output.extents().data[2u]),
          0,
      },
      vk_dim,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(gather),
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
      params.buffer());

  return convert(v_output);
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::gather"), TORCH_FN(gather));
}

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
