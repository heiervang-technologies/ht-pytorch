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

Tensor index_select(
    const Tensor& self_arg,
    const int64_t dim,
    const Tensor& index_arg) {
  TORCH_CHECK(
      self_arg.dim() >= 1 && self_arg.dim() <= 4,
      "Vulkan index_select: input must be 1-4D, got ", self_arg.dim(), "D");
  TORCH_CHECK(
      index_arg.dim() == 1,
      "Vulkan index_select: index must be 1D, got ", index_arg.dim(), "D");
  TORCH_CHECK(
      index_arg.scalar_type() == at::kInt || index_arg.scalar_type() == at::kLong,
      "Vulkan index_select: index must be int32 or int64");

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  // Index must be on CPU as int32 for the SSBO
  const Tensor index_cpu = index_arg.cpu().to(at::kInt).contiguous();
  const int64_t num_indices = index_cpu.numel();

  // Compute output sizes
  const int64_t ndim = self_arg.dim();
  const int64_t norm_dim = utils::normalize(dim, ndim);
  auto out_sizes = self_arg.sizes().vec();
  out_sizes[norm_dim] = num_indices;

  vTensor v_output{
      context,
      out_sizes,
      v_self.dtype(),
  };

  // Map tensor dim to Vulkan image dim
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

  // Upload indices as SSBO
  api::StorageBuffer index_buffer(context, api::kInt, num_indices);
  {
    api::MemoryMap mapping(index_buffer.buffer(), api::MemoryAccessType::WRITE);
    int32_t* data_ptr = mapping.template data<int32_t>();
    memcpy(data_ptr, index_cpu.data_ptr<int32_t>(), sizeof(int32_t) * num_indices);
  }

  const struct Block final {
    ivec4 out_extents;
    int32_t dim;
    int32_t num_indices;
    int32_t src_dim_size;
    int32_t fill0;
  } block{
      {
          safe_downcast<int32_t>(v_output.extents().data[0u]),
          safe_downcast<int32_t>(v_output.extents().data[1u]),
          safe_downcast<int32_t>(v_output.extents().data[2u]),
          0,
      },
      vk_dim,
      safe_downcast<int32_t>(num_indices),
      safe_downcast<int32_t>(self_arg.size(norm_dim)),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(index_select),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      index_buffer.buffer(),
      params.buffer());

  return convert(v_output);
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::index_select"), TORCH_FN(index_select));
}

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
