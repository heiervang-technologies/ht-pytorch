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

Tensor validate_indices(const Tensor& index_arg, const int64_t dim_size) {
  const Tensor index_cpu = index_arg.cpu().contiguous();
  if (index_cpu.scalar_type() == at::kLong) {
    const int64_t* const indices = index_cpu.const_data_ptr<int64_t>();
    for (const auto i : c10::irange(index_cpu.numel())) {
      TORCH_CHECK(
          indices[i] >= 0 && indices[i] < dim_size,
          "Vulkan index_select: index ",
          indices[i],
          " is out of bounds for dimension with size ",
          dim_size);
    }
  } else {
    const int32_t* const indices = index_cpu.const_data_ptr<int32_t>();
    for (const auto i : c10::irange(index_cpu.numel())) {
      TORCH_CHECK(
          indices[i] >= 0 && indices[i] < dim_size,
          "Vulkan index_select: index ",
          indices[i],
          " is out of bounds for dimension with size ",
          dim_size);
    }
  }
  return index_cpu.to(at::kInt);
}

Tensor index_select(
    const Tensor& self_arg,
    const int64_t dim,
    const Tensor& index_arg) {
  TORCH_CHECK(
      self_arg.dim() >= 1 && self_arg.dim() <= 4,
      "Vulkan index_select: input must be 1-4D, got ",
      self_arg.dim(),
      "D");
  TORCH_CHECK(
      index_arg.dim() == 1,
      "Vulkan index_select: index must be 1D, got ",
      index_arg.dim(),
      "D");
  TORCH_CHECK(
      index_arg.scalar_type() == at::kInt ||
          index_arg.scalar_type() == at::kLong,
      "Vulkan index_select: index must be int32 or int64");

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  // Compute output sizes
  const int64_t ndim = self_arg.dim();
  const int64_t norm_dim = utils::normalize(dim, ndim);
  TORCH_CHECK(
      norm_dim >= ndim - 2,
      "Vulkan index_select currently supports only the final two dimensions");
  const Tensor index_cpu = validate_indices(index_arg, self_arg.size(norm_dim));
  const int64_t num_indices = index_cpu.numel();
  auto out_sizes = self_arg.sizes().vec();
  out_sizes[norm_dim] = num_indices;

  vTensor v_output{
      context,
      out_sizes,
      v_self.dtype(),
  };

  // The final two logical dimensions map directly to image width and height.
  const int vk_dim = norm_dim == ndim - 1 ? 0 : 1;

  // Upload indices as SSBO
  api::StorageBuffer index_buffer(context, api::kInt, num_indices);
  {
    api::MemoryMap mapping(index_buffer.buffer(), api::MemoryAccessType::WRITE);
    int32_t* data_ptr = mapping.template data<int32_t>();
    memcpy(
        data_ptr, index_cpu.data_ptr<int32_t>(), sizeof(int32_t) * num_indices);
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
