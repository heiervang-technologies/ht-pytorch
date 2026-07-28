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
  const int64_t* const indices = index_cpu.const_data_ptr<int64_t>();
  for (const auto i : c10::irange(index_cpu.numel())) {
    TORCH_CHECK(
        indices[i] >= 0 && indices[i] < dim_size,
        "Vulkan gather: index ",
        indices[i],
        " is out of bounds for dimension with size ",
        dim_size);
  }
  return index_cpu.to(at::kInt);
}

Tensor gather(
    const Tensor& self_arg,
    const int64_t dim,
    const Tensor& index_arg,
    bool sparse_grad) {
  TORCH_CHECK(
      self_arg.dim() >= 1 && self_arg.dim() <= 4,
      "Vulkan gather: input must be 1-4D, got ",
      self_arg.dim(),
      "D");
  TORCH_CHECK(
      index_arg.scalar_type() == at::kLong,
      "Vulkan gather: index must have dtype int64");
  TORCH_CHECK(
      index_arg.dim() == self_arg.dim(),
      "Vulkan gather: input and index must have the same number of dimensions");

  const int64_t ndim = self_arg.dim();
  const int64_t norm_dim = utils::normalize(dim, ndim);
  for (const auto d : c10::irange(ndim)) {
    TORCH_CHECK(
        d == norm_dim || index_arg.size(d) <= self_arg.size(d),
        "Vulkan gather: index size must not exceed input size at dimension ",
        d);
  }

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);
  const Tensor index_cpu = validate_indices(index_arg, self_arg.size(norm_dim));

  vTensor v_output{
      context,
      index_arg.sizes().vec(),
      v_self.dtype(),
  };

  api::StorageBuffer index_buffer(context, api::kInt, index_cpu.numel());
  {
    api::MemoryMap mapping(index_buffer.buffer(), api::MemoryAccessType::WRITE);
    memcpy(
        mapping.template data<int32_t>(),
        index_cpu.const_data_ptr<int32_t>(),
        index_cpu.nbytes());
  }

  const struct Block final {
    ivec4 out_extents;
    ivec4 self_sizes;
    ivec4 index_sizes;
    int32_t dim;
  } block{
      {
          safe_downcast<int32_t>(v_output.extents().data[0u]),
          safe_downcast<int32_t>(v_output.extents().data[1u]),
          safe_downcast<int32_t>(v_output.extents().data[2u]),
          0,
      },
      {
          safe_downcast<int32_t>(get_dim<Dim4D::Width>(v_self)),
          safe_downcast<int32_t>(get_dim<Dim4D::Height>(v_self)),
          safe_downcast<int32_t>(get_dim<Dim4D::Channel>(v_self)),
          safe_downcast<int32_t>(get_dim<Dim4D::Batch>(v_self)),
      },
      {
          safe_downcast<int32_t>(get_dim<Dim4D::Width>(index_arg)),
          safe_downcast<int32_t>(get_dim<Dim4D::Height>(index_arg)),
          safe_downcast<int32_t>(get_dim<Dim4D::Channel>(index_arg)),
          safe_downcast<int32_t>(get_dim<Dim4D::Batch>(index_arg)),
      },
      safe_downcast<int32_t>(norm_dim + 4 - ndim),
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
      index_buffer.buffer(),
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
