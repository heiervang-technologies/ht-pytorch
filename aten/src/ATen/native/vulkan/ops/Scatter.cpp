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
  TORCH_CHECK(
      index_arg.scalar_type() == at::kLong,
      "Vulkan scatter: index must have dtype int64");

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const Tensor src = src_arg.is_vulkan() ? src_arg : src_arg.vulkan();

  const vTensor& v_self = convert(self);
  const vTensor& v_src = convert(src);
  const Tensor index_cpu = index_arg.cpu().to(at::kInt).contiguous();

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  const int64_t ndim = self_arg.dim();
  const int64_t norm_dim = utils::normalize(dim, ndim);

  api::StorageBuffer index_buffer(context, api::kInt, index_cpu.numel());
  {
    api::MemoryMap mapping(
        index_buffer.buffer(), api::MemoryAccessType::WRITE);
    memcpy(
        mapping.template data<int32_t>(),
        index_cpu.const_data_ptr<int32_t>(),
        index_cpu.nbytes());
  }

  int64_t src_dim_size = index_arg.size(norm_dim);

  const struct Block final {
    ivec4 out_extents;
    ivec4 self_sizes;
    ivec4 index_sizes;
    int32_t dim;
    int32_t src_dim_size;
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
      safe_downcast<int32_t>(src_dim_size),
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
      index_buffer.buffer(),
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
