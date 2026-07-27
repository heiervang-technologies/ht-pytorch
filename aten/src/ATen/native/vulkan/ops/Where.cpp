#ifdef USE_VULKAN_API

#include <ATen/ExpandUtils.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor where_self(
    const Tensor& condition_arg,
    const Tensor& self_arg,
    const Tensor& other_arg) {
  TORCH_CHECK(
      condition_arg.scalar_type() == at::kBool,
      "where expected condition to be a boolean tensor, but got ",
      condition_arg.scalar_type());
  TORCH_CHECK(
      condition_arg.dim() <= 4 && self_arg.dim() <= 4 && other_arg.dim() <= 4,
      "Vulkan where supports tensors up to 4 dimensions");

  api::Context* const context = api::context();

  auto out_sizes =
      at::infer_size_dimvector(self_arg.sizes(), other_arg.sizes());
  out_sizes = at::infer_size_dimvector(out_sizes, condition_arg.sizes());

  const ScalarType result_dtype = at::native::result_type(self_arg, other_arg);
  Tensor condition_cpu =
      condition_arg.is_vulkan() ? condition_arg.cpu() : condition_arg;
  if (condition_cpu.dim() == 0) {
    condition_cpu = condition_cpu.reshape({1});
  }
  const Tensor condition = condition_cpu.to(at::kFloat).vulkan();

  Tensor self_typed = self_arg.scalar_type() == result_dtype
      ? self_arg
      : self_arg.to(result_dtype);
  if (self_typed.dim() == 0) {
    self_typed = self_typed.reshape({1});
  }
  const Tensor self = self_typed.is_vulkan() ? self_typed : self_typed.vulkan();

  Tensor other_typed = other_arg.scalar_type() == result_dtype
      ? other_arg
      : other_arg.to(result_dtype);
  if (other_typed.dim() == 0) {
    other_typed = other_typed.reshape({1});
  }
  const Tensor other =
      other_typed.is_vulkan() ? other_typed : other_typed.vulkan();

  const vTensor& v_condition = convert(condition);
  const vTensor& v_self = convert(self);
  const vTensor& v_other = convert(other);

  vTensor v_output{
      context,
      out_sizes,
      v_self.dtype(),
  };

  const struct Block final {
    ivec4 output_sizes;
    ivec4 condition_sizes;
    ivec4 self_sizes;
    ivec4 other_sizes;
  } block{
      {
          safe_downcast<int32_t>(get_dim<Dim4D::Width>(v_output)),
          safe_downcast<int32_t>(get_dim<Dim4D::Height>(v_output)),
          safe_downcast<int32_t>(get_dim<Dim4D::Channel>(v_output)),
          safe_downcast<int32_t>(get_dim<Dim4D::Batch>(v_output)),
      },
      {
          safe_downcast<int32_t>(get_dim<Dim4D::Width>(v_condition)),
          safe_downcast<int32_t>(get_dim<Dim4D::Height>(v_condition)),
          safe_downcast<int32_t>(get_dim<Dim4D::Channel>(v_condition)),
          safe_downcast<int32_t>(get_dim<Dim4D::Batch>(v_condition)),
      },
      {
          safe_downcast<int32_t>(get_dim<Dim4D::Width>(v_self)),
          safe_downcast<int32_t>(get_dim<Dim4D::Height>(v_self)),
          safe_downcast<int32_t>(get_dim<Dim4D::Channel>(v_self)),
          safe_downcast<int32_t>(get_dim<Dim4D::Batch>(v_self)),
      },
      {
          safe_downcast<int32_t>(get_dim<Dim4D::Width>(v_other)),
          safe_downcast<int32_t>(get_dim<Dim4D::Height>(v_other)),
          safe_downcast<int32_t>(get_dim<Dim4D::Channel>(v_other)),
          safe_downcast<int32_t>(get_dim<Dim4D::Batch>(v_other)),
      },
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(where),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_condition.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::where.self"), TORCH_FN(where_self));
}

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
