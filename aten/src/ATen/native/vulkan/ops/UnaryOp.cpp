#include <ATen/ArrayRef.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <torch/library.h>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {
using namespace api::utils;

Tensor unary_op(
    const Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  const struct Block final {
    uvec3 extents;
    uint32_t fill0;
  } block{
      v_self.extents(),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor& unary_op_(Tensor& self_arg, const api::ShaderInfo& shader_descriptor) {
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);

  const struct Block final {
    uvec3 extents;
    uint32_t fill0;
  } block{
      v_self.extents(),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      // params buffer
      params.buffer());

  return self_arg;
}

Tensor exp(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(exp));
}

Tensor& exp_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(exp_inplace));
}

Tensor sqrt(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(sqrt));
}

Tensor& sqrt_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(sqrt_inplace));
}

Tensor log(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(log));
}

Tensor& log_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(log_inplace));
}

Tensor neg(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(neg));
}

Tensor& neg_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(neg_inplace));
}

Tensor floor(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(floor));
}

Tensor& floor_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(floor_inplace));
}

Tensor ceil(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(ceil));
}

Tensor& ceil_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(ceil_inplace));
}

Tensor round(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(round));
}

Tensor& round_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(round_inplace));
}

Tensor sign(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(sign));
}

Tensor sin(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(sin));
}

Tensor& sin_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(sin_inplace));
}

Tensor cos(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(cos));
}

Tensor& cos_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(cos_inplace));
}

Tensor rsqrt(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(rsqrt));
}

Tensor& rsqrt_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(rsqrt_inplace));
}

Tensor reciprocal(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(reciprocal));
}

Tensor& reciprocal_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(reciprocal_inplace));
}

Tensor silu(const Tensor& self_arg) {
  return unary_op(self_arg, VK_KERNEL(silu));
}

Tensor& silu_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(silu_inplace));
}

Tensor& silu_out(const Tensor& self_arg, Tensor& out) {
  Tensor result = silu(self_arg);
  out.resize_(result.sizes());
  out.copy_(result);
  return out;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::exp"), TORCH_FN(exp));
  m.impl(TORCH_SELECTIVE_NAME("aten::exp_"), TORCH_FN(exp_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sqrt"), TORCH_FN(sqrt));
  m.impl(TORCH_SELECTIVE_NAME("aten::sqrt_"), TORCH_FN(sqrt_));
  m.impl(TORCH_SELECTIVE_NAME("aten::log"), TORCH_FN(log));
  m.impl(TORCH_SELECTIVE_NAME("aten::log_"), TORCH_FN(log_));
  m.impl(TORCH_SELECTIVE_NAME("aten::neg"), TORCH_FN(neg));
  m.impl(TORCH_SELECTIVE_NAME("aten::neg_"), TORCH_FN(neg_));
  m.impl(TORCH_SELECTIVE_NAME("aten::floor"), TORCH_FN(floor));
  m.impl(TORCH_SELECTIVE_NAME("aten::floor_"), TORCH_FN(floor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::ceil"), TORCH_FN(ceil));
  m.impl(TORCH_SELECTIVE_NAME("aten::ceil_"), TORCH_FN(ceil_));
  m.impl(TORCH_SELECTIVE_NAME("aten::round"), TORCH_FN(round));
  m.impl(TORCH_SELECTIVE_NAME("aten::round_"), TORCH_FN(round_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sign"), TORCH_FN(sign));
  m.impl(TORCH_SELECTIVE_NAME("aten::sin"), TORCH_FN(sin));
  m.impl(TORCH_SELECTIVE_NAME("aten::sin_"), TORCH_FN(sin_));
  m.impl(TORCH_SELECTIVE_NAME("aten::cos"), TORCH_FN(cos));
  m.impl(TORCH_SELECTIVE_NAME("aten::cos_"), TORCH_FN(cos_));
  m.impl(TORCH_SELECTIVE_NAME("aten::rsqrt"), TORCH_FN(rsqrt));
  m.impl(TORCH_SELECTIVE_NAME("aten::rsqrt_"), TORCH_FN(rsqrt_));
  m.impl(TORCH_SELECTIVE_NAME("aten::reciprocal"), TORCH_FN(reciprocal));
  m.impl(TORCH_SELECTIVE_NAME("aten::reciprocal_"), TORCH_FN(reciprocal_));
  m.impl(TORCH_SELECTIVE_NAME("aten::silu"), TORCH_FN(silu));
  m.impl(TORCH_SELECTIVE_NAME("aten::silu_"), TORCH_FN(silu_));
  m.impl(TORCH_SELECTIVE_NAME("aten::silu.out"), TORCH_FN(silu_out));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
