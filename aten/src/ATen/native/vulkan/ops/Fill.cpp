#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor& fill_scalar(Tensor& self_arg, const Scalar& value) {
  TORCH_CHECK(
      self_arg.is_vulkan(), "Vulkan fill_: input must be a Vulkan tensor");

  api::Context* const context = api::context();
  vTensor& v_self = convert(self_arg);

  const bool is_int_dtype = v_self.dtype() == api::kBool;

  api::PipelineBarrier pipeline_barrier{};

  if (is_int_dtype) {
    const struct Block final {
      ivec4 extents;
      int32_t value;
      int32_t fill0;
      int32_t fill1;
      int32_t fill2;
    } block{
        {
            safe_downcast<int32_t>(v_self.extents().data[0u]),
            safe_downcast<int32_t>(v_self.extents().data[1u]),
            safe_downcast<int32_t>(v_self.extents().data[2u]),
            0,
        },
        value.to<int>(),
        0,
        0,
        0,
    };

    api::UniformParamsBuffer params(context, block);

    context->submit_compute_job(
        VK_KERNEL(fill_int),
        pipeline_barrier,
        v_self.extents(),
        adaptive_work_group_size(v_self.extents()),
        VK_NULL_HANDLE,
        v_self.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        params.buffer());
  } else {
    const struct Block final {
      ivec4 extents;
      float value;
      int32_t fill0;
      int32_t fill1;
      int32_t fill2;
    } block{
        {
            safe_downcast<int32_t>(v_self.extents().data[0u]),
            safe_downcast<int32_t>(v_self.extents().data[1u]),
            safe_downcast<int32_t>(v_self.extents().data[2u]),
            0,
        },
        value.to<float>(),
        0,
        0,
        0,
    };

    api::UniformParamsBuffer params(context, block);

    context->submit_compute_job(
        VK_KERNEL(fill),
        pipeline_barrier,
        v_self.extents(),
        adaptive_work_group_size(v_self.extents()),
        VK_NULL_HANDLE,
        v_self.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        params.buffer());
  }

  return self_arg;
}

Tensor& fill_tensor(Tensor& self_arg, const Tensor& value) {
  TORCH_CHECK(
      value.dim() == 0, "Vulkan fill_: value tensor must be 0-dimensional");
  return fill_scalar(self_arg, value.item());
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::fill_.Scalar"), TORCH_FN(fill_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::fill_.Tensor"), TORCH_FN(fill_tensor));
}

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
