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

Tensor amax_dim(const at::Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan amax supports 1-4D tensors, got ",
      self.dim(),
      "D");

  api::Context* const context = api::context();

  const Tensor input = self.is_vulkan() ? self : self.vulkan();
  const vTensor& v_input = convert(input);

  std::vector<int64_t> output_size = v_input.sizes();
  uint32_t dim_size = output_size[dim];
  if (keepdim) {
    output_size[dim] = 1;
  } else {
    output_size.erase(output_size.begin() + dim);
  }

  vTensor v_output{
      context,
      output_size,
      v_input.dtype(),
  };

  api::PipelineBarrier pipeline_barrier{};

  if (self.dim() < 4) {
    dim += (4 - self.dim());
  }

  const struct Block final {
    uvec2 dim_info;
    int32_t channel;
  } block{
      {static_cast<uint32_t>(dim), dim_size},
      static_cast<int32_t>(get_dim<Dim4D::Channel>(v_input)),
  };

  api::UniformParamsBuffer params(context, block);

  context->submit_compute_job(
      keepdim ? VK_KERNEL(amax_dim_keepdim) : VK_KERNEL(amax_dim),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return convert(v_output);
}

Tensor amax(const at::Tensor& self, IntArrayRef dims, bool keepdim) {
  TORCH_CHECK(!dims.empty(), "Vulkan amax requires at least one dim");

  std::set<int64_t> dims_set;
  for (const auto& d : dims) {
    TORCH_CHECK(
        d >= -self.dim() && d <= self.dim() - 1,
        "Vulkan amax dimension out of range [",
        -self.dim(),
        ",",
        self.dim() - 1,
        "], got ",
        d);
    dims_set.insert(utils::normalize(d, self.dim()));
  }

  Tensor result = self;
  for (auto it = dims_set.rbegin(); it != dims_set.rend(); ++it) {
    result = amax_dim(result, *it, keepdim);
  }
  return result;
}

Tensor amin_dim(const at::Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan amin supports 1-4D tensors, got ",
      self.dim(),
      "D");

  api::Context* const context = api::context();

  const Tensor input = self.is_vulkan() ? self : self.vulkan();
  const vTensor& v_input = convert(input);

  std::vector<int64_t> output_size = v_input.sizes();
  uint32_t dim_size = output_size[dim];
  if (keepdim) {
    output_size[dim] = 1;
  } else {
    output_size.erase(output_size.begin() + dim);
  }

  vTensor v_output{
      context,
      output_size,
      v_input.dtype(),
  };

  api::PipelineBarrier pipeline_barrier{};

  int64_t shifted_dim = dim;
  if (self.dim() < 4) {
    shifted_dim += (4 - self.dim());
  }

  const struct Block final {
    uvec2 dim_info;
    int32_t channel;
  } block{
      {static_cast<uint32_t>(shifted_dim), dim_size},
      static_cast<int32_t>(get_dim<Dim4D::Channel>(v_input)),
  };

  api::UniformParamsBuffer params(context, block);

  context->submit_compute_job(
      keepdim ? VK_KERNEL(amin_dim_keepdim) : VK_KERNEL(amin_dim),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
  return convert(v_output);
}

Tensor amin(const at::Tensor& self, IntArrayRef dims, bool keepdim) {
  TORCH_CHECK(!dims.empty(), "Vulkan amin requires at least one dim");

  std::set<int64_t> dims_set;
  for (const auto& d : dims) {
    TORCH_CHECK(
        d >= -self.dim() && d <= self.dim() - 1,
        "Vulkan amin dimension out of range [",
        -self.dim(),
        ",",
        self.dim() - 1,
        "], got ",
        d);
    dims_set.insert(utils::normalize(d, self.dim()));
  }

  Tensor result = self;
  for (auto it = dims_set.rbegin(); it != dims_set.rend(); ++it) {
    result = amin_dim(result, *it, keepdim);
  }
  return result;
}

Tensor& amax_out(
    const at::Tensor& self,
    IntArrayRef dims,
    bool keepdim,
    Tensor& out) {
  Tensor result = amax(self, dims, keepdim);
  out.resize_(result.sizes());
  out.copy_(result);
  return out;
}

Tensor& amin_out(
    const at::Tensor& self,
    IntArrayRef dims,
    bool keepdim,
    Tensor& out) {
  Tensor result = amin(self, dims, keepdim);
  out.resize_(result.sizes());
  out.copy_(result);
  return out;
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::amax"), TORCH_FN(amax));
  m.impl(TORCH_SELECTIVE_NAME("aten::amax.out"), TORCH_FN(amax_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::amin"), TORCH_FN(amin));
  m.impl(TORCH_SELECTIVE_NAME("aten::amin.out"), TORCH_FN(amin_out));
}

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
