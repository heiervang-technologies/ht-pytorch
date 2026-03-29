#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/Functions.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

std::tuple<Tensor, Tensor, Tensor> native_group_norm(
    const Tensor& input_arg,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps) {
  TORCH_CHECK(
      input_arg.dim() >= 2 && input_arg.dim() <= 4,
      "Vulkan group_norm: input must be 2-4D, got ", input_arg.dim(), "D");
  TORCH_CHECK(C % group == 0, "Vulkan group_norm: C must be divisible by group");

  // Compute mean and rstd per group on CPU (group statistics are small)
  const Tensor input_cpu = input_arg.is_vulkan() ? input_arg.cpu() : input_arg;
  const auto input_contig = input_cpu.contiguous();
  const float* input_data = input_contig.data_ptr<float>();

  int64_t channels_per_group = C / group;
  auto mean_out = at::empty({N, group}, input_arg.options().device(at::kCPU));
  auto rstd_out = at::empty({N, group}, input_arg.options().device(at::kCPU));
  float* mean_data = mean_out.data_ptr<float>();
  float* rstd_data = rstd_out.data_ptr<float>();

  for (int64_t n = 0; n < N; n++) {
    for (int64_t g = 0; g < group; g++) {
      float sum = 0.0f;
      float sq_sum = 0.0f;
      int64_t count = channels_per_group * HxW;
      for (int64_t c = g * channels_per_group;
           c < (g + 1) * channels_per_group; c++) {
        for (int64_t hw = 0; hw < HxW; hw++) {
          float val = input_data[n * C * HxW + c * HxW + hw];
          sum += val;
          sq_sum += val * val;
        }
      }
      float mean = sum / count;
      float var = sq_sum / count - mean * mean;
      int64_t idx = n * group + g;
      mean_data[idx] = mean;
      rstd_data[idx] = 1.0f / std::sqrt(var + static_cast<float>(eps));
    }
  }

  // Apply normalization on GPU via shader
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  vTensor v_output{
      context,
      v_input.sizes(),
      v_input.dtype(),
  };

  // Upload mean/rstd as SSBOs
  int64_t stat_count = N * group;
  api::StorageBuffer mean_buffer(context, api::kFloat, stat_count);
  api::StorageBuffer rstd_buffer(context, api::kFloat, stat_count);
  {
    api::MemoryMap mapping(mean_buffer.buffer(), api::MemoryAccessType::WRITE);
    memcpy(
        mapping.template data<float>(),
        mean_data,
        sizeof(float) * stat_count);
  }
  {
    api::MemoryMap mapping(rstd_buffer.buffer(), api::MemoryAccessType::WRITE);
    memcpy(
        mapping.template data<float>(),
        rstd_data,
        sizeof(float) * stat_count);
  }

  // Upload weight/bias as SSBOs
  bool has_weight = weight_opt.has_value() && weight_opt->defined();
  bool has_bias = bias_opt.has_value() && bias_opt->defined();

  api::StorageBuffer weight_buffer(context, api::kFloat, has_weight ? C : 1);
  api::StorageBuffer bias_buffer(context, api::kFloat, has_bias ? C : 1);

  if (has_weight) {
    const Tensor w = weight_opt->cpu().contiguous();
    api::MemoryMap mapping(weight_buffer.buffer(), api::MemoryAccessType::WRITE);
    memcpy(
        mapping.template data<float>(),
        w.data_ptr<float>(),
        sizeof(float) * C);
  }
  if (has_bias) {
    const Tensor b = bias_opt->cpu().contiguous();
    api::MemoryMap mapping(bias_buffer.buffer(), api::MemoryAccessType::WRITE);
    memcpy(
        mapping.template data<float>(),
        b.data_ptr<float>(),
        sizeof(float) * C);
  }

  const struct Block final {
    ivec4 extents;
    int32_t num_channels;
    int32_t num_groups;
    int32_t has_weight;
    int32_t has_bias;
  } block{
      {
          safe_downcast<int32_t>(v_output.extents().data[0u]),
          safe_downcast<int32_t>(v_output.extents().data[1u]),
          safe_downcast<int32_t>(v_output.extents().data[2u]),
          0,
      },
      safe_downcast<int32_t>(C),
      safe_downcast<int32_t>(group),
      has_weight ? 1 : 0,
      has_bias ? 1 : 0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(group_norm),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      mean_buffer.buffer(),
      rstd_buffer.buffer(),
      weight_buffer.buffer(),
      bias_buffer.buffer(),
      params.buffer());

  return std::make_tuple(convert(v_output), mean_out, rstd_out);
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::native_group_norm"),
      TORCH_FN(native_group_norm));
}

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
