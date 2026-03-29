#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/impl/Packing.h>
#include <ATen/Functions.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

// CPU function: quantize float weight to int8, upload to Vulkan as
// channels-packed int8 texture with height-packed layout.
//
// Weight (K, N) float → vTensor (1, 4, K/4, N) QInt8 channels-packed
// Texel at (n, k/4, 0) = ivec4(w[k*4+0,n], w[k*4+1,n], w[k*4+2,n], w[k*4+3,n])
//
// Per-channel symmetric quantization: each column n has scale[n] = max_abs(W[:,n]) / 127.
// Returns (weight_vulkan, bias_vulkan_or_empty, scale_vulkan, zero_point_tensor).
std::tuple<Tensor, Tensor, Tensor, Tensor> create_q8_linear(
    const Tensor& weight_arg,
    const std::optional<Tensor>& bias_arg) {
  TORCH_CHECK(weight_arg.dim() == 2, "Weight must be 2D (K, N)");
  TORCH_CHECK(weight_arg.device().is_cpu(), "Weight must be on CPU");

  const Tensor weight = weight_arg.contiguous();
  const int64_t K = weight.size(0);
  const int64_t N = weight.size(1);
  TORCH_CHECK(K % 4 == 0, "K (", K, ") must be divisible by 4");

  const float* w_ptr = weight.const_data_ptr<float>();

  // Per-channel symmetric quantization: one scale per output column
  std::vector<float> scales(N);
  std::vector<float> inv_scales(N);
  for (int64_t n = 0; n < N; n++) {
    float col_absmax = 0.0f;
    for (int64_t k = 0; k < K; k++) {
      col_absmax = std::max(col_absmax, std::abs(w_ptr[k * N + n]));
    }
    scales[n] = col_absmax > 0 ? col_absmax / 127.0f : 1.0f;
    inv_scales[n] = 1.0f / scales[n];
  }

  api::Context* const context = api::context();

  const int64_t K4 = K / 4;
  vTensor v_weight{context, {1, 4, K4, N}, api::ScalarType::QInt8};
  v_weight.set_is_quantized();
  v_weight.set_scale(1.0);
  v_weight.set_zero_point(0);

  api::StorageBuffer staging(context, api::kFloat, v_weight.gpu_numel());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
    int8_t* dst = mapping.template data<int8_t>();
    memset(dst, 0, v_weight.gpu_numel() * sizeof(int8_t));

    for (int64_t c = 0; c < 4; c++) {
      for (int64_t kq = 0; kq < K4; kq++) {
        for (int64_t n = 0; n < N; n++) {
          int64_t src_k = kq * 4 + c;
          float val = w_ptr[src_k * N + n];
          int32_t q = static_cast<int32_t>(std::nearbyintf(val * inv_scales[n]));
          q = std::max(-128, std::min(127, q));
          dst[c * K4 * N + kq * N + n] = static_cast<int8_t>(q);
        }
      }
    }
  }
  utils::pack_staging_to_vtensor(staging.buffer(), v_weight);

  // Upload bias to Vulkan if present
  Tensor bias_vk;
  if (bias_arg && bias_arg->defined()) {
    bias_vk = bias_arg->vulkan();
  } else {
    bias_vk = at::zeros({1}, at::kFloat).vulkan();
  }

  // Upload per-channel scales to Vulkan as 1D tensor
  Tensor scale_cpu = at::from_blob(scales.data(), {N}, at::kFloat).clone();
  Tensor scale_vk = scale_cpu.vulkan();

  Tensor zp_t = at::full({1}, 0, at::kInt);

  return std::make_tuple(convert(v_weight), bias_vk, scale_vk, zp_t);
}

// y = x @ dequant(W_int8) + bias
// Per-channel dequantization with scales stored as a Vulkan texture.
// Bias is fused into the shader to avoid an extra dispatch.
Tensor run_q8_linear(
    const Tensor& input_arg,
    const Tensor& weight_arg,
    const Tensor& bias_arg,
    const Tensor& scale_arg,
    int64_t zero_point) {
  api::Context* const context = api::context();

  // Reshape to 2D for matmul
  Tensor input_2d = input_arg;
  if (input_arg.dim() > 2) {
    const auto d = c10::multiply_integers(
        input_arg.sizes().begin(), input_arg.sizes().end() - 1);
    input_2d = input_arg.reshape({d, input_arg.size(-1)});
  }

  const Tensor input = input_2d.is_vulkan() ? input_2d : input_2d.vulkan();

  vTensor v_input = convert(input);
  if (v_input.gpu_memory_layout() ==
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED) {
    v_input = packing::convert_image_channels_packed_to_width_packed(v_input);
  }

  const vTensor& v_weight = convert(weight_arg);
  const Tensor bias = bias_arg.is_vulkan() ? bias_arg : bias_arg.vulkan();
  const vTensor& v_bias = convert(bias);
  const Tensor scale = scale_arg.is_vulkan() ? scale_arg : scale_arg.vulkan();
  const vTensor& v_scale = convert(scale);

  const int64_t N = weight_arg.size(3);
  const int64_t K_over_4 = weight_arg.size(2);
  const bool has_bias = bias_arg.numel() > 1;

  vTensor v_output{context, {1, N}, api::ScalarType::Float};

  const struct {
    ivec4 shader_extents;
    int32_t has_bias;
  } block{
      {
          safe_downcast<int32_t>(N),
          1,
          1,
          safe_downcast<int32_t>(K_over_4),
      },
      has_bias ? 1 : 0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(matvec_q8),
      pipeline_barrier,
      {
          safe_downcast<uint32_t>(N),
          1,
          1,
      },
      {64, 1, 1},
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_scale.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  Tensor output = convert(v_output);

  if (input_arg.dim() > 2) {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(input_arg.dim()));
    for (int64_t i = 0; i < input_arg.dim() - 1; i++) {
      shape.push_back(input_arg.size(i));
    }
    shape.push_back(N);
    output = output.reshape(shape);
  }

  return output;
}

// Per-group quantization: each group of G elements along K has its own scale.
// G must divide K and be divisible by 4. Default G=128.
// Scale tensor: 2D (n_groups, N).
std::tuple<Tensor, Tensor, Tensor, Tensor> create_q8g_linear(
    const Tensor& weight_arg,
    const std::optional<Tensor>& bias_arg,
    int64_t group_size) {
  TORCH_CHECK(weight_arg.dim() == 2, "Weight must be 2D (K, N)");
  TORCH_CHECK(weight_arg.device().is_cpu(), "Weight must be on CPU");

  const Tensor weight = weight_arg.contiguous();
  const int64_t K = weight.size(0);
  const int64_t N = weight.size(1);
  TORCH_CHECK(K % 4 == 0, "K must be divisible by 4");
  TORCH_CHECK(group_size % 4 == 0, "group_size must be divisible by 4");
  TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");

  const float* w_ptr = weight.const_data_ptr<float>();
  const int64_t n_groups = K / group_size;

  // Per-group scales: scale[g, n] = max_abs(W[g*G:(g+1)*G, n]) / 127
  std::vector<float> scales(n_groups * N);
  std::vector<float> inv_scales(n_groups * N);
  for (int64_t g = 0; g < n_groups; g++) {
    for (int64_t n = 0; n < N; n++) {
      float grp_absmax = 0.0f;
      for (int64_t k = g * group_size; k < (g + 1) * group_size; k++) {
        grp_absmax = std::max(grp_absmax, std::abs(w_ptr[k * N + n]));
      }
      float s = grp_absmax > 0 ? grp_absmax / 127.0f : 1.0f;
      scales[g * N + n] = s;
      inv_scales[g * N + n] = 1.0f / s;
    }
  }

  api::Context* const context = api::context();

  const int64_t K4 = K / 4;
  vTensor v_weight{context, {1, 4, K4, N}, api::ScalarType::QInt8};
  v_weight.set_is_quantized();
  v_weight.set_scale(1.0);
  v_weight.set_zero_point(0);

  api::StorageBuffer staging(context, api::kFloat, v_weight.gpu_numel());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
    int8_t* dst = mapping.template data<int8_t>();
    memset(dst, 0, v_weight.gpu_numel() * sizeof(int8_t));

    for (int64_t c = 0; c < 4; c++) {
      for (int64_t kq = 0; kq < K4; kq++) {
        for (int64_t n = 0; n < N; n++) {
          int64_t src_k = kq * 4 + c;
          int64_t g = src_k / group_size;
          float inv_s = inv_scales[g * N + n];
          float val = w_ptr[src_k * N + n];
          int32_t q = static_cast<int32_t>(std::nearbyintf(val * inv_s));
          q = std::max(-128, std::min(127, q));
          dst[c * K4 * N + kq * N + n] = static_cast<int8_t>(q);
        }
      }
    }
  }
  utils::pack_staging_to_vtensor(staging.buffer(), v_weight);

  Tensor bias_vk;
  if (bias_arg && bias_arg->defined()) {
    bias_vk = bias_arg->vulkan();
  } else {
    bias_vk = at::zeros({1}, at::kFloat).vulkan();
  }

  // Upload 2D scale tensor (n_groups, N)
  Tensor scale_cpu = at::from_blob(scales.data(), {n_groups, N}, at::kFloat).clone();
  Tensor scale_vk = scale_cpu.vulkan();

  // Pack group_size into the zero_point return for API simplicity
  Tensor gs_t = at::full({1}, group_size / 4, at::kInt); // group_size in vec4 steps

  return std::make_tuple(convert(v_weight), bias_vk, scale_vk, gs_t);
}

Tensor run_q8g_linear(
    const Tensor& input_arg,
    const Tensor& weight_arg,
    const Tensor& bias_arg,
    const Tensor& scale_arg,
    int64_t group_size_k4) {
  api::Context* const context = api::context();

  Tensor input_2d = input_arg;
  if (input_arg.dim() > 2) {
    const auto d = c10::multiply_integers(
        input_arg.sizes().begin(), input_arg.sizes().end() - 1);
    input_2d = input_arg.reshape({d, input_arg.size(-1)});
  }

  const Tensor input = input_2d.is_vulkan() ? input_2d : input_2d.vulkan();

  vTensor v_input = convert(input);
  if (v_input.gpu_memory_layout() ==
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED) {
    v_input = packing::convert_image_channels_packed_to_width_packed(v_input);
  }

  const vTensor& v_weight = convert(weight_arg);
  const Tensor bias = bias_arg.is_vulkan() ? bias_arg : bias_arg.vulkan();
  const vTensor& v_bias = convert(bias);
  const Tensor scale = scale_arg.is_vulkan() ? scale_arg : scale_arg.vulkan();
  const vTensor& v_scale = convert(scale);

  const int64_t N = weight_arg.size(3);
  const int64_t K_over_4 = weight_arg.size(2);
  const bool has_bias = bias_arg.numel() > 1;

  vTensor v_output{context, {1, N}, api::ScalarType::Float};

  const struct {
    ivec4 shader_extents;
    int32_t has_bias;
    int32_t group_size_k4;
  } block{
      {
          safe_downcast<int32_t>(N),
          1,
          1,
          safe_downcast<int32_t>(K_over_4),
      },
      has_bias ? 1 : 0,
      safe_downcast<int32_t>(group_size_k4),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(matvec_q8g),
      pipeline_barrier,
      {
          safe_downcast<uint32_t>(N),
          1,
          1,
      },
      {64, 1, 1},
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_scale.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  Tensor output = convert(v_output);

  if (input_arg.dim() > 2) {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(input_arg.dim()));
    for (int64_t i = 0; i < input_arg.dim() - 1; i++) {
      shape.push_back(input_arg.size(i));
    }
    shape.push_back(N);
    output = output.reshape(shape);
  }

  return output;
}

// Per-group INT4 quantization: 2 int4 values packed per byte.
// Each byte stores: low nibble = (val + 8) & 0xF, high nibble = (val + 8) >> 4.
// Weight texture: (1, 4, K/8, N) QInt8 channels-packed.
// texel (n, k8, 0): 4 bytes → 8 int4 values covering K indices [k8*8 .. k8*8+7].
std::tuple<Tensor, Tensor, Tensor, Tensor> create_q4g_linear(
    const Tensor& weight_arg,
    const std::optional<Tensor>& bias_arg,
    int64_t group_size) {
  TORCH_CHECK(weight_arg.dim() == 2, "Weight must be 2D (K, N)");
  TORCH_CHECK(weight_arg.device().is_cpu(), "Weight must be on CPU");

  const Tensor weight = weight_arg.contiguous();
  const int64_t K = weight.size(0);
  const int64_t N = weight.size(1);
  TORCH_CHECK(K % 8 == 0, "K must be divisible by 8 for int4 packing");
  TORCH_CHECK(group_size % 4 == 0, "group_size must be divisible by 4");
  TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");

  const float* w_ptr = weight.const_data_ptr<float>();
  const int64_t n_groups = K / group_size;

  // Per-group scales: scale[g, n] = max_abs(W[g*G:(g+1)*G, n]) / 7
  std::vector<float> scales(n_groups * N);
  std::vector<float> inv_scales(n_groups * N);
  for (int64_t g = 0; g < n_groups; g++) {
    for (int64_t n = 0; n < N; n++) {
      float grp_absmax = 0.0f;
      for (int64_t k = g * group_size; k < (g + 1) * group_size; k++) {
        grp_absmax = std::max(grp_absmax, std::abs(w_ptr[k * N + n]));
      }
      float s = grp_absmax > 0 ? grp_absmax / 7.0f : 1.0f;
      scales[g * N + n] = s;
      inv_scales[g * N + n] = 1.0f / s;
    }
  }

  api::Context* const context = api::context();

  const int64_t K8 = K / 8;
  // Packed as (1, 4, K/8, N) with each byte holding 2 int4 values
  vTensor v_weight{context, {1, 4, K8, N}, api::ScalarType::QInt8};
  v_weight.set_is_quantized();
  v_weight.set_scale(1.0);
  v_weight.set_zero_point(0);

  api::StorageBuffer staging(context, api::kFloat, v_weight.gpu_numel());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
    int8_t* dst = mapping.template data<int8_t>();
    memset(dst, 0, v_weight.gpu_numel() * sizeof(int8_t));

    // Channel c stores: pack(w[k8*8 + c*2], w[k8*8 + c*2 + 1])
    for (int64_t c = 0; c < 4; c++) {
      for (int64_t k8 = 0; k8 < K8; k8++) {
        for (int64_t n = 0; n < N; n++) {
          int64_t src_k_lo = k8 * 8 + c * 2;
          int64_t src_k_hi = k8 * 8 + c * 2 + 1;
          int64_t g_lo = src_k_lo / group_size;
          int64_t g_hi = src_k_hi / group_size;

          float val_lo = w_ptr[src_k_lo * N + n];
          float val_hi = w_ptr[src_k_hi * N + n];
          int32_t q_lo = static_cast<int32_t>(std::nearbyintf(val_lo * inv_scales[g_lo * N + n]));
          int32_t q_hi = static_cast<int32_t>(std::nearbyintf(val_hi * inv_scales[g_hi * N + n]));
          q_lo = std::max(-8, std::min(7, q_lo));
          q_hi = std::max(-8, std::min(7, q_hi));

          // Pack: low nibble = (q_lo + 8), high nibble = (q_hi + 8)
          uint8_t packed = static_cast<uint8_t>(
              ((q_lo + 8) & 0xF) | (((q_hi + 8) & 0xF) << 4));
          dst[c * K8 * N + k8 * N + n] = static_cast<int8_t>(packed);
        }
      }
    }
  }
  utils::pack_staging_to_vtensor(staging.buffer(), v_weight);

  Tensor bias_vk;
  if (bias_arg && bias_arg->defined()) {
    bias_vk = bias_arg->vulkan();
  } else {
    bias_vk = at::zeros({1}, at::kFloat).vulkan();
  }

  Tensor scale_cpu = at::from_blob(scales.data(), {n_groups, N}, at::kFloat).clone();
  Tensor scale_vk = scale_cpu.vulkan();

  Tensor gs_t = at::full({1}, group_size / 4, at::kInt);

  return std::make_tuple(convert(v_weight), bias_vk, scale_vk, gs_t);
}

Tensor run_q4g_linear(
    const Tensor& input_arg,
    const Tensor& weight_arg,
    const Tensor& bias_arg,
    const Tensor& scale_arg,
    int64_t group_size_k4) {
  api::Context* const context = api::context();

  Tensor input_2d = input_arg;
  if (input_arg.dim() > 2) {
    const auto d = c10::multiply_integers(
        input_arg.sizes().begin(), input_arg.sizes().end() - 1);
    input_2d = input_arg.reshape({d, input_arg.size(-1)});
  }

  const Tensor input = input_2d.is_vulkan() ? input_2d : input_2d.vulkan();

  vTensor v_input = convert(input);
  if (v_input.gpu_memory_layout() ==
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED) {
    v_input = packing::convert_image_channels_packed_to_width_packed(v_input);
  }

  const vTensor& v_weight = convert(weight_arg);
  const Tensor bias = bias_arg.is_vulkan() ? bias_arg : bias_arg.vulkan();
  const vTensor& v_bias = convert(bias);
  const Tensor scale = scale_arg.is_vulkan() ? scale_arg : scale_arg.vulkan();
  const vTensor& v_scale = convert(scale);

  const int64_t N = weight_arg.size(3);
  // K/8 is the weight texture height; K/4 is the number of input vec4 steps
  const int64_t K_over_8 = weight_arg.size(2);
  const int64_t K_over_4 = K_over_8 * 2;
  const bool has_bias = bias_arg.numel() > 1;

  vTensor v_output{context, {1, N}, api::ScalarType::Float};

  const struct {
    ivec4 shader_extents;
    int32_t has_bias;
    int32_t group_size_k4;
  } block{
      {
          safe_downcast<int32_t>(N),
          1,
          1,
          safe_downcast<int32_t>(K_over_4),
      },
      has_bias ? 1 : 0,
      safe_downcast<int32_t>(group_size_k4),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(matvec_q4g),
      pipeline_barrier,
      {
          safe_downcast<uint32_t>(N),
          1,
          1,
      },
      {64, 1, 1},
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_scale.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  Tensor output = convert(v_output);

  if (input_arg.dim() > 2) {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(input_arg.dim()));
    for (int64_t i = 0; i < input_arg.dim() - 1; i++) {
      shape.push_back(input_arg.size(i));
    }
    shape.push_back(N);
    output = output.reshape(shape);
  }

  return output;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(vulkan_prepack, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_q8_linear"),
      create_q8_linear);
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_q8g_linear"),
      create_q8g_linear);
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_q4g_linear"),
      create_q4g_linear);
}

TORCH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_q8_linear"),
      run_q8_linear);
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_q8g_linear"),
      run_q8g_linear);
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_q4g_linear"),
      run_q4g_linear);
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
