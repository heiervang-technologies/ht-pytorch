#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

// Fused RMS normalization: y = x * rsqrt(mean(x^2) + eps) * weight
// One workgroup per row. Reduction via shared memory.
// Operates on channels-packed tensors (default Vulkan layout).
// For 3D (B, S, D) with B small: width=D, height=S, depth=ceil(B/4)
// Each texel lane holds an independent row.

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uWeight;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 extents;
  // x: D (width), y: S (height), z: depth
  // w: unused
}
uBlock;

#define WG_SIZE 256

layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

shared vec4 partial_sums[WG_SIZE];

void main() {
  const uint tid = gl_LocalInvocationID.x;
  const int y = int(gl_WorkGroupID.x);
  const int z = int(gl_WorkGroupID.y);
  const int D = uBlock.extents.x;

  // Phase 1: compute partial sum of squares over the width dimension
  vec4 sum_sq = vec4(0.0);
  for (int w = int(tid); w < D; w += WG_SIZE) {
    vec4 val = texelFetch(uInput, ivec3(w, y, z), 0);
    sum_sq += val * val;
  }
  partial_sums[tid] = sum_sq;
  barrier();

  // Parallel reduction
  for (uint stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial_sums[tid] += partial_sums[tid + stride];
    }
    barrier();
  }

  // Phase 2: normalize and apply weight
  vec4 rms_inv = inversesqrt(partial_sums[0] / float(D) + vec4(1e-5));

  for (int w = int(tid); w < D; w += WG_SIZE) {
    vec4 val = texelFetch(uInput, ivec3(w, y, z), 0);
    float wt = texelFetch(uWeight, ivec3(w, 0, 0), 0).x;
    imageStore(uOutput, ivec3(w, y, z), val * rms_inv * wt);
  }
}
