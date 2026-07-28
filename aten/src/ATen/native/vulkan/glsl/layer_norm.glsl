#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

// Fused Layer Normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
// One workgroup per row. Two-pass reduction via shared memory.
// Channels-packed layout: width=D, height=S, depth=ceil(B/4).

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uWeight;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uBias;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 extents;
  // x: D (width), y: S (height), z: depth, w: unused
}
uBlock;

#define WG_SIZE 256

layout(local_size_x = WG_SIZE, local_size_y = 1, local_size_z = 1) in;

shared vec4 partial[WG_SIZE];

void main() {
  const uint tid = gl_LocalInvocationID.x;
  const int y = int(gl_WorkGroupID.x);
  const int z = int(gl_WorkGroupID.y);
  const int D = uBlock.extents.x;

  // Pass 1: compute mean
  vec4 sum_val = vec4(0.0);
  for (int w = int(tid); w < D; w += WG_SIZE) {
    sum_val += texelFetch(uInput, ivec3(w, y, z), 0);
  }
  partial[tid] = sum_val;
  barrier();

  for (uint stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    barrier();
  }

  vec4 mean = partial[0] / float(D);

  // Pass 2: compute variance
  vec4 sum_sq = vec4(0.0);
  for (int w = int(tid); w < D; w += WG_SIZE) {
    vec4 diff = texelFetch(uInput, ivec3(w, y, z), 0) - mean;
    sum_sq += diff * diff;
  }
  partial[tid] = sum_sq;
  barrier();

  for (uint stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    barrier();
  }

  vec4 inv_std = inversesqrt(partial[0] / float(D) + vec4(1e-5));

  // Pass 3: normalize, scale, shift
  for (int w = int(tid); w < D; w += WG_SIZE) {
    vec4 val = texelFetch(uInput, ivec3(w, y, z), 0);
    float wt = texelFetch(uWeight, ivec3(w, 0, 0), 0).x;
    float bi = texelFetch(uBias, ivec3(w, 0, 0), 0).x;
    vec4 normed = (val - mean) * inv_std * wt + vec4(bi);
    imageStore(uOutput, ivec3(w, y, z), normed);
  }
}
