#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

// Weight-only int8 matrix-vector multiplication with per-group quantization.
// y = x @ dequant(W) + bias
// Per-group dequantization: groups of G=128 along K dimension, each with own scale.
// scale_texture shape: (K/G, N) stored as channels-packed image3D.

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION isampler3D uWeight;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uBias;
layout(set = 0, binding = 4) uniform PRECISION sampler3D uScale;
layout(set = 0, binding = 5) uniform PRECISION restrict Block {
  ivec4 shader_extents;
  // x: output width (N), y: 1, z: batch
  // w: K/4 (number of vec4 steps along reduction dimension)
  int has_bias;
  int group_size_k4; // G/4 = group size in vec4 steps (32 for G=128)
}
uBlock;

#define LOCAL_SIZE 64
#define TILE_K 64

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

shared vec4 input_tile[TILE_K];

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const uint tid = gl_LocalInvocationID.x;

  const int k_steps = uBlock.shader_extents.w;
  const bool in_bounds = pos.x < uBlock.shader_extents.x &&
                         pos.z < uBlock.shader_extents.z;

  float result = 0.0;
  const int g_k4 = uBlock.group_size_k4;

  for (int tile_start = 0; tile_start < k_steps; tile_start += TILE_K) {
    // Cooperatively load float input tile into shared memory
    for (uint i = tid; i < TILE_K; i += LOCAL_SIZE) {
      int k_idx = tile_start + int(i);
      if (k_idx < k_steps) {
        input_tile[i] = texelFetch(uInput, ivec3(k_idx, 0, pos.z), 0);
      } else {
        input_tile[i] = vec4(0.0);
      }
    }
    barrier();

    if (in_bounds) {
      int tile_end = min(TILE_K, k_steps - tile_start);
      int prev_group = -1;
      float w_scale = 0.0;

      for (int i = 0; i < tile_end; i++) {
        // Determine which group this k4 index belongs to
        int global_k4 = tile_start + i;
        int group = global_k4 / g_k4;

        // Load scale when crossing group boundary
        if (group != prev_group) {
          w_scale = texelFetch(uScale, ivec3(pos.x, group, 0), 0).x;
          prev_group = group;
        }

        ivec4 w_int = texelFetch(uWeight, ivec3(pos.x, global_k4, pos.z), 0);
        vec4 w_float = w_scale * vec4(w_int);
        result += dot(input_tile[i], w_float);
      }
    }
    barrier();
  }

  if (in_bounds) {
    if (uBlock.has_bias != 0) {
      result += texelFetch(uBias, ivec3(pos.x, 0, 0), 0).x;
    }
    imageStore(uOutput, ivec3(pos.x, 0, pos.z), vec4(result, 0.0, 0.0, 0.0));
  }
}
