#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

// Weight-only int4 matrix-vector multiplication with per-group quantization.
// y = x @ dequant(W) + bias
// Weights packed as 2 int4 values per byte (unsigned offset format).
// Per-group dequantization: groups of G elements along K, each with own scale.
// Weight texture shape: (1, 4, K/8, N) QInt8 channels-packed.
// Each texel holds 8 int4 values covering 8 consecutive K steps for one N.

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION isampler3D uWeight;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uBias;
layout(set = 0, binding = 4) uniform PRECISION sampler3D uScale;
layout(set = 0, binding = 5) uniform PRECISION restrict Block {
  ivec4 shader_extents;
  // x: N (output width), y: 1, z: batch
  // w: K/4 (number of vec4 steps along input/reduction dimension)
  int has_bias;
  int group_size_k4; // G/4 = group size in vec4 steps
}
uBlock;

#define LOCAL_SIZE 64
#define TILE_K 64

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

shared vec4 input_tile[TILE_K];

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const uint tid = gl_LocalInvocationID.x;

  const int k4_steps = uBlock.shader_extents.w;
  // Weight texture has K/8 height (each texel = 8 int4 = 2 vec4 steps)
  const int k8_steps = k4_steps / 2;
  const bool in_bounds = pos.x < uBlock.shader_extents.x &&
                         pos.z < uBlock.shader_extents.z;

  float result = 0.0;
  const int g_k4 = uBlock.group_size_k4;

  // Iterate over K in tiles of TILE_K vec4 steps (= TILE_K/2 weight texels)
  for (int tile_start = 0; tile_start < k4_steps; tile_start += TILE_K) {
    // Cooperatively load float input tile into shared memory
    for (uint i = tid; i < TILE_K; i += LOCAL_SIZE) {
      int k_idx = tile_start + int(i);
      if (k_idx < k4_steps) {
        input_tile[i] = texelFetch(uInput, ivec3(k_idx, 0, pos.z), 0);
      } else {
        input_tile[i] = vec4(0.0);
      }
    }
    barrier();

    if (in_bounds) {
      int tile_k4_end = min(TILE_K, k4_steps - tile_start);
      int tile_k8_end = tile_k4_end / 2;
      int prev_group = -1;
      float w_scale = 0.0;

      for (int j = 0; j < tile_k8_end; j++) {
        int global_k8 = (tile_start / 2) + j;
        int global_k4 = tile_start + j * 2;

        // Group boundary check (based on vec4 index)
        int group = global_k4 / g_k4;
        if (group != prev_group) {
          w_scale = texelFetch(uScale, ivec3(pos.x, group, 0), 0).x;
          prev_group = group;
        }

        // Read packed int4 weight texel (4 bytes = 8 int4 values)
        ivec4 w_packed = texelFetch(uWeight, ivec3(pos.x, global_k8, pos.z), 0);

        // Unpack: each byte has 2 int4 values (low nibble = even K, high = odd K)
        // Unsigned offset: stored as [0,15], subtract 8 to get [-8,7]
        int b0 = w_packed.x & 0xFF;
        int b1 = w_packed.y & 0xFF;
        int b2 = w_packed.z & 0xFF;
        int b3 = w_packed.w & 0xFF;

        // First vec4: K indices k8*8+0..3
        vec4 w_a = w_scale * vec4(
            float((b0 & 0xF) - 8),
            float((b0 >> 4) - 8),
            float((b1 & 0xF) - 8),
            float((b1 >> 4) - 8)
        );
        // Second vec4: K indices k8*8+4..7
        vec4 w_b = w_scale * vec4(
            float((b2 & 0xF) - 8),
            float((b2 >> 4) - 8),
            float((b3 & 0xF) - 8),
            float((b3 >> 4) - 8)
        );

        result += dot(input_tile[j * 2], w_a);
        result += dot(input_tile[j * 2 + 1], w_b);
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
