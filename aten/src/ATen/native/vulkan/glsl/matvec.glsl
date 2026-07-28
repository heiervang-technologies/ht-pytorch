#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

// Matrix-vector multiplication: y = x @ W
// Optimized for M=1 (single token decode).
// Input x: (1, K) width-packed → image3D width = K/4
// Weight W: (K, N) height-packed → image3D width = N, height = K/4
// Output y: (1, N)
//
// Each thread computes one output element. Input vector is cooperatively
// loaded into shared memory in tiles to reduce texture fetches.

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uWeight;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 shader_extents;
  // x: output width (N), y: 1, z: batch
  // w: K/4 (number of vec4 steps along reduction dimension)
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

  // Process K dimension in tiles of TILE_K vec4s
  for (int tile_start = 0; tile_start < k_steps; tile_start += TILE_K) {
    // Cooperatively load input vector tile into shared memory
    for (uint i = tid; i < TILE_K; i += LOCAL_SIZE) {
      int k_idx = tile_start + int(i);
      if (k_idx < k_steps) {
        input_tile[i] = texelFetch(uInput, ivec3(k_idx, 0, pos.z), 0);
      } else {
        input_tile[i] = vec4(0.0);
      }
    }
    barrier();

    // Each thread dots its weight column against the cached input tile
    if (in_bounds) {
      int tile_end = min(TILE_K, k_steps - tile_start);
      for (int i = 0; i < tile_end; i++) {
        vec4 w_val = texelFetch(uWeight, ivec3(pos.x, tile_start + i, pos.z), 0);
        result += dot(input_tile[i], w_val);
      }
    }
    barrier();
  }

  if (in_bounds) {
    imageStore(uOutput, ivec3(pos.x, 0, pos.z), vec4(result, 0.0, 0.0, 0.0));
  }
}
