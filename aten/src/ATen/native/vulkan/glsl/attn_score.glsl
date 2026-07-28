#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

// Fused attention score: out[b,0,s] = scale * dot(Q[b,0,:], K[b,s,:])
// Q: (B, 1, D) channels-packed along B
// K: (B_kv, S_kv, D) channels-packed along B_kv
// Output: (B, 1, S_kv) channels-packed along B
//
// Avoids K transpose + contiguous dispatch.
// Supports GQA: B_q heads map to B_kv KV heads via h * n_kv / n_q.

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uQ;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uK;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 extents;
  // x: S_kv, y: D (head_dim), z: ceil(B/4), w: ceil(B_kv/4)
  float scale;
  int n_heads;
  int n_kv_heads;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  // pos.x = s_kv position, pos.z = packed head group
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (pos.x >= uBlock.extents.x || pos.z >= uBlock.extents.z) {
    return;
  }

  const int s_kv = pos.x;
  const int z_q = pos.z;
  const int D = uBlock.extents.y;
  const int n_heads = uBlock.n_heads;
  const int n_kv_heads = uBlock.n_kv_heads;

  // Compute dot product for 4 heads packed in this z-plane
  vec4 dot_val = vec4(0.0);

  if (n_heads == n_kv_heads) {
    // MHA: Q and K have same head count, same z-plane
    for (int d = 0; d < D; d++) {
      vec4 q_val = texelFetch(uQ, ivec3(d, 0, z_q), 0);
      vec4 k_val = texelFetch(uK, ivec3(d, s_kv, z_q), 0);
      dot_val += q_val * k_val;
    }
  } else {
    // GQA: map each Q head to its KV head
    int h_base = z_q * 4;
    for (int d = 0; d < D; d++) {
      vec4 q_val = texelFetch(uQ, ivec3(d, 0, z_q), 0);
      // Get K values from corresponding KV heads
      vec4 k_vals;
      for (int i = 0; i < 4; i++) {
        int h = h_base + i;
        if (h >= n_heads) break;
        int kv_h = h * n_kv_heads / n_heads;
        int kv_z = kv_h / 4;
        int kv_c = kv_h % 4;
        vec4 k_texel = texelFetch(uK, ivec3(d, s_kv, kv_z), 0);
        float k_val;
        if (kv_c == 0) k_val = k_texel.x;
        else if (kv_c == 1) k_val = k_texel.y;
        else if (kv_c == 2) k_val = k_texel.z;
        else k_val = k_texel.w;

        if (i == 0) k_vals.x = k_val;
        else if (i == 1) k_vals.y = k_val;
        else if (i == 2) k_vals.z = k_val;
        else k_vals.w = k_val;
      }
      dot_val += q_val * k_vals;
    }
  }

  dot_val *= uBlock.scale;
  imageStore(uOutput, ivec3(s_kv, 0, z_q), dot_val);
}
