#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

// Fused attention × value with GQA support.
// attn: (B, 1, S_kv) channels-packed along B  (after softmax)
// V: (B_kv, S_kv, D) channels-packed along B_kv
// Output: (B, 1, D) channels-packed along B
//
// out[b,0,d] = sum_s attn[b,0,s] * V[kv_h(b), s, d]
// Eliminates repeat_kv for V.

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uAttn;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uV;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 extents;
  // x: D (head_dim), y: S_kv, z: ceil(B/4), w: unused
  int n_heads;
  int n_kv_heads;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  // pos.x = d (head_dim index), pos.z = packed head group
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (pos.x >= uBlock.extents.x || pos.z >= uBlock.extents.z) {
    return;
  }

  const int d = pos.x;
  const int z_q = pos.z;
  const int S_kv = uBlock.extents.y;
  const int n_heads = uBlock.n_heads;
  const int n_kv_heads = uBlock.n_kv_heads;

  vec4 result = vec4(0.0);

  if (n_heads == n_kv_heads) {
    // MHA: same z-plane
    for (int s = 0; s < S_kv; s++) {
      vec4 a = texelFetch(uAttn, ivec3(s, 0, z_q), 0);
      vec4 v = texelFetch(uV, ivec3(d, s, z_q), 0);
      result += a * v;
    }
  } else {
    // GQA: gather V from KV heads
    int h_base = z_q * 4;
    for (int s = 0; s < S_kv; s++) {
      vec4 a = texelFetch(uAttn, ivec3(s, 0, z_q), 0);
      vec4 v_vals;
      for (int i = 0; i < 4; i++) {
        int h = h_base + i;
        if (h >= n_heads) break;
        int kv_h = h * n_kv_heads / n_heads;
        int kv_z = kv_h / 4;
        int kv_c = kv_h % 4;
        vec4 v_texel = texelFetch(uV, ivec3(d, s, kv_z), 0);
        float v_val;
        if (kv_c == 0) v_val = v_texel.x;
        else if (kv_c == 1) v_val = v_texel.y;
        else if (kv_c == 2) v_val = v_texel.z;
        else v_val = v_texel.w;

        if (i == 0) v_vals.x = v_val;
        else if (i == 1) v_vals.y = v_val;
        else if (i == 2) v_vals.z = v_val;
        else v_vals.w = v_val;
      }
      result += a * v_vals;
    }
  }

  imageStore(uOutput, ivec3(d, 0, z_q), result);
}
