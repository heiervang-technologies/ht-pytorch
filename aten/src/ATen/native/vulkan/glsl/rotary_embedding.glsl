#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

// Fused rotary positional embedding (RoPE).
// x_out[..., :half] = x[..., :half] * cos - x[..., half:] * sin
// x_out[..., half:] = x[..., half:] * cos + x[..., :half] * sin
//
// Channels-packed 3D tensors: (num_heads, seq_len, head_dim)
// Batch dim (num_heads) packed in groups of 4 along z-axis.
// cos, sin: (seq_len, head_dim/2) — batch dim = 1, packed z = 1

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uCos;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uSin;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 extents;
  // x: head_dim, y: seq_len, z: ceil(num_heads/4)
  // w: half = head_dim / 2
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.extents.x || pos.y >= uBlock.extents.y ||
      pos.z >= uBlock.extents.z) {
    return;
  }

  const int half_dim = uBlock.extents.w;
  const int w = pos.x;

  // Read all 4 channels (4 heads packed in this z-plane)
  vec4 x_val = texelFetch(uInput, pos, 0);
  vec4 result;

  if (w < half_dim) {
    // First half: x1 * cos - x2 * sin
    vec4 x2 = texelFetch(uInput, ivec3(w + half_dim, pos.y, pos.z), 0);
    float c = texelFetch(uCos, ivec3(w, pos.y, 0), 0).x;
    float s = texelFetch(uSin, ivec3(w, pos.y, 0), 0).x;
    result = x_val * c - x2 * s;
  } else {
    // Second half: x2 * cos + x1 * sin
    vec4 x1 = texelFetch(uInput, ivec3(w - half_dim, pos.y, pos.z), 0);
    float c = texelFetch(uCos, ivec3(w - half_dim, pos.y, 0), 0).x;
    float s = texelFetch(uSin, ivec3(w - half_dim, pos.y, 0), 0).x;
    result = x_val * c + x1 * s;
  }

  imageStore(uOutput, pos, result);
}
