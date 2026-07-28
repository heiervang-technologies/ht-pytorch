#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

// Fused SiLU(gate) * up: y = (gate / (1 + exp(-gate))) * up
// Input: concatenated [gate, up] along last dim (width).
// Output: half the width of input.
// Channels-packed layout.

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 extents;
  // x: half_width (output width), y: height, z: depth, w: unused
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (pos.x >= uBlock.extents.x || pos.y >= uBlock.extents.y || pos.z >= uBlock.extents.z) {
    return;
  }

  const int half_w = uBlock.extents.x;
  vec4 gate = texelFetch(uInput, pos, 0);
  vec4 up =
      texelFetch(uInput, ivec3(pos.x + half_w, pos.y, pos.z), 0);

  vec4 silu_gate = gate / (vec4(1.0) + exp(-gate));
  imageStore(uOutput, pos, silu_gate * up);
}
