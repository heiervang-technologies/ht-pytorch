#version 450 core
#define PRECISION ${PRECISION}

layout(std430) buffer;

layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION restrict Block {
  ivec4 extents;
  int value;
  int fill0;
  int fill1;
  int fill2;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.extents.xyz))) {
    imageStore(uOutput, pos, ivec4(uBlock.value));
  }
}
