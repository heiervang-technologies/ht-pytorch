#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 extents;
  int diagonal;
  int width;
  int height;
  int fill0;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.extents.xyz))) {
    vec4 val = texelFetch(uInput, pos, 0);
    // pos.x = width (col), pos.y = height (row)
    int col = pos.x;
    int row = pos.y;
    // triu: zero elements where col < row + diagonal
    if (col < row + uBlock.diagonal) {
      val = vec4(0.0);
    }
    imageStore(uOutput, pos, val);
  }
}
