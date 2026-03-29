#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uIndex;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 out_extents;
  // dim: 0=width(x), 1=height(y), 2=channels(z)
  int dim;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.out_extents.xyz))) {
    // Read the index value at this position
    vec4 idx_vec = texelFetch(uIndex, pos, 0);

    // For each component in the vec4, gather from input along dim
    vec4 result;
    for (int c = 0; c < 4; c++) {
      int idx = int(idx_vec[c]);
      ivec3 src_pos = pos;
      if (uBlock.dim == 0) {
        src_pos.x = idx;
      } else if (uBlock.dim == 1) {
        src_pos.y = idx;
      } else {
        src_pos.z = idx;
      }
      result[c] = texelFetch(uInput, src_pos, 0)[c];
    }

    imageStore(uOutput, pos, result);
  }
}
