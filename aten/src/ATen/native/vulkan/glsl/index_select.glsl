#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Indices {
  int data[];
}
uIndices;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 out_extents;
  // dim: 0=width(x), 1=height(y), 2=channels(z)
  int dim;
  int num_indices;
  int in_dim_size;
  int fill0;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.out_extents.xyz))) {
    // Determine which index along the selected dim this output position maps to
    int idx_along_dim;
    if (uBlock.dim == 0) {
      idx_along_dim = pos.x;
    } else if (uBlock.dim == 1) {
      idx_along_dim = pos.y;
    } else {
      idx_along_dim = pos.z;
    }

    int src_idx = uIndices.data[idx_along_dim];

    ivec3 src_pos = pos;
    if (uBlock.dim == 0) {
      src_pos.x = src_idx;
    } else if (uBlock.dim == 1) {
      src_pos.y = src_idx;
    } else {
      src_pos.z = src_idx;
    }

    vec4 val = texelFetch(uInput, src_pos, 0);
    imageStore(uOutput, pos, val);
  }
}
