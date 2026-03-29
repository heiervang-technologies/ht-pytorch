#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uSelf;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uIndex;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uSrc;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 out_extents;
  int dim;
  int src_dim_size;
  int fill0;
  int fill1;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.out_extents.xyz))) {
    // Start with the self value
    vec4 val = texelFetch(uSelf, pos, 0);

    // Check all positions in the src along the scatter dim to see if any
    // index maps to our position
    for (int i = 0; i < uBlock.src_dim_size; i++) {
      ivec3 idx_pos = pos;
      if (uBlock.dim == 0) {
        idx_pos.x = i;
      } else if (uBlock.dim == 1) {
        idx_pos.y = i;
      } else {
        idx_pos.z = i;
      }

      vec4 idx_val = texelFetch(uIndex, idx_pos, 0);
      vec4 src_val = texelFetch(uSrc, idx_pos, 0);

      // For each component, check if the index matches our position
      int target;
      if (uBlock.dim == 0) {
        target = pos.x;
      } else if (uBlock.dim == 1) {
        target = pos.y;
      } else {
        target = pos.z;
      }

      // Only first component matters for non-packed dimensions
      if (int(idx_val.x) == target) {
        val.x = src_val.x;
      }
    }

    imageStore(uOutput, pos, val);
  }
}
