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
  // Sizes are W, H, C, N; dim is in padded NCHW order.
  ivec4 self_sizes;
  ivec4 index_sizes;
  int dim;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.out_extents.xyz))) {
    const int index_c4 = (uBlock.index_sizes.z + 3) / 4;
    const int n = pos.z / index_c4;
    const int c_base = (pos.z % index_c4) * 4;
    vec4 result = vec4(0.0);

    for (int lane = 0; lane < 4; lane++) {
      const int c = c_base + lane;
      if (c >= uBlock.index_sizes.z) {
        break;
      }

      const int linear_index =
          ((n * uBlock.index_sizes.z + c) * uBlock.index_sizes.y + pos.y) *
              uBlock.index_sizes.x +
          pos.x;
      const int index = uIndices.data[linear_index];

      ivec4 logical_pos = ivec4(n, c, pos.y, pos.x);
      logical_pos[uBlock.dim] = index;
      const int src_c4 = (uBlock.self_sizes.z + 3) / 4;
      const ivec3 src_pos = ivec3(
          logical_pos.w,
          logical_pos.z,
          logical_pos.x * src_c4 + logical_pos.y / 4);
      result[lane] = texelFetch(uInput, src_pos, 0)[logical_pos.y % 4];
    }

    imageStore(uOutput, pos, result);
  }
}
