#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uSelf;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Indices {
  int data[];
}
uIndices;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uSrc;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 out_extents;
  // Sizes are W, H, C, N; dim is in padded NCHW order.
  ivec4 self_sizes;
  ivec4 index_sizes;
  ivec4 src_sizes;
  int dim;
  int src_dim_size;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.out_extents.xyz))) {
    vec4 val = texelFetch(uSelf, pos, 0);
    const int self_c4 = (uBlock.self_sizes.z + 3) / 4;
    const int n = pos.z / self_c4;
    const int c_base = (pos.z % self_c4) * 4;

    for (int lane = 0; lane < 4; lane++) {
      const int c = c_base + lane;
      if (c >= uBlock.self_sizes.z) {
        break;
      }

      const ivec4 output_logical = ivec4(n, c, pos.y, pos.x);
      const int target = output_logical[uBlock.dim];
      for (int i = 0; i < uBlock.src_dim_size; i++) {
        ivec4 source_logical = output_logical;
        source_logical[uBlock.dim] = i;
        if (source_logical.x >= uBlock.index_sizes.w ||
            source_logical.y >= uBlock.index_sizes.z ||
            source_logical.z >= uBlock.index_sizes.y ||
            source_logical.w >= uBlock.index_sizes.x) {
          continue;
        }

        const int linear_index =
            ((source_logical.x * uBlock.index_sizes.z + source_logical.y) *
                 uBlock.index_sizes.y +
             source_logical.z) *
                uBlock.index_sizes.x +
            source_logical.w;
        if (uIndices.data[linear_index] == target) {
          const int source_c4 = (uBlock.src_sizes.z + 3) / 4;
          const ivec3 source_pos = ivec3(
              source_logical.w,
              source_logical.z,
              source_logical.x * source_c4 + source_logical.y / 4);
          val[lane] =
              texelFetch(uSrc, source_pos, 0)[source_logical.y % 4];
        }
      }
    }

    imageStore(uOutput, pos, val);
  }
}
