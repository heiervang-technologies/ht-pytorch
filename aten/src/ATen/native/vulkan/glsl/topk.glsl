#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

// Output: top-k values
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uValues;
// Output: top-k indices (stored as float for image compatibility)
layout(set = 0, binding = 1, FORMAT) uniform PRECISION restrict writeonly image3D uIndices;
// Input tensor
layout(set = 0, binding = 2) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 out_extents;
  int k;
  int dim_size;
  int largest;
  int sorted;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.out_extents.xyz))) {
    // For simplicity, this shader handles topk along the width (x) dimension.
    // Each invocation handles one (y, z) position and finds top-k across x.
    //
    // This is a naive O(k*n) selection. For large k or n, a more sophisticated
    // approach (partial sort, heap) would be needed.

    int out_x = pos.x;
    if (out_x >= uBlock.k) return;

    // Collect values along the x dimension for this (y, z)
    // Use a simple selection: find the out_x-th largest/smallest
    vec4 selected_values;
    vec4 selected_indices;
    for (int lane = 0; lane < 4; lane++) {
      float selected_val = 0.0;
      int selected_idx = 0;
      float prev_val = uBlock.largest != 0 ? 1.0/0.0 : -1.0/0.0;
      int prev_idx = -1;

      // For the first element (out_x == 0), find the global max/min.
      // Subsequent elements find the next value beyond the prior selection.
      for (int rank = 0; rank <= out_x; rank++) {
        float best_val =
            uBlock.largest != 0 ? -1.0/0.0 : 1.0/0.0;
        int best_idx = 0;

        for (int i = 0; i < uBlock.dim_size; i++) {
          ivec3 read_pos = ivec3(i, pos.y, pos.z);
          float val = texelFetch(uInput, read_pos, 0)[lane];

          bool is_better = uBlock.largest != 0
              ? val > best_val
              : val < best_val;
          bool is_valid;
          if (rank == 0) {
            is_valid = true;
          } else if (uBlock.largest != 0) {
            is_valid =
                (val < prev_val) || (val == prev_val && i > prev_idx);
          } else {
            is_valid =
                (val > prev_val) || (val == prev_val && i > prev_idx);
          }

          if (is_better && is_valid) {
            best_val = val;
            best_idx = i;
          }
        }
        selected_val = best_val;
        selected_idx = best_idx;
        prev_val = best_val;
        prev_idx = best_idx;
      }
      selected_values[lane] = selected_val;
      selected_indices[lane] = float(selected_idx);
    }

    imageStore(uValues, pos, selected_values);
    imageStore(uIndices, pos, selected_indices);
  }
}
