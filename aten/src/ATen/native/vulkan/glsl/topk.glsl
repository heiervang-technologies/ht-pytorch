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
    float selected_val;
    int selected_idx;
    float prev_val = uBlock.largest != 0 ? 1.0/0.0 : -1.0/0.0;
    int prev_idx = -1;

    // For the first element (out_x == 0), find the global max/min
    // For subsequent elements, find the next max/min that's <= prev selection
    for (int rank = 0; rank <= out_x; rank++) {
      float best_val = uBlock.largest != 0 ? -1.0/0.0 : 1.0/0.0;
      int best_idx = 0;

      for (int i = 0; i < uBlock.dim_size; i++) {
        ivec3 read_pos = ivec3(i, pos.y, pos.z);
        float val = texelFetch(uInput, read_pos, 0).x;

        bool is_better;
        if (uBlock.largest != 0) {
          is_better = val > best_val;
        } else {
          is_better = val < best_val;
        }

        // Skip values we've already selected (by checking against threshold)
        bool is_valid;
        if (rank == 0) {
          is_valid = true;
        } else {
          if (uBlock.largest != 0) {
            is_valid = (val < prev_val) || (val == prev_val && i > prev_idx);
          } else {
            is_valid = (val > prev_val) || (val == prev_val && i > prev_idx);
          }
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

    imageStore(uValues, pos, vec4(selected_val, 0.0, 0.0, 0.0));
    imageStore(uIndices, pos, vec4(float(selected_idx), 0.0, 0.0, 0.0));
  }
}
