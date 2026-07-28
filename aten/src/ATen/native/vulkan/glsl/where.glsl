#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#include "texel_access.h"

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uCondition;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uSelf;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uOther;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 output_sizes;
  ivec4 condition_sizes;
  ivec4 self_sizes;
  ivec4 other_sizes;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec3 output_extents = ivec3(
      uBlock.output_sizes.x,
      uBlock.output_sizes.y,
      uBlock.output_sizes.w * ((uBlock.output_sizes.z + 3) / 4));
  if (all(lessThan(pos, output_extents))) {
    const ivec3 condition_pos = map_output_pos_to_input_pos(
        pos, uBlock.output_sizes, uBlock.condition_sizes);
    const ivec3 self_pos = map_output_pos_to_input_pos(
        pos, uBlock.output_sizes, uBlock.self_sizes);
    const ivec3 other_pos = map_output_pos_to_input_pos(
        pos, uBlock.output_sizes, uBlock.other_sizes);

    const vec4 cond = load_texel(
        condition_pos,
        uBlock.output_sizes,
        uBlock.condition_sizes,
        uCondition);
    const vec4 self_val = load_texel(
        self_pos, uBlock.output_sizes, uBlock.self_sizes, uSelf);
    const vec4 other_val = load_texel(
        other_pos, uBlock.output_sizes, uBlock.other_sizes, uOther);

    // condition is treated as boolean: nonzero = true
    vec4 result;
    result.x = (cond.x != 0.0) ? self_val.x : other_val.x;
    result.y = (cond.y != 0.0) ? self_val.y : other_val.y;
    result.z = (cond.z != 0.0) ? self_val.z : other_val.z;
    result.w = (cond.w != 0.0) ? self_val.w : other_val.w;

    imageStore(uOutput, pos, result);
  }
}
