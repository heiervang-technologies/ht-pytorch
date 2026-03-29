#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uCondition;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uSelf;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uOther;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec3 extents;
  int fill0;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.extents))) {
    const vec4 cond = texelFetch(uCondition, pos, 0);
    const vec4 self_val = texelFetch(uSelf, pos, 0);
    const vec4 other_val = texelFetch(uOther, pos, 0);

    // condition is treated as boolean: nonzero = true
    vec4 result;
    result.x = (cond.x != 0.0) ? self_val.x : other_val.x;
    result.y = (cond.y != 0.0) ? self_val.y : other_val.y;
    result.z = (cond.z != 0.0) ? self_val.z : other_val.z;
    result.w = (cond.w != 0.0) ? self_val.w : other_val.w;

    imageStore(uOutput, pos, result);
  }
}
