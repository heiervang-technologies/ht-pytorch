#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define OP(X, Y) ${OPERATOR}
// clang-format on

layout(std430) buffer;

vec4 signed_pow(const vec4 base, const vec4 exponent) {
  const vec4 magnitude = pow(abs(base), exponent);
  const vec4 negative_base = vec4(lessThan(base, vec4(0.0)));
  const vec4 finite_exponent =
      vec4(not(isinf(exponent))) * vec4(not(isnan(exponent)));
  const vec4 integer_exponent =
      vec4(equal(exponent, trunc(exponent))) * finite_exponent;
  const vec4 odd_exponent =
      vec4(notEqual(mod(abs(exponent), vec4(2.0)), vec4(0.0)));
  const vec4 sign =
      vec4(1.0) - vec4(2.0) * negative_base * integer_exponent * odd_exponent;
  const vec4 invalid =
      negative_base * finite_exponent * (vec4(1.0) - integer_exponent);
  return mix(
      magnitude * sign, sqrt(vec4(-1.0)), notEqual(invalid, vec4(0.0)));
}

// clang-format off
$if not INPLACE:
  layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
  layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
  layout(set = 0, binding = 2) uniform PRECISION restrict Block {
    ivec4 extents;
    // scalar argument
    float other;
  }
  uArgs;
$else:
  layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uOutput;
  layout(set = 0, binding = 1) uniform PRECISION restrict Block {
    ivec4 extents;
    // scalar argument
    float other;
  }
  uArgs;
// clang-format on

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/**
 * Performs a binary elementwise operation between uInput and uArgs.other,
 * writing the output to uOutput.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, uArgs.extents.xyz))) {
    return;
  }

  vec4 v_other = vec4(uArgs.other);

  // clang-format off
  $if not INPLACE:
    vec4 v = texelFetch(uInput, pos, 0);
    vec4 out_texel = OP(v, v_other);
  $else:
    vec4 out_texel = imageLoad(uOutput, pos);
    out_texel = OP(out_texel, v_other);
  // clang-format on

  imageStore(uOutput, pos, out_texel);
}
