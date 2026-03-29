#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)         uniform PRECISION restrict           Block {
  ivec4 size;
  float unused;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Abramowitz & Stegun approximation of erf (max error ~1.5e-7)
vec4 erf_approx(vec4 x) {
  vec4 a = abs(x);
  vec4 t = vec4(1.0) / (vec4(1.0) + vec4(0.3275911) * a);
  vec4 t2 = t * t;
  vec4 t3 = t2 * t;
  vec4 t4 = t3 * t;
  vec4 t5 = t4 * t;
  vec4 poly = vec4(0.254829592) * t
            - vec4(0.284496736) * t2
            + vec4(1.421413741) * t3
            - vec4(1.453152027) * t4
            + vec4(1.061405429) * t5;
  vec4 result = vec4(1.0) - poly * exp(-a * a);
  return sign(x) * result;
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const vec4 x = texelFetch(uInput, pos, 0);
    const vec4 outval = vec4(0.5) * x * (vec4(1.0) + erf_approx(x * vec4(0.7071067811865476)));
    imageStore(uOutput, pos, outval);
  }
}
