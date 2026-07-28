#version 450 core
#define PRECISION ${PRECISION}

layout(std430) buffer;

layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uOutput;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  imageStore(uOutput, pos, ivec4(0, 0, 0, 0));
}
