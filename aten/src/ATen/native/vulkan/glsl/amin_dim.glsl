#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  uvec2 dim_info;
  int channel;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  int flattened_channels = int(ceil(uBlock.channel / 4.0));
  vec4 out_texel = vec4(1.0/0.0);

  // Batch
  if (uBlock.dim_info.x == 0) {
    for (int batch = 0; batch < uBlock.dim_info.y; batch++) {
      int src_z = batch * flattened_channels + pos.z;
      vec4 v = texelFetch(uInput, ivec3(pos.x, pos.y, src_z), 0);
      out_texel = min(out_texel, v);
    }
    imageStore(uOutput, pos, out_texel);
  }

  // Channel
  else if (uBlock.dim_info.x == 1) {
    for (int out_index = 0; out_index < 4; out_index++) {
      float m = 1.0/0.0;
      for (int channel = 0; channel < uBlock.dim_info.y; channel++) {
        int src_z =
            (pos.z * 4 + out_index) * flattened_channels + int(channel / 4);
        vec4 v = texelFetch(uInput, ivec3(pos.x, pos.y, src_z), 0);
        m = min(m, v[channel % 4]);
      }
      out_texel[out_index] = m;
    }
    imageStore(uOutput, pos, out_texel);
  }

  // Height, Width
  else {
    for (int out_index = 0; out_index < 4; out_index++) {
      int src_z = (pos.z * 4 + out_index) * flattened_channels + pos.y / 4;
      float m = 1.0/0.0;
      for (int hw = 0; hw < uBlock.dim_info.y; hw++) {
        vec4 v = (uBlock.dim_info.x == 2)
            ? texelFetch(uInput, ivec3(pos.x, hw, src_z), 0)
            : texelFetch(uInput, ivec3(hw, pos.x, src_z), 0);
        m = min(m, v[pos.y % 4]);
      }
      out_texel[out_index] = m;
    }
    imageStore(uOutput, pos, out_texel);
  }
}
