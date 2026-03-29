#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Mean {
  float data[];
}
uMean;
layout(set = 0, binding = 3) buffer PRECISION restrict readonly Rstd {
  float data[];
}
uRstd;
layout(set = 0, binding = 4) buffer PRECISION restrict readonly Weight {
  float data[];
}
uWeight;
layout(set = 0, binding = 5) buffer PRECISION restrict readonly Bias {
  float data[];
}
uBias;
layout(set = 0, binding = 6) uniform PRECISION restrict Block {
  ivec4 extents;
  int num_channels;
  int num_groups;
  int has_weight;
  int has_bias;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.extents.xyz))) {
    // pos.z encodes batch*ceil(channels/4). Determine the channel indices.
    int channels_per_group = uBlock.num_channels / uBlock.num_groups;
    int c4_idx = pos.z; // which group of 4 channels
    int batch_idx = c4_idx / ((uBlock.num_channels + 3) / 4);
    int c_base = (c4_idx % ((uBlock.num_channels + 3) / 4)) * 4;

    vec4 val = texelFetch(uInput, pos, 0);
    vec4 result;

    for (int i = 0; i < 4; i++) {
      int c = c_base + i;
      if (c >= uBlock.num_channels) {
        result[i] = 0.0;
        continue;
      }
      int group = c / channels_per_group;
      int stat_idx = batch_idx * uBlock.num_groups + group;

      float mean = uMean.data[stat_idx];
      float rstd = uRstd.data[stat_idx];
      float normalized = (val[i] - mean) * rstd;

      if (uBlock.has_weight != 0) {
        normalized *= uWeight.data[c];
      }
      if (uBlock.has_bias != 0) {
        normalized += uBias.data[c];
      }
      result[i] = normalized;
    }

    imageStore(uOutput, pos, result);
  }
}
