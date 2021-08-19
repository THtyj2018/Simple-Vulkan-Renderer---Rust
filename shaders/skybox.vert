#version 450

#extension GL_ARB_separate_shader_objects : enable

layout(std140, set=0, binding=0) uniform PerFrame {
    mat4 camera;
} frame;

const vec3[8] positions = vec3[] (
    vec3(-1.0, -1.0, -1.0),
    vec3(1.0, -1.0, -1.0),
    vec3(-1.0, 1.0, -1.0),
    vec3(1.0, 1.0, -1.0),
    vec3(-1.0, -1.0, 1.0),
    vec3(1.0, -1.0, 1.0),
    vec3(-1.0, 1.0, 1.0),
    vec3(1.0, 1.0, 1.0)
);

const int[36] indices = int[] (
    0, 1, 2, 2, 1, 3,
    2, 3, 6, 6, 3, 7,
    6, 7, 4, 4, 7, 5,
    4, 5, 0, 5, 0, 1,
    4, 0, 6, 6, 0, 2,
    1, 5, 3, 3, 5, 7
);

layout(location=0) out vec3 f_texcoord;

void main() {
    vec3 position = positions[indices[gl_VertexIndex]];
    gl_Position = (frame.camera * vec4(position, 1.0)).xyww;
    f_texcoord = vec3(position.x, position.y, position.z);
}