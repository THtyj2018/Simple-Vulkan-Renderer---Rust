#version 460

#extension GL_ARB_separate_shader_objects : enable

layout(std140, set=0, binding=0) uniform PerFrame {
    mat4 light;
} frame;

struct PerMesh {
    mat4 transform;
};

layout(std140, set=1, binding=0) readonly buffer PerMeshBuffer {
    PerMesh meshes[];
} meshes;

layout(location=0) in vec3 position;

void main() {
    gl_Position = frame.light * meshes.meshes[gl_BaseInstance].transform * vec4(position, 1.0);
}