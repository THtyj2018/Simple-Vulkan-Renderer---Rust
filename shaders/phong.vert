#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable

#include "phong.h.glsl"

layout(std140, set=1, binding=0) uniform PerMesh {
    mat4 transform;
} mesh;

layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec4 tangent;
layout(location=3) in vec2 texcoord;

layout(location=0) out vec4 f_light_space_position;
layout(location=1) out vec3 f_normal;
layout(location=2) out vec3 f_view_dir;
layout(location=3) out vec3 f_world_view_dir;
layout(location=4) out vec3 f_light_dir;
layout(location=5) out vec2 f_texcoord;

void main() {
    vec4 p = mesh.transform * vec4(position, 1.0);
    gl_Position = frame.camera * p;
    f_light_space_position = light.matrix * p;
    f_normal = normalize(mat3(mesh.transform) * normal);
    f_view_dir = frame.eye - p.xyz;
    f_world_view_dir = f_view_dir;
    f_light_dir = light.direction;
    f_texcoord = texcoord;
    if (material.normal_map_index >= 0) {
        vec3 T = mat3(mesh.transform) * vec3(tangent);
        T = normalize(T - dot(T, f_normal) * f_normal);
        vec3 B = cross(T, f_normal);
        mat3 TBN_inv = transpose(mat3(T, B, f_normal));
        // normalize ?
        f_view_dir = TBN_inv * f_view_dir;
        f_light_dir = TBN_inv * f_light_dir;
    }
}