#version 450

#extension GL_ARB_separate_shader_objects : enable

layout(set=0, binding=1) uniform samplerCube skybox;

layout(location=0) in vec3 f_texcoord;
layout(location=0) out vec4 out_color;

void main() {
    out_color = texture(skybox, f_texcoord);
    out_color.a = 1.0;
}