#version 450

#extension GL_ARB_separate_shader_objects : enable

layout(std140, set=0, binding=1) uniform LightInfo {
    mat4 matrix;
    vec3 direction;
    vec4 color;
    float ambient;
    float intensity;
} light;

layout(std140, set=2, binding=0) uniform MaterialInfo {
    vec4 base_color;
    int base_color_map_index;
    int normal_map_index;
    float normal_scale;
    float specular;
} material;

layout(set=0, binding=2) uniform sampler2D shadow_map;
layout(set=0, binding=3) uniform samplerCube skybox;

layout(set=2, binding=1) uniform sampler2D sampled_textures[1000];

layout(location=0) in vec4 f_light_space_position;
layout(location=1) in vec3 f_normal;
layout(location=2) in vec3 f_view_dir;
layout(location=3) in vec3 f_world_view_dir;
layout(location=4) in vec3 f_light_dir;
layout(location=5) in vec2 f_texcoord;

layout(location=0) out vec4 out_color;

float calc_shaodow_factor() {
    vec3 coords = f_light_space_position.xyz / f_light_space_position.w;
    vec2 texel_size = 1.0 / textureSize(shadow_map, 0);
    float factor = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float depth = texture(shadow_map, coords.xy * 0.5 + 0.5 + texel_size * vec2(x, y)).r;
            if (coords.z > depth)
                factor += 1.0;
        }
    }
    return factor / 9.0;
}

vec3 fresnel_schlick(vec4 color, float cos_theta) {
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, color.rgb, material.specular);
    float F90 = 0.2 + 0.8 * material.specular;
    return F0 + (F90 - F0) * pow(1.0 - cos_theta, 5.0);
}

void main() {
    vec3 normal = normalize(f_normal);
    vec3 world_view_dir = normalize(f_world_view_dir);
    vec3 refl = reflect(-world_view_dir, normal);
    vec4 refl_color = texture(skybox, refl);
    float cos_theta = dot(normal, world_view_dir);

    if (material.normal_map_index >= 0) {
        normal = texture(sampled_textures[material.normal_map_index], f_texcoord).rgb;
        normal = (normal * 2.0 - 1.0) * vec3(material.normal_scale, material.normal_scale, 1.0);
        normal = normalize(normal);
    }
    vec3 view_dir = normalize(f_view_dir);
    vec3 light_dir = normalize(f_light_dir);
    vec3 halfv = normalize(light_dir + view_dir);
    vec4 base_color = material.base_color;
    if (material.base_color_map_index >= 0) {
        base_color *= texture(sampled_textures[material.base_color_map_index], f_texcoord);
    }
    vec4 diffuse = light.color * base_color * (0.5 + 0.5 * dot(normal, light_dir));
    vec4 specular = light.color * material.specular * pow(max(0.0, dot(normal, halfv)), 32);
    vec3 reflection = refl_color.rgb * fresnel_schlick(base_color, cos_theta);
    float shadow = calc_shaodow_factor();
    out_color.rgb = light.ambient * base_color.rgb + (1.0 - shadow) * light.intensity * (diffuse.rgb + specular.rgb + reflection);
    out_color.a = 1.0;
}