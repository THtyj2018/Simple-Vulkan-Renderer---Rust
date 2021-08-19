layout(std140, set=0, binding=0) uniform PerFrame {
    mat4 camera;
    vec3 eye;
} frame;

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