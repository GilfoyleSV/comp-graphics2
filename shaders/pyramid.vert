#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

struct DirectionalLight {
    vec4 direction;
    vec4 color;
};

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;
    vec4 cameraPos;
    vec4 ambientLight;
    DirectionalLight dirLight;
    int pointLightCount;
    int spotLightCount;
    int pad0;
    int pad1;
} scene;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specularColor;
} modelUbo;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;

void main() {
    vec4 worldPos = modelUbo.model * vec4(inPosition, 1.0);
    gl_Position = scene.viewProj * worldPos;
    fragWorldPos = worldPos.xyz;
    fragNormal = normalize(mat3(modelUbo.normalMatrix) * inNormal);
    fragColor = inColor;
}
