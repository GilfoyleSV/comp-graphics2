#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

struct DirectionalLight {
    vec4 direction;
    vec4 color;
};

struct PointLight {
    vec4 position;
    vec4 color;
};

struct SpotLight {
    vec4 position;
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

layout(std430, set = 0, binding = 1) readonly buffer PointLightBuffer {
    PointLight pointLights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLightBuffer {
    SpotLight spotLights[];
};

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specularColor;
} modelUbo;

void main() {
    vec3 diffColor = modelUbo.albedo.rgb * fragColor;
    vec3 specColor = modelUbo.specularColor.rgb;
    float shininess = max(modelUbo.specularColor.w, 1.0);

    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene.cameraPos.xyz - fragWorldPos);

    vec3 totalLight = scene.ambientLight.rgb * scene.ambientLight.w * diffColor;

    vec3 L_dir = normalize(-scene.dirLight.direction.xyz);
    float NdotL_dir = max(dot(N, L_dir), 0.0);
    vec3 H_dir = normalize(L_dir + V);
    float NdotH_dir = max(dot(N, H_dir), 0.0);
    float spec_dir = pow(NdotH_dir, shininess);
    vec3 dirRadiance = scene.dirLight.color.rgb * scene.dirLight.direction.w;
    totalLight += (diffColor * NdotL_dir + specColor * spec_dir) * dirRadiance;

    for (int i = 0; i < scene.pointLightCount; ++i) {
        vec3 toLight = pointLights[i].position.xyz - fragWorldPos;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        if (dist > 0.0001) {
            vec3 L = toLight / dist;
            float attenuation = 1.0 / max(distSq, 0.01);
            float NdotL = max(dot(N, L), 0.0);
            vec3 H = normalize(L + V);
            float NdotH = max(dot(N, H), 0.0);
            float spec = pow(NdotH, shininess);
            vec3 radiance = pointLights[i].color.rgb * (pointLights[i].position.w * attenuation);
            totalLight += (diffColor * NdotL + specColor * spec) * radiance;
        }
    }

    for (int i = 0; i < scene.spotLightCount; ++i) {
        vec3 toLight = spotLights[i].position.xyz - fragWorldPos;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        if (dist > 0.0001) {
            vec3 L = toLight / dist;
            vec3 spotDir = normalize(-spotLights[i].direction.xyz);
            float cosTheta = dot(L, spotDir);
            float cosInner = spotLights[i].direction.w;
            float cosOuter = spotLights[i].color.w;
            float spotFactor = 0.0;
            if (cosInner > cosOuter) {
                spotFactor = clamp((cosTheta - cosOuter) / (cosInner - cosOuter), 0.0, 1.0);
            } else if (cosTheta >= cosInner) {
                spotFactor = 1.0;
            }
            if (spotFactor > 0.0) {
                float attenuation = 1.0 / max(distSq, 0.01);
                float NdotL = max(dot(N, L), 0.0);
                vec3 H = normalize(L + V);
                float NdotH = max(dot(N, H), 0.0);
                float spec = pow(NdotH, shininess);
                vec3 radiance = spotLights[i].color.rgb * (spotLights[i].position.w * attenuation * spotFactor);
                totalLight += (diffColor * NdotL + specColor * spec) * radiance;
            }
        }
    }

    outColor = vec4(totalLight, 1.0);
}
