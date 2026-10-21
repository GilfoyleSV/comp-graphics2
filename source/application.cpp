#include "application.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>

#include <imgui.h>
#include "math3d.hpp"

namespace application {

namespace {

struct Vertex {
	math::Vec3 pos;
	math::Vec3 normal;
	math::Vec3 color;
};

struct GpuDirectionalLight {
	math::Vec4 direction;
	math::Vec4 color;
};

struct SceneUniformBuffer {
	math::Mat4 viewProj;
	math::Vec4 cameraPos;
	math::Vec4 ambientLight;
	GpuDirectionalLight dirLight;
	int32_t pointLightCount;
	int32_t spotLightCount;
	int32_t pad0;
	int32_t pad1;
};

struct GpuPointLight {
	math::Vec4 position;
	math::Vec4 color;
};

struct GpuSpotLight {
	math::Vec4 position;
	math::Vec4 direction;
	math::Vec4 color;
};

struct ModelUniformBuffer {
	math::Mat4 model;
	math::Mat4 normalMatrix;
	math::Vec4 albedo;
	math::Vec4 specularColor;
};

struct PointLightItem {
	float position[3] = { 1.5f, 1.0f, 1.0f };
	float color[3] = { 1.0f, 0.7f, 0.4f };
	float intensity = 2.0f;
};

struct SpotLightItem {
	float position[3] = { -1.5f, 2.0f, 1.5f };
	float direction[3] = { 0.8f, -1.0f, -0.8f };
	float color[3] = { 0.4f, 0.8f, 1.0f };
	float intensity = 4.0f;
	float inner_angle = 15.0f;
	float outer_angle = 25.0f;
};

constexpr size_t MAX_LIGHTS = 64;

math::Vec3 calcProceduralColor(const math::Vec3& p) {
	float r = (p.x + 0.6f) / 1.2f;
	float g = (p.y + 0.5f) / 1.2f;
	float b = (p.z + 0.6f) / 1.2f;
	return {
		0.2f + 0.8f * r,
		0.2f + 0.8f * g,
		0.2f + 0.8f * b
	};
}

std::vector<Vertex> buildPyramidVertices() {
	math::Vec3 top(0.0f, 0.7f, 0.0f);
	math::Vec3 fl(-0.6f, -0.5f,  0.6f);
	math::Vec3 fr( 0.6f, -0.5f,  0.6f);
	math::Vec3 br( 0.6f, -0.5f, -0.6f);
	math::Vec3 bl(-0.6f, -0.5f, -0.6f);

	auto addTri = [](std::vector<Vertex>& v, const math::Vec3& a, const math::Vec3& b, const math::Vec3& c) {
		math::Vec3 normal = math::Vec3::cross(b - a, c - a).normalized();
		v.push_back({ a, normal, calcProceduralColor(a) });
		v.push_back({ b, normal, calcProceduralColor(b) });
		v.push_back({ c, normal, calcProceduralColor(c) });
	};

	std::vector<Vertex> vertices;
	addTri(vertices, top, fl, fr);
	addTri(vertices, top, fr, br);
	addTri(vertices, top, br, bl);
	addTri(vertices, top, bl, fl);

	addTri(vertices, fl, bl, br);
	addTri(vertices, fl, br, fr);

	return vertices;
}

std::vector<Vertex> pyramid_vertices;
uint32_t vertex_count = 0;

VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_buffer_alloc = VK_NULL_HANDLE;

VkBuffer scene_ubo_buffer = VK_NULL_HANDLE;
VmaAllocation scene_ubo_alloc = VK_NULL_HANDLE;

VkBuffer point_lights_ssbo = VK_NULL_HANDLE;
VmaAllocation point_lights_ssbo_alloc = VK_NULL_HANDLE;

VkBuffer spot_lights_ssbo = VK_NULL_HANDLE;
VmaAllocation spot_lights_ssbo_alloc = VK_NULL_HANDLE;

VkBuffer model_ubo_buffer = VK_NULL_HANDLE;
VmaAllocation model_ubo_alloc = VK_NULL_HANDLE;

VkDescriptorSetLayout scene_desc_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_desc_layout = VK_NULL_HANDLE;
VkDescriptorPool desc_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_desc_set = VK_NULL_HANDLE;
VkDescriptorSet model_desc_set = VK_NULL_HANDLE;

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline graphics_pipeline = VK_NULL_HANDLE;

float camera_pos[3] = { 0.0f, 1.2f, 3.5f };
float camera_pitch = -12.0f;
float camera_yaw = 0.0f;
float camera_fov = 45.0f;
int projection_type = 0;
float ortho_size = 2.0f;
float near_plane = 0.1f;
float far_plane = 100.0f;

float dir_light_dir[3] = { -0.5f, -1.0f, -0.3f };
float dir_light_color[3] = { 1.0f, 1.0f, 1.0f };
float dir_light_intensity = 1.0f;

float ambient_color[3] = { 0.2f, 0.2f, 0.25f };
float ambient_intensity = 0.3f;

float material_albedo[3] = { 0.9f, 0.9f, 0.9f };
float material_specular[3] = { 1.0f, 1.0f, 1.0f };
float material_shininess = 32.0f;

float position[3] = { 0.0f, 0.0f, 0.0f };
float rotation_deg[3] = { 0.0f, 0.0f, 0.0f };
float scale[3] = { 1.0f, 1.0f, 1.0f };
bool uniform_scale = true;
float uniform_scale_val = 1.0f;

bool animation_enabled = false;
float animation_speed = 1.0f;
float trajectory_radius = 1.5f;
float trajectory_height = 0.5f;
int trajectory_type = 0;
double anim_time = 0.0;
double last_time = 0.0;

std::vector<PointLightItem> point_lights;
std::vector<SpotLightItem> spot_lights;

std::vector<char> readBinaryFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Failed to open shader file: " << filename << std::endl;
		return {};
	}

	size_t fileSize = static_cast<size_t>(file.tellg());
	std::vector<char> buffer(fileSize);
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	return buffer;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
	if (code.empty()) return VK_NULL_HANDLE;

	VkShaderModuleCreateInfo create_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = code.size(),
		.pCode = reinterpret_cast<const uint32_t*>(code.data()),
	};

	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(device, &create_info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan shader module\n";
		return VK_NULL_HANDLE;
	}
	return module;
}

}

bool initialize() {
	auto& context = graphics::internal::context;

	point_lights.push_back({ { 1.5f, 1.0f, 1.0f }, { 1.0f, 0.7f, 0.4f }, 2.5f });
	spot_lights.push_back({ { -1.5f, 2.0f, 1.5f }, { 0.8f, -1.0f, -0.8f }, { 0.4f, 0.8f, 1.0f }, 4.0f, 15.0f, 25.0f });

	pyramid_vertices = buildPyramidVertices();
	vertex_count = static_cast<uint32_t>(pyramid_vertices.size());
	VkDeviceSize vb_size = sizeof(Vertex) * vertex_count;

	VkBufferCreateInfo vb_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = vb_size,
		.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VmaAllocationCreateInfo vb_alloc_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	if (vmaCreateBuffer(context.allocator, &vb_info, &vb_alloc_info,
	                    &vertex_buffer, &vertex_buffer_alloc, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create vertex buffer\n";
		return false;
	}
	void* mapped_vb = nullptr;
	vmaMapMemory(context.allocator, vertex_buffer_alloc, &mapped_vb);
	std::memcpy(mapped_vb, pyramid_vertices.data(), vb_size);
	vmaUnmapMemory(context.allocator, vertex_buffer_alloc);

	VkBufferCreateInfo scene_ubo_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(SceneUniformBuffer),
		.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VmaAllocationCreateInfo ubo_alloc_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	if (vmaCreateBuffer(context.allocator, &scene_ubo_info, &ubo_alloc_info,
	                    &scene_ubo_buffer, &scene_ubo_alloc, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create scene ubo buffer\n";
		return false;
	}

	VkDeviceSize point_ssbo_size = sizeof(GpuPointLight) * MAX_LIGHTS;
	VkBufferCreateInfo point_ssbo_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = point_ssbo_size,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	if (vmaCreateBuffer(context.allocator, &point_ssbo_info, &ubo_alloc_info,
	                    &point_lights_ssbo, &point_lights_ssbo_alloc, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create point lights ssbo\n";
		return false;
	}

	VkDeviceSize spot_ssbo_size = sizeof(GpuSpotLight) * MAX_LIGHTS;
	VkBufferCreateInfo spot_ssbo_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = spot_ssbo_size,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	if (vmaCreateBuffer(context.allocator, &spot_ssbo_info, &ubo_alloc_info,
	                    &spot_lights_ssbo, &spot_lights_ssbo_alloc, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create spot lights ssbo\n";
		return false;
	}

	VkBufferCreateInfo model_ubo_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(ModelUniformBuffer),
		.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	if (vmaCreateBuffer(context.allocator, &model_ubo_info, &ubo_alloc_info,
	                    &model_ubo_buffer, &model_ubo_alloc, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create model ubo buffer\n";
		return false;
	}

	VkDescriptorSetLayoutBinding scene_bindings[3] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};
	VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 3,
		.pBindings = scene_bindings,
	};
	if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr, &scene_desc_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create scene descriptor set layout\n";
		return false;
	}

	VkDescriptorSetLayoutBinding model_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};
	VkDescriptorSetLayoutCreateInfo model_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};
	if (vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr, &model_desc_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create model descriptor set layout\n";
		return false;
	}

	VkDescriptorPoolSize pool_sizes[2] = {
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 2,
		},
		{
			.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 2,
		},
	};
	VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 2,
		.poolSizeCount = 2,
		.pPoolSizes = pool_sizes,
	};
	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &desc_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	VkDescriptorSetLayout set_layouts[2] = { scene_desc_layout, model_desc_layout };
	VkDescriptorSetAllocateInfo alloc_set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = desc_pool,
		.descriptorSetCount = 2,
		.pSetLayouts = set_layouts,
	};
	VkDescriptorSet allocated_sets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
	if (vkAllocateDescriptorSets(context.device, &alloc_set_info, allocated_sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		return false;
	}
	scene_desc_set = allocated_sets[0];
	model_desc_set = allocated_sets[1];

	VkDescriptorBufferInfo scene_ubo_desc = {
		.buffer = scene_ubo_buffer,
		.offset = 0,
		.range = sizeof(SceneUniformBuffer),
	};
	VkDescriptorBufferInfo point_ssbo_desc = {
		.buffer = point_lights_ssbo,
		.offset = 0,
		.range = point_ssbo_size,
	};
	VkDescriptorBufferInfo spot_ssbo_desc = {
		.buffer = spot_lights_ssbo,
		.offset = 0,
		.range = spot_ssbo_size,
	};
	VkDescriptorBufferInfo model_ubo_desc = {
		.buffer = model_ubo_buffer,
		.offset = 0,
		.range = sizeof(ModelUniformBuffer),
	};

	VkWriteDescriptorSet descriptor_writes[4] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_desc_set,
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &scene_ubo_desc,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_desc_set,
			.dstBinding = 1,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &point_ssbo_desc,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_desc_set,
			.dstBinding = 2,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &spot_ssbo_desc,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = model_desc_set,
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &model_ubo_desc,
		},
	};
	vkUpdateDescriptorSets(context.device, 4, descriptor_writes, 0, nullptr);

	VkPipelineLayoutCreateInfo pipeline_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = set_layouts,
	};
	if (vkCreatePipelineLayout(context.device, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

	auto vert_code = readBinaryFile("shaders/pyramid.vert.spv");
	auto frag_code = readBinaryFile("shaders/pyramid.frag.spv");
	if (vert_code.empty() || frag_code.empty()) {
		std::cerr << "Shader binaries are missing\n";
		return false;
	}

	VkShaderModule vert_module = createShaderModule(context.device, vert_code);
	VkShaderModule frag_module = createShaderModule(context.device, frag_code);

	VkPipelineShaderStageCreateInfo shader_stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vert_module,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = frag_module,
			.pName = "main",
		},
	};

	VkVertexInputBindingDescription binding_description = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};
	VkVertexInputAttributeDescription attribute_descriptions[3] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, pos),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		},
		{
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, color),
		},
	};
	VkPipelineVertexInputStateCreateInfo vertex_input_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding_description,
		.vertexAttributeDescriptionCount = 3,
		.pVertexAttributeDescriptions = attribute_descriptions,
	};

	VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE,
	};

	VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	VkPipelineRasterizationStateCreateInfo rasterizer = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f,
	};

	VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		.sampleShadingEnable = VK_FALSE,
	};

	VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
	};

	VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.blendEnable = VK_TRUE,
		.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
		.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
		.colorBlendOp = VK_BLEND_OP_ADD,
		.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
		.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
		.alphaBlendOp = VK_BLEND_OP_ADD,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	VkPipelineColorBlendStateCreateInfo color_blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOpEnable = VK_FALSE,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};
	VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};

	VkGraphicsPipelineCreateInfo pipeline_create_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input_info,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blending,
		.pDynamicState = &dynamic_state,
		.layout = pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &graphics_pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		vkDestroyShaderModule(context.device, frag_module, nullptr);
		vkDestroyShaderModule(context.device, vert_module, nullptr);
		return false;
	}

	vkDestroyShaderModule(context.device, frag_module, nullptr);
	vkDestroyShaderModule(context.device, vert_module, nullptr);

	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	if (graphics_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, graphics_pipeline, nullptr);
		graphics_pipeline = VK_NULL_HANDLE;
	}
	if (pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
		pipeline_layout = VK_NULL_HANDLE;
	}
	if (desc_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, desc_pool, nullptr);
		desc_pool = VK_NULL_HANDLE;
	}
	if (model_desc_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, model_desc_layout, nullptr);
		model_desc_layout = VK_NULL_HANDLE;
	}
	if (scene_desc_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, scene_desc_layout, nullptr);
		scene_desc_layout = VK_NULL_HANDLE;
	}
	if (model_ubo_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, model_ubo_buffer, model_ubo_alloc);
		model_ubo_buffer = VK_NULL_HANDLE;
	}
	if (spot_lights_ssbo != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, spot_lights_ssbo, spot_lights_ssbo_alloc);
		spot_lights_ssbo = VK_NULL_HANDLE;
	}
	if (point_lights_ssbo != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, point_lights_ssbo, point_lights_ssbo_alloc);
		point_lights_ssbo = VK_NULL_HANDLE;
	}
	if (scene_ubo_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, scene_ubo_buffer, scene_ubo_alloc);
		scene_ubo_buffer = VK_NULL_HANDLE;
	}
	if (vertex_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_buffer_alloc);
		vertex_buffer = VK_NULL_HANDLE;
	}
}

void update(double current_time) {
	double dt = current_time - last_time;
	if (last_time == 0.0) dt = 0.0;
	last_time = current_time;

	if (animation_enabled) {
		anim_time += dt * animation_speed;
		float t = static_cast<float>(anim_time);

		if (trajectory_type == 0) {
			position[0] = trajectory_radius * std::cos(t);
			position[2] = trajectory_radius * std::sin(t) * std::cos(t);
			position[1] = trajectory_height * std::sin(2.0f * t);
		} else if (trajectory_type == 1) {
			position[0] = trajectory_radius * std::cos(t);
			position[2] = trajectory_radius * std::sin(t);
			position[1] = trajectory_height * std::sin(t);
		} else {
			position[0] = trajectory_radius * std::cos(t * 1.5f);
			position[2] = trajectory_radius * std::sin(t * 1.5f);
			position[1] = trajectory_height * std::sin(t * 3.0f);
		}

		rotation_deg[1] += 45.0f * animation_speed * static_cast<float>(dt);
		rotation_deg[0] += 25.0f * animation_speed * static_cast<float>(dt);
		while (rotation_deg[1] >= 360.0f) rotation_deg[1] -= 360.0f;
		while (rotation_deg[0] >= 360.0f) rotation_deg[0] -= 360.0f;
	}

	ImGui::Begin("Camera and Lighting Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

	if (ImGui::CollapsingHeader("Camera Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Camera Position", camera_pos, 0.05f);
		ImGui::SliderFloat("Camera Pitch", &camera_pitch, -89.0f, 89.0f);
		ImGui::SliderFloat("Camera Yaw", &camera_yaw, -180.0f, 180.0f);
		ImGui::SliderFloat("Camera FOV", &camera_fov, 10.0f, 120.0f);

		ImGui::RadioButton("Perspective", &projection_type, 0);
		ImGui::SameLine();
		ImGui::RadioButton("Orthographic", &projection_type, 1);

		if (projection_type == 1) {
			ImGui::SliderFloat("Ortho Size", &ortho_size, 0.5f, 10.0f);
		}
		ImGui::DragFloat("Near Plane", &near_plane, 0.05f, 0.01f, 10.0f);
		ImGui::DragFloat("Far Plane", &far_plane, 1.0f, 10.0f, 500.0f);

		if (ImGui::Button("Reset Camera")) {
			camera_pos[0] = 0.0f;
			camera_pos[1] = 1.2f;
			camera_pos[2] = 3.5f;
			camera_pitch = -12.0f;
			camera_yaw = 0.0f;
			camera_fov = 45.0f;
		}
	}

	if (ImGui::CollapsingHeader("Directional & Ambient Light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Text("Ambient Light");
		ImGui::ColorEdit3("Ambient Color", ambient_color);
		ImGui::SliderFloat("Ambient Intensity", &ambient_intensity, 0.0f, 2.0f);

		ImGui::Separator();
		ImGui::Text("Directional Light");
		ImGui::DragFloat3("Dir Light Direction", dir_light_dir, 0.05f, -1.0f, 1.0f);
		ImGui::ColorEdit3("Dir Light Color", dir_light_color);
		ImGui::SliderFloat("Dir Light Intensity", &dir_light_intensity, 0.0f, 5.0f);
	}

	if (ImGui::CollapsingHeader("Material (Blinn-Phong)", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Albedo", material_albedo);
		ImGui::ColorEdit3("Specular Color", material_specular);
		ImGui::SliderFloat("Shininess", &material_shininess, 1.0f, 256.0f);
	}

	if (ImGui::CollapsingHeader("Point Lights (SSBO)")) {
		if (ImGui::Button("Add Point Light") && point_lights.size() < MAX_LIGHTS) {
			point_lights.push_back({ { 0.0f, 1.0f, 1.5f }, { 1.0f, 1.0f, 1.0f }, 2.0f });
		}
		ImGui::SameLine();
		ImGui::Text("Count: %zu / %zu", point_lights.size(), MAX_LIGHTS);

		int remove_point_idx = -1;
		for (size_t i = 0; i < point_lights.size(); ++i) {
			ImGui::PushID(static_cast<int>(i + 1000));
			std::string label = "Point Light #" + std::to_string(i + 1);
			if (ImGui::TreeNode(label.c_str())) {
				ImGui::DragFloat3("Position", point_lights[i].position, 0.05f);
				ImGui::ColorEdit3("Color", point_lights[i].color);
				ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.0f, 10.0f);
				if (ImGui::Button("Remove")) {
					remove_point_idx = static_cast<int>(i);
				}
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
		if (remove_point_idx >= 0) {
			point_lights.erase(point_lights.begin() + remove_point_idx);
		}
	}

	if (ImGui::CollapsingHeader("Spot Lights (SSBO)")) {
		if (ImGui::Button("Add Spot Light") && spot_lights.size() < MAX_LIGHTS) {
			spot_lights.push_back({ { 0.0f, 2.5f, 0.0f }, { 0.0f, -1.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 4.0f, 15.0f, 25.0f });
		}
		ImGui::SameLine();
		ImGui::Text("Count: %zu / %zu", spot_lights.size(), MAX_LIGHTS);

		int remove_spot_idx = -1;
		for (size_t i = 0; i < spot_lights.size(); ++i) {
			ImGui::PushID(static_cast<int>(i + 2000));
			std::string label = "Spot Light #" + std::to_string(i + 1);
			if (ImGui::TreeNode(label.c_str())) {
				ImGui::DragFloat3("Position", spot_lights[i].position, 0.05f);
				ImGui::DragFloat3("Direction", spot_lights[i].direction, 0.05f);
				ImGui::ColorEdit3("Color", spot_lights[i].color);
				ImGui::SliderFloat("Intensity", &spot_lights[i].intensity, 0.0f, 15.0f);
				ImGui::SliderFloat("Inner Angle (deg)", &spot_lights[i].inner_angle, 1.0f, 60.0f);
				if (spot_lights[i].outer_angle < spot_lights[i].inner_angle) {
					spot_lights[i].outer_angle = spot_lights[i].inner_angle;
				}
				ImGui::SliderFloat("Outer Angle (deg)", &spot_lights[i].outer_angle, spot_lights[i].inner_angle, 80.0f);
				if (ImGui::Button("Remove")) {
					remove_spot_idx = static_cast<int>(i);
				}
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
		if (remove_spot_idx >= 0) {
			spot_lights.erase(spot_lights.begin() + remove_spot_idx);
		}
	}

	if (ImGui::CollapsingHeader("Model Transformations")) {
		ImGui::DragFloat3("Position (X, Y, Z)", position, 0.05f, -10.0f, 10.0f);
		ImGui::DragFloat3("Rotation (deg)", rotation_deg, 1.0f, -360.0f, 360.0f);

		ImGui::Checkbox("Uniform Scale", &uniform_scale);
		if (uniform_scale) {
			if (ImGui::DragFloat("Scale", &uniform_scale_val, 0.05f, 0.1f, 10.0f)) {
				scale[0] = scale[1] = scale[2] = uniform_scale_val;
			}
		} else {
			ImGui::DragFloat3("Scale (X, Y, Z)", scale, 0.05f, 0.1f, 10.0f);
		}

		if (ImGui::Button("Reset Transformations")) {
			position[0] = position[1] = position[2] = 0.0f;
			rotation_deg[0] = rotation_deg[1] = rotation_deg[2] = 0.0f;
			scale[0] = scale[1] = scale[2] = uniform_scale_val = 1.0f;
		}

		ImGui::Separator();
		if (animation_enabled) {
			if (ImGui::Button("Pause Animation")) {
				animation_enabled = false;
			}
		} else {
			if (ImGui::Button("Play Animation")) {
				animation_enabled = true;
			}
		}

		const char* trajectories[] = { "Lemniscate (Figure 8)", "Circular Orbit", "3D Spiral" };
		ImGui::Combo("Trajectory", &trajectory_type, trajectories, IM_COUNTOF(trajectories));
		ImGui::SliderFloat("Speed", &animation_speed, 0.1f, 5.0f);
		ImGui::SliderFloat("Radius", &trajectory_radius, 0.5f, 4.0f);
		ImGui::SliderFloat("Height", &trajectory_height, 0.0f, 2.0f);
	}

	ImGui::End();

	auto& context = graphics::internal::context;
	float aspect = 1.0f;
	if (context.swapchain_extent.height > 0) {
		aspect = static_cast<float>(context.swapchain_extent.width) / static_cast<float>(context.swapchain_extent.height);
	}

	math::Vec3 eye(camera_pos[0], camera_pos[1], camera_pos[2]);
	float pitch_r = math::radians(camera_pitch);
	float yaw_r = math::radians(camera_yaw);
	math::Vec3 forward(
		std::sin(yaw_r) * std::cos(pitch_r),
		std::sin(pitch_r),
		-std::cos(yaw_r) * std::cos(pitch_r)
	);
	math::Vec3 target = eye + forward;
	math::Mat4 m_view = math::Mat4::lookAt(eye, target, { 0.0f, 1.0f, 0.0f });

	math::Mat4 m_proj;
	if (projection_type == 0) {
		m_proj = math::Mat4::perspective(math::radians(camera_fov), aspect, near_plane, far_plane);
	} else {
		m_proj = math::Mat4::orthographic(-ortho_size * aspect, ortho_size * aspect,
		                                  -ortho_size, ortho_size, near_plane, far_plane);
	}

	math::Mat4 m_model = math::Mat4::translate({ position[0], position[1], position[2] }) *
	                     math::Mat4::rotateZ(math::radians(rotation_deg[2])) *
	                     math::Mat4::rotateY(math::radians(rotation_deg[1])) *
	                     math::Mat4::rotateX(math::radians(rotation_deg[0])) *
	                     math::Mat4::scale({ scale[0], scale[1], scale[2] });

	math::Mat4 m_normal = m_model.inverse().transposed();

	std::vector<GpuPointLight> gpu_point_lights;
	gpu_point_lights.reserve(point_lights.size());
	for (const auto& pl : point_lights) {
		if (gpu_point_lights.size() >= MAX_LIGHTS) break;
		GpuPointLight gpl;
		gpl.position = { pl.position[0], pl.position[1], pl.position[2], pl.intensity };
		gpl.color = { pl.color[0], pl.color[1], pl.color[2], 0.0f };
		gpu_point_lights.push_back(gpl);
	}
	if (!gpu_point_lights.empty()) {
		void* mapped_pl = nullptr;
		vmaMapMemory(context.allocator, point_lights_ssbo_alloc, &mapped_pl);
		std::memcpy(mapped_pl, gpu_point_lights.data(), sizeof(GpuPointLight) * gpu_point_lights.size());
		vmaUnmapMemory(context.allocator, point_lights_ssbo_alloc);
	}

	std::vector<GpuSpotLight> gpu_spot_lights;
	gpu_spot_lights.reserve(spot_lights.size());
	for (const auto& sl : spot_lights) {
		if (gpu_spot_lights.size() >= MAX_LIGHTS) break;
		GpuSpotLight gsl;
		gsl.position = { sl.position[0], sl.position[1], sl.position[2], sl.intensity };
		math::Vec3 dir = math::Vec3(sl.direction[0], sl.direction[1], sl.direction[2]).normalized();
		float inner_rad = math::radians(sl.inner_angle);
		float outer_rad = math::radians(sl.outer_angle);
		gsl.direction = { dir.x, dir.y, dir.z, std::cos(inner_rad) };
		gsl.color = { sl.color[0], sl.color[1], sl.color[2], std::cos(outer_rad) };
		gpu_spot_lights.push_back(gsl);
	}
	if (!gpu_spot_lights.empty()) {
		void* mapped_sl = nullptr;
		vmaMapMemory(context.allocator, spot_lights_ssbo_alloc, &mapped_sl);
		std::memcpy(mapped_sl, gpu_spot_lights.data(), sizeof(GpuSpotLight) * gpu_spot_lights.size());
		vmaUnmapMemory(context.allocator, spot_lights_ssbo_alloc);
	}

	SceneUniformBuffer scene_ubo = {
		.viewProj = m_proj * m_view,
		.cameraPos = { eye.x, eye.y, eye.z, 1.0f },
		.ambientLight = { ambient_color[0], ambient_color[1], ambient_color[2], ambient_intensity },
		.dirLight = {
			.direction = { dir_light_dir[0], dir_light_dir[1], dir_light_dir[2], dir_light_intensity },
			.color = { dir_light_color[0], dir_light_color[1], dir_light_color[2], 0.0f },
		},
		.pointLightCount = static_cast<int32_t>(gpu_point_lights.size()),
		.spotLightCount = static_cast<int32_t>(gpu_spot_lights.size()),
		.pad0 = 0,
		.pad1 = 0,
	};
	void* mapped_scene = nullptr;
	vmaMapMemory(context.allocator, scene_ubo_alloc, &mapped_scene);
	std::memcpy(mapped_scene, &scene_ubo, sizeof(scene_ubo));
	vmaUnmapMemory(context.allocator, scene_ubo_alloc);

	ModelUniformBuffer model_ubo = {
		.model = m_model,
		.normalMatrix = m_normal,
		.albedo = { material_albedo[0], material_albedo[1], material_albedo[2], 1.0f },
		.specularColor = { material_specular[0], material_specular[1], material_specular[2], material_shininess },
	};
	void* mapped_model = nullptr;
	vmaMapMemory(context.allocator, model_ubo_alloc, &mapped_model);
	std::memcpy(mapped_model, &model_ubo, sizeof(model_ubo));
	vmaUnmapMemory(context.allocator, model_ubo_alloc);
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(fd.command_buffer, &begin_info);

	const VkClearValue clear_values[2] = {
		{ .color = { { 0.10f, 0.12f, 0.16f, 1.0f } } },
		{ .depthStencil = { 1.0f, 0 } },
	};

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	if (graphics_pipeline != VK_NULL_HANDLE) {
		vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

		VkViewport viewport = {
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(context.swapchain_extent.width),
			.height = static_cast<float>(context.swapchain_extent.height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};
		vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

		VkRect2D scissor = {
			.offset = { 0, 0 },
			.extent = context.swapchain_extent,
		};
		vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, offsets);

		VkDescriptorSet descriptor_sets[2] = { scene_desc_set, model_desc_set };
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        pipeline_layout, 0, 2, descriptor_sets, 0, nullptr);

		vkCmdDraw(fd.command_buffer, vertex_count, 1, 0, 0);
	}

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

}