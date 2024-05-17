#ifndef VULKAN_HPP
#define VULKAN_HPP

#include <vulkan/vulkan.h>
#include "quants.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <array>

#define SPLIT_RANGE_TO_THREADS(varStart, varEnd, rangeStart, rangeEnd, nThreads, threadIndex) \
    const unsigned int rangeLen = (rangeEnd - rangeStart); \
    const unsigned int rangeSlice = rangeLen / nThreads; \
    const unsigned int rangeRest = rangeLen % nThreads; \
    const unsigned int varStart = threadIndex * rangeSlice + (threadIndex < rangeRest ? threadIndex : rangeRest); \
    const unsigned int varEnd = varStart + rangeSlice + (threadIndex < rangeRest ? 1 : 0);

struct MatMulInfo {
    int n;
    int ds;
    int de;
};

enum VulkanPipelineType {
    F32_F32 = 0,
    F16_F32 = 1,
    Q40_F32 = 2,
    Q80_F32 = 3,
    Q40_Q80 = 4,
    Q80_Q80 = 5,
};

struct VulkanPipeline {
    FloatType weightsFloatType;
    FloatType inputFloatType;
    VkShaderModule shaderModule;
    VkCommandPool commandPool;
    VkDescriptorPool descriptorPool;
    VkDescriptorSetLayout descriptorSetLayoutSet0;
    VkDescriptorSetLayout descriptorSetLayoutSet1;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    VkCommandBuffer commandBuffer;
    VkDescriptorSet descriptorSets[2];
};

class VulkanContext {
private:
    VkInstance instance;
    VkApplicationInfo appInfo;
    void initialize();
    void createInstance();
    void getDevice();
    void printPhysicalDeviceMemoryProperties();
    VkShaderModule loadComputeShaderModule(const std::string &shaderPath);
    void createDescriptorSetLayout();
    void createPipeline(VulkanPipelineType pipelineType, const std::string &name);
public:
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    std::vector<VkPhysicalDevice> physicalDevices;
    int computeQueueFamilyIndex;
    VkQueue computeQueue;
    std::unordered_map<VulkanPipelineType, VulkanPipeline> pipelines;
    VulkanContext();
    ~VulkanContext();
    VulkanPipeline* getPipeline(VulkanPipelineType pipelineType);
};

void matmulVulkan(VulkanContext* vulkan, FloatType weightsFloatType, FloatType inputFloatType, float* output, void* input, void* weights, int n, int d, unsigned int nThreads, unsigned int threadIndex);

#endif