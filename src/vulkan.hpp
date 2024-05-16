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
    VkShaderModule shaderModule; //done
    VkCommandPool commandPool; //done
    VkDescriptorPool descriptorPool; //done
    VkDescriptorSetLayout descriptorSetLayoutSet0; //done
    VkDescriptorSetLayout descriptorSetLayoutSet1;//done
    VkPipelineLayout pipelineLayout; //done
    VkPipeline pipeline; //done
    VkCommandBuffer commandBuffer; //done
    VkDescriptorSet descriptorSets[2];
};

class VulkanContext {
private:
    VkInstance instance;
    VkApplicationInfo appInfo;
    /*
    VkDescriptorSet descriptorSet;
    VkBuffer inputBuffer, weightsBuffer, outputBuffer;
    VkDeviceMemory inputBufferMemory, weightsBufferMemory, outputBufferMemory;
    VkBuffer uniformBuffer;
    VkDeviceMemory uniformBufferMemory;
    VkDescriptorBufferInfo inputBufferInfo, weightsBufferInfo, outputBufferInfo, uniformBufferInfo;
    */
    void initialize();
    void createInstance();
    void getDevice();
    VkShaderModule loadComputeShaderModule(const std::string &shaderPath);
    void createDescriptorSetLayout();
    void createPipeline(VulkanPipelineType pipelineType);
public:
    VkDevice device; //done
    VkPhysicalDevice physicalDevice; //done
    std::vector<VkPhysicalDevice> physicalDevices; //done
    int computeQueueFamilyIndex; //done
    VkQueue computeQueue; //done
    std::unordered_map<VulkanPipelineType, VulkanPipeline> pipelines; //done
    VulkanContext();
    ~VulkanContext();
    VulkanPipeline* getPipeline(VulkanPipelineType pipelineType);
};

/*
 * Explanation of MatmulVulkanInfo members:
    * - vulkan context: Pointer to vulkan context
 * - float* output: Pointer to the output array where the result of matrix multiplication is stored.
 * - void* input: Pointer to the input data (either float or quantized).
 * - void* weights: Pointer to the weight data (either float or quantized).
 * - int n: Number of columns in the input matrix (dimensionality of input).
 * - int ds: Start index of the output array slice that the thread will compute.
 * - int de: End index (exclusive) of the output array slice that the thread will compute.
 */
struct MatmulVulkanInfo {
    VulkanContext* vulkan;
    float* output;
    void* input;
    void* weights;
    int n;
    int ds;
    int de;
};

void matmulVulkan(VulkanContext* vulkan, FloatType weightsFloatType, FloatType inputFloatType, float* output, void* input, void* weights, int n, int d, unsigned int nThreads, unsigned int threadIndex);

#endif