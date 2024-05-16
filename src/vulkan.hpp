#ifndef VULKAN_HPP
#define VULKAN_HPP

#include <vulkan/vulkan.h>
#include "quants.hpp"
#include <vector>
#include <string>

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

class VulkanContext {
private:
    VkInstance instance;
    VkApplicationInfo appInfo;
    VkShaderModule shaderModule;
    VkCommandPool commandPool;
    VkDescriptorPool descriptorPool;
    VkDescriptorSetLayout descriptorSetLayoutSet0;
    VkDescriptorSetLayout descriptorSetLayoutSet1;
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
    void loadComputeShaderModule(const std::string &shaderPath);
    void createDescriptorSetLayout();
    void createPipeline();
public:
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    std::vector<VkPhysicalDevice> physicalDevices;
    int computeQueueFamilyIndex;
    VkQueue computeQueue;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    VkCommandBuffer commandBuffer;
    VkDescriptorSet descriptorSets[2];
    VulkanContext();
    ~VulkanContext();
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