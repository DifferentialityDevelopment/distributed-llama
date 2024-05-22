#ifndef VULKAN_HPP
#define VULKAN_HPP

#include <vulkan/vulkan.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <array>
#include "quants.hpp"
#include "tasks.hpp"

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

enum class LayerElement {
    QUERY = 0,
    KEY = 1,
    VALUE = 2
};

//hash function for std::pair<int, LayerElement>
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2>& p) const {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ hash2;  // Combine the two hash values
    }
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
    std::unordered_map<std::pair<int, LayerElement>, std::pair<VkBuffer, VkDeviceMemory>, pair_hash> bufferMap;
    void initialize();
    void createInstance();
    void getDevice();
    void printPhysicalDeviceMemoryProperties();
    VkShaderModule loadComputeShaderModule(const std::string &shaderPath);
    void createDescriptorSetLayout();
    bool loadTransformerBlock(int blockIndex, TransformerBlock* block);
    void createPipeline(VulkanPipelineType pipelineType, const std::string &name);
public:
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    std::vector<VkPhysicalDevice> physicalDevices;
    int computeQueueFamilyIndex;
    VkQueue computeQueue;
    std::unordered_map<VulkanPipelineType, VulkanPipeline> pipelines;
    VulkanContext(Transformer* transformer);
    ~VulkanContext();
    std::pair<VkBuffer, VkDeviceMemory>* getLayerBufferData(int blockIndex, LayerElement type);
    VulkanPipeline* getPipeline(VulkanPipelineType pipelineType);
};

void matmulVulkan(TransformerContext* ctx, LayerElement layerElement, FloatType weightsFloatType, FloatType inputFloatType, float* output, void* input, void* weights, int n, int d, unsigned int nThreads, unsigned int threadIndex);

#endif