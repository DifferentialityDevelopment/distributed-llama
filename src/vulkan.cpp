#define VK_USE_PLATFORM_XCB_KHR
#include <cstdio>
#include <cassert>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vulkan/vulkan.h>
#include "transformer.hpp"
#include "quants.hpp"
#include "funcs.hpp"
#include "tasks.hpp"
#include "vulkan.hpp"
#include <functional>

// Create an array to map enum values to struct instances
std::unordered_map<VulkanPipelineType, std::string> shaderPathMap = {
    {F32_F32, "./shaders/matmulF32.spv"},
    {Q40_Q80, "./shaders/matmulQ40Q80.spv"}
};

// Validation layers to enable
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

void printDebugArray(const float* arr, int size, const std::string& name) {
    std::cout << name << ": ";
    for (int i = 0; i < size; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

// Enabled extensions
const char* enabledExtensions[] = {
    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    //VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME
};

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file!");
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

// Function to get the shader path from FloatType
std::string getShaderPath(VulkanPipelineType pipelineType) {
    auto it = shaderPathMap.find(pipelineType);
    if (it != shaderPathMap.end()) {
        return it->second;
    } else {
        return "Shader path not found!";
    }
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module!");
    }

    return shaderModule;
}

uint32_t findMemoryType(VkPhysicalDevice &physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    // Retrieve physical device memory properties
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    printf("ERROR: Failed to find suitable memory type!\n");
    throw std::runtime_error("Failed to find suitable memory type!");
}

void matmulVulkanF32(MatmulThreadInfo* a, LayerElement layerElement) {
    TransformerContext* ctx = (TransformerContext*)a->ctx;
    VulkanContext* vulkan = (VulkanContext*)ctx->vulkan;

    VulkanPipeline* pipeline = vulkan->getPipeline(VulkanPipelineType::F32_F32);

    #pragma region // Create the buffers
    VkBufferCreateInfo weightsBufferInfo = {};
    weightsBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    weightsBufferInfo.size = a->n * (a->de - a->ds) * sizeof(float);
    weightsBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    weightsBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBufferCreateInfo inputBufferInfo = {};
    inputBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    inputBufferInfo.size = a->n * sizeof(float);
    inputBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    inputBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBufferCreateInfo outputBufferInfo = {};
    outputBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    outputBufferInfo.size = (a->de - a->ds) * sizeof(float);
    outputBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    outputBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBufferCreateInfo matMulBufferInfo = {};
    matMulBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    matMulBufferInfo.size = sizeof(MatMulInfo);
    matMulBufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    matMulBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer weightsBuffer, inputBuffer, outputBuffer, matMulBuffer;
    vkCreateBuffer(vulkan->device, &weightsBufferInfo, nullptr, &weightsBuffer);
    vkCreateBuffer(vulkan->device, &inputBufferInfo, nullptr, &inputBuffer);
    vkCreateBuffer(vulkan->device, &outputBufferInfo, nullptr, &outputBuffer);
    vkCreateBuffer(vulkan->device, &matMulBufferInfo, nullptr, &matMulBuffer);
    #pragma endregion

    #pragma region // Get memory requirements for the buffers 
    VkMemoryRequirements memRequirementsWeights;
    vkGetBufferMemoryRequirements(vulkan->device, weightsBuffer, &memRequirementsWeights);
    VkMemoryRequirements memRequirementsInput;
    vkGetBufferMemoryRequirements(vulkan->device, inputBuffer, &memRequirementsInput);
    VkMemoryRequirements memoryRequirementsOutput;
    vkGetBufferMemoryRequirements(vulkan->device, outputBuffer, &memoryRequirementsOutput);
    VkMemoryRequirements memoryRequirementsMatMulInfo;
    vkGetBufferMemoryRequirements(vulkan->device, matMulBuffer, &memoryRequirementsMatMulInfo);
    
    VkMemoryAllocateInfo allocInfoWeights = {};
    allocInfoWeights.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfoWeights.allocationSize = memRequirementsWeights.size;
    allocInfoWeights.memoryTypeIndex = findMemoryType(vulkan->physicalDevices[0], memRequirementsWeights.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    VkMemoryAllocateInfo allocInfoInput = {};
    allocInfoInput.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfoInput.allocationSize = memRequirementsInput.size;
    allocInfoInput.memoryTypeIndex = findMemoryType(vulkan->physicalDevices[0], memRequirementsInput.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    VkMemoryAllocateInfo allocInfoOutput = {};
    allocInfoOutput.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfoOutput.allocationSize = memoryRequirementsOutput.size;
    allocInfoOutput.memoryTypeIndex = findMemoryType(vulkan->physicalDevices[0], memoryRequirementsOutput.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    VkMemoryAllocateInfo allocInfoMatMul = {};
    allocInfoMatMul.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfoMatMul.allocationSize = memoryRequirementsMatMulInfo.size;
    allocInfoMatMul.memoryTypeIndex = findMemoryType(vulkan->physicalDevices[0], memoryRequirementsMatMulInfo.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    #pragma endregion

    #pragma region // Allocate and map memory for the buffers
    //Allocate and map memory for the buffers
    VkDeviceMemory weightsBufferMemory, inputBufferMemory, outputBufferMemory, matMulBufferMemory;
    vkAllocateMemory(vulkan->device, &allocInfoWeights, nullptr, &weightsBufferMemory);
    vkAllocateMemory(vulkan->device, &allocInfoInput, nullptr, &inputBufferMemory);
    vkAllocateMemory(vulkan->device, &allocInfoOutput, nullptr, &outputBufferMemory);
    vkAllocateMemory(vulkan->device, &allocInfoMatMul, nullptr, &matMulBufferMemory);
    #pragma endregion

    #pragma region // Bind the memory to the buffers
    vkBindBufferMemory(vulkan->device, weightsBuffer, weightsBufferMemory, 0);
    vkBindBufferMemory(vulkan->device, inputBuffer, inputBufferMemory, 0);
    vkBindBufferMemory(vulkan->device, outputBuffer, outputBufferMemory, 0);
    vkBindBufferMemory(vulkan->device, matMulBuffer, matMulBufferMemory, 0);
    #pragma endregion

    #pragma region //Copy the weights to GPU memory
    float* weightsData;
    VkResult result = vkMapMemory(vulkan->device, weightsBufferMemory, 0, weightsBufferInfo.size, 0, (void**)&weightsData);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to map weight to GPU memory" << std::endl;
    }
    memcpy(weightsData, a->weights, weightsBufferInfo.size);
    vkUnmapMemory(vulkan->device, weightsBufferMemory);
    #pragma endregion
    
    #pragma region // Copy the input to GPU memory
    float* inputData;
    result = vkMapMemory(vulkan->device, inputBufferMemory, 0, inputBufferInfo.size, 0, (void**)&inputData);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to map input to GPU memory" << std::endl;
    }
    memcpy(inputData, a->input, inputBufferInfo.size);
    vkUnmapMemory(vulkan->device, inputBufferMemory);
    #pragma endregion
    
    #pragma region //Copy the matmul info to GPU memory
    MatMulInfo matmulInfo;
    matmulInfo.n = a->n;
    matmulInfo.de = a->de;
    matmulInfo.ds = a->ds;

    void* matMulData;
    result = vkMapMemory(vulkan->device, matMulBufferMemory, 0, matMulBufferInfo.size, 0, (void**)&matMulData);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to map matmul info to GPU memory" << std::endl;
    }
    memcpy(matMulData, &matmulInfo, matMulBufferInfo.size);
    vkUnmapMemory(vulkan->device, matMulBufferMemory);
    #pragma endregion

    #pragma region // Bind the buffers to the descriptor sets
    VkDescriptorBufferInfo weightsDescriptorBufferInfo = { weightsBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo inputDescriptorBufferInfo = { inputBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo outputDescriptorBufferInfo = { outputBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo matmulDescriptorBufferInfo = { matMulBuffer, 0, VK_WHOLE_SIZE };
    #pragma endregion

    #pragma region // Write and update the descriptor sets
    VkWriteDescriptorSet writeDescriptorSet1[3] = {};
    VkWriteDescriptorSet writeDescriptorSet2[1] = {};

    writeDescriptorSet1[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet1[0].dstBinding = 0;
    writeDescriptorSet1[0].dstSet = pipeline->descriptorSets[0];
    writeDescriptorSet1[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet1[0].descriptorCount = 1;
    writeDescriptorSet1[0].pBufferInfo = &weightsDescriptorBufferInfo;
    writeDescriptorSet1[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet1[1].dstBinding = 1;
    writeDescriptorSet1[1].dstSet = pipeline->descriptorSets[0];
    writeDescriptorSet1[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet1[1].descriptorCount = 1;
    writeDescriptorSet1[1].pBufferInfo = &inputDescriptorBufferInfo;
    writeDescriptorSet1[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet1[2].dstBinding = 2;
    writeDescriptorSet1[2].dstSet = pipeline->descriptorSets[0];
    writeDescriptorSet1[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet1[2].descriptorCount = 1;
    writeDescriptorSet1[2].pBufferInfo = &outputDescriptorBufferInfo;
    vkUpdateDescriptorSets(vulkan->device, 3, writeDescriptorSet1, 0, nullptr);
    
    writeDescriptorSet2[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet2[0].dstBinding = 0; 
    writeDescriptorSet2[0].dstSet = pipeline->descriptorSets[1];
    writeDescriptorSet2[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; 
    writeDescriptorSet2[0].descriptorCount = 1;
    writeDescriptorSet2[0].pBufferInfo = &matmulDescriptorBufferInfo;
    vkUpdateDescriptorSets(vulkan->device, 2, writeDescriptorSet2, 0, nullptr);

    VkCommandBufferBeginInfo cmdBufferBeginInfo = {};
    cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    #pragma endregion
        
    #pragma region // Create a pointer to the commandBuffer member
    VkCommandBuffer commandBuffer = pipeline->commandBuffer;
    vkBeginCommandBuffer(commandBuffer, &cmdBufferBeginInfo);
    #pragma endregion

    #pragma region // Bind pipeline and descriptor sets to the command buffer
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipelineLayout, 0, 2, pipeline->descriptorSets, 0, nullptr);
    #pragma endregion

    #pragma region // Dispatch the command buffer and stop recording
    uint numGroups = (a->de - a->ds + 15) / 16;  // Make sure to cover all elements
    vkCmdDispatch(commandBuffer, numGroups, 1, 1);
    vkEndCommandBuffer(commandBuffer);
    #pragma endregion

    // Measure execution time
    //auto start = std::chrono::high_resolution_clock::now();

    #pragma region // Wait for the compute shader to finish
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &pipeline->commandBuffer;
    vkQueueSubmit(vulkan->computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(vulkan->computeQueue);
    #pragma endregion

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double, std::milli> executionTime = end - start;
    //std::cout << "Shader execution time: " << executionTime.count() << " ms" << std::endl;

    #pragma region //Get the output from the compute shader
    float* outputData;
    result = vkMapMemory(vulkan->device, outputBufferMemory, 0, outputBufferInfo.size, 0, (void**)&outputData);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to map matrix info memory" << std::endl;
    }
    memcpy(a->output, outputData, outputBufferInfo.size);
    vkUnmapMemory(vulkan->device, outputBufferMemory);
    #pragma endregion
}

void matmulVulkanQ40vQ80(MatmulThreadInfo* a, LayerElement layerElement) {
    TransformerContext* ctx = (TransformerContext*)a->ctx;
    VulkanContext* vulkan = (VulkanContext*)ctx->vulkan;

    VulkanPipeline* pipeline = vulkan->getPipeline(VulkanPipelineType::Q40_Q80);


    auto* cachedWeightsData = vulkan->getLayerBufferData(ctx->currentBlockIndex, layerElement);
    bool weightsLoaded = cachedWeightsData != nullptr;

    /* if(weightsLoaded){
        printf("Weights cached\n");
    }
    else{
        printf("Weights uncached\n");
    } */

    #pragma region // Create buffers
    // Buffer creation info template
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer weightsBuffer, inputBuffer, outputBuffer, conversionBuffer, matMulBuffer;
    
    uint weightsSize = (a->n * (a->de - a->ds) * sizeof(BlockQ40)) / QK40;
    uint inputSize = a->n * sizeof(BlockQ80);
    uint conversionSize = sizeof(float) * 65536;

    if(!weightsLoaded){
        bufferCreateInfo.size = weightsSize;
        if (vkCreateBuffer(vulkan->device, &bufferCreateInfo, nullptr, &weightsBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create weights buffer!");
        }
    }

    bufferCreateInfo.size = inputSize;
    if (vkCreateBuffer(vulkan->device, &bufferCreateInfo, nullptr, &inputBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create input buffer!");
    }

    bufferCreateInfo.size = (a->de - a->ds) * sizeof(float);
    if (vkCreateBuffer(vulkan->device, &bufferCreateInfo, nullptr, &outputBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create output buffer!");
    }

    bufferCreateInfo.size = 65536 * sizeof(float);
    if (vkCreateBuffer(vulkan->device, &bufferCreateInfo, nullptr, &conversionBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create conversion buffer!");
    }

    bufferCreateInfo.size = sizeof(MatMulInfo);
    bufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (vkCreateBuffer(vulkan->device, &bufferCreateInfo, nullptr, &matMulBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create matmul info buffer!");
    }
    #pragma endregion

    #pragma region // Get memory requirements for the buffers 
    // Memory allocation and binding template
    VkMemoryRequirements memRequirements;
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;

    auto allocateAndBindMemory = [&](VkBuffer buffer, VkDeviceMemory& bufferMemory) {
        vkGetBufferMemoryRequirements(vulkan->device, buffer, &memRequirements);
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(vulkan->physicalDevices[0], memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(vulkan->device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate buffer memory!");
        }

        vkBindBufferMemory(vulkan->device, buffer, bufferMemory, 0);
    };

    // Allocate and bind memory
    VkDeviceMemory weightsBufferMemory, inputBufferMemory, outputBufferMemory, conversionBufferMemory, matMulBufferMemory;
    if(!weightsLoaded){
        allocateAndBindMemory(weightsBuffer, weightsBufferMemory);
    }
    allocateAndBindMemory(inputBuffer, inputBufferMemory);
    allocateAndBindMemory(outputBuffer, outputBufferMemory);
    allocateAndBindMemory(conversionBuffer, conversionBufferMemory);
    allocateAndBindMemory(matMulBuffer, matMulBufferMemory);
    #pragma endregion
    
    #pragma region // Convert and copy the data to GPU memory
    auto mapAndCopyData = [&](VkDeviceMemory bufferMemory, const void* data, VkDeviceSize size) {
        void* mappedData;
        if (vkMapMemory(vulkan->device, bufferMemory, 0, size, 0, (void**)&mappedData) != VK_SUCCESS) {
            throw std::runtime_error("Failed to map buffer memory!");
        }
        std::memcpy(mappedData, data, size);
        vkUnmapMemory(vulkan->device, bufferMemory);
    };

    if (!weightsLoaded){
        void* weightsData;
        if (vkMapMemory(vulkan->device, weightsBufferMemory, 0, weightsSize, 0, (void**)&weightsData) != VK_SUCCESS) {
            throw std::runtime_error("Failed to map buffer memory!");
        }
        std::memcpy(weightsData, a->weights, weightsSize);
        vkUnmapMemory(vulkan->device, weightsBufferMemory);
    }
    
    //mapAndCopyData(weightsBufferMemory, a->weights, a->n * (a->de - a->ds) * sizeof(BlockQ40));
    mapAndCopyData(inputBufferMemory, a->input, inputSize);
    mapAndCopyData(conversionBufferMemory, GetF16ToF32(), conversionSize);
    MatMulInfo matmulInfo = { a->n, a->ds, a->de };
    mapAndCopyData(matMulBufferMemory, &matmulInfo, sizeof(MatMulInfo));
    #pragma endregion

    #pragma region // Bind the buffers to the descriptor sets
    VkDescriptorBufferInfo weightsDescriptorBufferInfo;
    weightsDescriptorBufferInfo.offset = 0;
    weightsDescriptorBufferInfo.range = VK_WHOLE_SIZE;
    if(weightsLoaded){
        weightsDescriptorBufferInfo.buffer = cachedWeightsData->first;
    }
    else{
        weightsDescriptorBufferInfo.buffer = weightsBuffer;
    }
    VkDescriptorBufferInfo inputDescriptorBufferInfo = { inputBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo outputDescriptorBufferInfo = { outputBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo conversionDescriptorBufferInfo = { conversionBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo matmulDescriptorBufferInfo = { matMulBuffer, 0, VK_WHOLE_SIZE };
    #pragma endregion

    #pragma region // Write and update the descriptor sets
    VkWriteDescriptorSet writeDescriptorSet1[4] = {};
    VkWriteDescriptorSet writeDescriptorSet2[1] = {};

    writeDescriptorSet1[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet1[0].dstBinding = 0;
    writeDescriptorSet1[0].dstSet = pipeline->descriptorSets[0];
    writeDescriptorSet1[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet1[0].descriptorCount = 1;
    writeDescriptorSet1[0].pBufferInfo = &weightsDescriptorBufferInfo;
    writeDescriptorSet1[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet1[1].dstBinding = 1;
    writeDescriptorSet1[1].dstSet = pipeline->descriptorSets[0];
    writeDescriptorSet1[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet1[1].descriptorCount = 1;
    writeDescriptorSet1[1].pBufferInfo = &inputDescriptorBufferInfo;
    writeDescriptorSet1[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet1[2].dstBinding = 2;
    writeDescriptorSet1[2].dstSet = pipeline->descriptorSets[0];
    writeDescriptorSet1[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet1[2].descriptorCount = 1;
    writeDescriptorSet1[2].pBufferInfo = &outputDescriptorBufferInfo;
    writeDescriptorSet1[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet1[3].dstBinding = 3;
    writeDescriptorSet1[3].dstSet = pipeline->descriptorSets[0];
    writeDescriptorSet1[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet1[3].descriptorCount = 1;
    writeDescriptorSet1[3].pBufferInfo = &conversionDescriptorBufferInfo;
    vkUpdateDescriptorSets(vulkan->device, 4, writeDescriptorSet1, 0, nullptr);
    
    writeDescriptorSet2[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet2[0].dstBinding = 0; 
    writeDescriptorSet2[0].dstSet = pipeline->descriptorSets[1];
    writeDescriptorSet2[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; 
    writeDescriptorSet2[0].descriptorCount = 1;
    writeDescriptorSet2[0].pBufferInfo = &matmulDescriptorBufferInfo;
    vkUpdateDescriptorSets(vulkan->device, 1, writeDescriptorSet2, 0, nullptr);

    VkCommandBufferBeginInfo cmdBufferBeginInfo = {};
    cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    #pragma endregion
        
    #pragma region // Create a pointer to the commandBuffer member
    VkCommandBuffer commandBuffer = pipeline->commandBuffer;
    vkBeginCommandBuffer(commandBuffer, &cmdBufferBeginInfo);
    #pragma endregion

    #pragma region // Bind pipeline and descriptor sets to the command buffer
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipelineLayout, 0, 2, pipeline->descriptorSets, 0, nullptr);
    #pragma endregion

    #pragma region // Dispatch the command buffer and stop recording
    uint localGroupSize = 2;
    uint numGroups = (a->de - a->ds) / localGroupSize;
    vkCmdDispatch(pipeline->commandBuffer, numGroups, 1, 1);

    // Add a memory barrier to ensure all writes to the output buffer are visible
    VkBufferMemoryBarrier bufferMemoryBarrier = {};
    bufferMemoryBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bufferMemoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bufferMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferMemoryBarrier.buffer = outputBuffer;
    bufferMemoryBarrier.offset = 0;
    bufferMemoryBarrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0,
        0, nullptr,
        1, &bufferMemoryBarrier,
        0, nullptr
    );

    vkEndCommandBuffer(commandBuffer);
    #pragma endregion

    // Measure execution time
    //auto start = std::chrono::high_resolution_clock::now();

    #pragma region // Wait for the compute shader to finish
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &pipeline->commandBuffer;
    vkQueueSubmit(vulkan->computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(vulkan->computeQueue);
    #pragma endregion

    /* auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> executionTime = end - start;
    std::cout << "Shader execution time: " << executionTime.count() << " ms" << std::endl; */

    #pragma region // Get the output from the compute shader
    void* outputData;
    uint outputSize = (a->de - a->ds) * sizeof(float);
    if (vkMapMemory(vulkan->device, outputBufferMemory, 0, outputSize, 0, (void**)&outputData) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map buffer memory!");
    }
    std::memcpy(a->output, outputData, outputSize);
    vkUnmapMemory(vulkan->device, outputBufferMemory);
    #pragma endregion

    //Clear up data
    if(!weightsLoaded){
        vkDestroyBuffer(vulkan->device, weightsBuffer, nullptr);
        vkFreeMemory(vulkan->device, weightsBufferMemory, nullptr);
    }

    vkDestroyBuffer(vulkan->device, inputBuffer, nullptr);
    vkFreeMemory(vulkan->device, inputBufferMemory, nullptr);

    vkDestroyBuffer(vulkan->device, outputBuffer, nullptr);
    vkFreeMemory(vulkan->device, outputBufferMemory, nullptr);

    vkDestroyBuffer(vulkan->device, conversionBuffer, nullptr);
    vkFreeMemory(vulkan->device, conversionBufferMemory, nullptr);

    vkDestroyBuffer(vulkan->device, matMulBuffer, nullptr);
    vkFreeMemory(vulkan->device, matMulBufferMemory, nullptr);
}

void matmulVulkan(TransformerContext* ctx, LayerElement layerElement, FloatType weightsFloatType, FloatType inputFloatType, float* output, void* input, void* weights, int n, int d, unsigned int nThreads, unsigned int threadIndex){
    SPLIT_RANGE_TO_THREADS(ds, de, 0, d, nThreads, threadIndex);

    /*
    * Explanation of MatmulThreadInfo members:
    * - float* output: Pointer to the output array where the result of matrix multiplication is stored.
    * - void* input: Pointer to the input data (either float or quantized).
    * - void* weights: Pointer to the weight data (either float or quantized).
    * - int n: Number of columns in the input matrix (dimensionality of input).
    * - int ds: Start index of the output array slice that the thread will compute.
    * - int de: End index (exclusive) of the output array slice that the thread will compute.
    */
    MatmulThreadInfo s;
    s.ctx = ctx;
    s.output = output;
    s.input = input;
    s.weights = weights;
    s.n = n;
    s.ds = ds;
    s.de = de;

    if (inputFloatType == F32) {
        if (weightsFloatType == F32) {
            //matmulF32(&s);
            matmulVulkanF32(&s, layerElement);
            return;
        }
        if (weightsFloatType == F16) {
            matmulF16(&s);
            return;
        }
        if (weightsFloatType == Q40) {
            matmulQ40(&s);
            return;
        }
        if (weightsFloatType == Q80) {
            matmulQ80(&s);
            return;
        }
    } else if (inputFloatType == Q80) {
        if (weightsFloatType == Q40) {
            //matmulQ40vQ80(&s);
            //printf("N: %d, D: %d\n", n, d);
            matmulVulkanQ40vQ80(&s, layerElement);
            return;
        }
        if (weightsFloatType == Q80) {
            matmulQ80vQ80(&s);
            return;
        }
    }
    return;
    //exit(1);
}

void VulkanContext::printPhysicalDeviceMemoryProperties() {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevices[0], &memProperties);

    std::cout << "Memory Heaps: " << memProperties.memoryHeapCount << std::endl;
    for (uint32_t i = 0; i < memProperties.memoryHeapCount; ++i) {
        std::cout << "Heap " << i << ": " << memProperties.memoryHeaps[i].size / (1024 * 1024) << " MB" << std::endl;
    }

    std::cout << "Memory Types: " << memProperties.memoryTypeCount << std::endl;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        std::cout << "Type " << i << ": " << memProperties.memoryTypes[i].heapIndex << " " 
                  << ((memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "Device Local" : "")
                  << ((memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? " Host Visible" : "")
                  << ((memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? " Host Coherent" : "")
                  << std::endl;
    }
}

void VulkanContext::createInstance() { 
    appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan App";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = 1;
    createInfo.ppEnabledExtensionNames = enabledExtensions;
    createInfo.enabledLayerCount = 1;
    createInfo.ppEnabledLayerNames = validationLayers.data();

    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan instance." << std::endl;
        throw;
    }
    printf("Created Vulkan Instance!\n");
}

void VulkanContext::getDevice() {
    // Query for physical devices
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::cerr << "Failed to find any GPUs with Vulkan support." << std::endl;
        vkDestroyInstance(instance, nullptr);
        throw std::runtime_error("No GPUs with Vulkan support found");
    }

    physicalDevices = std::vector<VkPhysicalDevice>(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());

    // Query physical device properties
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevices[0], &deviceProperties);

    // Check device features and queues
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(physicalDevices[0], &deviceFeatures);

    // Query device extension properties
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevices[0], nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevices[0], nullptr, &extensionCount, availableExtensions.data());

    // Check if VK_KHR_8bit_storage is supported
    bool is8BitStorageSupported = false;
    for (const auto& extension : availableExtensions) {
        if (strcmp(extension.extensionName, VK_KHR_8BIT_STORAGE_EXTENSION_NAME) == 0) {
            is8BitStorageSupported = true;
            break;
        }
    }

    if (!is8BitStorageSupported) {
        std::cerr << "VK_KHR_8bit_storage extension is not supported by this device." << std::endl;
        vkDestroyInstance(instance, nullptr);
        throw std::runtime_error("VK_KHR_8bit_storage extension not supported");
    }

    // List of required extensions
    const char* enabledExtensions[] = {
        VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME,
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME
    };

    // Query the physical device features
    VkPhysicalDeviceFeatures2 physicalDeviceFeatures2 = {};
    physicalDeviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceVulkan12Features vk12Features = {};
    vk12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDevice16BitStorageFeatures storage16BitFeatures = {};
    storage16BitFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures = {};
    atomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;

    // Chain the features
    storage16BitFeatures.pNext = &vk12Features;
    vk12Features.pNext = &atomicFloatFeatures;
    physicalDeviceFeatures2.pNext = &storage16BitFeatures;

    // Query physical device features2
    vkGetPhysicalDeviceFeatures2(physicalDevices[0], &physicalDeviceFeatures2);

    // Enable the required features
    vk12Features.storageBuffer8BitAccess = VK_TRUE;
    vk12Features.uniformAndStorageBuffer8BitAccess = VK_TRUE;
    storage16BitFeatures.storageBuffer16BitAccess = VK_TRUE;

    // Print physical device properties
    std::cout << "Device Name: " << deviceProperties.deviceName << std::endl;
    std::cout << "API Version: " << VK_VERSION_MAJOR(deviceProperties.apiVersion) << "."
              << VK_VERSION_MINOR(deviceProperties.apiVersion) << "."
              << VK_VERSION_PATCH(deviceProperties.apiVersion) << std::endl;

    // Query device queue families
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevices[0], &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevices[0], &queueFamilyCount, queueFamilies.data());

    // Find a queue family that supports compute operations
    computeQueueFamilyIndex = -1;
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamilyIndex = i;
            break;
        }
    }

    if (computeQueueFamilyIndex == -1) {
        std::cerr << "No compute queue family found." << std::endl;
        vkDestroyInstance(instance, nullptr);
        throw std::runtime_error("No compute queue family found");
    }

    // Create logical device
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledExtensionCount = sizeof(enabledExtensions) / sizeof(enabledExtensions[0]);
    deviceCreateInfo.ppEnabledExtensionNames = enabledExtensions;
    deviceCreateInfo.pNext = &physicalDeviceFeatures2; // Chain the features

    VkResult result = vkCreateDevice(physicalDevices[0], &deviceCreateInfo, nullptr, &device);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create logical device." << std::endl;
        vkDestroyInstance(instance, nullptr);
        throw std::runtime_error("Failed to create logical device");
    }

    // Get the compute queue
    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
}

VkShaderModule VulkanContext::loadComputeShaderModule(const std::string &shaderPath) {
    std::vector<char> shaderCode = readFile(shaderPath);
    return createShaderModule(device, shaderCode);
}

void VulkanContext::createPipeline(VulkanPipelineType pipelineType, const std::string &name) {
    VulkanPipeline vulkanPipeline = {};
    if(pipelineType == VulkanPipelineType::F32_F32){
        vulkanPipeline.weightsFloatType = FloatType::F32;
        vulkanPipeline.inputFloatType = FloatType::F32;
    }
    else if(pipelineType == VulkanPipelineType::Q40_Q80){
        vulkanPipeline.weightsFloatType = FloatType::Q40;
        vulkanPipeline.inputFloatType = FloatType::Q80;
    }
    vulkanPipeline.shaderModule = loadComputeShaderModule(getShaderPath(pipelineType));
    
    // Define the descriptor set layout bindings
    VkDescriptorSetLayoutBinding weightsBufferBinding = {};
    weightsBufferBinding.binding = 0; // Binding point for the input buffer
    weightsBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    weightsBufferBinding.descriptorCount = 1;
    weightsBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding inputBufferBinding = {};
    inputBufferBinding.binding = 1; // Binding point for the input buffer
    inputBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    inputBufferBinding.descriptorCount = 1;
    inputBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding outputBufferBinding = {};
    outputBufferBinding.binding = 2; // Binding point for the output buffer
    outputBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    outputBufferBinding.descriptorCount = 1;
    outputBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding conversionBufferBinding = {};
    conversionBufferBinding.binding = 3; // Binding point for the output buffer
    conversionBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    conversionBufferBinding.descriptorCount = 1;
    conversionBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding matMulInfoBinding = {};
    matMulInfoBinding.binding = 0; // Binding point for the MatrixInfo uniform buffer
    matMulInfoBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    matMulInfoBinding.descriptorCount = 1;
    matMulInfoBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    const VkDescriptorSetLayoutBinding bindingsSet0[] = {
        weightsBufferBinding, inputBufferBinding, outputBufferBinding, conversionBufferBinding
    };

    const VkDescriptorSetLayoutBinding bindingsSet1[] = {
        matMulInfoBinding
    };

    // Create the descriptor set layout for set 0
    VkDescriptorSetLayoutCreateInfo descriptorLayoutCreateInfoSet0 = {};
    descriptorLayoutCreateInfoSet0.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorLayoutCreateInfoSet0.bindingCount = 4;
    descriptorLayoutCreateInfoSet0.pBindings = bindingsSet0;

    // Create the descriptor set layout for set 1
    VkDescriptorSetLayoutCreateInfo descriptorLayoutCreateInfoSet1 = {};
    descriptorLayoutCreateInfoSet1.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorLayoutCreateInfoSet1.bindingCount = 1;
    descriptorLayoutCreateInfoSet1.pBindings = bindingsSet1;

    VkResult result = vkCreateDescriptorSetLayout(device, &descriptorLayoutCreateInfoSet0, nullptr, &vulkanPipeline.descriptorSetLayoutSet0);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create descriptorSetLayoutSet0." << std::endl;
    }
    result = vkCreateDescriptorSetLayout(device, &descriptorLayoutCreateInfoSet1, nullptr, &vulkanPipeline.descriptorSetLayoutSet1);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create descriptorSetLayoutSet1." << std::endl;
    }

    VkDescriptorSetLayout layouts[] = { vulkanPipeline.descriptorSetLayoutSet0, vulkanPipeline.descriptorSetLayoutSet1 };

    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = vulkanPipeline.shaderModule;
    shaderStageInfo.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 2;
    pipelineLayoutInfo.pSetLayouts = layouts;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &vulkanPipeline.pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout!");
    }

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = vulkanPipeline.pipelineLayout;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &vulkanPipeline.pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline!");
    }

    // Allocate the descriptor sets
    VkDescriptorPoolSize poolSizeStorage = {};
    poolSizeStorage.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizeStorage.descriptorCount = 4;

    VkDescriptorPoolSize poolSizeUniform = {};
    poolSizeUniform.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizeUniform.descriptorCount = 1;

    VkDescriptorPoolSize poolSizes[] = {poolSizeStorage, poolSizeUniform};

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets = 2;
    descriptorPoolCreateInfo.poolSizeCount = 2;
    descriptorPoolCreateInfo.pPoolSizes = poolSizes;

    vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &vulkanPipeline.descriptorPool);

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = vulkanPipeline.descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 2;
    descriptorSetAllocateInfo.pSetLayouts = layouts;

    // Allocate the descriptor sets
    vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, vulkanPipeline.descriptorSets);

    // Create the command pool
    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &vulkanPipeline.commandPool);

    // Allocate and begin recording the command buffer
    VkCommandBufferAllocateInfo cmdBufferAllocateInfo = {};
    cmdBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufferAllocateInfo.commandPool = vulkanPipeline.commandPool;
    cmdBufferAllocateInfo.commandBufferCount = 1;

    vkAllocateCommandBuffers(device, &cmdBufferAllocateInfo, &vulkanPipeline.commandBuffer);

    pipelines[pipelineType] = vulkanPipeline;
    printf("Created pipeline %s\n", name.c_str());
}

VulkanPipeline* VulkanContext::getPipeline(VulkanPipelineType pipelineType){
    // Check if the pipeline exists in the map
    auto it = pipelines.find(pipelineType);
    if (it != pipelines.end()) {
        return &(it->second);
    } else {
        throw std::runtime_error("No pipeline exists for the specified floatType.");
        return nullptr;
    }
}

//If the block could be successfully loaded into VRAM, this returns true
bool VulkanContext::loadTransformerBlock(int blockIndex, TransformerBlock* block){
    #pragma region // Create the buffers
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer qBuffer, kBuffer, vBuffer;

    bufferCreateInfo.size = block->q0Slice->sliceBytes;
    if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &qBuffer) != VK_SUCCESS) {
        return false;
    }
    
    bufferCreateInfo.size = block->k0Slice->sliceBytes;
    if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &kBuffer) != VK_SUCCESS) {
        vkDestroyBuffer(device, qBuffer, nullptr);
        return false;
    }
    
    bufferCreateInfo.size = block->v0Slice->sliceBytes;
    if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &vBuffer) != VK_SUCCESS) {
        vkDestroyBuffer(device, qBuffer, nullptr);
        vkDestroyBuffer(device, vBuffer, nullptr);
        return false;
    }
    #pragma endregion
    
    #pragma region // Get the buffer memory requirements
    
    VkMemoryRequirements memRequirements;
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    std::function<bool(VkBuffer, VkDeviceMemory&)> allocateAndBindMemory = [&](VkBuffer buffer, VkDeviceMemory& bufferMemory) -> bool {
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(physicalDevices[0], memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            return false;
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
        return true;
    };
    VkDeviceMemory qBufferMemory, kBufferMemory, vBufferMemory;
    if(!allocateAndBindMemory(qBuffer, qBufferMemory)){
        vkDestroyBuffer(device, qBuffer, nullptr);
        vkDestroyBuffer(device, kBuffer, nullptr);
        vkDestroyBuffer(device, vBuffer, nullptr);
        return false;
    }
    if(!allocateAndBindMemory(kBuffer, kBufferMemory)){
        vkDestroyBuffer(device, qBuffer, nullptr);
        vkDestroyBuffer(device, kBuffer, nullptr);
        vkDestroyBuffer(device, vBuffer, nullptr);
        vkFreeMemory(device, qBufferMemory, nullptr);
        return false;
    }
    if(!allocateAndBindMemory(vBuffer, vBufferMemory)){
        vkDestroyBuffer(device, qBuffer, nullptr);
        vkDestroyBuffer(device, kBuffer, nullptr);
        vkDestroyBuffer(device, vBuffer, nullptr);
        vkFreeMemory(device, qBufferMemory, nullptr);
        vkFreeMemory(device, kBufferMemory, nullptr);
        return false;
    }
    #pragma endregion

    #pragma region // Copy the data to GPU memory
    auto mapAndCopyData = [&](VkDeviceMemory bufferMemory, const void* data, VkDeviceSize size) {
        void* mappedData;
        if (vkMapMemory(device, bufferMemory, 0, size, 0, (void**)&mappedData) != VK_SUCCESS) {
            return false;
        }
        std::memcpy(mappedData, data, size);
        vkUnmapMemory(device, bufferMemory);
        return true;
    };
    
    //Copy Query buffer data
    void* qData;
    if (vkMapMemory(device, qBufferMemory, 0, block->q0Slice->sliceBytes, 0, (void**)&qData) != VK_SUCCESS) {
        //Cleanup 
        vkDestroyBuffer(device, qBuffer, nullptr);
        vkDestroyBuffer(device, kBuffer, nullptr);
        vkDestroyBuffer(device, vBuffer, nullptr);
        vkFreeMemory(device, qBufferMemory, nullptr);
        vkFreeMemory(device, kBufferMemory, nullptr);
        vkFreeMemory(device, vBufferMemory, nullptr);
        return false;
    }
    std::memcpy(qData, block->q0, block->q0Slice->sliceBytes);
    vkUnmapMemory(device, qBufferMemory);

    //Copy Key buffer data
    void* kData;
    if (vkMapMemory(device, kBufferMemory, 0, block->k0Slice->sliceBytes, 0, (void**)&kData) != VK_SUCCESS) {
        //Cleanup 
        vkDestroyBuffer(device, qBuffer, nullptr);
        vkDestroyBuffer(device, kBuffer, nullptr);
        vkDestroyBuffer(device, vBuffer, nullptr);
        vkFreeMemory(device, qBufferMemory, nullptr);
        vkFreeMemory(device, kBufferMemory, nullptr);
        vkFreeMemory(device, vBufferMemory, nullptr);
        return false;
    }
    std::memcpy(qData, block->k0, block->k0Slice->sliceBytes);
    vkUnmapMemory(device, kBufferMemory);

    //Copy Value buffer data
    void* vData;
    if (vkMapMemory(device, vBufferMemory, 0, block->v0Slice->sliceBytes, 0, (void**)&vData) != VK_SUCCESS) {
        //Cleanup 
        vkDestroyBuffer(device, qBuffer, nullptr);
        vkDestroyBuffer(device, kBuffer, nullptr);
        vkDestroyBuffer(device, vBuffer, nullptr);
        vkFreeMemory(device, qBufferMemory, nullptr);
        vkFreeMemory(device, kBufferMemory, nullptr);
        vkFreeMemory(device, vBufferMemory, nullptr);
        return false;
    }
    std::memcpy(vData, block->v0, block->v0Slice->sliceBytes);
    vkUnmapMemory(device, vBufferMemory);

    #pragma endregion

    //Now that the layer has been mapped to GPU memory, we store the buffer

    bufferMap[std::make_pair(blockIndex, LayerElement::QUERY)] = std::make_pair(qBuffer, qBufferMemory);
    bufferMap[std::make_pair(blockIndex, LayerElement::KEY)] = std::make_pair(kBuffer, kBufferMemory);
    bufferMap[std::make_pair(blockIndex, LayerElement::VALUE)] = std::make_pair(vBuffer, vBufferMemory);

    return true;
}

std::pair<VkBuffer, VkDeviceMemory>* VulkanContext::getLayerBufferData(int blockIndex, LayerElement type) {
    auto it = bufferMap.find({blockIndex, type});
    if (it != bufferMap.end()) {
        // Return the address of the VkBuffer stored in the map
        return &it->second;
    }
    // Return nullptr if the buffer is not found
    return nullptr;
}

void VulkanContext::initialize() {
    // Create the Vulkan Instance
    createInstance();

    // Get the first* Vulkan enabled device
    getDevice();

    printPhysicalDeviceMemoryProperties();

    // Load the matmul F32 compute shader module
    //loadComputeShaderModule("./shaders/matmulF32.spv");

    // Create the descriptor set layout
    //createDescriptorSetLayout();

    // Create the pipeline for the matmul F32 compute shader
    //createPipeline(VulkanPipelineType::F32_F32, "F32_F32");

    // Create the pipeline for the matmul Q40 compute shader
    createPipeline(VulkanPipelineType::Q40_Q80, "Q40_Q80");
}

VulkanContext::VulkanContext(Transformer* transformer) {
    //Setup the Vulkan Instance
    initialize();

    if(transformer != nullptr){
    //Loop through transformer layers, storing them in GPU memory until we can't store anymore
        for (int i = 0; i < transformer->spec->nLayers; i++) {
            TransformerBlock* b = transformer->blocks[i];
            std::cout << "Layer " << i << " [Q] N: " <<  b->q0Slice->n << ", D: " << b->q0Slice->d0 << std::endl;
            std::cout << "Layer " << i << " [K] N: " <<  b->k0Slice->n << ", D: " << b->k0Slice->d0 << std::endl;
            std::cout << "Layer " << i << " [V] N: " <<  b->v0Slice->n << ", D: " << b->v0Slice->d0 << std::endl;
            if(!loadTransformerBlock(i, b)){
                //We can't load anymore layers into GPU memory
                break;
            }
            std::cout << "Loaded Layer " << i << " into GPU memory" << std::endl;
        }
    }
}

VulkanContext::~VulkanContext() {
    std::cout << "Destroying VulkanContext" << std::endl;

    // Destroy buffers
    for (const auto& pair : bufferMap) {
        vkDestroyBuffer(device, pair.second.first, nullptr);
        vkFreeMemory(device, pair.second.second, nullptr);
    }

    for (const auto& pipeline : pipelines) {
        vkDestroyShaderModule(device, pipeline.second.shaderModule, nullptr);

        vkDestroyCommandPool(device, pipeline.second.commandPool, nullptr);

        vkDestroyDescriptorPool(device, pipeline.second.descriptorPool, nullptr);

        vkDestroyDescriptorSetLayout(device, pipeline.second.descriptorSetLayoutSet0, nullptr);
        vkDestroyDescriptorSetLayout(device, pipeline.second.descriptorSetLayoutSet1, nullptr);

        vkDestroyPipelineLayout(device, pipeline.second.pipelineLayout, nullptr);
        vkDestroyPipeline(device, pipeline.second.pipeline, nullptr);
    }

    // Finally, destroy the Vulkan device and instance
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
}