#define VK_USE_PLATFORM_XCB_KHR
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <vulkan/vulkan.h>
#include "vulkan.hpp"
#include "quants.hpp"
#include "funcs.hpp"

// Validation layers to enable
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// Enabled extensions
const char* enabledExtensions[] = {
    //VK_NV_GLSL_SHADER_EXTENSION_NAME
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

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

void matmulVulkanF32(MatmulVulkanInfo* a){
    VulkanContext* vulkan = static_cast<VulkanContext*>(a->vulkan);

    // Create the buffers
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

    // Get memory requirements for the buffers
    VkMemoryRequirements memRequirementsWeights;
    vkGetBufferMemoryRequirements(vulkan->device, weightsBuffer, &memRequirementsWeights);
    VkMemoryRequirements memRequirementsInput;
    vkGetBufferMemoryRequirements(vulkan->device, inputBuffer, &memRequirementsInput);
    VkMemoryRequirements memoryRequirementsOutput;
    vkGetBufferMemoryRequirements(vulkan->device, outputBuffer, &memoryRequirementsOutput);
    VkMemoryRequirements memoryRequirementsMatMulInfo;
    vkGetBufferMemoryRequirements(vulkan->device, matMulBuffer, &memoryRequirementsMatMulInfo);

    // Get memory allocation requirements
    VkMemoryAllocateInfo allocInfoWeights = {};
    allocInfoWeights.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfoWeights.allocationSize = memRequirementsWeights.size;
    allocInfoWeights.memoryTypeIndex = findMemoryType(vulkan->physicalDevice, memRequirementsWeights.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo allocInfoInput = {};
    allocInfoInput.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfoInput.allocationSize = memRequirementsInput.size;
    allocInfoInput.memoryTypeIndex = findMemoryType(vulkan->physicalDevice, memRequirementsInput.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo allocInfoOutput = {};
    allocInfoOutput.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfoOutput.allocationSize = memoryRequirementsOutput.size;
    allocInfoOutput.memoryTypeIndex = findMemoryType(vulkan->physicalDevice, memoryRequirementsOutput.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo allocInfoMatMul = {};
    allocInfoMatMul.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfoMatMul.allocationSize = memoryRequirementsMatMulInfo.size;
    allocInfoMatMul.memoryTypeIndex = findMemoryType(vulkan->physicalDevice, memoryRequirementsMatMulInfo.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    //Allocate and map memory for the buffers
    VkDeviceMemory weightsBufferMemory, inputBufferMemory, outputBufferMemory, matMulBufferMemory;
    vkAllocateMemory(vulkan->device, &allocInfoWeights, nullptr, &weightsBufferMemory);
    vkAllocateMemory(vulkan->device, &allocInfoInput, nullptr, &inputBufferMemory);
    vkAllocateMemory(vulkan->device, &allocInfoOutput, nullptr, &outputBufferMemory);
    vkAllocateMemory(vulkan->device, &allocInfoMatMul, nullptr, &matMulBufferMemory);

    // Bind the memory to the buffers
    vkBindBufferMemory(vulkan->device, weightsBuffer, weightsBufferMemory, 0);
    vkBindBufferMemory(vulkan->device, inputBuffer, inputBufferMemory, 0);
    vkBindBufferMemory(vulkan->device, outputBuffer, outputBufferMemory, 0);
    vkBindBufferMemory(vulkan->device, matMulBuffer, matMulBufferMemory, 0);

    // Copy the weights to GPU memory
    float* weightsData;
    VkResult result = vkMapMemory(vulkan->device, weightsBufferMemory, 0, weightsBufferInfo.size, 0, (void**)&weightsData);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to map weight to GPU memory" << std::endl;
    }
    memcpy(weightsData, a->weights, weightsBufferInfo.size);
    vkUnmapMemory(vulkan->device, weightsBufferMemory);
    
    // Copy the input to GPU memory
    float* inputData;
    result = vkMapMemory(vulkan->device, inputBufferMemory, 0, inputBufferInfo.size, 0, (void**)&inputData);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to map input to GPU memory" << std::endl;
    }
    memcpy(inputData, a->input, inputBufferInfo.size);
    vkUnmapMemory(vulkan->device, inputBufferMemory);
    
    MatMulInfo matmulInfo;
    matmulInfo.n = a->n;
    matmulInfo.de = a->de;
    matmulInfo.ds = a->ds;

    // Copy the matmul info to GPU memory
    void* matMulData;
    result = vkMapMemory(vulkan->device, matMulBufferMemory, 0, matMulBufferInfo.size, 0, (void**)&matMulData);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to map matmul info to GPU memory" << std::endl;
    }
    memcpy(matMulData, &matmulInfo, matMulBufferInfo.size);
    vkUnmapMemory(vulkan->device, matMulBufferMemory);

    // Bind the buffers to the descriptor sets
    VkDescriptorBufferInfo weightsDescriptorBufferInfo = { weightsBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo inputDescriptorBufferInfo = { inputBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo outputDescriptorBufferInfo = { outputBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo matmulDescriptorBufferInfo = { matMulBuffer, 0, VK_WHOLE_SIZE };

    // Write and update the descriptor sets
    VkWriteDescriptorSet writeDescriptorSet1[3] = {};
    VkWriteDescriptorSet writeDescriptorSet2[1] = {};

    writeDescriptorSet1[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet1[0].dstBinding = 0;
    writeDescriptorSet1[0].dstSet = vulkan->descriptorSets[0];
    writeDescriptorSet1[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet1[0].descriptorCount = 1;
    writeDescriptorSet1[0].pBufferInfo = &weightsDescriptorBufferInfo;
    writeDescriptorSet1[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet1[1].dstBinding = 1;
    writeDescriptorSet1[1].dstSet = vulkan->descriptorSets[0];
    writeDescriptorSet1[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet1[1].descriptorCount = 1;
    writeDescriptorSet1[1].pBufferInfo = &inputDescriptorBufferInfo;
    writeDescriptorSet1[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet1[2].dstBinding = 2;
    writeDescriptorSet1[2].dstSet = vulkan->descriptorSets[0];
    writeDescriptorSet1[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet1[2].descriptorCount = 1;
    writeDescriptorSet1[2].pBufferInfo = &outputDescriptorBufferInfo;
    vkUpdateDescriptorSets(vulkan->device, 3, writeDescriptorSet1, 0, nullptr);
    
    writeDescriptorSet2[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet2[0].dstBinding = 0; 
    writeDescriptorSet2[0].dstSet = vulkan->descriptorSets[1];
    writeDescriptorSet2[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; 
    writeDescriptorSet2[0].descriptorCount = 1;
    writeDescriptorSet2[0].pBufferInfo = &matmulDescriptorBufferInfo;
    vkUpdateDescriptorSets(vulkan->device, 2, writeDescriptorSet2, 0, nullptr);

    VkCommandBufferBeginInfo cmdBufferBeginInfo = {};
    cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        
    // Create a pointer to the commandBuffer member
    VkCommandBuffer commandBuffer = vulkan->commandBuffer;

    vkBeginCommandBuffer(commandBuffer, &cmdBufferBeginInfo);

    // Bind pipeline and descriptor sets to the command buffer
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vulkan->pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vulkan->pipelineLayout, 0, 2, vulkan->descriptorSets, 0, nullptr);

    vkCmdDispatch(commandBuffer, 1, 1, 1);

    vkEndCommandBuffer(commandBuffer);
    
    // Wait for the compute shader to finish
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &vulkan->commandBuffer;

    vkQueueSubmit(vulkan->computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(vulkan->computeQueue);

    //Get the output from the compute shader
    float* outputData;
    result = vkMapMemory(vulkan->device, outputBufferMemory, 0, outputBufferInfo.size, 0, (void**)&outputData);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to map matrix info memory" << std::endl;
    }
    memcpy(a->output, &outputData, outputBufferInfo.size);
    vkUnmapMemory(vulkan->device, outputBufferMemory);
    printf("Processed output using vulkan!\n");
}

void matmulVulkan(VulkanContext* vulkan, FloatType weightsFloatType, FloatType inputFloatType, float* output, void* input, void* weights, int n, int d, unsigned int nThreads, unsigned int threadIndex){
    SPLIT_RANGE_TO_THREADS(ds, de, 0, d, nThreads, threadIndex);
    
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
    MatmulThreadInfo s;
    s.output = output;
    s.input = input;
    s.weights = weights;
    s.n = n;
    s.ds = ds;
    s.de = de;

    if (inputFloatType == F32) {
        if (weightsFloatType == F32) {
            MatmulVulkanInfo v;
            v.vulkan = vulkan;
            v.output = output;
            v.input = input;
            v.weights = weights;
            v.n = n;
            v.ds = ds;
            v.de = de;
            //matmulF32(&s);
            matmulVulkanF32(&v);
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
            matmulQ40vQ80(&s);
            return;
        }
        if (weightsFloatType == Q80) {
            matmulQ80vQ80(&s);
            return;
        }
    }
    
    printf("Success\n");
    exit(1);
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
    if (sizeof(enabledExtensions) > 0){
        createInfo.enabledExtensionCount = sizeof(enabledExtensions) / sizeof(enabledExtensions[0]);
        createInfo.ppEnabledExtensionNames = enabledExtensions;
    }
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
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
        throw;
    }

    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());

    // Choose a physical device (for simplicity, just choose the first one)
    VkPhysicalDevice physicalDevice = physicalDevices[0];

    // Query physical device properties
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);

    // Check device features and queues
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);

    // Print physical device properties
    std::cout << "Device Name: " << deviceProperties.deviceName << std::endl;
    std::cout << "API Version: " << VK_VERSION_MAJOR(deviceProperties.apiVersion) << "."
        << VK_VERSION_MINOR(deviceProperties.apiVersion) << "."
        << VK_VERSION_PATCH(deviceProperties.apiVersion) << std::endl;

    // Query device queue families
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

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
        throw;
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
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

    VkResult result = vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create logical device." << std::endl;
        vkDestroyInstance(instance, nullptr);
        throw;
    }

    // Get the compute queue
    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
}

void VulkanContext::loadComputeShaderModule(const std::string &shaderPath) {
    std::vector<char> shaderCode = readFile(shaderPath);
    shaderModule = createShaderModule(device, shaderCode);
}

void VulkanContext::createDescriptorSetLayout() {
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

    VkDescriptorSetLayoutBinding matMulInfoBinding = {};
    matMulInfoBinding.binding = 0; // Binding point for the MatrixInfo uniform buffer
    matMulInfoBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    matMulInfoBinding.descriptorCount = 1;
    matMulInfoBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    const VkDescriptorSetLayoutBinding bindingsSet0[] = {
        weightsBufferBinding, inputBufferBinding, outputBufferBinding
    };

    const VkDescriptorSetLayoutBinding bindingsSet1[] = {
        matMulInfoBinding
    };

    // Create the descriptor set layout for set 0
    VkDescriptorSetLayoutCreateInfo descriptorLayoutCreateInfoSet0 = {};
    descriptorLayoutCreateInfoSet0.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorLayoutCreateInfoSet0.bindingCount = 3;
    descriptorLayoutCreateInfoSet0.pBindings = bindingsSet0;

    // Create the descriptor set layout for set 1
    VkDescriptorSetLayoutCreateInfo descriptorLayoutCreateInfoSet1 = {};
    descriptorLayoutCreateInfoSet1.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorLayoutCreateInfoSet1.bindingCount = 1;
    descriptorLayoutCreateInfoSet1.pBindings = bindingsSet1;

    VkResult result = vkCreateDescriptorSetLayout(device, &descriptorLayoutCreateInfoSet0, nullptr, &descriptorSetLayoutSet0);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create descriptorSetLayoutSet0." << std::endl;
    }
    result = vkCreateDescriptorSetLayout(device, &descriptorLayoutCreateInfoSet1, nullptr, &descriptorSetLayoutSet1);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create descriptorSetLayoutSet1." << std::endl;
    }
}

void VulkanContext::createPipeline() {
    VkDescriptorSetLayout layouts[] = { descriptorSetLayoutSet0, descriptorSetLayoutSet1 };

    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = shaderModule;
    shaderStageInfo.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 2;
    pipelineLayoutInfo.pSetLayouts = layouts;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout!");
    }

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = pipelineLayout;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline!");
    }

    // Allocate the descriptor sets
    VkDescriptorPoolSize poolSizeStorage = {};
    poolSizeStorage.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizeStorage.descriptorCount = 3;

    VkDescriptorPoolSize poolSizeUniform = {};
    poolSizeUniform.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizeUniform.descriptorCount = 1;

    VkDescriptorPoolSize poolSizes[] = {poolSizeStorage, poolSizeUniform};

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets = 2;
    descriptorPoolCreateInfo.poolSizeCount = 2;
    descriptorPoolCreateInfo.pPoolSizes = poolSizes;

    vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool);

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 2;
    descriptorSetAllocateInfo.pSetLayouts = layouts;

    // Allocate the descriptor sets
    descriptorSets[2] = {};
    vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, descriptorSets);

    // Create the command pool
    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool);

    // Allocate and begin recording the command buffer
    VkCommandBufferAllocateInfo cmdBufferAllocateInfo = {};
    cmdBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufferAllocateInfo.commandPool = commandPool;
    cmdBufferAllocateInfo.commandBufferCount = 1;

    vkAllocateCommandBuffers(device, &cmdBufferAllocateInfo, &commandBuffer);
}

void VulkanContext::initialize() {
    // Create the Vulkan Instance
    createInstance();

    // Get the first* Vulkan enabled device
    getDevice();

    // Load the matmul F32 compute shader module
    loadComputeShaderModule("./shaders/matmulF32.spv");

    // Create the descriptor set layout
    createDescriptorSetLayout();

    // Create the pipeline for the matmul F32 compute shader
    createPipeline();
}

VulkanContext::VulkanContext() {
    initialize();
}

VulkanContext::~VulkanContext() {
    printf("Destroying Vulkan Instance\n");
    vkDestroyInstance(instance, nullptr);
}