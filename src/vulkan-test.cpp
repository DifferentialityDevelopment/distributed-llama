#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <iostream>
#include <ostream>
#include <chrono>
#include <random>
#include "transformer.hpp"
#include "vulkan.hpp"
#include "utils.hpp"
#include "funcs.hpp"

// Function to print the contents of an array, useful for debugging
void printArray(const float* arr, int size, const std::string& name) {
    std::cout << name << ": ";
    for (int i = 0; i < size; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

void printBlockQ40Row(BlockQ40* row, const std::string& name) {
    std::cout << name << " [D:"<< row->d << "," << convertF16ToF32(row->d) <<"]" << ": ";
    float* data = new float[16];
    dequantizeQ40Row(row, data, 32);
    for (int i = 0; i < 16; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

void testQuantizeQ40(){
    const int n = 4096;
    const int d = 4096;
    unsigned long long state = 88888888L;
    float* w = new float[n * d];  // weights matrix
    
    //generate fp32 weights
    int i;
    for (i = 0; i < n * d; i++) w[i] = randomF32(&state) / 127.0f;
    
    // Print some of the FP32 weights
    printf("FP32 Weights\n");
    for (int i = 0; i < 32 && i < n * d; i++) {
        std::cout << w[i] << ", ";
    }
    std::cout << "\n";

    // Quantize the weights to q4
    int size = getBatchBytes(Q40, n, d);
    char* wQ4 = new (std::nothrow) char[size];
    if (!wQ4) {
        std::cerr << "Memory allocation failed for wQ4." << std::endl;
        return;
    }
    quantizeQ40Row(w, (BlockQ40*)wQ4, n * d, 1, 0);

    // Dequantize the Q40 weights
    float* wDeq = new float[n * d];
    dequantizeQ40Row((BlockQ40*)wQ4, wDeq, n * d);

    // Print some of the dequantized Q40 weights
    printf("Dequantized Q40 Weights\n");
    uint diff = 0;
    for (int i = 0; i < 32; i++) {
        std::cout << wDeq[i] << ", ";
        if(std::fabs(wDeq[i]-w[i]) > 0.001){
            diff++;
        }
    }
    std::cout << "\n";
    std::cout << "Diff: " << diff << std::endl;

    delete[] wQ4;
    delete[] wDeq;
}

float getRandomFloat() {
    // Create a random device and seed the generator
    std::random_device rd;
    std::mt19937 gen(rd());
    // Define a uniform distribution in the range [0.0, 1.0)
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // Generate and return a random float
    return dis(gen);
}

void testMatmulQ80(VulkanContext* vulkan) {
    const int n = 2048;
    const int d = 2048;
    unsigned long long wstate = 46546541L;
    unsigned long long istate = 95874562L;
    float* x = new float[n];  // input matrix
    float* w = new float[n * d];  // weights matrix
    //float outputCPUComparison[d]; // output matrix for CPU
    float outputCPUQ4[d]; // output matrix for CPU
    float outputVulkanQ4[d]; // output matrix for Vulkan
    //float outputCPUQ8[d]; // output matrix for CPU

    TransformerContext* ctx = new TransformerContext();
    ctx->vulkan = vulkan;

    //float outputVulkanQ8[d]; // output matrix for Vulkan
    int i;
    for (i = 0; i < n; i++){
        x[i] = getRandomFloat() / 127.0f;//randomF32(&istate) / 127.0f;
    }
    for (i = 0; i < n * d; i++){
        w[i] = getRandomFloat() / 127.0f;//randomF32(&wstate) / 127.0f;
    }

    char* xQ = new char[getBatchBytes(Q80, n, 1)];
    char* wQ8 = new char[getBatchBytes(Q80, n, d)];
    char* wQ4 = new char[getBatchBytes(Q40, n, d)];

    quantizeQ80Row(x, (BlockQ80*)xQ, n, 1, 0);
    quantizeQ40Row(w, (BlockQ40*)wQ4, n * d, 1, 0);
    //quantizeQ80Row(w, (BlockQ80*)wQ8, n * d, 1, 0);

    //matmul(F32, F32, outputCPUComparison, x, w, n, d, 1, 0);
    //matmul(Q80, Q80, outputCPUQ8, xQ, wQ8, n, d, 1, 0);
    //matmulVulkan(vulkan, Q80, Q80, outputVulkanQ8, xQ, wQ8, n, d, 1, 0);
    auto start = std::chrono::high_resolution_clock::now();
    matmul(Q40, Q80, outputCPUQ4, xQ, wQ4, n, d, 1, 0);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> executionTime = end - start;
    std::cout << "CPU Q40/Q80 execution time: " << executionTime.count() << " ms" << std::endl;
    matmulVulkan(ctx, LayerElement::KEY, Q40, Q80, outputVulkanQ4, xQ, wQ4, n, d, 1, 0);

    printArray(x, 16, "Input");
    printf("\n");
    printBlockQ40Row((BlockQ40*)wQ4, "Weights");
    printf("\n");
    //printArray(outputCPUQ8, 32, "CPU Results @ Q80");
    //printf("\n");
    //printArray(outputVulkanQ8, 32, "Vulkan Results @ Q80");
    printf("\n");
    printArray(outputCPUQ4, 16, "CPU Results @ Q40");
    printf("\n");
    printArray(outputVulkanQ4, 16, "Vulkan Results @ Q40");
    printf("\n");

    uint diff = 0;
    for (int i = 0; i < d; i++) {
        if(std::fabs(outputCPUQ4[i]-outputVulkanQ4[i]) > 0.001){
            std::cout << i << ": " << "CPU: " << outputCPUQ4[i] << " != GPU: " << outputVulkanQ4[i] << std::endl;
            diff++;
        }
    }
    std::cout << "\n";
    std::cout << "Diff: " << diff << "/" << d << std::endl;
    
    delete[] xQ;
    delete[] wQ4;
    delete[] wQ8;
}

void runMatmulQ80Test(VulkanContext* vulkan) {
    const int n = 4096; //Crashes above this
    const int d = 4096; //Crashes above this
    unsigned long long state = 88888888L;
    float x[n]; // input matrix
    float w[n * d];  // weights matrix
    float yQ3[d]; // output matrix for CPU
    float yQ2[d]; // output matrix for Vulkan

    TransformerContext* ctx;
    ctx->vulkan = vulkan;

    // Initialize input and weights with random values
    for (int i = 0; i < n; i++) x[i] = randomF32(&state) / 127.0f;
    for (int i = 0; i < n * d; i++) w[i] = randomF32(&state) / 127.0f;

    // Allocate memory for input and weights
    char* xQ = new char[getBatchBytes(Q80, n, 1)];
    char* wQ4 = new char[getBatchBytes(Q40, n, d)];
    
    if (!xQ || !wQ4) {
        std::cerr << "Memory allocation failed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Take x and w and quantize them to Q40/Q80
    quantizeQ80Row(x, (BlockQ80*)xQ, n, 1, 0);
    //quantizeQ40Row(w, (BlockQ40*)wQ4, n * d, 1, 0);

    // Measure CPU matmul duration
    auto start = std::chrono::high_resolution_clock::now();
    matmul(Q40, Q80, yQ3, xQ, wQ4, n, d, 1, 0);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = end - start;

    // Print the durations
    std::cout << "CPU matmulQ40Q80 - Duration: " << cpuDuration.count() << " ms" << std::endl;

    // Measure Vulkan matmul duration
    start = std::chrono::high_resolution_clock::now();
    matmulVulkan(ctx, LayerElement::KEY, Q40, Q80, yQ2, xQ, wQ4, n, d, 1, 0);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> vulkanDuration = end - start;

    // Print the durations
    std::cout << "Vulkan matmulQ40Q80 - Duration: " << vulkanDuration.count() << " ms" << std::endl;

    // Clean up
    delete[] xQ;
    delete[] wQ4;
}

int main() {
    initQuants();

    void* t = nullptr;

    VulkanContext* vulkan = new VulkanContext((Transformer*)t);

    testMatmulQ80(vulkan);
    //runMatmulQ80Test(vulkan);
    //testQuantizeQ40();

    delete vulkan;

    return EXIT_SUCCESS;
}