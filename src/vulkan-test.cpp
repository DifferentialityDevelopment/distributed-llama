#include "vulkan.hpp"
#include "utils.hpp"
#include "funcs.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <iostream>
#include <ostream>

// Function to print the contents of an array, useful for debugging
void printArray(const float* arr, int size, const std::string& name) {
    std::cout << name << ": ";
    for (int i = 0; i < size; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

// Function to check if the quantize and dequantize operations are consistent
bool testQuantizeQ80Row() {
    printf("testQuantizeQ80Row\n");
    const int k = QK80;  // Assuming QK40 is the number of elements in a row (should match your setup)
    float input[k] = { 0, 0.19509, 0.382683, 0.55557, 0.707106, 0.831469, 0.923879, 0.980785, 1, 0.980786, 0.92388, 0.831471, 0.707108, 0.555572, 0.382685, 0.195093, 2.53518e-06, -0.195088, -0.382681, -0.555568, -0.707105, -0.831468, -0.923878, -0.980785, -1, -0.980786, -0.923881, -0.831472, -0.70711, -0.555574, -0.382688, -0.195095 };
    BlockQ80 output[1];  // Adjust size if your blocks are structured differently
    float dequantized[k];

    // Print input for verification
    printArray(input, k, "Input");

    // Quantize the input row
    quantizeQ80Row(input, output, k, 1, 0);  // Assuming single-threaded operation for simplicity

    // Dequantize the quantized data
    dequantizeQ80Row(output, dequantized, k, 1, 0);

    // Print dequantized for verification
    printArray(dequantized, k, "Dequantized");

    // Compare original input with dequantized output
    float maxError = 0.0f;
    for (int i = 0; i < k; ++i) {
        float error = fabs(input[i] - dequantized[i]);
        if (error > maxError) {
            maxError = error;
        }
    }

    std::cout << "Maximum error: " << maxError << std::endl;

    // Define an acceptable error threshold
    const float errorThreshold = 0.1f;  // Adjust based on your acceptable error tolerance

    return maxError <= errorThreshold;
}

// Function to check if the quantize and dequantize operations are consistent
bool testQuantizeQ40Row() {
    printf("testQuantizeQ40Row\n");
    const int k = QK40;  // Assuming QK40 is the number of elements in a row (should match your setup)
    float input[k] = { 0, 0.19509, 0.382683, 0.55557, 0.707106, 0.831469, 0.923879, 0.980785, 1, 0.980786, 0.92388, 0.831471, 0.707108, 0.555572, 0.382685, 0.195093, 2.53518e-06, -0.195088, -0.382681, -0.555568, -0.707105, -0.831468, -0.923878, -0.980785, -1, -0.980786, -0.923881, -0.831472, -0.70711, -0.555574, -0.382688, -0.195095 };
    BlockQ40 output[1];  // Adjust size if your blocks are structured differently
    float dequantized[k];

    // Print input for verification
    printArray(input, k, "Input");

    // Quantize the input row
    quantizeQ40Row(input, output, k, 1, 0);  // Assuming single-threaded operation for simplicity

    // Dequantize the quantized data
    dequantizeQ40Row(output, dequantized, k);

    // Print dequantized for verification
    printArray(dequantized, k, "Dequantized");

    // Compare original input with dequantized output
    float maxError = 0.0f;
    for (int i = 0; i < k; ++i) {
        float error = fabs(input[i] - dequantized[i]);
        if (error > maxError) {
            maxError = error;
        }
    }

    std::cout << "Maximum error: " << maxError << std::endl;

    // Define an acceptable error threshold
    const float errorThreshold = 0.1f;  // Adjust based on your acceptable error tolerance

    return maxError <= errorThreshold;
}

void testMatmulQ80(VulkanContext* vulkan) {
    const int n = 512;
    const int d = 256;
    unsigned long long state = 88888888L;
    float x[n]; // input matrix
    float w[n * d];  // weights matrix
    float y[d]; //output matrix
    float yQ0[d];
    float yQ1[d];
    float yQ2[d];
    float yQ3[d];
    int i;
    for (i = 0; i < n; i++) x[i] = randomF32(&state) / 127.0f;
    for (i = 0; i < n * d; i++) w[i] = randomF32(&state) / 127.0f;

    char* xQ = new char[getBatchBytes(Q80, n, 1)];
    char* wQ = new char[getBatchBytes(Q80, n, d)];
    quantizeQ80Row(x, (BlockQ80*)xQ, n, 1, 0);
    quantizeQ80Row(w, (BlockQ80*)wQ, n * d, 1, 0);
    char* wQ4 = new char[getBatchBytes(Q40, n, d)];
    quantizeQ40Row(w, (BlockQ40*)wQ4, n * d, 1, 0);

    matmul(F32, F32, y, x, w, n, d, 1, 0);
    matmulVulkan(vulkan, Q80, F32, yQ0, x, wQ, n, d, 1, 0);
    matmulVulkan(vulkan, Q80, Q80, yQ1, xQ, wQ, n, d, 1, 0);
    matmul(Q40, Q80, yQ3, xQ, wQ4, n, d, 1, 0);
    matmulVulkan(vulkan, Q40, Q80, yQ2, xQ, wQ4, n, d, 1, 0);

    for (i = 0; i < d; i++) {
        float diff = fabs(y[i] - yQ0[i]);
        if (diff > 0.001) {
            printf("❌ matmulQ80() ix=%d %f != %f diff=%f\n", i, y[i], yQ0[i], diff);
            exit(EXIT_FAILURE);
        }
    }
    printf("✅ matmulQ80\n");

    for (i = 0; i < d; i++) {
        float diff = fabs(y[i] - yQ1[i]);
        if (diff > 0.001) {
            printf("❌ matmulQ80vQ80() ix=%d %f != %f diff=%f\n", i, y[i], yQ1[i], diff);
            exit(EXIT_FAILURE);
        }
    }
    printf("✅ matmulQ80vQ80\n");

    
    printArray(yQ3, 16, "CPU Results");
    printArray(yQ2, 16, "Vulkan Results");

    for (i = 0; i < d; i++) {
        float diff = fabs(y[i] - yQ2[i]);
        if (diff > 0.001) {
            printf("❌ matmulQ40Q80() ix=%d %f != %f diff=%f\n", i, y[i], yQ2[i], diff);
            exit(EXIT_FAILURE);
        }
    }
    printf("✅ matmulQ40Q80\n");

    

    delete[] xQ;
    delete[] wQ;
}


int main() {
    initQuants();

    VulkanContext* vulkan = new VulkanContext();

    testMatmulQ80(vulkan);
    return EXIT_SUCCESS;
}