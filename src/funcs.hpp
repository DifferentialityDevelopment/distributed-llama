#ifndef FUNCS_HPP
#define FUNCS_HPP

#include "quants.hpp"

#define SPLIT_RANGE_TO_THREADS(varStart, varEnd, rangeStart, rangeEnd, nThreads, threadIndex) \
    const unsigned int rangeLen = (rangeEnd - rangeStart); \
    const unsigned int rangeSlice = rangeLen / nThreads; \
    const unsigned int rangeRest = rangeLen % nThreads; \
    const unsigned int varStart = threadIndex * rangeSlice + (threadIndex < rangeRest ? threadIndex : rangeRest); \
    const unsigned int varEnd = varStart + rangeSlice + (threadIndex < rangeRest ? 1 : 0);

/*
 * Explanation of MatmulThreadInfo members:
 * - pthread_t handler: Thread handler used for threading operations.
 * - float* output: Pointer to the output array where the result of matrix multiplication is stored.
 * - void* input: Pointer to the input data (either float or quantized).
 * - void* weights: Pointer to the weight data (either float or quantized).
 * - int n: Number of columns in the input matrix (dimensionality of input).
 * - int ds: Start index of the output array slice that the thread will compute.
 * - int de: End index (exclusive) of the output array slice that the thread will compute.
 */
struct MatmulThreadInfo {
    pthread_t handler;
    float* output;
    void* input;
    void* weights;
    int n;
    int ds;
    int de;
};

void softmax(float* x, const int size);
float rms(const float* x, const int size);
void rmsnorm(float* o, const float* x, const float ms, const float* weight, const int size, unsigned int nThreads, unsigned int threadIndex);
void matmul(FloatType weightsFloatType, FloatType inputFloatType, float* output, void* input, void* weights, int n, int d, unsigned int nThreads, unsigned int threadIndex);
float dotProduct(const float* a, const float* b, const int size);
void gelu(float* t, int n, unsigned int nThreads, unsigned int threadIndex);
void silu(float* t, int n, unsigned int nThreads, unsigned int threadIndex);
void mul(float* output, float* input, int n, unsigned int nThreads, unsigned int threadIndex);
void mulScalar(float* output, float c, int n, unsigned int nThreads, unsigned int threadIndex);
void add(float* output, float* input, int n, unsigned int nThreads, unsigned int threadIndex);

void matmulF32(MatmulThreadInfo* a);
void matmulF16(MatmulThreadInfo* a);
void matmulQ40(MatmulThreadInfo* a);
void matmulQ80(MatmulThreadInfo* a);
void matmulQ40vQ80(MatmulThreadInfo* a);
void matmulQ80vQ80(MatmulThreadInfo* a);

#endif
