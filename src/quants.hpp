#ifndef QUANTS_HPP
#define QUANTS_HPP

#include <cstdint>
#include <vector>

enum FloatType {
    F32 = 0,
    F16 = 1,
    Q40 = 2,
    Q80 = 3
};

#define QK40 32
#define QK80 32

typedef struct {
    uint16_t d; // delta
    uint8_t qs[QK40 / 2]; // nibbles / quants
} BlockQ40;

typedef struct {
    uint16_t d; // delta
    int8_t  qs[QK80]; // quants
} BlockQ80;

void initQuants();
float* GetF16ToF32();

int getNumbersPerBatch(FloatType type);
long getBatchBytes(FloatType type, int n, int d);
float convertF16ToF32(uint16_t value);

void quantizeQ40Row(float* input, BlockQ40* output, long k, unsigned int nThreads, unsigned int threadIndex);
void dequantizeQ40Row(const BlockQ40* x, float* y, int k);
void quantizeQ80Row(float* input, BlockQ80* output, int k, unsigned int nThreads, unsigned int threadIndex);
void dequantizeQ80Row(const BlockQ80* input, float* output, int k, unsigned int nThreads, unsigned int threadIndex);

#endif
