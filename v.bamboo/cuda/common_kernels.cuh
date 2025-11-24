#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

// Lightweight CUDA error check for debugging kernels
#define CUDA_CHECK(err)                                                                                \
    do {                                                                                               \
        cudaError_t _err = (err);                                                                      \
        if (_err != cudaSuccess) {                                                                     \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(_err), __FILE__, __LINE__);          \
        }                                                                                              \
    } while (0)

// Device helpers
__device__ __forceinline__ float clipf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

__device__ __forceinline__ float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}

__device__ __forceinline__ float fast_gelu(float x) {
    // tanh-based approximation
    const float k0 = 0.7978845608028654f;  // sqrt(2/pi)
    const float k1 = 0.044715f;
    float y = k0 * (x + k1 * x * x * x);
    return 0.5f * x * (1.f + tanhf(y));
}

__device__ __forceinline__ float rms_norm_scale(const float* x, int stride, int n, float eps) {
    float acc = 0.f;
    for (int i = 0; i < n; ++i) {
        float v = x[i * stride];
        acc += v * v;
    }
    acc = acc / static_cast<float>(n);
    return rsqrtf(acc + eps);
}

// Simple block-wide reduction for sum; assumes power-of-two blockDim.x
template <unsigned int blockSize>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    if (blockSize >= 64) val += __shfl_down_sync(0xffffffff, val, 32);
    if (blockSize >= 32) val += __shfl_down_sync(0xffffffff, val, 16);
    if (blockSize >= 16) val += __shfl_down_sync(0xffffffff, val, 8);
    if (blockSize >= 8)  val += __shfl_down_sync(0xffffffff, val, 4);
    if (blockSize >= 4)  val += __shfl_down_sync(0xffffffff, val, 2);
    if (blockSize >= 2)  val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

template <unsigned int blockSize>
__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];  // supports up to 1024 threads
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warp_reduce_sum<blockSize>(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : 0.f;
    if (wid == 0) val = warp_reduce_sum<blockSize>(val);
    return val;
}

// Utility for grid-stride loops
#define GRID_STRIDE_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

