#ifdef NOCTA_CUDA_ENABLED

#include "nocta/cuda/cuda_kernels.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
} while(0)

// ============================================
// ReLU
// ============================================

__global__ void kernel_relu_f32(float* out, const float* in, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] > 0.0f ? in[idx] : 0.0f;
    }
}

__global__ void kernel_relu_backward_f32(float* grad_in, const float* grad_out, 
                                          const float* input, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] = input[idx] > 0.0f ? grad_out[idx] : 0.0f;
    }
}

extern "C" void nc_cuda_relu_f32(float* out, const float* in, size_t n) {
    if (n == 0) return;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_relu_f32<<<blocks, BLOCK_SIZE>>>(out, in, n);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void nc_cuda_relu_backward_f32(float* grad_in, const float* grad_out, 
                                           const float* input, size_t n) {
    if (n == 0) return;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_relu_backward_f32<<<blocks, BLOCK_SIZE>>>(grad_in, grad_out, input, n);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================
// Sigmoid
// ============================================

__global__ void kernel_sigmoid_f32(float* out, const float* in, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        // Numerically stable sigmoid
        if (x >= 0) {
            out[idx] = 1.0f / (1.0f + expf(-x));
        } else {
            float ex = expf(x);
            out[idx] = ex / (1.0f + ex);
        }
    }
}

extern "C" void nc_cuda_sigmoid_f32(float* out, const float* in, size_t n) {
    if (n == 0) return;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_sigmoid_f32<<<blocks, BLOCK_SIZE>>>(out, in, n);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================
// Tanh
// ============================================

__global__ void kernel_tanh_f32(float* out, const float* in, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanhf(in[idx]);
    }
}

extern "C" void nc_cuda_tanh_f32(float* out, const float* in, size_t n) {
    if (n == 0) return;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_tanh_f32<<<blocks, BLOCK_SIZE>>>(out, in, n);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================
// Softmax (row-wise, numerically stable)
// ============================================

__global__ void kernel_softmax_f32(float* out, const float* in, size_t batch, size_t dim) {
    size_t row = blockIdx.x;
    if (row >= batch) return;
    
    const float* in_row = in + row * dim;
    float* out_row = out + row * dim;
    
    // Find max for numerical stability
    float max_val = in_row[0];
    for (size_t i = 1; i < dim; i++) {
        if (in_row[i] > max_val) max_val = in_row[i];
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        out_row[i] = expf(in_row[i] - max_val);
        sum += out_row[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (size_t i = 0; i < dim; i++) {
        out_row[i] *= inv_sum;
    }
}

extern "C" void nc_cuda_softmax_f32(float* out, const float* in, size_t batch, size_t dim) {
    if (batch == 0 || dim == 0) return;
    // One block per row
    kernel_softmax_f32<<<batch, 1>>>(out, in, batch, dim);
    CUDA_CHECK(cudaGetLastError());
}

#endif // NOCTA_CUDA_ENABLED
