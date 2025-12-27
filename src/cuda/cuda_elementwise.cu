#ifdef NOCTA_CUDA_ENABLED

#include "nocta/cuda/cuda_kernels.h"
#include <cuda_runtime.h>
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
// Fill Kernels
// ============================================

__global__ void kernel_fill_f32(float* data, float value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

__global__ void kernel_fill_f64(double* data, double value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

extern "C" void nc_cuda_fill_f32(float* data, float value, size_t n) {
    if (n == 0) return;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_fill_f32<<<blocks, BLOCK_SIZE>>>(data, value, n);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void nc_cuda_fill_f64(double* data, double value, size_t n) {
    if (n == 0) return;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_fill_f64<<<blocks, BLOCK_SIZE>>>(data, value, n);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================
// Copy Kernels
// ============================================

__global__ void kernel_copy_f32(float* dst, const float* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

__global__ void kernel_copy_f64(double* dst, const double* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

extern "C" void nc_cuda_copy_f32(float* dst, const float* src, size_t n) {
    if (n == 0) return;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_copy_f32<<<blocks, BLOCK_SIZE>>>(dst, src, n);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void nc_cuda_copy_f64(double* dst, const double* src, size_t n) {
    if (n == 0) return;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_copy_f64<<<blocks, BLOCK_SIZE>>>(dst, src, n);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================
// Arithmetic Kernels
// ============================================

__global__ void kernel_add_f32(float* out, const float* a, const float* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void kernel_sub_f32(float* out, const float* a, const float* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void kernel_mul_f32(float* out, const float* a, const float* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void kernel_div_f32(float* out, const float* a, const float* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void kernel_add_scalar_f32(float* out, const float* a, float scalar, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + scalar;
    }
}

__global__ void kernel_mul_scalar_f32(float* out, const float* a, float scalar, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

extern "C" void nc_cuda_add_f32(float* out, const float* a, const float* b, size_t n) {
    if (n == 0) return;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_add_f32<<<blocks, BLOCK_SIZE>>>(out, a, b, n);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void nc_cuda_sub_f32(float* out, const float* a, const float* b, size_t n) {
    if (n == 0) return;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_sub_f32<<<blocks, BLOCK_SIZE>>>(out, a, b, n);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void nc_cuda_mul_f32(float* out, const float* a, const float* b, size_t n) {
    if (n == 0) return;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_mul_f32<<<blocks, BLOCK_SIZE>>>(out, a, b, n);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void nc_cuda_div_f32(float* out, const float* a, const float* b, size_t n) {
    if (n == 0) return;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_div_f32<<<blocks, BLOCK_SIZE>>>(out, a, b, n);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void nc_cuda_add_scalar_f32(float* out, const float* a, float scalar, size_t n) {
    if (n == 0) return;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_add_scalar_f32<<<blocks, BLOCK_SIZE>>>(out, a, scalar, n);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void nc_cuda_mul_scalar_f32(float* out, const float* a, float scalar, size_t n) {
    if (n == 0) return;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_mul_scalar_f32<<<blocks, BLOCK_SIZE>>>(out, a, scalar, n);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================
// Broadcast Kernels
// ============================================

__global__ void kernel_add_broadcast_batch_f32(float* out, const float* a, const float* b, 
                                              int inner_dim, size_t total) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int c = idx % inner_dim;
        out[idx] = a[idx] + b[c];
    }
}

// Adds b (size C) to a (size N*C) where inner dimension is C
extern "C" void nc_cuda_add_broadcast_batch_f32(float* out, const float* a, const float* b, 
                                                size_t total_elements, int inner_dim) {
    if (total_elements == 0) return;
    int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_add_broadcast_batch_f32<<<blocks, BLOCK_SIZE>>>(out, a, b, inner_dim, total_elements);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================
// SGD Update Kernel
// ============================================

__global__ void kernel_sgd_step_f32(float* param, const float* grad, float lr, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        param[idx] -= lr * grad[idx];
    }
}

extern "C" void nc_cuda_sgd_step_f32(float* param, const float* grad, float lr, size_t n) {
    if (n == 0) return;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_sgd_step_f32<<<blocks, BLOCK_SIZE>>>(param, grad, lr, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void kernel_sgd_momentum_f32(float* param, float* velocity, const float* grad, 
                                        float lr, float momentum, float dampening, 
                                        int nesterov, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grad[idx];
        float v = velocity[idx];
        
        // v = momentum * v + (1 - dampening) * g
        v = momentum * v + (1.0f - dampening) * g;
        
        velocity[idx] = v;
        
        float update;
        if (nesterov) {
            update = g + momentum * v;
        } else {
            update = v;
        }
        
        param[idx] -= lr * update;
    }
}

extern "C" void nc_cuda_sgd_momentum_f32(float* param, float* velocity, const float* grad,
                                         float lr, float momentum, float dampening,
                                         int nesterov, size_t n) {
    if (n == 0) return;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_sgd_momentum_f32<<<blocks, BLOCK_SIZE>>>(param, velocity, grad, lr, momentum, dampening, nesterov, n);
    CUDA_CHECK(cudaGetLastError());
}

#endif // NOCTA_CUDA_ENABLED
