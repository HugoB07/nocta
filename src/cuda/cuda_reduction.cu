#ifdef NOCTA_CUDA_ENABLED

#include "nocta/cuda/cuda_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
} while(0)

// ============================================
// Parallel Reduction for Sum
// ============================================

__global__ void kernel_sum_reduce_f32(float* out, const float* in, size_t n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

extern "C" float nc_cuda_sum_f32(const float* data, size_t n) {
    if (n == 0) return 0.0f;
    
    float* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_sum_reduce_f32<<<blocks, BLOCK_SIZE>>>(d_result, data, n);
    CUDA_CHECK(cudaGetLastError());
    
    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));
    
    return result;
}

// ============================================
// Parallel Reduction for Max
// ============================================

__global__ void kernel_max_reduce_f32(float* out, const float* in, size_t n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? in[i] : -FLT_MAX;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        // Atomic max for float (using atomicCAS)
        float old = *out;
        float assumed;
        do {
            assumed = old;
            old = __int_as_float(atomicCAS((int*)out, 
                                            __float_as_int(assumed),
                                            __float_as_int(fmaxf(assumed, sdata[0]))));
        } while (assumed != old);
    }
}

extern "C" float nc_cuda_max_f32(const float* data, size_t n) {
    if (n == 0) return 0.0f;
    
    float* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    float init = -FLT_MAX;
    CUDA_CHECK(cudaMemcpy(d_result, &init, sizeof(float), cudaMemcpyHostToDevice));
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_max_reduce_f32<<<blocks, BLOCK_SIZE>>>(d_result, data, n);
    CUDA_CHECK(cudaGetLastError());
    
    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));
    
    return result;
}

// ============================================
// Argmax
// ============================================

__global__ void kernel_argmax_f32(size_t* out_idx, float* out_val, const float* in, size_t n) {
    __shared__ float sval[BLOCK_SIZE];
    __shared__ size_t sidx[BLOCK_SIZE];
    
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        sval[tid] = in[i];
        sidx[tid] = i;
    } else {
        sval[tid] = -FLT_MAX;
        sidx[tid] = 0;
    }
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sval[tid + s] > sval[tid]) {
                sval[tid] = sval[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        // Atomic compare and swap for argmax
        float old_val = *out_val;
        if (sval[0] > old_val) {
            float assumed;
            do {
                assumed = old_val;
                if (sval[0] <= assumed) break;
                old_val = __int_as_float(atomicCAS((int*)out_val,
                                                    __float_as_int(assumed),
                                                    __float_as_int(sval[0])));
                if (old_val == assumed) {
                    *out_idx = sidx[0];
                }
            } while (assumed != old_val);
        }
    }
}

extern "C" size_t nc_cuda_argmax_f32(const float* data, size_t n) {
    if (n == 0) return 0;
    
    float* d_val;
    size_t* d_idx;
    CUDA_CHECK(cudaMalloc(&d_val, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx, sizeof(size_t)));
    
    float init_val = -FLT_MAX;
    size_t init_idx = 0;
    CUDA_CHECK(cudaMemcpy(d_val, &init_val, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idx, &init_idx, sizeof(size_t), cudaMemcpyHostToDevice));
    
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_argmax_f32<<<blocks, BLOCK_SIZE>>>(d_idx, d_val, data, n);
    CUDA_CHECK(cudaGetLastError());
    
    size_t result;
    CUDA_CHECK(cudaMemcpy(&result, d_idx, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_idx));
    
    return result;
}

// ============================================
// Axis-wise Reductions
// ============================================

__global__ void kernel_sum_axis_f32(float* out, const float* in,
                                    size_t outer, size_t reduce_dim, size_t inner) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t out_size = outer * inner;
    
    if (out_idx >= out_size) return;
    
    size_t o = out_idx / inner;
    size_t i = out_idx % inner;
    
    float sum = 0.0f;
    for (size_t r = 0; r < reduce_dim; r++) {
        sum += in[(o * reduce_dim + r) * inner + i];
    }
    out[out_idx] = sum;
}

__global__ void kernel_max_axis_f32(float* out, const float* in,
                                    size_t outer, size_t reduce_dim, size_t inner) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t out_size = outer * inner;
    
    if (out_idx >= out_size) return;
    
    size_t o = out_idx / inner;
    size_t i = out_idx % inner;
    
    float max_val = -FLT_MAX;
    for (size_t r = 0; r < reduce_dim; r++) {
        float val = in[(o * reduce_dim + r) * inner + i];
        if (val > max_val) max_val = val;
    }
    out[out_idx] = max_val;
}

extern "C" void nc_cuda_sum_axis_f32(float* out, const float* in,
                                      size_t outer, size_t reduce_dim, size_t inner) {
    size_t out_size = outer * inner;
    if (out_size == 0) return;
    
    int blocks = (out_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_sum_axis_f32<<<blocks, BLOCK_SIZE>>>(out, in, outer, reduce_dim, inner);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void nc_cuda_max_axis_f32(float* out, const float* in,
                                      size_t outer, size_t reduce_dim, size_t inner) {
    size_t out_size = outer * inner;
    if (out_size == 0) return;
    
    int blocks = (out_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_max_axis_f32<<<blocks, BLOCK_SIZE>>>(out, in, outer, reduce_dim, inner);
    CUDA_CHECK(cudaGetLastError());
}

#endif // NOCTA_CUDA_ENABLED
