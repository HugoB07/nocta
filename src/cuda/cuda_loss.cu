#ifdef NOCTA_CUDA_ENABLED

#include "nocta/cuda/cuda_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
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
// Cross-Entropy Loss Forward
// ============================================

__global__ void kernel_cross_entropy_forward_f32(
    float* loss,
    const float* logits,  // (N, C)
    const int64_t* targets, // (N,)
    int N, int C) {
    
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_loss = 0.0f;
    
    if (i < N) {
        const float* row = logits + i * C;
        int target = (int)targets[i];
        
        // Find max for numerical stability
        float max_val = row[0];
        for (int c = 1; c < C; c++) {
            if (row[c] > max_val) max_val = row[c];
        }
        
        // Compute log-sum-exp
        float sum_exp = 0.0f;
        for (int c = 0; c < C; c++) {
            sum_exp += expf(row[c] - max_val);
        }
        float log_sum_exp = max_val + logf(sum_exp + 1e-10f);
        
        // Loss = -log(softmax[target]) = log_sum_exp - logits[target]
        local_loss = log_sum_exp - row[target];
    }
    
    sdata[tid] = local_loss;
    __syncthreads();
    
    // Reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(loss, sdata[0]);
    }
}

__global__ void kernel_scale_f32(float* ptr, float scale) {
    if (threadIdx.x == 0) *ptr *= scale;
}

extern "C" void nc_cuda_cross_entropy_forward_f32(
    float* loss,
    const float* logits,
    const int64_t* targets,
    int N, int C) {
    
    CUDA_CHECK(cudaMemset(loss, 0, sizeof(float)));
    
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_cross_entropy_forward_f32<<<blocks, BLOCK_SIZE>>>(loss, logits, targets, N, C);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA ERROR Launch: %s\n", cudaGetErrorString(err));
    
    // Divide by N on Device to avoid Sync
    if (N > 0) {
        kernel_scale_f32<<<1, 1>>>(loss, 1.0f / (float)N);
        err = cudaGetLastError();
        if (err != cudaSuccess) printf("CUDA ERROR Scale: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize(); // Force sync to print kernel output
}

// ============================================
// Cross-Entropy Loss Backward
// ============================================

__global__ void kernel_cross_entropy_backward_f32(
    float* grad_logits,
    const float* logits,
    const int64_t* targets,
    int N, int C) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx >= total) return;
    
    int n = idx / C;
    int c = idx % C;
    
    const float* row = logits + n * C;
    int target = (int)targets[n];
    
    // Compute softmax
    float max_val = row[0];
    for (int j = 1; j < C; j++) {
        if (row[j] > max_val) max_val = row[j];
    }
    
    float sum_exp = 0.0f;
    for (int j = 0; j < C; j++) {
        sum_exp += expf(row[j] - max_val);
    }
    
    float softmax = expf(row[c] - max_val) / (sum_exp + 1e-10f);
    
    // Gradient = softmax - 1{c == target}
    float grad = softmax - (c == target ? 1.0f : 0.0f);
    grad_logits[idx] = grad / N;  // Average over batch
}

extern "C" void nc_cuda_cross_entropy_backward_f32(
    float* grad_logits,
    const float* logits,
    const int64_t* targets,
    int N, int C) {
    
    int total = N * C;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_cross_entropy_backward_f32<<<blocks, BLOCK_SIZE>>>(grad_logits, logits, targets, N, C);
    CUDA_CHECK(cudaGetLastError());
}

#endif // NOCTA_CUDA_ENABLED
