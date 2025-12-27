#ifdef NOCTA_CUDA_ENABLED

#include "nocta/cuda/cuda_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
} while(0)

// ============================================
// BatchNorm Forward
// ============================================

// Compute mean per channel
__global__ void kernel_bn_mean_f32(float* mean, const float* input,
                                    int N, int C, int spatial) {
    int c = blockIdx.x;
    if (c >= C) return;
    
    float sum = 0.0f;
    int count = N * spatial;
    
    for (int n = 0; n < N; n++) {
        for (int s = 0; s < spatial; s++) {
            sum += input[(n * C + c) * spatial + s];
        }
    }
    
    mean[c] = sum / count;
}

// Compute variance per channel
__global__ void kernel_bn_var_f32(float* var, const float* input, const float* mean,
                                   int N, int C, int spatial) {
    int c = blockIdx.x;
    if (c >= C) return;
    
    float sum = 0.0f;
    float m = mean[c];
    int count = N * spatial;
    
    for (int n = 0; n < N; n++) {
        for (int s = 0; s < spatial; s++) {
            float diff = input[(n * C + c) * spatial + s] - m;
            sum += diff * diff;
        }
    }
    
    var[c] = sum / count;
}

// Normalize and scale
__global__ void kernel_bn_normalize_f32(float* output, const float* input,
                                         const float* mean, const float* var,
                                         const float* gamma, const float* beta,
                                         int N, int C, int spatial, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * spatial;
    if (idx >= total) return;
    
    int s = idx % spatial;
    int c = (idx / spatial) % C;
    int n = idx / (C * spatial);
    (void)n; (void)s;
    
    float x = input[idx];
    float m = mean[c];
    float v = var[c];
    float g = gamma[c];
    float b = beta[c];
    
    float x_hat = (x - m) / sqrtf(v + eps);
    output[idx] = g * x_hat + b;
}

// Update running stats
__global__ void kernel_bn_update_stats_f32(float* r_mean, float* r_var,
                                           const float* mean, const float* var,
                                           int C, float momentum) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    
    r_mean[c] = (1.0f - momentum) * r_mean[c] + momentum * mean[c];
    r_var[c] = (1.0f - momentum) * r_var[c] + momentum * var[c];
}

extern "C" void nc_cuda_batchnorm_forward_f32(
    float* output,
    const float* input,
    const float* gamma,
    const float* beta,
    float* running_mean,
    float* running_var,
    float* save_mean,
    float* save_invstd,
    int N, int C, int spatial,
    float momentum, float eps,
    bool training) {
    
    // Static cache for mean/var
    static float* mean_buffer = NULL;
    static float* var_buffer = NULL;
    static size_t mv_buffer_size = 0;
    
    size_t required = C * sizeof(float);
    if (!mean_buffer || mv_buffer_size < required) {
        if (mean_buffer) cudaFree(mean_buffer);
        if (var_buffer) cudaFree(var_buffer);
        CUDA_CHECK(cudaMalloc(&mean_buffer, required));
        CUDA_CHECK(cudaMalloc(&var_buffer, required));
        mv_buffer_size = required;
    }
    float* mean = mean_buffer;
    float* var = var_buffer;
    
    if (training) {
        // Compute batch statistics
        kernel_bn_mean_f32<<<C, 1>>>(mean, input, N, C, spatial);
        CUDA_CHECK(cudaGetLastError());
        
        kernel_bn_var_f32<<<C, 1>>>(var, input, mean, N, C, spatial);
        CUDA_CHECK(cudaGetLastError());
        
        // Save for backward
        if (save_mean) {
            CUDA_CHECK(cudaMemcpy(save_mean, mean, C * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        
        // Update running stats
        if (running_mean && running_var) {
            int blocks_update = (C + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernel_bn_update_stats_f32<<<blocks_update, BLOCK_SIZE>>>(
                running_mean, running_var, mean, var, C, momentum);
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        // Use running statistics
        CUDA_CHECK(cudaMemcpy(mean, running_mean, C * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(var, running_var, C * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    // Save var for backward (passed as save_invstd argument name, but treating as save_var)
    if (save_invstd) {
        CUDA_CHECK(cudaMemcpy(save_invstd, var, C * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    // Normalize
    int total = N * C * spatial;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_bn_normalize_f32<<<blocks, BLOCK_SIZE>>>(output, input, mean, var, gamma, beta,
                                                     N, C, spatial, eps);
    CUDA_CHECK(cudaGetLastError());
    
    // No free (static)
    // CUDA_CHECK(cudaFree(mean));
    // CUDA_CHECK(cudaFree(var));
}

// ============================================
// BatchNorm Backward
// ============================================

__global__ void kernel_bn_backward_f32(
    float* grad_input, float* grad_gamma, float* grad_beta,
    const float* grad_output, const float* input,
    const float* mean, const float* var, const float* gamma,
    int N, int C, int spatial, float eps) {
    
    int c = blockIdx.x;
    if (c >= C) return;
    
    float m = mean[c];
    float v = var[c];
    float inv = 1.0f / sqrtf(v + eps); // Compute invstd
    float g = gamma ? gamma[c] : 1.0f;
    int count = N * spatial;
    
    // Compute grad_gamma and grad_beta
    float dgamma = 0.0f;
    float dbeta = 0.0f;
    
    for (int n = 0; n < N; n++) {
        for (int s = 0; s < spatial; s++) {
            int idx = (n * C + c) * spatial + s;
            float x_hat = (input[idx] - m) * inv;
            dgamma += grad_output[idx] * x_hat;
            dbeta += grad_output[idx];
        }
    }
    
    if (grad_gamma) grad_gamma[c] = dgamma;
    if (grad_beta) grad_beta[c] = dbeta;
    
    // Compute grad_input
    float dx_hat_sum = 0.0f;
    float dx_hat_x_hat_sum = 0.0f;
    
    for (int n = 0; n < N; n++) {
        for (int s = 0; s < spatial; s++) {
            int idx = (n * C + c) * spatial + s;
            float x_hat = (input[idx] - m) * inv;
            float dy = grad_output[idx];
            dx_hat_sum += dy * g;
            dx_hat_x_hat_sum += dy * g * x_hat;
        }
    }
    
    for (int n = 0; n < N; n++) {
        for (int s = 0; s < spatial; s++) {
            int idx = (n * C + c) * spatial + s;
            float x_hat = (input[idx] - m) * inv;
            float dy = grad_output[idx];
            
            float dx_hat = dy * g;
            float dx = inv * (dx_hat - dx_hat_sum / count - x_hat * dx_hat_x_hat_sum / count);
            grad_input[idx] = dx;
        }
    }
}

extern "C" void nc_cuda_batchnorm_backward_f32(
    float* grad_input,
    float* grad_gamma,
    float* grad_beta,
    const float* grad_output,
    const float* input,
    const float* gamma,
    const float* save_mean,
    const float* save_var,
    int N, int C, int spatial,
    float eps) {
    
    kernel_bn_backward_f32<<<C, 1>>>(grad_input, grad_gamma, grad_beta,
                                      grad_output, input,
                                      save_mean, save_var, gamma,
                                      N, C, spatial, eps);
    CUDA_CHECK(cudaGetLastError());
}

#endif // NOCTA_CUDA_ENABLED
