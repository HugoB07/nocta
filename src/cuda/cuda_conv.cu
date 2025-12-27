#ifdef NOCTA_CUDA_ENABLED

#include "nocta/cuda/cuda_kernels.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include "nocta/core/device.h"

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
} while(0)

// Forward declaration for cuBLAS handle from cuda_matmul.cu
extern "C" {
    extern cublasHandle_t g_cublas_handle;
    void ensure_cublas(void);
}

// ============================================
// Im2Col Kernel
// ============================================

__global__ void kernel_im2col_f32(const float* data_im, 
                                   int C, int H, int W,
                                   int kH, int kW, int pad, int stride,
                                   float* data_col,
                                   int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * kH * kW * H_out * W_out;
    
    if (idx >= total) return;
    
    // Decode index
    int w_out = idx % W_out; idx /= W_out;
    int h_out = idx % H_out; idx /= H_out;
    int kw = idx % kW; idx /= kW;
    int kh = idx % kH; idx /= kH;
    int c = idx;
    
    int h_in = h_out * stride + kh - pad;
    int w_in = w_out * stride + kw - pad;
    
    float val = 0.0f;
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        val = data_im[(c * H + h_in) * W + w_in];
    }
    
    // Output layout: (C*kH*kW) x (H_out*W_out)
    int col_row = (c * kH + kh) * kW + kw;
    int col_col = h_out * W_out + w_out;
    data_col[col_row * (H_out * W_out) + col_col] = val;
}

extern "C" void nc_cuda_im2col_f32(const float* data_im, int C, int H, int W,
                                   int kH, int kW, int pad, int stride,
                                   float* data_col) {
    int H_out = (H + 2 * pad - kH) / stride + 1;
    int W_out = (W + 2 * pad - kW) / stride + 1;
    int total = C * kH * kW * H_out * W_out;
    
    if (total == 0) return;
    
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_im2col_f32<<<blocks, BLOCK_SIZE>>>(data_im, C, H, W, kH, kW, pad, stride,
                                               data_col, H_out, W_out);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================
// Col2Im Kernel
// ============================================

__global__ void kernel_col2im_f32(const float* data_col,
                                  int C, int H, int W,
                                  int kH, int kW, int pad, int stride,
                                  float* data_im,
                                  int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * H * W; // Output size (input image size)
    
    if (idx >= total) return;
    
    // Decode index -> (c, h, w) in image
    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    int c = tmp / H;
    
    // We need to sum contributions from all columns that include this pixel
    // h_in = h_out * stride + kh - pad  =>  h_out * stride = h_in - kh + pad
    // We iterate over possible kh and kw
    
    float val = 0.0f;
    
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            // Check if this (kh, kw) could have come from a valid h_out, w_out
            int h_in_shifted = h - kh + pad;
            int w_in_shifted = w - kw + pad;
            
            if (h_in_shifted % stride == 0 && w_in_shifted % stride == 0) {
                int h_out = h_in_shifted / stride;
                int w_out = w_in_shifted / stride;
                
                if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out) {
                    // This output pixel (h_out, w_out) contributed
                    // Col index: row=(c*kH+kh)*kW+kw, col=h_out*W_out+w_out
                    int col_row = (c * kH + kh) * kW + kw;
                    int col_col = h_out * W_out + w_out;
                    
                    val += data_col[col_row * (H_out * W_out) + col_col];
                }
            }
        }
    }
    
    data_im[idx] = val;
}

extern "C" void nc_cuda_col2im_f32(const float* data_col, int C, int H, int W,
                                   int kH, int kW, int pad, int stride,
                                   float* data_im) {
    int H_out = (H + 2 * pad - kH) / stride + 1;
    int W_out = (W + 2 * pad - kW) / stride + 1;
    int total = C * H * W;
    
    if (total == 0) return;
    
    // Initialize image with 0
    nc_cuda_memset((void*)data_im, 0, total * sizeof(float));
    
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_col2im_f32<<<blocks, BLOCK_SIZE>>>(data_col, C, H, W, kH, kW, pad, stride,
                                              data_im, H_out, W_out);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================
// Conv2D Forward (Im2Col + cuBLAS GEMM)
// ============================================

extern "C" void nc_cuda_conv2d_forward_f32(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride, int padding) {
    
    int H_out = (H + 2 * padding - kH) / stride + 1;
    int W_out = (W + 2 * padding - kW) / stride + 1;
    int K = C_in * kH * kW;
    int P = H_out * W_out;
    
    // Allocate workspace for im2col (CACHE)
    static float* col_buffer = NULL;
    static size_t col_buffer_size = 0;
    
    size_t required = K * P * sizeof(float);
    if (!col_buffer || col_buffer_size < required) {
        if (col_buffer) cudaFree(col_buffer);
        CUDA_CHECK(cudaMalloc(&col_buffer, required));
        col_buffer_size = required;
    }
    float* col = col_buffer;
    
    ensure_cublas();
    
    float alpha = 1.0f, beta = 0.0f;
    
    for (int n = 0; n < N; n++) {
        const float* input_n = input + n * C_in * H * W;
        float* output_n = output + n * C_out * H_out * W_out;
        
        // Im2col for this sample
        nc_cuda_im2col_f32(input_n, C_in, H, W, kH, kW, padding, stride, col);
        
        // GEMM: output_n = weight @ col
        // weight: (C_out, K), col: (K, P), output_n: (C_out, P)
        cublasSgemm(g_cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    P, C_out, K,
                    &alpha,
                    col, P,
                    weight, K,
                    &beta,
                    output_n, P);
        
        // Add bias if present
        if (bias) {
            // Launch kernel to add bias
            int total = C_out * P;
            int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            // Simple bias add kernel inline here
            // We'll need to define it
        }
    }
}

extern "C" void nc_cuda_conv2d_backward_input_f32(
    float* grad_input,
    const float* grad_output,
    const float* weight,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride, int padding) {

    int H_out = (H + 2 * padding - kH) / stride + 1;
    int W_out = (W + 2 * padding - kW) / stride + 1;
    int K = C_in * kH * kW; // kernel size flattened
    int P = H_out * W_out;  // output spatial size
    
    // grad_output: (N, C_out, P)
    // weight: (C_out, K)
    // grad_input: (N, C_in, H, W)
    
    // We need dCol = W^T * dY
    // Then dX = col2im(dCol)
    
    // Allocate col buffer (CACHE)
    static float* col_buffer = NULL;
    static size_t col_buffer_size = 0;
    
    size_t required = K * P * sizeof(float);
    if (!col_buffer || col_buffer_size < required) {
        if (col_buffer) cudaFree(col_buffer);
        CUDA_CHECK(cudaMalloc(&col_buffer, required));
        col_buffer_size = required;
    }
    float* col = col_buffer;
    
    ensure_cublas();
    
    float alpha = 1.0f, beta = 0.0f;
    
    for (int n = 0; n < N; n++) {
        // dCol = W^T * dY[n]
        // Row-major dims: W^T (K, C_out), dY (C_out, P) -> dCol (K, P)
        // Col-major logic: dCol^T = dY^T * W
        // dY^T (P, C_out), W (C_out, K) -> (P, K)
        // gemm(N, N, m=P, n=K, k=C_out, dY[n], W, col)
        
        const float* dY_n = grad_output + n * C_out * P;
        // W is constant
        // col is reused
        
        // C = AB (row-major) <=> C^T = B^T A^T (col-major)
        // We want C = W^T * dY
        // C^T = dY^T * W
        // dY is (C_out, P). In memory (col-major view): (P, C_out)
        // W is (C_out, K). In memory (col-major view): (K, C_out)
        // We want C^T (P, K).
        // (P, C_out) * (C_out, K) -> Need W transposed in col-major view! (K, C_out) -> (C_out, K)
        // So B = W^T (transa=T)
        // cublasSgemm(handle, N, T, P, K, C_out, ..., dY, ..., W, ..., col, ...)
        
        cublasSgemm(g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    P, K, C_out,
                    &alpha,
                    dY_n, P,          // A = dY^T (P x C_out)
                    weight, K,        // B = W (K x C_out) -> Transposed -> (C_out x K)
                    &beta,
                    col, P);          // C = col (P x K)
        
        // col2im
        float* dX_n = grad_input + n * C_in * H * W;
        nc_cuda_col2im_f32(col, C_in, H, W, kH, kW, padding, stride, dX_n);
    }
    
    // No free (static buffer)
    // CUDA_CHECK(cudaFree(col));
}

extern "C" void nc_cuda_conv2d_backward_weight_f32(
    float* grad_weight,
    const float* grad_output,
    const float* input,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride, int padding) {

    int H_out = (H + 2 * padding - kH) / stride + 1;
    int W_out = (W + 2 * padding - kW) / stride + 1;
    int K = C_in * kH * kW;
    int P = H_out * W_out;
    
    // grad_weight: (C_out, K) - accumulate here! (init with 0 outside or assume accumulation?)
    // Usually backward_weight accumulates. We assume grad_weight is initialized (or handled by caller).
    // But typically we compute proper gradient for this batch.
    // Let's assume we accumulate into grad_weight (beta=1.0).
    
    // grad_weight += dL/dY * col^T
    // dL/dY: (N, C_out, P)
    // col: (K, P) (from im2col(X))
    
    // Allocate col buffer (CACHE)
    static float* col_buffer = NULL;
    static size_t col_buffer_size = 0; // Distinct from other functions
    
    size_t required = K * P * sizeof(float);
    if (!col_buffer || col_buffer_size < required) {
        if (col_buffer) cudaFree(col_buffer);
        CUDA_CHECK(cudaMalloc(&col_buffer, required));
        col_buffer_size = required;
    }
    float* col = col_buffer;
    
    ensure_cublas();
    
    float alpha = 1.0f, beta = 1.0f; // Accumulate
    
    for (int n = 0; n < N; n++) {
        // im2col(X[n])
        const float* X_n = input + n * C_in * H * W;
        nc_cuda_im2col_f32(X_n, C_in, H, W, kH, kW, padding, stride, col);
        
        // dW += dY[n] * col^T
        // Row-major: (C_out, P) * (P, K) -> (C_out, K)
        // Col-major logic: dW^T = col * dY[n]^T
        // col memory: (P, K)
        // dY[n] memory: (dY[n]^T) (P, C_out)
        // (P, K)^T * (P, C_out) -> (K, P) * (P, C_out) -> (K, C_out) -> dW^T
        // So A = col (transA=T), B = dY[n] (transB=N)
        
        const float* dY_n = grad_output + n * C_out * P;
        
        cublasSgemm(g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    K, C_out, P,
                    &alpha,
                    col, P,           // A = col (P x K) -> Transpoed -> (K x P)
                    dY_n, P,          // B = dY^T (P x C_out)
                    &beta,
                    grad_weight, K);  // C = dW^T (K x C_out)
    }
    
    // No free
    // CUDA_CHECK(cudaFree(col));
}

// ============================================
// Conv2D Backward Bias
// ============================================

__global__ void kernel_conv2d_backward_bias_f32(float* grad_bias, const float* grad_output,
                                                int N, int C, int spatial) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    
    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int s = 0; s < spatial; s++) {
            sum += grad_output[(n * C + c) * spatial + s];
        }
    }
    grad_bias[c] = sum;
}

extern "C" void nc_cuda_conv2d_backward_bias_f32(float* grad_bias, const float* grad_output,
                                                 int N, int C_out, int H_out, int W_out) {
    int spatial = H_out * W_out;
    int blocks = (C_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    kernel_conv2d_backward_bias_f32<<<blocks, BLOCK_SIZE>>>(grad_bias, grad_output, N, C_out, spatial);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================
// MaxPool Forward
// ============================================

__global__ void kernel_maxpool_forward_f32(float* output, const float* input,
                                           int N, int C, int H, int W,
                                           int kH, int kW, int stride,
                                           int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    
    if (idx >= total) return;
    
    // Decode index -> (n, c, h_out, w_out)
    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;
    
    int h_start = h_out * stride;
    int w_start = w_out * stride;
    int h_end = min(h_start + kH, H);
    int w_end = min(w_start + kW, W);
    
    float max_val = -1e37f;
    
    for (int h = h_start; h < h_end; h++) {
        for (int w = w_start; w < w_end; w++) {
            float val = input[((n * C + c) * H + h) * W + w];
            if (val > max_val) max_val = val;
        }
    }
    
    output[idx] = max_val;
}

extern "C" void nc_cuda_maxpool2d_forward_f32(float* output, const float* input,
                                              int N, int C, int H, int W,
                                              int kH, int kW, int stride) {
    int H_out = (H - kH) / stride + 1;
    int W_out = (W - kW) / stride + 1;
    int total = N * C * H_out * W_out;
    
    if (total == 0) return;
    
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_maxpool_forward_f32<<<blocks, BLOCK_SIZE>>>(output, input, N, C, H, W, kH, kW, stride, H_out, W_out);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================
// MaxPool Backward
// ============================================

__global__ void kernel_maxpool_backward_f32(float* grad_input, const float* grad_output, const float* input,
                                            int N, int C, int H, int W,
                                            int kH, int kW, int stride,
                                            int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    
    if (idx >= total) return;
    
    // Decode index -> (n, c, h_out, w_out)
    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;
    
    // Gradient to propagate
    float g = grad_output[idx];
    
    // Find absolute max index in input window re-computing it
    int h_start = h_out * stride;
    int w_start = w_out * stride;
    int h_end = min(h_start + kH, H);
    int w_end = min(w_start + kW, W);
    
    float max_val = -1e37f; // -FLT_MAX
    int max_h = -1;
    int max_w = -1;
    
    for (int h = h_start; h < h_end; h++) {
        for (int w = w_start; w < w_end; w++) {
            float val = input[((n * C + c) * H + h) * W + w];
            if (val > max_val) {
                max_val = val;
                max_h = h;
                max_w = w;
            }
        }
    }
    
    // Propagate gradient to max index
    if (max_h != -1) {
        // Atomic add because multiple output windows might overlap (if stride < kernel)
        // For LeNet (stride=kernel=2), no overlap, but let's be safe or just standard assign?
        // Standard MaxPool usually doesn't overlap for assignment if disjoint.
        // But if overlap, atomic is needed.
        // We use atomicAdd.
        atomicAdd(&grad_input[((n * C + c) * H + max_h) * W + max_w], g);
    }
}

extern "C" void nc_cuda_maxpool2d_backward_f32(float* grad_input, const float* grad_output, const float* input, const float* output,
                                               int N, int C, int H, int W, int kH, int kW, int stride) {
    (void)output; // Unused, we recompute max from input
    
    int H_out = (H - kH) / stride + 1;
    int W_out = (W - kW) / stride + 1;
    
    int total = N * C * H_out * W_out;
    if (total == 0) return;
    
    // Initialize grad_input to 0
    nc_cuda_memset(grad_input, 0, N * C * H * W * sizeof(float));
    
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_maxpool_backward_f32<<<blocks, BLOCK_SIZE>>>(grad_input, grad_output, input,
                                                        N, C, H, W, kH, kW, stride,
                                                        H_out, W_out);
    CUDA_CHECK(cudaGetLastError());
}

// Bias add kernel
__global__ void kernel_bias_add_f32(float* output, const float* bias, 
                                     int C, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * spatial;
    if (idx >= total) return;
    
    int c = idx / spatial;
    output[idx] += bias[c];
}

#endif // NOCTA_CUDA_ENABLED
