#ifndef NOCTA_CUDA_KERNELS_H
#define NOCTA_CUDA_KERNELS_H

#include "nocta/core/device.h"
#include <stddef.h>

#ifdef NOCTA_CUDA_ENABLED

#ifdef __cplusplus
extern "C" {
#endif

// ============================================
// Memory Management (cuda_device.cu)
// ============================================
void* nc_cuda_malloc(size_t size);
void nc_cuda_free(void* ptr);
void nc_cuda_memcpy_h2d(void* dst, const void* src, size_t size);
void nc_cuda_memcpy_d2h(void* dst, const void* src, size_t size);
void nc_cuda_memcpy_d2d(void* dst, const void* src, size_t size);
void nc_cuda_memset(void* ptr, int value, size_t size);

// ============================================
// Element-wise Operations (cuda_elementwise.cu)
// ============================================
void nc_cuda_fill_f32(float* data, float value, size_t n);
void nc_cuda_fill_f64(double* data, double value, size_t n);
void nc_cuda_copy_f32(float* dst, const float* src, size_t n);
void nc_cuda_copy_f64(double* dst, const double* src, size_t n);

// Arithmetic
void nc_cuda_add_f32(float* out, const float* a, const float* b, size_t n);
void nc_cuda_sub_f32(float* out, const float* a, const float* b, size_t n);
void nc_cuda_mul_f32(float* out, const float* a, const float* b, size_t n);
void nc_cuda_div_f32(float* out, const float* a, const float* b, size_t n);

void nc_cuda_add_scalar_f32(float* out, const float* a, float scalar, size_t n);
void nc_cuda_mul_scalar_f32(float* out, const float* a, float scalar, size_t n);

// SGD optimizer step: param -= lr * grad
void nc_cuda_sgd_step_f32(float* param, const float* grad, float lr, size_t n);

void nc_cuda_sgd_momentum_f32(float* param, float* velocity, const float* grad,
                              float lr, float momentum, float dampening,
                              int nesterov, size_t n);

// Adds b (size C) to a (size N*C) where inner dimension is C
void nc_cuda_add_broadcast_batch_f32(float* out, const float* a, const float* b, 
                                     size_t total_elements, int inner_dim);

// ============================================
// Activations (cuda_activation.cu)
// ============================================
void nc_cuda_relu_f32(float* out, const float* in, size_t n);
void nc_cuda_relu_backward_f32(float* grad_in, const float* grad_out, const float* input, size_t n);
void nc_cuda_sigmoid_f32(float* out, const float* in, size_t n);
void nc_cuda_tanh_f32(float* out, const float* in, size_t n);
void nc_cuda_softmax_f32(float* out, const float* in, size_t batch, size_t dim);

// ============================================
// MatMul (cuda_matmul.cu) - using cuBLAS
// ============================================
void nc_cuda_matmul_f32(float* C, const float* A, const float* B,
                                   int M, int N, int K,
                                   float alpha, float beta,
                                   int transA, int transB);

void nc_cuda_matmul_f64(double* C, const double* A, const double* B,
                                   int M, int N, int K,
                                   double alpha, double beta,
                                   int transA, int transB);

// ============================================
// Reductions (cuda_reduction.cu)
// ============================================
float nc_cuda_sum_f32(const float* data, size_t n);
float nc_cuda_max_f32(const float* data, size_t n);
size_t nc_cuda_argmax_f32(const float* data, size_t n);

void nc_cuda_sum_axis_f32(float* out, const float* in, 
                          size_t outer, size_t reduce_dim, size_t inner);
void nc_cuda_max_axis_f32(float* out, const float* in,
                          size_t outer, size_t reduce_dim, size_t inner);

// ============================================
// Convolution (cuda_conv.cu)
// ============================================
void nc_cuda_im2col_f32(const float* data_im, int C, int H, int W,
                        int kH, int kW, int pad, int stride,
                        float* data_col);

void nc_cuda_conv2d_forward_f32(
    float* output,           // (N, C_out, H_out, W_out)
    const float* input,      // (N, C_in, H, W)
    const float* weight,     // (C_out, C_in, kH, kW)
    const float* bias,       // (C_out,) or NULL
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride, int padding);

void nc_cuda_col2im_f32(const float* data_col, int C, int H, int W,
                        int kH, int kW, int pad, int stride,
                        float* data_im);

void nc_cuda_conv2d_backward_input_f32(float* grad_input, const float* grad_output, const float* weight,
                                       int N, int C_in, int H, int W, int C_out, int kH, int kW, int stride, int padding);

void nc_cuda_conv2d_backward_weight_f32(float* grad_weight, const float* grad_output, const float* input,
                                        int N, int C_in, int H, int W, int C_out, int kH, int kW, int stride, int padding);

void nc_cuda_conv2d_backward_bias_f32(float* grad_bias, const float* grad_output,
                                      int N, int C_out, int H_out, int W_out);

void nc_cuda_maxpool2d_forward_f32(float* output, const float* input,
                                   int N, int C, int H, int W,
                                   int kH, int kW, int stride);

void nc_cuda_maxpool2d_backward_f32(float* grad_input, const float* grad_output, const float* input, const float* output,
                                    int N, int C, int H, int W, int kH, int kW, int stride);

// ============================================
// BatchNorm (cuda_batchnorm.cu)
// ============================================
void nc_cuda_batchnorm_forward_f32(
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
    bool training);

void nc_cuda_batchnorm_backward_f32(
    float* grad_input,
    float* grad_gamma,
    float* grad_beta,
    const float* grad_output,
    const float* input,
    const float* gamma,
    const float* save_mean,
    const float* save_var,
    int N, int C, int spatial,
    float eps);

// ============================================
// Loss (cuda_loss.cu)
// ============================================
void nc_cuda_cross_entropy_forward_f32(
    float* loss,             // scalar output
    const float* logits,     // (N, C)
    const int64_t* targets,  // (N,)
    int N, int C);

void nc_cuda_cross_entropy_backward_f32(
    float* grad_logits,      // (N, C)
    const float* logits,     // (N, C)
    const int64_t* targets,  // (N,)
    int N, int C);

#ifdef __cplusplus
}
#endif

#endif // NOCTA_CUDA_ENABLED

#endif // NOCTA_CUDA_KERNELS_H
