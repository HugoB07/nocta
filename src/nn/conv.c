#include "nocta/nn/conv.h"
#include "nocta/core/memory.h"
#include "nocta/autograd/node.h"
#include "nocta/autograd/backward.h"
#include "nocta/ops/matmul.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef NOCTA_OPENMP_ENABLED
#include <omp.h>
#endif

#ifdef NOCTA_CUDA_ENABLED
#include "nocta/cuda/cuda_kernels.h"
#endif

// Helper to check if tensor is on CUDA
static inline bool tensor_on_cuda(nc_tensor* t) {
#ifdef NOCTA_CUDA_ENABLED
    return t && t->storage && t->storage->device == NC_DEVICE_CUDA;
#else
    (void)t;
    return false;
#endif
}

// ============================================
// Conv2D Extra Data
// ============================================

typedef struct {
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    bool has_bias;
} nc_conv2d_data;

// ============================================
// Im2Col Implementation
// ============================================

static void im2col(const float* data_im, int channels, int height, int width,
                   int kernel_h, int kernel_w, int pad, int stride,
                   float* data_col, int col_stride) {
    int height_col = (height + 2 * pad - kernel_h) / stride + 1;
    int width_col = (width + 2 * pad - kernel_w) / stride + 1;
    int channels_col = channels * kernel_h * kernel_w;
    
    int c;
    (void)c;
    // No OpenMP here, we parallelize over batch
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / kernel_h / kernel_w;
        
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                // We use col_stride for the channel dimension
                int col_index = c * col_stride + h * width_col + w;
                
                im_row -= pad;
                im_col -= pad;
                
                if (im_row >= 0 && im_col >= 0 && im_row < height && im_col < width) {
                    data_col[col_index] = data_im[(c_im * height + im_row) * width + im_col];
                } else {
                    data_col[col_index] = 0;
                }
            }
        }
    }
}

static void im2col_f64(const double* data_im, int channels, int height, int width,
                       int kernel_h, int kernel_w, int pad, int stride,
                       double* data_col, int col_stride) {
    int height_col = (height + 2 * pad - kernel_h) / stride + 1;
    int width_col = (width + 2 * pad - kernel_w) / stride + 1;
    int channels_col = channels * kernel_h * kernel_w;
    
    int c;
    (void)c;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / kernel_h / kernel_w;
        
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = c * col_stride + h * width_col + w;
                
                im_row -= pad;
                im_col -= pad;
                
                if (im_row >= 0 && im_col >= 0 && im_row < height && im_col < width) {
                    data_col[col_index] = data_im[(c_im * height + im_row) * width + im_col];
                } else {
                    data_col[col_index] = 0;
                }
            }
        }
    }
}

// ============================================
// Conv2D Backward
// ============================================

nc_tensor** nc_backward_conv2d(nc_tensor* grad, nc_tensor** saved, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(3, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    nc_tensor* input = saved[0];    // (N, C_in, H, W)
    nc_tensor* weight = saved[1];   // (C_out, C_in, kH, kW)
    nc_tensor* bias = saved[2];     // (C_out,) or NULL
    nc_tensor* stride_t = saved[3]; // scalar
    nc_tensor* padding_t = saved[4]; // scalar
    
    size_t stride = (size_t)nc_tensor_get_flat(stride_t, 0);
    size_t padding = (size_t)nc_tensor_get_flat(padding_t, 0);
    
    size_t N = input->shape[0];
    size_t C_in = input->shape[1];
    size_t H = input->shape[2];
    size_t W = input->shape[3];
    size_t C_out = weight->shape[0];
    size_t kH = weight->shape[2];
    size_t kW = weight->shape[3];
    
    size_t H_out = (H + 2*padding - kH) / stride + 1;
    size_t W_out = (W + 2*padding - kW) / stride + 1;
    
    // Gradient w.r.t input
    grads[0] = nc_tensor_zeros(input->shape, input->ndim, input->dtype);
    
    // Gradient w.r.t weight
    grads[1] = nc_tensor_zeros(weight->shape, weight->ndim, weight->dtype);
    
    // Gradient w.r.t bias (if present)
    if (bias) {
        grads[2] = nc_tensor_zeros(bias->shape, bias->ndim, bias->dtype);
    }
    
    // CUDA Dispatch
#ifdef NOCTA_CUDA_ENABLED
    if (grad->storage && grad->storage->device == NC_DEVICE_CUDA) {
        // Prepare gradients on GPU
        nc_tensor_to_device(grads[0], NC_DEVICE_CUDA);
        nc_tensor_to_device(grads[1], NC_DEVICE_CUDA);
        if (grads[2]) nc_tensor_to_device(grads[2], NC_DEVICE_CUDA);
        
        // 1. dL/dX
        if (input->storage && input->storage->cuda_data && grads[0]->storage->cuda_data) {
            nc_cuda_conv2d_backward_input_f32(
                (float*)grads[0]->storage->cuda_data,
                (const float*)grad->storage->cuda_data,
                (const float*)weight->storage->cuda_data,
                N, C_in, H, W, C_out, kH, kW, stride, padding);
        }
        
        // 2. dL/dW
        if (grads[1]->storage->cuda_data) {
            nc_cuda_conv2d_backward_weight_f32(
                (float*)grads[1]->storage->cuda_data,
                (const float*)grad->storage->cuda_data,
                (const float*)input->storage->cuda_data,
                N, C_in, H, W, C_out, kH, kW, stride, padding);
        }
            
        // 3. dL/db
        if (grads[2] && grads[2]->storage->cuda_data) {
            nc_cuda_conv2d_backward_bias_f32(
                (float*)grads[2]->storage->cuda_data,
                (const float*)grad->storage->cuda_data,
                N, C_out, H_out, W_out);
        }
        
        return grads;
    }
#endif
    
    if (bias) {
        // CPU Bias Backward
        // dL/db = sum over (N, H_out, W_out) of grad
        // Parallelize over channels
        int c;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for
        #endif
        for (c = 0; c < (int)C_out; c++) {
            double sum = 0.0;
            for (size_t b = 0; b < N; b++) {
                for (size_t h = 0; h < H_out; h++) {
                    for (size_t w = 0; w < W_out; w++) {
                        sum += nc_tensor_get4(grad, b, c, h, w);
                    }
                }
            }
            nc_tensor_set1(grads[2], c, sum);
        }
    }
    
    // Ensure contiguous for raw pointer access
    nc_tensor* grad_c = nc_tensor_contiguous(grad);
    nc_tensor* input_c = nc_tensor_contiguous(input);
    nc_tensor* weight_c = nc_tensor_contiguous(weight);
    nc_tensor* dx_c = grads[0]; // zeros, contiguous
    nc_tensor* dw_c = grads[1]; // zeros, contiguous
    
    if (grad->dtype == NC_F32) {
        float* gp = nc_tensor_data_f32(grad_c);
        float* ip = nc_tensor_data_f32(input_c);
        float* wp = nc_tensor_data_f32(weight_c);
        float* dxp = nc_tensor_data_f32(dx_c);
        float* dwp = nc_tensor_data_f32(dw_c);
        
        // 1. Compute dL/dX (Input Gradient)
        // Parallelize over Batch (N) - No atomics needed
        int b_idx;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for
        #endif
        for (b_idx = 0; b_idx < (int)N; b_idx++) {
            for (size_t c_out = 0; c_out < C_out; c_out++) {
                for (size_t h_out = 0; h_out < H_out; h_out++) {
                    for (size_t w_out = 0; w_out < W_out; w_out++) {
                        float grad_val = gp[((b_idx * C_out + c_out) * H_out + h_out) * W_out + w_out];
                        
                        for (size_t c_in = 0; c_in < C_in; c_in++) {
                            for (size_t kh = 0; kh < kH; kh++) {
                                for (size_t kw = 0; kw < kW; kw++) {
                                    int h_in = (int)(h_out * stride + kh) - (int)padding;
                                    int w_in = (int)(w_out * stride + kw) - (int)padding;
                                    
                                    if (h_in >= 0 && h_in < (int)H && 
                                        w_in >= 0 && w_in < (int)W) {
                                        
                                        float w = wp[((c_out * C_in + c_in) * kH + kh) * kW + kw];
                                        size_t x_idx = ((b_idx * C_in + c_in) * H + h_in) * W + w_in;
                                        
                                        dxp[x_idx] += grad_val * w;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 2. Compute dL/dW (Weight Gradient)
        // Manual collapse of 4 loops: C_out, C_in, kH, kW
        size_t total_params = C_out * C_in * kH * kW;
        
        int p_idx;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for
        #endif
        for (p_idx = 0; p_idx < (int)total_params; p_idx++) {
            // Decode index
            size_t tmp = p_idx;
            size_t kw = tmp % kW; tmp /= kW;
            size_t kh = tmp % kH; tmp /= kH;
            size_t c_in = tmp % C_in; tmp /= C_in;
            size_t c_out = tmp;
            
            double sum = 0.0;
            
            // Sum over batch and spatial locations
            for (size_t b = 0; b < N; b++) {
                for (size_t h_out = 0; h_out < H_out; h_out++) {
                    for (size_t w_out = 0; w_out < W_out; w_out++) {
                        
                        int h_in = (int)(h_out * stride + kh) - (int)padding;
                        int w_in = (int)(w_out * stride + kw) - (int)padding;
                        
                        if (h_in >= 0 && h_in < (int)H && 
                            w_in >= 0 && w_in < (int)W) {
                            
                            float grad_val = gp[((b * C_out + c_out) * H_out + h_out) * W_out + w_out];
                            float inp = ip[((b * C_in + c_in) * H + h_in) * W + w_in];
                            sum += grad_val * inp;
                        }
                    }
                }
            }
            
            dwp[p_idx] = (float)sum;
        }
        
    } else {
        // F64 Fallback
        double* gp = nc_tensor_data_f64(grad_c);
        double* ip = nc_tensor_data_f64(input_c);
        double* wp = nc_tensor_data_f64(weight_c);
        double* dxp = nc_tensor_data_f64(dx_c);
        double* dwp = nc_tensor_data_f64(dw_c);
        
        // 1. dL/dX
        int b_idx;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for
        #endif
        for (b_idx = 0; b_idx < (int)N; b_idx++) {
            for (size_t c_out = 0; c_out < C_out; c_out++) {
                for (size_t h_out = 0; h_out < H_out; h_out++) {
                    for (size_t w_out = 0; w_out < W_out; w_out++) {
                        double grad_val = gp[((b_idx * C_out + c_out) * H_out + h_out) * W_out + w_out];
                        for (size_t c_in = 0; c_in < C_in; c_in++) {
                            for (size_t kh = 0; kh < kH; kh++) {
                                for (size_t kw = 0; kw < kW; kw++) {
                                    int h_in = (int)(h_out * stride + kh) - (int)padding;
                                    int w_in = (int)(w_out * stride + kw) - (int)padding;
                                    if (h_in >= 0 && h_in < (int)H && w_in >= 0 && w_in < (int)W) {
                                        double w = wp[((c_out * C_in + c_in) * kH + kh) * kW + kw];
                                        size_t x_idx = ((b_idx * C_in + c_in) * H + h_in) * W + w_in;
                                        dxp[x_idx] += grad_val * w;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 2. dL/dW
        size_t total_params = C_out * C_in * kH * kW;
        
        int p_idx;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for
        #endif
        for (p_idx = 0; p_idx < (int)total_params; p_idx++) {
            // Decode index
            size_t tmp = p_idx;
            size_t kw = tmp % kW; tmp /= kW;
            size_t kh = tmp % kH; tmp /= kH;
            size_t c_in = tmp % C_in; tmp /= C_in;
            size_t c_out = tmp;
            
            double sum = 0.0;
            for (size_t b = 0; b < N; b++) {
                for (size_t h_out = 0; h_out < H_out; h_out++) {
                    for (size_t w_out = 0; w_out < W_out; w_out++) {
                        int h_in = (int)(h_out * stride + kh) - (int)padding;
                        int w_in = (int)(w_out * stride + kw) - (int)padding;
                        if (h_in >= 0 && h_in < (int)H && w_in >= 0 && w_in < (int)W) {
                            double grad_val = gp[((b * C_out + c_out) * H_out + h_out) * W_out + w_out];
                            double inp = ip[((b * C_in + c_in) * H + h_in) * W + w_in];
                            sum += grad_val * inp;
                        }
                    }
                }
            }
            dwp[p_idx] = sum;
        }
    }
    
    if (grad_c != grad) nc_tensor_free(grad_c);
    if (input_c != input) nc_tensor_free(input_c);
    if (weight_c != weight) nc_tensor_free(weight_c);
    
    return grads;
}

// ============================================
// Conv2D Forward (Im2Col + GEMM)
// ============================================

nc_tensor* nc_conv2d_forward(nc_tensor* input, nc_tensor* weight, 
                             nc_tensor* bias, size_t stride, size_t padding) {
    if (!input || !weight) return NULL;
    if (input->ndim != 4 || weight->ndim != 4) {
        NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, "Conv2D requires 4D tensors");
        return NULL;
    }
    
    size_t N = input->shape[0];
    size_t C_in = input->shape[1];
    size_t H = input->shape[2];
    size_t W = input->shape[3];
    
    size_t C_out = weight->shape[0];
    size_t kH = weight->shape[2];
    size_t kW = weight->shape[3];
    
    if (weight->shape[1] != C_in) {
        NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, "Conv2D: input channels mismatch");
        return NULL;
    }
    
    size_t H_out = nc_conv_output_size(H, kH, stride, padding);
    size_t W_out = nc_conv_output_size(W, kW, stride, padding);
    
#ifdef NOCTA_CUDA_ENABLED
    // CUDA path for F32 tensors on GPU
    if (tensor_on_cuda(input) && tensor_on_cuda(weight) && 
        input->dtype == NC_F32 && nc_tensor_is_contiguous(input) && nc_tensor_is_contiguous(weight)) {
        
        size_t out_shape[] = {N, C_out, H_out, W_out};
        nc_tensor* out = nc_tensor_zeros(out_shape, 4, NC_F32);
        if (!out) return NULL;
        nc_storage_to_device(out->storage, NC_DEVICE_CUDA);
        
        const float* bias_ptr = (bias && tensor_on_cuda(bias)) ? 
            (const float*)bias->storage->cuda_data : NULL;
        
        nc_cuda_conv2d_forward_f32(
            (float*)out->storage->cuda_data,
            (const float*)input->storage->cuda_data,
            (const float*)weight->storage->cuda_data,
            bias_ptr,
            (int)N, (int)C_in, (int)H, (int)W,
            (int)C_out, (int)kH, (int)kW,
            (int)stride, (int)padding);
        
        // Setup autograd
        if (nc_grad_enabled() && (input->requires_grad || weight->requires_grad || 
            (bias && bias->requires_grad))) {
            out->requires_grad = true;
            out->is_leaf = false;
            nc_node* node = nc_node_create("conv2d", nc_backward_conv2d);
            if (node) {
                nc_node_add_input(node, input);
                nc_node_add_input(node, weight);
                if (bias) nc_node_add_input(node, bias);
                nc_node_save_tensor(node, input);
                nc_node_save_tensor(node, weight);
                nc_node_save_tensor(node, bias);
                nc_tensor* stride_t = nc_tensor_scalar((double)stride, NC_F64);
                nc_tensor* padding_t = nc_tensor_scalar((double)padding, NC_F64);
                nc_node_save_owned_tensor(node, stride_t);
                nc_node_save_owned_tensor(node, padding_t);
                node->output = out;
                out->grad_fn = node;
            }
        }
        return out;
    }
#endif
    
    // CPU path (im2col + GEMM)
    size_t out_shape[] = {N, C_out, H_out, W_out};
    nc_tensor* out = nc_tensor_zeros(out_shape, 4, input->dtype);
    if (!out) return NULL;
    
    // Reshape weight to (C_out, C_in * kH * kW)
    size_t K = C_in * kH * kW;
    size_t w_flat_shape[] = {C_out, K};
    nc_tensor* weight_flat = nc_tensor_reshape(weight, w_flat_shape, 2);
    if (!weight_flat) { nc_tensor_free(out); return NULL; }
    
    // Batched Im2Col
    // We want a matrix of shape (K, N * H_out * W_out)
    // But we allocate it as (K, N * P) where P = H_out * W_out
    size_t P = H_out * W_out;
    size_t col_shape[] = {K, N * P};
    nc_tensor* col = nc_tensor_empty(col_shape, 2, input->dtype);
    if (!col) {
        nc_tensor_free(out);
        nc_tensor_free(weight_flat);
        return NULL;
    }
    
    nc_tensor* input_c = nc_tensor_contiguous(input);
    
    // 1. Batched Im2Col
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)N; i++) {
        if (input->dtype == NC_F32) {
            float* input_data = nc_tensor_data_f32(input_c) + i * C_in * H * W;
            float* col_data = nc_tensor_data_f32(col) + i * P; // Start at offset i*P for each K-row
            // The stride between K-rows is N*P
            im2col(input_data, (int)C_in, (int)H, (int)W, (int)kH, (int)kW, 
                   (int)padding, (int)stride, col_data, (int)(N * P));
        } else {
            double* input_data = nc_tensor_data_f64(input_c) + i * C_in * H * W;
            double* col_data = nc_tensor_data_f64(col) + i * P;
            im2col_f64(input_data, (int)C_in, (int)H, (int)W, (int)kH, (int)kW, 
                       (int)padding, (int)stride, col_data, (int)(N * P));
        }
    }
    
    // 2. GEMM: (C_out, K) @ (K, N*P) -> (C_out, N*P)
    nc_tensor* out_gemm = nc_matmul(weight_flat, col);
    nc_tensor_free(col);
    nc_tensor_free(weight_flat);
    if (input_c != input) nc_tensor_free(input_c);
    
    if (!out_gemm) { nc_tensor_free(out); return NULL; }
    
    // 3. Add bias and shuffle to (N, C_out, P)
    // out_gemm is (C_out, N*P)
    // We want out (N, C_out, P)
    // out_gemm layout:
    // Row c: [ img0_p..., img1_p..., ... ]
    
    if (out->dtype == NC_F32) {
        float* src = nc_tensor_data_f32(out_gemm);
        float* dst = nc_tensor_data_f32(out);
        float* b_ptr = bias ? nc_tensor_data_f32(bias) : NULL;
        
        int n;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for
        #endif
        for (n = 0; n < (int)N; n++) {
            for (size_t c = 0; c < C_out; c++) {
                float b_val = b_ptr ? b_ptr[c] : 0.0f;
                
                // Src: Row c, Block n (offset n*P)
                float* s_ptr = src + c * (N * P) + n * P;
                // Dst: Batch n, Channel c (offset n*C_out*P + c*P)
                float* d_ptr = dst + n * (C_out * P) + c * P;
                
                for (size_t p = 0; p < P; p++) {
                    d_ptr[p] = s_ptr[p] + b_val;
                }
            }
        }
    } else {
        double* src = nc_tensor_data_f64(out_gemm);
        double* dst = nc_tensor_data_f64(out);
        double* b_ptr = bias ? nc_tensor_data_f64(bias) : NULL;
        
        int n;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for
        #endif
        for (n = 0; n < (int)N; n++) {
            for (size_t c = 0; c < C_out; c++) {
                double b_val = b_ptr ? b_ptr[c] : 0.0;
                double* s_ptr = src + c * (N * P) + n * P;
                double* d_ptr = dst + n * (C_out * P) + c * P;
                for (size_t p = 0; p < P; p++) {
                    d_ptr[p] = s_ptr[p] + b_val;
                }
            }
        }
    }
    
    nc_tensor_free(out_gemm);
    
    // Setup autograd
    if (nc_grad_enabled() && (input->requires_grad || weight->requires_grad)) {
        out->requires_grad = true;
        out->is_leaf = false;
        
        nc_node* node = nc_node_create("conv2d", nc_backward_conv2d);
        if (node) {
            nc_node_add_input(node, input);
            nc_node_add_input(node, weight);
            if (bias) nc_node_add_input(node, bias);
            
            // Save tensors for backward: input, weight, bias, stride, padding
            nc_node_save_tensor(node, input);
            nc_node_save_tensor(node, weight);
            nc_node_save_tensor(node, bias);  // Can be NULL
            
            nc_tensor* stride_t = nc_tensor_scalar((double)stride, NC_F32);
            nc_tensor* padding_t = nc_tensor_scalar((double)padding, NC_F32);
            nc_node_save_owned_tensor(node, stride_t);
            nc_node_save_owned_tensor(node, padding_t);
            
            node->output = out;
            out->grad_fn = node;
        }
    }
    
    return out;
}

// ============================================
// Conv2D Module
// ============================================

static nc_tensor* conv2d_module_forward(nc_module* self, nc_tensor* input) {
    nc_conv2d_data* data = self->extra;
    nc_tensor* weight = nc_module_get_param(self, "weight");
    nc_tensor* bias = nc_module_get_param(self, "bias");
    
    return nc_conv2d_forward(input, weight, bias, data->stride, data->padding);
}

// Custom print for Conv2D
static void conv2d_print(nc_module* m) {
    nc_conv2d_data* data = m->extra;
    printf("Conv2D(%zu, %zu, kernel_size=%zu, stride=%zu, padding=%zu, bias=%s)\n",
           data->in_channels, data->out_channels, data->kernel_size,
           data->stride, data->padding, data->has_bias ? "true" : "false");
}

nc_module* nc_conv2d(size_t in_channels, size_t out_channels,
                     size_t kernel_size, size_t stride,
                     size_t padding, bool bias) {
    nc_module* m = nc_module_create("Conv2D");
    if (!m) return NULL;
    
    nc_conv2d_data* data = nc_alloc(sizeof(nc_conv2d_data));
    if (!data) { nc_module_free(m); return NULL; }
    
    data->in_channels = in_channels;
    data->out_channels = out_channels;
    data->kernel_size = kernel_size;
    data->stride = stride;
    data->padding = padding;
    data->has_bias = bias;
    
    m->extra = data;
    m->free_extra = nc_free;
    m->forward = conv2d_module_forward;
    
    // Weight: (out_channels, in_channels, kernel_size, kernel_size)
    size_t w_shape[] = {out_channels, in_channels, kernel_size, kernel_size};
    nc_tensor* weight = nc_tensor_empty(w_shape, 4, NC_F32);
    if (!weight) { nc_module_free(m); return NULL; }
    
    // Kaiming initialization
    double fan_in = (double)(in_channels * kernel_size * kernel_size);
    double std = sqrt(2.0 / fan_in);
    for (size_t i = 0; i < weight->numel; i++) {
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        double z = sqrt(-2.0 * log(u1 + 1e-10)) * cos(2.0 * 3.14159265 * u2);
        nc_tensor_set_flat(weight, i, z * std);
    }
    nc_module_add_param(m, "weight", weight);
    
    // Bias
    if (bias) {
        size_t b_shape[] = {out_channels};
        nc_tensor* b = nc_tensor_zeros(b_shape, 1, NC_F32);
        if (!b) { nc_module_free(m); return NULL; }
        nc_module_add_param(m, "bias", b);
    }
    
    return m;
}

nc_module* nc_conv2d_simple(size_t in_channels, size_t out_channels,
                            size_t kernel_size) {
    return nc_conv2d(in_channels, out_channels, kernel_size, 1, 0, true);
}

nc_tensor* nc_conv2d_weight(nc_module* conv) {
    return nc_module_get_param(conv, "weight");
}

nc_tensor* nc_conv2d_bias(nc_module* conv) {
    return nc_module_get_param(conv, "bias");
}

// ============================================
// MaxPool2D Backward
// ============================================

nc_tensor** nc_backward_maxpool2d(nc_tensor* grad, nc_tensor** saved, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(1, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    nc_tensor* input = saved[0];
    size_t kernel_size = (size_t)nc_tensor_get_flat(saved[1], 0);
    size_t stride = (size_t)nc_tensor_get_flat(saved[2], 0);
    
    size_t N = input->shape[0];
    size_t C = input->shape[1];
    size_t H = input->shape[2];
    size_t W = input->shape[3];
    
    size_t H_out = (H - kernel_size) / stride + 1;
    size_t W_out = (W - kernel_size) / stride + 1;
    
    grads[0] = nc_tensor_zeros(input->shape, input->ndim, input->dtype);
    
#ifdef NOCTA_CUDA_ENABLED
    if (grad->storage && grad->storage->device == NC_DEVICE_CUDA &&
        input->storage && input->storage->device == NC_DEVICE_CUDA) {
        
        nc_tensor_to_device(grads[0], NC_DEVICE_CUDA);
        
        if (grads[0]->storage->cuda_data) {
            nc_cuda_maxpool2d_backward_f32(
                (float*)grads[0]->storage->cuda_data,
                (const float*)grad->storage->cuda_data,
                (const float*)input->storage->cuda_data,
                NULL, // output unused
                N, C, H, W, kernel_size, kernel_size, stride);
            return grads;
        }
    }
#endif
    
    int b;
    (void)b;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (b = 0; b < (int)N; b++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h_out = 0; h_out < H_out; h_out++) {
                for (size_t w_out = 0; w_out < W_out; w_out++) {
                    // Find max index
                    double max_val = -1e30;
                    size_t max_h = 0, max_w = 0;
                    
                    for (size_t kh = 0; kh < kernel_size; kh++) {
                        for (size_t kw = 0; kw < kernel_size; kw++) {
                            size_t h_in = h_out * stride + kh;
                            size_t w_in = w_out * stride + kw;
                            double val = nc_tensor_get4(input, b, c, h_in, w_in);
                            if (val > max_val) {
                                max_val = val;
                                max_h = h_in;
                                max_w = w_in;
                            }
                        }
                    }
                    
                    // Route gradient to max element
                    double g = nc_tensor_get4(grad, b, c, h_out, w_out);
                    double cur = nc_tensor_get4(grads[0], b, c, max_h, max_w);
                    nc_tensor_set4(grads[0], b, c, max_h, max_w, cur + g);
                }
            }
        }
    }
    
    return grads;
}

// ============================================
// MaxPool2D Forward
// ============================================

typedef struct {
    size_t kernel_size;
    size_t stride;
} nc_maxpool_data;

nc_tensor* nc_maxpool2d_forward(nc_tensor* input, size_t kernel_size, size_t stride) {
    if (!input || input->ndim != 4) return NULL;
    
    size_t N = input->shape[0];
    size_t C = input->shape[1];
    size_t H = input->shape[2];
    size_t W = input->shape[3];
    
    size_t H_out = (H - kernel_size) / stride + 1;
    size_t W_out = (W - kernel_size) / stride + 1;
    
    size_t out_shape[] = {N, C, H_out, W_out};
    nc_tensor* out = nc_tensor_empty(out_shape, 4, input->dtype);
    if (!out) return NULL;
    
#ifdef NOCTA_CUDA_ENABLED
    if (tensor_on_cuda(input)) {
        nc_tensor_to_device(out, NC_DEVICE_CUDA);
        if (out->storage->cuda_data) {
             nc_cuda_maxpool2d_forward_f32(
                (float*)out->storage->cuda_data,
                (const float*)input->storage->cuda_data,
                (int)N, (int)C, (int)H, (int)W, 
                (int)kernel_size, (int)kernel_size, (int)stride);
             
             // Setup autograd (same as CPU)
             if (nc_grad_enabled() && input->requires_grad) {
                out->requires_grad = true;
                out->is_leaf = false;
                nc_node* node = nc_node_create("maxpool2d", nc_backward_maxpool2d);
                if (node) {
                    nc_node_add_input(node, input);
                    nc_node_save_tensor(node, input);
                    nc_tensor* ks_t = nc_tensor_scalar((double)kernel_size, NC_F32);
                    nc_tensor* st_t = nc_tensor_scalar((double)stride, NC_F32);
                    nc_node_save_owned_tensor(node, ks_t);
                    nc_node_save_owned_tensor(node, st_t);
                    node->output = out;
                    out->grad_fn = node;
                }
             }
             return out;
        }
    }
#endif

    int b;
    (void)b;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (b = 0; b < (int)N; b++) {
        for (int c = 0; c < (int)C; c++) {
            for (size_t h_out = 0; h_out < H_out; h_out++) {
                for (size_t w_out = 0; w_out < W_out; w_out++) {
                    double max_val = -1e30;
                    
                    for (size_t kh = 0; kh < kernel_size; kh++) {
                        for (size_t kw = 0; kw < kernel_size; kw++) {
                            size_t h_in = h_out * stride + kh;
                            size_t w_in = w_out * stride + kw;
                            double val = nc_tensor_get4(input, b, c, h_in, w_in);
                            if (val > max_val) max_val = val;
                        }
                    }
                    
                    nc_tensor_set4(out, b, c, h_out, w_out, max_val);
                }
            }
        }
    }
    
    // Setup autograd
    if (nc_grad_enabled() && input->requires_grad) {
        out->requires_grad = true;
        out->is_leaf = false;
        
        nc_node* node = nc_node_create("maxpool2d", nc_backward_maxpool2d);
        if (node) {
            nc_node_add_input(node, input);
            nc_node_save_tensor(node, input);
            
            nc_tensor* ks_t = nc_tensor_scalar((double)kernel_size, NC_F32);
            nc_tensor* st_t = nc_tensor_scalar((double)stride, NC_F32);
            nc_node_save_owned_tensor(node, ks_t);
            nc_node_save_owned_tensor(node, st_t);
            
            node->output = out;
            out->grad_fn = node;
        }
    }
    
    return out;
}

static nc_tensor* maxpool_forward(nc_module* self, nc_tensor* input) {
    nc_maxpool_data* data = self->extra;
    return nc_maxpool2d_forward(input, data->kernel_size, data->stride);
}

// Custom print for MaxPool2D
static void maxpool_print(nc_module* m) {
    nc_maxpool_data* data = m->extra;
    printf("MaxPool2D(kernel_size=%zu, stride=%zu)\n",
           data->kernel_size, data->stride);
}

nc_module* nc_maxpool2d(size_t kernel_size, size_t stride) {
    nc_module* m = nc_module_create("MaxPool2D");
    if (!m) return NULL;
    
    nc_maxpool_data* data = nc_alloc(sizeof(nc_maxpool_data));
    if (!data) { nc_module_free(m); return NULL; }
    
    data->kernel_size = kernel_size;
    data->stride = stride;
    
    m->extra = data;
    m->free_extra = nc_free;
    m->forward = maxpool_forward;
    
    return m;
}

nc_module* nc_maxpool2d_simple(size_t kernel_size) {
    return nc_maxpool2d(kernel_size, kernel_size);
}

// ============================================
// AvgPool2D
// ============================================

nc_tensor* nc_avgpool2d_forward(nc_tensor* input, size_t kernel_size, size_t stride) {
    if (!input || input->ndim != 4) return NULL;
    
    size_t N = input->shape[0];
    size_t C = input->shape[1];
    size_t H = input->shape[2];
    size_t W = input->shape[3];
    
    size_t H_out = (H - kernel_size) / stride + 1;
    size_t W_out = (W - kernel_size) / stride + 1;
    
    size_t out_shape[] = {N, C, H_out, W_out};
    nc_tensor* out = nc_tensor_zeros(out_shape, 4, input->dtype);
    if (!out) return NULL;
    
    double pool_size = (double)(kernel_size * kernel_size);
    
    int b;
    (void)b;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (b = 0; b < (int)N; b++) {
        for (int c = 0; c < (int)C; c++) {
            for (size_t h_out = 0; h_out < H_out; h_out++) {
                for (size_t w_out = 0; w_out < W_out; w_out++) {
                    double sum = 0.0;
                    
                    for (size_t kh = 0; kh < kernel_size; kh++) {
                        for (size_t kw = 0; kw < kernel_size; kw++) {
                            size_t h_in = h_out * stride + kh;
                            size_t w_in = w_out * stride + kw;
                            sum += nc_tensor_get4(input, b, c, h_in, w_in);
                        }
                    }
                    
                    nc_tensor_set4(out, b, c, h_out, w_out, sum / pool_size);
                }
            }
        }
    }
    
    return out;
}

// ============================================
// Flatten Layer
// ============================================

static nc_tensor* flatten_forward(nc_module* self, nc_tensor* input) {
    (void)self;
    if (!input) return NULL;
    
    // Keep batch dimension, flatten the rest
    size_t batch = input->shape[0];
    size_t flat_size = input->numel / batch;
    
    size_t out_shape[] = {batch, flat_size};
    return nc_tensor_reshape(input, out_shape, 2);
}

nc_module* nc_flatten(void) {
    nc_module* m = nc_module_create("Flatten");
    if (!m) return NULL;
    m->forward = flatten_forward;
    return m;
}