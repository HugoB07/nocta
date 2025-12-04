#ifndef NOCTA_CONV_H
#define NOCTA_CONV_H

#include "nocta/nn/module.h"
#include "nocta/core/tensor.h"

// ============================================
// Conv2D Layer
// ============================================
// Input:  (N, C_in, H, W)
// Output: (N, C_out, H_out, W_out)
// Where:
//   H_out = (H + 2*padding - kernel_size) / stride + 1
//   W_out = (W + 2*padding - kernel_size) / stride + 1

typedef struct {
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    bool has_bias;
} nc_conv2d_config;

// Create Conv2D layer
nc_module* nc_conv2d(size_t in_channels, size_t out_channels, 
                     size_t kernel_size, size_t stride, 
                     size_t padding, bool bias);

// Convenience: Conv2D with defaults (stride=1, padding=0, bias=true)
nc_module* nc_conv2d_simple(size_t in_channels, size_t out_channels, 
                            size_t kernel_size);

// Get weight/bias from conv module
nc_tensor* nc_conv2d_weight(nc_module* conv);
nc_tensor* nc_conv2d_bias(nc_module* conv);

// ============================================
// Pooling Layers
// ============================================

// MaxPool2D
// Input:  (N, C, H, W)
// Output: (N, C, H/kernel_size, W/kernel_size)
nc_module* nc_maxpool2d(size_t kernel_size, size_t stride);

// Convenience: MaxPool2D with stride = kernel_size
nc_module* nc_maxpool2d_simple(size_t kernel_size);

// AvgPool2D
nc_module* nc_avgpool2d(size_t kernel_size, size_t stride);

// Global Average Pooling (reduces H,W to 1,1)
nc_module* nc_global_avgpool2d(void);

// ============================================
// Functional API (direct tensor operations)
// ============================================

// Conv2D forward pass
nc_tensor* nc_conv2d_forward(nc_tensor* input, nc_tensor* weight, 
                             nc_tensor* bias, size_t stride, size_t padding);

// MaxPool2D forward pass
nc_tensor* nc_maxpool2d_forward(nc_tensor* input, size_t kernel_size, 
                                size_t stride);

// AvgPool2D forward pass
nc_tensor* nc_avgpool2d_forward(nc_tensor* input, size_t kernel_size,
                                size_t stride);

// ============================================
// Utility
// ============================================

// Compute output size for conv/pool
static inline size_t nc_conv_output_size(size_t input_size, size_t kernel_size,
                                         size_t stride, size_t padding) {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

// Flatten layer (for transitioning from conv to linear)
nc_module* nc_flatten(void);

#endif // NOCTA_CONV_H