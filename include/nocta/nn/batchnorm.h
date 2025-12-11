#ifndef NOCTA_BATCHNORM_H
#define NOCTA_BATCHNORM_H

#include "nocta/nn/module.h"
#include "nocta/core/tensor.h"

// ============================================
// BatchNorm2D Layer
// ============================================
// Applies Batch Normalization over a 4D input (N, C, H, W)
// y = (x - mean) / sqrt(var + eps) * gamma + beta
//
// During training: uses batch statistics
// During inference: uses running statistics

typedef struct {
    size_t num_features;    // Number of channels (C)
    double eps;             // Small constant for numerical stability
    double momentum;        // Momentum for running stats update
    bool affine;            // Whether to learn gamma/beta
    bool track_running_stats;
} nc_batchnorm2d_config;

// Default config
static const nc_batchnorm2d_config NC_BATCHNORM2D_DEFAULT = {
    .num_features = 0,
    .eps = 1e-5,
    .momentum = 0.1,
    .affine = true,
    .track_running_stats = true
};

// Create BatchNorm2D layer
nc_module* nc_batchnorm2d(size_t num_features);

// Create BatchNorm2D with custom config
nc_module* nc_batchnorm2d_ex(nc_batchnorm2d_config config);

// Get parameters
nc_tensor* nc_batchnorm2d_weight(nc_module* bn);  // gamma
nc_tensor* nc_batchnorm2d_bias(nc_module* bn);    // beta
nc_tensor* nc_batchnorm2d_running_mean(nc_module* bn);
nc_tensor* nc_batchnorm2d_running_var(nc_module* bn);

// ============================================
// BatchNorm1D Layer
// ============================================
// Applies Batch Normalization over a 2D or 3D input
// 2D input: (N, C) - normalizes over N
// 3D input: (N, C, L) - normalizes over N and L

nc_module* nc_batchnorm1d(size_t num_features);
nc_module* nc_batchnorm1d_ex(nc_batchnorm2d_config config);

// ============================================
// LayerNorm
// ============================================
// Applies Layer Normalization over the last D dimensions
// Commonly used in transformers

typedef struct {
    size_t* normalized_shape;
    size_t normalized_ndim;
    double eps;
    bool elementwise_affine;
} nc_layernorm_config;

nc_module* nc_layernorm(const size_t* normalized_shape, size_t ndim);
nc_module* nc_layernorm_ex(nc_layernorm_config config);

// ============================================
// Functional API
// ============================================

// BatchNorm forward (functional)
nc_tensor* nc_batchnorm2d_forward_fn(
    nc_tensor* input,
    nc_tensor* running_mean,
    nc_tensor* running_var,
    nc_tensor* weight,
    nc_tensor* bias,
    bool training,
    double momentum,
    double eps
);

nc_tensor* nc_batchnorm1d_forward_fn(
    nc_tensor* input,
    nc_tensor* running_mean,
    nc_tensor* running_var,
    nc_tensor* weight,
    nc_tensor* bias,
    bool training,
    double momentum,
    double eps
);

// LayerNorm forward (functional)
nc_tensor* nc_layernorm_forward_fn(
    nc_tensor* input,
    const size_t* normalized_shape,
    size_t normalized_ndim,
    nc_tensor* weight,
    nc_tensor* bias,
    double eps
);

#endif // NOCTA_BATCHNORM_H