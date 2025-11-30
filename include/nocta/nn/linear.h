#ifndef NOCTA_LINEAR_H
#define NOCTA_LINEAR_H

#include "nocta/nn/module.h"
#include "nocta/core/tensor.h"

// Linear (fully connected) layer
// y = x @ W^T + b

// Create linear layer
// in_features: input dimension
// out_features: output dimension
// bias: whether to include bias term
nc_module* nc_linear(size_t in_features, size_t out_features, bool bias);

// Get weight tensor from linear module
nc_tensor* nc_linear_weight(nc_module* linear);

// Get bias tensor from linear module
nc_tensor* nc_linear_bias(nc_module* linear);

// ============================================
// Weight initialization
// ============================================

typedef enum {
    NC_INIT_ZEROS,
    NC_INIT_ONES,
    NC_INIT_UNIFORM,        // Uniform [-1/sqrt(fan_in), 1/sqrt(fan_in)]
    NC_INIT_NORMAL,         // Normal(0, 1/sqrt(fan_in))
    NC_INIT_XAVIER_UNIFORM, // Xavier/Glorot uniform
    NC_INIT_XAVIER_NORMAL,  // Xavier/Glorot normal
    NC_INIT_KAIMING_UNIFORM,// He uniform (for ReLU)
    NC_INIT_KAIMING_NORMAL, // He normal (for ReLU)
    NC_INIT_ORTHOGONAL,     // Orthogonal initialization
} nc_init_type;

// Initialize tensor with specified method
void nc_init_(nc_tensor* t, nc_init_type type, double gain);

// Initialize linear layer weights
void nc_linear_init_(nc_module* linear, nc_init_type type);

#endif // NOCTA_LINEAR_H