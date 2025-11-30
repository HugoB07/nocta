#include "nocta/core/tensor.h"
#include "nocta/core/memory.h"
#include "nocta/nn/linear.h"
#include "nocta/nn/module.h"
#include "nocta/ops/matmul.h"
#include "nocta/ops/arithmetic.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Explicit declaration to ensure correct prototype on 64-bit
extern nc_tensor* nc_tensor_empty(const size_t* shape, size_t ndim, nc_dtype dtype);
extern nc_tensor* nc_tensor_zeros(const size_t* shape, size_t ndim, nc_dtype dtype);
extern void nc_tensor_requires_grad_(nc_tensor* t, bool requires_grad);

// Extra data for linear layer
typedef struct {
    size_t in_features;
    size_t out_features;
    bool has_bias;
} nc_linear_data;

// Forward pass: y = x @ W^T + b
static nc_tensor* linear_forward(nc_module* self, nc_tensor* input) {
    nc_tensor* weight = nc_module_get_param(self, "weight");
    nc_tensor* bias = nc_module_get_param(self, "bias");
    
    if (!weight || !input) return NULL;
    
    // Weight is (out_features, in_features), need to transpose
    nc_tensor* wt = nc_tensor_t(weight);
    if (!wt) return NULL;
    
    // x @ W^T
    nc_tensor* out = nc_matmul(input, wt);
    nc_tensor_free(wt);
    
    if (!out) return NULL;
    
    // Add bias if present
    if (bias) {
        nc_tensor* out_bias = nc_add(out, bias);
        nc_tensor_free(out);
        out = out_bias;
    }
    
    return out;
}

nc_module* nc_linear(size_t in_features, size_t out_features, bool bias) {
    nc_module* m = nc_module_create("Linear");
    if (!m) return NULL;
    
    // Store config
    nc_linear_data* data = nc_alloc(sizeof(nc_linear_data));
    if (!data) {
        nc_module_free(m);
        return NULL;
    }
    data->in_features = in_features;
    data->out_features = out_features;
    data->has_bias = bias;
    m->extra = data;
    m->free_extra = nc_free;
    
    // Set forward function
    m->forward = linear_forward;
    
    // Create weight: (out_features, in_features)
    size_t w_shape[] = {out_features, in_features};
    nc_tensor* weight = nc_tensor_empty(w_shape, 2, NC_F32);
    if (!weight) {
        nc_module_free(m);
        return NULL;
    }
    
    // Kaiming uniform initialization
    double bound = 1.0 / sqrt((double)in_features);
    for (size_t i = 0; i < weight->numel; i++) {
        double r = (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
        nc_tensor_set_flat(weight, i, r * bound);
    }
    nc_module_add_param(m, "weight", weight);
    
    // Create bias
    if (bias) {
        size_t b_shape[] = {out_features};
        nc_tensor* b = nc_tensor_empty(b_shape, 1, NC_F32);
        if (!b) {
            nc_module_free(m);
            return NULL;
        }
        for (size_t i = 0; i < b->numel; i++) {
            double r = (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
            nc_tensor_set_flat(b, i, r * bound);
        }
        nc_module_add_param(m, "bias", b);
    }
    
    return m;
}

nc_tensor* nc_linear_weight(nc_module* linear) {
    return nc_module_get_param(linear, "weight");
}

nc_tensor* nc_linear_bias(nc_module* linear) {
    return nc_module_get_param(linear, "bias");
}

// ============================================
// Initialization functions
// ============================================

void nc_init_(nc_tensor* t, nc_init_type type, double gain) {
    if (!t) return;
    
    size_t fan_in = t->ndim >= 2 ? t->shape[1] : t->shape[0];
    size_t fan_out = t->shape[0];
    
    switch (type) {
        case NC_INIT_ZEROS:
            nc_tensor_fill_(t, 0.0);
            break;
            
        case NC_INIT_ONES:
            nc_tensor_fill_(t, 1.0);
            break;
            
        case NC_INIT_UNIFORM: {
            double bound = gain / sqrt((double)fan_in);
            for (size_t i = 0; i < t->numel; i++) {
                double r = (double)rand() / RAND_MAX * 2.0 - 1.0;
                nc_tensor_set_flat(t, i, r * bound);
            }
            break;
        }
        
        case NC_INIT_NORMAL: {
            double std = gain / sqrt((double)fan_in);
            for (size_t i = 0; i < t->numel; i++) {
                // Box-Muller transform
                double u1 = (double)rand() / RAND_MAX;
                double u2 = (double)rand() / RAND_MAX;
                double z = sqrt(-2.0 * log(u1 + 1e-10)) * cos(2.0 * M_PI * u2);
                nc_tensor_set_flat(t, i, z * std);
            }
            break;
        }
        
        case NC_INIT_XAVIER_UNIFORM: {
            double bound = gain * sqrt(6.0 / (double)(fan_in + fan_out));
            for (size_t i = 0; i < t->numel; i++) {
                double r = (double)rand() / RAND_MAX * 2.0 - 1.0;
                nc_tensor_set_flat(t, i, r * bound);
            }
            break;
        }
        
        case NC_INIT_XAVIER_NORMAL: {
            double std = gain * sqrt(2.0 / (double)(fan_in + fan_out));
            for (size_t i = 0; i < t->numel; i++) {
                double u1 = (double)rand() / RAND_MAX;
                double u2 = (double)rand() / RAND_MAX;
                double z = sqrt(-2.0 * log(u1 + 1e-10)) * cos(2.0 * M_PI * u2);
                nc_tensor_set_flat(t, i, z * std);
            }
            break;
        }
        
        case NC_INIT_KAIMING_UNIFORM: {
            double bound = gain * sqrt(3.0 / (double)fan_in);
            for (size_t i = 0; i < t->numel; i++) {
                double r = (double)rand() / RAND_MAX * 2.0 - 1.0;
                nc_tensor_set_flat(t, i, r * bound);
            }
            break;
        }
        
        case NC_INIT_KAIMING_NORMAL: {
            double std = gain / sqrt((double)fan_in);
            for (size_t i = 0; i < t->numel; i++) {
                double u1 = (double)rand() / RAND_MAX;
                double u2 = (double)rand() / RAND_MAX;
                double z = sqrt(-2.0 * log(u1 + 1e-10)) * cos(2.0 * M_PI * u2);
                nc_tensor_set_flat(t, i, z * std);
            }
            break;
        }
        
        case NC_INIT_ORTHOGONAL:
            // Simplified: use normal for now
            // Full implementation would use QR decomposition
            nc_init_(t, NC_INIT_NORMAL, gain);
            break;
    }
}

void nc_linear_init_(nc_module* linear, nc_init_type type) {
    if (!linear) return;
    
    nc_tensor* weight = nc_linear_weight(linear);
    nc_tensor* bias = nc_linear_bias(linear);
    
    double gain = (type == NC_INIT_KAIMING_UNIFORM || type == NC_INIT_KAIMING_NORMAL) 
                  ? sqrt(2.0) : 1.0;
    
    if (weight) nc_init_(weight, type, gain);
    if (bias) nc_init_(bias, NC_INIT_ZEROS, 1.0);
}