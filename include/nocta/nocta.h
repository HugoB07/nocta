#ifndef NOCTA_H
#define NOCTA_H

// Nocta - A lightweight deep learning library in C
// Version 0.1.0

// ============================================
// Core
// ============================================
#include "nocta/core/dtype.h"
#include "nocta/core/error.h"
#include "nocta/core/memory.h"
#include "nocta/core/tensor.h"
#include "nocta/core/serialize.h"

// ============================================
// Autograd
// ============================================
#include "nocta/autograd/node.h"
#include "nocta/autograd/backward.h"

// ============================================
// Operations
// ============================================
#include "nocta/ops/arithmetic.h"
#include "nocta/ops/matmul.h"
#include "nocta/ops/activation.h"
#include "nocta/ops/reduction.h"

// ============================================
// Neural Network Modules
// ============================================
#include "nocta/nn/module.h"
#include "nocta/nn/linear.h"

// ============================================
// Optimizers
// ============================================
#include "nocta/optim/optimizer.h"
#include "nocta/optim/sgd.h"
#include "nocta/optim/adam.h"

// ============================================
// Version info
// ============================================
#define NOCTA_VERSION_MAJOR 0
#define NOCTA_VERSION_MINOR 1
#define NOCTA_VERSION_PATCH 0
#define NOCTA_VERSION "0.1.0"

// ============================================
// Global initialization / cleanup
// ============================================

// Initialize Nocta (call once at startup)
static inline void nc_init(void) {
    // Future: initialize RNG, thread pool, etc.
}

// Cleanup Nocta (call once at shutdown)
static inline void nc_cleanup(void) {
    nc_memory_report();
}

// ============================================
// Loss functions (convenience)
// ============================================

// Mean Squared Error: mean((pred - target)^2)
static inline nc_tensor* nc_mse_loss(nc_tensor* pred, nc_tensor* target) {
    nc_tensor* diff = nc_sub(pred, target);
    nc_tensor* sq = nc_square(diff);
    nc_tensor* loss = nc_mean_all(sq);
    nc_tensor_free(diff);
    nc_tensor_free(sq);
    return loss;
}

// Binary Cross Entropy: -mean(target * log(pred) + (1-target) * log(1-pred))
static inline nc_tensor* nc_bce_loss(nc_tensor* pred, nc_tensor* target) {
    nc_tensor* log_pred = nc_log(nc_clamp(pred, 1e-7, 1.0 - 1e-7));
    nc_tensor* log_1mp = nc_log(nc_clamp(nc_sub(nc_tensor_ones(pred->shape, pred->ndim, pred->dtype), pred), 1e-7, 1.0 - 1e-7));
    nc_tensor* one_mt = nc_sub(nc_tensor_ones(target->shape, target->ndim, target->dtype), target);
    
    nc_tensor* term1 = nc_mul(target, log_pred);
    nc_tensor* term2 = nc_mul(one_mt, log_1mp);
    nc_tensor* sum = nc_add(term1, term2);
    nc_tensor* neg = nc_neg(sum);
    nc_tensor* loss = nc_mean_all(neg);
    
    // Cleanup intermediates
    nc_tensor_free(log_pred);
    nc_tensor_free(log_1mp);
    nc_tensor_free(one_mt);
    nc_tensor_free(term1);
    nc_tensor_free(term2);
    nc_tensor_free(sum);
    nc_tensor_free(neg);
    
    return loss;
}

// Cross Entropy Loss (with log_softmax)
static inline nc_tensor* nc_cross_entropy_loss(nc_tensor* logits, nc_tensor* targets) {
    nc_tensor* log_probs = nc_log_softmax(logits, -1);
    
    // For simplicity, assume targets are class indices (1D)
    // Full implementation would handle one-hot and class indices
    nc_tensor* loss = nc_neg(nc_mean_all(log_probs));
    
    nc_tensor_free(log_probs);
    return loss;
}

#endif // NOCTA_H