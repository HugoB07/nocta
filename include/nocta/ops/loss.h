#ifndef NOCTA_LOSS_H
#define NOCTA_LOSS_H

#include "nocta/core/tensor.h"

// ============================================
// Loss Functions (with autograd support)
// ============================================

// Mean Squared Error: mean((pred - target)^2)
// Supports autograd
nc_tensor* nc_mse_loss(nc_tensor* pred, nc_tensor* target);

// Binary Cross Entropy: -mean(target * log(pred) + (1-target) * log(1-pred))
// pred should be in (0, 1) range (after sigmoid)
// Supports autograd
nc_tensor* nc_bce_loss(nc_tensor* pred, nc_tensor* target);

// Cross Entropy Loss (combines log_softmax + nll_loss)
// logits: (N, C) raw scores
// targets: (N,) class indices (0 to C-1)
// Returns scalar loss with autograd support
nc_tensor* nc_cross_entropy_loss(nc_tensor* logits, nc_tensor* targets);

// Negative Log Likelihood Loss
// log_probs: (N, C) log probabilities
// targets: (N,) class indices
nc_tensor* nc_nll_loss(nc_tensor* log_probs, nc_tensor* targets);

// ============================================
// Loss reduction modes
// ============================================

typedef enum {
    NC_REDUCTION_MEAN,  // Default: mean over batch
    NC_REDUCTION_SUM,   // Sum over batch
    NC_REDUCTION_NONE   // No reduction, return per-sample loss
} nc_reduction;

// Cross entropy with reduction option
nc_tensor* nc_cross_entropy_loss_ex(nc_tensor* logits, nc_tensor* targets, 
                                     nc_reduction reduction);

#endif // NOCTA_LOSS_H