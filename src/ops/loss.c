#include "nocta/ops/loss.h"
#include "nocta/ops/activation.h"
#include "nocta/ops/arithmetic.h"
#include "nocta/ops/reduction.h"
#include "nocta/autograd/node.h"
#include "nocta/autograd/backward.h"
#include "nocta/core/memory.h"
#include <math.h>

#ifdef NOCTA_OPENMP_ENABLED
#include <omp.h>
#endif

// ============================================
// Cross Entropy Backward
// ============================================

// Combined softmax + cross entropy backward
// This is more numerically stable than separate backward passes
// grad_logits = softmax(logits) - one_hot(targets)
nc_tensor** nc_backward_cross_entropy(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(2, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    nc_tensor* logits = inputs[0];   // (N, C)
    nc_tensor* targets = inputs[1];  // (N,)
    
    size_t batch = logits->shape[0];
    size_t n_classes = logits->shape[1];
    
    // Compute softmax
    nc_tensor* probs = nc_softmax(logits, 1);
    
    // grad_logits = probs - one_hot(targets)
    grads[0] = nc_tensor_clone(probs);
    
    int b;
    (void)b;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (b = 0; b < (int)batch; b++) {
        int label = (int)nc_tensor_get1(targets, b);
        if (label >= 0 && (size_t)label < n_classes) {
            double p = nc_tensor_get2(grads[0], b, label);
            nc_tensor_set2(grads[0], b, label, p - 1.0);
        }
    }
    
    // Scale by upstream gradient and 1/batch for mean reduction
    double scale = nc_tensor_get_flat(grad, 0) / (double)batch;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)grads[0]->numel; i++) {
        nc_tensor_set_flat(grads[0], i, nc_tensor_get_flat(grads[0], i) * scale);
    }
    
    nc_tensor_free(probs);
    return grads;
}

// ============================================
// MSE Backward
// ============================================

nc_tensor** nc_backward_mse(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(2, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    nc_tensor* pred = inputs[0];
    nc_tensor* target = inputs[1];
    
    // d(MSE)/d(pred) = 2 * (pred - target) / n
    double scale = 2.0 * nc_tensor_get_flat(grad, 0) / (double)pred->numel;
    
    grads[0] = nc_tensor_empty(pred->shape, pred->ndim, pred->dtype);
    grads[1] = nc_tensor_empty(target->shape, target->ndim, target->dtype);
    
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)pred->numel; i++) {
        double diff = nc_tensor_get_flat(pred, i) - nc_tensor_get_flat(target, i);
        nc_tensor_set_flat(grads[0], i, scale * diff);
        nc_tensor_set_flat(grads[1], i, -scale * diff);
    }
    
    return grads;
}

// ============================================
// BCE Backward
// ============================================

nc_tensor** nc_backward_bce(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(2, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    nc_tensor* pred = inputs[0];
    nc_tensor* target = inputs[1];
    
    double g = nc_tensor_get_flat(grad, 0) / (double)pred->numel;
    
    grads[0] = nc_tensor_empty(pred->shape, pred->ndim, pred->dtype);
    grads[1] = NULL; // Usually don't need gradient w.r.t. target
    
    // d(BCE)/d(pred) = -target/pred + (1-target)/(1-pred)
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)pred->numel; i++) {
        double p = nc_tensor_get_flat(pred, i);
        double t = nc_tensor_get_flat(target, i);
        
        // Clamp for numerical stability
        p = fmax(1e-7, fmin(1.0 - 1e-7, p));
        
        double grad_p = (-t / p + (1.0 - t) / (1.0 - p)) * g;
        nc_tensor_set_flat(grads[0], i, grad_p);
    }
    
    return grads;
}

// ============================================
// Loss Functions Implementation
// ============================================

nc_tensor* nc_cross_entropy_loss(nc_tensor* logits, nc_tensor* targets) {
    return nc_cross_entropy_loss_ex(logits, targets, NC_REDUCTION_MEAN);
}

nc_tensor* nc_cross_entropy_loss_ex(nc_tensor* logits, nc_tensor* targets,
                                     nc_reduction reduction) {
    if (!logits || !targets) return NULL;
    if (logits->ndim != 2) {
        NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, "logits must be 2D (N, C)");
        return NULL;
    }
    
    size_t batch = logits->shape[0];
    size_t n_classes = logits->shape[1];
    
    // Compute log_softmax for numerical stability
    nc_tensor* log_probs = nc_log_softmax(logits, 1);
    if (!log_probs) return NULL;
    
    // Compute NLL loss
    double loss_sum = 0.0;
    int b;
    (void)b;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for reduction(+:loss_sum)
    #endif
    for (b = 0; b < (int)batch; b++) {
        int label = (int)nc_tensor_get1(targets, b);
        if (label >= 0 && (size_t)label < n_classes) {
            loss_sum -= nc_tensor_get2(log_probs, b, label);
        }
    }
    
    nc_tensor_free(log_probs);
    
    // Apply reduction
    double loss_val;
    switch (reduction) {
        case NC_REDUCTION_MEAN:
            loss_val = loss_sum / (double)batch;
            break;
        case NC_REDUCTION_SUM:
            loss_val = loss_sum;
            break;
        case NC_REDUCTION_NONE:
        default:
            loss_val = loss_sum;  // Simplified, should return tensor
            break;
    }
    
    nc_tensor* loss = nc_tensor_scalar(loss_val, logits->dtype);
    if (!loss) return NULL;
    
    // Setup autograd
    if (nc_grad_enabled() && logits->requires_grad) {
        loss->requires_grad = true;
        loss->is_leaf = false;
        
        nc_node* node = nc_node_create("cross_entropy", nc_backward_cross_entropy);
        if (node) {
            nc_node_add_input(node, logits);
            nc_node_add_input(node, targets);
            nc_node_save_tensor(node, logits);
            nc_node_save_tensor(node, targets);
            node->output = loss;
            loss->grad_fn = node;
        }
    }
    
    return loss;
}

nc_tensor* nc_mse_loss(nc_tensor* pred, nc_tensor* target) {
    if (!pred || !target) return NULL;
    
    // Compute MSE
    double sum = 0.0;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for reduction(+:sum)
    #endif
    for (i = 0; i < (int)pred->numel; i++) {
        double diff = nc_tensor_get_flat(pred, i) - nc_tensor_get_flat(target, i);
        sum += diff * diff;
    }
    
    nc_tensor* loss = nc_tensor_scalar(sum / (double)pred->numel, pred->dtype);
    if (!loss) return NULL;
    
    // Setup autograd
    if (nc_grad_enabled() && pred->requires_grad) {
        loss->requires_grad = true;
        loss->is_leaf = false;
        
        nc_node* node = nc_node_create("mse_loss", nc_backward_mse);
        if (node) {
            nc_node_add_input(node, pred);
            nc_node_add_input(node, target);
            nc_node_save_tensor(node, pred);
            nc_node_save_tensor(node, target);
            node->output = loss;
            loss->grad_fn = node;
        }
    }
    
    return loss;
}

nc_tensor* nc_bce_loss(nc_tensor* pred, nc_tensor* target) {
    if (!pred || !target) return NULL;
    
    // Compute BCE
    double sum = 0.0;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for reduction(-:sum)
    #endif
    for (i = 0; i < (int)pred->numel; i++) {
        double p = nc_tensor_get_flat(pred, i);
        double t = nc_tensor_get_flat(target, i);
        
        // Clamp for numerical stability
        p = fmax(1e-7, fmin(1.0 - 1e-7, p));
        
        sum -= t * log(p) + (1.0 - t) * log(1.0 - p);
    }
    
    nc_tensor* loss = nc_tensor_scalar(sum / (double)pred->numel, pred->dtype);
    if (!loss) return NULL;
    
    // Setup autograd
    if (nc_grad_enabled() && pred->requires_grad) {
        loss->requires_grad = true;
        loss->is_leaf = false;
        
        nc_node* node = nc_node_create("bce_loss", nc_backward_bce);
        if (node) {
            nc_node_add_input(node, pred);
            nc_node_add_input(node, target);
            nc_node_save_tensor(node, pred);
            nc_node_save_tensor(node, target);
            node->output = loss;
            loss->grad_fn = node;
        }
    }
    
    return loss;
}

nc_tensor* nc_nll_loss(nc_tensor* log_probs, nc_tensor* targets) {
    if (!log_probs || !targets) return NULL;
    
    size_t batch = log_probs->shape[0];
    size_t n_classes = log_probs->shape[1];
    
    double loss_sum = 0.0;
    int b;
    (void)b;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for reduction(-:loss_sum)
    #endif
    for (b = 0; b < (int)batch; b++) {
        int label = (int)nc_tensor_get1(targets, b);
        if (label >= 0 && (size_t)label < n_classes) {
            loss_sum -= nc_tensor_get2(log_probs, b, label);
        }
    }
    
    return nc_tensor_scalar(loss_sum / (double)batch, log_probs->dtype);
}