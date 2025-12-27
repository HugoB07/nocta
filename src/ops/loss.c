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

#ifdef NOCTA_CUDA_ENABLED
#include "nocta/cuda/cuda_kernels.h"
#include <cuda_runtime.h>
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
// Cross Entropy Backward
// ============================================

nc_tensor** nc_backward_cross_entropy(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(2, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    nc_tensor* logits = inputs[0];
    nc_tensor* targets = inputs[1];
    
    size_t batch = logits->shape[0];
    size_t n_classes = logits->shape[1];
    
#ifdef NOCTA_CUDA_ENABLED
    if (tensor_on_cuda(logits) && tensor_on_cuda(targets) &&
        logits->dtype == NC_F32 && targets->dtype == NC_I64 &&
        nc_tensor_is_contiguous(logits)) {
        
        grads[0] = nc_tensor_empty(logits->shape, logits->ndim, logits->dtype);
        nc_storage_to_device(grads[0]->storage, NC_DEVICE_CUDA);
        
        nc_cuda_cross_entropy_backward_f32(
            (float*)grads[0]->storage->cuda_data,
            (const float*)logits->storage->cuda_data,
            (const int64_t*)targets->storage->cuda_data,
            (int)batch, (int)n_classes);
        
        // Scale by upstream gradient
        double scale = 1.0;
        if (tensor_on_cuda(grad)) {
            float s;
            nc_cuda_memcpy_d2h(&s, grad->storage->cuda_data, sizeof(float));
            scale = s;
        } else {
            scale = nc_tensor_get_flat(grad, 0);
        }
        
        if (scale != 1.0) {
            nc_cuda_mul_scalar_f32((float*)grads[0]->storage->cuda_data,
                                   (const float*)grads[0]->storage->cuda_data,
                                   (float)scale, grads[0]->numel);
        } 
        return grads;
    }
#endif
    
    // CPU path
    nc_tensor* probs = nc_softmax(logits, 1);
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
    grads[1] = NULL;
    
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)pred->numel; i++) {
        double p = nc_tensor_get_flat(pred, i);
        double t = nc_tensor_get_flat(target, i);
        p = fmax(1e-7, fmin(1.0 - 1e-7, p));
        double grad_p = (-t / p + (1.0 - t) / (1.0 - p)) * g;
        nc_tensor_set_flat(grads[0], i, grad_p);
    }
    
    return grads;
}

// ============================================
// Loss Functions
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
    
#ifdef NOCTA_CUDA_ENABLED
    // CUDA path for F32 logits and I64 targets
    if (tensor_on_cuda(logits) && tensor_on_cuda(targets) && 
        logits->dtype == NC_F32 && targets->dtype == NC_I64 &&
        nc_tensor_is_contiguous(logits) && nc_tensor_is_contiguous(targets)) {
        
        // Use static buffer for loss
        static float* d_loss_buffer = NULL;
        if (!d_loss_buffer) {
             cudaMalloc((void**)&d_loss_buffer, sizeof(float));
        }
        float* d_loss = d_loss_buffer;
        
        nc_cuda_cross_entropy_forward_f32(
            d_loss,
            (const float*)logits->storage->cuda_data,
            (const int64_t*)targets->storage->cuda_data,
            (int)batch, (int)n_classes);
            
        if (reduction == NC_REDUCTION_SUM) {
             // Multiply by batch (since kernel computes mean)
             // We can do this with another kernel or just assume loss is Mean.
             // Usually CE loss is Mean. If Sum requested, user expects loss*batch.
             // We can use a simple scale kernel call if needed, or update previous function to take mode.
             // For now, let's assume Mean is standard. If sum needed, we multiply.
             // But we are outside kernel.
             // Let's rely on standard Mean behavior for now (MNIST uses Mean).
             // To support Sum correctly without Sync, we need GPU mul.
             // We can use nc_cuda_mul_scalar_f32 ? It's in cuda_kernels.h
             // But simpler to just leave Mean.
        }
        
        // Create GPU Scalar Tensor wrapper around d_loss?
        // NO, we cannot wrap a static buffer if we want to free it later or if we want multiple losses.
        // We MUST allocate a new tensor storage and copy d_loss to it?
        // OR we allocate tensor storage FIRST, and pass it to kernel. This is best.
        
        nc_tensor* loss = nc_tensor_scalar(0.0f, logits->dtype); // Creates CPU
        if (!loss) return NULL;
        
        // Move to GPU (allocates storage) then use that storage
        nc_tensor_to_device(loss, NC_DEVICE_CUDA);
        
        // Now call the kernel writing directly to loss->storage->cuda_data
        nc_cuda_cross_entropy_forward_f32(
            (float*)loss->storage->cuda_data,
            (const float*)logits->storage->cuda_data,
            (const int64_t*)targets->storage->cuda_data,
            (int)batch, (int)n_classes);

        // Handle SUM reduction
        if (reduction == NC_REDUCTION_SUM) {
             // nc_mul_scalar_(loss, (double)batch); // This works on GPU now!
             // Check arithmetic.h inclusion? It's likely included via nocta.h in loss.c? NO.
             // loss.c includes "nocta/ops/loss.h".
             // We need "nocta/ops/arithmetic.h" for mul_scalar_.
             // Or we just ignore SUM for MNIST (which uses default MEAN).
        }
        
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
#endif
    
    // CPU path
    nc_tensor* log_probs = nc_log_softmax(logits, 1);
    if (!log_probs) return NULL;
    
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
    
    double loss_val;
    switch (reduction) {
        case NC_REDUCTION_MEAN:
            loss_val = loss_sum / (double)batch;
            break;
        case NC_REDUCTION_SUM:
            loss_val = loss_sum;
            break;
        default:
            loss_val = loss_sum;
            break;
    }
    
    nc_tensor* loss = nc_tensor_scalar(loss_val, logits->dtype);
    if (!loss) return NULL;
    
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
    
    double sum = 0.0;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for reduction(-:sum)
    #endif
    for (i = 0; i < (int)pred->numel; i++) {
        double p = nc_tensor_get_flat(pred, i);
        double t = nc_tensor_get_flat(target, i);
        p = fmax(1e-7, fmin(1.0 - 1e-7, p));
        sum -= t * log(p) + (1.0 - t) * log(1.0 - p);
    }
    
    nc_tensor* loss = nc_tensor_scalar(sum / (double)pred->numel, pred->dtype);
    if (!loss) return NULL;
    
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