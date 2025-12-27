#include "nocta/ops/activation.h"
#include "nocta/ops/arithmetic.h"
#include "nocta/autograd/node.h"
#include "nocta/autograd/backward.h"
#include <math.h>

#ifdef NOCTA_OPENMP_ENABLED
#include <omp.h>
#endif

#ifdef NOCTA_CUDA_ENABLED
#include "nocta/cuda/cuda_kernels.h"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
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
// Backward functions
// ============================================

nc_tensor** nc_backward_relu(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(1, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    nc_tensor* x = inputs[0];
    grads[0] = nc_tensor_empty(x->shape, x->ndim, x->dtype);
    if (!grads[0]) { free(grads); return NULL; }
    
#ifdef NOCTA_CUDA_ENABLED
    if (tensor_on_cuda(grad) && tensor_on_cuda(x)) {
        nc_storage_to_device(grads[0]->storage, NC_DEVICE_CUDA);
        if (grads[0]->storage->cuda_data) {
            nc_cuda_relu_backward_f32(
                (float*)grads[0]->storage->cuda_data,
                (const float*)grad->storage->cuda_data,
                (const float*)x->storage->cuda_data,
                x->numel
            );
            return grads;
        }
    }
#endif
    
    // d(relu)/dx = 1 if x > 0, else 0
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        double xi = nc_tensor_get_flat(x, i);
        double gi = nc_tensor_get_flat(grad, i);
        nc_tensor_set_flat(grads[0], i, xi > 0 ? gi : 0);
    }
    
    return grads;
}

nc_tensor** nc_backward_sigmoid(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(1, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    // inputs[0] = x (original input)
    // inputs[1] = sigmoid output (saved during forward)
    nc_tensor* sig_out = (n > 1 && inputs[1]) ? inputs[1] : NULL;
    nc_tensor* x = inputs[0];
    
    grads[0] = nc_tensor_empty(x->shape, x->ndim, x->dtype);
    if (!grads[0]) { free(grads); return NULL; }
    
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        double s;
        if (sig_out) {
            s = nc_tensor_get_flat(sig_out, i);
        } else {
            double xi = nc_tensor_get_flat(x, i);
            s = 1.0 / (1.0 + exp(-xi));
        }
        double gi = nc_tensor_get_flat(grad, i);
        // d(sigmoid)/dx = sigmoid * (1 - sigmoid)
        nc_tensor_set_flat(grads[0], i, gi * s * (1.0 - s));
    }
    
    return grads;
}

nc_tensor** nc_backward_tanh(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(1, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    nc_tensor* x = inputs[0];
    grads[0] = nc_tensor_empty(x->shape, x->ndim, x->dtype);
    if (!grads[0]) { free(grads); return NULL; }
    
    // d(tanh)/dx = 1 - tanh(x)^2
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        double t = tanh(nc_tensor_get_flat(x, i));
        double gi = nc_tensor_get_flat(grad, i);
        nc_tensor_set_flat(grads[0], i, gi * (1.0 - t * t));
    }
    
    return grads;
}

nc_tensor** nc_backward_softmax(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(1, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    // inputs[1] = softmax output (saved)
    nc_tensor* s = (n > 1) ? inputs[1] : inputs[0];
    
    grads[0] = nc_tensor_empty(grad->shape, grad->ndim, grad->dtype);
    if (!grads[0]) { free(grads); return NULL; }
    
    // Simplified: assume last dim softmax
    size_t batch_size = (grad->ndim > 1) ? grad->numel / grad->shape[grad->ndim - 1] : 1;
    size_t dim_size = grad->shape[grad->ndim - 1];
    
    int b;
    (void)b;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (b = 0; b < (int)batch_size; b++) {
        double dot = 0.0;
        for (size_t i = 0; i < dim_size; i++) {
            size_t idx = b * dim_size + i;
            dot += nc_tensor_get_flat(grad, idx) * nc_tensor_get_flat(s, idx);
        }
        
        for (size_t i = 0; i < dim_size; i++) {
            size_t idx = b * dim_size + i;
            double si = nc_tensor_get_flat(s, idx);
            double gi = nc_tensor_get_flat(grad, idx);
            nc_tensor_set_flat(grads[0], idx, si * (gi - dot));
        }
    }
    
    return grads;
}

// ============================================
// Helper: setup autograd
// ============================================

static void setup_grad(nc_tensor* out, nc_tensor* x, const char* op,
                       nc_backward_fn backward, nc_tensor* extra_saved) {
    if (!nc_grad_enabled() || !x->requires_grad) return;
    
    out->requires_grad = true;
    out->is_leaf = false;
    
    nc_node* node = nc_node_create(op, backward);
    if (!node) return;
    
    nc_node_add_input(node, x);
    nc_node_save_tensor(node, x);
    if (extra_saved) {
        nc_node_save_tensor(node, extra_saved);
    }
    node->output = out;
    out->grad_fn = node;
}

// ============================================
// Activations
// ============================================

nc_tensor* nc_relu(nc_tensor* x) {
    NC_CHECK_NULL(x);
    
    nc_tensor* out = nc_tensor_empty(x->shape, x->ndim, x->dtype);
    if (!out) return NULL;
    
#ifdef NOCTA_CUDA_ENABLED
    if (tensor_on_cuda(x) && x->dtype == NC_F32 && nc_tensor_is_contiguous(x)) {
        nc_storage_to_device(out->storage, NC_DEVICE_CUDA);
        nc_cuda_relu_f32((float*)out->storage->cuda_data,
                         (const float*)x->storage->cuda_data,
                         out->numel);
        setup_grad(out, x, "relu", nc_backward_relu, NULL);
        return out;
    }
#endif
    
    // CPU path
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        double v = nc_tensor_get_flat(x, i);
        nc_tensor_set_flat(out, i, v > 0 ? v : 0);
    }
    
    setup_grad(out, x, "relu", nc_backward_relu, NULL);
    return out;
}

nc_tensor* nc_leaky_relu(nc_tensor* x, double alpha) {
    NC_CHECK_NULL(x);
    
    nc_tensor* out = nc_tensor_empty(x->shape, x->ndim, x->dtype);
    if (!out) return NULL;
    
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        double v = nc_tensor_get_flat(x, i);
        nc_tensor_set_flat(out, i, v > 0 ? v : alpha * v);
    }
    
    return out;
}

nc_tensor* nc_elu(nc_tensor* x, double alpha) {
    NC_CHECK_NULL(x);
    
    nc_tensor* out = nc_tensor_empty(x->shape, x->ndim, x->dtype);
    if (!out) return NULL;
    
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        double v = nc_tensor_get_flat(x, i);
        nc_tensor_set_flat(out, i, v > 0 ? v : alpha * (exp(v) - 1));
    }
    
    return out;
}

nc_tensor* nc_selu(nc_tensor* x) {
    // SELU constants
    const double alpha = 1.6732632423543772;
    const double scale = 1.0507009873554805;
    
    NC_CHECK_NULL(x);
    
    nc_tensor* out = nc_tensor_empty(x->shape, x->ndim, x->dtype);
    if (!out) return NULL;
    
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        double v = nc_tensor_get_flat(x, i);
        double r = v > 0 ? v : alpha * (exp(v) - 1);
        nc_tensor_set_flat(out, i, scale * r);
    }
    
    return out;
}

nc_tensor* nc_gelu(nc_tensor* x) {
    NC_CHECK_NULL(x);
    
    nc_tensor* out = nc_tensor_empty(x->shape, x->ndim, x->dtype);
    if (!out) return NULL;
    
    const double c = sqrt(2.0 / M_PI);
    
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        double v = nc_tensor_get_flat(x, i);
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        double inner = c * (v + 0.044715 * v * v * v);
        nc_tensor_set_flat(out, i, 0.5 * v * (1.0 + tanh(inner)));
    }
    
    return out;
}

nc_tensor* nc_sigmoid(nc_tensor* x) {
    NC_CHECK_NULL(x);
    
    nc_tensor* out = nc_tensor_empty(x->shape, x->ndim, x->dtype);
    if (!out) return NULL;
    
#ifdef NOCTA_CUDA_ENABLED
    if (tensor_on_cuda(x) && x->dtype == NC_F32 && nc_tensor_is_contiguous(x)) {
        nc_storage_to_device(out->storage, NC_DEVICE_CUDA);
        nc_cuda_sigmoid_f32((float*)out->storage->cuda_data,
                            (const float*)x->storage->cuda_data,
                            out->numel);
        setup_grad(out, x, "sigmoid", nc_backward_sigmoid, out);
        return out;
    }
#endif
    
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        double v = nc_tensor_get_flat(x, i);
        nc_tensor_set_flat(out, i, 1.0 / (1.0 + exp(-v)));
    }
    
    setup_grad(out, x, "sigmoid", nc_backward_sigmoid, out);
    return out;
}

nc_tensor* nc_tanh_act(nc_tensor* x) {
    NC_CHECK_NULL(x);
    
    nc_tensor* out = nc_tensor_empty(x->shape, x->ndim, x->dtype);
    if (!out) return NULL;
    
#ifdef NOCTA_CUDA_ENABLED
    if (tensor_on_cuda(x) && x->dtype == NC_F32 && nc_tensor_is_contiguous(x)) {
        nc_storage_to_device(out->storage, NC_DEVICE_CUDA);
        nc_cuda_tanh_f32((float*)out->storage->cuda_data,
                         (const float*)x->storage->cuda_data,
                         out->numel);
        setup_grad(out, x, "tanh", nc_backward_tanh, NULL);
        return out;
    }
#endif
    
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        nc_tensor_set_flat(out, i, tanh(nc_tensor_get_flat(x, i)));
    }
    
    setup_grad(out, x, "tanh", nc_backward_tanh, NULL);
    return out;
}

nc_tensor* nc_softmax(nc_tensor* x, int dim) {
    NC_CHECK_NULL(x);
    
    if (dim < 0) dim = (int)x->ndim + dim;
    if (dim < 0 || (size_t)dim >= x->ndim) {
        NC_SET_ERROR(NC_ERR_INVALID_AXIS, "Invalid softmax dim");
        return NULL;
    }
    
    nc_tensor* out = nc_tensor_empty(x->shape, x->ndim, x->dtype);
    if (!out) return NULL;
    
#ifdef NOCTA_CUDA_ENABLED
    // Fast path: 2D tensor, softmax on last dim, contiguous
    if (tensor_on_cuda(x) && x->dtype == NC_F32 && nc_tensor_is_contiguous(x) &&
        x->ndim == 2 && (size_t)dim == x->ndim - 1) {
        nc_storage_to_device(out->storage, NC_DEVICE_CUDA);
        nc_cuda_softmax_f32((float*)out->storage->cuda_data,
                            (const float*)x->storage->cuda_data,
                            x->shape[0], x->shape[1]);
        setup_grad(out, x, "softmax", nc_backward_softmax, out);
        return out;
    }
#endif
    
    size_t dim_size = x->shape[dim];
    size_t outer_size = 1, inner_size = 1;
    
    for (size_t i = 0; i < (size_t)dim; i++) outer_size *= x->shape[i];
    for (size_t i = (size_t)dim + 1; i < x->ndim; i++) inner_size *= x->shape[i];
    
    int o;
    (void)o;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (o = 0; o < (int)outer_size; o++) {
        for (size_t in = 0; in < inner_size; in++) {
            double max_val = -INFINITY;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = (o * dim_size + d) * inner_size + in;
                double v = nc_tensor_get_flat(x, idx);
                if (v > max_val) max_val = v;
            }
            
            double sum = 0.0;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = (o * dim_size + d) * inner_size + in;
                double v = exp(nc_tensor_get_flat(x, idx) - max_val);
                nc_tensor_set_flat(out, idx, v);
                sum += v;
            }
            
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = (o * dim_size + d) * inner_size + in;
                nc_tensor_set_flat(out, idx, nc_tensor_get_flat(out, idx) / sum);
            }
        }
    }
    
    setup_grad(out, x, "softmax", nc_backward_softmax, out);
    return out;
}

nc_tensor* nc_log_softmax(nc_tensor* x, int dim) {
    NC_CHECK_NULL(x);
    
    if (dim < 0) dim = (int)x->ndim + dim;
    if (dim < 0 || (size_t)dim >= x->ndim) {
        NC_SET_ERROR(NC_ERR_INVALID_AXIS, "Invalid log_softmax dim");
        return NULL;
    }
    
    nc_tensor* out = nc_tensor_empty(x->shape, x->ndim, x->dtype);
    if (!out) return NULL;
    
    size_t dim_size = x->shape[dim];
    size_t outer_size = 1, inner_size = 1;
    
    for (size_t i = 0; i < (size_t)dim; i++) outer_size *= x->shape[i];
    for (size_t i = (size_t)dim + 1; i < x->ndim; i++) inner_size *= x->shape[i];
    
    int o;
    (void)o;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (o = 0; o < (int)outer_size; o++) {
        for (size_t in = 0; in < inner_size; in++) {
            // Find max
            double max_val = -INFINITY;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = (o * dim_size + d) * inner_size + in;
                double v = nc_tensor_get_flat(x, idx);
                if (v > max_val) max_val = v;
            }
            
            // Compute log-sum-exp
            double sum = 0.0;
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = (o * dim_size + d) * inner_size + in;
                sum += exp(nc_tensor_get_flat(x, idx) - max_val);
            }
            double log_sum = log(sum) + max_val;
            
            // log_softmax = x - log_sum_exp
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = (o * dim_size + d) * inner_size + in;
                nc_tensor_set_flat(out, idx, nc_tensor_get_flat(x, idx) - log_sum);
            }
        }
    }
    
    return out;
}

nc_tensor* nc_swish(nc_tensor* x) {
    NC_CHECK_NULL(x);
    
    nc_tensor* out = nc_tensor_empty(x->shape, x->ndim, x->dtype);
    if (!out) return NULL;
    
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        double v = nc_tensor_get_flat(x, i);
        nc_tensor_set_flat(out, i, v / (1.0 + exp(-v)));
    }
    
    return out;
}

nc_tensor* nc_silu(nc_tensor* x) {
    return nc_swish(x);
}

nc_tensor* nc_mish(nc_tensor* x) {
    NC_CHECK_NULL(x);
    
    nc_tensor* out = nc_tensor_empty(x->shape, x->ndim, x->dtype);
    if (!out) return NULL;
    
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        double v = nc_tensor_get_flat(x, i);
        double sp = log(1.0 + exp(v));  // softplus
        nc_tensor_set_flat(out, i, v * tanh(sp));
    }
    
    return out;
}

nc_tensor* nc_relu6(nc_tensor* x) {
    return nc_clamp(nc_relu(x), 0, 6);
}

// ============================================
// In-place
// ============================================

void nc_relu_(nc_tensor* x) {
    if (!x) return;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        double v = nc_tensor_get_flat(x, i);
        if (v < 0) nc_tensor_set_flat(x, i, 0);
    }
}

void nc_sigmoid_(nc_tensor* x) {
    if (!x) return;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        double v = nc_tensor_get_flat(x, i);
        nc_tensor_set_flat(x, i, 1.0 / (1.0 + exp(-v)));
    }
}

void nc_tanh_(nc_tensor* x) {
    if (!x) return;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        nc_tensor_set_flat(x, i, tanh(nc_tensor_get_flat(x, i)));
    }
}