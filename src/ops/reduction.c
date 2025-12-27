#include "nocta/ops/reduction.h"
#include "nocta/autograd/node.h"
#include "nocta/autograd/backward.h"
#include <math.h>
#include <float.h>

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
// Backward functions
// ============================================

nc_tensor** nc_backward_sum(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(1, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    nc_tensor* x = inputs[0];
    grads[0] = nc_tensor_empty(x->shape, x->ndim, x->dtype);
    if (!grads[0]) { 
        nc_free(grads); 
        return NULL; 
    }
    
    double g = nc_tensor_get_flat(grad, 0);
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        nc_tensor_set_flat(grads[0], i, g);
    }
    
    return grads;
}

nc_tensor** nc_backward_mean(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(1, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    nc_tensor* x = inputs[0];
    grads[0] = nc_tensor_empty(x->shape, x->ndim, x->dtype);
    if (!grads[0]) { 
        nc_free(grads); 
        return NULL; 
    }
    
    double g = nc_tensor_get_flat(grad, 0) / (double)x->numel;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        nc_tensor_set_flat(grads[0], i, g);
    }
    
    return grads;
}

// ============================================
// Helper: compute output shape after reduction
// ============================================

static void compute_reduced_shape(const nc_tensor* x, int axis, bool keepdim,
                                  size_t* out_shape, size_t* out_ndim) {
    if (axis < 0) axis = (int)x->ndim + axis;
    
    if (keepdim) {
        *out_ndim = x->ndim;
        for (size_t i = 0; i < x->ndim; i++) {
            out_shape[i] = (i == (size_t)axis) ? 1 : x->shape[i];
        }
    } else {
        *out_ndim = x->ndim - 1;
        size_t j = 0;
        for (size_t i = 0; i < x->ndim; i++) {
            if (i != (size_t)axis) {
                out_shape[j++] = x->shape[i];
            }
        }
    }
}

// ============================================
// Sum
// ============================================

nc_tensor* nc_sum_all(nc_tensor* x) {
    NC_CHECK_NULL(x);
    
    double sum = 0.0;
    
#ifdef NOCTA_CUDA_ENABLED
    if (tensor_on_cuda(x) && x->dtype == NC_F32 && nc_tensor_is_contiguous(x)) {
        sum = (double)nc_cuda_sum_f32((const float*)x->storage->cuda_data, x->numel);
        nc_tensor* out = nc_tensor_scalar(sum, x->dtype);
        if (out && nc_grad_enabled() && x->requires_grad) {
            out->requires_grad = true;
            out->is_leaf = false;
            nc_node* node = nc_node_create("sum", nc_backward_sum);
            if (node) {
                nc_node_add_input(node, x);
                nc_node_save_tensor(node, x);
                node->output = out;
                out->grad_fn = node;
            }
        }
        return out;
    }
#endif
    
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for reduction(+:sum)
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        sum += nc_tensor_get_flat(x, i);
    }
    
    nc_tensor* out = nc_tensor_scalar(sum, x->dtype);
    
    if (out && nc_grad_enabled() && x->requires_grad) {
        out->requires_grad = true;
        out->is_leaf = false;
        
        nc_node* node = nc_node_create("sum", nc_backward_sum);
        if (node) {
            nc_node_add_input(node, x);
            nc_node_save_tensor(node, x);
            node->output = out;
            out->grad_fn = node;
        }
    }
    
    return out;
}

nc_tensor* nc_sum(nc_tensor* x, int axis, bool keepdim) {
    NC_CHECK_NULL(x);
    
    if (axis < 0) axis = (int)x->ndim + axis;
    if (axis < 0 || (size_t)axis >= x->ndim) {
        NC_SET_ERROR(NC_ERR_INVALID_AXIS, "Invalid axis %d for ndim %zu", axis, x->ndim);
        return NULL;
    }
    
    size_t out_shape[NC_MAX_DIMS], out_ndim;
    compute_reduced_shape(x, axis, keepdim, out_shape, &out_ndim);
    
    nc_tensor* out = nc_tensor_zeros(out_shape, out_ndim, x->dtype);
    if (!out) return NULL;
    
    size_t axis_size = x->shape[axis];
    size_t outer_size = 1, inner_size = 1;
    
    for (int i = 0; i < axis; i++) outer_size *= x->shape[i];
    for (size_t i = (size_t)axis + 1; i < x->ndim; i++) inner_size *= x->shape[i];
    
    int o;
    (void)o;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (o = 0; o < (int)outer_size; o++) {
        for (size_t in = 0; in < inner_size; in++) {
            double sum = 0.0;
            for (size_t a = 0; a < axis_size; a++) {
                size_t idx = (o * axis_size + a) * inner_size + in;
                sum += nc_tensor_get_flat(x, idx);
            }
            
            size_t out_idx = o * inner_size + in;
            nc_tensor_set_flat(out, out_idx, sum);
        }
    }
    
    return out;
}

// ============================================
// Mean
// ============================================

nc_tensor* nc_mean_all(nc_tensor* x) {
    NC_CHECK_NULL(x);
    
    double sum = 0.0;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for reduction(+:sum)
    #endif
    for (i = 0; i < (int)x->numel; i++) {
        sum += nc_tensor_get_flat(x, i);
    }
    
    nc_tensor* out = nc_tensor_scalar(sum / (double)x->numel, x->dtype);
    
    if (out && nc_grad_enabled() && x->requires_grad) {
        out->requires_grad = true;
        out->is_leaf = false;
        
        nc_node* node = nc_node_create("mean", nc_backward_mean);
        if (node) {
            nc_node_add_input(node, x);
            nc_node_save_tensor(node, x);
            node->output = out;
            out->grad_fn = node;
        }
    }
    
    return out;
}

nc_tensor* nc_mean(nc_tensor* x, int axis, bool keepdim) {
    NC_CHECK_NULL(x);
    
    nc_tensor* s = nc_sum(x, axis, keepdim);
    if (!s) return NULL;
    
    if (axis < 0) axis = (int)x->ndim + axis;
    double n = (double)x->shape[axis];
    
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)s->numel; i++) {
        nc_tensor_set_flat(s, i, nc_tensor_get_flat(s, i) / n);
    }
    
    return s;
}

// ============================================
// Variance / Std
// ============================================

nc_tensor* nc_var(nc_tensor* x, int axis, bool keepdim, bool unbiased) {
    NC_CHECK_NULL(x);
    
    if (axis < 0) axis = (int)x->ndim + axis;
    
    nc_tensor* m = nc_mean(x, axis, true);
    if (!m) return NULL;
    
    size_t out_shape[NC_MAX_DIMS], out_ndim;
    compute_reduced_shape(x, axis, keepdim, out_shape, &out_ndim);
    
    nc_tensor* out = nc_tensor_zeros(out_shape, out_ndim, x->dtype);
    if (!out) { nc_tensor_free(m); return NULL; }
    
    size_t axis_size = x->shape[axis];
    size_t outer_size = 1, inner_size = 1;
    
    for (int i = 0; i < axis; i++) outer_size *= x->shape[i];
    for (size_t i = (size_t)axis + 1; i < x->ndim; i++) inner_size *= x->shape[i];
    
    double denom = unbiased ? (double)(axis_size - 1) : (double)axis_size;
    
    int o;
    (void)o;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (o = 0; o < (int)outer_size; o++) {
        for (size_t in = 0; in < inner_size; in++) {
            double mean_val = nc_tensor_get_flat(m, o * inner_size + in);
            double var_sum = 0.0;
            
            for (size_t a = 0; a < axis_size; a++) {
                size_t idx = (o * axis_size + a) * inner_size + in;
                double diff = nc_tensor_get_flat(x, idx) - mean_val;
                var_sum += diff * diff;
            }
            
            nc_tensor_set_flat(out, o * inner_size + in, var_sum / denom);
        }
    }
    
    nc_tensor_free(m);
    return out;
}

nc_tensor* nc_std(nc_tensor* x, int axis, bool keepdim, bool unbiased) {
    nc_tensor* v = nc_var(x, axis, keepdim, unbiased);
    if (!v) return NULL;
    
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)v->numel; i++) {
        nc_tensor_set_flat(v, i, sqrt(nc_tensor_get_flat(v, i)));
    }
    
    return v;
}

// ============================================
// Max / Min
// ============================================

nc_tensor* nc_max_all(nc_tensor* x) {
    NC_CHECK_NULL(x);
    
#ifdef NOCTA_CUDA_ENABLED
    if (tensor_on_cuda(x) && x->dtype == NC_F32 && nc_tensor_is_contiguous(x)) {
        float max_val = nc_cuda_max_f32((const float*)x->storage->cuda_data, x->numel);
        return nc_tensor_scalar((double)max_val, x->dtype);
    }
#endif
    
    double max_val = -DBL_MAX;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel
    {
        double local_max = -DBL_MAX;
        #pragma omp for nowait
        for (i = 0; i < (int)x->numel; i++) {
            double v = nc_tensor_get_flat(x, i);
            if (v > local_max) local_max = v;
        }
        #pragma omp critical
        {
            if (local_max > max_val) max_val = local_max;
        }
    }
    #else
    for (i = 0; i < (int)x->numel; i++) {
        double v = nc_tensor_get_flat(x, i);
        if (v > max_val) max_val = v;
    }
    #endif
    
    return nc_tensor_scalar(max_val, x->dtype);
}

nc_tensor* nc_min_all(nc_tensor* x) {
    NC_CHECK_NULL(x);
    
    double min_val = DBL_MAX;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel
    {
        double local_min = DBL_MAX;
        #pragma omp for nowait
        for (i = 0; i < (int)x->numel; i++) {
            double v = nc_tensor_get_flat(x, i);
            if (v < local_min) local_min = v;
        }
        #pragma omp critical
        {
            if (local_min < min_val) min_val = local_min;
        }
    }
    #else
    for (i = 0; i < (int)x->numel; i++) {
        double v = nc_tensor_get_flat(x, i);
        if (v < min_val) min_val = v;
    }
    #endif
    
    return nc_tensor_scalar(min_val, x->dtype);
}

nc_tensor* nc_max(nc_tensor* x, int axis, bool keepdim) {
    NC_CHECK_NULL(x);
    
    if (axis < 0) axis = (int)x->ndim + axis;
    
    size_t out_shape[NC_MAX_DIMS], out_ndim;
    compute_reduced_shape(x, axis, keepdim, out_shape, &out_ndim);
    
    nc_tensor* out = nc_tensor_empty(out_shape, out_ndim, x->dtype);
    if (!out) return NULL;
    nc_tensor_fill_(out, -DBL_MAX);
    
    size_t axis_size = x->shape[axis];
    size_t outer_size = 1, inner_size = 1;
    
    for (int i = 0; i < axis; i++) outer_size *= x->shape[i];
    for (size_t i = (size_t)axis + 1; i < x->ndim; i++) inner_size *= x->shape[i];
    
    int o;
    (void)o;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (o = 0; o < (int)outer_size; o++) {
        for (size_t in = 0; in < inner_size; in++) {
            double max_val = -DBL_MAX;
            for (size_t a = 0; a < axis_size; a++) {
                size_t idx = (o * axis_size + a) * inner_size + in;
                double v = nc_tensor_get_flat(x, idx);
                if (v > max_val) max_val = v;
            }
            nc_tensor_set_flat(out, o * inner_size + in, max_val);
        }
    }
    
    return out;
}

nc_tensor* nc_min(nc_tensor* x, int axis, bool keepdim) {
    NC_CHECK_NULL(x);
    
    if (axis < 0) axis = (int)x->ndim + axis;
    
    size_t out_shape[NC_MAX_DIMS], out_ndim;
    compute_reduced_shape(x, axis, keepdim, out_shape, &out_ndim);
    
    nc_tensor* out = nc_tensor_empty(out_shape, out_ndim, x->dtype);
    if (!out) return NULL;
    nc_tensor_fill_(out, DBL_MAX);
    
    size_t axis_size = x->shape[axis];
    size_t outer_size = 1, inner_size = 1;
    
    for (int i = 0; i < axis; i++) outer_size *= x->shape[i];
    for (size_t i = (size_t)axis + 1; i < x->ndim; i++) inner_size *= x->shape[i];
    
    int o;
    (void)o;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (o = 0; o < (int)outer_size; o++) {
        for (size_t in = 0; in < inner_size; in++) {
            double min_val = DBL_MAX;
            for (size_t a = 0; a < axis_size; a++) {
                size_t idx = (o * axis_size + a) * inner_size + in;
                double v = nc_tensor_get_flat(x, idx);
                if (v < min_val) min_val = v;
            }
            nc_tensor_set_flat(out, o * inner_size + in, min_val);
        }
    }
    
    return out;
}

// ============================================
// Argmax / Argmin
// ============================================

nc_tensor* nc_argmax(nc_tensor* x, int axis, bool keepdim) {
    NC_CHECK_NULL(x);
    
    if (axis < 0) axis = (int)x->ndim + axis;
    
    size_t out_shape[NC_MAX_DIMS], out_ndim;
    compute_reduced_shape(x, axis, keepdim, out_shape, &out_ndim);
    
    nc_tensor* out = nc_tensor_zeros(out_shape, out_ndim, NC_I64);
    if (!out) return NULL;
    
    size_t axis_size = x->shape[axis];
    size_t outer_size = 1, inner_size = 1;
    
    for (int i = 0; i < axis; i++) outer_size *= x->shape[i];
    for (size_t i = (size_t)axis + 1; i < x->ndim; i++) inner_size *= x->shape[i];
    
    int o;
    (void)o;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (o = 0; o < (int)outer_size; o++) {
        for (size_t in = 0; in < inner_size; in++) {
            double max_val = -DBL_MAX;
            size_t max_idx = 0;
            
            for (size_t a = 0; a < axis_size; a++) {
                size_t idx = (o * axis_size + a) * inner_size + in;
                double v = nc_tensor_get_flat(x, idx);
                if (v > max_val) {
                    max_val = v;
                    max_idx = a;
                }
            }
            nc_tensor_set_flat(out, o * inner_size + in, (double)max_idx);
        }
    }
    
    return out;
}

nc_tensor* nc_argmin(nc_tensor* x, int axis, bool keepdim) {
    NC_CHECK_NULL(x);
    
    if (axis < 0) axis = (int)x->ndim + axis;
    
    size_t out_shape[NC_MAX_DIMS], out_ndim;
    compute_reduced_shape(x, axis, keepdim, out_shape, &out_ndim);
    
    nc_tensor* out = nc_tensor_zeros(out_shape, out_ndim, NC_I64);
    if (!out) return NULL;
    
    size_t axis_size = x->shape[axis];
    size_t outer_size = 1, inner_size = 1;
    
    for (int i = 0; i < axis; i++) outer_size *= x->shape[i];
    for (size_t i = (size_t)axis + 1; i < x->ndim; i++) inner_size *= x->shape[i];
    
    int o;
    (void)o;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (o = 0; o < (int)outer_size; o++) {
        for (size_t in = 0; in < inner_size; in++) {
            double min_val = DBL_MAX;
            size_t min_idx = 0;
            
            for (size_t a = 0; a < axis_size; a++) {
                size_t idx = (o * axis_size + a) * inner_size + in;
                double v = nc_tensor_get_flat(x, idx);
                if (v < min_val) {
                    min_val = v;
                    min_idx = a;
                }
            }
            nc_tensor_set_flat(out, o * inner_size + in, (double)min_idx);
        }
    }
    
    return out;
}

// ============================================
// Norm
// ============================================

nc_tensor* nc_norm(nc_tensor* x, double p) {
    NC_CHECK_NULL(x);
    
    double sum = 0.0;
    
    if (p == 2.0) {
        int i;
        (void)i;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for reduction(+:sum)
        #endif
        for (i = 0; i < (int)x->numel; i++) {
            double v = nc_tensor_get_flat(x, i);
            sum += v * v;
        }
        return nc_tensor_scalar(sqrt(sum), x->dtype);
    } else if (p == 1.0) {
        int i;
        (void)i;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for reduction(+:sum)
        #endif
        for (i = 0; i < (int)x->numel; i++) {
            sum += fabs(nc_tensor_get_flat(x, i));
        }
        return nc_tensor_scalar(sum, x->dtype);
    } else if (p == INFINITY) {
        double max_val = 0.0;
        int i;
        (void)i;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel
        {
            double local_max = 0.0;
            #pragma omp for nowait
            for (i = 0; i < (int)x->numel; i++) {
                double v = fabs(nc_tensor_get_flat(x, i));
                if (v > local_max) local_max = v;
            }
            #pragma omp critical
            {
                if (local_max > max_val) max_val = local_max;
            }
        }
        #else
        for (i = 0; i < (int)x->numel; i++) {
            double v = fabs(nc_tensor_get_flat(x, i));
            if (v > max_val) max_val = v;
        }
        #endif
        return nc_tensor_scalar(max_val, x->dtype);
    } else {
        int i;
        (void)i;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for reduction(+:sum)
        #endif
        for (i = 0; i < (int)x->numel; i++) {
            sum += pow(fabs(nc_tensor_get_flat(x, i)), p);
        }
        return nc_tensor_scalar(pow(sum, 1.0 / p), x->dtype);
    }
}

// ============================================
// Logsumexp
// ============================================

nc_tensor* nc_logsumexp(nc_tensor* x, int axis, bool keepdim) {
    NC_CHECK_NULL(x);
    
    nc_tensor* max_vals = nc_max(x, axis, true);
    if (!max_vals) return NULL;
    
    if (axis < 0) axis = (int)x->ndim + axis;
    
    size_t out_shape[NC_MAX_DIMS], out_ndim;
    compute_reduced_shape(x, axis, keepdim, out_shape, &out_ndim);
    
    nc_tensor* out = nc_tensor_zeros(out_shape, out_ndim, x->dtype);
    if (!out) { nc_tensor_free(max_vals); return NULL; }
    
    size_t axis_size = x->shape[axis];
    size_t outer_size = 1, inner_size = 1;
    
    for (int i = 0; i < axis; i++) outer_size *= x->shape[i];
    for (size_t i = (size_t)axis + 1; i < x->ndim; i++) inner_size *= x->shape[i];
    
    int o;
    (void)o;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (o = 0; o < (int)outer_size; o++) {
        for (size_t in = 0; in < inner_size; in++) {
            double max_val = nc_tensor_get_flat(max_vals, o * inner_size + in);
            double sum = 0.0;
            
            for (size_t a = 0; a < axis_size; a++) {
                size_t idx = (o * axis_size + a) * inner_size + in;
                sum += exp(nc_tensor_get_flat(x, idx) - max_val);
            }
            
            nc_tensor_set_flat(out, o * inner_size + in, log(sum) + max_val);
        }
    }
    
    nc_tensor_free(max_vals);
    return out;
}