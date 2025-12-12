#include "nocta/ops/arithmetic.h"
#include "nocta/autograd/node.h"
#include "nocta/autograd/backward.h"
#include <math.h>
#include <string.h>

#ifdef NOCTA_OPENMP_ENABLED
#include <omp.h>
#endif

// ============================================
// Helper: compute broadcast shape
// ============================================

static int broadcast_shape(const nc_tensor* a, const nc_tensor* b, 
                           size_t* out_shape, size_t* out_ndim) {
    size_t max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    *out_ndim = max_ndim;
    
    for (size_t i = 0; i < max_ndim; i++) {
        size_t da = i < a->ndim ? a->shape[a->ndim - 1 - i] : 1;
        size_t db = i < b->ndim ? b->shape[b->ndim - 1 - i] : 1;
        
        if (da != db && da != 1 && db != 1) {
            return -1; // Not broadcastable
        }
        out_shape[max_ndim - 1 - i] = da > db ? da : db;
    }
    return 0;
}

// Get broadcasted index
static size_t broadcast_index(const nc_tensor* t, const size_t* out_idx, size_t out_ndim) {
    size_t idx = 0;
    for (size_t i = 0; i < t->ndim; i++) {
        size_t out_i = out_ndim - t->ndim + i;
        size_t ti = out_idx[out_i] % t->shape[i];
        idx += ti * t->strides[i];
    }
    return idx + t->offset;
}

// ============================================
// Backward functions
// ============================================

// Helper: reduce gradient to match input shape (for broadcasting)
static nc_tensor* reduce_grad_to_shape(nc_tensor* grad, nc_tensor* target) {
    if (!grad || !target) return NULL;
    
    // If same shape, just clone
    if (nc_tensor_shape_eq(grad, target)) {
        return nc_tensor_clone(grad);
    }
    
    // Need to sum over broadcasted dimensions
    nc_tensor* result = nc_tensor_zeros(target->shape, target->ndim, target->dtype);
    if (!result) return NULL;
    
    // Simple approach: iterate and accumulate
    size_t target_numel = target->numel;
    size_t grad_numel = grad->numel;
    
    if (target_numel == 1) {
        // Scalar target - sum all gradients
        double sum = 0;
        int i;
        (void)i;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for reduction(+:sum)
        #endif
        for (i = 0; i < (int)grad_numel; i++) {
            sum += nc_tensor_get_flat(grad, i);
        }
        nc_tensor_set_flat(result, 0, sum);
    } else if (target->ndim == 1 && grad->ndim == 2) {
        // Bias case: (N,) from (batch, N) - sum over batch
        size_t batch = grad->shape[0];
        size_t n = grad->shape[1];
        int j;
        (void)j;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for
        #endif
        for (j = 0; j < (int)n; j++) {
            double sum = 0;
            for (size_t i = 0; i < batch; i++) {
                sum += nc_tensor_get2(grad, i, j);
            }
            nc_tensor_set1(result, j, sum);
        }
    } else {
        // General case - copy what fits
        // This part is tricky to parallelize generally without atomic adds or complex logic
        // Leaving serial for safety in general broadcasting case
        for (size_t i = 0; i < target_numel && i < grad_numel; i++) {
            nc_tensor_set_flat(result, i, nc_tensor_get_flat(grad, i));
        }
    }
    
    return result;
}

nc_tensor** nc_backward_add(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(2, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    // d(a+b)/da = 1, d(a+b)/db = 1
    grads[0] = reduce_grad_to_shape(grad, inputs[0]);
    grads[1] = reduce_grad_to_shape(grad, inputs[1]);
    
    return grads;
}

nc_tensor** nc_backward_sub(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(2, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    grads[0] = reduce_grad_to_shape(grad, inputs[0]);
    
    // d(a-b)/db = -1
    nc_tensor* neg = nc_neg(grad);
    grads[1] = reduce_grad_to_shape(neg, inputs[1]);
    nc_tensor_free(neg);
    
    return grads;
}

nc_tensor** nc_backward_mul(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(2, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    // d(a*b)/da = b, d(a*b)/db = a
    nc_tensor* grad_a = nc_mul(grad, inputs[1]);
    nc_tensor* grad_b = nc_mul(grad, inputs[0]);
    
    grads[0] = reduce_grad_to_shape(grad_a, inputs[0]);
    grads[1] = reduce_grad_to_shape(grad_b, inputs[1]);
    
    nc_tensor_free(grad_a);
    nc_tensor_free(grad_b);
    
    return grads;
}

nc_tensor** nc_backward_div(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(2, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    // d(a/b)/da = 1/b
    // d(a/b)/db = -a/b^2
    nc_tensor* grad_a = nc_div(grad, inputs[1]);
    grads[0] = reduce_grad_to_shape(grad_a, inputs[0]);
    nc_tensor_free(grad_a);
    
    nc_tensor* b_sq = nc_mul(inputs[1], inputs[1]);
    nc_tensor* a_div_bsq = nc_div(inputs[0], b_sq);
    nc_tensor* neg = nc_neg(a_div_bsq);
    nc_tensor* grad_b = nc_mul(grad, neg);
    grads[1] = reduce_grad_to_shape(grad_b, inputs[1]);
    
    nc_tensor_free(b_sq);
    nc_tensor_free(a_div_bsq);
    nc_tensor_free(neg);
    nc_tensor_free(grad_b);
    
    return grads;
}

nc_tensor** nc_backward_neg(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)inputs; (void)n;
    nc_tensor** grads = nc_calloc(1, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    grads[0] = nc_neg(grad);
    return grads;
}

nc_tensor** nc_backward_exp(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(1, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    // d(e^x)/dx = e^x
    nc_tensor* exp_x = nc_exp(inputs[0]);
    grads[0] = nc_mul(grad, exp_x);
    nc_tensor_free(exp_x);
    
    return grads;
}

nc_tensor** nc_backward_log(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(1, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    // d(ln(x))/dx = 1/x
    grads[0] = nc_div(grad, inputs[0]);
    return grads;
}

nc_tensor** nc_backward_pow(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(2, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    nc_tensor* a = inputs[0];
    nc_tensor* b = inputs[1];
    
    // d(a^b)/da = b * a^(b-1)
    nc_tensor* one = nc_tensor_full(b->shape, b->ndim, b->dtype, 1.0);
    nc_tensor* b_m1 = nc_sub(b, one);
    nc_tensor* a_pow_bm1 = nc_pow(a, b_m1);
    nc_tensor* da = nc_mul(b, a_pow_bm1);
    nc_tensor* grad_a = nc_mul(grad, da);
    grads[0] = reduce_grad_to_shape(grad_a, a);
    
    // d(a^b)/db = a^b * ln(a)
    nc_tensor* a_pow_b = nc_pow(a, b);
    nc_tensor* ln_a = nc_log(a);
    nc_tensor* db = nc_mul(a_pow_b, ln_a);
    nc_tensor* grad_b = nc_mul(grad, db);
    grads[1] = reduce_grad_to_shape(grad_b, b);
    
    nc_tensor_free(one);
    nc_tensor_free(b_m1);
    nc_tensor_free(a_pow_bm1);
    nc_tensor_free(da);
    nc_tensor_free(grad_a);
    nc_tensor_free(a_pow_b);
    nc_tensor_free(ln_a);
    nc_tensor_free(db);
    nc_tensor_free(grad_b);
    
    return grads;
}

// ============================================
// Setup autograd for binary op
// ============================================

static void setup_grad_binary(nc_tensor* out, nc_tensor* a, nc_tensor* b,
                              const char* op, nc_backward_fn backward) {
    if (!nc_grad_enabled()) return;
    if (!a->requires_grad && !b->requires_grad) return;
    
    out->requires_grad = true;
    out->is_leaf = false;
    
    nc_node* node = nc_node_create(op, backward);
    if (!node) return;
    
    nc_node_add_input(node, a);
    nc_node_add_input(node, b);
    nc_node_save_tensor(node, a);
    nc_node_save_tensor(node, b);
    node->output = out;
    
    out->grad_fn = node;
}

static void setup_grad_unary(nc_tensor* out, nc_tensor* a,
                             const char* op, nc_backward_fn backward) {
    if (!nc_grad_enabled()) return;
    if (!a->requires_grad) return;
    
    out->requires_grad = true;
    out->is_leaf = false;
    
    nc_node* node = nc_node_create(op, backward);
    if (!node) return;
    
    nc_node_add_input(node, a);
    nc_node_save_tensor(node, a);
    node->output = out;
    
    out->grad_fn = node;
}

// ============================================
// Binary operations
// ============================================

#define BINARY_OP(name, op, backward_fn) \
nc_tensor* nc_##name(nc_tensor* a, nc_tensor* b) { \
    NC_CHECK_NULL(a); NC_CHECK_NULL(b); \
    \
    size_t out_shape[NC_MAX_DIMS], out_ndim; \
    if (broadcast_shape(a, b, out_shape, &out_ndim) < 0) { \
        NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, "Cannot broadcast shapes"); \
        return NULL; \
    } \
    \
    nc_dtype dtype = nc_dtype_promote(a->dtype, b->dtype); \
    nc_tensor* out = nc_tensor_empty(out_shape, out_ndim, dtype); \
    if (!out) return NULL; \
    \
    if (nc_tensor_shape_eq(a, b) && nc_tensor_is_contiguous(a) && nc_tensor_is_contiguous(b)) { \
        /* Fast path for contiguous same-shape tensors */ \
        int i; \
        (void)i; \
        _Pragma("omp parallel for") \
        for (i = 0; i < (int)out->numel; i++) { \
            double va = nc_tensor_get_flat(a, i); \
            double vb = nc_tensor_get_flat(b, i); \
            nc_tensor_set_flat(out, i, va op vb); \
        } \
    } else { \
        /* Slow path with broadcasting */ \
        size_t idx[NC_MAX_DIMS] = {0}; \
        for (size_t i = 0; i < out->numel; i++) { \
            size_t ai = broadcast_index(a, idx, out_ndim); \
            size_t bi = broadcast_index(b, idx, out_ndim); \
            double va = nc_tensor_get_flat(a, ai - a->offset); \
            double vb = nc_tensor_get_flat(b, bi - b->offset); \
            nc_tensor_set_flat(out, i, va op vb); \
            \
            for (int d = (int)out_ndim - 1; d >= 0; d--) { \
                if (++idx[d] < out_shape[d]) break; \
                idx[d] = 0; \
            } \
        } \
    } \
    \
    setup_grad_binary(out, a, b, #name, backward_fn); \
    return out; \
}

BINARY_OP(add, +, nc_backward_add)
BINARY_OP(sub, -, nc_backward_sub)
BINARY_OP(mul, *, nc_backward_mul)
BINARY_OP(div, /, nc_backward_div)

nc_tensor* nc_pow(nc_tensor* a, nc_tensor* b) {
    NC_CHECK_NULL(a); NC_CHECK_NULL(b);
    
    size_t out_shape[NC_MAX_DIMS], out_ndim;
    if (broadcast_shape(a, b, out_shape, &out_ndim) < 0) {
        NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, "Cannot broadcast shapes");
        return NULL;
    }
    
    nc_dtype dtype = nc_dtype_promote(a->dtype, b->dtype);
    nc_tensor* out = nc_tensor_empty(out_shape, out_ndim, dtype);
    if (!out) return NULL;
    
    if (nc_tensor_shape_eq(a, b) && nc_tensor_is_contiguous(a) && nc_tensor_is_contiguous(b)) {
        int i;
        (void)i;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for
        #endif
        for (i = 0; i < (int)out->numel; i++) {
            double va = nc_tensor_get_flat(a, i);
            double vb = nc_tensor_get_flat(b, i);
            nc_tensor_set_flat(out, i, pow(va, vb));
        }
    } else {
        size_t idx[NC_MAX_DIMS] = {0};
        for (size_t i = 0; i < out->numel; i++) {
            size_t ai = broadcast_index(a, idx, out_ndim);
            size_t bi = broadcast_index(b, idx, out_ndim);
            double va = nc_tensor_get_flat(a, ai - a->offset);
            double vb = nc_tensor_get_flat(b, bi - b->offset);
            nc_tensor_set_flat(out, i, pow(va, vb));
            
            for (int d = (int)out_ndim - 1; d >= 0; d--) {
                if (++idx[d] < out_shape[d]) break;
                idx[d] = 0;
            }
        }
    }
    
    setup_grad_binary(out, a, b, "pow", nc_backward_pow);
    return out;
}

// ============================================
// Scalar operations
// ============================================

nc_tensor* nc_add_scalar(nc_tensor* a, double s) {
    NC_CHECK_NULL(a);
    nc_tensor* out = nc_tensor_clone(a);
    if (!out) return NULL;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)out->numel; i++) {
        nc_tensor_set_flat(out, i, nc_tensor_get_flat(a, i) + s);
    }
    return out;
}

nc_tensor* nc_sub_scalar(nc_tensor* a, double s) {
    return nc_add_scalar(a, -s);
}

nc_tensor* nc_mul_scalar(nc_tensor* a, double s) {
    NC_CHECK_NULL(a);
    nc_tensor* out = nc_tensor_clone(a);
    if (!out) return NULL;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)out->numel; i++) {
        nc_tensor_set_flat(out, i, nc_tensor_get_flat(a, i) * s);
    }
    return out;
}

nc_tensor* nc_div_scalar(nc_tensor* a, double s) {
    return nc_mul_scalar(a, 1.0 / s);
}

nc_tensor* nc_pow_scalar(nc_tensor* a, double s) {
    NC_CHECK_NULL(a);
    nc_tensor* out = nc_tensor_clone(a);
    if (!out) return NULL;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)out->numel; i++) {
        nc_tensor_set_flat(out, i, pow(nc_tensor_get_flat(a, i), s));
    }
    return out;
}

// ============================================
// Unary operations
// ============================================

#define UNARY_OP(name, func, backward_fn) \
nc_tensor* nc_##name(nc_tensor* a) { \
    NC_CHECK_NULL(a); \
    nc_tensor* out = nc_tensor_empty(a->shape, a->ndim, a->dtype); \
    if (!out) return NULL; \
    int i; \
    (void)i; \
    _Pragma("omp parallel for") \
    for (i = 0; i < (int)a->numel; i++) { \
        nc_tensor_set_flat(out, i, func(nc_tensor_get_flat(a, i))); \
    } \
    if (backward_fn) setup_grad_unary(out, a, #name, backward_fn); \
    return out; \
}

static double neg_fn(double x) { return -x; }
static double abs_fn(double x) { return fabs(x); }
static double square_fn(double x) { return x * x; }
static double sign_fn(double x) { return (x > 0) - (x < 0); }

UNARY_OP(neg, neg_fn, nc_backward_neg)
UNARY_OP(abs, abs_fn, NULL)
UNARY_OP(square, square_fn, NULL)
UNARY_OP(sqrt, sqrt, NULL)
UNARY_OP(exp, exp, nc_backward_exp)
UNARY_OP(log, log, nc_backward_log)
UNARY_OP(log10, log10, NULL)
UNARY_OP(log2, log2, NULL)
UNARY_OP(sin, sin, NULL)
UNARY_OP(cos, cos, NULL)
UNARY_OP(tan, tan, NULL)
UNARY_OP(asin, asin, NULL)
UNARY_OP(acos, acos, NULL)
UNARY_OP(atan, atan, NULL)
UNARY_OP(sinh, sinh, NULL)
UNARY_OP(cosh, cosh, NULL)
UNARY_OP(sign, sign_fn, NULL)
UNARY_OP(floor, floor, NULL)
UNARY_OP(ceil, ceil, NULL)
UNARY_OP(round, round, NULL)

nc_tensor* nc_clamp(nc_tensor* a, double min_val, double max_val) {
    NC_CHECK_NULL(a);
    nc_tensor* out = nc_tensor_clone(a);
    if (!out) return NULL;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)a->numel; i++) {
        double v = nc_tensor_get_flat(a, i);
        if (v < min_val) v = min_val;
        if (v > max_val) v = max_val;
        nc_tensor_set_flat(out, i, v);
    }
    return out;
}

nc_tensor* nc_reciprocal(nc_tensor* a) {
    NC_CHECK_NULL(a);
    nc_tensor* out = nc_tensor_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)a->numel; i++) {
        nc_tensor_set_flat(out, i, 1.0 / nc_tensor_get_flat(a, i));
    }
    return out;
}

// ============================================
// In-place operations
// ============================================

void nc_add_(nc_tensor* a, nc_tensor* b) {
    if (!a || !b || a->numel != b->numel) return;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)a->numel; i++) {
        nc_tensor_set_flat(a, i, nc_tensor_get_flat(a, i) + nc_tensor_get_flat(b, i));
    }
}

void nc_mul_(nc_tensor* a, nc_tensor* b) {
    if (!a || !b || a->numel != b->numel) return;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)a->numel; i++) {
        nc_tensor_set_flat(a, i, nc_tensor_get_flat(a, i) * nc_tensor_get_flat(b, i));
    }
}

void nc_add_scalar_(nc_tensor* a, double s) {
    if (!a) return;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)a->numel; i++) {
        nc_tensor_set_flat(a, i, nc_tensor_get_flat(a, i) + s);
    }
}

void nc_mul_scalar_(nc_tensor* a, double s) {
    if (!a) return;
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)a->numel; i++) {
        nc_tensor_set_flat(a, i, nc_tensor_get_flat(a, i) * s);
    }
}

// ============================================
// Comparison operations
// ============================================

#define CMP_OP(name, op) \
nc_tensor* nc_##name(nc_tensor* a, nc_tensor* b) { \
    NC_CHECK_NULL(a); NC_CHECK_NULL(b); \
    size_t out_shape[NC_MAX_DIMS], out_ndim; \
    if (broadcast_shape(a, b, out_shape, &out_ndim) < 0) return NULL; \
    nc_tensor* out = nc_tensor_empty(out_shape, out_ndim, NC_BOOL); \
    if (!out) return NULL; \
    if (nc_tensor_shape_eq(a, b) && nc_tensor_is_contiguous(a) && nc_tensor_is_contiguous(b)) { \
        int i; \
        (void)i; \
        _Pragma("omp parallel for") \
        for (i = 0; i < (int)out->numel; i++) { \
            double va = nc_tensor_get_flat(a, i); \
            double vb = nc_tensor_get_flat(b, i); \
            nc_tensor_set_flat(out, i, va op vb); \
        } \
    } else { \
        size_t idx[NC_MAX_DIMS] = {0}; \
        for (size_t i = 0; i < out->numel; i++) { \
            size_t ai = broadcast_index(a, idx, out_ndim); \
            size_t bi = broadcast_index(b, idx, out_ndim); \
            double va = nc_tensor_get_flat(a, ai - a->offset); \
            double vb = nc_tensor_get_flat(b, bi - b->offset); \
            nc_tensor_set_flat(out, i, va op vb); \
            for (int d = (int)out_ndim - 1; d >= 0; d--) { \
                if (++idx[d] < out_shape[d]) break; \
                idx[d] = 0; \
            } \
        } \
    } \
    return out; \
}

CMP_OP(eq, ==)
CMP_OP(ne, !=)
CMP_OP(lt, <)
CMP_OP(le, <=)
CMP_OP(gt, >)
CMP_OP(ge, >=)