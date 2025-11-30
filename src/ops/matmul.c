#include "nocta/ops/matmul.h"
#include "nocta/autograd/node.h"
#include "nocta/autograd/backward.h"
#include <string.h>

// ============================================
// Backward
// ============================================

nc_tensor** nc_backward_matmul(nc_tensor* grad, nc_tensor** inputs, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(2, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    nc_tensor* a = inputs[0];  // (M, K)
    nc_tensor* b = inputs[1];  // (K, N)
    // grad is (M, N)
    
    // grad_a = grad @ b^T  -> (M, N) @ (N, K) = (M, K)
    // grad_b = a^T @ grad  -> (K, M) @ (M, N) = (K, N)
    
    if (b->ndim == 2) {
        nc_tensor* bt = nc_tensor_t(b);
        grads[0] = nc_matmul(grad, bt);
        nc_tensor_free(bt);
    } else {
        grads[0] = nc_tensor_clone(grad);
    }
    
    if (a->ndim == 2) {
        nc_tensor* at = nc_tensor_t(a);
        grads[1] = nc_matmul(at, grad);
        nc_tensor_free(at);
    } else {
        grads[1] = nc_tensor_clone(grad);
    }
    
    return grads;
}

// ============================================
// Core matmul implementation
// ============================================

// Simple 2D matmul: (M, K) @ (K, N) -> (M, N)
static nc_tensor* matmul_2d(nc_tensor* a, nc_tensor* b) {
    size_t M = a->shape[0];
    size_t K = a->shape[1];
    size_t N = b->shape[1];
    
    if (b->shape[0] != K) {
        NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, 
            "matmul: %zu x %zu @ %zu x %zu", M, K, b->shape[0], N);
        return NULL;
    }
    
    nc_dtype dtype = nc_dtype_promote(a->dtype, b->dtype);
    size_t out_shape[] = {M, N};
    nc_tensor* out = nc_tensor_zeros(out_shape, 2, dtype);
    if (!out) return NULL;
    
    // Basic triple loop (will optimize with SIMD later)
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                double va = nc_tensor_get2(a, i, k);
                double vb = nc_tensor_get2(b, k, j);
                sum += va * vb;
            }
            nc_tensor_set2(out, i, j, sum);
        }
    }
    
    return out;
}

// Matrix-vector: (M, K) @ (K,) -> (M,)
static nc_tensor* matmul_mv(nc_tensor* mat, nc_tensor* vec) {
    size_t M = mat->shape[0];
    size_t K = mat->shape[1];
    
    if (vec->shape[0] != K) {
        NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, "mv: K mismatch");
        return NULL;
    }
    
    size_t out_shape[] = {M};
    nc_tensor* out = nc_tensor_zeros(out_shape, 1, 
        nc_dtype_promote(mat->dtype, vec->dtype));
    if (!out) return NULL;
    
    for (size_t i = 0; i < M; i++) {
        double sum = 0.0;
        for (size_t k = 0; k < K; k++) {
            sum += nc_tensor_get2(mat, i, k) * nc_tensor_get1(vec, k);
        }
        nc_tensor_set1(out, i, sum);
    }
    
    return out;
}

// Vector-matrix: (K,) @ (K, N) -> (N,)
static nc_tensor* matmul_vm(nc_tensor* vec, nc_tensor* mat) {
    size_t K = vec->shape[0];
    size_t N = mat->shape[1];
    
    if (mat->shape[0] != K) {
        NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, "vm: K mismatch");
        return NULL;
    }
    
    size_t out_shape[] = {N};
    nc_tensor* out = nc_tensor_zeros(out_shape, 1,
        nc_dtype_promote(vec->dtype, mat->dtype));
    if (!out) return NULL;
    
    for (size_t j = 0; j < N; j++) {
        double sum = 0.0;
        for (size_t k = 0; k < K; k++) {
            sum += nc_tensor_get1(vec, k) * nc_tensor_get2(mat, k, j);
        }
        nc_tensor_set1(out, j, sum);
    }
    
    return out;
}

nc_tensor* nc_matmul(nc_tensor* a, nc_tensor* b) {
    NC_CHECK_NULL(a);
    NC_CHECK_NULL(b);
    
    nc_tensor* out = NULL;
    
    // Dispatch based on dimensions
    if (a->ndim == 2 && b->ndim == 2) {
        out = matmul_2d(a, b);
    } else if (a->ndim == 2 && b->ndim == 1) {
        out = matmul_mv(a, b);
    } else if (a->ndim == 1 && b->ndim == 2) {
        out = matmul_vm(a, b);
    } else if (a->ndim == 1 && b->ndim == 1) {
        // Dot product
        out = nc_dot(a, b);
    } else if (a->ndim >= 3 || b->ndim >= 3) {
        // Batched matmul
        out = nc_bmm(a, b);
    } else {
        NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, "Invalid matmul dimensions");
        return NULL;
    }
    
    // Setup autograd
    if (out && nc_grad_enabled() && (a->requires_grad || b->requires_grad)) {
        out->requires_grad = true;
        out->is_leaf = false;
        
        nc_node* node = nc_node_create("matmul", nc_backward_matmul);
        if (node) {
            nc_node_add_input(node, a);
            nc_node_add_input(node, b);
            nc_node_save_tensor(node, a);
            nc_node_save_tensor(node, b);
            node->output = out;
            out->grad_fn = node;
        }
    }
    
    return out;
}

nc_tensor* nc_bmm(nc_tensor* a, nc_tensor* b) {
    NC_CHECK_NULL(a);
    NC_CHECK_NULL(b);
    
    if (a->ndim < 3 || b->ndim < 3) {
        NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, "bmm requires 3D+ tensors");
        return NULL;
    }
    
    // Get batch dimensions
    size_t batch = a->shape[0];
    size_t M = a->shape[a->ndim - 2];
    size_t K = a->shape[a->ndim - 1];
    size_t N = b->shape[b->ndim - 1];
    
    if (b->shape[0] != batch || b->shape[b->ndim - 2] != K) {
        NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, "bmm shape mismatch");
        return NULL;
    }
    
    size_t out_shape[] = {batch, M, N};
    nc_tensor* out = nc_tensor_zeros(out_shape, 3,
        nc_dtype_promote(a->dtype, b->dtype));
    if (!out) return NULL;
    
    // Loop over batch
    for (size_t bi = 0; bi < batch; bi++) {
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                double sum = 0.0;
                for (size_t k = 0; k < K; k++) {
                    sum += nc_tensor_get3(a, bi, i, k) * 
                           nc_tensor_get3(b, bi, k, j);
                }
                nc_tensor_set3(out, bi, i, j, sum);
            }
        }
    }
    
    return out;
}

nc_tensor* nc_dot(nc_tensor* a, nc_tensor* b) {
    NC_CHECK_NULL(a);
    NC_CHECK_NULL(b);
    
    if (a->ndim != 1 || b->ndim != 1 || a->numel != b->numel) {
        NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, "dot requires equal-length 1D tensors");
        return NULL;
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < a->numel; i++) {
        sum += nc_tensor_get_flat(a, i) * nc_tensor_get_flat(b, i);
    }
    
    return nc_tensor_scalar(sum, nc_dtype_promote(a->dtype, b->dtype));
}

nc_tensor* nc_outer(nc_tensor* a, nc_tensor* b) {
    NC_CHECK_NULL(a);
    NC_CHECK_NULL(b);
    
    if (a->ndim != 1 || b->ndim != 1) {
        NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, "outer requires 1D tensors");
        return NULL;
    }
    
    size_t M = a->numel;
    size_t N = b->numel;
    size_t out_shape[] = {M, N};
    
    nc_tensor* out = nc_tensor_empty(out_shape, 2,
        nc_dtype_promote(a->dtype, b->dtype));
    if (!out) return NULL;
    
    for (size_t i = 0; i < M; i++) {
        double va = nc_tensor_get_flat(a, i);
        for (size_t j = 0; j < N; j++) {
            nc_tensor_set2(out, i, j, va * nc_tensor_get_flat(b, j));
        }
    }
    
    return out;
}

nc_tensor* nc_mv(nc_tensor* mat, nc_tensor* vec) {
    return matmul_mv(mat, vec);
}

nc_tensor* nc_vm(nc_tensor* vec, nc_tensor* mat) {
    return matmul_vm(vec, mat);
}

nc_tensor* nc_addmm(nc_tensor* c, nc_tensor* a, nc_tensor* b,
                    double alpha, double beta) {
    NC_CHECK_NULL(a);
    NC_CHECK_NULL(b);
    NC_CHECK_NULL(c);
    
    nc_tensor* ab = nc_matmul(a, b);
    if (!ab) return NULL;
    
    nc_tensor* out = nc_tensor_clone(c);
    if (!out) {
        nc_tensor_free(ab);
        return NULL;
    }
    
    // out = beta * c + alpha * (a @ b)
    for (size_t i = 0; i < out->numel; i++) {
        double v = beta * nc_tensor_get_flat(c, i) + 
                   alpha * nc_tensor_get_flat(ab, i);
        nc_tensor_set_flat(out, i, v);
    }
    
    nc_tensor_free(ab);
    return out;
}