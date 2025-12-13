#include "nocta/ops/matmul.h"
#include "nocta/autograd/node.h"
#include "nocta/autograd/backward.h"
#include <string.h>
#include <stdio.h>

#ifdef NOCTA_OPENMP_ENABLED
#include <omp.h>
#endif

// ============================================
// Backward
// ============================================

nc_tensor** nc_backward_matmul(nc_tensor* grad, nc_tensor** saved, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(2, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    nc_tensor* a = saved[0];  // (M, K)
    nc_tensor* b = saved[1];  // (K, N)
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
    
    nc_tensor* a_cont = nc_tensor_is_contiguous(a) ? a : nc_tensor_contiguous(a);
    nc_tensor* b_cont = nc_tensor_is_contiguous(b) ? b : nc_tensor_contiguous(b);
    
    if (!a_cont || !b_cont) {
        if (a_cont != a) nc_tensor_free(a_cont);
        if (b_cont != b) nc_tensor_free(b_cont);
        nc_tensor_free(out);
        return NULL;
    }

    // Optimized implementation using raw pointers and IKJ loop order
    if (dtype == NC_F32) {
        float* ap = nc_tensor_data_f32(a_cont);
        float* bp = nc_tensor_data_f32(b_cont);
        float* cp = nc_tensor_data_f32(out);
        
        int i;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for
        #endif
        for (i = 0; i < (int)M; i++) {
            for (size_t k = 0; k < K; k++) {
                float a_val = ap[i * K + k];
                for (size_t j = 0; j < N; j++) {
                    cp[i * N + j] += a_val * bp[k * N + j];
                }
            }
        }
    } else {
        double* ap = nc_tensor_data_f64(a_cont);
        double* bp = nc_tensor_data_f64(b_cont);
        double* cp = nc_tensor_data_f64(out);
        
        int i;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for
        #endif
        for (i = 0; i < (int)M; i++) {
            for (size_t k = 0; k < K; k++) {
                double a_val = ap[i * K + k];
                for (size_t j = 0; j < N; j++) {
                    cp[i * N + j] += a_val * bp[k * N + j];
                }
            }
        }
    }
    
    if (a_cont != a) nc_tensor_free(a_cont);
    if (b_cont != b) nc_tensor_free(b_cont);
    
    return out;
}

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
    
    nc_tensor* m_cont = nc_tensor_is_contiguous(mat) ? mat : nc_tensor_contiguous(mat);
    nc_tensor* v_cont = nc_tensor_is_contiguous(vec) ? vec : nc_tensor_contiguous(vec);

    if (out->dtype == NC_F32) {
        float* mp = nc_tensor_data_f32(m_cont);
        float* vp = nc_tensor_data_f32(v_cont);
        float* op = nc_tensor_data_f32(out);
        
        int i;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for
        #endif
        for (i = 0; i < (int)M; i++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                sum += (double)mp[i * K + k] * (double)vp[k];
            }
            op[i] = (float)sum;
        }
    } else {
        double* mp = nc_tensor_data_f64(m_cont);
        double* vp = nc_tensor_data_f64(v_cont);
        double* op = nc_tensor_data_f64(out);
        
        int i;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for
        #endif
        for (i = 0; i < (int)M; i++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                sum += mp[i * K + k] * vp[k];
            }
            op[i] = sum;
        }
    }

    if (m_cont != mat) nc_tensor_free(m_cont);
    if (v_cont != vec) nc_tensor_free(v_cont);
    
    return out;
}

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
    
    nc_tensor* v_cont = nc_tensor_is_contiguous(vec) ? vec : nc_tensor_contiguous(vec);
    nc_tensor* m_cont = nc_tensor_is_contiguous(mat) ? mat : nc_tensor_contiguous(mat);

    if (out->dtype == NC_F32) {
        float* vp = nc_tensor_data_f32(v_cont);
        float* mp = nc_tensor_data_f32(m_cont);
        float* op = nc_tensor_data_f32(out);
        
        int j;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for
        #endif
        for (j = 0; j < (int)N; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                sum += (double)vp[k] * (double)mp[k * N + j];
            }
            op[j] = (float)sum;
        }
    } else {
        double* vp = nc_tensor_data_f64(v_cont);
        double* mp = nc_tensor_data_f64(m_cont);
        double* op = nc_tensor_data_f64(out);
        
        int j;
        #ifdef NOCTA_OPENMP_ENABLED
        #pragma omp parallel for
        #endif
        for (j = 0; j < (int)N; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                sum += vp[k] * mp[k * N + j];
            }
            op[j] = sum;
        }
    }

    if (v_cont != vec) nc_tensor_free(v_cont);
    if (m_cont != mat) nc_tensor_free(m_cont);
    
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
        out = nc_dot(a, b);
    } else if (a->ndim >= 3 || b->ndim >= 3) {
        out = nc_bmm(a, b);
    } else {
        NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, "Invalid matmul dimensions");
        return NULL;
    }
    
    if (!out) return NULL;
    
    // Setup autograd - check if EITHER input requires grad
    bool needs_grad = nc_grad_enabled() && (a->requires_grad || b->requires_grad);
    
    if (needs_grad) {
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
    
    int bi;
    (void)bi;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (bi = 0; bi < (int)batch; bi++) {
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
    
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for reduction(+:sum)
    #endif
    for (i = 0; i < (int)a->numel; i++) {
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
    
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)M; i++) {
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
    
    int i;
    (void)i;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (i = 0; i < (int)out->numel; i++) {
        double v = beta * nc_tensor_get_flat(c, i) + 
                   alpha * nc_tensor_get_flat(ab, i);
        nc_tensor_set_flat(out, i, v);
    }
    
    nc_tensor_free(ab);
    return out;
}