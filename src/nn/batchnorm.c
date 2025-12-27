#include "nocta/nn/batchnorm.h"
#include "nocta/core/memory.h"
#include "nocta/autograd/node.h"
#include "nocta/autograd/backward.h"
#include <math.h>
#include <string.h>

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
// BatchNorm2D Extra Data
// ============================================

typedef struct {
    size_t num_features;
    double eps;
    double momentum;
    bool affine;
    bool track_running_stats;
    size_t num_batches_tracked;
} nc_batchnorm2d_data;

// ============================================
// BatchNorm2D Backward
// ============================================

nc_tensor** nc_backward_batchnorm2d(nc_tensor* grad, nc_tensor** saved, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(3, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    nc_tensor* input = saved[0];      // (N, C, H, W)
    nc_tensor* weight = saved[1];     // (C,) gamma - may be NULL
    nc_tensor* mean = saved[2];       // (C,) batch mean
    nc_tensor* var = saved[3];        // (C,) batch variance
    nc_tensor* eps_t = saved[4];      // scalar eps
    
    double eps = nc_tensor_get_flat(eps_t, 0);
    
    size_t N = input->shape[0];
    size_t C = input->shape[1];
    size_t H = input->shape[2];
    size_t W = input->shape[3];
    size_t spatial = H * W;
    size_t M = N * spatial;  // Total elements per channel
    
#ifdef NOCTA_CUDA_ENABLED
    if (grad->storage->device == NC_DEVICE_CUDA && 
        input->storage->device == NC_DEVICE_CUDA) {
        
        // Prepare gradients
        grads[0] = nc_tensor_zeros(input->shape, input->ndim, input->dtype);
        nc_tensor_to_device(grads[0], NC_DEVICE_CUDA);
        
        if (weight) {
            grads[1] = nc_tensor_zeros(weight->shape, weight->ndim, weight->dtype);
            nc_tensor_to_device(grads[1], NC_DEVICE_CUDA);
        }
        
        size_t b_shape[] = {C};
        grads[2] = nc_tensor_zeros(b_shape, 1, input->dtype);
        nc_tensor_to_device(grads[2], NC_DEVICE_CUDA);
        
        nc_cuda_batchnorm_backward_f32(
            (float*)grads[0]->storage->cuda_data,
            grads[1] ? (float*)grads[1]->storage->cuda_data : NULL,
            (float*)grads[2]->storage->cuda_data,
            (const float*)grad->storage->cuda_data,
            (const float*)input->storage->cuda_data,
            weight ? (const float*)weight->storage->cuda_data : NULL, // gamma
            (const float*)mean->storage->cuda_data,
            (const float*)var->storage->cuda_data,
            (int)N, (int)C, (int)spatial,
            (float)eps
        );
        return grads;
    }
#endif

    // Gradient w.r.t input
    grads[0] = nc_tensor_zeros(input->shape, input->ndim, input->dtype);
    
    // Gradient w.r.t weight (gamma)
    if (weight) {
        grads[1] = nc_tensor_zeros(weight->shape, weight->ndim, weight->dtype);
    }
    
    // Gradient w.r.t bias (beta) - sum of upstream gradients
    size_t bias_shape[] = {C};
    grads[2] = nc_tensor_zeros(bias_shape, 1, input->dtype);
    
    int c;
    (void)c;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (c = 0; c < (int)C; c++) {
        double mu = nc_tensor_get1(mean, c);
        double v = nc_tensor_get1(var, c);
        double std = sqrt(v + eps);
        double inv_std = 1.0 / std;
        double gamma = weight ? nc_tensor_get1(weight, c) : 1.0;
        
        // Accumulate for this channel
        double sum_grad = 0.0;
        double sum_grad_xhat = 0.0;
        
        // First pass: compute sums
        for (size_t b = 0; b < N; b++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    double g = nc_tensor_get4(grad, b, c, h, w);
                    double x = nc_tensor_get4(input, b, c, h, w);
                    double x_hat = (x - mu) * inv_std;
                    
                    sum_grad += g;
                    sum_grad_xhat += g * x_hat;
                }
            }
        }
        
        // d_beta = sum(grad)
        nc_tensor_set1(grads[2], c, sum_grad);
        
        // d_gamma = sum(grad * x_hat)
        if (weight) {
            nc_tensor_set1(grads[1], c, sum_grad_xhat);
        }
        
        // d_input: using the standard batchnorm backward formula
        // dx = (1/M) * gamma * inv_std * (M * dout - sum(dout) - x_hat * sum(dout * x_hat))
        double coeff = gamma * inv_std / (double)M;
        
        for (size_t b = 0; b < N; b++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    double g = nc_tensor_get4(grad, b, c, h, w);
                    double x = nc_tensor_get4(input, b, c, h, w);
                    double x_hat = (x - mu) * inv_std;
                    
                    double dx = coeff * ((double)M * g - sum_grad - x_hat * sum_grad_xhat);
                    nc_tensor_set4(grads[0], b, c, h, w, dx);
                }
            }
        }
    }
    
    return grads;
}

// ============================================
// BatchNorm2D Forward (Functional)
// ============================================

nc_tensor* nc_batchnorm2d_forward_fn(
    nc_tensor* input,
    nc_tensor* running_mean,
    nc_tensor* running_var,
    nc_tensor* weight,
    nc_tensor* bias,
    bool training,
    double momentum,
    double eps
) {
    if (!input || input->ndim != 4) {
        NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, "BatchNorm2D requires 4D input (N,C,H,W)");
        return NULL;
    }
    
    size_t N = input->shape[0];
    size_t C = input->shape[1];
    size_t H = input->shape[2];
    size_t W = input->shape[3];
    size_t spatial = H * W;
    size_t M = N * spatial;
    
#ifdef NOCTA_CUDA_ENABLED
    // CUDA path for F32 tensors on GPU
    if (tensor_on_cuda(input) && input->dtype == NC_F32 && nc_tensor_is_contiguous(input)) {
        nc_tensor* out = nc_tensor_empty(input->shape, input->ndim, input->dtype);
        if (!out) return NULL;
        nc_storage_to_device(out->storage, NC_DEVICE_CUDA);
        
        size_t c_shape[] = {C};
        nc_tensor* batch_mean = nc_tensor_zeros(c_shape, 1, NC_F32);
        nc_tensor* batch_var = nc_tensor_zeros(c_shape, 1, NC_F32);
        if (!batch_mean || !batch_var) {
            nc_tensor_free(out);
            nc_tensor_free(batch_mean);
            nc_tensor_free(batch_var);
            return NULL;
        }
        nc_storage_to_device(batch_mean->storage, NC_DEVICE_CUDA);
        nc_storage_to_device(batch_var->storage, NC_DEVICE_CUDA);
        
        const float* gamma_ptr = (weight && tensor_on_cuda(weight)) ? 
            (const float*)weight->storage->cuda_data : NULL;
        const float* beta_ptr = (bias && tensor_on_cuda(bias)) ? 
            (const float*)bias->storage->cuda_data : NULL;
        float* rm_ptr = (running_mean && tensor_on_cuda(running_mean)) ?
            (float*)running_mean->storage->cuda_data : NULL;
        float* rv_ptr = (running_var && tensor_on_cuda(running_var)) ?
            (float*)running_var->storage->cuda_data : NULL;
        
        nc_cuda_batchnorm_forward_f32(
            (float*)out->storage->cuda_data,
            (const float*)input->storage->cuda_data,
            gamma_ptr, beta_ptr,
            rm_ptr, rv_ptr,
            (float*)batch_mean->storage->cuda_data,
            (float*)batch_var->storage->cuda_data,
            (int)N, (int)C, (int)spatial,
            (float)momentum, (float)eps,
            training);
        
        // Setup autograd
        if (nc_grad_enabled() && (input->requires_grad || 
            (weight && weight->requires_grad) || (bias && bias->requires_grad))) {
            out->requires_grad = true;
            out->is_leaf = false;
            nc_node* node = nc_node_create("batchnorm2d", nc_backward_batchnorm2d);
            if (node) {
                nc_node_add_input(node, input);
                if (weight) nc_node_add_input(node, weight);
                if (bias) nc_node_add_input(node, bias);
                nc_node_save_tensor(node, input);
                nc_node_save_tensor(node, weight);
                nc_node_save_owned_tensor(node, batch_mean);
                nc_node_save_owned_tensor(node, batch_var);
                nc_tensor* eps_t = nc_tensor_scalar(eps, NC_F64);
                nc_node_save_owned_tensor(node, eps_t);
                node->output = out;
                out->grad_fn = node;
            }
        } else {
            nc_tensor_free(batch_mean);
            nc_tensor_free(batch_var);
        }
        return out;
    }
#endif
    
    // CPU path
    nc_tensor* out = nc_tensor_empty(input->shape, input->ndim, input->dtype);
    if (!out) return NULL;
    
    // Tensors for batch statistics (used in backward)
    size_t c_shape[] = {C};
    nc_tensor* batch_mean = nc_tensor_zeros(c_shape, 1, input->dtype);
    nc_tensor* batch_var = nc_tensor_zeros(c_shape, 1, input->dtype);
    
    if (!batch_mean || !batch_var) {
        nc_tensor_free(out);
        nc_tensor_free(batch_mean);
        nc_tensor_free(batch_var);
        return NULL;
    }
    
    int c;
    (void)c;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (c = 0; c < (int)C; c++) {
        double mean_val, var_val;
        
        if (training) {
            // Compute batch mean
            double sum = 0.0;
            for (size_t b = 0; b < N; b++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        sum += nc_tensor_get4(input, b, c, h, w);
                    }
                }
            }
            mean_val = sum / (double)M;
            
            // Compute batch variance
            double var_sum = 0.0;
            for (size_t b = 0; b < N; b++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        double diff = nc_tensor_get4(input, b, c, h, w) - mean_val;
                        var_sum += diff * diff;
                    }
                }
            }
            var_val = var_sum / (double)M;
            
            // Update running statistics
            if (running_mean && running_var) {
                double rm = nc_tensor_get1(running_mean, c);
                double rv = nc_tensor_get1(running_var, c);
                nc_tensor_set1(running_mean, c, (1.0 - momentum) * rm + momentum * mean_val);
                double unbiased_var = (M > 1) ? var_sum / (double)(M - 1) : var_val;
                nc_tensor_set1(running_var, c, (1.0 - momentum) * rv + momentum * unbiased_var);
            }
        } else {
            // Use running statistics for inference
            mean_val = running_mean ? nc_tensor_get1(running_mean, c) : 0.0;
            var_val = running_var ? nc_tensor_get1(running_var, c) : 1.0;
        }
        
        // Store batch stats for backward
        nc_tensor_set1(batch_mean, c, mean_val);
        nc_tensor_set1(batch_var, c, var_val);
        
        // Normalize and apply affine transform
        double inv_std = 1.0 / sqrt(var_val + eps);
        double gamma = weight ? nc_tensor_get1(weight, c) : 1.0;
        double beta = bias ? nc_tensor_get1(bias, c) : 0.0;
        
        for (size_t b = 0; b < N; b++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t w = 0; w < W; w++) {
                    double x = nc_tensor_get4(input, b, c, h, w);
                    double x_norm = (x - mean_val) * inv_std;
                    double y = gamma * x_norm + beta;
                    nc_tensor_set4(out, b, c, h, w, y);
                }
            }
        }
    }
    
    // Setup autograd
    if (nc_grad_enabled() && (input->requires_grad || 
        (weight && weight->requires_grad) || (bias && bias->requires_grad))) {
        out->requires_grad = true;
        out->is_leaf = false;
        
        nc_node* node = nc_node_create("batchnorm2d", nc_backward_batchnorm2d);
        if (node) {
            nc_node_add_input(node, input);
            if (weight) nc_node_add_input(node, weight);
            if (bias) nc_node_add_input(node, bias);
            
            nc_node_save_tensor(node, input);
            nc_node_save_tensor(node, weight);
            nc_node_save_owned_tensor(node, batch_mean);
            nc_node_save_owned_tensor(node, batch_var);
            
            nc_tensor* eps_t = nc_tensor_scalar(eps, NC_F64);
            nc_node_save_owned_tensor(node, eps_t);
            
            node->output = out;
            out->grad_fn = node;
        }
    } else {
        nc_tensor_free(batch_mean);
        nc_tensor_free(batch_var);
    }
    
    return out;
}

// ============================================
// BatchNorm2D Module
// ============================================

static nc_tensor* batchnorm2d_forward(nc_module* self, nc_tensor* input) {
    nc_batchnorm2d_data* data = self->extra;
    
    nc_tensor* weight = nc_module_get_param(self, "weight");
    nc_tensor* bias = nc_module_get_param(self, "bias");
    nc_tensor* running_mean = nc_module_get_param(self, "running_mean");
    nc_tensor* running_var = nc_module_get_param(self, "running_var");
    
    return nc_batchnorm2d_forward_fn(
        input,
        running_mean,
        running_var,
        weight,
        bias,
        self->training,
        data->momentum,
        data->eps
    );
}

nc_module* nc_batchnorm2d(size_t num_features) {
    nc_batchnorm2d_config cfg = NC_BATCHNORM2D_DEFAULT;
    cfg.num_features = num_features;
    return nc_batchnorm2d_ex(cfg);
}

nc_module* nc_batchnorm2d_ex(nc_batchnorm2d_config config) {
    nc_module* m = nc_module_create("BatchNorm2D");
    if (!m) return NULL;
    
    nc_batchnorm2d_data* data = nc_alloc(sizeof(nc_batchnorm2d_data));
    if (!data) { nc_module_free(m); return NULL; }
    
    data->num_features = config.num_features;
    data->eps = config.eps;
    data->momentum = config.momentum;
    data->affine = config.affine;
    data->track_running_stats = config.track_running_stats;
    data->num_batches_tracked = 0;
    
    m->extra = data;
    m->free_extra = nc_free;
    m->forward = batchnorm2d_forward;
    
    size_t shape[] = {config.num_features};
    
    // Learnable parameters (gamma, beta)
    if (config.affine) {
        nc_tensor* weight = nc_tensor_ones(shape, 1, NC_F32);
        nc_tensor* bias = nc_tensor_zeros(shape, 1, NC_F32);
        if (!weight || !bias) {
            nc_tensor_free(weight);
            nc_tensor_free(bias);
            nc_module_free(m);
            return NULL;
        }
        nc_module_add_param(m, "weight", weight);
        nc_module_add_param(m, "bias", bias);
    }
    
    // Running statistics (not learnable, but saved)
    if (config.track_running_stats) {
        nc_tensor* running_mean = nc_tensor_zeros(shape, 1, NC_F32);
        nc_tensor* running_var = nc_tensor_ones(shape, 1, NC_F32);
        if (!running_mean || !running_var) {
            nc_tensor_free(running_mean);
            nc_tensor_free(running_var);
            nc_module_free(m);
            return NULL;
        }
        // These are buffers, not parameters (no gradient)
        running_mean->requires_grad = false;
        running_var->requires_grad = false;
        nc_module_add_param(m, "running_mean", running_mean);
        nc_module_add_param(m, "running_var", running_var);
    }
    
    return m;
}

nc_tensor* nc_batchnorm2d_weight(nc_module* bn) {
    return nc_module_get_param(bn, "weight");
}

nc_tensor* nc_batchnorm2d_bias(nc_module* bn) {
    return nc_module_get_param(bn, "bias");
}

nc_tensor* nc_batchnorm2d_running_mean(nc_module* bn) {
    return nc_module_get_param(bn, "running_mean");
}

nc_tensor* nc_batchnorm2d_running_var(nc_module* bn) {
    return nc_module_get_param(bn, "running_var");
}

// ============================================
// BatchNorm1D Backward
// ============================================

nc_tensor** nc_backward_batchnorm1d(nc_tensor* grad, nc_tensor** saved, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(3, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    nc_tensor* input = saved[0];
    nc_tensor* weight = saved[1];
    nc_tensor* mean = saved[2];
    nc_tensor* var = saved[3];
    nc_tensor* eps_t = saved[4];
    
    double eps = nc_tensor_get_flat(eps_t, 0);
    
    size_t N = input->shape[0];
    size_t C = input->shape[1];
    size_t L = (input->ndim == 3) ? input->shape[2] : 1;
    size_t M = N * L;
    
    grads[0] = nc_tensor_zeros(input->shape, input->ndim, input->dtype);
    
#ifdef NOCTA_CUDA_ENABLED
    if (grad->storage->device == NC_DEVICE_CUDA && 
        input->storage->device == NC_DEVICE_CUDA) {
        
        nc_tensor_to_device(grads[0], NC_DEVICE_CUDA);
        
        if (weight) {
            grads[1] = nc_tensor_zeros(weight->shape, weight->ndim, weight->dtype);
            nc_tensor_to_device(grads[1], NC_DEVICE_CUDA);
        }
        
        size_t b_shape[] = {C};
        grads[2] = nc_tensor_zeros(b_shape, 1, input->dtype);
        nc_tensor_to_device(grads[2], NC_DEVICE_CUDA);
        
        nc_cuda_batchnorm_backward_f32(
            (float*)grads[0]->storage->cuda_data,
            grads[1] ? (float*)grads[1]->storage->cuda_data : NULL,
            (float*)grads[2]->storage->cuda_data,
            (const float*)grad->storage->cuda_data,
            (const float*)input->storage->cuda_data,
            weight ? (const float*)weight->storage->cuda_data : NULL,
            (const float*)mean->storage->cuda_data,
            (const float*)var->storage->cuda_data,
            (int)N, (int)C, (int)L,
            (float)eps
        );
        return grads;
    }
#endif
    
    if (weight) {
        grads[1] = nc_tensor_zeros(weight->shape, weight->ndim, weight->dtype);
    }
    
    size_t bias_shape[] = {C};
    grads[2] = nc_tensor_zeros(bias_shape, 1, input->dtype);
    
    int c;
    (void)c;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (c = 0; c < (int)C; c++) {
        double mu = nc_tensor_get1(mean, c);
        double v = nc_tensor_get1(var, c);
        double inv_std = 1.0 / sqrt(v + eps);
        double gamma = weight ? nc_tensor_get1(weight, c) : 1.0;
        
        double sum_grad = 0.0;
        double sum_grad_xhat = 0.0;
        
        for (size_t b = 0; b < N; b++) {
            if (input->ndim == 2) {
                double g = nc_tensor_get2(grad, b, c);
                double x = nc_tensor_get2(input, b, c);
                double x_hat = (x - mu) * inv_std;
                sum_grad += g;
                sum_grad_xhat += g * x_hat;
            } else {
                for (size_t l = 0; l < L; l++) {
                    double g = nc_tensor_get3(grad, b, c, l);
                    double x = nc_tensor_get3(input, b, c, l);
                    double x_hat = (x - mu) * inv_std;
                    sum_grad += g;
                    sum_grad_xhat += g * x_hat;
                }
            }
        }
        
        nc_tensor_set1(grads[2], c, sum_grad);
        if (weight) {
            nc_tensor_set1(grads[1], c, sum_grad_xhat);
        }
        
        double coeff = gamma * inv_std / (double)M;
        
        for (size_t b = 0; b < N; b++) {
            if (input->ndim == 2) {
                double g = nc_tensor_get2(grad, b, c);
                double x = nc_tensor_get2(input, b, c);
                double x_hat = (x - mu) * inv_std;
                double dx = coeff * ((double)M * g - sum_grad - x_hat * sum_grad_xhat);
                nc_tensor_set2(grads[0], b, c, dx);
            } else {
                for (size_t l = 0; l < L; l++) {
                    double g = nc_tensor_get3(grad, b, c, l);
                    double x = nc_tensor_get3(input, b, c, l);
                    double x_hat = (x - mu) * inv_std;
                    double dx = coeff * ((double)M * g - sum_grad - x_hat * sum_grad_xhat);
                    nc_tensor_set3(grads[0], b, c, l, dx);
                }
            }
        }
    }
    
    return grads;
}

// ============================================
// LayerNorm Backward
// ============================================

nc_tensor** nc_backward_layernorm(nc_tensor* grad, nc_tensor** saved, size_t n) {
    (void)n;
    nc_tensor** grads = nc_calloc(3, sizeof(nc_tensor*));
    if (!grads) return NULL;
    
    nc_tensor* input = saved[0];
    nc_tensor* weight = saved[1];
    nc_tensor* norm_size_t = saved[2];
    nc_tensor* eps_t = saved[3];
    
    size_t norm_size = (size_t)nc_tensor_get_flat(norm_size_t, 0);
    double eps = nc_tensor_get_flat(eps_t, 0);
    size_t outer_size = input->numel / norm_size;
    
    grads[0] = nc_tensor_zeros(input->shape, input->ndim, input->dtype);
    
    if (weight) {
        size_t w_shape[] = {norm_size};
        grads[1] = nc_tensor_zeros(w_shape, 1, input->dtype);
        grads[2] = nc_tensor_zeros(w_shape, 1, input->dtype);
    }
    
    int o;
    (void)o;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (o = 0; o < (int)outer_size; o++) {
        // Compute mean and var for this instance
        double sum = 0.0;
        for (size_t i = 0; i < norm_size; i++) {
            sum += nc_tensor_get_flat(input, o * norm_size + i);
        }
        double mean = sum / (double)norm_size;
        
        double var_sum = 0.0;
        for (size_t i = 0; i < norm_size; i++) {
            double diff = nc_tensor_get_flat(input, o * norm_size + i) - mean;
            var_sum += diff * diff;
        }
        double var = var_sum / (double)norm_size;
        double inv_std = 1.0 / sqrt(var + eps);
        
        // Compute gradient sums
        double sum_grad = 0.0;
        double sum_grad_xhat = 0.0;
        
        for (size_t i = 0; i < norm_size; i++) {
            double g = nc_tensor_get_flat(grad, o * norm_size + i);
            double x = nc_tensor_get_flat(input, o * norm_size + i);
            double x_hat = (x - mean) * inv_std;
            double gamma = weight ? nc_tensor_get_flat(weight, i) : 1.0;
            
            sum_grad += g * gamma;
            sum_grad_xhat += g * gamma * x_hat;
            
            // Accumulate weight/bias gradients
            if (weight) {
                if (grads[1]->dtype == NC_F32) {
                    float* w_data = nc_tensor_data_f32(grads[1]);
                    float* b_data = nc_tensor_data_f32(grads[2]);
                    
                    #ifdef NOCTA_OPENMP_ENABLED
                    #pragma omp atomic
                    #endif
                    w_data[i] += (float)(g * x_hat);
                    
                    #ifdef NOCTA_OPENMP_ENABLED
                    #pragma omp atomic
                    #endif
                    b_data[i] += (float)g;
                } else {
                    double* w_data = nc_tensor_data_f64(grads[1]);
                    double* b_data = nc_tensor_data_f64(grads[2]);
                    
                    #ifdef NOCTA_OPENMP_ENABLED
                    #pragma omp atomic
                    #endif
                    w_data[i] += g * x_hat;
                    
                    #ifdef NOCTA_OPENMP_ENABLED
                    #pragma omp atomic
                    #endif
                    b_data[i] += g;
                }
            }
        }
        
        // Compute input gradient
        double coeff = inv_std / (double)norm_size;
        for (size_t i = 0; i < norm_size; i++) {
            double g = nc_tensor_get_flat(grad, o * norm_size + i);
            double x = nc_tensor_get_flat(input, o * norm_size + i);
            double x_hat = (x - mean) * inv_std;
            double gamma = weight ? nc_tensor_get_flat(weight, i) : 1.0;
            
            double dx = coeff * ((double)norm_size * g * gamma - sum_grad - x_hat * sum_grad_xhat);
            nc_tensor_set_flat(grads[0], o * norm_size + i, dx);
        }
    }
    
    return grads;
}

// ============================================
// BatchNorm1D Implementation
// ============================================

nc_tensor* nc_batchnorm1d_forward_fn(
    nc_tensor* input,
    nc_tensor* running_mean,
    nc_tensor* running_var,
    nc_tensor* weight,
    nc_tensor* bias,
    bool training,
    double momentum,
    double eps
) {
    if (!input || (input->ndim != 2 && input->ndim != 3)) {
        NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, "BatchNorm1D requires 2D or 3D input");
        return NULL;
    }
    
    size_t N = input->shape[0];
    size_t C = input->shape[1];
    size_t L = (input->ndim == 3) ? input->shape[2] : 1;
    size_t M = N * L;
    
    nc_tensor* out = nc_tensor_empty(input->shape, input->ndim, input->dtype);
    if (!out) return NULL;
    
#ifdef NOCTA_CUDA_ENABLED
    if (tensor_on_cuda(input) && input->dtype == NC_F32) {
        nc_storage_to_device(out->storage, NC_DEVICE_CUDA);
        
        size_t c_shape[] = {C};
        nc_tensor* batch_mean = nc_tensor_zeros(c_shape, 1, NC_F32);
        nc_tensor* batch_var = nc_tensor_zeros(c_shape, 1, NC_F32);
        if (!batch_mean || !batch_var) {
            nc_tensor_free(out);
            nc_tensor_free(batch_mean);
            nc_tensor_free(batch_var);
            return NULL;
        }
        nc_storage_to_device(batch_mean->storage, NC_DEVICE_CUDA);
        nc_storage_to_device(batch_var->storage, NC_DEVICE_CUDA);
        
        const float* gamma_ptr = (weight && tensor_on_cuda(weight)) ? 
            (const float*)weight->storage->cuda_data : NULL;
        const float* beta_ptr = (bias && tensor_on_cuda(bias)) ? 
            (const float*)bias->storage->cuda_data : NULL;
        float* rm_ptr = (running_mean && tensor_on_cuda(running_mean)) ?
            (float*)running_mean->storage->cuda_data : NULL;
        float* rv_ptr = (running_var && tensor_on_cuda(running_var)) ?
            (float*)running_var->storage->cuda_data : NULL;
        
        nc_cuda_batchnorm_forward_f32(
            (float*)out->storage->cuda_data,
            (const float*)input->storage->cuda_data,
            gamma_ptr, beta_ptr,
            rm_ptr, rv_ptr,
            (float*)batch_mean->storage->cuda_data,
            (float*)batch_var->storage->cuda_data,
            (int)N, (int)C, (int)L,
            (float)momentum, (float)eps,
            training);
            
         // Setup autograd
        if (nc_grad_enabled() && (input->requires_grad || 
            (weight && weight->requires_grad) || (bias && bias->requires_grad))) {
            out->requires_grad = true;
            out->is_leaf = false;
            nc_node* node = nc_node_create("batchnorm1d", nc_backward_batchnorm1d);
            if (node) {
                nc_node_add_input(node, input);
                if (weight) nc_node_add_input(node, weight);
                if (bias) nc_node_add_input(node, bias);
                nc_node_save_tensor(node, input);
                nc_node_save_tensor(node, weight);
                nc_node_save_owned_tensor(node, batch_mean);
                nc_node_save_owned_tensor(node, batch_var);
                nc_tensor* eps_t = nc_tensor_scalar(eps, NC_F64);
                nc_node_save_owned_tensor(node, eps_t);
                node->output = out;
                out->grad_fn = node;
            } else {
                nc_tensor_free(batch_mean);
                nc_tensor_free(batch_var);
            }
        } else {
            nc_tensor_free(batch_mean);
            nc_tensor_free(batch_var);
        }
        return out;
    }
#endif

    // Tensors for batch statistics (used in backward)
    size_t c_shape[] = {C};
    nc_tensor* batch_mean = nc_tensor_zeros(c_shape, 1, input->dtype);
    nc_tensor* batch_var = nc_tensor_zeros(c_shape, 1, input->dtype);
    
    if (!batch_mean || !batch_var) {
        nc_tensor_free(out);
        nc_tensor_free(batch_mean);
        nc_tensor_free(batch_var);
        return NULL;
    }
    
    int c;
    (void)c;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (c = 0; c < (int)C; c++) {
        double mean_val, var_val;
        
        if (training) {
            // Compute batch mean
            double sum = 0.0;
            for (size_t b = 0; b < N; b++) {
                if (input->ndim == 2) {
                    sum += nc_tensor_get2(input, b, c);
                } else {
                    for (size_t l = 0; l < L; l++) {
                        sum += nc_tensor_get3(input, b, c, l);
                    }
                }
            }
            mean_val = sum / (double)M;
            
            // Compute variance
            double var_sum = 0.0;
            for (size_t b = 0; b < N; b++) {
                if (input->ndim == 2) {
                    double diff = nc_tensor_get2(input, b, c) - mean_val;
                    var_sum += diff * diff;
                } else {
                    for (size_t l = 0; l < L; l++) {
                        double diff = nc_tensor_get3(input, b, c, l) - mean_val;
                        var_sum += diff * diff;
                    }
                }
            }
            var_val = var_sum / (double)M;
            
            // Update running stats
            if (running_mean && running_var) {
                double rm = nc_tensor_get1(running_mean, c);
                double rv = nc_tensor_get1(running_var, c);
                nc_tensor_set1(running_mean, c, (1.0 - momentum) * rm + momentum * mean_val);
                double unbiased_var = var_sum / (double)(M - 1);
                nc_tensor_set1(running_var, c, (1.0 - momentum) * rv + momentum * unbiased_var);
            }
        } else {
            mean_val = running_mean ? nc_tensor_get1(running_mean, c) : 0.0;
            var_val = running_var ? nc_tensor_get1(running_var, c) : 1.0;
        }
        
        double inv_std = 1.0 / sqrt(var_val + eps);
        double gamma = weight ? nc_tensor_get1(weight, c) : 1.0;
        double beta = bias ? nc_tensor_get1(bias, c) : 0.0;
        
        for (size_t b = 0; b < N; b++) {
            if (input->ndim == 2) {
                double x = nc_tensor_get2(input, b, c);
                double y = gamma * (x - mean_val) * inv_std + beta;
                nc_tensor_set2(out, b, c, y);
            } else {
                for (size_t l = 0; l < L; l++) {
                    double x = nc_tensor_get3(input, b, c, l);
                    double y = gamma * (x - mean_val) * inv_std + beta;
                    nc_tensor_set3(out, b, c, l, y);
                }
            }
        }
        
        nc_tensor_set1(batch_mean, c, mean_val);
        nc_tensor_set1(batch_var, c, var_val);
    }
    
    // Setup autograd
    if (nc_grad_enabled() && (input->requires_grad || 
        (weight && weight->requires_grad) || (bias && bias->requires_grad))) {
        out->requires_grad = true;
        out->is_leaf = false;
        
        nc_node* node = nc_node_create("batchnorm1d", nc_backward_batchnorm1d);
        if (node) {
            nc_node_add_input(node, input);
            if (weight) nc_node_add_input(node, weight);
            if (bias) nc_node_add_input(node, bias);
            
            nc_node_save_tensor(node, input);
            nc_node_save_tensor(node, weight);
            nc_node_save_owned_tensor(node, batch_mean);
            nc_node_save_owned_tensor(node, batch_var);
            
            nc_tensor* eps_t = nc_tensor_scalar(eps, NC_F64);
            nc_node_save_owned_tensor(node, eps_t);
            
            node->output = out;
            out->grad_fn = node;
        }
    } else {
        nc_tensor_free(batch_mean);
        nc_tensor_free(batch_var);
    }
    
    return out;
}

static nc_tensor* batchnorm1d_forward(nc_module* self, nc_tensor* input) {
    nc_batchnorm2d_data* data = self->extra;
    
    return nc_batchnorm1d_forward_fn(
        input,
        nc_module_get_param(self, "running_mean"),
        nc_module_get_param(self, "running_var"),
        nc_module_get_param(self, "weight"),
        nc_module_get_param(self, "bias"),
        self->training,
        data->momentum,
        data->eps
    );
}

nc_module* nc_batchnorm1d(size_t num_features) {
    nc_batchnorm2d_config cfg = NC_BATCHNORM2D_DEFAULT;
    cfg.num_features = num_features;
    return nc_batchnorm1d_ex(cfg);
}

nc_module* nc_batchnorm1d_ex(nc_batchnorm2d_config config) {
    nc_module* m = nc_batchnorm2d_ex(config);
    if (m) {
        strncpy(m->name, "BatchNorm1D", NC_MAX_NAME_LEN - 1);
        m->forward = batchnorm1d_forward;
    }
    return m;
}

// ============================================
// LayerNorm Implementation
// ============================================

nc_tensor* nc_layernorm_forward_fn(
    nc_tensor* input,
    const size_t* normalized_shape,
    size_t normalized_ndim,
    nc_tensor* weight,
    nc_tensor* bias,
    double eps
) {
    if (!input) return NULL;
    
    // Compute the size of normalized dimensions
    size_t norm_size = 1;
    for (size_t i = 0; i < normalized_ndim; i++) {
        norm_size *= normalized_shape[i];
    }
    
    // Compute the number of instances to normalize
    size_t outer_size = input->numel / norm_size;
    
    nc_tensor* out = nc_tensor_empty(input->shape, input->ndim, input->dtype);
    if (!out) return NULL;
    
    int o;
    (void)o;
    #ifdef NOCTA_OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for (o = 0; o < (int)outer_size; o++) {
        // Compute mean
        double sum = 0.0;
        for (size_t i = 0; i < norm_size; i++) {
            sum += nc_tensor_get_flat(input, o * norm_size + i);
        }
        double mean = sum / (double)norm_size;
        
        // Compute variance
        double var_sum = 0.0;
        for (size_t i = 0; i < norm_size; i++) {
            double diff = nc_tensor_get_flat(input, o * norm_size + i) - mean;
            var_sum += diff * diff;
        }
        double var = var_sum / (double)norm_size;
        double inv_std = 1.0 / sqrt(var + eps);
        
        // Normalize and apply affine
        for (size_t i = 0; i < norm_size; i++) {
            double x = nc_tensor_get_flat(input, o * norm_size + i);
            double x_norm = (x - mean) * inv_std;
            double gamma = weight ? nc_tensor_get_flat(weight, i) : 1.0;
            double beta = bias ? nc_tensor_get_flat(bias, i) : 0.0;
            nc_tensor_set_flat(out, o * norm_size + i, gamma * x_norm + beta);
        }
    }
    
    // Setup autograd
    if (nc_grad_enabled() && (input->requires_grad || 
        (weight && weight->requires_grad) || (bias && bias->requires_grad))) {
        out->requires_grad = true;
        out->is_leaf = false;
        
        nc_node* node = nc_node_create("layernorm", nc_backward_layernorm);
        if (node) {
            nc_node_add_input(node, input);
            if (weight) nc_node_add_input(node, weight);
            if (bias) nc_node_add_input(node, bias);
            
            nc_node_save_tensor(node, input);
            nc_node_save_tensor(node, weight);
            
            nc_tensor* norm_size_t = nc_tensor_scalar((double)norm_size, NC_F64);
            nc_tensor* eps_t = nc_tensor_scalar(eps, NC_F64);
            nc_node_save_owned_tensor(node, norm_size_t);
            nc_node_save_owned_tensor(node, eps_t);
            
            node->output = out;
            out->grad_fn = node;
        }
    }
    
    return out;
}

typedef struct {
    size_t normalized_shape[NC_MAX_DIMS];
    size_t normalized_ndim;
    double eps;
    bool elementwise_affine;
} nc_layernorm_data;

static nc_tensor* layernorm_forward(nc_module* self, nc_tensor* input) {
    nc_layernorm_data* data = self->extra;
    
    return nc_layernorm_forward_fn(
        input,
        data->normalized_shape,
        data->normalized_ndim,
        nc_module_get_param(self, "weight"),
        nc_module_get_param(self, "bias"),
        data->eps
    );
}

nc_module* nc_layernorm(const size_t* normalized_shape, size_t ndim) {
    nc_module* m = nc_module_create("LayerNorm");
    if (!m) return NULL;
    
    nc_layernorm_data* data = nc_alloc(sizeof(nc_layernorm_data));
    if (!data) { nc_module_free(m); return NULL; }
    
    data->normalized_ndim = ndim;
    data->eps = 1e-5;
    data->elementwise_affine = true;
    
    size_t total_size = 1;
    for (size_t i = 0; i < ndim; i++) {
        data->normalized_shape[i] = normalized_shape[i];
        total_size *= normalized_shape[i];
    }
    
    m->extra = data;
    m->free_extra = nc_free;
    m->forward = layernorm_forward;
    
    size_t shape[] = {total_size};
    
    if (data->elementwise_affine) {
        nc_tensor* weight = nc_tensor_ones(shape, 1, NC_F32);
        nc_tensor* bias = nc_tensor_zeros(shape, 1, NC_F32);
        if (!weight || !bias) {
            nc_tensor_free(weight);
            nc_tensor_free(bias);
            nc_module_free(m);
            return NULL;
        }
        nc_module_add_param(m, "weight", weight);
        nc_module_add_param(m, "bias", bias);
    }
    
    return m;
}