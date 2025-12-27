#include "nocta/optim/sgd.h"
#include <string.h>
#include "nocta/core/device.h"
#ifdef NOCTA_CUDA_ENABLED
#include "nocta/cuda/cuda_kernels.h"
#endif

typedef struct {
    nc_sgd_config config;
    nc_tensor* velocity[NC_MAX_OPTIM_PARAMS];  // Momentum buffers
} nc_sgd_state;

static void sgd_step(nc_optimizer* self) {
    nc_sgd_state* state = self->state;
    nc_sgd_config* cfg = &state->config;
    
    for (size_t i = 0; i < self->n_params; i++) {
        nc_tensor* param = self->params[i];
        nc_tensor* grad = param->grad;
        
        if (!grad) continue;
        
        // Weight decay (L2 regularization)
        if (cfg->weight_decay != 0.0) {
            for (size_t j = 0; j < grad->numel; j++) {
                double g = nc_tensor_get_flat(grad, j);
                double p = nc_tensor_get_flat(param, j);
                nc_tensor_set_flat(grad, j, g + cfg->weight_decay * p);
            }
        }
        
        // Momentum
        if (cfg->momentum != 0.0) {
            nc_tensor* v = state->velocity[i];
            
            // Initialize velocity buffer if needed
            if (!v) {
                v = nc_tensor_zeros(param->shape, param->ndim, param->dtype);
#ifdef NOCTA_CUDA_ENABLED
                if (param->storage->device == NC_DEVICE_CUDA) {
                    nc_tensor_to_device(v, NC_DEVICE_CUDA);
                }
#endif
                state->velocity[i] = v;
            }
            
#ifdef NOCTA_CUDA_ENABLED
            if (param->storage->device == NC_DEVICE_CUDA && param->dtype == NC_F32) {
                // Ensure velocity is on CUDA
                if (v->storage->device != NC_DEVICE_CUDA) {
                    nc_tensor_to_device(v, NC_DEVICE_CUDA);
                }
                 
                nc_cuda_sgd_momentum_f32(
                    (float*)param->storage->cuda_data,
                    (float*)v->storage->cuda_data,
                    (const float*)grad->storage->cuda_data,
                    (float)self->lr,
                    (float)cfg->momentum,
                    (float)cfg->dampening,
                    (int)cfg->nesterov,
                    param->numel
                );
            
                continue;
            }
#endif
            
            for (size_t j = 0; j < param->numel; j++) {
                double g = nc_tensor_get_flat(grad, j);
                double vj = nc_tensor_get_flat(v, j);
                
                // v = momentum * v + (1 - dampening) * grad
                if (self->t > 0) {
                    vj = cfg->momentum * vj + (1.0 - cfg->dampening) * g;
                } else {
                    vj = g;
                }
                nc_tensor_set_flat(v, j, vj);
                
                // Nesterov momentum
                double update;
                if (cfg->nesterov) {
                    update = g + cfg->momentum * vj;
                } else {
                    update = vj;
                }
                
                // param = param - lr * update
                double p = nc_tensor_get_flat(param, j);
                nc_tensor_set_flat(param, j, p - self->lr * update);
            }
        } else {
            // Simple SGD without momentum
#ifdef NOCTA_CUDA_ENABLED
            if (param->storage->device == NC_DEVICE_CUDA && param->dtype == NC_F32) {
                nc_cuda_sgd_step_f32(
                    (float*)param->storage->cuda_data,
                    (const float*)grad->storage->cuda_data,
                    (float)self->lr,
                    param->numel
                );
                continue;
            }
#endif
            for (size_t j = 0; j < param->numel; j++) {
                double p = nc_tensor_get_flat(param, j);
                double g = nc_tensor_get_flat(grad, j);
                nc_tensor_set_flat(param, j, p - self->lr * g);
            }
        }
    }
    
    self->t++;
}

static void sgd_free_state(void* state) {
    nc_sgd_state* s = state;
    for (size_t i = 0; i < NC_MAX_OPTIM_PARAMS; i++) {
        if (s->velocity[i]) {
            nc_tensor_free(s->velocity[i]);
        }
    }
    nc_free(s);
}

nc_optimizer* nc_sgd(double lr, nc_sgd_config config) {
    nc_optimizer* opt = nc_calloc(1, sizeof(nc_optimizer));
    if (!opt) return NULL;
    
    opt->lr = lr;
    opt->step = sgd_step;
    opt->t = 0;
    
    nc_sgd_state* state = nc_calloc(1, sizeof(nc_sgd_state));
    if (!state) {
        nc_free(opt);
        return NULL;
    }
    state->config = config;
    
    opt->state = state;
    opt->free_state = sgd_free_state;
    
    return opt;
}

// Callback for collecting params
static void add_param_cb(const char* name, nc_tensor* param, void* ctx) {
    (void)name;
    nc_optimizer* opt = ctx;
    nc_optimizer_add_param(opt, param);
}

nc_optimizer* nc_sgd_from_module(nc_module* m, double lr, nc_sgd_config config) {
    nc_optimizer* opt = nc_sgd(lr, config);
    if (!opt) return NULL;
    
    nc_module_parameters(m, add_param_cb, opt);
    return opt;
}

nc_optimizer* nc_sgd_simple(nc_module* m, double lr) {
    return nc_sgd_from_module(m, lr, NC_SGD_DEFAULT);
}

nc_optimizer* nc_sgd_momentum(nc_module* m, double lr, double momentum) {
    nc_sgd_config cfg = NC_SGD_DEFAULT;
    cfg.momentum = momentum;
    return nc_sgd_from_module(m, lr, cfg);
}