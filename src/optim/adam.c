#include "nocta/optim/adam.h"
#include <math.h>

typedef struct {
    nc_adam_config config;
    nc_tensor* m[NC_MAX_OPTIM_PARAMS];      // First moment
    nc_tensor* v[NC_MAX_OPTIM_PARAMS];      // Second moment
    nc_tensor* v_max[NC_MAX_OPTIM_PARAMS];  // Max second moment (AMSGrad)
} nc_adam_state;

static void adam_step(nc_optimizer* self) {
    nc_adam_state* state = self->state;
    nc_adam_config* cfg = &state->config;
    
    self->t++;
    
    // Bias correction
    double bias_corr1 = 1.0 - pow(cfg->beta1, (double)self->t);
    double bias_corr2 = 1.0 - pow(cfg->beta2, (double)self->t);
    
    for (size_t i = 0; i < self->n_params; i++) {
        nc_tensor* param = self->params[i];
        nc_tensor* grad = param->grad;
        
        if (!grad) continue;
        
        // Initialize state tensors if needed
        if (!state->m[i]) {
            state->m[i] = nc_tensor_zeros(param->shape, param->ndim, param->dtype);
            state->v[i] = nc_tensor_zeros(param->shape, param->ndim, param->dtype);
            if (cfg->amsgrad) {
                state->v_max[i] = nc_tensor_zeros(param->shape, param->ndim, param->dtype);
            }
        }
        
        nc_tensor* m = state->m[i];
        nc_tensor* v = state->v[i];
        
        for (size_t j = 0; j < param->numel; j++) {
            double g = nc_tensor_get_flat(grad, j);
            double p = nc_tensor_get_flat(param, j);
            
            // Weight decay (AdamW style: decoupled)
            if (cfg->weight_decay != 0.0) {
                p = p - self->lr * cfg->weight_decay * p;
            }
            
            // Update biased first moment estimate
            double mj = nc_tensor_get_flat(m, j);
            mj = cfg->beta1 * mj + (1.0 - cfg->beta1) * g;
            nc_tensor_set_flat(m, j, mj);
            
            // Update biased second moment estimate
            double vj = nc_tensor_get_flat(v, j);
            vj = cfg->beta2 * vj + (1.0 - cfg->beta2) * g * g;
            nc_tensor_set_flat(v, j, vj);
            
            // Compute bias-corrected estimates
            double m_hat = mj / bias_corr1;
            double v_hat = vj / bias_corr2;
            
            // AMSGrad: use max of v_hat
            if (cfg->amsgrad) {
                double v_max_j = nc_tensor_get_flat(state->v_max[i], j);
                v_max_j = fmax(v_max_j, v_hat);
                nc_tensor_set_flat(state->v_max[i], j, v_max_j);
                v_hat = v_max_j;
            }
            
            // Update parameters
            p = p - self->lr * m_hat / (sqrt(v_hat) + cfg->eps);
            nc_tensor_set_flat(param, j, p);
        }
    }
}

static void adam_free_state(void* state) {
    nc_adam_state* s = state;
    for (size_t i = 0; i < NC_MAX_OPTIM_PARAMS; i++) {
        if (s->m[i]) nc_tensor_free(s->m[i]);
        if (s->v[i]) nc_tensor_free(s->v[i]);
        if (s->v_max[i]) nc_tensor_free(s->v_max[i]);
    }
    nc_free(s);
}

nc_optimizer* nc_adam(double lr, nc_adam_config config) {
    nc_optimizer* opt = nc_calloc(1, sizeof(nc_optimizer));
    if (!opt) return NULL;
    
    opt->lr = lr;
    opt->step = adam_step;
    opt->t = 0;
    
    nc_adam_state* state = nc_calloc(1, sizeof(nc_adam_state));
    if (!state) {
        nc_free(opt);
        return NULL;
    }
    state->config = config;
    
    opt->state = state;
    opt->free_state = adam_free_state;
    
    return opt;
}

// Callback for collecting params
static void add_param_cb(const char* name, nc_tensor* param, void* ctx) {
    (void)name;
    nc_optimizer* opt = ctx;
    nc_optimizer_add_param(opt, param);
}

nc_optimizer* nc_adam_from_module(nc_module* m, double lr, nc_adam_config config) {
    nc_optimizer* opt = nc_adam(lr, config);
    if (!opt) return NULL;
    
    nc_module_parameters(m, add_param_cb, opt);
    return opt;
}

nc_optimizer* nc_adam_default(nc_module* m, double lr) {
    return nc_adam_from_module(m, lr, NC_ADAM_DEFAULT);
}

nc_optimizer* nc_adamw(nc_module* m, double lr, double weight_decay) {
    nc_adam_config cfg = NC_ADAM_DEFAULT;
    cfg.weight_decay = weight_decay;
    return nc_adam_from_module(m, lr, cfg);
}