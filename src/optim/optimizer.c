#include "nocta/optim/optimizer.h"

void nc_optimizer_free(nc_optimizer* opt) {
    if (!opt) return;
    
    if (opt->state && opt->free_state) {
        opt->free_state(opt->state);
    }
    
    nc_free(opt);
}

void nc_optimizer_step(nc_optimizer* opt) {
    if (opt && opt->step) {
        opt->step(opt);
    }
}

void nc_optimizer_zero_grad(nc_optimizer* opt) {
    if (!opt) return;
    
    for (size_t i = 0; i < opt->n_params; i++) {
        nc_tensor_zero_grad_(opt->params[i]);
    }
}

void nc_optimizer_set_lr(nc_optimizer* opt, double lr) {
    if (opt) opt->lr = lr;
}

double nc_optimizer_get_lr(nc_optimizer* opt) {
    return opt ? opt->lr : 0.0;
}

static void add_param_cb(const char* name, nc_tensor* param, void* ctx) {
    (void)name;
    nc_optimizer* opt = ctx;
    nc_optimizer_add_param(opt, param);
}

void nc_optimizer_add_module(nc_optimizer* opt, nc_module* m) {
    if (!opt || !m) return;
    nc_module_parameters(m, add_param_cb, opt);
}

void nc_optimizer_add_param(nc_optimizer* opt, nc_tensor* param) {
    if (!opt || !param) return;
    if (opt->n_params >= NC_MAX_OPTIM_PARAMS) return;
    
    opt->params[opt->n_params++] = param;
}