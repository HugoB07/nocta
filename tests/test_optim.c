#include "minunit.h"
#include "nocta/nocta.h"
#include "nocta/optim/sgd.h"
#include <math.h>

static char* test_sgd_step() {
    // w = 1.0, grad = 0.5, lr = 0.1
    // w_new = 1.0 - 0.1 * 0.5 = 0.95
    
    size_t shape[] = {1};
    nc_tensor* w = nc_tensor_zeros(shape, 1, NC_F32);
    nc_tensor_set_flat(w, 0, 1.0f);
    nc_tensor_requires_grad_(w, true);
    
    // Manually set gradient
    w->grad = nc_tensor_zeros(shape, 1, NC_F32);
    nc_tensor_set_flat(w->grad, 0, 0.5f);
    
    nc_optimizer* opt = nc_sgd(0.1, NC_SGD_DEFAULT);
    nc_optimizer_add_param(opt, w);
    
    nc_optimizer_step(opt);
    
    mu_assert("SGD update wrong", fabs(nc_tensor_get_flat(w, 0) - 0.95f) < 1e-6);
    
    nc_optimizer_free(opt);
    nc_tensor_free(w);
    return NULL;
}

static char* test_sgd_momentum() {
    // w = 0, grad = 1, lr = 0.1, momentum = 0.9
    // v0 = 0
    // v1 = 0.9*v0 + grad = 1
    // w1 = w0 - lr*v1 = -0.1
    
    size_t shape[] = {1};
    nc_tensor* w = nc_tensor_zeros(shape, 1, NC_F32);
    nc_tensor_requires_grad_(w, true);
    
    // Gradient 1
    w->grad = nc_tensor_zeros(shape, 1, NC_F32);
    nc_tensor_set_flat(w->grad, 0, 1.0f);
    
    nc_sgd_config conf = NC_SGD_DEFAULT;
    conf.momentum = 0.9;
    nc_optimizer* opt = nc_sgd(0.1, conf);
    nc_optimizer_add_param(opt, w);
    
    nc_optimizer_step(opt);
    
    mu_assert("SGD momentum step 1 wrong", fabs(nc_tensor_get_flat(w, 0) + 0.1f) < 1e-6);
    
    // Step 2
    // v2 = 0.9*v1 + grad = 0.9*1 + 1 = 1.9
    // w2 = w1 - lr*v2 = -0.1 - 0.1*1.9 = -0.29
    nc_optimizer_step(opt);
    
    mu_assert("SGD momentum step 2 wrong", fabs(nc_tensor_get_flat(w, 0) + 0.29f) < 1e-6);
    
    nc_optimizer_free(opt);
    nc_tensor_free(w);
    return NULL;
}

char* test_optim_suite() {
    mu_run_test(test_sgd_step);
    mu_run_test(test_sgd_momentum);
    return NULL;
}
