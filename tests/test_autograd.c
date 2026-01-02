#include "minunit.h"
#include "nocta/nocta.h"
#include <math.h>

static char* test_basic_backward() {
    // y = 2*x, dy/dx = 2
    size_t shape[] = {1};
    nc_tensor* x = nc_tensor_zeros(shape, 1, NC_F32);
    nc_tensor_set_flat(x, 0, 3.0f);
    nc_tensor_requires_grad_(x, true);
    
    nc_tensor* two = nc_tensor_zeros(shape, 1, NC_F32);
    nc_tensor_set_flat(two, 0, 2.0f);
    
    nc_tensor* y = nc_mul(x, two);
    
    nc_backward_scalar(y);
    
    mu_assert("Grad is NULL", x->grad != NULL);
    mu_assert("Grad value wrong", fabs(nc_tensor_get_flat(x->grad, 0) - 2.0f) < 1e-6);
    
    nc_tensor_free(x); nc_tensor_free(two); nc_tensor_free(y);
    return NULL;
}

static char* test_chain_rule() {
    // y = (x + 1)^2 at x=2
    // z = x + 1 = 3
    // y = z^2 = 9
    // dy/dx = dy/dz * dz/dx = 2z * 1 = 6
    
    size_t shape[] = {1};
    nc_tensor* x = nc_tensor_zeros(shape, 1, NC_F32);
    nc_tensor_set_flat(x, 0, 2.0f);
    nc_tensor_requires_grad_(x, true);
    
    nc_tensor* one = nc_tensor_zeros(shape, 1, NC_F32);
    nc_tensor_set_flat(one, 0, 1.0f);
    
    nc_tensor* z = nc_add(x, one);
    nc_tensor* y = nc_mul(z, z); // z^2
    
    nc_backward_scalar(y);
    
    mu_assert("Grad wrong", fabs(nc_tensor_get_flat(x->grad, 0) - 6.0f) < 1e-6);
    
    nc_tensor_free(x); nc_tensor_free(one); nc_tensor_free(z); nc_tensor_free(y);
    return NULL;
}

char* test_autograd_suite() {
    mu_run_test(test_basic_backward);
    mu_run_test(test_chain_rule);
    return NULL;
}
