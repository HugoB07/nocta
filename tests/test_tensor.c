#include "minunit.h"
#include "nocta/nocta.h"
#include <math.h>

static char* test_create() {
    size_t shape[] = {2, 3};
    nc_tensor* t = nc_tensor_empty(shape, 2, NC_F32);
    mu_assert("Creation failed", t != NULL);
    mu_assert("Wrong ndim", t->ndim == 2);
    mu_assert("Wrong shape[0]", t->shape[0] == 2);
    mu_assert("Wrong shape[1]", t->shape[1] == 3);
    mu_assert("Wrong numel", t->numel == 6);
    
    nc_tensor_free(t);
    return NULL;
}

static char* test_zeros() {
    size_t shape[] = {5};
    nc_tensor* t = nc_tensor_zeros(shape, 1, NC_F32);
    mu_assert("Zeros creation failed", t != NULL);
    
    for(size_t i=0; i<5; i++) {
        double v = nc_tensor_get_flat(t, i);
        mu_assert("Not zero", fabs(v) < 1e-6);
    }
    nc_tensor_free(t);
    return NULL;
}

static char* test_view() {
    size_t shape[] = {2, 2};
    nc_tensor* t = nc_tensor_zeros(shape, 2, NC_F32);
    nc_tensor_set_flat(t, 0, 1.0f);
    
    size_t new_shape[] = {4};
    nc_tensor* v = nc_tensor_reshape(t, new_shape, 1);
    
    mu_assert("View creation failed", v != NULL);
    mu_assert("View shape wrong", v->shape[0] == 4);
    mu_assert("View content mismatch", fabs(nc_tensor_get_flat(v, 0) - 1.0f) < 1e-6);
    
    // Test shared storage
    nc_tensor_set_flat(v, 1, 2.0f);
    mu_assert("Shared storage check failed", fabs(nc_tensor_get_flat(t, 1) - 2.0f) < 1e-6);
    
    nc_tensor_free(t);
    nc_tensor_free(v);
    return NULL;
}

char* test_tensor_suite() {
    mu_run_test(test_create);
    mu_run_test(test_zeros);
    mu_run_test(test_view);
    return NULL;
}
