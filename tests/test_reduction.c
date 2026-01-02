#include "minunit.h"
#include "nocta/nocta.h"
#include <math.h>

static char* test_sum() {
    size_t shape[] = {2, 3};
    nc_tensor* t = nc_tensor_ones(shape, 2, NC_F32);
    
    // Sum all = 6
    nc_tensor* s = nc_sum_all(t);
    mu_assert("Sum all wrong", fabs(nc_tensor_get_flat(s, 0) - 6.0f) < 1e-6);
    nc_tensor_free(s);
    
    // Sum dim 0 -> shape [3] -> [2, 2, 2]
    s = nc_sum(t, 0, false);
    mu_assert("Sum dim 0 ndim wrong", s->ndim == 1);
    mu_assert("Sum dim 0 shape wrong", s->shape[0] == 3);
    mu_assert("Sum dim 0 val wrong", fabs(nc_tensor_get_flat(s, 0) - 2.0f) < 1e-6);
    
    nc_tensor_free(s);
    nc_tensor_free(t);
    return NULL;
}

static char* test_mean() {
    size_t shape[] = {4};
    nc_tensor* t = nc_tensor_zeros(shape, 1, NC_F32);
    nc_tensor_set_flat(t, 0, 2.0f);
    nc_tensor_set_flat(t, 1, 4.0f);
    nc_tensor_set_flat(t, 2, 6.0f);
    nc_tensor_set_flat(t, 3, 8.0f);
    
    // Mean = 20 / 4 = 5
    nc_tensor* m = nc_mean_all(t);
    mu_assert("Mean wrong", fabs(nc_tensor_get_flat(m, 0) - 5.0f) < 1e-6);
    
    nc_tensor_free(m);
    nc_tensor_free(t);
    return NULL;
}

static char* test_argmax() {
    size_t shape[] = {4};
    nc_tensor* t = nc_tensor_zeros(shape, 1, NC_F32);
    nc_tensor_set_flat(t, 2, 10.0f); // Max at index 2
    
    nc_tensor* idx = nc_argmax(t, 0, false);
    
    // Argmax returns float indices in this lib currently? Or int64?
    // Based on mnist.c it seems to return a tensor that we can read.
    // Assuming float storage for simplicity unless checked. 
    // Wait, mnist.c uses nc_tensor_get1 which returns double, so it works.
    
    mu_assert("Argmax wrong", fabs(nc_tensor_get_flat(idx, 0) - 2.0f) < 1e-6);
    
    nc_tensor_free(idx);
    nc_tensor_free(t);
    return NULL;
}

char* test_reduction_suite() {
    mu_run_test(test_sum);
    mu_run_test(test_mean);
    mu_run_test(test_argmax);
    return NULL;
}
