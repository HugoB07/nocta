#include "minunit.h"
#include "nocta/nocta.h"
#include <math.h>

static char* test_add() {
    size_t shape[] = {2};
    nc_tensor* a = nc_tensor_zeros(shape, 1, NC_F32);
    nc_tensor* b = nc_tensor_zeros(shape, 1, NC_F32);
    
    nc_tensor_set_flat(a, 0, 1.0f);
    nc_tensor_set_flat(b, 0, 2.0f);
    
    nc_tensor* c = nc_add(a, b);
    mu_assert("Add failed", c != NULL);
    mu_assert("Add result wrong", fabs(nc_tensor_get_flat(c, 0) - 3.0f) < 1e-6);
    
    nc_tensor_free(a); nc_tensor_free(b); nc_tensor_free(c);
    return NULL;
}

static char* test_matmul() {
    // A: 2x3, B: 3x2 -> C: 2x2
    size_t sA[] = {2, 3};
    size_t sB[] = {3, 2};
    
    nc_tensor* A = nc_tensor_ones(sA, 2, NC_F32);
    nc_tensor* B = nc_tensor_ones(sB, 2, NC_F32);
    
    nc_tensor* C = nc_matmul(A, B);
    mu_assert("Matmul failed", C != NULL);
    mu_assert("Matmul shape[0] wrong", C->shape[0] == 2);
    mu_assert("Matmul shape[1] wrong", C->shape[1] == 2);
    
    // Result should be dot product of ones vector of len 3 -> 3.0
    mu_assert("Matmul value wrong", fabs(nc_tensor_get_flat(C, 0) - 3.0f) < 1e-6);
    
    nc_tensor_free(A); nc_tensor_free(B); nc_tensor_free(C);
    return NULL;
}

static char* test_relu() {
    size_t shape[] = {3};
    nc_tensor* t = nc_tensor_zeros(shape, 1, NC_F32);
    nc_tensor_set_flat(t, 0, -1.0f);
    nc_tensor_set_flat(t, 1, 0.0f);
    nc_tensor_set_flat(t, 2, 2.0f);
    
    nc_tensor* out = nc_relu(t);
    
    mu_assert("ReLU neg failed", fabs(nc_tensor_get_flat(out, 0)) < 1e-6);
    mu_assert("ReLU zero failed", fabs(nc_tensor_get_flat(out, 1)) < 1e-6);
    mu_assert("ReLU pos failed", fabs(nc_tensor_get_flat(out, 2) - 2.0f) < 1e-6);
    
    nc_tensor_free(t); nc_tensor_free(out);
    return NULL;
}

char* test_ops_suite() {
    mu_run_test(test_add);
    mu_run_test(test_matmul);
    mu_run_test(test_relu);
    return NULL;
}
