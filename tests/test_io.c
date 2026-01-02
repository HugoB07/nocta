#include "minunit.h"
#include "nocta/nocta.h"
#include <math.h>

static char* test_save_load_tensor() {
    const char* filename = "test_tensor.ncta";
    
    size_t shape[] = {2, 2};
    nc_tensor* t1 = nc_tensor_zeros(shape, 2, NC_F32);
    nc_tensor_set_flat(t1, 0, 1.23f);
    nc_tensor_set_flat(t1, 3, 4.56f);
    
    // Save
    nc_tensor_save(t1, filename);
    
    // Load
    nc_tensor* t2 = nc_tensor_load(filename);
    
    mu_assert("Load creation failed", t2 != NULL);
    mu_assert("Load ndim wrong", t2->ndim == 2);
    mu_assert("Load value[0] wrong", fabs(nc_tensor_get_flat(t2, 0) - 1.23f) < 1e-6);
    mu_assert("Load value[3] wrong", fabs(nc_tensor_get_flat(t2, 3) - 4.56f) < 1e-6);
    
    nc_tensor_free(t1);
    nc_tensor_free(t2);
    remove(filename);
    return NULL;
}

char* test_io_suite() {
    mu_run_test(test_save_load_tensor);
    return NULL;
}
