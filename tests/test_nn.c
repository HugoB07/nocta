#include "minunit.h"
#include "nocta/nocta.h"
#include "nocta/nn/linear.h"
#include "nocta/nn/conv.h"
#include <math.h>

static char* test_linear() {
    // Linear 2->2, weights=ones, bias=ones
    nc_module* m = nc_linear(2, 2, true);
    
    // Set weights to 1, bias to 1 for deterministic test
    nc_tensor_fill_(nc_linear_weight(m), 1.0f);
    nc_tensor_fill_(nc_linear_bias(m), 1.0f);
    
    size_t shape[] = {1, 2};
    nc_tensor* input = nc_tensor_zeros(shape, 2, NC_F32);
    nc_tensor_set_flat(input, 0, 2.0f);
    nc_tensor_set_flat(input, 1, 3.0f);
    
    // forward = input @ weight.T + bias
    // input = [2, 3]
    // weight = [[1, 1], [1, 1]]
    // input @ weight.T = [2*1+3*1, 2*1+3*1] = [5, 5]
    // bias = [1, 1]
    // result = [6, 6]
    
    nc_tensor* out = nc_module_forward(m, input);
    
    mu_assert("Linear output wrong[0]", fabs(nc_tensor_get_flat(out, 0) - 6.0f) < 1e-6);
    mu_assert("Linear output wrong[1]", fabs(nc_tensor_get_flat(out, 1) - 6.0f) < 1e-6);
    
    nc_tensor_free(out);
    nc_tensor_free(input);
    nc_module_free(m);
    return NULL;
}

static char* test_conv2d() {
    // Input: 1 batch, 1 ch, 3x3 img. All ones.
    // Kernel: 1 out_ch, 1 in_ch, 2x2. All ones.
    // No bias.
    // Stride 1, padding 0.
    // Output should be: 
    // [[1,1], [1,1]] * [[1,1], [1,1]] = 4 for top-left
    // Output size: (3-2)/1 + 1 = 2. So 2x2 output.
    
    nc_module* conv = nc_conv2d(1, 1, 2, 1, 0, false);
    nc_tensor_fill_(nc_conv2d_weight(conv), 1.0);
    
    size_t in_shape[] = {1, 1, 3, 3};
    nc_tensor* input = nc_tensor_ones(in_shape, 4, NC_F32);
    
    nc_tensor* out = nc_module_forward(conv, input);
    
    mu_assert("Conv out dim wrong", out->ndim == 4);
    mu_assert("Conv out H wrong", out->shape[2] == 2);
    mu_assert("Conv out W wrong", out->shape[3] == 2);
    
    // Check value: sum of 2x2 ones = 4
    mu_assert("Conv val wrong", fabs(nc_tensor_get_flat(out, 0) - 4.0f) < 1e-6);
    
    nc_tensor_free(out);
    nc_tensor_free(input);
    nc_module_free(conv);
    return NULL;
}

char* test_nn_suite() {
    mu_run_test(test_linear);
    mu_run_test(test_conv2d);
    return NULL;
}
