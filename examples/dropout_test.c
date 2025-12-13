#include "nocta/nocta.h"
#include <stdio.h>
#include <math.h>

int main() {
    printf("Running Dropout Test...\n");
    
    // Create input
    size_t shape[] = {10, 10};
    nc_tensor* x = nc_tensor_ones(shape, 2, NC_F32);
    nc_tensor_requires_grad_(x, true);
    
    // Create dropout module
    double p = 0.5;
    nc_module* dropout = nc_dropout(p);
    
    // Test 1: Training mode (should drop approx 50%)
    nc_module_train(dropout, true);
    nc_tensor* out_train = nc_module_forward(dropout, x);
    
    double sum = nc_tensor_get_flat(nc_sum_all(out_train), 0);
    double expected_sum = 100.0; // Since input is all ones, sum is 100. Expect approx 100 because of scaling.
    // With p=0.5, we keep 50% elements, but scale them by 2. So sum should be roughly same.
    
    printf("Input sum: 100.0\n");
    printf("Output sum (training): %f\n", sum);
    
    // Check if some are zero
    int zeros = 0;
    for (size_t i = 0; i < 100; i++) {
        if (nc_tensor_get_flat(out_train, i) == 0.0) {
            zeros++;
        }
    }
    printf("Zeros count (expected ~50): %d\n", zeros);
    
    // Test 2: Eval mode (should be identity)
    nc_module_train(dropout, false);
    nc_tensor* out_eval = nc_module_forward(dropout, x);
    
    double sum_eval = nc_tensor_get_flat(nc_sum_all(out_eval), 0);
    printf("Output sum (eval): %f\n", sum_eval);
    
    if (fabs(sum_eval - 100.0) < 1e-5) {
        printf("Eval mode check: PASSED\n");
    } else {
        printf("Eval mode check: FAILED\n");
    }
    
    // Test 3: Backward
    nc_module_train(dropout, true);
    nc_tensor* out_back = nc_module_forward(dropout, x);
    nc_tensor* loss = nc_mean_all(out_back);
    
    nc_backward_scalar(loss);
    
    // Gradient should be mask * scale / numel
    // mask is 0 or 1. scale is 2. numel is 100.
    // So grad entries should be 0 or 0.02.
    
    nc_tensor* grad = x->grad;
    int grad_zeros = 0;
    int grad_nonzeros = 0;
    
    for (size_t i = 0; i < 100; i++) {
        double g = nc_tensor_get_flat(grad, i);
        if (g == 0.0) {
            grad_zeros++;
        } else if (fabs(g - (1.0/(1.0-p))/100.0) < 1e-5) {
            grad_nonzeros++;
        } else {
            printf("Unexpected gradient value: %f\n", g);
        }
    }
    
    printf("Gradient zeros: %d\n", grad_zeros);
    printf("Gradient non-zeros: %d\n", grad_nonzeros);
    
    if (grad_zeros + grad_nonzeros == 100) {
        printf("Backward check: PASSED\n");
    } else {
        printf("Backward check: FAILED\n");
    }
    
    // Cleanup
    nc_module_free(dropout);
    nc_tensor_free(x);
    nc_tensor_free(out_train);
    nc_tensor_free(out_eval);
    nc_tensor_free(out_back);
    nc_tensor_free(loss);
    
    return 0;
}
