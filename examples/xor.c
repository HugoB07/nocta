// XOR Example - Testing Nocta's autograd
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "nocta/nocta.h"

int main(void) {
    srand((unsigned)time(NULL));
    nc_init();
    
    printf("=== Nocta XOR Example (Autograd) ===\n\n");
    
    // XOR dataset
    float X_data[] = {0, 0,  0, 1,  1, 0,  1, 1};
    float Y_data[] = {0, 1, 1, 0};
    
    size_t x_shape[] = {4, 2};
    size_t y_shape[] = {4, 1};
    
    nc_tensor* X = nc_tensor_from_data(X_data, x_shape, 2, NC_F32);
    nc_tensor* Y = nc_tensor_from_data(Y_data, y_shape, 2, NC_F32);
    
    // Network: 2 -> 8 -> 1
    size_t hidden = 8;
    
    // Use shapes that don't require transpose in forward pass
    size_t w1_shape[] = {2, hidden};  // (2, 8) so X @ W1 = (4,2) @ (2,8) = (4,8)
    size_t b1_shape[] = {1, hidden};  // (1, 8)
    size_t w2_shape[] = {hidden, 1};  // (8, 1) so h @ W2 = (4,8) @ (8,1) = (4,1)
    size_t b2_shape[] = {1, 1};       // (1, 1)
    
    nc_tensor* W1 = nc_tensor_randn(w1_shape, 2, NC_F32);
    nc_tensor* b1 = nc_tensor_zeros(b1_shape, 2, NC_F32);
    nc_tensor* W2 = nc_tensor_randn(w2_shape, 2, NC_F32);
    nc_tensor* b2 = nc_tensor_zeros(b2_shape, 2, NC_F32);
    
    nc_mul_scalar_(W1, 0.5);
    nc_mul_scalar_(W2, 0.5);
    
    // Enable gradients
    W1->is_leaf = true;
    b1->is_leaf = true;
    W2->is_leaf = true;
    b2->is_leaf = true;
    
    nc_tensor_requires_grad_(W1, true);
    nc_tensor_requires_grad_(b1, true);
    nc_tensor_requires_grad_(W2, true);
    nc_tensor_requires_grad_(b2, true);
    
    printf("W1: "); nc_tensor_print_shape(W1);
    printf("b1: "); nc_tensor_print_shape(b1);
    printf("W2: "); nc_tensor_print_shape(W2);
    printf("b2: "); nc_tensor_print_shape(b2);

    size_t n_params = W1->numel + b1->numel + W2->numel + b2->numel;
    printf("-------------------------------\n");
    printf("Total parameters: %zu\n", n_params);
    
    double lr = 1.0;
    int epochs = 5000;
    
    printf("\nTraining...\n");
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Zero gradients
        nc_tensor_zero_grad_(W1);
        nc_tensor_zero_grad_(b1);
        nc_tensor_zero_grad_(W2);
        nc_tensor_zero_grad_(b2);
        
        // Forward: h = relu(X @ W1 + b1)
        nc_tensor* h1 = nc_matmul(X, W1);            // (4,2) @ (2,8) = (4,8)
        nc_tensor* h1_bias = nc_add(h1, b1);         // (4,8) + (1,8) = (4,8)
        nc_tensor* h1_act = nc_relu(h1_bias);        // (4,8)
        
        // Forward: out = sigmoid(h @ W2 + b2)
        nc_tensor* h2 = nc_matmul(h1_act, W2);       // (4,8) @ (8,1) = (4,1)
        nc_tensor* h2_bias = nc_add(h2, b2);         // (4,1) + (1,1) = (4,1)
        nc_tensor* pred = nc_sigmoid(h2_bias);       // (4,1)
        
        // Loss: MSE = mean((pred - Y)^2)
        nc_tensor* diff = nc_sub(pred, Y);
        nc_tensor* diff_sq = nc_mul(diff, diff);
        nc_tensor* loss = nc_mean_all(diff_sq);
        
        // Backward
        nc_backward_scalar(loss);
        
        // Update weights using gradients
        if (W1->grad) {
            for (size_t i = 0; i < W1->numel; i++) {
                double w = nc_tensor_get_flat(W1, i);
                double g = nc_tensor_get_flat(W1->grad, i);
                nc_tensor_set_flat(W1, i, w - lr * g);
            }
        }
        if (b1->grad) {
            for (size_t i = 0; i < b1->numel; i++) {
                double w = nc_tensor_get_flat(b1, i);
                double g = nc_tensor_get_flat(b1->grad, i);
                nc_tensor_set_flat(b1, i, w - lr * g);
            }
        }
        if (W2->grad) {
            for (size_t i = 0; i < W2->numel; i++) {
                double w = nc_tensor_get_flat(W2, i);
                double g = nc_tensor_get_flat(W2->grad, i);
                nc_tensor_set_flat(W2, i, w - lr * g);
            }
        }
        if (b2->grad) {
            for (size_t i = 0; i < b2->numel; i++) {
                double w = nc_tensor_get_flat(b2, i);
                double g = nc_tensor_get_flat(b2->grad, i);
                nc_tensor_set_flat(b2, i, w - lr * g);
            }
        }
        
        if (epoch % 500 == 0 || epoch == epochs - 1) {
            double loss_val = nc_tensor_get_flat(loss, 0);
            double grad_sum = 0;
            if (W1->grad) {
                for (size_t i = 0; i < W1->grad->numel; i++) {
                    grad_sum += fabs(nc_tensor_get_flat(W1->grad, i));
                }
            }
            printf("Epoch %5d | Loss: %.6f | grad_sum: %.4f\n", epoch, loss_val, grad_sum);
        }
        
        // Free intermediate tensors
        nc_tensor_free(h1);
        nc_tensor_free(h1_bias);
        nc_tensor_free(h1_act);
        nc_tensor_free(h2);
        nc_tensor_free(h2_bias);
        nc_tensor_free(pred);
        nc_tensor_free(diff);
        nc_tensor_free(diff_sq);
        nc_tensor_free(loss);
    }
    
    // Final predictions
    printf("\n=== Final Predictions ===\n");
    nc_tensor* h1 = nc_matmul(X, W1);
    nc_tensor* h1_bias = nc_add(h1, b1);
    nc_tensor* h1_act = nc_relu(h1_bias);
    nc_tensor* h2 = nc_matmul(h1_act, W2);
    nc_tensor* h2_bias = nc_add(h2, b2);
    nc_tensor* pred = nc_sigmoid(h2_bias);
    
    for (size_t i = 0; i < 4; i++) {
        double x0 = nc_tensor_get2(X, i, 0);
        double x1 = nc_tensor_get2(X, i, 1);
        double y_true = nc_tensor_get2(Y, i, 0);
        double y_pred = nc_tensor_get2(pred, i, 0);
        printf("  %.0f XOR %.0f = %.0f (pred: %.4f)\n", x0, x1, y_true, y_pred);
    }
    
    nc_tensor_free(h1);
    nc_tensor_free(h1_bias);
    nc_tensor_free(h1_act);
    nc_tensor_free(h2);
    nc_tensor_free(h2_bias);
    nc_tensor_free(pred);
    
    // Cleanup
    nc_tensor_free(X);
    nc_tensor_free(Y);
    nc_tensor_free(W1);
    nc_tensor_free(b1);
    nc_tensor_free(W2);
    nc_tensor_free(b2);
    
    printf("\n");
    nc_cleanup();
    
    return 0;
}