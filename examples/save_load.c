// Save/Load Example - Demonstrating Nocta's serialization
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "nocta/nocta.h"

// Forward pass helper
static nc_tensor* forward(nc_tensor* X, nc_tensor* W1, nc_tensor* b1, 
                          nc_tensor* W2, nc_tensor* b2) {
    nc_tensor* h1 = nc_matmul(X, W1);
    nc_tensor* h1_bias = nc_add(h1, b1);
    nc_tensor* h1_act = nc_relu(h1_bias);
    nc_tensor* h2 = nc_matmul(h1_act, W2);
    nc_tensor* h2_bias = nc_add(h2, b2);
    nc_tensor* pred = nc_sigmoid(h2_bias);
    
    nc_tensor_free(h1);
    nc_tensor_free(h1_bias);
    nc_tensor_free(h1_act);
    nc_tensor_free(h2);
    nc_tensor_free(h2_bias);
    
    return pred;
}

int main(void) {
    srand((unsigned)time(NULL));
    nc_init();
    
    printf("=== Nocta Save/Load Example ===\n\n");
    
    // XOR dataset
    float X_data[] = {0, 0,  0, 1,  1, 0,  1, 1};
    float Y_data[] = {0, 1, 1, 0};
    
    size_t x_shape[] = {4, 2};
    size_t y_shape[] = {4, 1};
    
    nc_tensor* X = nc_tensor_from_data(X_data, x_shape, 2, NC_F32);
    nc_tensor* Y = nc_tensor_from_data(Y_data, y_shape, 2, NC_F32);
    
    // ========================================
    // Part 1: Train a model and save it
    // ========================================
    printf("--- Part 1: Training and Saving ---\n\n");
    
    size_t hidden = 8;
    size_t w1_shape[] = {2, hidden};
    size_t b1_shape[] = {1, hidden};
    size_t w2_shape[] = {hidden, 1};
    size_t b2_shape[] = {1, 1};
    
    nc_tensor* W1 = nc_tensor_randn(w1_shape, 2, NC_F32);
    nc_tensor* b1 = nc_tensor_zeros(b1_shape, 2, NC_F32);
    nc_tensor* W2 = nc_tensor_randn(w2_shape, 2, NC_F32);
    nc_tensor* b2 = nc_tensor_zeros(b2_shape, 2, NC_F32);
    
    nc_mul_scalar_(W1, 0.5);
    nc_mul_scalar_(W2, 0.5);
    
    W1->is_leaf = true; b1->is_leaf = true;
    W2->is_leaf = true; b2->is_leaf = true;
    nc_tensor_requires_grad_(W1, true);
    nc_tensor_requires_grad_(b1, true);
    nc_tensor_requires_grad_(W2, true);
    nc_tensor_requires_grad_(b2, true);
    
    double lr = 1.0;
    int epochs = 3000;
    
    printf("Training for %d epochs...\n", epochs);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        nc_tensor_zero_grad_(W1);
        nc_tensor_zero_grad_(b1);
        nc_tensor_zero_grad_(W2);
        nc_tensor_zero_grad_(b2);
        
        nc_tensor* h1 = nc_matmul(X, W1);
        nc_tensor* h1_bias = nc_add(h1, b1);
        nc_tensor* h1_act = nc_relu(h1_bias);
        nc_tensor* h2 = nc_matmul(h1_act, W2);
        nc_tensor* h2_bias = nc_add(h2, b2);
        nc_tensor* pred = nc_sigmoid(h2_bias);
        
        nc_tensor* diff = nc_sub(pred, Y);
        nc_tensor* diff_sq = nc_mul(diff, diff);
        nc_tensor* loss = nc_mean_all(diff_sq);
        
        nc_backward_scalar(loss);
        
        // SGD update
        for (size_t i = 0; i < W1->numel; i++) {
            nc_tensor_set_flat(W1, i, nc_tensor_get_flat(W1, i) - lr * nc_tensor_get_flat(W1->grad, i));
        }
        for (size_t i = 0; i < b1->numel; i++) {
            nc_tensor_set_flat(b1, i, nc_tensor_get_flat(b1, i) - lr * nc_tensor_get_flat(b1->grad, i));
        }
        for (size_t i = 0; i < W2->numel; i++) {
            nc_tensor_set_flat(W2, i, nc_tensor_get_flat(W2, i) - lr * nc_tensor_get_flat(W2->grad, i));
        }
        for (size_t i = 0; i < b2->numel; i++) {
            nc_tensor_set_flat(b2, i, nc_tensor_get_flat(b2, i) - lr * nc_tensor_get_flat(b2->grad, i));
        }
        
        if (epoch % 500 == 0 || epoch == epochs - 1) {
            printf("Epoch %4d | Loss: %.6f\n", epoch, nc_tensor_get_flat(loss, 0));
        }
        
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
    
    // Show predictions before saving
    printf("\nPredictions before saving:\n");
    nc_tensor* pred = forward(X, W1, b1, W2, b2);
    for (size_t i = 0; i < 4; i++) {
        printf("  %.0f XOR %.0f = %.0f (pred: %.4f)\n",
            nc_tensor_get2(X, i, 0), nc_tensor_get2(X, i, 1),
            nc_tensor_get2(Y, i, 0), nc_tensor_get2(pred, i, 0));
    }
    nc_tensor_free(pred);
    
    // Save model using state dict
    printf("\nSaving model to 'xor_model.ncta'...\n");
    
    nc_state_dict* sd = nc_state_dict_create();
    nc_state_dict_add(sd, "W1", nc_tensor_clone(W1));
    nc_state_dict_add(sd, "b1", nc_tensor_clone(b1));
    nc_state_dict_add(sd, "W2", nc_tensor_clone(W2));
    nc_state_dict_add(sd, "b2", nc_tensor_clone(b2));
    
    nc_error err = nc_state_dict_save(sd, "xor_model.ncta");
    if (err == NC_OK) {
        printf("Model saved successfully!\n");
    } else {
        printf("Failed to save model: %s\n", nc_error_string(err));
    }
    nc_state_dict_free(sd);
    
    // Show file info
    printf("\nFile info:\n");
    nc_file_info("xor_model.ncta");
    
    // Free training weights
    nc_tensor_free(W1);
    nc_tensor_free(b1);
    nc_tensor_free(W2);
    nc_tensor_free(b2);
    
    // ========================================
    // Part 2: Load model and make predictions
    // ========================================
    printf("\n--- Part 2: Loading and Inference ---\n\n");
    
    // Create new tensors (uninitialized)
    nc_tensor* W1_loaded = nc_tensor_zeros(w1_shape, 2, NC_F32);
    nc_tensor* b1_loaded = nc_tensor_zeros(b1_shape, 2, NC_F32);
    nc_tensor* W2_loaded = nc_tensor_zeros(w2_shape, 2, NC_F32);
    nc_tensor* b2_loaded = nc_tensor_zeros(b2_shape, 2, NC_F32);
    
    // Load state dict
    printf("Loading model from 'xor_model.ncta'...\n");
    nc_state_dict* loaded_sd = nc_state_dict_load("xor_model.ncta");
    
    if (loaded_sd) {
        printf("Loaded %zu tensors\n", loaded_sd->n_tensors);
        
        // Copy loaded weights
        nc_tensor* t;
        if ((t = nc_state_dict_get(loaded_sd, "W1"))) nc_tensor_copy_(W1_loaded, t);
        if ((t = nc_state_dict_get(loaded_sd, "b1"))) nc_tensor_copy_(b1_loaded, t);
        if ((t = nc_state_dict_get(loaded_sd, "W2"))) nc_tensor_copy_(W2_loaded, t);
        if ((t = nc_state_dict_get(loaded_sd, "b2"))) nc_tensor_copy_(b2_loaded, t);
        
        nc_state_dict_free(loaded_sd);
        
        // Make predictions with loaded model
        printf("\nPredictions after loading:\n");
        pred = forward(X, W1_loaded, b1_loaded, W2_loaded, b2_loaded);
        for (size_t i = 0; i < 4; i++) {
            printf("  %.0f XOR %.0f = %.0f (pred: %.4f)\n",
                nc_tensor_get2(X, i, 0), nc_tensor_get2(X, i, 1),
                nc_tensor_get2(Y, i, 0), nc_tensor_get2(pred, i, 0));
        }
        nc_tensor_free(pred);
    } else {
        printf("Failed to load model!\n");
    }
    
    // ========================================
    // Part 3: Demonstrate single tensor save/load
    // ========================================
    printf("\n--- Part 3: Single Tensor Save/Load ---\n\n");
    
    printf("Saving W1 tensor...\n");
    nc_tensor_save(W1_loaded, "w1_tensor.ncta");
    
    printf("Loading W1 tensor...\n");
    nc_tensor* w1_reloaded = nc_tensor_load("w1_tensor.ncta");
    if (w1_reloaded) {
        printf("Loaded tensor: ");
        nc_tensor_print_shape(w1_reloaded);
        
        // Verify values match
        bool match = true;
        for (size_t i = 0; i < W1_loaded->numel && match; i++) {
            if (fabs(nc_tensor_get_flat(W1_loaded, i) - nc_tensor_get_flat(w1_reloaded, i)) > 1e-6) {
                match = false;
            }
        }
        printf("Values match: %s\n", match ? "YES" : "NO");
        nc_tensor_free(w1_reloaded);
    }
    
    // ========================================
    // Part 4: Checkpoint save/load
    // ========================================
    printf("\n--- Part 4: Checkpoint Save/Load ---\n\n");
    
    nc_checkpoint ckpt = {0};
    ckpt.model_state = nc_state_dict_create();
    nc_state_dict_add(ckpt.model_state, "W1", nc_tensor_clone(W1_loaded));
    nc_state_dict_add(ckpt.model_state, "b1", nc_tensor_clone(b1_loaded));
    nc_state_dict_add(ckpt.model_state, "W2", nc_tensor_clone(W2_loaded));
    nc_state_dict_add(ckpt.model_state, "b2", nc_tensor_clone(b2_loaded));
    ckpt.epoch = 3000;
    ckpt.loss = 0.001;
    
    printf("Saving checkpoint...\n");
    nc_checkpoint_save(&ckpt, "xor_checkpoint.ncta");
    nc_state_dict_free(ckpt.model_state);
    
    printf("Loading checkpoint...\n");
    nc_checkpoint* loaded_ckpt = nc_checkpoint_load("xor_checkpoint.ncta");
    if (loaded_ckpt) {
        printf("  Epoch: %zu\n", loaded_ckpt->epoch);
        printf("  Loss: %.6f\n", loaded_ckpt->loss);
        printf("  Model tensors: %zu\n", 
            loaded_ckpt->model_state ? loaded_ckpt->model_state->n_tensors : 0);
        nc_checkpoint_free(loaded_ckpt);
    }
    
    // Cleanup
    nc_tensor_free(W1_loaded);
    nc_tensor_free(b1_loaded);
    nc_tensor_free(W2_loaded);
    nc_tensor_free(b2_loaded);
    nc_tensor_free(X);
    nc_tensor_free(Y);
    
    printf("\n");
    nc_cleanup();
    
    // Clean up temp files
    remove("xor_model.ncta");
    remove("w1_tensor.ncta");
    remove("xor_checkpoint.ncta");
    
    printf("\nDone!\n");
    return 0;
}