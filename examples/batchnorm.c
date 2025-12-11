// BatchNorm Example - Demonstrating Batch Normalization in Nocta
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "nocta/nocta.h"
#include "nocta/nn/batchnorm.h"

// Simple CNN with BatchNorm: Conv -> BN -> ReLU -> Pool -> FC
typedef struct {
    nc_module *conv1, *bn1, *pool, *fc;
} simple_cnn;

static simple_cnn* cnn_create(void) {
    simple_cnn* m = nc_alloc(sizeof(simple_cnn));
    m->conv1 = nc_conv2d(1, 8, 3, 1, 1, false);  // No bias before BN
    m->bn1 = nc_batchnorm2d(8);
    m->pool = nc_maxpool2d_simple(2);
    // After conv(28x28, pad=1) -> 28x28, after pool(2) -> 14x14
    m->fc = nc_linear(8 * 14 * 14, 10, true);  // 8 channels * 14 * 14 = 1568
    return m;
}

static void cnn_free(simple_cnn* m) {
    nc_module_free(m->conv1);
    nc_module_free(m->bn1);
    nc_module_free(m->pool);
    nc_module_free(m->fc);
    nc_free(m);
}

static void cnn_train_mode(simple_cnn* m, bool training) {
    nc_module_train(m->conv1, training);
    nc_module_train(m->bn1, training);
    nc_module_train(m->pool, training);
    nc_module_train(m->fc, training);
}

static nc_tensor* cnn_forward(simple_cnn* m, nc_tensor* x) {
    // Conv -> BN -> ReLU -> Pool
    nc_tensor* h = nc_module_forward(m->conv1, x);
    if (!h) { printf("conv1 failed\n"); return NULL; }
    
    nc_tensor* bn = nc_module_forward(m->bn1, h);
    nc_tensor_free(h);
    if (!bn) { printf("bn1 failed\n"); return NULL; }
    
    h = nc_relu(bn);
    nc_tensor_free(bn);
    if (!h) { printf("relu failed\n"); return NULL; }
    
    nc_tensor* pooled = nc_module_forward(m->pool, h);
    nc_tensor_free(h);
    if (!pooled) { printf("pool failed\n"); return NULL; }
    
    // Flatten: (batch, channels, H, W) -> (batch, channels*H*W)
    size_t batch = pooled->shape[0];
    size_t channels = pooled->shape[1];
    size_t pH = pooled->shape[2];
    size_t pW = pooled->shape[3];
    size_t flat_size = channels * pH * pW;
    
    size_t flat_shape[] = {batch, flat_size};
    nc_tensor* flat = nc_tensor_empty(flat_shape, 2, pooled->dtype);
    if (!flat) { nc_tensor_free(pooled); return NULL; }
    
    for (size_t b = 0; b < batch; b++) {
        size_t idx = 0;
        for (size_t c = 0; c < channels; c++) {
            for (size_t i = 0; i < pH; i++) {
                for (size_t j = 0; j < pW; j++) {
                    nc_tensor_set2(flat, b, idx++, nc_tensor_get4(pooled, b, c, i, j));
                }
            }
        }
    }
    nc_tensor_free(pooled);
    
    // FC
    nc_tensor* out = nc_module_forward(m->fc, flat);
    nc_tensor_free(flat);
    
    return out;
}

// Test BatchNorm2D standalone
static void test_batchnorm2d(void) {
    printf("=== Testing BatchNorm2D ===\n\n");
    
    // Create a small 4D tensor: (2, 3, 4, 4) - batch=2, channels=3, H=W=4
    size_t shape[] = {2, 3, 4, 4};
    nc_tensor* x = nc_tensor_randn(shape, 4, NC_F32);
    
    // Scale up the first channel to have different statistics
    for (size_t b = 0; b < 2; b++) {
        for (size_t h = 0; h < 4; h++) {
            for (size_t w = 0; w < 4; w++) {
                double v = nc_tensor_get4(x, b, 0, h, w);
                nc_tensor_set4(x, b, 0, h, w, v * 10.0 + 5.0);  // mean~5, std~10
            }
        }
    }
    
    printf("Input stats per channel (before BN):\n");
    for (size_t c = 0; c < 3; c++) {
        double sum = 0, sum_sq = 0;
        size_t count = 2 * 4 * 4;
        for (size_t b = 0; b < 2; b++) {
            for (size_t h = 0; h < 4; h++) {
                for (size_t w = 0; w < 4; w++) {
                    double v = nc_tensor_get4(x, b, c, h, w);
                    sum += v;
                    sum_sq += v * v;
                }
            }
        }
        double mean = sum / count;
        double var = sum_sq / count - mean * mean;
        printf("  Channel %zu: mean=%.3f, std=%.3f\n", c, mean, sqrt(fmax(0, var)));
    }
    
    // Create BatchNorm layer
    nc_module* bn = nc_batchnorm2d(3);
    nc_module_train(bn, true);
    
    // Forward pass
    nc_tensor* y = nc_module_forward(bn, x);
    
    printf("\nOutput stats per channel (after BN, training=true):\n");
    for (size_t c = 0; c < 3; c++) {
        double sum = 0, sum_sq = 0;
        size_t count = 2 * 4 * 4;
        for (size_t b = 0; b < 2; b++) {
            for (size_t h = 0; h < 4; h++) {
                for (size_t w = 0; w < 4; w++) {
                    double v = nc_tensor_get4(y, b, c, h, w);
                    sum += v;
                    sum_sq += v * v;
                }
            }
        }
        double mean = sum / count;
        double var = sum_sq / count - mean * mean;
        printf("  Channel %zu: mean=%.4f, std=%.4f (expected: ~0, ~1)\n", c, mean, sqrt(fmax(0, var)));
    }
    
    // Check running statistics
    nc_tensor* running_mean = nc_batchnorm2d_running_mean(bn);
    nc_tensor* running_var = nc_batchnorm2d_running_var(bn);
    
    printf("\nRunning statistics (after 1 batch, momentum=0.1):\n");
    for (size_t c = 0; c < 3; c++) {
        printf("  Channel %zu: running_mean=%.3f, running_var=%.3f\n",
               c, nc_tensor_get1(running_mean, c), nc_tensor_get1(running_var, c));
    }
    
    // Test eval mode
    nc_tensor_free(y);
    nc_module_train(bn, false);
    y = nc_module_forward(bn, x);
    
    printf("\nOutput stats (eval mode, using running stats):\n");
    for (size_t c = 0; c < 3; c++) {
        double sum = 0, sum_sq = 0;
        size_t count = 2 * 4 * 4;
        for (size_t b = 0; b < 2; b++) {
            for (size_t h = 0; h < 4; h++) {
                for (size_t w = 0; w < 4; w++) {
                    double v = nc_tensor_get4(y, b, c, h, w);
                    sum += v;
                    sum_sq += v * v;
                }
            }
        }
        double mean = sum / count;
        double var = sum_sq / count - mean * mean;
        printf("  Channel %zu: mean=%.3f, std=%.3f\n", c, mean, sqrt(fmax(0, var)));
    }
    
    nc_tensor_free(x);
    nc_tensor_free(y);
    nc_module_free(bn);
    
    printf("\n");
}

// Test BatchNorm1D
static void test_batchnorm1d(void) {
    printf("=== Testing BatchNorm1D ===\n\n");
    
    // Create 2D tensor: (4, 8) - batch=4, features=8
    size_t shape[] = {4, 8};
    nc_tensor* x = nc_tensor_randn(shape, 2, NC_F32);
    
    // Scale feature 0 to have different stats
    for (size_t b = 0; b < 4; b++) {
        double v = nc_tensor_get2(x, b, 0);
        nc_tensor_set2(x, b, 0, v * 5.0 + 10.0);
    }
    
    printf("Feature 0 before BN: ");
    for (size_t b = 0; b < 4; b++) {
        printf("%.2f ", nc_tensor_get2(x, b, 0));
    }
    printf("\n");
    
    nc_module* bn = nc_batchnorm1d(8);
    nc_tensor* y = nc_module_forward(bn, x);
    
    printf("Feature 0 after BN:  ");
    for (size_t b = 0; b < 4; b++) {
        printf("%.2f ", nc_tensor_get2(y, b, 0));
    }
    printf("\n");
    
    // Verify mean~0, std~1
    double sum = 0, sum_sq = 0;
    for (size_t b = 0; b < 4; b++) {
        double v = nc_tensor_get2(y, b, 0);
        sum += v;
        sum_sq += v * v;
    }
    double mean = sum / 4;
    double std = sqrt(fmax(0, sum_sq / 4 - mean * mean));
    printf("Feature 0 stats: mean=%.4f, std=%.4f\n\n", mean, std);
    
    nc_tensor_free(x);
    nc_tensor_free(y);
    nc_module_free(bn);
}

// Test LayerNorm
static void test_layernorm(void) {
    printf("=== Testing LayerNorm ===\n\n");
    
    // Create tensor: (2, 3, 4) - batch=2, seq=3, features=4
    size_t shape[] = {2, 3, 4};
    nc_tensor* x = nc_tensor_randn(shape, 3, NC_F32);
    
    // LayerNorm over last dimension (features)
    size_t norm_shape[] = {4};
    nc_module* ln = nc_layernorm(norm_shape, 1);
    
    nc_tensor* y = nc_module_forward(ln, x);
    
    printf("Normalized output sample [0,0,:]:\n  ");
    for (size_t i = 0; i < 4; i++) {
        printf("%.3f ", nc_tensor_get3(y, 0, 0, i));
    }
    
    // Verify each position is normalized
    double sum = 0, sum_sq = 0;
    for (size_t i = 0; i < 4; i++) {
        double v = nc_tensor_get3(y, 0, 0, i);
        sum += v;
        sum_sq += v * v;
    }
    double mean = sum / 4;
    double std = sqrt(fmax(0, sum_sq / 4 - mean * mean));
    printf("\n  mean=%.4f, std=%.4f\n\n", mean, std);
    
    nc_tensor_free(x);
    nc_tensor_free(y);
    nc_module_free(ln);
}

// Test CNN with BatchNorm on synthetic data
static void test_cnn_with_bn(void) {
    printf("=== Testing CNN with BatchNorm ===\n\n");
    
    simple_cnn* model = cnn_create();
    
    // Create synthetic batch: (4, 1, 28, 28)
    size_t x_shape[] = {4, 1, 28, 28};
    nc_tensor* x = nc_tensor_randn(x_shape, 4, NC_F32);
    
    printf("Model architecture:\n");
    printf("  Conv2D(1, 8, 3x3, pad=1) -> BatchNorm2D(8) -> ReLU -> MaxPool(2)\n");
    printf("  -> Flatten(8*14*14=1568) -> Linear(1568, 10)\n\n");
    
    // Training mode
    cnn_train_mode(model, true);
    
    printf("Training mode forward pass...\n");
    nc_tensor* logits = cnn_forward(model, x);
    
    if (logits) {
        printf("  Output shape: (%zu, %zu)\n", logits->shape[0], logits->shape[1]);
        
        // Check BN running stats updated
        nc_tensor* rm = nc_batchnorm2d_running_mean(model->bn1);
        nc_tensor* rv = nc_batchnorm2d_running_var(model->bn1);
        printf("  BN running_mean[0]: %.4f\n", nc_tensor_get1(rm, 0));
        printf("  BN running_var[0]: %.4f\n", nc_tensor_get1(rv, 0));
        
        nc_tensor_free(logits);
        
        // Eval mode
        cnn_train_mode(model, false);
        printf("\nEval mode forward pass...\n");
        logits = cnn_forward(model, x);
        
        if (logits) {
            printf("  Output shape: (%zu, %zu)\n", logits->shape[0], logits->shape[1]);
            
            // Predictions
            nc_tensor* preds = nc_argmax(logits, 1, false);
            printf("  Predictions: ");
            for (size_t i = 0; i < 4; i++) {
                printf("%d ", (int)nc_tensor_get1(preds, i));
            }
            printf("\n");
            
            nc_tensor_free(preds);
            nc_tensor_free(logits);
        } else {
            printf("  Eval forward failed!\n");
        }
    } else {
        printf("  Training forward failed!\n");
    }
    
    nc_tensor_free(x);
    cnn_free(model);
    
    printf("\n");
}

int main(void) {
    srand((unsigned)time(NULL));
    nc_init();
    
    printf("=== Nocta BatchNorm Example ===\n\n");
    
    test_batchnorm2d();
    test_batchnorm1d();
    test_layernorm();
    test_cnn_with_bn();
    
    printf("All tests completed!\n\n");
    nc_cleanup();
    
    return 0;
}