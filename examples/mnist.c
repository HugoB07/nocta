// MNIST Example - CNN for handwritten digit classification
// Demonstrates Conv2D, MaxPool2D, Linear layers with Nocta
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "nocta/nocta.h"
#include "nocta/nn/conv.h"

// ============================================
// MNIST Data Loading
// ============================================

#define MNIST_IMAGE_MAGIC 0x00000803
#define MNIST_LABEL_MAGIC 0x00000801

typedef struct {
    nc_tensor* images;
    nc_tensor* labels;
    size_t n_samples;
} mnist_dataset;

static uint32_t read_be32(FILE* fp) {
    unsigned char buf[4];
    if (fread(buf, 1, 4, fp) != 4) return 0;
    return (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
}

static mnist_dataset* load_mnist(const char* img_path, const char* lbl_path) {
    FILE* img_fp = fopen(img_path, "rb");
    FILE* lbl_fp = fopen(lbl_path, "rb");
    if (!img_fp || !lbl_fp) { 
        if (img_fp) fclose(img_fp); 
        if (lbl_fp) fclose(lbl_fp); 
        return NULL; 
    }
    
    if (read_be32(img_fp) != MNIST_IMAGE_MAGIC) { fclose(img_fp); fclose(lbl_fp); return NULL; }
    uint32_t n = read_be32(img_fp);
    read_be32(img_fp); read_be32(img_fp); // rows, cols
    
    if (read_be32(lbl_fp) != MNIST_LABEL_MAGIC) { fclose(img_fp); fclose(lbl_fp); return NULL; }
    read_be32(lbl_fp);
    
    mnist_dataset* ds = nc_alloc(sizeof(mnist_dataset));
    ds->n_samples = n;
    size_t img_shape[] = {n, 1, 28, 28}, lbl_shape[] = {n};
    ds->images = nc_tensor_empty(img_shape, 4, NC_F32);
    ds->labels = nc_tensor_empty(lbl_shape, 1, NC_I64);
    
    unsigned char* buf = nc_alloc(784);
    for (uint32_t i = 0; i < n; i++) {
        fread(buf, 1, 784, img_fp);
        for (int j = 0; j < 784; j++)
            nc_tensor_set4(ds->images, i, 0, j/28, j%28, buf[j]/255.0);
    }
    nc_free(buf);
    
    buf = nc_alloc(n);
    fread(buf, 1, n, lbl_fp);
    for (uint32_t i = 0; i < n; i++)
        nc_tensor_set1(ds->labels, i, (double)buf[i]);
    nc_free(buf);
    
    fclose(img_fp); fclose(lbl_fp);
    return ds;
}

static mnist_dataset* create_synthetic(size_t n) {
    mnist_dataset* ds = nc_alloc(sizeof(mnist_dataset));
    ds->n_samples = n;
    size_t img_shape[] = {n, 1, 28, 28}, lbl_shape[] = {n};
    ds->images = nc_tensor_rand(img_shape, 4, NC_F32);
    ds->labels = nc_tensor_empty(lbl_shape, 1, NC_I64);
    for (size_t i = 0; i < n; i++)
        nc_tensor_set1(ds->labels, i, (double)(rand() % 10));
    return ds;
}

static void free_mnist(mnist_dataset* ds) {
    if (!ds) return;
    nc_tensor_free(ds->images);
    nc_tensor_free(ds->labels);
    nc_free(ds);
}

static void get_batch(mnist_dataset* ds, size_t start, size_t bs, nc_tensor** x, nc_tensor** y) {
    if (start + bs > ds->n_samples) bs = ds->n_samples - start;
    size_t xs[] = {bs, 1, 28, 28}, ys[] = {bs};
    *x = nc_tensor_empty(xs, 4, NC_F32);
    *y = nc_tensor_empty(ys, 1, NC_I64);
    for (size_t i = 0; i < bs; i++) {
        for (int j = 0; j < 784; j++)
            nc_tensor_set4(*x, i, 0, j/28, j%28, nc_tensor_get4(ds->images, start+i, 0, j/28, j%28));
        nc_tensor_set1(*y, i, nc_tensor_get1(ds->labels, start+i));
    }
}

// ============================================
// LeNet Model
// ============================================

typedef struct {
    nc_module *conv1, *conv2, *pool, *fc1, *fc2;
} lenet;

static lenet* lenet_create(void) {
    lenet* m = nc_alloc(sizeof(lenet));
    // Smaller network for faster testing
    m->conv1 = nc_conv2d(1, 8, 3, 1, 1, true);    // 1->8 (was 16)
    m->conv2 = nc_conv2d(8, 16, 3, 1, 1, true);   // 8->16 (was 32)
    m->pool = nc_maxpool2d_simple(2);
    m->fc1 = nc_linear(16*7*7, 64, true);         // 784->64 (was 1568->128)
    m->fc2 = nc_linear(64, 10, true);             // 64->10 (was 128->10)
    return m;
}

static void lenet_free(lenet* m) {
    nc_module_free(m->conv1); nc_module_free(m->conv2);
    nc_module_free(m->pool); nc_module_free(m->fc1);
    nc_module_free(m->fc2); nc_free(m);
}

static void lenet_print(lenet* m) {
    printf("LeNet Architecture:\n");
    printf("-------------------\n");
    nc_module_print(m->conv1);
    nc_module_print(m->pool);
    nc_module_print(m->conv2);
    nc_module_print(m->pool);
    nc_module_print(m->fc1);
    nc_module_print(m->fc2);
    printf("-------------------\n");
    size_t total = nc_module_num_parameters(m->conv1) + nc_module_num_parameters(m->conv2) +
                   nc_module_num_parameters(m->fc1) + nc_module_num_parameters(m->fc2);
    printf("Total parameters: %zu\n\n", total);
}

static nc_tensor* lenet_forward(lenet* m, nc_tensor* x, nc_tensor** intermediates, size_t* n_inter) {
    *n_inter = 0;
    
    // Enable gradients on input to propagate through network
    x->requires_grad = true;
    
    // Conv1 -> ReLU -> Pool
    nc_tensor* h = nc_module_forward(m->conv1, x);
    intermediates[(*n_inter)++] = h;
    nc_tensor* a = nc_relu(h);
    intermediates[(*n_inter)++] = a;
    h = nc_module_forward(m->pool, a);
    intermediates[(*n_inter)++] = h;
    
    // Conv2 -> ReLU -> Pool
    a = nc_module_forward(m->conv2, h);
    intermediates[(*n_inter)++] = a;
    h = nc_relu(a);
    intermediates[(*n_inter)++] = h;
    a = nc_module_forward(m->pool, h);
    intermediates[(*n_inter)++] = a;
    
    // Flatten - needs requires_grad!
    size_t batch = a->shape[0];
    size_t channels = a->shape[1];
    size_t flat_size = channels * 7 * 7;
    size_t flat_shape[] = {batch, flat_size};
    nc_tensor* flat = nc_tensor_empty(flat_shape, 2, a->dtype);
    flat->requires_grad = true;
    for (size_t b = 0; b < batch; b++) {
        size_t idx = 0;
        for (size_t c = 0; c < channels; c++)
            for (size_t i = 0; i < 7; i++)
                for (size_t j = 0; j < 7; j++)
                    nc_tensor_set2(flat, b, idx++, nc_tensor_get4(a, b, c, i, j));
    }
    intermediates[(*n_inter)++] = flat;
    
    // FC1 -> ReLU -> FC2
    h = nc_module_forward(m->fc1, flat);
    intermediates[(*n_inter)++] = h;
    a = nc_relu(h);
    intermediates[(*n_inter)++] = a;
    nc_tensor* logits = nc_module_forward(m->fc2, a);
    
    return logits;
}

static void lenet_zero_grad(lenet* m) {
    nc_module_zero_grad(m->conv1);
    nc_module_zero_grad(m->conv2);
    nc_module_zero_grad(m->fc1);
    nc_module_zero_grad(m->fc2);
}

static void lenet_sgd_step(lenet* m, double lr) {
    nc_tensor* params[] = {
        nc_conv2d_weight(m->conv1), nc_conv2d_bias(m->conv1),
        nc_conv2d_weight(m->conv2), nc_conv2d_bias(m->conv2),
        nc_linear_weight(m->fc1), nc_linear_bias(m->fc1),
        nc_linear_weight(m->fc2), nc_linear_bias(m->fc2)
    };
    for (int i = 0; i < 8; i++) {
        if (!params[i] || !params[i]->grad) continue;
        for (size_t j = 0; j < params[i]->numel; j++) {
            double p = nc_tensor_get_flat(params[i], j);
            double g = nc_tensor_get_flat(params[i]->grad, j);
            nc_tensor_set_flat(params[i], j, p - lr * g);
        }
    }
}

// ============================================
// Save/Load
// ============================================

static void save_model(lenet* m, const char* path) {
    nc_state_dict* sd = nc_state_dict_create();
    nc_state_dict_add(sd, "conv1.weight", nc_tensor_clone(nc_conv2d_weight(m->conv1)));
    nc_state_dict_add(sd, "conv1.bias", nc_tensor_clone(nc_conv2d_bias(m->conv1)));
    nc_state_dict_add(sd, "conv2.weight", nc_tensor_clone(nc_conv2d_weight(m->conv2)));
    nc_state_dict_add(sd, "conv2.bias", nc_tensor_clone(nc_conv2d_bias(m->conv2)));
    nc_state_dict_add(sd, "fc1.weight", nc_tensor_clone(nc_linear_weight(m->fc1)));
    nc_state_dict_add(sd, "fc1.bias", nc_tensor_clone(nc_linear_bias(m->fc1)));
    nc_state_dict_add(sd, "fc2.weight", nc_tensor_clone(nc_linear_weight(m->fc2)));
    nc_state_dict_add(sd, "fc2.bias", nc_tensor_clone(nc_linear_bias(m->fc2)));
    nc_state_dict_save(sd, path);
    nc_state_dict_free(sd);
    printf("Model saved to %s\n", path);
}

static void load_model(lenet* m, const char* path) {
    nc_state_dict* sd = nc_state_dict_load(path);
    if (!sd) { printf("Failed to load %s\n", path); return; }
    nc_tensor* t;
    if ((t = nc_state_dict_get(sd, "conv1.weight"))) nc_tensor_copy_(nc_conv2d_weight(m->conv1), t);
    if ((t = nc_state_dict_get(sd, "conv1.bias"))) nc_tensor_copy_(nc_conv2d_bias(m->conv1), t);
    if ((t = nc_state_dict_get(sd, "conv2.weight"))) nc_tensor_copy_(nc_conv2d_weight(m->conv2), t);
    if ((t = nc_state_dict_get(sd, "conv2.bias"))) nc_tensor_copy_(nc_conv2d_bias(m->conv2), t);
    if ((t = nc_state_dict_get(sd, "fc1.weight"))) nc_tensor_copy_(nc_linear_weight(m->fc1), t);
    if ((t = nc_state_dict_get(sd, "fc1.bias"))) nc_tensor_copy_(nc_linear_bias(m->fc1), t);
    if ((t = nc_state_dict_get(sd, "fc2.weight"))) nc_tensor_copy_(nc_linear_weight(m->fc2), t);
    if ((t = nc_state_dict_get(sd, "fc2.bias"))) nc_tensor_copy_(nc_linear_bias(m->fc2), t);
    nc_state_dict_free(sd);
    printf("Model loaded from %s\n", path);
}

// ============================================
// Accuracy
// ============================================

// Forward without keeping intermediates (for inference)
static nc_tensor* lenet_forward_infer(lenet* m, nc_tensor* x) {
    nc_tensor* h = nc_module_forward(m->conv1, x);
    nc_tensor* a = nc_relu(h); nc_tensor_free(h);
    h = nc_module_forward(m->pool, a); nc_tensor_free(a);
    
    a = nc_module_forward(m->conv2, h); nc_tensor_free(h);
    h = nc_relu(a); nc_tensor_free(a);
    a = nc_module_forward(m->pool, h); nc_tensor_free(h);
    
    size_t batch = a->shape[0];
    size_t channels = a->shape[1];
    size_t flat_size = channels * 7 * 7;
    size_t flat_shape[] = {batch, flat_size};
    nc_tensor* flat = nc_tensor_empty(flat_shape, 2, a->dtype);
    for (size_t b = 0; b < batch; b++) {
        size_t idx = 0;
        for (size_t c = 0; c < channels; c++)
            for (size_t i = 0; i < 7; i++)
                for (size_t j = 0; j < 7; j++)
                    nc_tensor_set2(flat, b, idx++, nc_tensor_get4(a, b, c, i, j));
    }
    nc_tensor_free(a);
    
    h = nc_module_forward(m->fc1, flat); nc_tensor_free(flat);
    a = nc_relu(h); nc_tensor_free(h);
    nc_tensor* logits = nc_module_forward(m->fc2, a);
    nc_tensor_free(a);
    
    return logits;
}

static double compute_accuracy(lenet* m, mnist_dataset* ds, size_t max_n) {
    size_t correct = 0, total = (max_n < ds->n_samples) ? max_n : ds->n_samples;
    for (size_t i = 0; i < total; i += 32) {
        size_t bs = (i + 32 <= total) ? 32 : (total - i);
        nc_tensor *x, *y;
        get_batch(ds, i, bs, &x, &y);
        nc_tensor* logits = lenet_forward_infer(m, x);  // Use inference forward
        nc_tensor* preds = nc_argmax(logits, 1, false);
        for (size_t j = 0; j < bs; j++)
            if ((int)nc_tensor_get1(preds, j) == (int)nc_tensor_get1(y, j)) correct++;
        nc_tensor_free(x); nc_tensor_free(y);
        nc_tensor_free(logits); nc_tensor_free(preds);
    }
    return 100.0 * correct / total;
}

// ============================================
// Main
// ============================================

int main(int argc, char** argv) {
    srand((unsigned)time(NULL));
    nc_init();
    
    printf("=== Nocta MNIST CNN Example ===\n\n");
    
    // Load data
    mnist_dataset *train = NULL, *test = NULL;
    if (argc >= 5) {
        train = load_mnist(argv[1], argv[2]);
        test = load_mnist(argv[3], argv[4]);
    }
    if (!train) {
        printf("Using synthetic data. For real MNIST:\n");
        printf("  %s train-images train-labels test-images test-labels\n\n", argv[0]);
        train = create_synthetic(100);  // Smaller for testing
        test = create_synthetic(50);
    }
    printf("Train: %zu, Test: %zu\n\n", train->n_samples, test->n_samples);
    
    // Create and print model
    lenet* model = lenet_create();
    lenet_print(model);
    
    // Training
    double lr = 0.01;
    size_t epochs = 2, batch_size = 16;  // Smaller batch
    printf("Training: lr=%.3f, batch=%zu, epochs=%zu\n\n", lr, batch_size, epochs);
    
    for (size_t e = 0; e < epochs; e++) {
        double loss_sum = 0;
        size_t n_batches = 0;
        
        for (size_t i = 0; i < train->n_samples; i += batch_size) {
            nc_tensor *x, *y;
            get_batch(train, i, batch_size, &x, &y);
            
            printf("\r  Batch %zu/%zu [fwd...", i/batch_size + 1, (train->n_samples + batch_size - 1)/batch_size);
            fflush(stdout);
            
            // Zero gradients
            lenet_zero_grad(model);
            
            // Forward - keep intermediates for backward
            nc_tensor* intermediates[20];
            size_t n_inter = 0;
            nc_tensor* logits = lenet_forward(model, x, intermediates, &n_inter);
            
            printf("loss...");
            fflush(stdout);
            
            // Loss (with autograd!)
            nc_tensor* loss = nc_cross_entropy_loss(logits, y);
            loss_sum += nc_tensor_get_flat(loss, 0);
            n_batches++;
            
            printf("bwd(loss=%p, logits=%p)...", (void*)loss, (void*)logits);
            fflush(stdout);
            
            // Backward - autograd does all the work!
            nc_backward_scalar(loss);
            
            printf("sgd]");
            fflush(stdout);
            
            // SGD step
            lenet_sgd_step(model, lr);
            
            // Now free everything
            nc_tensor_free(x); nc_tensor_free(y);
            nc_tensor_free(logits); nc_tensor_free(loss);
            for (size_t j = 0; j < n_inter; j++) {
                nc_tensor_free(intermediates[j]);
            }
        }
        printf("\r                              \r");
        
        double avg_loss = loss_sum / n_batches;
        double train_acc = compute_accuracy(model, train, 500);
        double test_acc = compute_accuracy(model, test, 200);
        printf("Epoch %zu: Loss=%.4f, Train=%.1f%%, Test=%.1f%%\n", 
               e+1, avg_loss, train_acc, test_acc);
    }
    
    // Save model
    printf("\n");
    save_model(model, "mnist_cnn.ncta");
    nc_file_info("mnist_cnn.ncta");
    
    // Test load
    printf("\nTesting load...\n");
    lenet* loaded = lenet_create();
    load_model(loaded, "mnist_cnn.ncta");
    printf("Loaded model accuracy: %.1f%%\n", compute_accuracy(loaded, test, 200));
    
    // Cleanup
    lenet_free(model);
    lenet_free(loaded);
    free_mnist(train);
    free_mnist(test);
    
    printf("\n");
    nc_cleanup();
    return 0;
}