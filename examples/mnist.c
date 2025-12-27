// MNIST Example - CNN with BatchNorm for handwritten digit classification
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "nocta/nocta.h"
#include "nocta/nn/conv.h"
#include "nocta/nn/batchnorm.h"
#include "nocta/optim/sgd.h"

#ifdef NOCTA_OPENMP_ENABLED
#include <omp.h>
#endif

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

static void get_batch(mnist_dataset* ds, size_t start, size_t bs, nc_tensor** x, nc_tensor** y, nc_device_type device) {
    if (start + bs > ds->n_samples) bs = ds->n_samples - start;
    size_t xs[] = {bs, 1, 28, 28}, ys[] = {bs};
    *x = nc_tensor_empty(xs, 4, NC_F32);
    *y = nc_tensor_empty(ys, 1, NC_I64);
    for (size_t i = 0; i < bs; i++) {
        for (int j = 0; j < 784; j++)
            nc_tensor_set4(*x, i, 0, j/28, j%28, nc_tensor_get4(ds->images, start+i, 0, j/28, j%28));
        nc_tensor_set1(*y, i, nc_tensor_get1(ds->labels, start+i));
    }
#ifdef NOCTA_CUDA_ENABLED
    if (device == NC_DEVICE_CUDA) {
        nc_tensor_to_device(*x, NC_DEVICE_CUDA);
        nc_tensor_to_device(*y, NC_DEVICE_CUDA);
    }
#else
    (void)device;
#endif
}

// ============================================
// LeNet with BatchNorm
// ============================================

typedef struct {
    nc_module *conv1, *bn1, *conv2, *bn2, *pool, *fc1, *bn_fc, *fc2;
} lenet;

static lenet* lenet_create(void) {
    lenet* m = nc_alloc(sizeof(lenet));
    // Conv layers without bias (BatchNorm has its own bias)
    m->conv1 = nc_conv2d(1, 8, 3, 1, 1, false);    // 1->8, no bias
    m->bn1 = nc_batchnorm2d(8);
    m->conv2 = nc_conv2d(8, 16, 3, 1, 1, false);   // 8->16, no bias
    m->bn2 = nc_batchnorm2d(16);
    m->pool = nc_maxpool2d_simple(2);
    m->fc1 = nc_linear(16*7*7, 64, false);         // no bias before bn
    m->bn_fc = nc_batchnorm1d(64);
    m->fc2 = nc_linear(64, 10, true);
    return m;
}

static void lenet_free(lenet* m) {
    nc_module_free(m->conv1); nc_module_free(m->bn1);
    nc_module_free(m->conv2); nc_module_free(m->bn2);
    nc_module_free(m->pool); 
    nc_module_free(m->fc1); nc_module_free(m->bn_fc);
    nc_module_free(m->fc2); 
    nc_free(m);
}

#ifdef NOCTA_CUDA_ENABLED
static void module_to_device(nc_module* m, nc_device_type device) {
    if (!m) return;
    for (size_t i = 0; i < m->n_params; i++) {
        if (m->params[i]) {
            nc_tensor_to_device(m->params[i], device);
        }
    }
}

static void lenet_to_device(lenet* m, nc_device_type device) {
    module_to_device(m->conv1, device);
    module_to_device(m->bn1, device);
    module_to_device(m->conv2, device);
    module_to_device(m->bn2, device);
    module_to_device(m->fc1, device);
    module_to_device(m->bn_fc, device);
    module_to_device(m->fc2, device);
}
#endif

static void lenet_train_mode(lenet* m, bool training) {
    nc_module_train(m->conv1, training);
    nc_module_train(m->bn1, training);
    nc_module_train(m->conv2, training);
    nc_module_train(m->bn2, training);
    nc_module_train(m->pool, training);
    nc_module_train(m->fc1, training);
    nc_module_train(m->bn_fc, training);
    nc_module_train(m->fc2, training);
}

static void lenet_print(lenet* m) {
    printf("LeNet Architecture:\n");
    printf("-------------------\n");
    nc_module_print(m->conv1);
    nc_module_print(m->bn1);
    nc_module_print(m->pool);
    nc_module_print(m->conv2);
    nc_module_print(m->bn2);
    nc_module_print(m->pool);
    nc_module_print(m->fc1);
    nc_module_print(m->bn_fc);
    nc_module_print(m->fc2);
    printf("-------------------\n");
    size_t total = nc_module_num_parameters(m->conv1) + nc_module_num_parameters(m->bn1) +
                    nc_module_num_parameters(m->conv2) + nc_module_num_parameters(m->bn2) +
                    nc_module_num_parameters(m->fc1) + nc_module_num_parameters(m->bn_fc) +
                    nc_module_num_parameters(m->fc2);
    printf("Total parameters: %zu\n\n", total);
    printf("Total parameters: %zu\n\n", total);
}

// Backward for flatten/reshape
static nc_tensor** lenet_backward_reshape(nc_tensor* grad, nc_tensor** saved, size_t n) {
    nc_tensor* input = saved[0];
    nc_tensor** grads = nc_calloc(1, sizeof(nc_tensor*));
    grads[0] = nc_tensor_reshape(grad, input->shape, input->ndim);
    return grads;
}

static nc_tensor* lenet_flatten(nc_tensor* x) {    // Reshape for fully connected layers
    size_t batch_size = x->shape[0];
    size_t flat_size = x->numel / batch_size;
    size_t new_shape[] = {batch_size, flat_size};
    nc_tensor* out = nc_tensor_reshape(x, new_shape, 2);
    
    if (out && x->requires_grad) {
        // Create autograd node to connect graph
        nc_node* node = nc_node_create("flatten", lenet_backward_reshape);
        nc_node_add_input(node, x);
        nc_node_save_tensor(node, x);
        node->output = out;
        out->grad_fn = node;
        out->is_leaf = false;
        out->requires_grad = true;
    }
    return out;
}

static nc_tensor* lenet_forward(lenet* m, nc_tensor* x, nc_tensor** intermediates, size_t* n_inter) {
    *n_inter = 0;
    x->requires_grad = true;
    fflush(stdout);
    
    // Conv1 -> BN1 -> ReLU -> Pool
    nc_tensor* h = nc_module_forward(m->conv1, x);
    intermediates[(*n_inter)++] = h;
    nc_tensor* bn = nc_module_forward(m->bn1, h);
    intermediates[(*n_inter)++] = bn;
    nc_tensor* a = nc_relu(bn);
    intermediates[(*n_inter)++] = a;
    h = nc_module_forward(m->pool, a);
    intermediates[(*n_inter)++] = h;
    
    // Conv2 -> BN2 -> ReLU -> Pool
    a = nc_module_forward(m->conv2, h);
    intermediates[(*n_inter)++] = a;
    bn = nc_module_forward(m->bn2, a);
    intermediates[(*n_inter)++] = bn;
    h = nc_relu(bn);
    intermediates[(*n_inter)++] = h;
    a = nc_module_forward(m->pool, h);
    intermediates[(*n_inter)++] = a;
    
    // Flatten
    nc_tensor* flat = lenet_flatten(a);
    intermediates[(*n_inter)++] = flat;
    
    // FC1 -> BN_FC -> ReLU -> FC2
    h = nc_module_forward(m->fc1, flat);
    intermediates[(*n_inter)++] = h;
    bn = nc_module_forward(m->bn_fc, h);
    intermediates[(*n_inter)++] = bn;
    a = nc_relu(bn);
    intermediates[(*n_inter)++] = a;
    nc_tensor* logits = nc_module_forward(m->fc2, a);
    
    return logits;
}

static void lenet_zero_grad(lenet* m) {
    nc_module_zero_grad(m->conv1);
    nc_module_zero_grad(m->bn1);
    nc_module_zero_grad(m->conv2);
    nc_module_zero_grad(m->bn2);
    nc_module_zero_grad(m->fc1);
    nc_module_zero_grad(m->bn_fc);
    nc_module_zero_grad(m->fc2);
}

// ============================================
// Save/Load
// ============================================

static void save_model(lenet* m, const char* path) {
    nc_state_dict* sd = nc_state_dict_create();
    nc_state_dict_add(sd, "conv1.weight", nc_tensor_clone(nc_conv2d_weight(m->conv1)));
    nc_state_dict_add(sd, "bn1.weight", nc_tensor_clone(nc_batchnorm2d_weight(m->bn1)));
    nc_state_dict_add(sd, "bn1.bias", nc_tensor_clone(nc_batchnorm2d_bias(m->bn1)));
    nc_state_dict_add(sd, "bn1.running_mean", nc_tensor_clone(nc_batchnorm2d_running_mean(m->bn1)));
    nc_state_dict_add(sd, "bn1.running_var", nc_tensor_clone(nc_batchnorm2d_running_var(m->bn1)));
    nc_state_dict_add(sd, "conv2.weight", nc_tensor_clone(nc_conv2d_weight(m->conv2)));
    nc_state_dict_add(sd, "bn2.weight", nc_tensor_clone(nc_batchnorm2d_weight(m->bn2)));
    nc_state_dict_add(sd, "bn2.bias", nc_tensor_clone(nc_batchnorm2d_bias(m->bn2)));
    nc_state_dict_add(sd, "bn2.running_mean", nc_tensor_clone(nc_batchnorm2d_running_mean(m->bn2)));
    nc_state_dict_add(sd, "bn2.running_var", nc_tensor_clone(nc_batchnorm2d_running_var(m->bn2)));
    nc_state_dict_add(sd, "fc1.weight", nc_tensor_clone(nc_linear_weight(m->fc1)));
    nc_state_dict_add(sd, "bn_fc.weight", nc_tensor_clone(nc_module_get_param(m->bn_fc, "weight")));
    nc_state_dict_add(sd, "bn_fc.bias", nc_tensor_clone(nc_module_get_param(m->bn_fc, "bias")));
    nc_state_dict_add(sd, "bn_fc.running_mean", nc_tensor_clone(nc_module_get_param(m->bn_fc, "running_mean")));
    nc_state_dict_add(sd, "bn_fc.running_var", nc_tensor_clone(nc_module_get_param(m->bn_fc, "running_var")));
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
    if ((t = nc_state_dict_get(sd, "bn1.weight"))) nc_tensor_copy_(nc_batchnorm2d_weight(m->bn1), t);
    if ((t = nc_state_dict_get(sd, "bn1.bias"))) nc_tensor_copy_(nc_batchnorm2d_bias(m->bn1), t);
    if ((t = nc_state_dict_get(sd, "bn1.running_mean"))) nc_tensor_copy_(nc_batchnorm2d_running_mean(m->bn1), t);
    if ((t = nc_state_dict_get(sd, "bn1.running_var"))) nc_tensor_copy_(nc_batchnorm2d_running_var(m->bn1), t);
    if ((t = nc_state_dict_get(sd, "conv2.weight"))) nc_tensor_copy_(nc_conv2d_weight(m->conv2), t);
    if ((t = nc_state_dict_get(sd, "bn2.weight"))) nc_tensor_copy_(nc_batchnorm2d_weight(m->bn2), t);
    if ((t = nc_state_dict_get(sd, "bn2.bias"))) nc_tensor_copy_(nc_batchnorm2d_bias(m->bn2), t);
    if ((t = nc_state_dict_get(sd, "bn2.running_mean"))) nc_tensor_copy_(nc_batchnorm2d_running_mean(m->bn2), t);
    if ((t = nc_state_dict_get(sd, "bn2.running_var"))) nc_tensor_copy_(nc_batchnorm2d_running_var(m->bn2), t);
    if ((t = nc_state_dict_get(sd, "fc1.weight"))) nc_tensor_copy_(nc_linear_weight(m->fc1), t);
    if ((t = nc_state_dict_get(sd, "bn_fc.weight"))) nc_tensor_copy_(nc_module_get_param(m->bn_fc, "weight"), t);
    if ((t = nc_state_dict_get(sd, "bn_fc.bias"))) nc_tensor_copy_(nc_module_get_param(m->bn_fc, "bias"), t);
    if ((t = nc_state_dict_get(sd, "bn_fc.running_mean"))) nc_tensor_copy_(nc_module_get_param(m->bn_fc, "running_mean"), t);
    if ((t = nc_state_dict_get(sd, "bn_fc.running_var"))) nc_tensor_copy_(nc_module_get_param(m->bn_fc, "running_var"), t);
    if ((t = nc_state_dict_get(sd, "fc2.weight"))) nc_tensor_copy_(nc_linear_weight(m->fc2), t);
    if ((t = nc_state_dict_get(sd, "fc2.bias"))) nc_tensor_copy_(nc_linear_bias(m->fc2), t);
    nc_state_dict_free(sd);
    printf("Model loaded from %s\n", path);
}

// ============================================
// Accuracy (inference mode)
// ============================================

static nc_tensor* lenet_forward_infer(lenet* m, nc_tensor* x) {
    // Conv1 -> BN1 -> ReLU -> Pool
    nc_tensor* h = nc_module_forward(m->conv1, x);
    nc_tensor* bn = nc_module_forward(m->bn1, h); nc_tensor_free(h);
    nc_tensor* a = nc_relu(bn); nc_tensor_free(bn);
    h = nc_module_forward(m->pool, a); nc_tensor_free(a);
    
    // Conv2 -> BN2 -> ReLU -> Pool
    a = nc_module_forward(m->conv2, h); nc_tensor_free(h);
    bn = nc_module_forward(m->bn2, a); nc_tensor_free(a);
    h = nc_relu(bn); nc_tensor_free(bn);
    a = nc_module_forward(m->pool, h); nc_tensor_free(h);
    
    // Flatten
    nc_tensor* flat = lenet_flatten(a);
    nc_tensor_free(a);
    
    // FC1 -> BN_FC -> ReLU -> FC2
    h = nc_module_forward(m->fc1, flat); nc_tensor_free(flat);
    bn = nc_module_forward(m->bn_fc, h); nc_tensor_free(h);
    a = nc_relu(bn); nc_tensor_free(bn);
    nc_tensor* logits = nc_module_forward(m->fc2, a);
    nc_tensor_free(a);
    
    return logits;
}

static double compute_accuracy(lenet* m, mnist_dataset* ds, size_t max_n, nc_device_type device) {
    lenet_train_mode(m, false);  // Eval mode for BatchNorm
    
    size_t correct = 0, total = (max_n < ds->n_samples) ? max_n : ds->n_samples;
    for (size_t i = 0; i < total; i += 32) {
        size_t bs = (i + 32 <= total) ? 32 : (total - i);
        nc_tensor *x, *y;
        get_batch(ds, i, bs, &x, &y, device);
        nc_tensor* logits = lenet_forward_infer(m, x);
        
        // nc_argmax doesn't have CUDA support - must copy logits to CPU first
        nc_tensor_to_device(logits, NC_DEVICE_CPU);
        nc_tensor* preds = nc_argmax(logits, 1, false);
        
        // Also copy labels to CPU for reading values
        nc_tensor_to_device(y, NC_DEVICE_CPU);
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
    
    printf("=== Nocta MNIST CNN+BatchNorm Example ===\n\n");

    #ifdef NOCTA_OPENMP_ENABLED
    printf("OpenMP Enabled. Max threads: %d\n\n", omp_get_max_threads());
    #else
    printf("OpenMP Disabled.\n\n");
    #endif
    
    // Load data
    mnist_dataset *train = NULL, *test = NULL;
    if (argc >= 5) {
        train = load_mnist(argv[1], argv[2]);
        test = load_mnist(argv[3], argv[4]);
    }
    if (!train) {
        if (test) free_mnist(test);
        printf("Using synthetic data. For real MNIST:\n");
        printf("  %s train-images train-labels test-images test-labels\n\n", argv[0]);
        train = create_synthetic(100);
        test = create_synthetic(50);
    }
    printf("Train: %zu, Test: %zu\n\n", train->n_samples, test->n_samples);
    
    // Create and print model
    lenet* model = lenet_create();
    lenet_print(model);
    
    // Detect GPU
    nc_device_type device = NC_DEVICE_CPU;
#ifdef NOCTA_CUDA_ENABLED
    if (nc_cuda_available()) {
        device = NC_DEVICE_CUDA;
        printf("GPU detected! Moving model to CUDA...\n");
        lenet_to_device(model, NC_DEVICE_CUDA);
        nc_device_info info = nc_cuda_device_info(0);
        printf("Using: %s\n\n", info.name);
    } else {
        printf("No GPU available. Using CPU.\n\n");
    }
#else
    printf("CUDA not compiled. Using CPU.\n\n");
#endif
    
    // Training parameters
    double lr = 0.01;
    size_t epochs = 5, batch_size = 256;

    // Optimizer Setup
    nc_tensor* params[32];
    size_t n_params = 0;
    
    // Collect params
    params[n_params++] = nc_conv2d_weight(model->conv1);
    params[n_params++] = nc_conv2d_bias(model->conv1);
    params[n_params++] = nc_batchnorm2d_weight(model->bn1);
    params[n_params++] = nc_batchnorm2d_bias(model->bn1);
    params[n_params++] = nc_conv2d_weight(model->conv2);
    params[n_params++] = nc_conv2d_bias(model->conv2);
    params[n_params++] = nc_batchnorm2d_weight(model->bn2);
    params[n_params++] = nc_batchnorm2d_bias(model->bn2);
    params[n_params++] = nc_linear_weight(model->fc1); // fc1 has no bias
    params[n_params++] = nc_module_get_param(model->bn_fc, "weight");
    params[n_params++] = nc_module_get_param(model->bn_fc, "bias");
    params[n_params++] = nc_linear_weight(model->fc2);
    params[n_params++] = nc_linear_bias(model->fc2);

    // Create SGD optimizer with momentum
    nc_sgd_config sgd_conf = NC_SGD_DEFAULT;
    sgd_conf.momentum = 0.9;
    sgd_conf.weight_decay = 0.0;
    nc_optimizer* optimizer = nc_sgd(lr, sgd_conf);
    
    // Add parameters to optimizer
    for (size_t i = 0; i < n_params; i++) {
        if (params[i]) nc_optimizer_add_param(optimizer, params[i]);
    }
    
    printf("Training: lr=%.3f, momentum=%.1f, batch=%zu, epochs=%zu\n\n", 
           lr, sgd_conf.momentum, batch_size, epochs);
    
    for (size_t e = 0; e < epochs; e++) {
        lenet_train_mode(model, true);
        double loss_sum = 0;
        size_t n_batches = 0;
        
        nc_tensor* total_loss_t = nc_tensor_scalar(0.0f, NC_F32);
        #ifdef NOCTA_CUDA_ENABLED
        if (device == NC_DEVICE_CUDA) nc_tensor_to_device(total_loss_t, NC_DEVICE_CUDA);
        #endif
        
        for (size_t i = 0; i < train->n_samples; i += batch_size) {
            nc_tensor *x, *y;
            get_batch(train, i, batch_size, &x, &y, device);
            
            printf("\r  Batch %zu/%zu", i/batch_size + 1, (train->n_samples + batch_size - 1)/batch_size);
            fflush(stdout);
            
            lenet_zero_grad(model);
            
            nc_tensor* intermediates[20];
            size_t n_inter = 0;
            nc_tensor* logits = lenet_forward(model, x, intermediates, &n_inter);
            
            nc_tensor* loss = nc_cross_entropy_loss(logits, y);
            nc_add_(total_loss_t, loss);
            n_batches++;
            
            nc_backward_scalar(loss);
            
            // Use optimizer to update weights
            nc_optimizer_step(optimizer);
            
            nc_tensor_free(x); nc_tensor_free(y);
            nc_tensor_free(logits); nc_tensor_free(loss);
            for (size_t j = 0; j < n_inter; j++) {
                nc_tensor_free(intermediates[j]);
            }
        }
        printf("\r                              \r"); // Clear the batch progress line
        
        nc_tensor_to_device(total_loss_t, NC_DEVICE_CPU);
        loss_sum = nc_tensor_get_flat(total_loss_t, 0);
        nc_tensor_free(total_loss_t);
        
        double avg_loss = loss_sum / n_batches;
        double train_acc = compute_accuracy(model, train, 500, device);
        double test_acc = compute_accuracy(model, test, 200, device);
        printf("Epoch %zu: Loss=%.4f, Train=%.1f%%, Test=%.1f%%\n", 
               e+1, avg_loss, train_acc, test_acc);
    }
    
    // Save model
    printf("\n");
    save_model(model, "mnist_cnn_bn.ncta");
    nc_file_info("mnist_cnn_bn.ncta");
    
    // Test load
    printf("\nTesting load...\n");
    lenet* loaded = lenet_create();
    load_model(loaded, "mnist_cnn_bn.ncta");
    printf("Loaded model accuracy: %.1f%%\n", compute_accuracy(loaded, test, 200, NC_DEVICE_CPU));
    
    // Cleanup
    nc_optimizer_free(optimizer);
    lenet_free(model);
    lenet_free(loaded);
    free_mnist(train);
    free_mnist(test);
    
    printf("\n");
    nc_cleanup();
    return 0;
}