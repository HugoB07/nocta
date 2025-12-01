# Nocta

A lightweight deep learning library written in pure C.

Nocta provides PyTorch-like functionality with automatic differentiation, tensor operations, and neural network modules — all in a single, dependency-free C library.

## Features

- **Tensors**: N-dimensional arrays with broadcasting, views, and type support (float32, float64, int32, int64)
- **Automatic Differentiation**: Dynamic computation graph with backpropagation
- **Neural Network Modules**: Linear layers, activations (ReLU, Sigmoid, Tanh, Softmax, GELU, etc.)
- **Optimizers**: SGD (with momentum, Nesterov), Adam, AdamW
- **Serialization**: Save/load tensors, models, and training checkpoints in custom .ncta format
- **Memory Efficient**: Reference-counted storage, aligned allocations, detailed memory tracking
- **Cross-Platform**: Works on Windows (MSVC), Linux (GCC), and macOS (Clang)
- **Zero Dependencies**: No external libraries required

## Quick Start

```c
#include "nocta/nocta.h"

int main() {
    // Create tensors
    size_t shape[] = {2, 3};
    nc_tensor* a = nc_tensor_randn(shape, 2, NC_F32);
    nc_tensor* b = nc_tensor_randn(shape, 2, NC_F32);
    
    // Enable gradients
    nc_tensor_requires_grad_(a, true);
    
    // Operations
    nc_tensor* c = nc_add(a, b);
    nc_tensor* d = nc_relu(c);
    nc_tensor* loss = nc_mean_all(d);
    
    // Backward pass
    nc_backward_scalar(loss);
    
    // Access gradients
    printf("Gradient: %f\n", nc_tensor_get_flat(a->grad, 0));
    
    // Cleanup
    nc_tensor_free(a);
    nc_tensor_free(b);
    nc_tensor_free(c);
    nc_tensor_free(d);
    nc_tensor_free(loss);
    
    return 0;
}
```

## Building

### Using CMake

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `NOCTA_BUILD_EXAMPLES` | ON | Build example programs |
| `NOCTA_ENABLE_SIMD` | OFF | Enable AVX2 SIMD optimizations |

## Example: XOR Neural Network

```c
#include "nocta/nocta.h"

int main() {
    // XOR dataset
    float X_data[] = {0,0, 0,1, 1,0, 1,1};
    float Y_data[] = {0, 1, 1, 0};
    
    size_t x_shape[] = {4, 2};
    size_t y_shape[] = {4, 1};
    
    nc_tensor* X = nc_tensor_from_data(X_data, x_shape, 2, NC_F32);
    nc_tensor* Y = nc_tensor_from_data(Y_data, y_shape, 2, NC_F32);
    
    // Create weights: 2 -> 8 -> 1
    size_t w1_shape[] = {2, 8};
    size_t w2_shape[] = {8, 1};
    
    nc_tensor* W1 = nc_tensor_randn(w1_shape, 2, NC_F32);
    nc_tensor* W2 = nc_tensor_randn(w2_shape, 2, NC_F32);
    
    nc_tensor_requires_grad_(W1, true);
    nc_tensor_requires_grad_(W2, true);
    
    double lr = 1.0;
    
    for (int epoch = 0; epoch < 5000; epoch++) {
        nc_tensor_zero_grad_(W1);
        nc_tensor_zero_grad_(W2);
        
        // Forward: relu(X @ W1) @ W2 -> sigmoid
        nc_tensor* h = nc_relu(nc_matmul(X, W1));
        nc_tensor* pred = nc_sigmoid(nc_matmul(h, W2));
        
        // MSE Loss
        nc_tensor* diff = nc_sub(pred, Y);
        nc_tensor* loss = nc_mean_all(nc_mul(diff, diff));
        
        // Backward
        nc_backward_scalar(loss);
        
        // Update weights
        for (size_t i = 0; i < W1->numel; i++) {
            nc_tensor_set_flat(W1, i, 
                nc_tensor_get_flat(W1, i) - lr * nc_tensor_get_flat(W1->grad, i));
        }
        // ... same for W2
    }
    
    return 0;
}
```

## API Overview

### Tensor Creation

```c
nc_tensor* nc_tensor_empty(shape, ndim, dtype);    // Uninitialized
nc_tensor* nc_tensor_zeros(shape, ndim, dtype);    // Filled with 0
nc_tensor* nc_tensor_ones(shape, ndim, dtype);     // Filled with 1
nc_tensor* nc_tensor_randn(shape, ndim, dtype);    // Normal distribution
nc_tensor* nc_tensor_from_data(data, shape, ndim, dtype);  // From array
```

### Operations

```c
// Arithmetic
nc_tensor* nc_add(a, b);
nc_tensor* nc_sub(a, b);
nc_tensor* nc_mul(a, b);
nc_tensor* nc_div(a, b);

// Matrix
nc_tensor* nc_matmul(a, b);
nc_tensor* nc_tensor_t(a);  // Transpose

// Activations
nc_tensor* nc_relu(x);
nc_tensor* nc_sigmoid(x);
nc_tensor* nc_tanh_act(x);
nc_tensor* nc_softmax(x, dim);

// Reductions
nc_tensor* nc_sum_all(x);
nc_tensor* nc_mean_all(x);
nc_tensor* nc_max_all(x);
```

### Autograd

```c
nc_tensor_requires_grad_(t, true);  // Enable gradient tracking
nc_backward_scalar(loss);            // Backpropagation
nc_tensor_zero_grad_(t);             // Reset gradients
```

### Neural Network Modules

```c
nc_module* layer = nc_linear(in_features, out_features, bias);
nc_tensor* output = nc_module_forward(layer, input);
nc_module_free(layer);
```

### Optimizers

```c
nc_optimizer* opt = nc_sgd_momentum(module, lr, momentum);
nc_optimizer* opt = nc_adam_default(module, lr);

nc_optimizer_zero_grad(opt);
nc_optimizer_step(opt);
nc_optimizer_free(opt);
```

### Serialization

```c
// Save/load single tensor
nc_tensor_save(tensor, "weights.ncta");
nc_tensor* loaded = nc_tensor_load("weights.ncta");

// Save/load model
nc_module_save(model, "model.ncta");
nc_module_load(model, "model.ncta");

// State dict (PyTorch-style)
nc_state_dict* sd = nc_module_state_dict(model);
nc_state_dict_save(sd, "checkpoint.ncta");
nc_state_dict* loaded_sd = nc_state_dict_load("checkpoint.ncta");

// Training checkpoints
nc_checkpoint ckpt = {
    .model_state = nc_module_state_dict(model),
    .optimizer_state = optimizer_state_dict,
    .epoch = 100,
    .loss = 0.001
};
nc_checkpoint_save(&ckpt, "training.ncta");
nc_checkpoint* loaded = nc_checkpoint_load("training.ncta");

// File utilities
nc_file_info("model.ncta");           // Print file contents
bool valid = nc_file_verify("model.ncta");  // Verify integrity
```

## Project Structure

```
nocta/
├── include/nocta/
│   ├── nocta.h          # Main header
│   ├── core/            # Tensor, memory, types, serialization
│   ├── autograd/        # Automatic differentiation
│   ├── ops/             # Operations (arithmetic, matmul, activations)
│   ├── nn/              # Neural network modules
│   └── optim/           # Optimizers
├── src/                 # Implementation files
├── examples/            # Example programs
└── CMakeLists.txt
```

## Performance

Memory efficiency on XOR example (5000 epochs):
```
Total allocated: 97.9 MB
Peak usage:      15.6 KB
Memory leaks:    0 bytes
```

## Roadmap

- [x] Model serialization (save/load tensors, modules, checkpoints)
- [ ] Convolution layers (Conv2D, MaxPool)
- [ ] Batch normalization
- [ ] Dropout
- [ ] SIMD optimizations (AVX2/AVX512)
- [ ] GPU support (OpenCL/CUDA)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

Inspired by PyTorch, Tinygrad, and micrograd.