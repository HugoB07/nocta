# Nocta

A lightweight deep learning library written in pure C.

Nocta provides PyTorch-like functionality with automatic differentiation, tensor operations, and neural network modules — all in a single, dependency-free C library.

## Features

- **Tensors**: N-dimensional arrays with broadcasting, views, and type support (float32, float64, int32, int64)
- **Automatic Differentiation**: Dynamic computation graph with backpropagation
- **Neural Network Modules**: Linear layers, Convolutional layers (Conv2d, MaxPool2d), Normalization (BatchNorm1d, BatchNorm2d, LayerNorm), Dropout, activations (ReLU, Sigmoid, Tanh, Softmax, GELU, etc.)
- **Loss Functions**: CrossEntropy, MSE, BCE
- **Optimizers**: SGD (with momentum, Nesterov), Adam, AdamW
- **Serialization**: Save/load tensors, models, and training checkpoints in custom .ncta format
- **Memory Efficient**: Reference-counted storage, aligned allocations, detailed memory tracking
- **Hardware Acceleration**: 
  - **CUDA/cuBLAS** for GPU acceleration (NVIDIA GPUs)
  - Multi-threading support via OpenMP
  - SIMD optimizations (AVX2/FMA)
- **Cross-Platform**: Works on Windows (MSVC), Linux (GCC), and macOS (Clang)
- **Zero Dependencies**: No external libraries required (CUDA optional)
- **Nocta Language**: Native scripting language (`.noc`) for rapid prototyping and training without recompilation

## Quick Start

### 1. Nocta Language (Scripting)

The fastest way to use Nocta is via its built-in scripting language, which offers Python-like syntax with native performance.

**`mnist.noc`**
```cs
// Define a LeNet-like model
class LeNet {
    void init() {
        this.c1 = randn(6, 1, 5, 5) * 0.1; requires_grad(this.c1, true);
        this.l1 = randn(120, 16*4*4) * 0.1; requires_grad(this.l1, true);
    }
    
    var forward(x) {
        x = conv2d(x, this.c1, nil, 1, 0);
        x = relu(x);
        x = max_pool2d(x, 2, 2);
        // ... more layers ...
        return x;
    }
}

// Training loop
var model = LeNet();
model.init();

var data = load_mnist(); // Custom helper
for (var i in range(0, 1000)) {
    var pred = model.forward(data.x);
    var loss = cross_entropy(pred, data.y);
    
    backward(loss);
    // optimizer step...
    print("Loss: " + loss);
}
```

Run it immediately:
```bash
./nocta_cli examples/mnist.noc
```

### 2. C API (Embedded)

You can also use Nocta as a pure C library in your own applications.

```c
#include "nocta/nocta.h"

int main() {
    // Operations
    nc_tensor* a = nc_tensor_randn((size_t[]){2, 3}, 2, NC_F32);
    nc_tensor* b = nc_tensor_randn((size_t[]){2, 3}, 2, NC_F32);
    
    nc_tensor_requires_grad_(a, true);
    
    nc_tensor* c = nc_add(a, b);
    nc_tensor* loss = nc_mean_all(nc_relu(c));
    
    // Backward pass
    nc_backward_scalar(loss);
    
    printf("Gradient: %f\n", nc_tensor_get_flat(a->grad, 0));
    
    nc_tensor_free(a); nc_tensor_free(b); nc_tensor_free(c); nc_tensor_free(loss);
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

### Running Tests

To run the unit test suite:

```bash
# Run all tests
ctest -C Release

# Run with verbose output
ctest -C Release --verbose
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `NOCTA_BUILD_EXAMPLES` | ON | Build example programs |
| `NOCTA_ENABLE_SIMD` | ON | Enable AVX2 SIMD optimizations |
| `NOCTA_ENABLE_OPENMP` | ON | Enable OpenMP multi-threading |
| `NOCTA_ENABLE_CUDA` | OFF | Enable CUDA/cuBLAS GPU acceleration |

> **Note**: For best performance, always build in **Release** mode. Debug builds disable optimizations and can be significantly slower.

### Building with CUDA

To enable GPU acceleration, you need CUDA Toolkit installed. Then configure CMake with:

```bash
mkdir build && cd build
cmake .. -DNOCTA_ENABLE_CUDA=ON \
         -DCUDAToolkit_ROOT=/usr/local/cuda-12.x/ \
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.x/bin/nvcc
cmake --build . --config Release
```

Replace `cuda-12.x` with your installed CUDA version (e.g., `cuda-12.9`).

### Nocta CLI

The library now includes a CLI for running `.noc` scripts directly.

```bash
# Start REPL
./nocta_cli

# Run a script
./nocta_cli examples/mnist.noc
```

## Example: CNN with BatchNorm

```c
#include "nocta/nocta.h"
#include "nocta/nn/batchnorm.h"

int main() {
    // Create layers
    nc_module* conv = nc_conv2d(1, 16, 3, 1, 1, false);  // No bias before BN
    nc_module* bn = nc_batchnorm2d(16);
    nc_module* fc = nc_linear(16 * 14 * 14, 10, true);
    
    // Input: (batch=4, channels=1, H=28, W=28)
    size_t shape[] = {4, 1, 28, 28};
    nc_tensor* x = nc_tensor_randn(shape, 4, NC_F32);
    
    // Forward: Conv -> BatchNorm -> ReLU -> Pool -> FC
    nc_tensor* h = nc_module_forward(conv, x);
    nc_tensor* bn_out = nc_module_forward(bn, h);
    nc_tensor* act = nc_relu(bn_out);
    // ... continue with pooling, flatten, fc
    
    // Training vs Eval mode
    nc_module_train(bn, true);   // Use batch statistics
    nc_module_train(bn, false);  // Use running statistics
    
    // Cleanup
    nc_module_free(conv);
    nc_module_free(bn);
    nc_module_free(fc);
    nc_tensor_free(x);
    // ... free other tensors
    
    return 0;
}
```

## API Overview

### Tensor Creation

```c
nc_tensor* nc_tensor_empty(shape, ndim, dtype);
nc_tensor* nc_tensor_zeros(shape, ndim, dtype);
nc_tensor* nc_tensor_ones(shape, ndim, dtype);
nc_tensor* nc_tensor_randn(shape, ndim, dtype);
nc_tensor* nc_tensor_from_data(data, shape, ndim, dtype);
```

### Operations

```c
// Arithmetic
nc_tensor* nc_add(a, b);
nc_tensor* nc_sub(a, b);
nc_tensor* nc_mul(a, b);
nc_tensor* nc_matmul(a, b);

// Activations
nc_tensor* nc_relu(x);
nc_tensor* nc_sigmoid(x);
nc_tensor* nc_softmax(x, dim);

// Reductions
nc_tensor* nc_sum_all(x);
nc_tensor* nc_mean_all(x);
```

### Neural Network Modules

```c
// Linear
nc_module* nc_linear(in_features, out_features, bias);

// Convolution
nc_module* nc_conv2d(in_ch, out_ch, kernel, stride, padding, bias);
nc_module* nc_maxpool2d(kernel_size, stride);

// Normalization
nc_module* nc_batchnorm1d(num_features);
nc_module* nc_batchnorm2d(num_features);
nc_module* nc_layernorm(normalized_shape, ndim);
nc_module* nc_dropout(p);

// Forward pass
nc_tensor* output = nc_module_forward(module, input);

// Training/Eval mode (affects BatchNorm, Dropout)
nc_module_train(module, true);   // Training mode
nc_module_train(module, false);  // Eval mode
```

### Optimizers

```c
nc_optimizer* nc_sgd_momentum(module, lr, momentum);
nc_optimizer* nc_adam_default(module, lr);

nc_optimizer_zero_grad(opt);
nc_optimizer_step(opt);
```

## Project Structure

```
nocta/
├── include/nocta/
│   ├── nocta.h          # Main header
│   ├── core/            # Tensor, memory, types, serialization
│   ├── lang/            # Nocta language (compiler, vm, ast)
│   ├── autograd/        # Automatic differentiation
│   ├── ops/             # Operations (arithmetic, matmul, activations)
│   ├── nn/              # Neural network modules (linear, conv, batchnorm)
│   ├── optim/           # Optimizers
│   └── cuda/            # CUDA kernel declarations
├── src/
│   ├── autograd/        # Automatic differentiation implementations
│   ├── core/            # Core implementations
│   ├── lang/            # Language implementation (lexer, parser, compiler, vm)
│   ├── ops/             # Operation implementations
│   ├── nn/              # Module implementations
│   ├── optim/           # Optimizer implementations
│   ├── cuda/            # CUDA kernel implementations (.cu files)
│   └── main.c           # CLI entry point
├── examples/            # Example programs (C and .noc scripts)
└── CMakeLists.txt
```

## Roadmap

- [x] Model serialization (save/load tensors, modules, checkpoints)
- [x] Convolution layers (Conv2D, MaxPool)
- [x] Batch normalization (BatchNorm1D, BatchNorm2D, LayerNorm)
- [x] SIMD optimizations (AVX2/FMA)
- [x] Multi-threading (OpenMP)
- [x] Dropout
- [x] Nocta Language (Compiler/VM)
- [x] GPU support (CUDA/cuBLAS)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

Inspired by PyTorch, Tinygrad, and micrograd.