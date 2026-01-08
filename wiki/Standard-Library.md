# Standard Library

Nocta provides a rich set of built-in functions for tensor manipulation, math operations, and neural network training. All functions are available in the global scope.

## Tensor Creation

*   `zeros(dim1, [dim2, ...])`: Creates a tensor of zeros with the specified shape (float32).
*   `ones(dim1, [dim2, ...])`: Creates a tensor of ones with the specified shape.
*   `randn(dim1, [dim2, ...])`: Creates a tensor with random normal values (mean 0, std 1).
*   `full(value, dim1, [dim2, ...])`: Creates a tensor filled with `value`.
*   `range(start, end)`: Creates a range object (iterable).
*   `clone(tensor)`: Returns a deep copy of the tensor.
*   `detach(tensor)`: Returns a new tensor detached from the autograd graph.

## Tensor Operations

*   `tensor_get(tensor, flat_index)`: Returns the scalar value at `flat_index`.
*   `tensor_set(tensor, flat_index, value)`: Sets the scalar value at `flat_index`.
*   `reshape(tensor, dim1, [dim2, ...])`: Returns a view of the tensor with a new shape.
*   `flatten(tensor)`: Flattens the tensor to 1D.
*   `transpose(tensor)`: Transposes the last two dimensions.
*   `contiguous(tensor)`: Returns a contiguous copy of the tensor (if not already contiguous).
*   `slice(tensor, start_idx, length)`: Returns a 1D slice of the tensor (flattened view/copy).

## Math Operations

Arithmetic operations are supported via standard operators which support automatic broadcasting:

*   `+`: Addition. `a + b`
*   `*`: Multiplication. `a * b`
*   `-`: Subtraction. `a - b`
*   `/`: Division. `a / b`
*   `%`: Modulo. `a % b`

### Logical Operations

*   `and`: Logical AND. `a and b`
*   `or`: Logical OR. `a or b`
*   `!`: Logical NOT. `!a`
*   `==`, `!=`: Equality checks.
*   `<`, `>`, `<=`, `>=`: Comparisons.

### Math Functions

*   `matmul(a, b)`: Matrix multiplication.
*   `pow(tensor, exponent)`: Element-wise power.
*   `sqrt(tensor)`: Element-wise square root.
*   `exp(tensor)`: Element-wise exponential.
*   `log(tensor)`: Element-wise natural logarithm.
*   `abs(tensor)`: Element-wise absolute value.
*   `clamp(tensor, min, max)`: Clamps values between min and max.

## Activations

*   `relu(x)`: Rectified Linear Unit. `max(0, x)`.
*   `sigmoid(x)`: Sigmoid activation. `1 / (1 + exp(-x))`.
*   `tanh(x)`: Hyperbolic Tangent activation.
*   `softmax(x, [dim])`: Softmax function along `dim` (default: last dim).
*   `leaky_relu(x, [alpha])`: Leaky ReLU with slope `alpha` (default 0.01).
*   `gelu(x)`: Gaussian Error Linear Unit.
*   `silu(x)`: Sigmoid Linear Unit (SiLU/Swish). `x * sigmoid(x)`.

## Neural Network Layers

*   `linear(input, weight, [bias])`: Linear transformation. `x @ w.T + b`.
*   `conv2d(input, weight, [bias], [stride], [padding])`: 2D Convolution.
*   `max_pool2d(input, kernel_size, [stride])`: 2D Max Pooling.
*   `avg_pool2d(input, kernel_size, [stride])`: 2D Average Pooling.
*   `batch_norm2d(input, running_mean, running_var, weight, bias, training)`: Batch Normalization.
    *   `training`: boolean, true to update running stats, false to use them.
*   `dropout(input, p, [training])`: Randomly zeroes elements with probability `p` (default 0.5) during training.

## Loss Functions

*   `cross_entropy(logits, targets)`: Cross Entropy Loss (LogSoftmax + NLLLoss).
*   `mse_loss(predictions, targets)`: Mean Squared Error Loss.
*   `bce_loss(predictions, targets)`: Binary Cross Entropy Loss.

## Reductions

*   `sum(tensor, [dim], [keepdim])`: Sum along dimension.
*   `sum_all(tensor)`: Sum of all elements.
*   `mean(tensor, [dim], [keepdim])`: Mean along dimension.
*   `mean_all(tensor)`: Mean of all elements.
*   `var(tensor, [dim], [keepdim])`: Variance along dimension.
*   `std(tensor, [dim], [keepdim])`: Standard Deviation along dimension.
*   `norm(tensor, [p])`: Matrix or vector norm (p-norm, default 2.0).
*   `min_all(tensor)`: Minimum value in tensor (scalar).
*   `max_all(tensor)`: Maximum value in tensor (scalar).
*   `argmax(tensor, dim)`: Indices of the maximum values along a dimension.

## Autograd & Optimization

*   `requires_grad(tensor, bool)`: Enables or disables gradient tracking.
*   `backward(loss)`: Computes derivatives.
*   `clear_grad(tensor)`: Zeroes out the gradient of `tensor`.
*   `apply_sgd(param, lr)`: Updates `param` in-place using SGD: `p -= lr * p.grad`.
*   `apply_sgd_momentum(param, velocity, lr, momentum)`: Updates `param` using SGD with momentum.
*   `apply_adam(param, m, v, t, lr, beta1, beta2, eps)`: Updates `param` using Adam optimizer.

## Serialization & I/O

*   `print(...)`: Prints values to stdout. Tensors are printed with shape and data.
*   `read_file(path)`: Reads a file into a 1D float32 tensor (byte values). Useful for loading binary datasets (MNIST).
*   `tensor_save(tensor, path)`: Saves a single tensor to file.
*   `tensor_load(path)`: Loads a single tensor from file.
*   `model_save(t1, t2, ..., path)`: Saves multiple tensors (state dict) to a `.ncta` file.
*   `model_load(path)`: Loads a state dict from file. Current implementation returns the first tensor (legacy) or requires specific handling.
*   `checkpoint_save(epoch, loss, t1, t2, ..., path)`: Saves training checkpoint including metadata.
*   `checkpoint_info(path)`: Prints checkpoint metadata and returns the epoch number.
*   `checkpoint_load_tensor(path, index)`: Loads a specific tensor from a checkpoint by index.

## Hardware

*   `cuda_available()`: Returns `true` if CUDA is supported and available.
*   `to_cuda(tensor)`: Moves tensor to GPU memory.
*   `to_cpu(tensor)`: Moves tensor to CPU memory.

## Introspection / Debugging

*   `tensor_info(tensor)`: Prints detailed tensor metadata (shape, stride, dtype, grad_fn).
*   `count_params(t1, [t2, ...])`: Returns the total number of elements in the given tensors.
*   `print_model(name1, t1, [name2, t2, ...])`: Prints a formatted table of model layers and parameter counts.
