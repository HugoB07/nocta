# Noc Language Specification (Draft 0.1)

**"Noc"** is a high-performance, domain-specific scripting language designed for Artificial Intelligence. It treats Tensors as first-class citizens and eliminates the Python Global Interpreter Lock (GIL) bottleneck.

## 1. Core Philosophy
1.  **Tensors are Primitives**: Not objects. `tensor<2x2>` is as fundamental as `int`.
2.  **Shape Safety**: Optional static shape checking where possible.
3.  **No Global State**: Thread-safe by default.
4.  **Method Chaining**: Transformations apply via methods (`x.view()`).

## 2. Syntax Properties

### 2.1 Basic Types
- `int`, `float`, `bool`, `string`
- `tensor`: The core type.
    - `tensor` (dynamic)
    - `tensor<4, 64>` (static shape)
    - `tensor<B, C, H, W>` (named dimensions / generic)

### 2.2 Variable Declaration
Using `var` (type inference) or explicit types. Semicolons are mandatory.

```noc
int batch_size = 32;
float learning_rate = 1e-3;

// Explicit type
tensor<784, 10> weights = randn(784, 10);
// Implicit type
var bias = zeros(10);
```

### 2.3 Method Chaining
Standard object-oriented chaining.

```noc
tensor x = load_image("digit.png")
      .resize(28, 28)
      .to_tensor()
      .normalize(0.5, 0.5);
```

## 3. Neural Network Definition
Instead of classes (like Python), Noc favors **Composition**.

```noc
// Define a model as a pipeline
Module create_lenet() {
    return sequence(
        conv2d(1, 6, 5),
        relu(),
        maxpool2d(2),
        conv2d(6, 16, 5),
        relu(),
        maxpool2d(2),
        flatten(),
        linear(16*4*4, 120),
        relu(),
        linear(120, 84),
        relu(),
        linear(84, 10)
    );
}

// Training Loop
void train(Module model, DataLoader data) {
    Optimizer opt = optimizer.sgd(model.params, lr=0.01);
    
    for (var batch in data) {
        // Forward
        tensor pred = model(batch.x);
        tensor loss = cross_entropy(pred, batch.y);
        
        // Backward
        backward(loss); // or loss.backward();
        
        // Step
        opt.step();
        opt.zero_grad();
    }
}
```

## 4. Automatic Differentiation
Autograd is built into the language runtime.
- Any expression involving a `tensor` that requires grad automatically builds the graph in the VM.
- `backward(scalar)` triggers the C engine's traversal.

## 5. Parallelism
Since there is no GIL, we can have a `spawn` keyword for true parallel data loading.

```noc
// Parallel data loader
var loader = spawn dataloader("mnist.data");
// Main thread consumes
for (var batch in loader) {
    train_step(batch);
}
```

## 6. Interop with C Engine
The VM will map Noc opcodes directly to `nc_*` C functions.
- `OP_ADD` -> `nc_add(s[0], s[1])`
- `OP_MATMUL` -> `nc_matmul(s[0], s[1])`

## 7. Confirmed Design Choices
1.  **Extension**: `.noc`
2.  **Syntax**: C/C# Style with braces `{ }`.
    - Function definitions: `void name() { ... }`, `int add(int a, int b) { ... }`
    - Control flow: `if (cond) { ... }`, `for (var x in data) { ... }`
3.  **Type System**: Gradual Typing.
    - Dynamic: `let x = ...` (Type inferred, shape checked at runtime)
    - Static: `let w: tensor<10, 10> = ...` (Shape checked at compile time where possible)
