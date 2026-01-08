# Nocta Language Guide

The Nocta Language (`.noc`) is a C-like scripting language designed specifically for deep learning. It treats tensors as first-class citizens and provides high-level abstractions over the C library.

## Syntax Overview

### Variables

Variables are dynamically typed but can store strictly typed tensors.

```cs
var x = 10;           // Integer
var y = 3.14;         // Float
var s = "hello";      // String
var t = randn(5, 5);  // Tensor
```

### Functions

Functions are defined with the `var` keyword for dynamic return types, or explicit types like `void`.

```cs
// Function definition
var add(int a, int b) {
    return a + b;
}

// Void function
void log(string msg) {
    print("[LOG] " + msg);
}
```

### Accessing Tensors

Tensors support basic arithmetic operators which are broadcasted.

```cs
var a = ones(2, 2);
var b = ones(2, 2) * 2;
var c = a + b; // Element-wise addition
```

### Control Flow

Standard C-style control structures.

```cs
// If-else
if (loss < 0.01) {
    print("Converged");
} else {
    print("Continuing");
}

// While loop
var i = 0;
while (i < epochs) {
    train_one_epoch();
    i = i + 1;
}

// For loop (range)
// Syntax: for (var item in iterable)
for (var i in range(0, 10)) {
    print(i);
}
```

### Classes

Classes allow you to bundle state and behavior, essential for defining Neural Network modules.

```cs
class Linear {
    // Constructor method
    void init(int in, int out) {
        this.weight = randn(in, out) * 0.01;
        this.bias = zeros(out);
        
        // Enable gradients
        requires_grad(this.weight, true);
        requires_grad(this.bias, true);
    }

    var forward(var x) {
        return matmul(x, this.weight) + this.bias;
    }
}

// Usage
var layer = Linear();
layer.init(784, 128);
var out = layer.forward(input);
```

### Comments

```cs
// Single line comment
```
