#ifndef NOCTA_ACTIVATION_H
#define NOCTA_ACTIVATION_H

#include "nocta/core/tensor.h"

// ============================================
// Activation functions
// ============================================

// ReLU: max(0, x)
nc_tensor* nc_relu(nc_tensor* x);

// Leaky ReLU: x if x > 0 else alpha * x
nc_tensor* nc_leaky_relu(nc_tensor* x, double alpha);

// ELU: x if x > 0 else alpha * (e^x - 1)
nc_tensor* nc_elu(nc_tensor* x, double alpha);

// SELU: scale * (x if x > 0 else alpha * (e^x - 1))
nc_tensor* nc_selu(nc_tensor* x);

// GELU: x * Φ(x) (Gaussian Error Linear Unit)
nc_tensor* nc_gelu(nc_tensor* x);

// Sigmoid: 1 / (1 + e^(-x))
nc_tensor* nc_sigmoid(nc_tensor* x);

// Tanh: (e^x - e^(-x)) / (e^x + e^(-x))
nc_tensor* nc_tanh_act(nc_tensor* x);

// Softmax: e^x_i / sum(e^x_j)
// dim: dimension along which to compute softmax
nc_tensor* nc_softmax(nc_tensor* x, int dim);

// Log-softmax: log(softmax(x))
nc_tensor* nc_log_softmax(nc_tensor* x, int dim);

// Softplus: log(1 + e^x)
nc_tensor* nc_softplus(nc_tensor* x, double beta, double threshold);

// Softsign: x / (1 + |x|)
nc_tensor* nc_softsign(nc_tensor* x);

// Swish / SiLU: x * sigmoid(x)
nc_tensor* nc_swish(nc_tensor* x);
nc_tensor* nc_silu(nc_tensor* x);  // Alias

// Mish: x * tanh(softplus(x))
nc_tensor* nc_mish(nc_tensor* x);

// Hardtanh: clamp(x, min, max)
nc_tensor* nc_hardtanh(nc_tensor* x, double min_val, double max_val);

// Hardsigmoid: clamp((x + 3) / 6, 0, 1)
nc_tensor* nc_hardsigmoid(nc_tensor* x);

// Hardswish: x * hardsigmoid(x)
nc_tensor* nc_hardswish(nc_tensor* x);

// ReLU6: min(max(0, x), 6)
nc_tensor* nc_relu6(nc_tensor* x);

// PReLU: parametric ReLU
nc_tensor* nc_prelu(nc_tensor* x, nc_tensor* weight);

// ============================================
// In-place activations
// ============================================

void nc_relu_(nc_tensor* x);
void nc_sigmoid_(nc_tensor* x);
void nc_tanh_(nc_tensor* x);

#endif // NOCTA_ACTIVATION_H