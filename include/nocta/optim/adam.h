#ifndef NOCTA_ADAM_H
#define NOCTA_ADAM_H

#include "nocta/optim/optimizer.h"

// Adam optimizer (Adaptive Moment Estimation)
//
// m = beta1 * m + (1 - beta1) * grad
// v = beta2 * v + (1 - beta2) * grad^2
// m_hat = m / (1 - beta1^t)
// v_hat = v / (1 - beta2^t)
// param = param - lr * m_hat / (sqrt(v_hat) + eps)

typedef struct {
    double beta1;       // Exponential decay for first moment (default: 0.9)
    double beta2;       // Exponential decay for second moment (default: 0.999)
    double eps;         // Numerical stability (default: 1e-8)
    double weight_decay; // L2 regularization (default: 0)
    bool amsgrad;       // Use AMSGrad variant
} nc_adam_config;

// Default config
static const nc_adam_config NC_ADAM_DEFAULT = {
    .beta1 = 0.9,
    .beta2 = 0.999,
    .eps = 1e-8,
    .weight_decay = 0.0,
    .amsgrad = false
};

// Create Adam optimizer
nc_optimizer* nc_adam(double lr, nc_adam_config config);

// Create Adam from module
nc_optimizer* nc_adam_from_module(nc_module* m, double lr, nc_adam_config config);

// Convenience: Adam with default config
nc_optimizer* nc_adam_default(nc_module* m, double lr);

// AdamW (Adam with decoupled weight decay)
nc_optimizer* nc_adamw(nc_module* m, double lr, double weight_decay);

// Functional API: Single tensor step
void nc_adam_step_single(nc_tensor* param, nc_tensor* grad, 
                        nc_tensor* m, nc_tensor* v, nc_tensor* v_max, 
                        nc_adam_config* config, double lr, int t);

#endif // NOCTA_ADAM_H