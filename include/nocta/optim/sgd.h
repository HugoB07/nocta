#ifndef NOCTA_SGD_H
#define NOCTA_SGD_H

#include "nocta/optim/optimizer.h"

// SGD optimizer with optional momentum and weight decay
// 
// v = momentum * v + grad
// param = param - lr * v - weight_decay * param
//
// With Nesterov momentum:
// v = momentum * v + grad
// param = param - lr * (grad + momentum * v)

typedef struct {
    double momentum;
    double dampening;
    double weight_decay;
    bool nesterov;
} nc_sgd_config;

// Default config
static const nc_sgd_config NC_SGD_DEFAULT = {
    .momentum = 0.0,
    .dampening = 0.0,
    .weight_decay = 0.0,
    .nesterov = false
};

// Create SGD optimizer
nc_optimizer* nc_sgd(double lr, nc_sgd_config config);

// Create SGD from module
nc_optimizer* nc_sgd_from_module(nc_module* m, double lr, nc_sgd_config config);

// Convenience: simple SGD (no momentum)
nc_optimizer* nc_sgd_simple(nc_module* m, double lr);

// Convenience: SGD with momentum
nc_optimizer* nc_sgd_momentum(nc_module* m, double lr, double momentum);

#endif // NOCTA_SGD_H