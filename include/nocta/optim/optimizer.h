#ifndef NOCTA_OPTIMIZER_H
#define NOCTA_OPTIMIZER_H

#include "nocta/core/tensor.h"
#include "nocta/nn/module.h"

#define NC_MAX_OPTIM_PARAMS 256

// Forward declaration
typedef struct nc_optimizer nc_optimizer;

// Step function signature
typedef void (*nc_optim_step_fn)(nc_optimizer* self);

// Optimizer base structure
struct nc_optimizer {
    // Parameters to optimize
    nc_tensor* params[NC_MAX_OPTIM_PARAMS];
    size_t n_params;
    
    // Learning rate
    double lr;
    
    // Step function
    nc_optim_step_fn step;
    
    // Optimizer state (momentum, adam states, etc.)
    void* state;
    void (*free_state)(void*);
    
    // Step counter
    size_t t;
};

// ============================================
// Optimizer API
// ============================================

// Free optimizer
void nc_optimizer_free(nc_optimizer* opt);

// Perform optimization step
void nc_optimizer_step(nc_optimizer* opt);

// Zero all gradients
void nc_optimizer_zero_grad(nc_optimizer* opt);

// Set learning rate
void nc_optimizer_set_lr(nc_optimizer* opt, double lr);

// Get current learning rate
double nc_optimizer_get_lr(nc_optimizer* opt);

// ============================================
// Collect parameters from module
// ============================================

// Create optimizer from module
void nc_optimizer_add_module(nc_optimizer* opt, nc_module* m);

// Add single parameter
void nc_optimizer_add_param(nc_optimizer* opt, nc_tensor* param);

#endif // NOCTA_OPTIMIZER_H