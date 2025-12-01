#ifndef NOCTA_MODULE_H
#define NOCTA_MODULE_H

#include "nocta/core/tensor.h"
#include "nocta/core/error.h"
#include <stdbool.h>
#include <string.h>

#define NC_MAX_PARAMS 64
#define NC_MAX_SUBMODULES 32
#define NC_MAX_NAME_LEN 64

// Forward declaration
typedef struct nc_module nc_module;

// Forward function signature
typedef nc_tensor* (*nc_forward_fn)(nc_module* self, nc_tensor* input);

// Module structure (base for all layers)
struct nc_module {
    char name[NC_MAX_NAME_LEN];
    
    // Parameters (weights, biases)
    nc_tensor* params[NC_MAX_PARAMS];
    char* param_names[NC_MAX_PARAMS];
    size_t n_params;
    
    // Submodules
    nc_module* submodules[NC_MAX_SUBMODULES];
    char* submodule_names[NC_MAX_SUBMODULES];
    size_t n_submodules;
    
    // Forward function
    nc_forward_fn forward;
    
    // Training mode
    bool training;
    
    // Module-specific data
    void* extra;
    
    // Cleanup function
    void (*free_extra)(void*);
};

// ============================================
// Module API
// ============================================

// Create empty module
nc_module* nc_module_create(const char* name);

// Free module and all parameters
void nc_module_free(nc_module* m);

// Forward pass
nc_tensor* nc_module_forward(nc_module* m, nc_tensor* input);

// Add parameter to module
void nc_module_add_param(nc_module* m, const char* name, nc_tensor* param);

// Add submodule
void nc_module_add_submodule(nc_module* m, const char* name, nc_module* sub);

// Get parameter by name
nc_tensor* nc_module_get_param(nc_module* m, const char* name);

// Get submodule by name
nc_module* nc_module_get_submodule(nc_module* m, const char* name);

// ============================================
// Parameter iteration
// ============================================

// Callback for parameter iteration
typedef void (*nc_param_callback)(const char* name, nc_tensor* param, void* ctx);

// Iterate over all parameters (including submodules)
void nc_module_parameters(nc_module* m, nc_param_callback cb, void* ctx);

// Count total parameters
size_t nc_module_num_parameters(nc_module* m);

// ============================================
// Training mode
// ============================================

// Set training mode (affects dropout, batchnorm, etc.)
void nc_module_train(nc_module* m, bool mode);

// Set eval mode
static inline void nc_module_eval(nc_module* m) {
    nc_module_train(m, false);
}

// Check if in training mode
static inline bool nc_module_is_training(nc_module* m) {
    return m ? m->training : false;
}

// ============================================
// Gradient management
// ============================================

// Zero gradients for all parameters
void nc_module_zero_grad(nc_module* m);

// Enable/disable gradients for all parameters
void nc_module_requires_grad_(nc_module* m, bool requires_grad);

// ============================================
// Debug
// ============================================

// Print module summary
void nc_module_print(nc_module* m);

#endif // NOCTA_MODULE_H