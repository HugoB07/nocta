#ifndef NOCTA_NODE_H
#define NOCTA_NODE_H

#include "nocta/core/tensor.h"
#include <stdbool.h>

#define NC_MAX_INPUTS 8

// Forward declaration
typedef struct nc_node nc_node;

// Backward function signature
// Takes: output gradient, input tensors, number of inputs
// Returns: array of input gradients (caller must free)
typedef nc_tensor** (*nc_backward_fn)(nc_tensor* grad_output, nc_tensor** inputs, size_t n_inputs);

// Node in computation graph
struct nc_node {
    // Operation info
    const char* op_name;            // Name for debugging ("add", "matmul", etc.)
    nc_backward_fn backward;        // Backward function
    
    // Inputs
    nc_tensor* inputs[NC_MAX_INPUTS];
    size_t n_inputs;
    
    // Saved tensors for backward (e.g., inputs needed for gradient)
    nc_tensor* saved_tensors[NC_MAX_INPUTS];
    bool saved_tensors_owned[NC_MAX_INPUTS]; // Whether node owns the tensor (should free it)
    size_t n_saved;
    
    // Output (weak reference, not owned)
    nc_tensor* output;
    
    // Graph traversal
    int visited;                    // For topological sort
    int ref_count;                  // Number of outputs depending on this
};

// Create a new node
nc_node* nc_node_create(const char* op_name, nc_backward_fn backward);

// Add input to node
void nc_node_add_input(nc_node* node, nc_tensor* input);

// Save tensor for backward
void nc_node_save_tensor(nc_node* node, nc_tensor* t);

// Save tensor for backward and take ownership (will be freed with node)
void nc_node_save_owned_tensor(nc_node* node, nc_tensor* t);

// Free node
void nc_node_free(nc_node* node);

// ============================================
// Common backward functions
// ============================================

// Element-wise ops
nc_tensor** nc_backward_add(nc_tensor* grad, nc_tensor** inputs, size_t n);
nc_tensor** nc_backward_sub(nc_tensor* grad, nc_tensor** inputs, size_t n);
nc_tensor** nc_backward_mul(nc_tensor* grad, nc_tensor** inputs, size_t n);
nc_tensor** nc_backward_div(nc_tensor* grad, nc_tensor** inputs, size_t n);
nc_tensor** nc_backward_neg(nc_tensor* grad, nc_tensor** inputs, size_t n);

// Matrix ops
nc_tensor** nc_backward_matmul(nc_tensor* grad, nc_tensor** inputs, size_t n);
nc_tensor** nc_backward_transpose(nc_tensor* grad, nc_tensor** inputs, size_t n);

// Activations
nc_tensor** nc_backward_relu(nc_tensor* grad, nc_tensor** inputs, size_t n);
nc_tensor** nc_backward_sigmoid(nc_tensor* grad, nc_tensor** inputs, size_t n);
nc_tensor** nc_backward_tanh(nc_tensor* grad, nc_tensor** inputs, size_t n);
nc_tensor** nc_backward_softmax(nc_tensor* grad, nc_tensor** inputs, size_t n);

// Reductions
nc_tensor** nc_backward_sum(nc_tensor* grad, nc_tensor** inputs, size_t n);
nc_tensor** nc_backward_mean(nc_tensor* grad, nc_tensor** inputs, size_t n);

// Power / exp / log
nc_tensor** nc_backward_pow(nc_tensor* grad, nc_tensor** inputs, size_t n);
nc_tensor** nc_backward_exp(nc_tensor* grad, nc_tensor** inputs, size_t n);
nc_tensor** nc_backward_log(nc_tensor* grad, nc_tensor** inputs, size_t n);

// Convolution
nc_tensor** nc_backward_conv2d(nc_tensor* grad, nc_tensor** inputs, size_t n);
nc_tensor** nc_backward_maxpool2d(nc_tensor* grad, nc_tensor** inputs, size_t n);

// Normalization
nc_tensor** nc_backward_batchnorm2d(nc_tensor* grad, nc_tensor** inputs, size_t n);
nc_tensor** nc_backward_batchnorm1d(nc_tensor* grad, nc_tensor** inputs, size_t n);
nc_tensor** nc_backward_layernorm(nc_tensor* grad, nc_tensor** inputs, size_t n);

#endif // NOCTA_NODE_H