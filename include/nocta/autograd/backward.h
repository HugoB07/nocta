#ifndef NOCTA_BACKWARD_H
#define NOCTA_BACKWARD_H

#include "nocta/core/tensor.h"
#include "nocta/autograd/node.h"

// Run backpropagation from a tensor
// The tensor should be a scalar (loss) or grad_output should be provided
nc_error nc_backward(nc_tensor* t, nc_tensor* grad_output);

// Convenience: backward from scalar loss (grad_output = 1)
nc_error nc_backward_scalar(nc_tensor* loss);

// Build topological order of computation graph
// Returns array of nodes in reverse topological order (caller must free)
nc_node** nc_build_topo(nc_tensor* t, size_t* n_nodes);

// Gradient accumulation mode
typedef enum {
    NC_GRAD_REPLACE,    // Replace existing gradients
    NC_GRAD_ACCUMULATE  // Add to existing gradients
} nc_grad_mode;

// Set gradient accumulation mode
void nc_set_grad_mode(nc_grad_mode mode);

// Get current gradient accumulation mode
nc_grad_mode nc_get_grad_mode(void);

// Context manager style: disable gradient tracking temporarily
typedef struct {
    int prev_enabled;
} nc_no_grad_guard;

nc_no_grad_guard nc_no_grad_begin(void);
void nc_no_grad_end(nc_no_grad_guard* guard);

// Check if gradients are enabled globally
int nc_grad_enabled(void);

// Enable/disable gradients globally
void nc_set_grad_enabled(int enabled);

// Macro for no_grad block
#define NC_NO_GRAD_BEGIN() nc_no_grad_guard _ng = nc_no_grad_begin()
#define NC_NO_GRAD_END() nc_no_grad_end(&_ng)

#endif // NOCTA_BACKWARD_H