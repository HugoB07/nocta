#include "nocta/nn/dropout.h"
#include "nocta/autograd/node.h"
#include "nocta/core/memory.h"
#include <stdlib.h>
#include <math.h>

typedef struct {
    double p;
} nc_dropout_data;

// Backward function for dropout
// saved_tensors[0] is the mask
// saved_tensors[1] is the scale factor (as a scalar tensor)
nc_tensor** nc_backward_dropout(nc_tensor* grad_output, nc_tensor** saved_tensors, size_t n_saved) {
    if (n_saved < 2) return NULL;
    
    nc_tensor* mask = saved_tensors[0];
    nc_tensor* scale_t = saved_tensors[1];
    double scale = nc_tensor_get_flat(scale_t, 0);
    
    // grad_input = grad_output * mask * scale
    // We can do this in place on a clone of grad_output or create new tensor
    // nc_mul creates a new tensor
    
    // Since mask contains 0s and 1s, and we scale, we can just loop
    nc_tensor* grad_input = nc_tensor_empty(grad_output->shape, grad_output->ndim, grad_output->dtype);
    
    size_t n = nc_tensor_numel(grad_output);
    double* g_out = nc_tensor_data_f64(grad_output); // Assuming f64 for now, need to handle types
    double* g_in = nc_tensor_data_f64(grad_input);
    double* m = nc_tensor_data_f64(mask);
    
    // TODO: Handle types properly. For now assuming all are same type (likely float/double)
    // The library seems to use double for get_flat/set_flat but stores as dtype.
    // For performance, we should switch on dtype.
    
    // Using get_flat/set_flat is slow but safe.
    // Let's use get_flat/set_flat for simplicity first, or check how other ops do it.
    
    for (size_t i = 0; i < n; i++) {
        double val = nc_tensor_get_flat(grad_output, i);
        double mask_val = nc_tensor_get_flat(mask, i);
        nc_tensor_set_flat(grad_input, i, val * mask_val * scale);
    }
    
    nc_tensor** grads = nc_alloc(1 * sizeof(nc_tensor*));
    grads[0] = grad_input;
    
    return grads;
}

nc_tensor* nc_dropout_fn(nc_tensor* x, double p, bool training) {
    if (!training || p == 0.0) {
        return nc_tensor_clone(x);
    }
    
    if (p < 0.0 || p >= 1.0) {
        // Invalid p, return clone or error?
        // Return clone for now
        return nc_tensor_clone(x);
    }
    
    double scale = 1.0 / (1.0 - p);
    
    nc_tensor* out = nc_tensor_empty(x->shape, x->ndim, x->dtype);
    nc_tensor* mask = nc_tensor_empty(x->shape, x->ndim, x->dtype); // We need to save this
    
    size_t n = nc_tensor_numel(x);
    
    // Generate mask and output
    for (size_t i = 0; i < n; i++) {
        double rand_val = (double)rand() / RAND_MAX;
        double m = (rand_val > p) ? 1.0 : 0.0;
        
        double val = nc_tensor_get_flat(x, i);
        nc_tensor_set_flat(mask, i, m);
        nc_tensor_set_flat(out, i, val * m * scale);
    }
    
    if (x->requires_grad) {
        nc_node* node = nc_node_create("dropout", nc_backward_dropout);
        nc_node_add_input(node, x);
        
        // Save mask
        nc_node_save_owned_tensor(node, mask);
        
        // Save scale as tensor
        nc_tensor* scale_t = nc_tensor_scalar(scale, NC_F64);
        nc_node_save_owned_tensor(node, scale_t);
        
        out->grad_fn = node;
        out->requires_grad = true;
    } else {
        nc_tensor_free(mask);
    }
    
    return out;
}

static nc_tensor* dropout_forward(nc_module* self, nc_tensor* input) {
    nc_dropout_data* data = (nc_dropout_data*)self->extra;
    return nc_dropout_fn(input, data->p, self->training);
}

nc_module* nc_dropout(double p) {
    nc_module* m = nc_module_create("Dropout");
    if (!m) return NULL;
    
    nc_dropout_data* data = nc_alloc(sizeof(nc_dropout_data));
    if (!data) {
        nc_module_free(m);
        return NULL;
    }
    
    data->p = p;
    
    m->extra = data;
    m->free_extra = nc_free;
    m->forward = dropout_forward;
    
    return m;
}
