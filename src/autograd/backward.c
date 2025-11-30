#include "nocta/autograd/backward.h"
#include "nocta/ops/arithmetic.h"
#include "nocta/core/memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Global state
static nc_grad_mode g_grad_mode = NC_GRAD_ACCUMULATE;
static int g_grad_enabled = 1;

void nc_set_grad_mode(nc_grad_mode mode) {
    g_grad_mode = mode;
}

nc_grad_mode nc_get_grad_mode(void) {
    return g_grad_mode;
}

int nc_grad_enabled(void) {
    return g_grad_enabled;
}

void nc_set_grad_enabled(int enabled) {
    g_grad_enabled = enabled;
}

nc_no_grad_guard nc_no_grad_begin(void) {
    nc_no_grad_guard guard = { .prev_enabled = g_grad_enabled };
    g_grad_enabled = 0;
    return guard;
}

void nc_no_grad_end(nc_no_grad_guard* guard) {
    if (guard) {
        g_grad_enabled = guard->prev_enabled;
    }
}

// Reset visited flags
static void reset_visited_nodes(nc_node** nodes, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (nodes[i]) nodes[i]->visited = 0;
    }
}

// Build topological order using DFS
static void topo_sort_dfs(nc_tensor* t, nc_tensor*** list, size_t* n, size_t* cap, int depth) {
    if (!t || !t->grad_fn || t->grad_fn->visited) return;
    if (depth > 100) return;
    
    t->grad_fn->visited = 1;
    
    nc_node* node = t->grad_fn;
    for (size_t i = 0; i < node->n_inputs; i++) {
        topo_sort_dfs(node->inputs[i], list, n, cap, depth + 1);
    }
    
    if (*n >= *cap) {
        size_t new_cap = (*cap == 0) ? 16 : (*cap * 2);
        nc_tensor** new_list = nc_alloc(new_cap * sizeof(nc_tensor*));
        if (*list && *n > 0) {
            memcpy(new_list, *list, *n * sizeof(nc_tensor*));
            nc_free(*list);
        }
        *list = new_list;
        *cap = new_cap;
    }
    (*list)[(*n)++] = t;
}

nc_node** nc_build_topo(nc_tensor* t, size_t* n_nodes) {
    *n_nodes = 0;
    if (!t || !t->grad_fn) return NULL;
    
    nc_tensor** tensors = NULL;
    size_t n = 0, cap = 0;
    
    topo_sort_dfs(t, &tensors, &n, &cap, 0);
    
    if (n == 0) {
        if (tensors) nc_free(tensors);
        return NULL;
    }
    
    nc_node** nodes = nc_alloc(n * sizeof(nc_node*));
    for (size_t i = 0; i < n; i++) {
        nodes[i] = tensors[i]->grad_fn;
    }
    
    nc_free(tensors);
    *n_nodes = n;
    
    reset_visited_nodes(nodes, n);
    
    return nodes;
}

// Accumulate gradient into tensor
static void accumulate_grad(nc_tensor* t, nc_tensor* grad) {
    if (!t || !grad || !t->requires_grad) return;
    
    if (t->numel != grad->numel) return;
    
    if (!t->grad) {
        t->grad = nc_tensor_zeros(t->shape, t->ndim, t->dtype);
        if (!t->grad) return;
    }
    
    for (size_t i = 0; i < t->numel; i++) {
        double cur = nc_tensor_get_flat(t->grad, i);
        double add = nc_tensor_get_flat(grad, i);
        nc_tensor_set_flat(t->grad, i, cur + add);
    }
}

nc_error nc_backward(nc_tensor* t, nc_tensor* grad_output) {
    if (!t) return NC_ERR_NULL_PTR;
    
    nc_tensor* out_grad = grad_output;
    int free_out_grad = 0;
    
    if (!out_grad) {
        if (t->numel != 1) {
            return NC_ERR_SHAPE_MISMATCH;
        }
        out_grad = nc_tensor_ones(t->shape, t->ndim, t->dtype);
        free_out_grad = 1;
    }
    
    if (!t->grad_fn) {
        if (t->requires_grad) {
            accumulate_grad(t, out_grad);
        }
        if (free_out_grad) nc_tensor_free(out_grad);
        return NC_OK;
    }
    
    size_t n_nodes;
    nc_node** topo = nc_build_topo(t, &n_nodes);
    
    if (!topo || n_nodes == 0) {
        if (free_out_grad) nc_tensor_free(out_grad);
        return NC_OK;
    }
    
    // Map from node -> gradient tensor
    nc_tensor** node_grads = nc_calloc(n_nodes, sizeof(nc_tensor*));
    node_grads[n_nodes - 1] = nc_tensor_clone(out_grad);
    
    // Disable autograd during backward
    int prev_grad_enabled = g_grad_enabled;
    g_grad_enabled = 0;
    
    // Backward pass
    for (int i = (int)n_nodes - 1; i >= 0; i--) {
        nc_node* node = topo[i];
        nc_tensor* grad = node_grads[i];
        
        if (!node || !grad || !node->backward) continue;
        
        nc_tensor** input_grads = node->backward(grad, node->saved_tensors, node->n_saved);
        
        if (!input_grads) continue;
        
        for (size_t j = 0; j < node->n_inputs; j++) {
            nc_tensor* inp = node->inputs[j];
            nc_tensor* inp_grad = input_grads[j];
            
            if (!inp || !inp_grad) continue;
            
            if (inp->grad_fn) {
                // Find node index for this input
                for (size_t k = 0; k < n_nodes; k++) {
                    if (topo[k] == inp->grad_fn) {
                        if (!node_grads[k]) {
                            node_grads[k] = nc_tensor_clone(inp_grad);
                        } else {
                            for (size_t m = 0; m < inp_grad->numel && m < node_grads[k]->numel; m++) {
                                double cur = nc_tensor_get_flat(node_grads[k], m);
                                double add = nc_tensor_get_flat(inp_grad, m);
                                nc_tensor_set_flat(node_grads[k], m, cur + add);
                            }
                        }
                        break;
                    }
                }
            } 
            
            // Always accumulate to leaf tensors
            if (inp->requires_grad && inp->is_leaf) {
                accumulate_grad(inp, inp_grad);
            }
            
            nc_tensor_free(inp_grad);
        }
        
        nc_free(input_grads);
    }
    
    g_grad_enabled = prev_grad_enabled;
    
    // Cleanup
    for (size_t i = 0; i < n_nodes; i++) {
        if (node_grads[i]) nc_tensor_free(node_grads[i]);
    }
    nc_free(node_grads);
    nc_free(topo);
    
    if (free_out_grad) nc_tensor_free(out_grad);
    
    return NC_OK;
}

nc_error nc_backward_scalar(nc_tensor* loss) {
    return nc_backward(loss, NULL);
}