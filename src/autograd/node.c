#include "nocta/autograd/node.h"
#include "nocta/core/memory.h"
#include <string.h>

nc_node* nc_node_create(const char* op_name, nc_backward_fn backward) {
    nc_node* node = nc_calloc(1, sizeof(nc_node));
    if (!node) return NULL;
    
    node->op_name = op_name;
    node->backward = backward;
    node->n_inputs = 0;
    node->n_saved = 0;
    node->visited = 0;
    node->ref_count = 0;
    node->output = NULL;
    
    return node;
}

void nc_node_add_input(nc_node* node, nc_tensor* input) {
    if (!node || node->n_inputs >= NC_MAX_INPUTS) return;
    node->inputs[node->n_inputs++] = input;
}

void nc_node_save_tensor(nc_node* node, nc_tensor* t) {
    if (!node || node->n_saved >= NC_MAX_INPUTS) return;
    node->saved_tensors[node->n_saved] = t;
    node->saved_tensors_owned[node->n_saved] = false;
    node->n_saved++;
}

void nc_node_save_owned_tensor(nc_node* node, nc_tensor* t) {
    if (!node || node->n_saved >= NC_MAX_INPUTS) return;
    node->saved_tensors[node->n_saved] = t;
    node->saved_tensors_owned[node->n_saved] = true;
    node->n_saved++;
}

void nc_node_free(nc_node* node) {
    if (!node) return;
    // Free owned saved tensors
    for (size_t i = 0; i < node->n_saved; i++) {
        if (node->saved_tensors_owned[i]) {
            nc_tensor_free(node->saved_tensors[i]);
        }
    }
    
    // Just free the node itself
    nc_free(node);
}