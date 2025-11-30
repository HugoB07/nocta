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
    node->saved_tensors[node->n_saved++] = t;
}

void nc_node_free(nc_node* node) {
    if (!node) return;
    // Don't free inputs or saved_tensors - they're owned elsewhere
    // Just free the node itself
    nc_free(node);
}