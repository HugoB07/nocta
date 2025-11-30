#include "nocta/nn/module.h"
#include <string.h>
#include <stdio.h>

nc_module* nc_module_create(const char* name) {
    nc_module* m = nc_calloc(1, sizeof(nc_module));
    if (!m) return NULL;
    
    if (name) {
        strncpy(m->name, name, NC_MAX_NAME_LEN - 1);
    }
    m->training = true;
    
    return m;
}

void nc_module_free(nc_module* m) {
    if (!m) return;
    
    // Free parameters
    for (size_t i = 0; i < m->n_params; i++) {
        nc_tensor_free(m->params[i]);
        nc_free(m->param_names[i]);
    }
    
    // Free submodules
    for (size_t i = 0; i < m->n_submodules; i++) {
        nc_module_free(m->submodules[i]);
        nc_free(m->submodule_names[i]);
    }
    
    // Free extra data
    if (m->extra && m->free_extra) {
        m->free_extra(m->extra);
    }
    
    nc_free(m);
}

nc_tensor* nc_module_forward(nc_module* m, nc_tensor* input) {
    if (!m || !m->forward) return NULL;
    return m->forward(m, input);
}

void nc_module_add_param(nc_module* m, const char* name, nc_tensor* param) {
    if (!m || !param || m->n_params >= NC_MAX_PARAMS) return;
    
    m->params[m->n_params] = param;
    m->param_names[m->n_params] = nc_alloc(strlen(name) + 1);
    if (m->param_names[m->n_params]) {
        strcpy(m->param_names[m->n_params], name);
    }
    m->n_params++;
    
    // Enable gradients
    nc_tensor_requires_grad_(param, true);
}

void nc_module_add_submodule(nc_module* m, const char* name, nc_module* sub) {
    if (!m || !sub || m->n_submodules >= NC_MAX_SUBMODULES) return;
    
    m->submodules[m->n_submodules] = sub;
    m->submodule_names[m->n_submodules] = nc_alloc(strlen(name) + 1);
    if (m->submodule_names[m->n_submodules]) {
        strcpy(m->submodule_names[m->n_submodules], name);
    }
    m->n_submodules++;
}

nc_tensor* nc_module_get_param(nc_module* m, const char* name) {
    if (!m || !name) return NULL;
    
    for (size_t i = 0; i < m->n_params; i++) {
        if (strcmp(m->param_names[i], name) == 0) {
            return m->params[i];
        }
    }
    return NULL;
}

nc_module* nc_module_get_submodule(nc_module* m, const char* name) {
    if (!m || !name) return NULL;
    
    for (size_t i = 0; i < m->n_submodules; i++) {
        if (strcmp(m->submodule_names[i], name) == 0) {
            return m->submodules[i];
        }
    }
    return NULL;
}

// Recursive parameter iteration
static void iterate_params(nc_module* m, const char* prefix, 
                          nc_param_callback cb, void* ctx) {
    char full_name[256];
    
    // Own parameters
    for (size_t i = 0; i < m->n_params; i++) {
        if (prefix[0]) {
            snprintf(full_name, sizeof(full_name), "%s.%s", prefix, m->param_names[i]);
        } else {
            strncpy(full_name, m->param_names[i], sizeof(full_name) - 1);
        }
        cb(full_name, m->params[i], ctx);
    }
    
    // Submodule parameters
    for (size_t i = 0; i < m->n_submodules; i++) {
        if (prefix[0]) {
            snprintf(full_name, sizeof(full_name), "%s.%s", prefix, m->submodule_names[i]);
        } else {
            strncpy(full_name, m->submodule_names[i], sizeof(full_name) - 1);
        }
        iterate_params(m->submodules[i], full_name, cb, ctx);
    }
}

void nc_module_parameters(nc_module* m, nc_param_callback cb, void* ctx) {
    if (!m || !cb) return;
    iterate_params(m, "", cb, ctx);
}

static void count_params_cb(const char* name, nc_tensor* param, void* ctx) {
    (void)name;
    size_t* count = ctx;
    *count += nc_tensor_numel(param);
}

size_t nc_module_num_parameters(nc_module* m) {
    size_t count = 0;
    nc_module_parameters(m, count_params_cb, &count);
    return count;
}

void nc_module_train(nc_module* m, bool mode) {
    if (!m) return;
    m->training = mode;
    
    for (size_t i = 0; i < m->n_submodules; i++) {
        nc_module_train(m->submodules[i], mode);
    }
}

static void zero_grad_cb(const char* name, nc_tensor* param, void* ctx) {
    (void)name; (void)ctx;
    nc_tensor_zero_grad_(param);
}

void nc_module_zero_grad(nc_module* m) {
    nc_module_parameters(m, zero_grad_cb, NULL);
}

static void set_grad_cb(const char* name, nc_tensor* param, void* ctx) {
    (void)name;
    bool* requires_grad = ctx;
    nc_tensor_requires_grad_(param, *requires_grad);
}

void nc_module_requires_grad_(nc_module* m, bool requires_grad) {
    nc_module_parameters(m, set_grad_cb, &requires_grad);
}

static void print_param_cb(const char* name, nc_tensor* param, void* ctx) {
    size_t* total = ctx;
    size_t n = nc_tensor_numel(param);
    *total += n;
    
    printf("  %s: ", name);
    nc_tensor_print_shape(param);
}

void nc_module_print(nc_module* m) {
    if (!m) return;
    
    printf("Module: %s\n", m->name[0] ? m->name : "(unnamed)");
    printf("Parameters:\n");
    
    size_t total = 0;
    nc_module_parameters(m, print_param_cb, &total);
    
    printf("Total parameters: %zu\n", total);
}