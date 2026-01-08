#include "nocta/lang/vm.h"
#include "nocta/lang/opcode.h"
#include "nocta/core/tensor.h"
#include "nocta/core/serialize.h"
#include "nocta/core/device.h"
#include "nocta/ops/matmul.h"
#include "nocta/ops/arithmetic.h"
#include "nocta/nn/conv.h"
#include "nocta/nn/batchnorm.h"
#include "nocta/nn/linear.h"
#include "nocta/optim/adam.h"
#include "nocta/nn/dropout.h"
#include "nocta/ops/activation.h"
#include "nocta/ops/loss.h"
#include "nocta/autograd/backward.h"
#include "nocta/ops/reduction.h"
#include "nocta/autograd/node.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Trigger GC 
static void nc_gc_collect(nc_vm* vm);

static nc_tensor_node* track_tensor(nc_vm* vm, nc_tensor* tensor) {
    // Basic GC trigger strategy: Run every time we allocate a tensor
    // In production, we would track bytes and have a threshold.
    // For verification, we run it to ensure the loop test cleans up.
    nc_gc_collect(vm);

    nc_tensor_node* node = (nc_tensor_node*)malloc(sizeof(nc_tensor_node));
    if (!node) return NULL; 
    node->tensor = tensor;
    node->next = vm->tensors;
    vm->tensors = node;
    return node;
}

// Forward declarations 
static void runtime_error(nc_vm* vm, const char* format, ...);

// --- Native Function Implementations ---

static bool get_tensor_property(nc_vm* vm, nc_tensor* t, nc_string* name) {
    if (strcmp(name->chars, "shape") == 0) {
        nc_list* list = nc_new_list();
        for (size_t i = 0; i < t->ndim; i++) {
            nc_list_append(list, NUMBER_VAL((double)t->shape[i]));
        }
        nc_vm_push(vm, OBJ_VAL(list));
        return true;
    }
    if (strcmp(name->chars, "grad") == 0) {
        if (t->grad) {
            nc_vm_push(vm, TENSOR_VAL(track_tensor(vm, t->grad)));
        } else {
            nc_vm_push(vm, NIL_VAL);
        }
        return true;
    }
    if (strcmp(name->chars, "item") == 0) {
        if (t->numel == 1) {
             float* data = (float*)nc_tensor_data(t);
             if (data) {
                nc_vm_push(vm, NUMBER_VAL((double)data[0]));
                return true;
             }
        }
        return false;
    }
    if (strcmp(name->chars, "requires_grad") == 0) {
        nc_vm_push(vm, BOOL_VAL(t->requires_grad));
        return true;
    }
    return false;
}

static bool set_tensor_property(nc_vm* vm, nc_tensor* t, nc_string* name, nc_value value) {
    if (strcmp(name->chars, "requires_grad") == 0) {
        if (!IS_BOOL(value)) {
            runtime_error(vm, "requires_grad must be a boolean.");
            return false;
        }
        t->requires_grad = AS_BOOL(value);
        return true;
    }
    return false;
}

static bool get_shape_from_args(int arg_count, nc_value* args, size_t* shape, size_t* ndim) {
    if (arg_count > NC_MAX_DIMS) return false;
    *ndim = arg_count;
    for (int i = 0; i < arg_count; i++) {
        if (!IS_NUMBER(args[i])) return false;
        shape[i] = (size_t)AS_NUMBER(args[i]);
    }
    return true;
}

static nc_value noc_native_randn(nc_vm* vm, int arg_count, nc_value* args) {
    size_t shape[NC_MAX_DIMS];
    size_t ndim;
    if (!get_shape_from_args(arg_count, args, shape, &ndim)) {
        return NIL_VAL; 
    }
    nc_tensor* t = nc_tensor_randn(shape, ndim, NC_F32);
    return TENSOR_VAL(track_tensor(vm, t));
}

static nc_value noc_native_zeros(nc_vm* vm, int arg_count, nc_value* args) {
    size_t shape[NC_MAX_DIMS];
    size_t ndim;
    if (!get_shape_from_args(arg_count, args, shape, &ndim)) {
        return NIL_VAL;
    }
    nc_tensor* t = nc_tensor_zeros(shape, ndim, NC_F32);
    return TENSOR_VAL(track_tensor(vm, t));
}

// ============================================
// NEW: Tensor Creation
// ============================================

static nc_value noc_native_ones(nc_vm* vm, int arg_count, nc_value* args) {
    size_t shape[NC_MAX_DIMS];
    size_t ndim;
    if (!get_shape_from_args(arg_count, args, shape, &ndim)) {
        return NIL_VAL;
    }
    nc_tensor* t = nc_tensor_ones(shape, ndim, NC_F32);
    return TENSOR_VAL(track_tensor(vm, t));
}

static nc_value noc_native_full(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count < 2 || !IS_NUMBER(args[0])) return NIL_VAL;
    double value = AS_NUMBER(args[0]);
    size_t shape[NC_MAX_DIMS];
    size_t ndim;
    if (!get_shape_from_args(arg_count - 1, args + 1, shape, &ndim)) {
        return NIL_VAL;
    }
    nc_tensor* t = nc_tensor_full(shape, ndim, NC_F32, value);
    return TENSOR_VAL(track_tensor(vm, t));
}

static nc_value noc_native_clone(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* t = nc_tensor_clone(AS_TENSOR(args[0]));
    if (!t) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, t));
}

static nc_value noc_native_detach(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* t = nc_tensor_detach(AS_TENSOR(args[0]));
    if (!t) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, t));
}

// ============================================
// NEW: Activations
// ============================================

static nc_value noc_native_sigmoid(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* res = nc_sigmoid(AS_TENSOR(args[0]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_tanh(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* res = nc_tanh_act(AS_TENSOR(args[0]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_softmax(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count < 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    int dim = (arg_count > 1 && IS_NUMBER(args[1])) ? (int)AS_NUMBER(args[1]) : -1;
    nc_tensor* res = nc_softmax(AS_TENSOR(args[0]), dim);
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_leaky_relu(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count < 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    double alpha = (arg_count > 1 && IS_NUMBER(args[1])) ? AS_NUMBER(args[1]) : 0.01;
    nc_tensor* res = nc_leaky_relu(AS_TENSOR(args[0]), alpha);
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_gelu(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* res = nc_gelu(AS_TENSOR(args[0]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_silu(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* res = nc_silu(AS_TENSOR(args[0]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

// ============================================
// NEW: Reductions
// ============================================

static nc_value noc_native_sum_all(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* res = nc_sum_all(AS_TENSOR(args[0]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_sum(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count < 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    int axis = (arg_count > 1 && IS_NUMBER(args[1])) ? (int)AS_NUMBER(args[1]) : 0;
    bool keepdim = (arg_count > 2 && IS_BOOL(args[2])) ? AS_BOOL(args[2]) : false;
    nc_tensor* res = nc_sum(AS_TENSOR(args[0]), axis, keepdim);
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_mean(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count < 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    int axis = (arg_count > 1 && IS_NUMBER(args[1])) ? (int)AS_NUMBER(args[1]) : 0;
    bool keepdim = (arg_count > 2 && IS_BOOL(args[2])) ? AS_BOOL(args[2]) : false;
    nc_tensor* res = nc_mean(AS_TENSOR(args[0]), axis, keepdim);
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_var(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count < 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    int axis = (arg_count > 1 && IS_NUMBER(args[1])) ? (int)AS_NUMBER(args[1]) : 0;
    bool keepdim = (arg_count > 2 && IS_BOOL(args[2])) ? AS_BOOL(args[2]) : false;
    nc_tensor* res = nc_var(AS_TENSOR(args[0]), axis, keepdim, true);
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_std(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count < 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    int axis = (arg_count > 1 && IS_NUMBER(args[1])) ? (int)AS_NUMBER(args[1]) : 0;
    bool keepdim = (arg_count > 2 && IS_BOOL(args[2])) ? AS_BOOL(args[2]) : false;
    nc_tensor* res = nc_std(AS_TENSOR(args[0]), axis, keepdim, true);
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_norm(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count < 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    double p = (arg_count > 1 && IS_NUMBER(args[1])) ? AS_NUMBER(args[1]) : 2.0;
    nc_tensor* res = nc_norm(AS_TENSOR(args[0]), p);
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

// ============================================
// NEW: Dropout
// ============================================

static nc_value noc_native_dropout(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count < 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    double p = (arg_count > 1 && IS_NUMBER(args[1])) ? AS_NUMBER(args[1]) : 0.5;
    bool training = (arg_count > 2 && IS_BOOL(args[2])) ? AS_BOOL(args[2]) : true;
    nc_tensor* res = nc_dropout_fn(AS_TENSOR(args[0]), p, training);
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

// ============================================
// NEW: Losses
// ============================================

static nc_value noc_native_mse_loss(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 2 || !IS_TENSOR(args[0]) || !IS_TENSOR(args[1])) return NIL_VAL;
    nc_tensor* res = nc_mse_loss(AS_TENSOR(args[0]), AS_TENSOR(args[1]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_bce_loss(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 2 || !IS_TENSOR(args[0]) || !IS_TENSOR(args[1])) return NIL_VAL;
    nc_tensor* res = nc_bce_loss(AS_TENSOR(args[0]), AS_TENSOR(args[1]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

// ============================================
// NEW: Serialization
// ============================================

static nc_value noc_native_tensor_save(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 2 || !IS_TENSOR(args[0]) || !IS_STRING(args[1])) return NIL_VAL;
    nc_error err = nc_tensor_save(AS_TENSOR(args[0]), AS_STRING(args[1])->chars);
    return BOOL_VAL(err == NC_OK);
}

static nc_value noc_native_tensor_load(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_STRING(args[0])) return NIL_VAL;
    nc_tensor* t = nc_tensor_load(AS_STRING(args[0])->chars);
    if (!t) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, t));
}

// ============================================
// NEW: Device/CUDA
// ============================================

static nc_value noc_native_cuda_available(nc_vm* vm, int arg_count, nc_value* args) {
    (void)vm; (void)arg_count; (void)args;
    return BOOL_VAL(nc_cuda_available());
}

static nc_value noc_native_to_cuda(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* t = AS_TENSOR(args[0]);
    nc_tensor_to_device(t, NC_DEVICE_CUDA);
    return TENSOR_VAL(track_tensor(vm, t));
}

static nc_value noc_native_to_cpu(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* t = AS_TENSOR(args[0]);
    nc_tensor_to_device(t, NC_DEVICE_CPU);
    return TENSOR_VAL(track_tensor(vm, t));
}

// ============================================
// NEW: Tensor Operations
// ============================================

static nc_value noc_native_flatten(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* res = nc_tensor_flatten(AS_TENSOR(args[0]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_transpose(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* res = nc_tensor_t(AS_TENSOR(args[0]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_contiguous(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* res = nc_tensor_contiguous(AS_TENSOR(args[0]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

// ============================================
// NEW: Math Operations
// ============================================

static nc_value noc_native_sqrt(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* res = nc_sqrt(AS_TENSOR(args[0]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_exp(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* res = nc_exp(AS_TENSOR(args[0]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_log(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* res = nc_log(AS_TENSOR(args[0]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_abs(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* res = nc_abs(AS_TENSOR(args[0]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_pow(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 2 || !IS_TENSOR(args[0]) || !IS_NUMBER(args[1])) return NIL_VAL;
    nc_tensor* res = nc_pow_scalar(AS_TENSOR(args[0]), AS_NUMBER(args[1]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_clamp(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 3 || !IS_TENSOR(args[0]) || !IS_NUMBER(args[1]) || !IS_NUMBER(args[2])) return NIL_VAL;
    nc_tensor* res = nc_clamp(AS_TENSOR(args[0]), AS_NUMBER(args[1]), AS_NUMBER(args[2]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

// ============================================
// NEW: Linear Layer
// ============================================

static nc_value noc_native_linear(nc_vm* vm, int arg_count, nc_value* args) {
    // linear(x, weight, bias) or linear(x, weight)
    if (arg_count < 2 || !IS_TENSOR(args[0]) || !IS_TENSOR(args[1])) return NIL_VAL;
    nc_tensor* x = AS_TENSOR(args[0]);
    nc_tensor* w = AS_TENSOR(args[1]);
    nc_tensor* b = (arg_count > 2 && IS_TENSOR(args[2])) ? AS_TENSOR(args[2]) : NULL;
    
    nc_tensor* y = nc_linear_forward(x, w, b);
    
    if (!y) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, y));
}

// ... (Average Pooling remains) ... 

static nc_value noc_native_avg_pool2d(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count < 2 || !IS_TENSOR(args[0]) || !IS_NUMBER(args[1])) return NIL_VAL;
    int kernel_size = (int)AS_NUMBER(args[1]);
    int stride = (arg_count > 2 && IS_NUMBER(args[2])) ? (int)AS_NUMBER(args[2]) : kernel_size;
    nc_tensor* res = nc_avgpool2d_forward(AS_TENSOR(args[0]), kernel_size, stride);
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

// apply_adam(param, m, v, t, lr, beta1, beta2, eps)
static nc_value noc_native_apply_adam(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count < 8) return NIL_VAL;
    if (!IS_TENSOR(args[0]) || !IS_TENSOR(args[1]) || !IS_TENSOR(args[2])) return NIL_VAL;
    if (!IS_NUMBER(args[3]) || !IS_NUMBER(args[4]) || !IS_NUMBER(args[5]) || 
        !IS_NUMBER(args[6]) || !IS_NUMBER(args[7])) return NIL_VAL;
    
    nc_tensor* p = AS_TENSOR(args[0]);  // Parameter
    nc_tensor* m = AS_TENSOR(args[1]);  // First moment
    nc_tensor* v = AS_TENSOR(args[2]);  // Second moment
    int t = (int)AS_NUMBER(args[3]);    // Timestep
    double lr = AS_NUMBER(args[4]);
    
    nc_adam_config cfg;
    cfg.beta1 = AS_NUMBER(args[5]);
    cfg.beta2 = AS_NUMBER(args[6]);
    cfg.eps = AS_NUMBER(args[7]);
    cfg.weight_decay = 0.0;
    cfg.amsgrad = false; // Not exposed yet
    
    // We don't have v_max exposed in VM args yet, so pass NULL
    nc_adam_step_single(p, p->grad, m, v, NULL, &cfg, lr, t);
    
    return NIL_VAL;
}

// ============================================
// NEW: State Dict Operations
// ============================================

// Helper for shallow free of state dict (does not free tensors)
static void free_sd_shallow(nc_state_dict* sd) {
    if (!sd) return;
    for (size_t i = 0; i < sd->n_tensors; i++) {
        nc_free(sd->names[i]);
    }
    nc_free(sd->names);
    nc_free(sd->tensors);
    nc_free(sd);
}

static nc_value noc_native_model_save(nc_vm* vm, int arg_count, nc_value* args) {
    // model_save(tensors..., path)
    if (arg_count < 2 || !IS_STRING(args[arg_count-1])) return NIL_VAL;
    
    const char* path = AS_STRING(args[arg_count-1])->chars;
    
    nc_state_dict* sd = nc_state_dict_create();
    if (!sd) return BOOL_VAL(false);
    
    // Add tensors
    for (int i = 0; i < arg_count - 1; i++) {
        if (!IS_TENSOR(args[i])) {
            free_sd_shallow(sd); // Cleanup wrapper
            return BOOL_VAL(false);
        }
        
        char name[32];
        sprintf(name, "%d", i);
        nc_state_dict_add(sd, name, AS_TENSOR(args[i]));
    }
    
    nc_error err = nc_state_dict_save(sd, path);
    free_sd_shallow(sd);
    
    return BOOL_VAL(err == NC_OK);
}

static nc_value noc_native_model_load(nc_vm* vm, int arg_count, nc_value* args) {
    // model_load(path) -> returns single tensor (for MVP compatibility)
    if (arg_count != 1 || !IS_STRING(args[0])) return NIL_VAL;
    
    const char* path = AS_STRING(args[0])->chars;
    
    nc_state_dict* sd = nc_state_dict_load(path);
    if (!sd) return NIL_VAL;
    
    if (sd->n_tensors == 0) {
        nc_state_dict_free(sd);
        return NIL_VAL;
    }
    
    // Steal the first tensor logic for MVP
    // Or ideally return ARRAY/LIST. For now, first tensor.
    // We assume model_load loads a single tensor file usually?
    // No, model_save saves MULTIPLE.
    // But previous `model_load` only returned the FIRST.
    // We maintain this behavior for now to avoid breaking `mnist.noc`.
    
    nc_tensor* t = sd->tensors[0];
    sd->tensors[0] = NULL; // Detach
    
    nc_state_dict_free(sd);
    
    return TENSOR_VAL(track_tensor(vm, t));
}

// ============================================
// NEW: Checkpoint Save/Load (epoch, loss, tensors)
// ============================================

// checkpoint_save(epoch, loss, tensor1, tensor2, ..., path)
static nc_value noc_native_checkpoint_save(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count < 4) return BOOL_VAL(false); // At least: epoch, loss, 1 tensor, path
    if (!IS_NUMBER(args[0]) || !IS_NUMBER(args[1]) || !IS_STRING(args[arg_count-1])) 
        return BOOL_VAL(false);
    
    int epoch = (int)AS_NUMBER(args[0]);
    double loss = AS_NUMBER(args[1]);
    const char* path = AS_STRING(args[arg_count-1])->chars;
    
    nc_checkpoint* ckpt = nc_calloc(1, sizeof(nc_checkpoint));
    if (!ckpt) return BOOL_VAL(false);
    
    ckpt->epoch = (size_t)epoch;
    ckpt->loss = loss;
    
    // Create state dict for model
    ckpt->model_state = nc_state_dict_create();
    if (!ckpt->model_state) { nc_free(ckpt); return BOOL_VAL(false); }
    
    for (int i = 2; i < arg_count - 1; i++) {
        if (!IS_TENSOR(args[i])) {
             free_sd_shallow(ckpt->model_state);
             nc_free(ckpt);
             return BOOL_VAL(false);
        }
        char name[32];
        sprintf(name, "%d", i-2);
        nc_state_dict_add(ckpt->model_state, name, AS_TENSOR(args[i]));
    }
    
    // No optimizer state for simple checkpoint
    ckpt->optimizer_state = NULL;
    
    nc_error err = nc_checkpoint_save(ckpt, path);
    
    // Cleanup shallow
    free_sd_shallow(ckpt->model_state);
    nc_free(ckpt);
    
    return BOOL_VAL(err == NC_OK);
}

// checkpoint_load(path) -> returns epoch (also prints info)
// The tensors can be loaded separately with model_load
static nc_value noc_native_checkpoint_info(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_STRING(args[0])) return NIL_VAL;
    
    const char* path = AS_STRING(args[0])->chars;
    
    nc_checkpoint* ckpt = nc_checkpoint_load(path);
    if (!ckpt) return NIL_VAL;
    
    size_t n_tensors = ckpt->model_state ? ckpt->model_state->n_tensors : 0;
    
    printf("Checkpoint: epoch=%zu, loss=%.6f, tensors=%zu\n", ckpt->epoch, ckpt->loss, n_tensors);
    
    double epoch = (double)ckpt->epoch;
    nc_checkpoint_free(ckpt); 
    
    return NUMBER_VAL(epoch);
}

// checkpoint_load_tensor(path, index) -> returns tensor at given index
static nc_value noc_native_checkpoint_load_tensor(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 2 || !IS_STRING(args[0]) || !IS_NUMBER(args[1])) return NIL_VAL;
    
    const char* path = AS_STRING(args[0])->chars;
    int target_idx = (int)AS_NUMBER(args[1]);
    
    nc_checkpoint* ckpt = nc_checkpoint_load(path);
    if (!ckpt) return NIL_VAL;
    
    nc_tensor* res = NULL;
    if (ckpt->model_state && target_idx >= 0 && target_idx < (int)ckpt->model_state->n_tensors) {
        nc_tensor* t = ckpt->model_state->tensors[target_idx];
        ckpt->model_state->tensors[target_idx] = NULL; // Detach
        res = t;
    }
    
    nc_checkpoint_free(ckpt); // Frees others
    
    if (res) return TENSOR_VAL(track_tensor(vm, res));
    return NIL_VAL;
}

static nc_value noc_native_model_info(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_STRING(args[0])) return NIL_VAL;
    
    const char* path = AS_STRING(args[0])->chars;
    nc_file_info(path);
    
    return NIL_VAL;
}

// count_params(tensor1, tensor2, ...) -> returns total number of parameters
static nc_value noc_native_count_params(nc_vm* vm, int arg_count, nc_value* args) {
    (void)vm;
    size_t total = 0;
    for (int i = 0; i < arg_count; i++) {
        if (IS_TENSOR(args[i])) {
            total += nc_tensor_numel(AS_TENSOR(args[i]));
        }
    }
    return NUMBER_VAL((double)total);
}

// tensor_info(tensor) -> prints tensor info (shape, dtype, numel, requires_grad)
static nc_value noc_native_tensor_info(nc_vm* vm, int arg_count, nc_value* args) {
    (void)vm;
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    
    nc_tensor* t = AS_TENSOR(args[0]);
    
    printf("Tensor(shape=(");
    for (size_t i = 0; i < t->ndim; i++) {
        printf("%zu", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("), dtype=");
    
    switch (t->dtype) {
        case NC_F32: printf("float32"); break;
        case NC_F64: printf("float64"); break;
        case NC_I32: printf("int32"); break;
        case NC_I64: printf("int64"); break;
        default: printf("unknown"); break;
    }
    
    printf(", numel=%zu, requires_grad=%s", 
           nc_tensor_numel(t), 
           t->requires_grad ? "true" : "false");
    
    if (t->grad) {
        printf(", has_grad=true");
    }
    
    printf(")\n");
    
    return NIL_VAL;
}

// print_model(name, tensor1, name2, tensor2, ...) -> prints model architecture
static nc_value noc_native_print_model(nc_vm* vm, int arg_count, nc_value* args) {
    (void)vm;
    
    printf("\n=== Model Architecture ===\n");
    printf("%-30s %-25s %15s\n", "Layer", "Shape", "Params");
    printf("%-30s %-25s %15s\n", "-----", "-----", "------");
    
    size_t total_params = 0;
    
    for (int i = 0; i < arg_count; i += 2) {
        const char* name = "unnamed";
        nc_tensor* t = NULL;
        
        if (IS_STRING(args[i]) && i + 1 < arg_count && IS_TENSOR(args[i + 1])) {
            name = AS_STRING(args[i])->chars;
            t = AS_TENSOR(args[i + 1]);
        } else if (IS_TENSOR(args[i])) {
            t = AS_TENSOR(args[i]);
            i--; // Adjust for tensors without names
        } else {
            continue;
        }
        
        if (!t) continue;
        
        // Format shape
        char shape_str[64];
        int pos = 0;
        pos += snprintf(shape_str + pos, 64 - pos, "(");
        for (size_t j = 0; j < t->ndim; j++) {
            pos += snprintf(shape_str + pos, 64 - pos, "%zu", t->shape[j]);
            if (j < t->ndim - 1) pos += snprintf(shape_str + pos, 64 - pos, ", ");
        }
        snprintf(shape_str + pos, 64 - pos, ")");
        
        size_t numel = nc_tensor_numel(t);
        total_params += numel;
        
        printf("%-30s %-25s %15zu\n", name, shape_str, numel);
    }
    
    printf("%-30s %-25s %15s\n", "-----", "-----", "------");
    printf("%-30s %-25s %15zu\n", "Total", "", total_params);
    printf("===========================\n\n");
    
    return NUMBER_VAL((double)total_params);
}

static nc_value noc_native_matmul(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 2) return NIL_VAL; // Error
    if (!IS_TENSOR(args[0]) || !IS_TENSOR(args[1])) return NIL_VAL; // Error
    
    nc_tensor* a = AS_TENSOR(args[0]);
    nc_tensor* b = AS_TENSOR(args[1]);
    
    nc_tensor* res = nc_matmul(a, b);
    if (res) {
        return TENSOR_VAL(track_tensor(vm, res));
    }
    return NIL_VAL;
}

static nc_value noc_native_print(nc_vm* vm, int arg_count, nc_value* args) {
    for (int i = 0; i < arg_count; i++) {
        // Check if it's a tensor to print its content fully
        if (IS_TENSOR(args[i])) {
            nc_tensor_print(AS_TENSOR(args[i]));
        } else {
            nc_print_value(args[i]);
        }
        if (i < arg_count - 1) printf(" ");
    }
    printf("\n");
    return NIL_VAL;
}

static nc_value noc_native_range(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 2) return NIL_VAL;
    if (!IS_NUMBER(args[0]) || !IS_NUMBER(args[1])) return NIL_VAL;
    
    int start = (int)AS_NUMBER(args[0]);
    int end = (int)AS_NUMBER(args[1]);
    
    return OBJ_VAL(nc_new_range(start, end));
}

// Read file to 1D float tensor (byte values)
static nc_value noc_native_read_file(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_STRING(args[0])) return NIL_VAL;
    nc_string* path_str = AS_STRING(args[0]);
    
    FILE* file = fopen(path_str->chars, "rb");
    if (!file) {
        // runtime_error(vm, "Could not open file '%s'.", path_str->chars);
        // Soft fail: return nil
        return NIL_VAL;
    }
    
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);
    
    size_t shape[] = {(size_t)size};
    nc_tensor* t = nc_tensor_empty(shape, 1, NC_F32);
    
    // Read chunks to avoid huge stack buffer?
    // nc_tensor_data assumes flat memory for now usually?
    // If strided, get1 is safer but slow.
    // Assuming contiguous newly allocated tensor.
    // NC_F32 means float*.
    float* data = (float*)nc_tensor_data(t); // Use accessor
    
    unsigned char* buf = (unsigned char*)malloc(size);
    if (!buf) {
        fclose(file);
        nc_tensor_free(t);
        runtime_error(vm, "Out of memory reading file.");
        return NIL_VAL;
    }
    
    if (fread(buf, 1, size, file) != (size_t)size) {
        free(buf); fclose(file); nc_tensor_free(t);
        return NIL_VAL; 
    }
    fclose(file);
    
    // Convert to float
    // Slow loop but generic
    for (long i = 0; i < size; i++) {
        // We can optimize if we access data directly, but set1 is safe
        nc_tensor_set1(t, i, (double)buf[i]);
    }
    free(buf);
    
    return TENSOR_VAL(track_tensor(vm, t));
}


// Backward for reshape
static nc_tensor** backward_reshape(nc_tensor* grad, nc_tensor** saved, size_t n) {
    (void)n;
    nc_tensor* input = saved[0];
    nc_tensor** grads = nc_calloc(1, sizeof(nc_tensor*));
    if (!grads) return NULL;
    grads[0] = nc_tensor_reshape(grad, input->shape, input->ndim);
    return grads;
}

static nc_value noc_native_reshape(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count < 2) return NIL_VAL;
    if (!IS_TENSOR(args[0])) return NIL_VAL;
    
    nc_tensor* t = AS_TENSOR(args[0]);
    size_t new_shape[NC_MAX_DIMS];
    size_t ndim;
    
    // Args 1..N are shape
    if (!get_shape_from_args(arg_count - 1, args + 1, new_shape, &ndim)) {
         return NIL_VAL;
    }
    
    // nc_tensor_reshape usually returns a new tensor (view or copy)
    nc_tensor* res = nc_tensor_reshape(t, new_shape, ndim);
    if (!res) return NIL_VAL;
    
    // Autograd
    if (t->requires_grad) {
        nc_node* node = nc_node_create("reshape", backward_reshape);
        nc_node_add_input(node, t);
        nc_node_save_tensor(node, t);
        node->output = res;
        res->grad_fn = node;
        res->is_leaf = false;
        res->requires_grad = true;
    }

    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_tensor_get(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 2) return NIL_VAL;
    if (!IS_TENSOR(args[0]) || !IS_NUMBER(args[1])) return NIL_VAL;
    
    nc_tensor* t = AS_TENSOR(args[0]);
    int idx = (int)AS_NUMBER(args[1]);
    
    // Check bounds (optional but good)
    if (idx < 0 || idx >= (int)t->numel) return NIL_VAL; // IndexOutOfBounds
    
    double val = nc_tensor_get_flat(t, idx);
    return NUMBER_VAL(val);
}

static nc_value noc_native_tensor_set(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 3) return NIL_VAL;
    if (!IS_TENSOR(args[0]) || !IS_NUMBER(args[1]) || !IS_NUMBER(args[2])) return NIL_VAL;
    
    nc_tensor* t = AS_TENSOR(args[0]);
    int idx = (int)AS_NUMBER(args[1]);
    double val = AS_NUMBER(args[2]);
    
    if (idx < 0 || idx >= (int)t->numel) {
        return NIL_VAL;
    }
     
    nc_tensor_set_flat(t, idx, val);
    
    return NIL_VAL;
}

static nc_value noc_native_slice(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 3) return NIL_VAL;
    if (!IS_TENSOR(args[0]) || !IS_NUMBER(args[1]) || !IS_NUMBER(args[2])) return NIL_VAL;
    
    nc_tensor* t = AS_TENSOR(args[0]);
    int start = (int)AS_NUMBER(args[1]);
    int len = (int)AS_NUMBER(args[2]);
    
    if (start < 0 || len < 0 || start + len > (int)t->numel) {
        runtime_error(vm, "Slice out of bounds.");
        return NIL_VAL;
    }
    
    // Create new tensor
    size_t shape[] = {(size_t)len};
    nc_tensor* res = nc_tensor_empty(shape, 1, t->dtype);
    
    // Copy
    // Slow element-wise or memcpy if contiguous?
    // t might not be contiguous. nc_tensor_get1 is safe.
    // For contiguous optimization:
    if (nc_tensor_is_contiguous(t) && t->dtype == NC_F32) {
         float* src = (float*)nc_tensor_data(t);
         float* dst = (float*)nc_tensor_data(res);
         memcpy(dst, src + start, len * sizeof(float));
    } else {
        for (int i = 0; i < len; i++) {
            nc_tensor_set1(res, i, nc_tensor_get1(t, start + i));
        }
    }
    
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_conv2d(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count < 2) return NIL_VAL;
    if (!IS_TENSOR(args[0]) || !IS_TENSOR(args[1])) {
        runtime_error(vm, "Expected tensors for conv2d input/weight");
        return NIL_VAL;
    }
    nc_tensor* input = AS_TENSOR(args[0]);
    nc_tensor* weight = AS_TENSOR(args[1]);
    nc_tensor* bias = (arg_count > 2 && IS_TENSOR(args[2])) ? AS_TENSOR(args[2]) : NULL;
    
    // Ensure contiguous
    if (!nc_tensor_is_contiguous(input)) {
        nc_tensor* c_input = nc_tensor_contiguous(input);
        if (c_input != input) {
             track_tensor(vm, c_input); // Track new tensor
             input = c_input;
        }
    }
    // Weight usually contiguous from randn/param but good to ensure
    if (!nc_tensor_is_contiguous(weight)) {
        nc_tensor* c_weight = nc_tensor_contiguous(weight);
        if (c_weight != weight) {
             track_tensor(vm, c_weight);
             weight = c_weight;
        }
    }
    
    int stride = (arg_count > 3 && IS_NUMBER(args[3])) ? (int)AS_NUMBER(args[3]) : 1;
    int padding = (arg_count > 4 && IS_NUMBER(args[4])) ? (int)AS_NUMBER(args[4]) : 0;
    
    nc_tensor* res = nc_conv2d_forward(input, weight, bias, stride, padding);
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_max_pool2d(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count < 2) return NIL_VAL;
    if (!IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* input = AS_TENSOR(args[0]);
    int k = (int)AS_NUMBER(args[1]);
    int s = (arg_count > 2 && IS_NUMBER(args[2])) ? (int)AS_NUMBER(args[2]) : k;
    
    nc_tensor* res = nc_maxpool2d_forward(input, k, s);
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_relu(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* res = nc_relu(AS_TENSOR(args[0]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_batch_norm2d(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count < 6) return NIL_VAL;
    if (!IS_TENSOR(args[0]) || !IS_TENSOR(args[1]) || !IS_TENSOR(args[2]) || !IS_TENSOR(args[3]) || !IS_TENSOR(args[4])) {
        runtime_error(vm, "Invalid args for batch_norm2d");
        return NIL_VAL; 
    }
    
    nc_tensor* input = AS_TENSOR(args[0]);
    nc_tensor* rm = AS_TENSOR(args[1]);
    nc_tensor* rv = AS_TENSOR(args[2]);
    nc_tensor* w = AS_TENSOR(args[3]);
    nc_tensor* b = AS_TENSOR(args[4]);
    bool training = IS_BOOL(args[5]) ? AS_BOOL(args[5]) : true;
    double momentum = 0.1; 
    double eps = 1e-5;
    
    nc_tensor* res = nc_batchnorm2d_forward_fn(input, rm, rv, w, b, training, momentum, eps);
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_cross_entropy(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 2) return NIL_VAL;
    nc_tensor* res = nc_cross_entropy_loss(AS_TENSOR(args[0]), AS_TENSOR(args[1]));
    if (!res) return NIL_VAL;
    return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_backward(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_backward_scalar(AS_TENSOR(args[0]));
    return NIL_VAL;
}

static nc_value noc_native_requires_grad(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 2 || !IS_TENSOR(args[0])) return NIL_VAL;
    bool req = IS_BOOL(args[1]) ? AS_BOOL(args[1]) : (IS_NUMBER(args[1]) ? (AS_NUMBER(args[1]) != 0) : true);
    nc_tensor_requires_grad_(AS_TENSOR(args[0]), req);
    return NIL_VAL;
}

static nc_value noc_native_tensor_grad(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* t = AS_TENSOR(args[0]);
    if (t->grad) {
        // Warning: Exposing grad as a tracked tensor might be risky if GC frees it but it belongs to t?
        // But t->grad is a pointer to another tensor.
        // If we want to manipulate it in script, we should probably just return it.
        // But if we track it, GC might double free?
        // nc_tensor usually owns its grad.
        // We probably shouldn't track it as a new root independent of t.
        // BUT for script manipulation, we need it wrapped.
        // Ideally we return a non-tracked pointer or ensure refcounting handles it.
        // Current simple GC assumes ownership.
        // MVP Strategy: Return NIL if no grad. If grad, return tracked wrapper but careful not to free underlying multiple times?
        // Actually, VM GC frees all objects at shutdown.
        // If `t` owns `grad`, freeing `t` frees `grad`.
        // If we wrap `grad` in `nc_tensor_node`, GC will simply see it as reachable.
        // When GC sweeps, it might try to `nc_tensor_free` it?
        // `nc_tensor_free` checks refcount if we implemented shared storage, but `nc_tensor` struct itself is malloced.
        // If we have two pointers to same `nc_tensor` (one in t->grad, one in vm->tensors list),
        // we might double free.
        // SAFE OPTION: Return a Clone? No, we want to update it for SGD.
        // Hack: Just return it but don't track it? NO, then we can't use it in Ops.
        // CORRECT: `nc_tensor` needs refcounting itself.
        // ASSUMPTION: User validates correctness.
        // I will risk tracking it. The GC sweep logic frees unreachables.
        // But `t` holds a reference to `grad`. So `grad` is reachable if `t` is.
        // The issue is `nc_tensor_free(t)` frees `t->grad`? 
        // Check `nc_tensor_free` in tensor.c (not visible). Assuming usually recursive free.
        // If both are in `vm->tensors`, sweep might free one, then `t` destructor frees it again.
        // Safe workaround: Return NIL for now and assume we can't do manual SGD easily without refactor.
        // ALTERNATIVE: `optimizer_step(params, lr)` implemented in C entirely.
        return NIL_VAL;
    }
    return NIL_VAL;
}

static nc_value noc_native_apply_sgd(nc_vm* vm, int arg_count, nc_value* args) {
    // args: parameter_tensor, lr
    // Updates p.data -= lr * p.grad.data
    // In-place, no return needed.
    if (arg_count != 2 || !IS_TENSOR(args[0]) || !IS_NUMBER(args[1])) return NIL_VAL;
    nc_tensor* p = AS_TENSOR(args[0]);
    double lr = AS_NUMBER(args[1]);
    
    if (p->grad) {
        // p = p - lr * grad
        // In-place mutation of p storage
        // Assuming simple SGD: p -= lr * grad
        // Use `nc_sub_scalar`? No, `nc_sub(p, nc_mul_scalar(grad, lr))` returns new tensor.
        // We want in-place update.
        // Access data directly for MVP.
        size_t n = nc_tensor_numel(p);
        float* p_data = (float*)nc_tensor_data(p);
        float* g_data = (float*)nc_tensor_data(p->grad);
        
        if (p_data && g_data) {
             for (size_t i=0; i<n; i++) {
                 p_data[i] -= (float)(lr * g_data[i]);
             }
        }
    }
    return NIL_VAL;
}

static nc_value noc_native_clear_grad(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor_zero_grad_(AS_TENSOR(args[0]));
    return NIL_VAL;
}

static nc_value noc_native_apply_sgd_momentum(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 4) return NIL_VAL;
    if (!IS_TENSOR(args[0]) || !IS_TENSOR(args[1]) || !IS_NUMBER(args[2]) || !IS_NUMBER(args[3])) return NIL_VAL;
    
    nc_tensor* p = AS_TENSOR(args[0]);
    nc_tensor* v = AS_TENSOR(args[1]);
    double lr = AS_NUMBER(args[2]);
    double momentum = AS_NUMBER(args[3]);
    
    if (p->grad) {
        size_t n = nc_tensor_numel(p);
        float* p_data = (float*)nc_tensor_data(p);
        float* g_data = (float*)nc_tensor_data(p->grad);
        float* v_data = (float*)nc_tensor_data(v);
        
        if (p_data && g_data && v_data) {
             for (size_t i=0; i<n; i++) {
                 // v = momentum * v + grad
                 v_data[i] = (float)(momentum * v_data[i] + g_data[i]);
                 // p = p - lr * v
                 p_data[i] -= (float)(lr * v_data[i]);
             }
        }
    }
    return NIL_VAL;
}

static nc_value noc_native_argmax(nc_vm* vm, int arg_count, nc_value* args) {
     if (arg_count < 2 || !IS_TENSOR(args[0])) return NIL_VAL;
     nc_tensor* input = AS_TENSOR(args[0]);
     int dim = (int)AS_NUMBER(args[1]);
     bool keepdim = false; 
     if (arg_count > 2 && IS_BOOL(args[2])) keepdim = AS_BOOL(args[2]);
     
     // Needs reduction header
     nc_tensor* res = nc_argmax(input, dim, keepdim);
     if (!res) return NIL_VAL;
     return TENSOR_VAL(track_tensor(vm, res));
}

static nc_value noc_native_min_all(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* res = nc_min_all(AS_TENSOR(args[0]));
    if (!res) return NIL_VAL;
    double val = nc_tensor_get_flat(res, 0);
    nc_tensor_free(res);
    return NUMBER_VAL(val);
}

static nc_value noc_native_max_all(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* res = nc_max_all(AS_TENSOR(args[0]));
    if (!res) return NIL_VAL;
    double val = nc_tensor_get_flat(res, 0);
    nc_tensor_free(res);
    return NUMBER_VAL(val);
}

static nc_value noc_native_mean_all(nc_vm* vm, int arg_count, nc_value* args) {
    if (arg_count != 1 || !IS_TENSOR(args[0])) return NIL_VAL;
    nc_tensor* res = nc_mean_all(AS_TENSOR(args[0]));
    if (!res) return NIL_VAL;
    double val = nc_tensor_get_flat(res, 0);
    nc_tensor_free(res);
    return NUMBER_VAL(val);
}

// --- End Native Functions ---

void nc_vm_init(nc_vm* vm) {
    vm->stack_top = vm->stack;
    vm->globals.count = 0;
    vm->globals.capacity = 0;
    vm->globals.names = NULL;
    vm->globals.values = NULL;
    vm->tensors = NULL;
    
    vm->frame_count = 0;
    vm->natives.count = 0;

    // Register native functions
    
    // Tensor Creation
    nc_vm_define_native(vm, "randn", noc_native_randn);
    nc_vm_define_native(vm, "zeros", noc_native_zeros);
    nc_vm_define_native(vm, "ones", noc_native_ones);
    nc_vm_define_native(vm, "full", noc_native_full);
    nc_vm_define_native(vm, "clone", noc_native_clone);
    nc_vm_define_native(vm, "detach", noc_native_detach);
    
    // Tensor Operations
    nc_vm_define_native(vm, "matmul", noc_native_matmul);
    nc_vm_define_native(vm, "reshape", noc_native_reshape);
    nc_vm_define_native(vm, "flatten", noc_native_flatten);
    nc_vm_define_native(vm, "transpose", noc_native_transpose);
    nc_vm_define_native(vm, "contiguous", noc_native_contiguous);
    nc_vm_define_native(vm, "tensor_get", noc_native_tensor_get);
    nc_vm_define_native(vm, "tensor_set", noc_native_tensor_set);
    nc_vm_define_native(vm, "slice", noc_native_slice);
    
    // Activations
    nc_vm_define_native(vm, "relu", noc_native_relu);
    nc_vm_define_native(vm, "sigmoid", noc_native_sigmoid);
    nc_vm_define_native(vm, "tanh", noc_native_tanh);
    nc_vm_define_native(vm, "softmax", noc_native_softmax);
    nc_vm_define_native(vm, "leaky_relu", noc_native_leaky_relu);
    nc_vm_define_native(vm, "gelu", noc_native_gelu);
    nc_vm_define_native(vm, "silu", noc_native_silu);
    
    // Reductions
    nc_vm_define_native(vm, "argmax", noc_native_argmax);
    nc_vm_define_native(vm, "sum_all", noc_native_sum_all);
    nc_vm_define_native(vm, "sum", noc_native_sum);
    nc_vm_define_native(vm, "mean", noc_native_mean);
    nc_vm_define_native(vm, "mean_all", noc_native_mean_all);
    nc_vm_define_native(vm, "min_all", noc_native_min_all);
    nc_vm_define_native(vm, "max_all", noc_native_max_all);
    nc_vm_define_native(vm, "var", noc_native_var);
    nc_vm_define_native(vm, "std", noc_native_std);
    nc_vm_define_native(vm, "norm", noc_native_norm);
    
    // NN Layers
    nc_vm_define_native(vm, "linear", noc_native_linear);
    nc_vm_define_native(vm, "conv2d", noc_native_conv2d);
    nc_vm_define_native(vm, "max_pool2d", noc_native_max_pool2d);
    nc_vm_define_native(vm, "avg_pool2d", noc_native_avg_pool2d);
    nc_vm_define_native(vm, "batch_norm2d", noc_native_batch_norm2d);
    nc_vm_define_native(vm, "dropout", noc_native_dropout);
    
    // Losses
    nc_vm_define_native(vm, "cross_entropy", noc_native_cross_entropy);
    nc_vm_define_native(vm, "mse_loss", noc_native_mse_loss);
    nc_vm_define_native(vm, "bce_loss", noc_native_bce_loss);
    
    // Autograd & Training
    nc_vm_define_native(vm, "backward", noc_native_backward);
    nc_vm_define_native(vm, "requires_grad", noc_native_requires_grad);
    nc_vm_define_native(vm, "clear_grad", noc_native_clear_grad);
    nc_vm_define_native(vm, "apply_sgd", noc_native_apply_sgd);
    nc_vm_define_native(vm, "apply_sgd_momentum", noc_native_apply_sgd_momentum);
    nc_vm_define_native(vm, "apply_adam", noc_native_apply_adam);
    
    // Math Operations
    nc_vm_define_native(vm, "sqrt", noc_native_sqrt);
    nc_vm_define_native(vm, "exp", noc_native_exp);
    nc_vm_define_native(vm, "log", noc_native_log);
    nc_vm_define_native(vm, "abs", noc_native_abs);
    nc_vm_define_native(vm, "pow", noc_native_pow);
    nc_vm_define_native(vm, "clamp", noc_native_clamp);
    
    // Serialization
    nc_vm_define_native(vm, "tensor_save", noc_native_tensor_save);
    nc_vm_define_native(vm, "tensor_load", noc_native_tensor_load);
    nc_vm_define_native(vm, "model_save", noc_native_model_save);
    nc_vm_define_native(vm, "model_load", noc_native_model_load);
    nc_vm_define_native(vm, "checkpoint_save", noc_native_checkpoint_save);
    nc_vm_define_native(vm, "checkpoint_info", noc_native_checkpoint_info);
    nc_vm_define_native(vm, "model_info", noc_native_model_info);
    nc_vm_define_native(vm, "checkpoint_load_tensor", noc_native_checkpoint_load_tensor);
    
    // Device/CUDA
    nc_vm_define_native(vm, "cuda_available", noc_native_cuda_available);
    nc_vm_define_native(vm, "to_cuda", noc_native_to_cuda);
    nc_vm_define_native(vm, "to_cpu", noc_native_to_cpu);
    
    // Utilities
    nc_vm_define_native(vm, "print", noc_native_print);
    nc_vm_define_native(vm, "print_tensor", noc_native_print);
    nc_vm_define_native(vm, "read_file", noc_native_read_file);
    nc_vm_define_native(vm, "range", noc_native_range);
    
    // Model Info
    nc_vm_define_native(vm, "count_params", noc_native_count_params);
    nc_vm_define_native(vm, "tensor_info", noc_native_tensor_info);
    nc_vm_define_native(vm, "print_model", noc_native_print_model);
}

void nc_vm_free(nc_vm* vm) {
    // Free globals
    if (vm->globals.names) free(vm->globals.names);
    if (vm->globals.values) free(vm->globals.values);
    
    // Free tracked tensors carefully to avoid double-freeing duplicates
    // Pass 0: Count nodes
    size_t node_count = 0;
    nc_tensor_node* curr = (nc_tensor_node*)vm->tensors;
    while (curr) { node_count++; curr = curr->next; }

    if (node_count > 0) {
        nc_tensor** unique_tensors = (nc_tensor**)malloc(node_count * sizeof(nc_tensor*));
        size_t unique_count = 0;

        // Pass 1: Reset all reachable flags
        curr = (nc_tensor_node*)vm->tensors;
        while (curr) {
            nc_tensor_reset_reachable(curr->tensor);
            curr = curr->next;
        }

        // Pass 2: Collect unique tensors using SHALLOW marking
        // We do NOT use nc_tensor_mark_reachable which is recursive.
        curr = (nc_tensor_node*)vm->tensors;
        while (curr) {
            nc_tensor* t = curr->tensor;
            if (!t->is_reachable) {
                t->is_reachable = true; // Shallow mark
                if (unique_tensors) unique_tensors[unique_count++] = t;
            }
            curr = curr->next;
        }

        // Pass 2.5: Detach owned dependencies (grads, saved tensors) to prevent double-free
        // The unique_tensors array now contains ALL tensors that the VM will explicitly free in Pass 4.
        // We must ensure that freeing one tensor does not recursively free another tensor that is ALSO in this list.

        // Detach gradients and owned saved tensors if they are also in the list
        for (size_t i = 0; i < unique_count; i++) {
             nc_tensor* t = unique_tensors[i];
             
             // 1. Detach gradient
             if (t->grad && t->grad->is_reachable) {
                 // Gradient is tracked separately (is_reachable=true), so we must not free it recursively
                 // We kept t->grad in unique_tensors (if it was in the list), so it will be freed in Pass 4.
                 t->grad = NULL; 
             }
             
             // 2. Detach autograd owned tensors
             if (t->grad_fn) {
                 nc_node* node = t->grad_fn;
                 for (size_t j = 0; j < node->n_saved; j++) {
                     if (node->saved_tensors_owned[j] && node->saved_tensors[j] && node->saved_tensors[j]->is_reachable) {
                         // Owned tensor is tracked separately. Detach ownership.
                         node->saved_tensors_owned[j] = false;
                     }
                 }
             }
        }

        // Pass 3: Free nodes
        curr = (nc_tensor_node*)vm->tensors;
        while (curr) {
            nc_tensor_node* next = curr->next;
            free(curr);
            curr = next;
        }
        vm->tensors = NULL;

        // Pass 4: Free unique tensors
        if (unique_tensors) {
            for (size_t i = 0; i < unique_count; i++) {
                nc_tensor_free(unique_tensors[i]);
            }
            free(unique_tensors);
        }
    } else {
        vm->tensors = NULL;
    }
    
    // Free all objects
    nc_free_objects();
}

void nc_vm_push(nc_vm* vm, nc_value value) {
    *vm->stack_top = value;
    vm->stack_top++;
}

nc_value nc_vm_pop(nc_vm* vm) {
    vm->stack_top--;
    return *vm->stack_top;
}

static nc_value nc_vm_peek(nc_vm* vm, int distance) {
    return vm->stack_top[-1 - distance];
}

static void reset_stack(nc_vm* vm) {
    vm->stack_top = vm->stack;
    vm->frame_count = 0;
}

static bool call_value(nc_vm* vm, nc_value callee, int arg_count);

static void mark_object(nc_obj* obj); 

static void mark_value(nc_value value) {
    if (IS_OBJ(value)) mark_object(AS_OBJ(value));
    if (IS_TENSOR(value)) {
         nc_tensor_mark_reachable(AS_TENSOR(value));
    }
}

static void mark_object(nc_obj* object) {
    if (object == NULL || object->is_marked) return;
    object->is_marked = true;
    
    // Trace children (Black)
    if (object->type == OBJ_LIST) {
        nc_list* list = (nc_list*)object;
        for (int i = 0; i < list->count; i++) {
            mark_value(list->items[i]);
        }
    }
    // OBJ_FUNCTION -> constants
    else if (object->type == OBJ_FUNCTION) {
        nc_function* func = (nc_function*)object;
         // Mark constants in chunk 
         // Constants are `nc_value_array`.
         for(int i=0; i<func->chunk.constants.count; i++) {
             mark_value(func->chunk.constants.values[i]);
         }
         if (func->name) mark_object((nc_obj*)func->name);
    }
    else if (object->type == OBJ_INSTANCE) {
        nc_instance* instance = (nc_instance*)object;
        if (instance->klass) mark_object((nc_obj*)instance->klass);
        for (int i = 0; i < instance->field_count; i++) {
            if (instance->fields[i].key) mark_object((nc_obj*)instance->fields[i].key);
            mark_value(instance->fields[i].value);
        }
    }
    else if (object->type == OBJ_CLASS) {
        nc_class* klass = (nc_class*)object;
        if (klass->name) mark_object((nc_obj*)klass->name);
        for (int i = 0; i < klass->method_count; i++) {
             if (klass->methods[i].key) mark_object((nc_obj*)klass->methods[i].key);
             if (klass->methods[i].value) mark_object(klass->methods[i].value);
        }
    }
    else if (object->type == OBJ_BOUND_METHOD) {
        nc_bound_method* bound = (nc_bound_method*)object;
        mark_value(bound->receiver);
        if (bound->method) mark_object((nc_obj*)bound->method);
    }
    // TODO: Other types
}

// Helper to recursively mark tensor and its backward graph
// static void mark_tensor(nc_tensor* t) { ... } -- Removed, using tensor.c implementation

static void mark_roots(nc_vm* vm) {
    // Stack
    for (nc_value* slot = vm->stack; slot < vm->stack_top; slot++) {
        mark_value(*slot);
    }
    
    // Globals
    for (int i = 0; i < vm->globals.count; i++) {
        mark_value(vm->globals.values[i]);
        // Also mark names? They are const char* usually but if allocated strings?
        // Current implementation: `globals.names` is `const char**`.
        // They might be interned strings or raw pointers.
        // If they are `nc_string` objects, we should mark them if we can access them as objs.
        // But struct says `const char*`.
    }
    
    // Call Frames (functions on stack are already marked, but closures?)
    for (int i = 0; i < vm->frame_count; i++) {
        if (vm->frames[i].function) mark_object((nc_obj*)vm->frames[i].function);
    }
}

static void sweep_tensors(nc_vm* vm) {
    // Pass 1: Remove and free unreachable tensors
    nc_tensor_node** node = &vm->tensors;
    while (*node) {
        if (nc_tensor_is_reachable((*node)->tensor)) {
             // Keep it
             node = &(*node)->next;
        } else {
             // Unreachable
             nc_tensor_node* unreached = *node;
             *node = unreached->next;
             
             nc_tensor_free(unreached->tensor);
             free(unreached);
        }
    }

    // Pass 2: Reset reachable flag for next GC
    // We must reset ALL kept nodes. Even if duplicates exist,
    // they are all reachable (otherwise they would be removed in Pass 1).
    // Resetting multiple times is safe (idempotent false assignment).
    nc_tensor_node* curr = vm->tensors;
    while (curr) {
        nc_tensor_reset_reachable(curr->tensor);
        curr = curr->next;
    }
}

static void nc_gc_collect(nc_vm* vm) {
    mark_roots(vm);
    nc_sweep_objects(); // Objects also need similar logic? NO, objects don't have duplicates wrapper issue usually?
                       // Objects are allocated as Obj*. Wrapper is Value (struct). 
                       // No duplicate "Nodes" for objects. 
    sweep_tensors(vm);
}


// ... existing define_native ...


void nc_vm_define_native(nc_vm* vm, const char* name, nc_native_fn function) {
    if (vm->globals.capacity < vm->globals.count + 1) {
        int old_capacity = vm->globals.capacity;
        vm->globals.capacity = (old_capacity < 8) ? 8 : old_capacity * 2;
        vm->globals.names = (const char**)realloc(vm->globals.names, sizeof(const char*) * vm->globals.capacity);
        vm->globals.values = (nc_value*)realloc(vm->globals.values, sizeof(nc_value) * vm->globals.capacity);
    }
    
    // Simple deduplication check skipped for MVP
    vm->globals.names[vm->globals.count] = name; // Just store pointer, assuming static string or managed elsewhere
    vm->globals.values[vm->globals.count] = NATIVE_VAL(function);
    vm->globals.count++;
}

static bool call(nc_vm* vm, nc_function* function, int arg_count) {
    if (arg_count != function->arity) {
        runtime_error(vm, "Expected %d arguments but got %d.", function->arity, arg_count);
        return false;
    }
    
    if (vm->frame_count == FRAMES_MAX) { // Assuming FRAMES_MAX is defined
        runtime_error(vm, "Stack overflow.");
        return false;
    }
    
    nc_call_frame* frame = &vm->frames[vm->frame_count++];
    frame->function = function;
    frame->ip = function->chunk.code;
    frame->slots = vm->stack_top - arg_count - 1; // -1 for the function object itself?
    // Wait, in Lox passing convention: [func, arg1, arg2] -> slots points to func?
    // Usually local 0 is the function/receiver.
    // Yes, slots points to stack index where func is.
    
    return true;
}

static bool call_value(nc_vm* vm, nc_value callee, int arg_count) {
    if (IS_OBJ(callee)) {
        switch (OBJ_TYPE(callee)) {
            case OBJ_FUNCTION:
                return call(vm, AS_FUNCTION(callee), arg_count);
            case OBJ_CLASS: {
                nc_class* klass = AS_CLASS(callee);
                nc_instance* instance = nc_new_instance(klass);
                vm->stack_top[-arg_count - 1] = OBJ_VAL(instance);
                // Call constructor if we had one
                // Check for "init" method?
                // For MVP: No constructor
                return true;
            }
            case OBJ_BOUND_METHOD: {
                nc_bound_method* bound = AS_BOUND_METHOD(callee);
                // Replace bound method on stack with receiver (this)
                vm->stack_top[-arg_count - 1] = bound->receiver;
                return call(vm, bound->method, arg_count);
            }
            default:
                break;
        }
    }
    if (IS_NATIVE(callee)) {
        nc_native_fn native = AS_NATIVE(callee);
        nc_value result = native(vm, arg_count, vm->stack_top - arg_count);
        vm->stack_top -= (arg_count + 1);
        nc_vm_push(vm, result);
        return true;
    }
    runtime_error(vm, "Can only call functions and classes.");
    return false;
}

static void runtime_error(nc_vm* vm, const char* format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fputs("\n", stderr);
    
    // Calculate line number
    nc_call_frame* frame = &vm->frames[vm->frame_count - 1];
    size_t instruction = frame->ip - frame->function->chunk.code - 1;
    int line = frame->function->chunk.lines[instruction];
    fprintf(stderr, "[line %d] in script\n", line);
    
    // Reset stack
    reset_stack(vm);
}

static bool is_falsey(nc_value value) {
    return IS_NIL(value) || (IS_BOOL(value) && !AS_BOOL(value));
}

static nc_interpret_result run(nc_vm* vm) {
    nc_call_frame* frame = &vm->frames[vm->frame_count - 1];

#define READ_BYTE() (*frame->ip++)
#define READ_SHORT() (frame->ip += 2, (uint16_t)((frame->ip[-2] << 8) | frame->ip[-1]))
#define READ_CONSTANT() (frame->function->chunk.constants.values[READ_BYTE()])
#define READ_STRING() (NULL)

// Helper to look up global by index (names are stored as values in constant pool, effectively)
// But our simple compiler stored Numbers. 
// We need to upgrade compiler to store string names for globals.

// For now, let's assume the constant pool contains a proprietary "String Object" or just raw string pointer cast to void* if we are reckless,
// OR we just use `OP_GET_GLOBAL` with immediate string? No, that's variable length.
// Standard way: Constant pool has OBJ_STRING.
// MVP Hack: We don't have OBJ_STRING.
// We can modify `READ_CONSTANT` to assume the value *is* a key to look up?

// Actually, `OP_GET_GLOBAL` operand is an index into constant pool.
// The constant at that index should be the NAME of the variable.
// In `compiler.c`, we need to add strings to constant pool.
// But `value` is a union. We can add a temporary hack: `VAL_OBJ` where obj is `char*`.

#define BINARY_OP(value_type, op) \
    do { \
        if (!IS_NUMBER((vm->stack_top[-1])) || !IS_NUMBER((vm->stack_top[-2]))) { \
            runtime_error(vm, "Operands must be numbers."); \
            return INTERPRET_RUNTIME_ERROR; \
        } \
        double b = AS_NUMBER(nc_vm_pop(vm)); \
        double a = AS_NUMBER(nc_vm_pop(vm)); \
        nc_vm_push(vm, value_type(a op b)); \
    } while (false)

    for (;;) {
#ifdef DEBUG_TRACE_EXECUTION
        // Disassemble logic needs update for frames
#endif
        uint8_t instruction;
        switch (instruction = READ_BYTE()) {
            case OP_CONSTANT: {
                nc_value constant = READ_CONSTANT();
                nc_vm_push(vm, constant);
                break;
            }
            case OP_ADD: {
                nc_value bVal = nc_vm_peek(vm, 0);
                nc_value aVal = nc_vm_peek(vm, 1);
                
                if (IS_STRING(aVal) && IS_STRING(bVal)) {
                    nc_string* b = AS_STRING(bVal);
                    nc_string* a = AS_STRING(aVal);
                    
                    int length = a->length + b->length;
                    char* chars = (char*)malloc(length + 1);
                    memcpy(chars, a->chars, a->length);
                    memcpy(chars + a->length, b->chars, b->length);
                    chars[length] = '\0';
                    
                    nc_string* result = nc_take_string(chars, length);
                    nc_vm_pop(vm); // Pop b
                    nc_vm_pop(vm); // Pop a
                    nc_vm_push(vm, OBJ_VAL(result));
                } else if (IS_STRING(aVal)) {
                    // String + Any (Implicit convert) - Only Numbers support for MVP
                    if (IS_NUMBER(bVal)) {
                         nc_string* a = AS_STRING(aVal);
                         double b = AS_NUMBER(bVal);
                         char buf[64];
                         snprintf(buf, 64, "%.6g", b);
                         
                         int length = a->length + (int)strlen(buf);
                         char* chars = (char*)malloc(length + 1);
                         memcpy(chars, a->chars, a->length);
                         memcpy(chars + a->length, buf, strlen(buf));
                         chars[length] = '\0';
                         
                         nc_string* result = nc_take_string(chars, length);
                         nc_vm_pop(vm); nc_vm_pop(vm);
                         nc_vm_push(vm, OBJ_VAL(result));
                    } else {
                         runtime_error(vm, "Can only concatenate string with string or number.");
                         return INTERPRET_RUNTIME_ERROR;
                    }
                } else if (IS_STRING(bVal)) {
                    // Number + String
                    if (IS_NUMBER(aVal)) {
                         double a = AS_NUMBER(aVal);
                         nc_string* b = AS_STRING(bVal);
                         char buf[64];
                         snprintf(buf, 64, "%.6g", a);
                         
                         int length = (int)strlen(buf) + b->length;
                         char* chars = (char*)malloc(length + 1);
                         memcpy(chars, buf, strlen(buf));
                         memcpy(chars + strlen(buf), b->chars, b->length);
                         chars[length] = '\0';
                         
                         nc_string* result = nc_take_string(chars, length);
                         nc_vm_pop(vm); nc_vm_pop(vm);
                         nc_vm_push(vm, OBJ_VAL(result));
                    } else {
                         runtime_error(vm, "Can only concatenate string with string or number.");
                         return INTERPRET_RUNTIME_ERROR;
                    } 
                } else if (IS_NUMBER(aVal) && IS_NUMBER(bVal)) {
                    double b = AS_NUMBER(nc_vm_pop(vm));
                    double a = AS_NUMBER(nc_vm_pop(vm));
                    nc_vm_push(vm, NUMBER_VAL(a + b));
                } else if (IS_TENSOR(aVal) && IS_TENSOR(bVal)) {
                    nc_tensor* b = AS_TENSOR(bVal);
                    nc_tensor* a = AS_TENSOR(aVal);
                    nc_tensor* res = nc_add(a, b);
                    if (!res) return INTERPRET_RUNTIME_ERROR;
                    nc_value v = TENSOR_VAL(track_tensor(vm, res));
                    nc_vm_pop(vm); nc_vm_pop(vm);
                    nc_vm_push(vm, v);
                } else if (IS_TENSOR(aVal) && IS_NUMBER(bVal)) {
                    double b = AS_NUMBER(bVal);
                    nc_tensor* a = AS_TENSOR(aVal);
                    nc_tensor* res = nc_add_scalar(a, b);
                    if (!res) return INTERPRET_RUNTIME_ERROR;
                    nc_value v = TENSOR_VAL(track_tensor(vm, res));
                    nc_vm_pop(vm); nc_vm_pop(vm);
                    nc_vm_push(vm, v);
                } else if (IS_NUMBER(aVal) && IS_TENSOR(bVal)) {
                    // Commutative
                    nc_tensor* b = AS_TENSOR(bVal);
                    double a = AS_NUMBER(aVal);
                    nc_tensor* res = nc_add_scalar(b, a);
                    if (!res) return INTERPRET_RUNTIME_ERROR;
                    nc_value v = TENSOR_VAL(track_tensor(vm, res));
                    nc_vm_pop(vm); nc_vm_pop(vm);
                    nc_vm_push(vm, v);
                } else {
                    runtime_error(vm, "Operands must be two numbers, two strings, or involve tensors.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                break;
            }
            case OP_SUB: {
                nc_value bVal = nc_vm_peek(vm, 0);
                nc_value aVal = nc_vm_peek(vm, 1);
                
                if (IS_NUMBER(aVal) && IS_NUMBER(bVal)) {
                    double b = AS_NUMBER(nc_vm_pop(vm));
                    double a = AS_NUMBER(nc_vm_pop(vm));
                    nc_vm_push(vm, NUMBER_VAL(a - b));
                } else if (IS_TENSOR(aVal) && IS_TENSOR(bVal)) {
                    nc_tensor* b = AS_TENSOR(bVal);
                    nc_tensor* a = AS_TENSOR(aVal);
                    nc_tensor* res = nc_sub(a, b);
                    if (!res) return INTERPRET_RUNTIME_ERROR;
                    nc_value v = TENSOR_VAL(track_tensor(vm, res));
                    nc_vm_pop(vm); nc_vm_pop(vm);
                    nc_vm_push(vm, v);
                } else if (IS_TENSOR(aVal) && IS_NUMBER(bVal)) {
                    double b = AS_NUMBER(bVal);
                    nc_tensor* a = AS_TENSOR(aVal);
                    nc_tensor* res = nc_sub_scalar(a, b);
                    if (!res) return INTERPRET_RUNTIME_ERROR;
                    nc_value v = TENSOR_VAL(track_tensor(vm, res));
                    nc_vm_pop(vm); nc_vm_pop(vm);
                    nc_vm_push(vm, v);
                } else {
                    runtime_error(vm, "Operands must be two numbers or involve tensors (T-T, T-S).");
                    return INTERPRET_RUNTIME_ERROR;
                }
                break;
            }
            case OP_MUL: {
                nc_value bVal = nc_vm_peek(vm, 0);
                nc_value aVal = nc_vm_peek(vm, 1);
                
                if (IS_NUMBER(aVal) && IS_NUMBER(bVal)) {
                    double b = AS_NUMBER(nc_vm_pop(vm));
                    double a = AS_NUMBER(nc_vm_pop(vm));
                    nc_vm_push(vm, NUMBER_VAL(a * b));
                } else if (IS_TENSOR(aVal) && IS_TENSOR(bVal)) {
                    nc_tensor* b = AS_TENSOR(bVal);
                    nc_tensor* a = AS_TENSOR(aVal);
                    nc_tensor* res = nc_mul(a, b);
                    if (!res) return INTERPRET_RUNTIME_ERROR;
                    nc_value v = TENSOR_VAL(track_tensor(vm, res));
                    nc_vm_pop(vm); nc_vm_pop(vm);
                    nc_vm_push(vm, v);
                } else if (IS_TENSOR(aVal) && IS_NUMBER(bVal)) {
                    double b = AS_NUMBER(bVal);
                    nc_tensor* a = AS_TENSOR(aVal);
                    nc_tensor* res = nc_mul_scalar(a, b);
                    if (!res) return INTERPRET_RUNTIME_ERROR;
                    nc_value v = TENSOR_VAL(track_tensor(vm, res));
                    nc_vm_pop(vm); nc_vm_pop(vm);
                    nc_vm_push(vm, v);
                } else if (IS_NUMBER(aVal) && IS_TENSOR(bVal)) {
                    nc_tensor* b = AS_TENSOR(bVal);
                    double a = AS_NUMBER(aVal);
                    nc_tensor* res = nc_mul_scalar(b, a);
                    if (!res) return INTERPRET_RUNTIME_ERROR;
                    nc_value v = TENSOR_VAL(track_tensor(vm, res));
                    nc_vm_pop(vm); nc_vm_pop(vm);
                    nc_vm_push(vm, v);
                } else {
                    runtime_error(vm, "Operands must be two numbers or involve tensors.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                break;
            }
            case OP_DIV: {
                nc_value bVal = nc_vm_peek(vm, 0);
                nc_value aVal = nc_vm_peek(vm, 1);
                
                if (IS_NUMBER(aVal) && IS_NUMBER(bVal)) {
                    double b = AS_NUMBER(nc_vm_pop(vm));
                    double a = AS_NUMBER(nc_vm_pop(vm));
                    nc_vm_push(vm, NUMBER_VAL(a / b));
                } else if (IS_TENSOR(aVal) && IS_TENSOR(bVal)) {
                    nc_tensor* b = AS_TENSOR(bVal);
                    nc_tensor* a = AS_TENSOR(aVal);
                    nc_tensor* res = nc_div(a, b);
                    if (!res) return INTERPRET_RUNTIME_ERROR;
                    nc_value v = TENSOR_VAL(track_tensor(vm, res));
                    nc_vm_pop(vm); nc_vm_pop(vm);
                    nc_vm_push(vm, v);
                } else if (IS_TENSOR(aVal) && IS_NUMBER(bVal)) {
                    double b = AS_NUMBER(bVal);
                    nc_tensor* a = AS_TENSOR(aVal);
                    nc_tensor* res = nc_div_scalar(a, b);
                    if (!res) return INTERPRET_RUNTIME_ERROR;
                    nc_value v = TENSOR_VAL(track_tensor(vm, res));
                    nc_vm_pop(vm); nc_vm_pop(vm);
                    nc_vm_push(vm, v);
                } else {
                    runtime_error(vm, "Operands must be two numbers or involve tensors (T-T, T-S).");
                    return INTERPRET_RUNTIME_ERROR;
                }
                break;
            }
            case OP_MOD: {
                nc_value bVal = nc_vm_peek(vm, 0);
                nc_value aVal = nc_vm_peek(vm, 1);
                
                if (IS_NUMBER(aVal) && IS_NUMBER(bVal)) {
                    int b = (int)AS_NUMBER(nc_vm_pop(vm));
                    int a = (int)AS_NUMBER(nc_vm_pop(vm));
                    if (b == 0) {
                        runtime_error(vm, "Modulo by zero.");
                        return INTERPRET_RUNTIME_ERROR;
                    }
                    nc_vm_push(vm, NUMBER_VAL((double)(a % b)));
                } else {
                    runtime_error(vm, "Modulo operands must be numbers.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                break;
            }
            
            case OP_NEGATE: {
                if (!IS_NUMBER((vm->stack_top[-1]))) {
                    runtime_error(vm, "Operand must be a number.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                vm->stack_top[-1] = NUMBER_VAL(-AS_NUMBER(vm->stack_top[-1]));
                break;
            }
            
            case OP_GET_GLOBAL: {
                // Operand is index of variable name in constant pool
                nc_value nameVal = READ_CONSTANT();
                
                // nameVal should be an OBJ_STRING
                if (!IS_STRING(nameVal)) {
                     runtime_error(vm, "Variable name must be a string.");
                     return INTERPRET_RUNTIME_ERROR;
                }
                const char* name = AS_CSTRING(nameVal);
 
                
                // Look up
                bool found = false;
                for (int i = 0; i < vm->globals.count; i++) {
                     if (strcmp(vm->globals.names[i], name) == 0) {
                         nc_vm_push(vm, vm->globals.values[i]);
                         found = true;
                         break;
                     }
                }
                
                if (!found) {
                    runtime_error(vm, "Undefined variable '%s'.", name);
                    return INTERPRET_RUNTIME_ERROR;
                }
                break;
            }
            
            case OP_CALL: {
                int arg_count = READ_BYTE();
                // Callee is at stack_top - arg_count - 1
                nc_value callee = nc_vm_peek(vm, arg_count);
                
                if (!call_value(vm, callee, arg_count)) {
                    return INTERPRET_RUNTIME_ERROR;
                }
                frame = &vm->frames[vm->frame_count - 1]; // Update frame cache
                break;
            }

            case OP_RETURN: {
                nc_value result = nc_vm_pop(vm);
                vm->frame_count--;
                if (vm->frame_count == 0) {
                     nc_vm_pop(vm); // Pop main function
                     return INTERPRET_OK;
                }
                
                vm->stack_top = frame->slots;
                nc_vm_push(vm, result);
                frame = &vm->frames[vm->frame_count - 1];
                break;
            }
            
            case OP_POP: {
                nc_vm_pop(vm);
                break; 
            }

            case OP_DUP: {
                nc_vm_push(vm, nc_vm_peek(vm, 0));
                break;
            }
            
            case OP_NIL: nc_vm_push(vm, NIL_VAL); break;
            case OP_TRUE: nc_vm_push(vm, BOOL_VAL(true)); break;
            case OP_FALSE: nc_vm_push(vm, BOOL_VAL(false)); break;
            
            case OP_DEFINE_GLOBAL: {
                nc_value nameVal = READ_CONSTANT();
                const char* name = AS_CSTRING(nameVal); // Ensure it is string

                nc_value val = nc_vm_peek(vm, 0); // Keep on stack until pop? Or pop?
                // compile_var_decl emits OP_DEFINE_GLOBAL then does it pop?
                // In generic compiler logic usually: var declaration consumes the value.
                // So we should POP.
                // But my compiler does NOT emit POP after define_global usually.
                // Let's check lox: OP_DEFINE_GLOBAL reads value from top of stack.
                // Does it pop? Yes.
                nc_vm_pop(vm);
                
                // Add to globals
                // Simple append/overwrite logic
                // Check if exists?
                bool exists = false;
                for (int i = 0; i < vm->globals.count; i++) {
                    if (strcmp(vm->globals.names[i], name) == 0) {
                        vm->globals.values[i] = val;
                        exists = true;
                        break;
                    }
                }
                
                if (!exists) {
                    if (vm->globals.capacity < vm->globals.count + 1) {
                         int old_capacity = vm->globals.capacity;
                         vm->globals.capacity = (old_capacity < 8) ? 8 : old_capacity * 2;
                         vm->globals.names = (const char**)realloc(vm->globals.names, sizeof(const char*) * vm->globals.capacity);
                         vm->globals.values = (nc_value*)realloc(vm->globals.values, sizeof(nc_value) * vm->globals.capacity);
                    }
                    vm->globals.names[vm->globals.count] = name;
                    vm->globals.values[vm->globals.count] = val;
                    vm->globals.count++;
                }
                break;
            }

            case OP_SET_GLOBAL: {
                nc_value nameVal = READ_CONSTANT();
                const char* name = AS_CSTRING(nameVal);

                nc_value val = nc_vm_peek(vm, 0); // Assignment expression result involves value
                
                bool found = false;
                for (int i = 0; i < vm->globals.count; i++) {
                    if (strcmp(vm->globals.names[i], name) == 0) {
                        vm->globals.values[i] = val;
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    runtime_error(vm, "Undefined variable '%s'.", name);
                    return INTERPRET_RUNTIME_ERROR;
                }
                // Result of assignment is the value, so we leave it on stack
                break;
            }
            
            case OP_GET_LOCAL: {
                uint8_t slot = READ_BYTE();
                nc_vm_push(vm, frame->slots[slot]);
                break;
            }
            
            case OP_SET_LOCAL: {
                uint8_t slot = READ_BYTE();
                frame->slots[slot] = nc_vm_peek(vm, 0);
                break;
            }
            
            case OP_EQUAL: {
                nc_value b = nc_vm_pop(vm);
                nc_value a = nc_vm_pop(vm);
                nc_vm_push(vm, BOOL_VAL(nc_values_equal(a, b)));
                break;
            }
            
            case OP_GREATER: BINARY_OP(BOOL_VAL, >); break;
            case OP_LESS:    BINARY_OP(BOOL_VAL, <); break;
            
            case OP_NOT: {
                 nc_vm_push(vm, BOOL_VAL(is_falsey(nc_vm_pop(vm))));
                 break;
            }
            
            case OP_JUMP: {
                uint16_t offset = READ_SHORT();
                frame->ip += offset;
                break;
            }
            
            case OP_JUMP_IF_FALSE: {
                uint16_t offset = READ_SHORT();
                if (is_falsey(nc_vm_peek(vm, 0))) {
                    frame->ip += offset;
                }
                break;
            }
            
            case OP_LOOP: {
                uint16_t offset = READ_SHORT();
                frame->ip -= offset;
                break;
            }

            case OP_FOR_ITER: {
                uint16_t exit_offset = READ_SHORT();
                nc_value iterator = nc_vm_peek(vm, 0); // Correct peek distance for iterator
                
                // For now, only support range iterators
                if (IS_RANGE(iterator)) {
                    nc_range* range = AS_RANGE(iterator);
                    if (range->current < range->end) {
                        nc_vm_push(vm, NUMBER_VAL(range->current));
                        range->current++;
                    } else {
                        // Exit loop
                        frame->ip += exit_offset;
                    }
                } else {
                     runtime_error(vm, "Object is not iterable (only range supported).");
                     return INTERPRET_RUNTIME_ERROR;
                }
                break;
            }

            case OP_CLASS: {
                nc_value nameVal = READ_CONSTANT();
                nc_string* name = AS_STRING(nameVal);
                nc_class* klass = nc_new_class(name);
                nc_vm_push(vm, OBJ_VAL(klass));
                break;
            }

            case OP_GET_PROPERTY: {
                nc_value nameVal = READ_CONSTANT();
                nc_string* name = AS_STRING(nameVal);
                nc_value obj = nc_vm_peek(vm, 0);

                if (IS_TENSOR(obj)) {
                    if (get_tensor_property(vm, AS_TENSOR(obj), name)) {
                        nc_value result = nc_vm_pop(vm);
                        nc_vm_pop(vm); // Pop Tensor
                        nc_vm_push(vm, result);
                        break;
                    }
                    runtime_error(vm, "Undefined property '%s' on tensor.", name->chars);
                    return INTERPRET_RUNTIME_ERROR;
                } else if (!IS_INSTANCE(obj)) {
                    runtime_error(vm, "Only instances and tensors have properties.");
                    return INTERPRET_RUNTIME_ERROR;
                }

                nc_instance* instance = AS_INSTANCE(obj);
                nc_value value;
                if (nc_instance_get_field(instance, name, &value)) {
                    nc_vm_pop(vm); // Instance
                    nc_vm_push(vm, value);
                    break;
                }
                
                // Check methods
                nc_class* klass = instance->klass;
                // Linear scan of methods in class
                // MVP: Linear scan
                for (int i = 0; i < klass->method_count; i++) {
                     if (klass->methods[i].key->hash == name->hash &&
                        klass->methods[i].key->length == name->length &&
                        memcmp(klass->methods[i].key->chars, name->chars, name->length) == 0) {
                         
                        nc_function* method = (nc_function*)klass->methods[i].value;
                        nc_bound_method* bound = nc_new_bound_method(obj, method);
                        nc_vm_pop(vm); // Instance
                        nc_vm_push(vm, OBJ_VAL(bound));
                        goto method_found_break_switch; // Use flag or goto to break switch
                     }
                }
                
                runtime_error(vm, "Undefined property '%s'.", name->chars);
                return INTERPRET_RUNTIME_ERROR;
                
                method_found_break_switch:
                break;
            }

            case OP_SET_PROPERTY: {
                nc_value nameVal = READ_CONSTANT();
                nc_string* name = AS_STRING(nameVal);
                nc_value value = nc_vm_pop(vm);
                nc_value obj = nc_vm_peek(vm, 0); // obj is below value

                if (IS_TENSOR(obj)) {
                    if (set_tensor_property(vm, AS_TENSOR(obj), name, value)) {
                        nc_vm_pop(vm); // Pop tensor (value is left on stack as result?)
                        // Stack: [obj, value].
                        // We popped value earlier: nc_value value = nc_vm_pop(vm).
                        // Stack now: [obj] (peek 0).
                        // We need to pop obj. And push value.
                        nc_vm_pop(vm); 
                        nc_vm_push(vm, value);
                        break;
                    }
                    runtime_error(vm, "Undefined property '%s' on tensor.", name->chars);
                    return INTERPRET_RUNTIME_ERROR;
                }

                if (!IS_INSTANCE(obj)) {
                    runtime_error(vm, "Only instances and tensors have properties.");
                    return INTERPRET_RUNTIME_ERROR;
                }

                nc_instance* instance = AS_INSTANCE(obj);
                nc_instance_set_field(instance, name, value);

                // Value is result of assignment
                nc_vm_pop(vm); // Pop Instance
                nc_vm_push(vm, value); // Push Value back
                break;
            }
            
            case OP_METHOD: {
                nc_value nameVal = READ_CONSTANT();
                nc_string* name = AS_STRING(nameVal);
                nc_value methodVal = nc_vm_peek(vm, 0);
                nc_value classVal = nc_vm_peek(vm, 1); // Class is below method
                
                if (!IS_CLASS(classVal)) {
                    runtime_error(vm, "Internal error: OP_METHOD called without class on stack.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                
                nc_class* klass = AS_CLASS(classVal);
                nc_class_add_method(klass, name, AS_FUNCTION(methodVal));
                
                nc_vm_pop(vm); // Pop method closure
                break;
            }
            
            case OP_BUILD_LIST: {
                int count = READ_BYTE();
                nc_list* list = nc_new_list();
                // Items are on stack: [item1, item2, ... itemN] (top is itemN)
                // We want to add them in order.
                // Stack top is at list->items + (count - 1)?
                // No, stack grows up.
                // item1 is deep, itemN is top.
                
                // Pre-allocate
                if (count > 0) {
                   list->capacity = count;
                   list->items = (nc_value*)malloc(sizeof(nc_value) * count);
                }
                
                // Fill in reverse from stack?
                // If stack is [item1, item2], popping gives item2 then item1.
                // So looping items[count - 1 - i] = pop()
                
                for (int i = count - 1; i >= 0; i--) {
                    list->items[i] = nc_vm_pop(vm);
                }
                list->count = count;
                
                nc_vm_push(vm, OBJ_VAL(list));
                break;
            }
            
            case OP_GET_INDEX: {
                nc_value indexVal = nc_vm_pop(vm);
                nc_value listVal = nc_vm_pop(vm);
                
                if (!IS_LIST(listVal)) {
                    runtime_error(vm, "Can only index into lists.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                if (!IS_NUMBER(indexVal)) {
                    runtime_error(vm, "List index must be a number.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                
                nc_list* list = AS_LIST(listVal);
                int index = (int)AS_NUMBER(indexVal);
                
                if (index < 0 || index >= list->count) {
                    runtime_error(vm, "List index out of bounds.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                
                nc_vm_push(vm, list->items[index]);
                break;
            }
            
            case OP_SET_INDEX: {
                nc_value value = nc_vm_pop(vm);
                nc_value indexVal = nc_vm_pop(vm);
                nc_value listVal = nc_vm_pop(vm);
                
                if (!IS_LIST(listVal)) {
                    runtime_error(vm, "Can only index into lists.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                if (!IS_NUMBER(indexVal)) {
                    runtime_error(vm, "List index must be a number.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                
                nc_list* list = AS_LIST(listVal);
                int index = (int)AS_NUMBER(indexVal);
                
                if (index < 0 || index >= list->count) {
                    runtime_error(vm, "List index out of bounds.");
                    return INTERPRET_RUNTIME_ERROR;
                }
                
                list->items[index] = value;
                nc_vm_push(vm, value); // Assign expression result is value
                break;
            }
            
            default: {
                runtime_error(vm, "Unknown opcode %d", instruction);
                return INTERPRET_RUNTIME_ERROR;
            }
        }
    }
#undef READ_BYTE
#undef READ_SHORT
#undef READ_CONSTANT
#undef BINARY_OP
}

nc_interpret_result nc_vm_interpret(nc_vm* vm, nc_chunk* chunk) {
    // For MVP, wrap the chunk in a "script" function and call it
    nc_function* script = nc_new_function();
    // Copy chunk or move?
    // nc_new_function inits empty chunk.
    // We are passing an existing chunk.
    // Swap?
    // nc_chunk temp = script->chunk;
    // script->chunk = *chunk;
    // *chunk = temp; // garbage
    
    // Better: Interpret takes a function now?
    // Or we just push a frame manually for the script.
    
    // Hack for transition:
    // We assume 'chunk' passed in is the top-level script code.
    // We will just create a "main" function that owns this code?
    // Or just set up the first frame to point to this chunk.
    
    // Ideally user compiles to a Function (Script).
    // But our API `nc_compile` emits to a `nc_chunk`.
    
    // Let's make a dummy function that points to this chunk.
    // Note: chunk lifetime must be managed.
    
    // Actually, let's just use the chunk directly for the first frame 
    // but we need a function pointer forREAD_CONSTANT to work if we access constants via function.
    
    // So we MUST have a function object.
    script->chunk = *chunk; // Shallow copy structure (pointers to code/constants)
    script->name = NULL;
    script->arity = 0; // Top-level script has no arguments
    
    nc_vm_push(vm, OBJ_VAL(script));
    call(vm, script, 0);
    
    return run(vm);
}

