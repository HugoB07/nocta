#ifndef NOCTA_H
#define NOCTA_H

// Nocta - A lightweight deep learning library in C
// Version 0.5.0

// ============================================
// Core
// ============================================
#include "nocta/core/dtype.h"
#include "nocta/core/error.h"
#include "nocta/core/memory.h"
#include "nocta/core/tensor.h"
#include "nocta/core/serialize.h"
#include "nocta/core/device.h"

// ============================================
// Autograd
// ============================================
#include "nocta/autograd/node.h"
#include "nocta/autograd/backward.h"

// ============================================
// Operations
// ============================================
#include "nocta/ops/arithmetic.h"
#include "nocta/ops/matmul.h"
#include "nocta/ops/activation.h"
#include "nocta/ops/reduction.h"
#include "nocta/ops/loss.h"

// ============================================
// Neural Network Modules
// ============================================
#include "nocta/nn/module.h"
#include "nocta/nn/linear.h"
#include "nocta/nn/conv.h"
#include "nocta/nn/batchnorm.h"
#include "nocta/nn/dropout.h"

// ============================================
// Optimizers
// ============================================
#include "nocta/optim/optimizer.h"
#include "nocta/optim/sgd.h"
#include "nocta/optim/adam.h"

// ============================================
// Version info
// ============================================
#define NOCTA_VERSION_MAJOR 0
#define NOCTA_VERSION_MINOR 5
#define NOCTA_VERSION_PATCH 0
#define NOCTA_VERSION "0.5.0"

// ============================================
// Global initialization / cleanup
// ============================================

static inline void nc_init(void) {
    // Future: initialize RNG, thread pool, etc.
#ifdef NOCTA_CUDA_ENABLED
    if (nc_cuda_available()) {
        nc_cuda_init();
    }
#endif
}

static inline void nc_cleanup(void) {
#ifdef NOCTA_CUDA_ENABLED
    nc_cuda_cleanup();
#endif
    nc_memory_report();
}

#endif // NOCTA_H