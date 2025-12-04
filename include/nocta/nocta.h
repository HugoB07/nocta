#ifndef NOCTA_H
#define NOCTA_H

// Nocta - A lightweight deep learning library in C
// Version 0.2.0

// ============================================
// Core
// ============================================
#include "nocta/core/dtype.h"
#include "nocta/core/error.h"
#include "nocta/core/memory.h"
#include "nocta/core/tensor.h"
#include "nocta/core/serialize.h"

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
#define NOCTA_VERSION_MINOR 2
#define NOCTA_VERSION_PATCH 0
#define NOCTA_VERSION "0.2.0"

// ============================================
// Global initialization / cleanup
// ============================================

static inline void nc_init(void) {
    // Future: initialize RNG, thread pool, etc.
}

static inline void nc_cleanup(void) {
    nc_memory_report();
}

#endif // NOCTA_H