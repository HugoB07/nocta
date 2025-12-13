#ifndef NOCTA_NN_DROPOUT_H
#define NOCTA_NN_DROPOUT_H

#include "nocta/nn/module.h"
#include "nocta/core/tensor.h"

// Dropout operation (functional)
// x: input tensor
// p: probability of an element to be zeroed (0 <= p <= 1)
// training: if true, apply dropout; if false, return x (or clone)
nc_tensor* nc_dropout_fn(nc_tensor* x, double p, bool training);

// Create Dropout module
// p: probability of an element to be zeroed
nc_module* nc_dropout(double p);

#endif // NOCTA_NN_DROPOUT_H
