#ifndef NOCTA_REDUCTION_H
#define NOCTA_REDUCTION_H

#include "nocta/core/tensor.h"

// ============================================
// Reduction operations
// ============================================

// Sum all elements -> scalar
nc_tensor* nc_sum_all(nc_tensor* x);

// Sum along axis
// keepdim: if true, keep reduced dimension as size 1
nc_tensor* nc_sum(nc_tensor* x, int axis, bool keepdim);

// Sum along multiple axes
nc_tensor* nc_sum_axes(nc_tensor* x, const int* axes, size_t n_axes, bool keepdim);

// Mean all elements -> scalar
nc_tensor* nc_mean_all(nc_tensor* x);

// Mean along axis
nc_tensor* nc_mean(nc_tensor* x, int axis, bool keepdim);

// Variance and standard deviation
nc_tensor* nc_var(nc_tensor* x, int axis, bool keepdim, bool unbiased);
nc_tensor* nc_std(nc_tensor* x, int axis, bool keepdim, bool unbiased);

// Max/Min
nc_tensor* nc_max_all(nc_tensor* x);
nc_tensor* nc_min_all(nc_tensor* x);

nc_tensor* nc_max(nc_tensor* x, int axis, bool keepdim);
nc_tensor* nc_min(nc_tensor* x, int axis, bool keepdim);

// Argmax/Argmin (returns indices)
nc_tensor* nc_argmax(nc_tensor* x, int axis, bool keepdim);
nc_tensor* nc_argmin(nc_tensor* x, int axis, bool keepdim);

// Product
nc_tensor* nc_prod_all(nc_tensor* x);
nc_tensor* nc_prod(nc_tensor* x, int axis, bool keepdim);

// Norms
nc_tensor* nc_norm(nc_tensor* x, double p);  // L-p norm of all elements
nc_tensor* nc_norm_axis(nc_tensor* x, double p, int axis, bool keepdim);

// Any/All (boolean tensors)
nc_tensor* nc_any(nc_tensor* x, int axis, bool keepdim);
nc_tensor* nc_all(nc_tensor* x, int axis, bool keepdim);

// Count non-zero
nc_tensor* nc_count_nonzero(nc_tensor* x, int axis, bool keepdim);

// Cumulative operations
nc_tensor* nc_cumsum(nc_tensor* x, int axis);
nc_tensor* nc_cumprod(nc_tensor* x, int axis);

// Logsumexp (numerically stable)
nc_tensor* nc_logsumexp(nc_tensor* x, int axis, bool keepdim);

#endif // NOCTA_REDUCTION_H