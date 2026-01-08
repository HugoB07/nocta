#ifndef NOCTA_MATMUL_H
#define NOCTA_MATMUL_H

#include "nocta/core/tensor.h"

// Matrix multiplication: C = A @ B
// Supports:
// - 2D: (M, K) @ (K, N) -> (M, N)
// - Batched: (..., M, K) @ (..., K, N) -> (..., M, N)
// - Matrix-vector: (M, K) @ (K,) -> (M,)
// - Vector-matrix: (K,) @ (K, N) -> (N,)
nc_tensor* nc_matmul(nc_tensor* a, nc_tensor* b);

// Batched matrix multiply (explicit batch dims)
nc_tensor* nc_bmm(nc_tensor* a, nc_tensor* b);

// Vector dot product
nc_tensor* nc_dot(nc_tensor* a, nc_tensor* b);

// Outer product: a ⊗ b
nc_tensor* nc_outer(nc_tensor* a, nc_tensor* b);

// Matrix-vector multiply: A @ x
nc_tensor* nc_mv(nc_tensor* mat, nc_tensor* vec);

// Vector-matrix multiply: x @ A
nc_tensor* nc_vec_mat(nc_tensor* vec, nc_tensor* mat);

// Add matrix multiply: C = alpha * A @ B + beta * C
nc_tensor* nc_addmm(nc_tensor* c, nc_tensor* a, nc_tensor* b, 
                    double alpha, double beta);

// Inner product (generalized dot)
nc_tensor* nc_inner(nc_tensor* a, nc_tensor* b);

// Einsum (basic patterns)
// Supports: "ij,jk->ik", "bij,bjk->bik", "i,i->", etc.
nc_tensor* nc_einsum(const char* equation, nc_tensor** tensors, size_t n_tensors);

#endif // NOCTA_MATMUL_H