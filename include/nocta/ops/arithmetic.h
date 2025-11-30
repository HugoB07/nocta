#ifndef NOCTA_ARITHMETIC_H
#define NOCTA_ARITHMETIC_H

#include "nocta/core/tensor.h"

// ============================================
// Binary operations (element-wise)
// ============================================

// Addition: a + b
nc_tensor* nc_add(nc_tensor* a, nc_tensor* b);

// Subtraction: a - b
nc_tensor* nc_sub(nc_tensor* a, nc_tensor* b);

// Multiplication: a * b (element-wise)
nc_tensor* nc_mul(nc_tensor* a, nc_tensor* b);

// Division: a / b
nc_tensor* nc_div(nc_tensor* a, nc_tensor* b);

// Power: a ^ b (element-wise)
nc_tensor* nc_pow(nc_tensor* a, nc_tensor* b);

// Maximum: max(a, b) element-wise
nc_tensor* nc_maximum(nc_tensor* a, nc_tensor* b);

// Minimum: min(a, b) element-wise
nc_tensor* nc_minimum(nc_tensor* a, nc_tensor* b);

// ============================================
// Scalar operations
// ============================================

nc_tensor* nc_add_scalar(nc_tensor* a, double s);
nc_tensor* nc_sub_scalar(nc_tensor* a, double s);
nc_tensor* nc_mul_scalar(nc_tensor* a, double s);
nc_tensor* nc_div_scalar(nc_tensor* a, double s);
nc_tensor* nc_pow_scalar(nc_tensor* a, double s);

// Scalar on left: s - a, s / a
nc_tensor* nc_rsub_scalar(double s, nc_tensor* a);
nc_tensor* nc_rdiv_scalar(double s, nc_tensor* a);

// ============================================
// Unary operations
// ============================================

// Negation: -a
nc_tensor* nc_neg(nc_tensor* a);

// Absolute value: |a|
nc_tensor* nc_abs(nc_tensor* a);

// Square: a^2
nc_tensor* nc_square(nc_tensor* a);

// Square root: sqrt(a)
nc_tensor* nc_sqrt(nc_tensor* a);

// Exponential: e^a
nc_tensor* nc_exp(nc_tensor* a);

// Natural log: ln(a)
nc_tensor* nc_log(nc_tensor* a);

// Log base 10
nc_tensor* nc_log10(nc_tensor* a);

// Log base 2
nc_tensor* nc_log2(nc_tensor* a);

// Trigonometric
nc_tensor* nc_sin(nc_tensor* a);
nc_tensor* nc_cos(nc_tensor* a);
nc_tensor* nc_tan(nc_tensor* a);

// Inverse trig
nc_tensor* nc_asin(nc_tensor* a);
nc_tensor* nc_acos(nc_tensor* a);
nc_tensor* nc_atan(nc_tensor* a);

// Hyperbolic
nc_tensor* nc_sinh(nc_tensor* a);
nc_tensor* nc_cosh(nc_tensor* a);

// Sign: -1, 0, or 1
nc_tensor* nc_sign(nc_tensor* a);

// Floor / Ceil / Round
nc_tensor* nc_floor(nc_tensor* a);
nc_tensor* nc_ceil(nc_tensor* a);
nc_tensor* nc_round(nc_tensor* a);

// Clamp values to [min, max]
nc_tensor* nc_clamp(nc_tensor* a, double min_val, double max_val);

// Reciprocal: 1/a
nc_tensor* nc_reciprocal(nc_tensor* a);

// ============================================
// In-place operations (with underscore suffix)
// ============================================

void nc_add_(nc_tensor* a, nc_tensor* b);
void nc_sub_(nc_tensor* a, nc_tensor* b);
void nc_mul_(nc_tensor* a, nc_tensor* b);
void nc_div_(nc_tensor* a, nc_tensor* b);

void nc_add_scalar_(nc_tensor* a, double s);
void nc_sub_scalar_(nc_tensor* a, double s);
void nc_mul_scalar_(nc_tensor* a, double s);
void nc_div_scalar_(nc_tensor* a, double s);

void nc_neg_(nc_tensor* a);
void nc_abs_(nc_tensor* a);
void nc_exp_(nc_tensor* a);
void nc_log_(nc_tensor* a);
void nc_sqrt_(nc_tensor* a);
void nc_clamp_(nc_tensor* a, double min_val, double max_val);

// ============================================
// Comparison operations (return bool tensor)
// ============================================

nc_tensor* nc_eq(nc_tensor* a, nc_tensor* b);   // a == b
nc_tensor* nc_ne(nc_tensor* a, nc_tensor* b);   // a != b
nc_tensor* nc_lt(nc_tensor* a, nc_tensor* b);   // a < b
nc_tensor* nc_le(nc_tensor* a, nc_tensor* b);   // a <= b
nc_tensor* nc_gt(nc_tensor* a, nc_tensor* b);   // a > b
nc_tensor* nc_ge(nc_tensor* a, nc_tensor* b);   // a >= b

nc_tensor* nc_eq_scalar(nc_tensor* a, double s);
nc_tensor* nc_ne_scalar(nc_tensor* a, double s);
nc_tensor* nc_lt_scalar(nc_tensor* a, double s);
nc_tensor* nc_le_scalar(nc_tensor* a, double s);
nc_tensor* nc_gt_scalar(nc_tensor* a, double s);
nc_tensor* nc_ge_scalar(nc_tensor* a, double s);

// ============================================
// Logical operations
// ============================================

nc_tensor* nc_logical_and(nc_tensor* a, nc_tensor* b);
nc_tensor* nc_logical_or(nc_tensor* a, nc_tensor* b);
nc_tensor* nc_logical_not(nc_tensor* a);
nc_tensor* nc_logical_xor(nc_tensor* a, nc_tensor* b);

// Where: condition ? x : y (element-wise)
nc_tensor* nc_where(nc_tensor* condition, nc_tensor* x, nc_tensor* y);

#endif // NOCTA_ARITHMETIC_H