#ifndef NOCTA_TENSOR_H
#define NOCTA_TENSOR_H

#include "dtype.h"
#include "memory.h"
#include "error.h"
#include <stddef.h>
#include <stdbool.h>

#define NC_MAX_DIMS 8

// Forward declarations
typedef struct nc_node nc_node;

// Tensor structure 
typedef struct nc_tensor {
    // Data
    nc_storage* storage; // Shared storage (ref counted)
    size_t offset;    // Offset in storage

    // Shape info
    size_t ndim;          // Number of dimensions
    size_t shape[NC_MAX_DIMS]; // Shape
    size_t strides[NC_MAX_DIMS]; // Strides in elements
    size_t numel; 

    // Type
    nc_dtype dtype;

    // Autograd
    bool requires_grad; // Track gradients?
    struct nc_tensor* grad; // Gradient tensor
    nc_node* grad_fn; // Backward function node
    bool is_leaf; // Leaf node in graph?

    // Flags
    bool is_contiguous;
    bool is_reachable;
} nc_tensor;

// ============================================
// Creation
// ============================================

// Create empty tensor with shape
nc_tensor* nc_tensor_empty(const size_t* shape, size_t ndim, nc_dtype dtype);

// Create tensor filled with zeros
nc_tensor* nc_tensor_zeros(const size_t* shape, size_t ndim, nc_dtype dtype);

// Create tensor filled with ones
nc_tensor* nc_tensor_ones(const size_t* shape, size_t ndim, nc_dtype dtype);

// Create tensor filled with value
nc_tensor* nc_tensor_full(const size_t* shape, size_t ndim, nc_dtype dtype, double value);

// Create tensor from data (copies data)
nc_tensor* nc_tensor_from_data(const void* data, const size_t* shape, size_t ndim, nc_dtype dtype);

// Create scalar tensor
nc_tensor* nc_tensor_scalar(double value, nc_dtype dtype);

// Create 1D tensor (vector)
nc_tensor* nc_tensor_arange(double start, double end, double step, nc_dtype dtype);

// Create linearly spaced tensor
nc_tensor* nc_tensor_linspace(double start, double end, size_t num, nc_dtype dtype);

// Random tensors
nc_tensor* nc_tensor_rand(const size_t* shape, size_t ndim, nc_dtype dtype);    // Uniform [0, 1]
nc_tensor* nc_tensor_randn(const size_t* shape, size_t ndim, nc_dtype dtype);   // Normal (0, 1)
nc_tensor* nc_tensor_randint(const size_t* shape, size_t ndim, int64_t low, int64_t high);

// Identify matrix
nc_tensor* nc_tensor_eye(size_t n, nc_dtype dtype);

// Clone tensor (deep copy)
nc_tensor* nc_tensor_clone(const nc_tensor* r);

// Free tensor
void nc_tensor_free(nc_tensor* t);

// ============================================
// Properties
// ============================================

// Get total number of elements
static inline size_t nc_tensor_numel(const nc_tensor* t) {
    return t ? t->numel : 0;
}

// Get number of dimensions
static inline size_t nc_tensor_ndim(const nc_tensor* t) {
    return t ? t->ndim : 0;
}

// Get shape at dimension
static inline size_t nc_tensor_shape(const nc_tensor* t, size_t dim) {
    return (t && dim < t->ndim) ? t->shape[dim] : 0;
}

// Get stride at dimension
static inline size_t nc_tensor_stride(const nc_tensor* t, size_t dim) {
    return (t && dim < t->ndim) ? t->strides[dim] : 0;
}

// Get data type
static inline nc_dtype nc_tensor_dtype(const nc_tensor* t) {
    return t ? t->dtype : NC_F32;
}

// Check if contiguous
static inline bool nc_tensor_is_contiguous(const nc_tensor* t) {
    return t ? t->is_contiguous : false;
}

// Get raw data pointer
static inline void* nc_tensor_data(const nc_tensor* t) {
    if (!t || !t->storage) return NULL;
    return (char*)t->storage->data + t->offset * nc_dtype_sizeof(t->dtype);
}

// Get typed data pointer
#define nc_tensor_data_f32(t) ((float*)nc_tensor_data(t))
#define nc_tensor_data_f64(t) ((double*)nc_tensor_data(t))
#define nc_tensor_data_i32(t) ((int32_t*)nc_tensor_data(t))
#define nc_tensor_data_i64(t) ((int64_t*)nc_tensor_data(t))

// Get total size in bytes
static inline size_t nc_tensor_nbytes(const nc_tensor* t) {
    return t ? t->numel * nc_dtype_sizeof(t->dtype) : 0;
}

// ============================================
// Indexing & Access
// ============================================

// Get element at flat index
double nc_tensor_get_flat(const nc_tensor* t, size_t idx);

// Set element at flat index
void nc_tensor_set_flat(nc_tensor* t, size_t idx, double value);

// Get element at indices
double nc_tensor_get(const nc_tensor* t, const size_t* indices);

// Set element at indices
void nc_tensor_set(nc_tensor* t, const size_t* indices, double value);

// Get element (variadic, up to 4 dims convenience)
double nc_tensor_get1(const nc_tensor* t, size_t i);
double nc_tensor_get2(const nc_tensor* t, size_t i, size_t j);
double nc_tensor_get3(const nc_tensor* t, size_t i, size_t j, size_t k);
double nc_tensor_get4(const nc_tensor* t, size_t i, size_t j, size_t k, size_t l);

// Set element (variadic)
void nc_tensor_set1(nc_tensor* t, size_t i, double value);
void nc_tensor_set2(nc_tensor* t, size_t i, size_t j, double value);
void nc_tensor_set3(nc_tensor* t, size_t i, size_t j, size_t k, double value);
void nc_tensor_set4(nc_tensor* t, size_t i, size_t j, size_t k, size_t l, double value);

// ============================================
// Shape operations
// ============================================

// Reshape (returns view if possible)
nc_tensor* nc_tensor_reshape(nc_tensor* t, const size_t* shape, size_t ndim);

// Reshape with inferred dimension (-1)
nc_tensor* nc_tensor_reshape_infer(nc_tensor* t, const size_t* shape, size_t ndim);

// Flatten to 1D
nc_tensor* nc_tensor_flatten(nc_tensor* t);

// Squeeze (remove dims of size 1)
nc_tensor* nc_tensor_squeeze(nc_tensor* t, int64_t dim);

// Unsqueeze (add dim of size 1)
nc_tensor* nc_tensor_unsqueeze(nc_tensor* t, size_t dim);

// Transpose
nc_tensor* nc_tensor_transpose(nc_tensor* t, size_t dim0, size_t dim1);

// Transpose 2D (convenience)
nc_tensor* nc_tensor_t(nc_tensor* t);

// Make contiguous (copy if needed)
nc_tensor* nc_tensor_contiguous(nc_tensor* t);

// Expand (broadcast)
nc_tensor* nc_tensor_expand(nc_tensor* t, const size_t* shape, size_t ndim);

// ============================================
// Autograd
// ============================================

// Enable gradient tracking
void nc_tensor_requires_grad_(nc_tensor* t, bool requires_grad);

// Detach from graph (returns new tensor)
nc_tensor* nc_tensor_detach(const nc_tensor* t);

// Zero gradient in-place
void nc_tensor_zero_grad_(nc_tensor* t);

// ============================================
// GC Support
// ============================================

// Recursively mark tensor and its graph as reachable
void nc_tensor_mark_reachable(nc_tensor* t);

// Check if tensor is marked reachable
bool nc_tensor_is_reachable(const nc_tensor* t);

// Reset reachable flag (for sweep phase)
void nc_tensor_reset_reachable(nc_tensor* t);

// ============================================
// Type conversion
// ============================================

// Convert to dtype
nc_tensor* nc_tensor_to(const nc_tensor* t, nc_dtype dtype);

// Convenience
#define nc_tensor_float(t) nc_tensor_to(t, NC_F32)
#define nc_tensor_double(t) nc_tensor_to(t, NC_F64)
#define nc_tensor_int(t) nc_tensor_to(t, NC_I32)
#define nc_tensor_long(t) nc_tensor_to(t, NC_I64)

// ============================================
// Utility
// ============================================

// Print tensor
void nc_tensor_print(const nc_tensor* t);

// Print tensor with name
void nc_tensor_print_named(const nc_tensor* t, const char* name);

// Print shape only
void nc_tensor_print_shape(const nc_tensor* t);

// Check if shapes are equal
bool nc_tensor_shape_eq(const nc_tensor* a, const nc_tensor* b);

// Check if shapes are broadcastable
bool nc_tensor_broadcastable(const nc_tensor* a, const nc_tensor* b);

// Fill tensor with value
void nc_tensor_fill_(nc_tensor* t, double value);

// Copy data from src to dst
nc_error nc_tensor_copy_(nc_tensor* dst, const nc_tensor* src);

// ============================================
// Device Transfer
// ============================================

// Move tensor to specified device (CPU or CUDA)
void nc_tensor_to_device(nc_tensor* t, nc_device_type device);

#endif // NOCTA_TENSOR_H