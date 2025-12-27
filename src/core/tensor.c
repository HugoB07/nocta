#include "nocta/core/tensor.h"
#include "nocta/autograd/node.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#ifdef NOCTA_CUDA_ENABLED
#include "nocta/cuda/cuda_kernels.h"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Global error state
nc_error nc_last_error = NC_OK;
char nc_error_context[256] = {0};

// RNG state
static uint64_t rng_state[2] = {0};

static void rng_init(void) {
    if (rng_state[0] == 0) {
        rng_state[0] = (uint64_t)time(NULL);
        rng_state[1] = rng_state[0] ^ 0x5DEECE66DULL;
    }
}

static uint64_t xorshift128plus(void) {
    uint64_t x = rng_state[0];
    uint64_t y = rng_state[1];
    rng_state[0] = y;
    x ^= x << 23;
    rng_state[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return rng_state[1] + y;
}

static double rand_uniform(void) {
    return (xorshift128plus() >> 11) * (1.0 / 9007199254740992.0);
}

static double rand_normal(void) {
    double u1 = rand_uniform();
    double u2 = rand_uniform();
    return sqrt(-2.0 * log(u1 + 1e-10)) * cos(2.0 * M_PI * u2);
}

// Compute strides for contiguous tensor
static void compute_strides(size_t* strides, const size_t* shape, size_t ndim) {
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

// Compute total elements
static size_t compute_numel(const size_t* shape, size_t ndim) {
    if (ndim == 0) return 1; // scalar
    size_t n = 1;
    for (size_t i = 0; i < ndim; i++) {
        n *= shape[i];
    }
    return n;
}

// Allocate tensor structure
static nc_tensor* tensor_alloc(void) {
    nc_tensor* t = nc_calloc(1, sizeof(nc_tensor));
    if (!t) {
        NC_SET_ERROR(NC_ERR_ALLOC, "Failed to allocate tensor");
        return NULL;
    }
    t->is_leaf = true;
    t->is_contiguous = true;
    t->requires_grad = false;
    t->grad = NULL;
    t->grad_fn = NULL;
    return t;
}

// ============================================
// Creation
// ============================================

nc_tensor* nc_tensor_empty(const size_t* shape, size_t ndim, nc_dtype dtype) {
    if (ndim > NC_MAX_DIMS) {
        NC_SET_ERROR(NC_ERR_INVALID_SHAPE, "ndim %zu exceeds NC_MAX_DIMS", ndim);
        return NULL;
    }
    
    nc_tensor* t = tensor_alloc();
    if (!t) return NULL;
    
    t->ndim = ndim;
    t->dtype = dtype;
    t->numel = compute_numel(shape, ndim);
    
    for (size_t i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
    }
    compute_strides(t->strides, t->shape, ndim);
    
    size_t data_size = t->numel * nc_dtype_sizeof(dtype);
    if (data_size == 0) data_size = 1;
    
    t->storage = nc_storage_create(data_size);
    if (!t->storage) {
        nc_free(t);
        return NULL;
    }
    
    t->offset = 0;
    return t;
}

nc_tensor* nc_tensor_zeros(const size_t* shape, size_t ndim, nc_dtype dtype) {
    nc_tensor* t = nc_tensor_empty(shape, ndim, dtype);
    if (t) {
        memset(t->storage->data, 0, t->numel * nc_dtype_sizeof(dtype));
    }
    return t;
}

nc_tensor* nc_tensor_ones(const size_t* shape, size_t ndim, nc_dtype dtype) {
    nc_tensor* t = nc_tensor_empty(shape, ndim, dtype);
    if (t) {
        nc_tensor_fill_(t, 1.0);
    }
    return t;
}

nc_tensor* nc_tensor_full(const size_t* shape, size_t ndim, nc_dtype dtype, double value) {
    nc_tensor* t = nc_tensor_empty(shape, ndim, dtype);
    if (t) {
        nc_tensor_fill_(t, value);
    }
    return t;
}

nc_tensor* nc_tensor_from_data(const void* data, const size_t* shape, size_t ndim, nc_dtype dtype) {
    NC_CHECK_NULL(data);
    
    nc_tensor* t = nc_tensor_empty(shape, ndim, dtype);
    if (t) {
        memcpy(t->storage->data, data, t->numel * nc_dtype_sizeof(dtype));
    }
    return t;
}

nc_tensor* nc_tensor_scalar(double value, nc_dtype dtype) {
    nc_tensor* t = nc_tensor_empty(NULL, 0, dtype);
    if (t) {
        nc_tensor_fill_(t, value);
    }
    return t;
}

nc_tensor* nc_tensor_arange(double start, double end, double step, nc_dtype dtype) {
    if (step == 0 || (step > 0 && start >= end) || (step < 0 && start <= end)) {
        NC_SET_ERROR(NC_ERR_INVALID_SHAPE, "Invalid arange parameters");
        return NULL;
    }
    
    size_t n = (size_t)ceil((end - start) / step);
    size_t shape[] = {n};
    nc_tensor* t = nc_tensor_empty(shape, 1, dtype);
    if (!t) return NULL;
    
    for (size_t i = 0; i < n; i++) {
        nc_tensor_set_flat(t, i, start + i * step);
    }
    return t;
}

nc_tensor* nc_tensor_linspace(double start, double end, size_t num, nc_dtype dtype) {
    if (num == 0) {
        NC_SET_ERROR(NC_ERR_INVALID_SHAPE, "num must be > 0");
        return NULL;
    }
    
    size_t shape[] = {num};
    nc_tensor* t = nc_tensor_empty(shape, 1, dtype);
    if (!t) return NULL;
    
    double step = (num > 1) ? (end - start) / (num - 1) : 0;
    for (size_t i = 0; i < num; i++) {
        nc_tensor_set_flat(t, i, start + i * step);
    }
    return t;
}

nc_tensor* nc_tensor_rand(const size_t* shape, size_t ndim, nc_dtype dtype) {
    rng_init();
    nc_tensor* t = nc_tensor_empty(shape, ndim, dtype);
    if (!t) return NULL;
    
    for (size_t i = 0; i < t->numel; i++) {
        nc_tensor_set_flat(t, i, rand_uniform());
    }
    return t;
}

nc_tensor* nc_tensor_randn(const size_t* shape, size_t ndim, nc_dtype dtype) {
    rng_init();
    nc_tensor* t = nc_tensor_empty(shape, ndim, dtype);
    if (!t) return NULL;
    
    for (size_t i = 0; i < t->numel; i++) {
        nc_tensor_set_flat(t, i, rand_normal());
    }
    return t;
}

nc_tensor* nc_tensor_randint(const size_t* shape, size_t ndim, int64_t low, int64_t high) {
    rng_init();
    nc_tensor* t = nc_tensor_empty(shape, ndim, NC_I64);
    if (!t) return NULL;
    
    int64_t range = high - low;
    int64_t* data = nc_tensor_data_i64(t);
    for (size_t i = 0; i < t->numel; i++) {
        data[i] = low + (int64_t)(xorshift128plus() % (uint64_t)range);
    }
    return t;
}

nc_tensor* nc_tensor_eye(size_t n, nc_dtype dtype) {
    size_t shape[] = {n, n};
    nc_tensor* t = nc_tensor_zeros(shape, 2, dtype);
    if (!t) return NULL;
    
    for (size_t i = 0; i < n; i++) {
        nc_tensor_set2(t, i, i, 1.0);
    }
    return t;
}

nc_tensor* nc_tensor_clone(const nc_tensor* t) {
    NC_CHECK_NULL(t);
    
    nc_tensor* clone = nc_tensor_empty(t->shape, t->ndim, t->dtype);
    if (!clone) return NULL;
    
    clone->requires_grad = t->requires_grad;
    
#ifdef NOCTA_CUDA_ENABLED
    // If source is on GPU, clone should also be on GPU
    if (t->storage && t->storage->device == NC_DEVICE_CUDA && t->storage->cuda_data) {
        nc_storage_to_device(clone->storage, NC_DEVICE_CUDA);
        if (t->is_contiguous && clone->storage->cuda_data) {
            // GPU to GPU copy
            nc_cuda_copy_f32((float*)clone->storage->cuda_data, 
                            (const float*)t->storage->cuda_data, 
                            t->numel);
            return clone;
        }
    }
#endif
    
    // CPU path
    if (t->is_contiguous) {
        memcpy(clone->storage->data, nc_tensor_data(t), t->numel * nc_dtype_sizeof(t->dtype));
    } else {
        for (size_t i = 0; i < t->numel; i++) {
            nc_tensor_set_flat(clone, i, nc_tensor_get_flat(t, i));
        }
    }
    return clone;
}

void nc_tensor_free(nc_tensor* t) {
    if (!t) return;
    
    // Free the grad_fn node
    if (t->grad_fn) {
        nc_node_free(t->grad_fn);
        t->grad_fn = NULL;
    }
    
    // Free gradient tensor
    if (t->grad) {
        nc_tensor_free(t->grad);
        t->grad = NULL;
    }
    
    // Release storage (ref-counted)
    if (t->storage) {
        nc_storage_release(t->storage);
        t->storage = NULL;
    }
    
    nc_free(t);
}

// ============================================
// Indexing & Access
// ============================================

static size_t flat_to_offset(const nc_tensor* t, size_t flat_idx) {
    size_t offset = t->offset;
    for (int i = (int)t->ndim - 1; i >= 0; i--) {
        size_t idx = flat_idx % t->shape[i];
        flat_idx /= t->shape[i];
        offset += idx * t->strides[i];
    }
    return offset;
}

static size_t indices_to_offset(const nc_tensor* t, const size_t* indices) {
    size_t offset = t->offset;
    for (size_t i = 0; i < t->ndim; i++) {
        offset += indices[i] * t->strides[i];
    }
    return offset;
}

double nc_tensor_get_flat(const nc_tensor* t, size_t idx) {
    if (!t || idx >= t->numel) return 0.0;
    if (!t->storage || !t->storage->data) return 0.0;
    
    size_t offset = t->is_contiguous ? (t->offset + idx) : flat_to_offset(t, idx);
    size_t byte_offset = offset * nc_dtype_sizeof(t->dtype);
    const char* base = (const char*)t->storage->data;
    
    switch (t->dtype) {
        case NC_F32: return (double)(*((float*)(base + byte_offset)));
        case NC_F64: return *((double*)(base + byte_offset));
        case NC_I32: return (double)(*((int32_t*)(base + byte_offset)));
        case NC_I64: return (double)(*((int64_t*)(base + byte_offset)));
        case NC_U8:  return (double)(*((uint8_t*)(base + byte_offset)));
        case NC_BOOL: return (double)(*((uint8_t*)(base + byte_offset)));
        default: return 0.0;
    }
}

void nc_tensor_set_flat(nc_tensor* t, size_t idx, double value) {
    if (!t || idx >= t->numel) return;
    if (!t->storage || !t->storage->data) return;
    
    size_t offset = t->is_contiguous ? (t->offset + idx) : flat_to_offset(t, idx);
    size_t byte_offset = offset * nc_dtype_sizeof(t->dtype);
    char* base = (char*)t->storage->data;
    
    switch (t->dtype) {
        case NC_F32: *((float*)(base + byte_offset)) = (float)value; break;
        case NC_F64: *((double*)(base + byte_offset)) = value; break;
        case NC_I32: *((int32_t*)(base + byte_offset)) = (int32_t)value; break;
        case NC_I64: *((int64_t*)(base + byte_offset)) = (int64_t)value; break;
        case NC_U8:  *((uint8_t*)(base + byte_offset)) = (uint8_t)value; break;
        case NC_BOOL: *((uint8_t*)(base + byte_offset)) = value != 0; break;
        default: break;
    }
}

double nc_tensor_get(const nc_tensor* t, const size_t* indices) {
    if (!t) return 0.0;
    size_t offset = indices_to_offset(t, indices);
    void* ptr = (char*)t->storage->data + offset * nc_dtype_sizeof(t->dtype);
    
    switch (t->dtype) {
        case NC_F32: return ((float*)ptr)[0];
        case NC_F64: return ((double*)ptr)[0];
        case NC_I32: return ((int32_t*)ptr)[0];
        case NC_I64: { int64_t val = ((int64_t*)ptr)[0]; return (double)val; }
        case NC_U8:  return ((uint8_t*)ptr)[0];
        default: return 0.0;
    }
}

void nc_tensor_set(nc_tensor* t, const size_t* indices, double value) {
    if (!t) return;
    size_t offset = indices_to_offset(t, indices);
    void* ptr = (char*)t->storage->data + offset * nc_dtype_sizeof(t->dtype);
    
    switch (t->dtype) {
        case NC_F32: ((float*)ptr)[0] = (float)value; break;
        case NC_F64: ((double*)ptr)[0] = value; break;
        case NC_I32: ((int32_t*)ptr)[0] = (int32_t)value; break;
        case NC_I64: ((int64_t*)ptr)[0] = (int64_t)value; break;
        case NC_U8:  ((uint8_t*)ptr)[0] = (uint8_t)value; break;
        default: break;
    }
}

// Convenience accessors
double nc_tensor_get1(const nc_tensor* t, size_t i) {
    size_t idx[] = {i};
    return nc_tensor_get(t, idx);
}

double nc_tensor_get2(const nc_tensor* t, size_t i, size_t j) {
    size_t idx[] = {i, j};
    return nc_tensor_get(t, idx);
}

double nc_tensor_get3(const nc_tensor* t, size_t i, size_t j, size_t k) {
    size_t idx[] = {i, j, k};
    return nc_tensor_get(t, idx);
}

double nc_tensor_get4(const nc_tensor* t, size_t i, size_t j, size_t k, size_t l) {
    size_t idx[] = {i, j, k, l};
    return nc_tensor_get(t, idx);
}

void nc_tensor_set1(nc_tensor* t, size_t i, double val) {
    size_t idx[] = {i};
    nc_tensor_set(t, idx, val);
}

void nc_tensor_set2(nc_tensor* t, size_t i, size_t j, double val) {
    size_t idx[] = {i, j};
    nc_tensor_set(t, idx, val);
}

void nc_tensor_set3(nc_tensor* t, size_t i, size_t j, size_t k, double val) {
    size_t idx[] = {i, j, k};
    nc_tensor_set(t, idx, val);
}

void nc_tensor_set4(nc_tensor* t, size_t i, size_t j, size_t k, size_t l, double val) {
    size_t idx[] = {i, j, k, l};
    nc_tensor_set(t, idx, val);
}

// ============================================
// Shape operations
// ============================================

nc_tensor* nc_tensor_reshape(nc_tensor* t, const size_t* shape, size_t ndim) {
    NC_CHECK_NULL(t);
    
    size_t new_numel = compute_numel(shape, ndim);
    if (new_numel != t->numel) {
        NC_SET_ERROR(NC_ERR_SHAPE_MISMATCH, "Cannot reshape %zu to %zu elements", t->numel, new_numel);
        return NULL;
    }
    
    // If contiguous, return view
    if (t->is_contiguous) {
        nc_tensor* v = tensor_alloc();
        if (!v) return NULL;
        
        v->storage = nc_storage_retain(t->storage);
        v->offset = t->offset;
        v->ndim = ndim;
        v->numel = new_numel;
        v->dtype = t->dtype;
        v->requires_grad = t->requires_grad;
        v->is_contiguous = true;
        
        for (size_t i = 0; i < ndim; i++) {
            v->shape[i] = shape[i];
        }
        compute_strides(v->strides, v->shape, ndim);
        
        return v;
    }
    
    // Non-contiguous: must copy
    nc_tensor* c = nc_tensor_contiguous(t);
    if (!c) return NULL;
    
    nc_tensor* v = nc_tensor_reshape(c, shape, ndim);
    nc_tensor_free(c);
    return v;
}

nc_tensor* nc_tensor_flatten(nc_tensor* t) {
    size_t shape[] = {t->numel};
    return nc_tensor_reshape(t, shape, 1);
}

nc_tensor* nc_tensor_squeeze(nc_tensor* t, int64_t dim) {
    NC_CHECK_NULL(t);
    
    size_t new_shape[NC_MAX_DIMS];
    size_t new_ndim = 0;
    
    for (size_t i = 0; i < t->ndim; i++) {
        if (dim < 0) {
            // Squeeze all dims of size 1
            if (t->shape[i] != 1) {
                new_shape[new_ndim++] = t->shape[i];
            }
        } else {
            // Squeeze specific dim
            if (i == (size_t)dim && t->shape[i] == 1) continue;
            new_shape[new_ndim++] = t->shape[i];
        }
    }
    
    return nc_tensor_reshape(t, new_shape, new_ndim);
}

nc_tensor* nc_tensor_unsqueeze(nc_tensor* t, size_t dim) {
    NC_CHECK_NULL(t);
    
    if (dim > t->ndim) {
        NC_SET_ERROR(NC_ERR_INVALID_AXIS, "dim %zu > ndim %zu", dim, t->ndim);
        return NULL;
    }
    
    size_t new_shape[NC_MAX_DIMS];
    size_t j = 0;
    for (size_t i = 0; i <= t->ndim; i++) {
        if (i == dim) {
            new_shape[j++] = 1;
        }
        if (i < t->ndim) {
            new_shape[j++] = t->shape[i];
        }
    }
    
    return nc_tensor_reshape(t, new_shape, t->ndim + 1);
}

nc_tensor* nc_tensor_transpose(nc_tensor* t, size_t dim0, size_t dim1) {
    NC_CHECK_NULL(t);
    
    if (dim0 >= t->ndim || dim1 >= t->ndim) {
        NC_SET_ERROR(NC_ERR_INVALID_AXIS, "Invalid transpose dims");
        return NULL;
    }
    
    nc_tensor* v = tensor_alloc();
    if (!v) return NULL;
    
    v->storage = nc_storage_retain(t->storage);
    v->offset = t->offset;
    v->ndim = t->ndim;
    v->numel = t->numel;
    v->dtype = t->dtype;
    v->requires_grad = t->requires_grad;  // <-- IMPORTANT: propagate requires_grad
    
    memcpy(v->shape, t->shape, sizeof(t->shape));
    memcpy(v->strides, t->strides, sizeof(t->strides));
    
    // Swap
    size_t tmp = v->shape[dim0];
    v->shape[dim0] = v->shape[dim1];
    v->shape[dim1] = tmp;
    
    tmp = v->strides[dim0];
    v->strides[dim0] = v->strides[dim1];
    v->strides[dim1] = tmp;
    
    v->is_contiguous = false;
    return v;
}

nc_tensor* nc_tensor_t(nc_tensor* t) {
    if (!t || t->ndim != 2) return NULL;
    return nc_tensor_transpose(t, 0, 1);
}

nc_tensor* nc_tensor_contiguous(nc_tensor* t) {
    NC_CHECK_NULL(t);
    
    if (t->is_contiguous) {
        return nc_tensor_clone(t);
    }
    
    nc_tensor* c = nc_tensor_empty(t->shape, t->ndim, t->dtype);
    if (!c) return NULL;
    
    for (size_t i = 0; i < t->numel; i++) {
        nc_tensor_set_flat(c, i, nc_tensor_get_flat(t, i));
    }
    
    c->requires_grad = t->requires_grad;
    return c;
}

// ============================================
// Autograd
// ============================================

void nc_tensor_requires_grad_(nc_tensor* t, bool requires_grad) {
    if (!t) return;
    t->requires_grad = requires_grad;
    if (requires_grad && !t->grad) {
        t->grad = nc_tensor_zeros(t->shape, t->ndim, t->dtype);
    }
}

nc_tensor* nc_tensor_detach(const nc_tensor* t) {
    NC_CHECK_NULL(t);
    nc_tensor* d = nc_tensor_clone(t);
    if (d) {
        d->requires_grad = false;
        d->grad_fn = NULL;
        d->is_leaf = true;
    }
    return d;
}

void nc_tensor_zero_grad_(nc_tensor* t) {
    if (t && t->grad) {
        nc_tensor_fill_(t->grad, 0.0);
    }
}

// ============================================
// Type conversion
// ============================================

nc_tensor* nc_tensor_to(const nc_tensor* t, nc_dtype dtype) {
    NC_CHECK_NULL(t);
    
    if (t->dtype == dtype) {
        return nc_tensor_clone(t);
    }
    
    nc_tensor* out = nc_tensor_empty(t->shape, t->ndim, dtype);
    if (!out) return NULL;
    
    for (size_t i = 0; i < t->numel; i++) {
        nc_tensor_set_flat(out, i, nc_tensor_get_flat(t, i));
    }
    return out;
}

// ============================================
// Utility
// ============================================

void nc_tensor_fill_(nc_tensor* t, double value) {
    if (!t) return;

#ifdef NOCTA_CUDA_ENABLED
    if (t->storage && t->storage->cuda_data && t->is_contiguous) {
        if (t->dtype == NC_F32) {
            nc_cuda_fill_f32((float*)t->storage->cuda_data, (float)value, t->numel);
            return;
        } else if (t->dtype == NC_F64) {
            nc_cuda_fill_f64((double*)t->storage->cuda_data, value, t->numel);
            return;
        }
    }
#endif

    for (size_t i = 0; i < t->numel; i++) {
        nc_tensor_set_flat(t, i, value);
    }
}

nc_error nc_tensor_copy_(nc_tensor* dst, const nc_tensor* src) {
    if (!dst || !src) return NC_ERR_NULL_PTR;
    if (dst->numel != src->numel) return NC_ERR_SHAPE_MISMATCH;
    
#ifdef NOCTA_CUDA_ENABLED
    if (dst->storage && dst->storage->cuda_data && 
        src->storage && src->storage->cuda_data &&
        dst->is_contiguous && src->is_contiguous) {
        
        if (dst->dtype == NC_F32 && src->dtype == NC_F32) {
            nc_cuda_copy_f32((float*)dst->storage->cuda_data, (const float*)src->storage->cuda_data, dst->numel);
            return NC_OK;
        } else if (dst->dtype == NC_F64 && src->dtype == NC_F64) {
            nc_cuda_copy_f64((double*)dst->storage->cuda_data, (const double*)src->storage->cuda_data, dst->numel);
            return NC_OK;
        }
    }
#endif

    for (size_t i = 0; i < src->numel; i++) {
        nc_tensor_set_flat(dst, i, nc_tensor_get_flat(src, i));
    }
    return NC_OK;
}

bool nc_tensor_shape_eq(const nc_tensor* a, const nc_tensor* b) {
    if (!a || !b || a->ndim != b->ndim) return false;
    for (size_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return false;
    }
    return true;
}

bool nc_tensor_broadcastable(const nc_tensor* a, const nc_tensor* b) {
    if (!a || !b) return false;
    
    size_t max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    
    for (size_t i = 0; i < max_ndim; i++) {
        size_t da = i < a->ndim ? a->shape[a->ndim - 1 - i] : 1;
        size_t db = i < b->ndim ? b->shape[b->ndim - 1 - i] : 1;
        if (da != db && da != 1 && db != 1) return false;
    }
    return true;
}

void nc_tensor_print_shape(const nc_tensor* t) {
    if (!t) {
        printf("(null)\n");
        return;
    }
    printf("(");
    for (size_t i = 0; i < t->ndim; i++) {
        printf("%zu%s", t->shape[i], i < t->ndim - 1 ? ", " : "");
    }
    printf(")\n");
}

void nc_tensor_print(const nc_tensor* t) {
    nc_tensor_print_named(t, NULL);
}

void nc_tensor_print_named(const nc_tensor* t, const char* name) {
    if (!t) {
        printf("%s: (null)\n", name ? name : "tensor");
        return;
    }
    
    if (name) printf("%s: ", name);
    printf("Tensor(shape=(");
    for (size_t i = 0; i < t->ndim; i++) {
        printf("%zu%s", t->shape[i], i < t->ndim - 1 ? ", " : "");
    }
    printf("), dtype=%s", nc_dtype_to_string(t->dtype));
    if (t->requires_grad) printf(", requires_grad=true");
    printf(")\n");
    
    // Print data for small tensors
    if (t->numel <= 20) {
        printf("[");
        for (size_t i = 0; i < t->numel; i++) {
            printf("%.4f%s", nc_tensor_get_flat(t, i), i < t->numel - 1 ? ", " : "");
        }
        printf("]\n");
    } else {
        printf("[%.4f, %.4f, %.4f, ..., %.4f, %.4f, %.4f]\n",
            nc_tensor_get_flat(t, 0),
            nc_tensor_get_flat(t, 1),
            nc_tensor_get_flat(t, 2),
            nc_tensor_get_flat(t, t->numel - 3),
            nc_tensor_get_flat(t, t->numel - 2),
            nc_tensor_get_flat(t, t->numel - 1));
    }
}

// ============================================
// Device Transfer
// ============================================

void nc_tensor_to_device(nc_tensor* t, nc_device_type device) {
    if (!t || !t->storage) return;
    nc_storage_to_device(t->storage, device);
    
    // Also move gradient if it exists
    if (t->grad) {
        nc_tensor_to_device(t->grad, device);
    }
}