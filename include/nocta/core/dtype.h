#ifndef NOCTA_DTYPE_H
#define NOCTA_DTYPE_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    NC_F32,     // float 32-bit
    NC_F64,     // float 64-bit
    NC_I32,     // int 32-bit
    NC_I64,     // int 64-bit
    NC_U8,      // unsigned 8-bit (images)
    NC_BOOL,    // boolean
    NC_DTYPE_COUNT
} nc_dtype;

// Type sizes in bytes
static const size_t nc_dtype_size[] = {
    sizeof(float),      // NC_F32
    sizeof(double),     // NC_F64
    sizeof(int32_t),    // NC_I32
    sizeof(int64_t),    // NC_I64
    sizeof(uint8_t),    // NC_U8
    sizeof(uint8_t)     // NC_BOOL
};

// Type names for debug
static const char* nc_dtype_name[] = {
    "float32",      // NC_F32
    "float64",      // NC_F64
    "int32",        // NC_I32
    "int64",        // NC_I64
    "uint8",        // NC_U8
    "bool"          // NC_BOOL
};

// Get size of dtype
static inline size_t nc_dtype_sizeof(nc_dtype dtype) {
    return (dtype < NC_DTYPE_COUNT) ? nc_dtype_size[dtype] : 0;
}

// Get name of dtype
static inline const char* nc_dtype_to_string(nc_dtype dtype) {
    return (dtype < NC_DTYPE_COUNT) ? nc_dtype_name[dtype] : "unknown";
}

// Check if dtype is integer
static inline int nc_dtype_is_integer(nc_dtype dtype) {
    return dtype == NC_I32 || dtype == NC_I64 || dtype == NC_U8;
}

// Promote dtypes (for mixed operations)
static inline nc_dtype nc_dtype_promote(nc_dtype a, nc_dtype b) {
    if (a == NC_F64 || b == NC_F64) return NC_F64;
    if (a == NC_F32 || b == NC_F32) return NC_F32;
    if (a == NC_I64 || b == NC_I64) return NC_I64;
    if (a == NC_I32 || b == NC_I32) return NC_I32;
    return NC_U8;
}

#endif // NOCTA_DTYPE_H