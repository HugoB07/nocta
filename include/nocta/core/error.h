#ifndef NOCTA_ERROR_H
#define NOCTA_ERROR_H

#include <stdio.h>
#include <stdlib.h>

typedef enum {
    NC_OK = 0,
    NC_ERR_NULL_PTR,
    NC_ERR_ALLOC,
    NC_ERR_SHAPE_MISMATCH,
    NC_ERR_DTYPE_MISMATCH,
    NC_ERR_INDEX_OUT_OF_BOUNDS,
    NC_ERR_INVALID_AXIS,
    NC_ERR_INVALID_SHAPE,
    NC_ERR_INVALID_DTYPE,
    NC_ERR_NOT_CONTIGUOUS,
    NC_ERR_NO_GRAD,
    NC_ERR_GRAPH_CYCLE,
    NC_ERR_IO,
    NC_ERR_UNKNOWN
} nc_error;

// Error messages
static const char* nc_error_messages[] = {
    [NC_OK]                    = "Success",
    [NC_ERR_NULL_PTR]          = "Null pointer",
    [NC_ERR_ALLOC]             = "Memory allocation failed",
    [NC_ERR_SHAPE_MISMATCH]    = "Shape mismatch",
    [NC_ERR_DTYPE_MISMATCH]    = "Data type mismatch",
    [NC_ERR_INDEX_OUT_OF_BOUNDS] = "Index out of bounds",
    [NC_ERR_INVALID_AXIS]      = "Invalid axis",
    [NC_ERR_INVALID_SHAPE]     = "Invalid shape",
    [NC_ERR_INVALID_DTYPE]     = "Invalid data type",
    [NC_ERR_NOT_CONTIGUOUS]    = "Tensor not contiguous",
    [NC_ERR_NO_GRAD]           = "Gradient not enabled",
    [NC_ERR_GRAPH_CYCLE]       = "Cycle detected in computation graph",
    [NC_ERR_IO]                = "I/O error",
    [NC_ERR_UNKNOWN]           = "Unknown error"
};

// Get error message
static inline const char* nc_error_string(nc_error err) {
    return (err <= NC_ERR_UNKNOWN) ? nc_error_messages[err] : "Unknown error";
}

// Global error state (thread-local in future)
extern nc_error nc_last_error;
extern char nc_error_context[256];

// Set error with context
#define NC_SET_ERROR(err, ...) do { \
    nc_last_error = (err); \
    snprintf(nc_error_context, sizeof(nc_error_context), __VA_ARGS__); \
} while(0)

// Get last error
static inline nc_error nc_get_error(void) {
    return nc_last_error;
}

// Clear error
static inline void nc_clear_error(void) {
    nc_last_error = NC_OK;
    nc_error_context[0] = '\0';
}

// Check and return on error
#define NC_CHECK(expr) do { \
    nc_error err = (expr); \
    if (err != NC_OK) return err; \
} while(0)

// Check pointer, return NULL if null
#define NC_CHECK_NULL(ptr) do { \
    if((ptr) == NULL) { \
        NC_SET_ERROR(NC_ERR_NULL_PTR, "Null pointer: %s", #ptr); \
        return NULL; \
    } \
} while(0)

// Assert with message (debug builds)
#ifdef NOCTA_DEBUG
#define NC_ASSERT(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "NOCTA ASSERT FAILED: %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        abort(); \
    } \
} while(0)
#else
#define NC_ASSERT(cond, ...) ((void)0)
#endif

// Log macros
#ifdef NOCTA_DEBUG
#define NC_LOG(fmt, ...) fprintf(stderr, "[NOCTA] " fmt "\n", ##__VA_ARGS__)
#define NC_WARN(fmt, ...) fprintf(stderr, "[NOCTA WARN] " fmt "\n", ##__VA_ARGS__)
#else
#define NC_LOG(fmt, ...) ((void)0)
#define NC_WARN(fmt, ...) ((void)0)
#endif

#define NC_ERR(fmt, ...) fprintf(stderr, "[NOCTA ERROR] " fmt "\n", ##__VA_ARGS__)

#endif // NOCTA_ERROR_H