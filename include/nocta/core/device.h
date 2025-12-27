#ifndef NOCTA_DEVICE_H
#define NOCTA_DEVICE_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Device types
typedef enum {
    NC_DEVICE_CPU = 0,
    NC_DEVICE_CUDA = 1
} nc_device_type;

// Device info structure
typedef struct {
    nc_device_type type;
    int index;          // Device index (for multi-GPU)
    char name[256];     // Device name
    size_t total_memory;
    size_t free_memory;
} nc_device_info;

// ============================================
// Global Device Control
// ============================================

// Set/get default device for new tensor allocations
void nc_set_default_device(nc_device_type type);
nc_device_type nc_get_default_device(void);

// ============================================
// CUDA-specific Functions
// ============================================

// Check if CUDA is available at runtime
bool nc_cuda_available(void);

// Get number of CUDA devices
int nc_cuda_device_count(void);

// Get info about a CUDA device
nc_device_info nc_cuda_device_info(int device_index);

// Set current CUDA device
void nc_cuda_set_device(int index);

// Synchronize current CUDA device
void nc_cuda_synchronize(void);

// Initialize CUDA backend (called automatically on first use)
void nc_cuda_init(void);

// Cleanup CUDA resources
void nc_cuda_cleanup(void);

// ============================================
// CUDA Memory Management (internal)
// ============================================

void* nc_cuda_malloc(size_t size);
void nc_cuda_free(void* ptr);
void nc_cuda_memcpy_h2d(void* dst, const void* src, size_t size);
void nc_cuda_memcpy_d2h(void* dst, const void* src, size_t size);
void nc_cuda_memcpy_d2d(void* dst, const void* src, size_t size);
void nc_cuda_memset(void* ptr, int value, size_t size);

#ifdef __cplusplus
}
#endif

#endif // NOCTA_DEVICE_H
