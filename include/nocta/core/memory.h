#ifndef NOCTA_MEMORY_H
#define NOCTA_MEMORY_H

#include <stddef.h>
#include <stdint.h>
#include "device.h"

// Platform-specific atomics
#ifdef _MSC_VER
    #include <windows.h>
    typedef volatile LONG nc_atomic_int;
    #define nc_atomic_inc(x) InterlockedIncrement((x))
    #define nc_atomic_dec(x) InterlockedDecrement((x))
    #define nc_atomic_load(x) (*(x))
    #define nc_atomic_init(x, v) do { *(x) = (v); } while(0)
#else
    #include <stdatomic.h>
    typedef _Atomic int nc_atomic_int;
    #define nc_atomic_inc(x) (atomic_fetch_add((x), 1) + 1)
    #define nc_atomic_dec(x) (atomic_fetch_sub((x), 1) - 1)
    #define nc_atomic_load(x) atomic_load((x))
    #define nc_atomic_init(x, v) atomic_init((x), (v))
#endif

// Alignment for SIMD (AVX-256 = 32 bytes, AVX-512 = 64 bytes)
#define NC_MEMORY_ALIGN 32

// Allocate aligned memory
void* nc_alloc(size_t size);

// Allocate zeroed aligned memory
void* nc_calloc(size_t count, size_t size);

// Reallocate aligned memory
void* nc_realloc(void* ptr, size_t old_size, size_t new_size);

// Free aligned memory
void nc_free(void* ptr);

// Memory statistics
typedef struct {
    size_t total_allocated;
    size_t total_freed;
    size_t current_usage;
    size_t peak_usage;
    size_t allocation_count;
    size_t free_count;
} nc_mem_stats;

// Get memory statistics
nc_mem_stats nc_memory_stats(void);

// Reset statistics
void nc_memory_stats_reset(void);

// Print memory report
void nc_memory_report(void);

// ============================================
// Device-aware Storage
// ============================================

// Reference counting for shared data with device support
typedef struct {
    void* data;              // CPU data pointer (NULL if on GPU only)
    void* cuda_data;         // CUDA device pointer (NULL if on CPU only)
    size_t size;
    nc_atomic_int refcount;
    nc_device_type device;   // Primary device location
} nc_storage;

// Create storage on default device
nc_storage* nc_storage_create(size_t size);

// Create storage on specific device
nc_storage* nc_storage_create_on(size_t size, nc_device_type device);

// Increment refcount
nc_storage* nc_storage_retain(nc_storage* storage);

// Decrement refcount, free if zero
void nc_storage_release(nc_storage* storage);

// Get refcount
int nc_storage_refcount(const nc_storage* storage);

// ============================================
// Device Transfer
// ============================================

// Transfer storage to target device (creates copy on target if needed)
void nc_storage_to_device(nc_storage* s, nc_device_type target);

// Ensure data is available on target device (syncs if needed)
void nc_storage_ensure_on(nc_storage* s, nc_device_type target);

// Get pointer to data on current device
void* nc_storage_data_ptr(nc_storage* s);

// Synchronize storage from GPU to CPU (for reading)
void nc_storage_sync_to_cpu(nc_storage* s);

#endif // NOCTA_MEMORY_H