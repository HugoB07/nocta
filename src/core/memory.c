#include "nocta/core/memory.h"
#include "nocta/core/error.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef NOCTA_CUDA_ENABLED
#include "nocta/cuda/cuda_kernels.h"
#endif

// Memory statistics
static nc_mem_stats g_mem_stats = {0};

// We store the size before each allocation to track frees
typedef struct {
    size_t size;
} alloc_header;

#define HEADER_SIZE sizeof(alloc_header)

// Platform-specific aligned allocation
#if defined(_WIN32)
    #include <malloc.h>
    #define platform_aligned_alloc(align, size) _aligned_malloc(size, align)
    #define platform_aligned_free(ptr) _aligned_free(ptr)
#else
    static void* platform_aligned_alloc(size_t align, size_t size) {
        void* ptr = NULL;
        if (posix_memalign(&ptr, align, size) != 0) return NULL;
        return ptr;
    }
    #define platform_aligned_free(ptr) free(ptr)
#endif

void* nc_alloc(size_t size) {
    if (size == 0) size = 1;
    
    // Allocate extra space for header
    size_t total_size = size + HEADER_SIZE + NC_MEMORY_ALIGN;
    size_t aligned_size = (total_size + NC_MEMORY_ALIGN - 1) & ~(NC_MEMORY_ALIGN - 1);
    
    void* raw = platform_aligned_alloc(NC_MEMORY_ALIGN, aligned_size);
    if (!raw) {
        fprintf(stderr, "[nc_alloc] Failed to allocate %zu bytes\n", size);
        return NULL;
    }
    
    // Store size in header
    alloc_header* header = (alloc_header*)raw;
    header->size = aligned_size;
    
    // Return pointer after header (aligned)
    void* ptr = (char*)raw + NC_MEMORY_ALIGN;
    
    g_mem_stats.total_allocated += aligned_size;
    g_mem_stats.current_usage += aligned_size;
    g_mem_stats.allocation_count++;
    
    if (g_mem_stats.current_usage > g_mem_stats.peak_usage) {
        g_mem_stats.peak_usage = g_mem_stats.current_usage;
    }
    
    return ptr;
}

void* nc_calloc(size_t count, size_t size) {
    size_t total = count * size;
    void* ptr = nc_alloc(total);
    if (ptr) {
        memset(ptr, 0, total);
    }
    return ptr;
}

void* nc_realloc(void* ptr, size_t old_size, size_t new_size) {
    (void)old_size; // We track size internally now
    
    if (new_size == 0) {
        nc_free(ptr);
        return NULL;
    }
    
    if (!ptr) {
        return nc_alloc(new_size);
    }
    
    // Get old size from header
    void* raw = (char*)ptr - NC_MEMORY_ALIGN;
    alloc_header* header = (alloc_header*)raw;
    size_t old_alloc_size = header->size;
    
    void* new_ptr = nc_alloc(new_size);
    if (!new_ptr) return NULL;
    
    // Copy old data
    size_t copy_size = old_alloc_size - HEADER_SIZE - NC_MEMORY_ALIGN;
    if (new_size < copy_size) copy_size = new_size;
    memcpy(new_ptr, ptr, copy_size);
    
    nc_free(ptr);
    return new_ptr;
}

void nc_free(void* ptr) {
    if (!ptr) return;
    
    // Get header
    void* raw = (char*)ptr - NC_MEMORY_ALIGN;
    alloc_header* header = (alloc_header*)raw;
    size_t size = header->size;
    
    g_mem_stats.total_freed += size;
    g_mem_stats.current_usage -= size;
    g_mem_stats.free_count++;
    
    platform_aligned_free(raw);
}

nc_mem_stats nc_memory_stats(void) {
    return g_mem_stats;
}

void nc_memory_stats_reset(void) {
    memset(&g_mem_stats, 0, sizeof(g_mem_stats));
}

void nc_memory_report(void) {
    printf("=== Nocta Memory Report ===\n");
    printf("Total allocated: %zu bytes\n", g_mem_stats.total_allocated);
    printf("Total freed:     %zu bytes\n", g_mem_stats.total_freed);
    printf("Current usage:   %zu bytes\n", g_mem_stats.current_usage);
    printf("Peak usage:      %zu bytes\n", g_mem_stats.peak_usage);
    printf("Allocations:     %zu\n", g_mem_stats.allocation_count);
    printf("Frees:           %zu\n", g_mem_stats.free_count);
    if (g_mem_stats.current_usage > 0) {
        printf("WARNING: Memory leak detected!\n");
    }
    printf("===========================\n");
}

// ============================================
// Device-aware Storage
// ============================================

nc_storage* nc_storage_create(size_t size) {
    return nc_storage_create_on(size, nc_get_default_device());
}

nc_storage* nc_storage_create_on(size_t size, nc_device_type device) {
    if (size == 0) size = 1;
    
    nc_storage* s = (nc_storage*)nc_alloc(sizeof(nc_storage));
    if (!s) {
        fprintf(stderr, "[nc_storage_create] Failed to alloc storage struct\n");
        return NULL;
    }
    
    s->data = NULL;
    s->cuda_data = NULL;
    s->size = size;
    s->device = device;
    nc_atomic_init(&s->refcount, 1);
    
    if (device == NC_DEVICE_CPU) {
        s->data = nc_alloc(size);
        if (!s->data) {
            fprintf(stderr, "[nc_storage_create] Failed to alloc CPU data of size %zu\n", size);
            nc_free(s);
            return NULL;
        }
    }
#ifdef NOCTA_CUDA_ENABLED
    else if (device == NC_DEVICE_CUDA) {
        s->cuda_data = nc_cuda_malloc(size);
        if (!s->cuda_data) {
            fprintf(stderr, "[nc_storage_create] Failed to alloc CUDA data of size %zu\n", size);
            nc_free(s);
            return NULL;
        }
    }
#endif
    else {
        // Fallback to CPU
        s->device = NC_DEVICE_CPU;
        s->data = nc_alloc(size);
        if (!s->data) {
            nc_free(s);
            return NULL;
        }
    }
    
    return s;
}

nc_storage* nc_storage_retain(nc_storage* storage) {
    if (storage) {
        nc_atomic_inc(&storage->refcount);
    }
    return storage;
}

void nc_storage_release(nc_storage* storage) {
    if (!storage) return;
    
    int new_count = nc_atomic_dec(&storage->refcount);
    if (new_count == 0) {
        if (storage->data) {
            nc_free(storage->data);
        }
#ifdef NOCTA_CUDA_ENABLED
        if (storage->cuda_data) {
            nc_cuda_free(storage->cuda_data);
        }
#endif
        nc_free(storage);
    }
}

int nc_storage_refcount(const nc_storage* storage) {
    return storage ? nc_atomic_load(&storage->refcount) : 0;
}

// ============================================
// Device Transfer
// ============================================

void nc_storage_to_device(nc_storage* s, nc_device_type target) {
    if (!s || s->device == target) return;
    
#ifdef NOCTA_CUDA_ENABLED
    if (target == NC_DEVICE_CUDA) {
        // CPU -> CUDA
        if (!s->cuda_data) {
            s->cuda_data = nc_cuda_malloc(s->size);
        }
        if (s->data && s->cuda_data) {
            nc_cuda_memcpy_h2d(s->cuda_data, s->data, s->size);
        }
        s->device = NC_DEVICE_CUDA;
    } else if (target == NC_DEVICE_CPU) {
        // CUDA -> CPU
        if (!s->data) {
            s->data = nc_alloc(s->size);
        }
        if (s->cuda_data && s->data) {
            nc_cuda_memcpy_d2h(s->data, s->cuda_data, s->size);
        }
        s->device = NC_DEVICE_CPU;
    }
#else
    (void)target;
#endif
}

void nc_storage_ensure_on(nc_storage* s, nc_device_type target) {
    if (!s) return;
    
#ifdef NOCTA_CUDA_ENABLED
    if (target == NC_DEVICE_CUDA && !s->cuda_data) {
        // Need to allocate and copy to CUDA
        s->cuda_data = nc_cuda_malloc(s->size);
        if (s->data && s->cuda_data) {
            nc_cuda_memcpy_h2d(s->cuda_data, s->data, s->size);
        }
    } else if (target == NC_DEVICE_CPU && !s->data) {
        // Need to allocate and copy to CPU
        s->data = nc_alloc(s->size);
        if (s->cuda_data && s->data) {
            nc_cuda_memcpy_d2h(s->data, s->cuda_data, s->size);
        }
    }
#else
    (void)target;
#endif
}

void* nc_storage_data_ptr(nc_storage* s) {
    if (!s) return NULL;
    
#ifdef NOCTA_CUDA_ENABLED
    if (s->device == NC_DEVICE_CUDA) {
        return s->cuda_data;
    }
#endif
    return s->data;
}

void nc_storage_sync_to_cpu(nc_storage* s) {
#ifdef NOCTA_CUDA_ENABLED
    if (s && s->device == NC_DEVICE_CUDA && s->cuda_data) {
        if (!s->data) {
            s->data = nc_alloc(s->size);
        }
        if (s->data) {
            nc_cuda_memcpy_d2h(s->data, s->cuda_data, s->size);
        }
    }
#else
    (void)s;
#endif
}