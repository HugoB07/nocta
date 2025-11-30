#include "nocta/core/memory.h"
#include "nocta/core/error.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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
// Reference-counted storage
// ============================================

nc_storage* nc_storage_create(size_t size) {
    if (size == 0) size = 1;
    
    nc_storage* s = (nc_storage*)nc_alloc(sizeof(nc_storage));
    if (!s) {
        fprintf(stderr, "[nc_storage_create] Failed to alloc storage struct\n");
        return NULL;
    }
    
    s->data = nc_alloc(size);
    if (!s->data) {
        fprintf(stderr, "[nc_storage_create] Failed to alloc data of size %zu\n", size);
        nc_free(s);
        return NULL;
    }
    
    s->size = size;
    nc_atomic_init(&s->refcount, 1);
    
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
        nc_free(storage->data);
        nc_free(storage);
    }
}

int nc_storage_refcount(const nc_storage* storage) {
    return storage ? nc_atomic_load(&storage->refcount) : 0;
}