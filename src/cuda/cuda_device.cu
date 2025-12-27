#ifdef NOCTA_CUDA_ENABLED

#include "nocta/core/device.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

// CUDA initialization state
static bool g_cuda_initialized = false;
static int g_cuda_device_count = 0;
static int g_current_cuda_device = 0;

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
} while(0)

void nc_cuda_init(void) {
    if (g_cuda_initialized) return;
    
    cudaError_t err = cudaGetDeviceCount(&g_cuda_device_count);
    if (err != cudaSuccess || g_cuda_device_count == 0) {
        g_cuda_device_count = 0;
        fprintf(stderr, "[CUDA] No CUDA devices found\n");
        return;
    }
    
    // Set first device as default
    CUDA_CHECK(cudaSetDevice(0));
    g_current_cuda_device = 0;
    g_cuda_initialized = true;
    
    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("[CUDA] Initialized: %s (Compute %d.%d, %.1f GB)\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}

bool nc_cuda_available(void) {
    if (!g_cuda_initialized) {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        return (err == cudaSuccess && count > 0);
    }
    return g_cuda_device_count > 0;
}

int nc_cuda_device_count(void) {
    if (!g_cuda_initialized) nc_cuda_init();
    return g_cuda_device_count;
}

nc_device_info nc_cuda_device_info(int device_index) {
    nc_device_info info;
    memset(&info, 0, sizeof(info));
    info.type = NC_DEVICE_CUDA;
    info.index = device_index;
    
    if (device_index < 0 || device_index >= g_cuda_device_count) {
        strcpy(info.name, "Invalid device");
        return info;
    }
    
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_index) == cudaSuccess) {
        strncpy(info.name, prop.name, sizeof(info.name) - 1);
        info.total_memory = prop.totalGlobalMem;
        
        size_t free_mem, total_mem;
        int prev_device;
        cudaGetDevice(&prev_device);
        cudaSetDevice(device_index);
        cudaMemGetInfo(&free_mem, &total_mem);
        cudaSetDevice(prev_device);
        info.free_memory = free_mem;
    }
    
    return info;
}

void nc_cuda_set_device(int index) {
    if (index >= 0 && index < g_cuda_device_count) {
        CUDA_CHECK(cudaSetDevice(index));
        g_current_cuda_device = index;
    }
}

void nc_cuda_synchronize(void) {
    CUDA_CHECK(cudaDeviceSynchronize());
}

void nc_cuda_cleanup(void) {
    if (g_cuda_initialized) {
        cudaDeviceReset();
        g_cuda_initialized = false;
    }
}

// ============================================
// CUDA Memory Management
// ============================================

void* nc_cuda_malloc(size_t size) {
    void* ptr = NULL;
    if (!g_cuda_initialized) nc_cuda_init();
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void nc_cuda_free(void* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void nc_cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void nc_cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void nc_cuda_memcpy_d2d(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

void nc_cuda_memset(void* ptr, int value, size_t size) {
    CUDA_CHECK(cudaMemset(ptr, value, size));
}

#endif // NOCTA_CUDA_ENABLED
