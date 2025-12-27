#include "nocta/core/device.h"
#include <stdio.h>
#include <string.h>

// Global default device - starts as CPU
static nc_device_type g_default_device = NC_DEVICE_CPU;

void nc_set_default_device(nc_device_type type) {
#ifdef NOCTA_CUDA_ENABLED
    if (type == NC_DEVICE_CUDA) {
        if (!nc_cuda_available()) {
            fprintf(stderr, "[WARNING] CUDA not available, falling back to CPU\n");
            g_default_device = NC_DEVICE_CPU;
            return;
        }
        nc_cuda_init();  // Ensure CUDA is initialized
    }
#else
    if (type == NC_DEVICE_CUDA) {
        fprintf(stderr, "[WARNING] CUDA support not compiled, falling back to CPU\n");
        type = NC_DEVICE_CPU;
    }
#endif
    g_default_device = type;
}

nc_device_type nc_get_default_device(void) {
    return g_default_device;
}

// CPU-only stubs when CUDA is not enabled
#ifndef NOCTA_CUDA_ENABLED

bool nc_cuda_available(void) {
    return false;
}

int nc_cuda_device_count(void) {
    return 0;
}

nc_device_info nc_cuda_device_info(int device_index) {
    (void)device_index;
    nc_device_info info = {0};
    info.type = NC_DEVICE_CPU;
    strcpy(info.name, "No CUDA available");
    return info;
}

void nc_cuda_set_device(int index) {
    (void)index;
}

void nc_cuda_synchronize(void) {
}

void nc_cuda_init(void) {
}

void nc_cuda_cleanup(void) {
}

#endif // !NOCTA_CUDA_ENABLED
