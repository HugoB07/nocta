#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nocta/nocta.h"
#include "nocta/nn/conv.h"

#ifdef _WIN32
#include <windows.h>
double get_time() {
    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    return (double)start.QuadPart / frequency.QuadPart;
}
#else
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

void benchmark_matmul(size_t M, size_t K, size_t N) {
    printf("Benchmarking MatMul (%zu x %zu) @ (%zu x %zu)... ", M, K, K, N);
    fflush(stdout);

    size_t shape_a[] = {M, K};
    size_t shape_b[] = {K, N};
    
    nc_tensor* a = nc_tensor_rand(shape_a, 2, NC_F32);
    nc_tensor* b = nc_tensor_rand(shape_b, 2, NC_F32);
    
    // Warmup
    nc_tensor* c = nc_matmul(a, b);
    nc_tensor_free(c);
    
    double start = get_time();
    int iters = 10;
    for (int i = 0; i < iters; i++) {
        c = nc_matmul(a, b);
        nc_tensor_free(c);
    }
    double end = get_time();
    
    double avg_time = (end - start) / iters;
    double ops = 2.0 * M * N * K;
    double gflops = (ops / avg_time) * 1e-9;
    
    printf("Time: %.4f s, Performance: %.2f GFLOPS\n", avg_time, gflops);
    
    nc_tensor_free(a);
    nc_tensor_free(b);
}

void benchmark_conv2d(size_t N, size_t C_in, size_t H, size_t W, size_t C_out, size_t K) {
    printf("Benchmarking Conv2D (N=%zu, C=%zu, H=%zu, W=%zu, K=%zu, Out=%zu)... ", 
           N, C_in, H, W, K, C_out);
    fflush(stdout);
    
    size_t input_shape[] = {N, C_in, H, W};
    nc_tensor* input = nc_tensor_rand(input_shape, 4, NC_F32);
    
    nc_module* conv = nc_conv2d(C_in, C_out, K, 1, 1, false);
    
    // Warmup
    nc_tensor* out = nc_module_forward(conv, input);
    nc_tensor_free(out);
    
    double start = get_time();
    int iters = 5;
    for (int i = 0; i < iters; i++) {
        out = nc_module_forward(conv, input);
        nc_tensor_free(out);
    }
    double end = get_time();
    
    double avg_time = (end - start) / iters;
    // Approx ops: 2 * N * H * W * C_out * C_in * K * K
    double ops = 2.0 * N * H * W * C_out * C_in * K * K;
    double gflops = (ops / avg_time) * 1e-9;
    
    printf("Time: %.4f s, Performance: %.2f GFLOPS\n", avg_time, gflops);
    
    nc_tensor_free(input);
    nc_module_free(conv);
}

int main() {
    nc_init();
    
    printf("=== Nocta Benchmarks ===\n\n");
    
    benchmark_matmul(512, 512, 512);
    benchmark_matmul(1024, 1024, 1024);
    benchmark_matmul(2048, 2048, 2048);
    
    printf("\n");
    
    // Standard ResNet-like layers
    benchmark_conv2d(4, 64, 56, 56, 64, 3);   // Early layer
    benchmark_conv2d(4, 256, 14, 14, 256, 3); // Mid layer
    benchmark_conv2d(4, 512, 7, 7, 512, 3);   // Deep layer
    
    nc_cleanup();
    return 0;
}
