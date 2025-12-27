/**
 * Nocta CPU vs GPU Benchmark
 * 
 * Compares performance of key operations on CPU and GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nocta/nocta.h"
#include "nocta/nn/conv.h"
#include "nocta/nn/batchnorm.h"

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#include <sys/time.h>
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif

// ============================================
// Benchmark Helpers
// ============================================

typedef struct {
    double cpu_time_ms;
    double gpu_time_ms;
    double speedup;
} benchmark_result;

static void print_result(const char* op, benchmark_result* r) {
    if (r->gpu_time_ms > 0) {
        printf("%-25s | CPU: %8.2f ms | GPU: %8.2f ms | Speedup: %.2fx\n",
               op, r->cpu_time_ms, r->gpu_time_ms, r->speedup);
    } else {
        printf("%-25s | CPU: %8.2f ms | GPU: N/A\n", op, r->cpu_time_ms);
    }
}

// ============================================
// MatMul Benchmark
// ============================================

static benchmark_result bench_matmul(size_t M, size_t K, size_t N, int warmup, int iters) {
    benchmark_result result = {0};
    
    size_t a_shape[] = {M, K};
    size_t b_shape[] = {K, N};
    
    // Create tensors on CPU
    nc_tensor* a_cpu = nc_tensor_rand(a_shape, 2, NC_F32);
    nc_tensor* b_cpu = nc_tensor_rand(b_shape, 2, NC_F32);
    
    // Warmup CPU
    for (int i = 0; i < warmup; i++) {
        nc_tensor* c = nc_matmul(a_cpu, b_cpu);
        nc_tensor_free(c);
    }
    
    // Benchmark CPU
    double start = get_time_ms();
    for (int i = 0; i < iters; i++) {
        nc_tensor* c = nc_matmul(a_cpu, b_cpu);
        nc_tensor_free(c);
    }
    result.cpu_time_ms = (get_time_ms() - start) / iters;
    
#ifdef NOCTA_CUDA_ENABLED
    if (nc_cuda_available()) {
        // Create tensors on GPU
        nc_tensor* a_gpu = nc_tensor_rand(a_shape, 2, NC_F32);
        nc_tensor* b_gpu = nc_tensor_rand(b_shape, 2, NC_F32);
        nc_tensor_to_device(a_gpu, NC_DEVICE_CUDA);
        nc_tensor_to_device(b_gpu, NC_DEVICE_CUDA);
        
        // Warmup GPU
        for (int i = 0; i < warmup; i++) {
            nc_tensor* c = nc_matmul(a_gpu, b_gpu);
            nc_cuda_synchronize();
            nc_tensor_free(c);
        }
        
        // Benchmark GPU
        start = get_time_ms();
        for (int i = 0; i < iters; i++) {
            nc_tensor* c = nc_matmul(a_gpu, b_gpu);
            nc_cuda_synchronize();
            nc_tensor_free(c);
        }
        result.gpu_time_ms = (get_time_ms() - start) / iters;
        result.speedup = result.cpu_time_ms / result.gpu_time_ms;
        
        nc_tensor_free(a_gpu);
        nc_tensor_free(b_gpu);
    }
#endif
    
    nc_tensor_free(a_cpu);
    nc_tensor_free(b_cpu);
    
    return result;
}

// ============================================
// Conv2D Benchmark
// ============================================

static benchmark_result bench_conv2d(size_t batch, size_t C_in, size_t H, size_t W,
                                     size_t C_out, size_t ksize, int warmup, int iters) {
    benchmark_result result = {0};
    
    size_t in_shape[] = {batch, C_in, H, W};
    size_t w_shape[] = {C_out, C_in, ksize, ksize};
    size_t b_shape[] = {C_out};
    
    // Create tensors on CPU
    nc_tensor* input_cpu = nc_tensor_rand(in_shape, 4, NC_F32);
    nc_tensor* weight_cpu = nc_tensor_rand(w_shape, 4, NC_F32);
    nc_tensor* bias_cpu = nc_tensor_zeros(b_shape, 1, NC_F32);
    
    // Warmup CPU
    for (int i = 0; i < warmup; i++) {
        nc_tensor* out = nc_conv2d_forward(input_cpu, weight_cpu, bias_cpu, 1, 1);
        nc_tensor_free(out);
    }
    
    // Benchmark CPU
    double start = get_time_ms();
    for (int i = 0; i < iters; i++) {
        nc_tensor* out = nc_conv2d_forward(input_cpu, weight_cpu, bias_cpu, 1, 1);
        nc_tensor_free(out);
    }
    result.cpu_time_ms = (get_time_ms() - start) / iters;
    
#ifdef NOCTA_CUDA_ENABLED
    if (nc_cuda_available()) {
        // Create tensors on GPU
        nc_tensor* input_gpu = nc_tensor_rand(in_shape, 4, NC_F32);
        nc_tensor* weight_gpu = nc_tensor_rand(w_shape, 4, NC_F32);
        nc_tensor* bias_gpu = nc_tensor_zeros(b_shape, 1, NC_F32);
        nc_tensor_to_device(input_gpu, NC_DEVICE_CUDA);
        nc_tensor_to_device(weight_gpu, NC_DEVICE_CUDA);
        nc_tensor_to_device(bias_gpu, NC_DEVICE_CUDA);
        
        // Warmup GPU
        for (int i = 0; i < warmup; i++) {
            nc_tensor* out = nc_conv2d_forward(input_gpu, weight_gpu, bias_gpu, 1, 1);
            nc_cuda_synchronize();
            nc_tensor_free(out);
        }
        
        // Benchmark GPU
        start = get_time_ms();
        for (int i = 0; i < iters; i++) {
            nc_tensor* out = nc_conv2d_forward(input_gpu, weight_gpu, bias_gpu, 1, 1);
            nc_cuda_synchronize();
            nc_tensor_free(out);
        }
        result.gpu_time_ms = (get_time_ms() - start) / iters;
        result.speedup = result.cpu_time_ms / result.gpu_time_ms;
        
        nc_tensor_free(input_gpu);
        nc_tensor_free(weight_gpu);
        nc_tensor_free(bias_gpu);
    }
#endif
    
    nc_tensor_free(input_cpu);
    nc_tensor_free(weight_cpu);
    nc_tensor_free(bias_cpu);
    
    return result;
}

// ============================================
// BatchNorm Benchmark
// ============================================

static benchmark_result bench_batchnorm(size_t batch, size_t C, size_t H, size_t W,
                                        int warmup, int iters) {
    benchmark_result result = {0};
    
    size_t in_shape[] = {batch, C, H, W};
    size_t c_shape[] = {C};
    
    // Create tensors on CPU
    nc_tensor* input_cpu = nc_tensor_rand(in_shape, 4, NC_F32);
    nc_tensor* gamma_cpu = nc_tensor_ones(c_shape, 1, NC_F32);
    nc_tensor* beta_cpu = nc_tensor_zeros(c_shape, 1, NC_F32);
    nc_tensor* rmean_cpu = nc_tensor_zeros(c_shape, 1, NC_F32);
    nc_tensor* rvar_cpu = nc_tensor_ones(c_shape, 1, NC_F32);
    
    // Warmup CPU
    for (int i = 0; i < warmup; i++) {
        nc_tensor* out = nc_batchnorm2d_forward_fn(input_cpu, rmean_cpu, rvar_cpu,
                                                    gamma_cpu, beta_cpu, false, 0.1, 1e-5);
        nc_tensor_free(out);
    }
    
    // Benchmark CPU
    double start = get_time_ms();
    for (int i = 0; i < iters; i++) {
        nc_tensor* out = nc_batchnorm2d_forward_fn(input_cpu, rmean_cpu, rvar_cpu,
                                                    gamma_cpu, beta_cpu, false, 0.1, 1e-5);
        nc_tensor_free(out);
    }
    result.cpu_time_ms = (get_time_ms() - start) / iters;
    
#ifdef NOCTA_CUDA_ENABLED
    if (nc_cuda_available()) {
        // Create tensors on GPU
        nc_tensor* input_gpu = nc_tensor_rand(in_shape, 4, NC_F32);
        nc_tensor* gamma_gpu = nc_tensor_ones(c_shape, 1, NC_F32);
        nc_tensor* beta_gpu = nc_tensor_zeros(c_shape, 1, NC_F32);
        nc_tensor* rmean_gpu = nc_tensor_zeros(c_shape, 1, NC_F32);
        nc_tensor* rvar_gpu = nc_tensor_ones(c_shape, 1, NC_F32);
        nc_tensor_to_device(input_gpu, NC_DEVICE_CUDA);
        nc_tensor_to_device(gamma_gpu, NC_DEVICE_CUDA);
        nc_tensor_to_device(beta_gpu, NC_DEVICE_CUDA);
        nc_tensor_to_device(rmean_gpu, NC_DEVICE_CUDA);
        nc_tensor_to_device(rvar_gpu, NC_DEVICE_CUDA);
        
        // Warmup GPU
        for (int i = 0; i < warmup; i++) {
            nc_tensor* out = nc_batchnorm2d_forward_fn(input_gpu, rmean_gpu, rvar_gpu,
                                                        gamma_gpu, beta_gpu, false, 0.1, 1e-5);
            nc_cuda_synchronize();
            nc_tensor_free(out);
        }
        
        // Benchmark GPU
        start = get_time_ms();
        for (int i = 0; i < iters; i++) {
            nc_tensor* out = nc_batchnorm2d_forward_fn(input_gpu, rmean_gpu, rvar_gpu,
                                                        gamma_gpu, beta_gpu, false, 0.1, 1e-5);
            nc_cuda_synchronize();
            nc_tensor_free(out);
        }
        result.gpu_time_ms = (get_time_ms() - start) / iters;
        result.speedup = result.cpu_time_ms / result.gpu_time_ms;
        
        nc_tensor_free(input_gpu);
        nc_tensor_free(gamma_gpu);
        nc_tensor_free(beta_gpu);
        nc_tensor_free(rmean_gpu);
        nc_tensor_free(rvar_gpu);
    }
#endif
    
    nc_tensor_free(input_cpu);
    nc_tensor_free(gamma_cpu);
    nc_tensor_free(beta_cpu);
    nc_tensor_free(rmean_cpu);
    nc_tensor_free(rvar_cpu);
    
    return result;
}

// ============================================
// Main
// ============================================

int main(void) {
    nc_init();
    
    printf("\n");
    printf("============================================================\n");
    printf("           NOCTA CPU vs GPU BENCHMARK                       \n");
    printf("============================================================\n");
    
#ifdef NOCTA_CUDA_ENABLED
    if (nc_cuda_available()) {
        nc_device_info info = nc_cuda_device_info(0);
        printf("\nGPU: %s\n", info.name);
        printf("Memory: %.1f GB\n", info.total_memory / (1024.0 * 1024 * 1024));
    } else {
        printf("\nCUDA compiled but no GPU available.\n");
    }
#else
    printf("\nCUDA not enabled. Showing CPU-only results.\n");
#endif
    
    int warmup = 3;
    int iters = 10;
    benchmark_result r;
    
    // ========================================
    // MatMul Benchmarks
    // ========================================
    printf("\n--- MATRIX MULTIPLICATION (GEMM) ---\n");
    printf("%-25s | %-18s | %-18s | Speedup\n", "Size", "CPU", "GPU");
    printf("----------------------------------------------------------------------\n");
    
    r = bench_matmul(512, 512, 512, warmup, iters);
    print_result("512x512 @ 512x512", &r);
    
    r = bench_matmul(1024, 1024, 1024, warmup, iters);
    print_result("1024x1024 @ 1024x1024", &r);
    
    r = bench_matmul(2048, 2048, 2048, warmup, iters);
    print_result("2048x2048 @ 2048x2048", &r);
    
    // ========================================
    // Conv2D Benchmarks
    // ========================================
    printf("\n--- CONVOLUTION 2D (3x3, stride=1, pad=1) ---\n");
    printf("%-25s | %-18s | %-18s | Speedup\n", "Input", "CPU", "GPU");
    printf("----------------------------------------------------------------------\n");
    
    r = bench_conv2d(16, 64, 56, 56, 64, 3, warmup, iters);
    print_result("16x64x56x56 -> 64", &r);
    
    r = bench_conv2d(16, 128, 28, 28, 128, 3, warmup, iters);
    print_result("16x128x28x28 -> 128", &r);
    
    r = bench_conv2d(32, 256, 14, 14, 256, 3, warmup, iters);
    print_result("32x256x14x14 -> 256", &r);
    
    // ========================================
    // BatchNorm Benchmarks
    // ========================================
    printf("\n--- BATCH NORMALIZATION (inference) ---\n");
    printf("%-25s | %-18s | %-18s | Speedup\n", "Input", "CPU", "GPU");
    printf("----------------------------------------------------------------------\n");
    
    r = bench_batchnorm(16, 64, 56, 56, warmup, iters);
    print_result("16x64x56x56", &r);
    
    r = bench_batchnorm(32, 128, 28, 28, warmup, iters);
    print_result("32x128x28x28", &r);
    
    r = bench_batchnorm(64, 256, 14, 14, warmup, iters);
    print_result("64x256x14x14", &r);
    
    printf("\n============================================================\n");
    printf("Benchmark complete!\n\n");
    
    nc_cleanup();
    return 0;
}
