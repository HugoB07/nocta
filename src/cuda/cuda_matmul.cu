#ifdef NOCTA_CUDA_ENABLED

#include "nocta/cuda/cuda_kernels.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

// Shared cuBLAS handle - accessible from other CUDA files
cublasHandle_t g_cublas_handle = NULL;

extern "C" void ensure_cublas(void) {
    if (g_cublas_handle == NULL) {
        cublasStatus_t status = cublasCreate(&g_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[cuBLAS ERROR] Failed to create handle\n");
        }
    }
}

// ============================================
// Matrix Multiplication using cuBLAS
// ============================================
// C = alpha * A @ B + beta * C
// A is M x K, B is K x N, C is M x N
// cuBLAS uses column-major, so we compute C^T = B^T @ A^T

extern "C" void nc_cuda_matmul_f32(float* C, const float* A, const float* B,
                                   int M, int N, int K,
                                   float alpha, float beta,
                                   int transA, int transB) {
    ensure_cublas();
    
    // Logic:
    // C (RowMajor) = C^T (ColMajor).
    // Formula: C^T = B^T @ A^T.
    // Inputs: A (MxK), B (KxN).
    // Sgemm args: Mat1 (B), Mat2 (A).
    
    // Handle B (Mat1):
    // If B RowMajor: Memory view is B^T (NxK). OpN -> B^T. ldb = N.
    // If B ColMajor: Memory view is B (KxN). OpT -> B^T. ldb = K.
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int ldb = transB ? K : N;
    
    // Handle A (Mat2):
    // If A RowMajor: Memory view is A^T (KxM). OpN -> A^T. lda = K.
    // If A ColMajor: Memory view is A (MxK). OpT -> A^T. lda = M.
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = transA ? M : K;
    
    cublasStatus_t status = cublasSgemm(
        g_cublas_handle,
        opB,
        opA,
        N,                  // rows of Result^T (ColMajor) = N
        M,                  // cols of Result^T (ColMajor) = M
        K,                  // inner dim
        &alpha,
        B, ldb,
        A, lda,
        &beta,
        C, N                // C (RowMajor) view as C^T (ColMajor, NxM). ld = N.
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[cuBLAS ERROR] SGEMM failed: %d\n", status);
    }
}

extern "C" void nc_cuda_matmul_f64(double* C, const double* A, const double* B,
                                   int M, int N, int K,
                                   double alpha, double beta,
                                   int transA, int transB) {
    ensure_cublas();
    
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int ldb = transB ? K : N;
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = transA ? M : K;
    
    cublasStatus_t status = cublasDgemm(
        g_cublas_handle,
        opB,
        opA,
        N, M, K,
        &alpha,
        B, ldb,
        A, lda,
        &beta,
        C, N
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[cuBLAS ERROR] DGEMM failed: %d\n", status);
    }
}

// Cleanup function (called on library unload)
__attribute__((destructor))
static void cleanup_cublas(void) {
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = NULL;
    }
}

#endif // NOCTA_CUDA_ENABLED
