#pragma once

#include <cublas_v2.h>
#include "runner.cuh"

inline void run_cublas_fp32(int M, int N, int K, float alpha, 
                            float* A, float* B, float beta, float* C,
                            cudaStream_t stream) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Set stream if provided
    if (stream != nullptr) {
        cublasSetStream(handle, stream);
    }
    
    // cuBLAS uses column-major, but we have row-major matrices
    // C = A * B  (row-major)
    // is equivalent to:
    // C^T = B^T * A^T  (column-major)
    // So we call: cublasSgemm(handle, N, M, K, B, A, C)
    
    cublasSgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,           // dimensions (swapped for row-major)
                &alpha,
                B, N,              // B^T with leading dim N
                A, K,              // A^T with leading dim K  
                &beta,
                C, N);             // C^T with leading dim N
    
    cublasDestroy(handle);
}
