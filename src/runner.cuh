#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>

// Error checking macros
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t err, char const* func, char const* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
inline void checkLast(char const* file, int line) {
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Legacy macro for compatibility
#define CUDA_CHECK(call) CHECK_CUDA_ERROR(call)

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// Utility functions
void CudaDeviceInfo();
void randomize_matrix_fp32(float* mat, int N);
void zero_init_matrix_fp32(float* mat, int N);
bool verify_matrix_fp32(float* matRef, float* matOut, int N, float tolerance = 0.01f);

// Kernel launcher declarations
template <typename T>
void launch_naive_gemm(size_t M, size_t N, size_t K, 
                       T const* alpha, T const* A,
                       T const* B, T const* beta,
                       T* C, cudaStream_t stream);

// cuBLAS reference
void run_cublas_fp32(int M, int N, int K, float alpha, 
                     float* A, float* B, float beta, float* C,
                     cudaStream_t stream = nullptr);

// Kernel runner (simplified interface for benchmarking)
void run_kernel(int kernel_num, int M, int N, int K, float alpha, 
                float* A, float* B, float beta, float* C,
                cudaStream_t stream = nullptr);

// Kernel names for display
const char* get_kernel_name(int kernel_num);
