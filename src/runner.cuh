#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// Utility functions
void CudaDeviceInfo();
void randomize_matrix_fp32(float *mat, int N);
void zero_init_matrix_fp32(float *mat, int N);
bool verify_matrix_fp32(float *matRef, float *matOut, int N, float tolerance = 0.01);

// cuBLAS reference
void run_cublas_fp32(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);

// Kernel runner functions
void run_kernel(int kernel_num, int M, int N, int K, float alpha, 
                float *A, float *B, float beta, float *C);

// Kernel names for display
const char* get_kernel_name(int kernel_num);
