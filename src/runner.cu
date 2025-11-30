#include "kernels.cuh"
#include "runner.cuh"
#include "gemm_cublas.cuh"
#include <cmath>
#include <cstdio>

void CudaDeviceInfo() {
    int deviceId;
    CUDA_CHECK(cudaGetDevice(&deviceId));

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));

    printf("\n");
    printf("========================================\n");
    printf("GPU Device Information\n");
    printf("========================================\n");
    printf("Device ID: %d\n", deviceId);
    printf("Name: %s\n", props.name);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);
    printf("Memory Bus Width: %d bits\n", props.memoryBusWidth);
    printf("Peak Memory Bandwidth: %.2f GB/s\n", 
           2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6);
    printf("Total Global Memory: %zu MB\n", props.totalGlobalMem / 1024 / 1024);
    printf("Shared Memory Per Block: %zu KB\n", props.sharedMemPerBlock / 1024);
    printf("Max Threads Per Block: %d\n", props.maxThreadsPerBlock);
    printf("Number of SMs: %d\n", props.multiProcessorCount);
    printf("Warp Size: %d\n", props.warpSize);
    printf("========================================\n\n");
}

void randomize_matrix_fp32(float *mat, int N) {
    struct timeval time {};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

void zero_init_matrix_fp32(float *mat, int N) {
    for (int i = 0; i < N; i++) {
        mat[i] = 0.0f;
    }
}

bool verify_matrix_fp32(float *matRef, float *matOut, int N, float tolerance) {
    double diff = 0.0;
    int errors = 0;
    const int max_errors_to_print = 10;
    
    for (int i = 0; i < N; i++) {
        diff = std::fabs(matRef[i] - matOut[i]);
        if (std::isnan(diff) || diff > tolerance) {
            if (errors < max_errors_to_print) {
                printf("Divergence at %d: Expected %5.2f, Got %5.2f (Diff %5.2f)\n",
                       i, matRef[i], matOut[i], diff);
            }
            errors++;
        }
    }
    
    if (errors > 0) {
        printf("Total errors: %d out of %d elements (%.2f%%)\n", 
               errors, N, 100.0f * errors / N);
        return false;
    }
    return true;
}

void run_naive_gemm_fp32(int M, int N, int K, float alpha, 
                         float *A, float *B, float beta, float *C) {
    dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32), 1);
    dim3 blockDim(32, 32, 1);
    naive_gemm<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, 
                float *A, float *B, float beta, float *C) {
    switch (kernel_num) {
        case 0:
            run_cublas_fp32(M, N, K, alpha, A, B, beta, C);
            break;
        case 1:
            run_naive_gemm_fp32(M, N, K, alpha, A, B, beta, C);
            break;
        default:
            printf("Invalid kernel number: %d\n", kernel_num);
            exit(1);
    }
}

const char* get_kernel_name(int kernel_num) {
    switch (kernel_num) {
        case 0: return "cuBLAS SGEMM (Gold Standard)";
        case 1: return "Naive GEMM (FP32)";
        default: return "Unknown";
    }
}
