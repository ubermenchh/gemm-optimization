#include "kernels.cuh"
#include "runner.cuh"
#include "gemm_cublas.cuh"
#include <cmath>
#include <cstdio>

void CudaDeviceInfo() {
    int deviceId;
    CHECK_CUDA_ERROR(cudaGetDevice(&deviceId));

    cudaDeviceProp props{};
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&props, deviceId));

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "GPU Device Information" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Device ID: " << deviceId << std::endl;
    std::cout << "Name: " << props.name << std::endl;
    std::cout << "Compute Capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "Memory Bus Width: " << props.memoryBusWidth << " bits" << std::endl;
    std::cout << "Peak Memory Bandwidth: " 
              << 2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6 
              << " GB/s" << std::endl;
    std::cout << "Total Global Memory: " << props.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Shared Memory Per Block: " << props.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Max Threads Per Block: " << props.maxThreadsPerBlock << std::endl;
    std::cout << "Number of SMs: " << props.multiProcessorCount << std::endl;
    std::cout << "Warp Size: " << props.warpSize << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
}

void randomize_matrix_fp32(float* mat, int N) {
    struct timeval time {};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = static_cast<float>(rand() % 5) + 0.01f * static_cast<float>(rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.0f);
        mat[i] = tmp;
    }
}

void zero_init_matrix_fp32(float* mat, int N) {
    for (int i = 0; i < N; i++) {
        mat[i] = 0.0f;
    }
}

bool verify_matrix_fp32(float* matRef, float* matOut, int N, float tolerance) {
    double diff = 0.0;
    int errors = 0;
    int const max_errors_to_print = 10;
    
    for (int i = 0; i < N; i++) {
        diff = std::fabs(matRef[i] - matOut[i]);
        if (std::isnan(diff) || diff > tolerance) {
            if (errors < max_errors_to_print) {
                std::cerr << "Divergence at " << i 
                          << ": Expected " << matRef[i] 
                          << ", Got " << matOut[i] 
                          << " (Diff " << diff << ")" << std::endl;
            }
            errors++;
        }
    }
    
    if (errors > 0) {
        std::cerr << "Total errors: " << errors << " out of " << N 
                  << " elements (" << 100.0f * errors / N << "%)" << std::endl;
        return false;
    }
    return true;
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, 
                float* A, float* B, float beta, float* C,
                cudaStream_t stream) {
    switch (kernel_num) {
        case 0:
            run_cublas_fp32(M, N, K, alpha, A, B, beta, C, stream);
            break;
        case 1:
            launch_naive_gemm<float>(
                static_cast<size_t>(M), 
                static_cast<size_t>(N), 
                static_cast<size_t>(K),
                &alpha, A, B, &beta, C, stream
            );
            break;
        default:
            std::cerr << "Invalid kernel number: " << kernel_num << std::endl;
            std::exit(EXIT_FAILURE);
    }
}

const char* get_kernel_name(int kernel_num) {
    switch (kernel_num) {
        case 0: return "cuBLAS SGEMM (Gold Standard)";
        case 1: return "Naive GEMM (FP32)";
        default: return "Unknown";
    }
}
