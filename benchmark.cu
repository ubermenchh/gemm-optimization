#include "src/runner.cuh"
#include "src/kernels.cuh"
#include "src/gemm_cublas.cuh"
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <kernel_num> <M> <N> <K> [iterations]" << std::endl;
        std::cout << "  kernel_num: 0=cuBLAS, 1=Naive" << std::endl;
        return 1;
    }
    
    int const kernel_num = std::atoi(argv[1]);
    int const M = std::atoi(argv[2]);
    int const N = std::atoi(argv[3]);
    int const K = std::atoi(argv[4]);
    int const iterations = (argc > 5) ? std::atoi(argv[5]) : 100;
    
    CudaDeviceInfo();
    
    std::cout << "Running: " << get_kernel_name(kernel_num) << std::endl;
    std::cout << "Matrix size: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Iterations: " << iterations << std::endl << std::endl;
    
    // Allocate host memory
    size_t const size_A = static_cast<size_t>(M) * K * sizeof(float);
    size_t const size_B = static_cast<size_t>(K) * N * sizeof(float);
    size_t const size_C = static_cast<size_t>(M) * N * sizeof(float);
    
    float* h_A = static_cast<float*>(std::malloc(size_A));
    float* h_B = static_cast<float*>(std::malloc(size_B));
    float* h_C = static_cast<float*>(std::malloc(size_C));
    float* h_C_ref = static_cast<float*>(std::malloc(size_C));
    
    // Initialize matrices
    randomize_matrix_fp32(h_A, M * K);
    randomize_matrix_fp32(h_B, K * N);
    zero_init_matrix_fp32(h_C, M * N);
    zero_init_matrix_fp32(h_C_ref, M * N);
    
    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    float* d_C_ref;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, size_A));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, size_B));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, size_C));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C_ref, size_C));
    
    // Copy to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_C_ref, h_C_ref, size_C, cudaMemcpyHostToDevice));
    
    float const alpha = 1.0f;
    float const beta = 0.0f;
    
    // Create stream for kernel execution
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    // Compute cuBLAS reference (always)
    std::cout << "Computing cuBLAS reference..." << std::endl;
    run_cublas_fp32(M, N, K, alpha, d_A, d_B, beta, d_C_ref, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    
    // Warmup
    std::cout << "Warming up kernel..." << std::endl;
    for (int i = 0; i < 5; ++i) {
        run_kernel(kernel_num, M, N, K, alpha, d_A, d_B, beta, d_C, stream);
    }
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    
    // Benchmark
    std::cout << "Benchmarking..." << std::endl;
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) {
        run_kernel(kernel_num, M, N, K, alpha, d_A, d_B, beta, d_C, stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float ms_total = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms_total, start, stop));
    float const ms_per_iter = ms_total / static_cast<float>(iterations);
    
    // Calculate TFLOPS: 2*M*N*K operations for GEMM
    double const tflops = (2.0 * M * N * K) / (ms_per_iter * 1e9);

    // Calculate effective memory bandwidth (GB/s)
    // Reads: A(M*K) + B(K*N), Writes: C(M*N)
    double const bytes_accessed = static_cast<double>(M * K + K * N + M * N) * sizeof(float);
    double const bandwidth_gbs = bytes_accessed / (ms_per_iter * 1e6);
    
    // Verify against cuBLAS (skip if kernel_num == 0)
    bool verified = true;
    if (kernel_num != 0) {
        std::cout << std::endl << "Verifying against cuBLAS..." << std::endl;
        CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_C_ref, d_C_ref, size_C, cudaMemcpyDeviceToHost));
        verified = verify_matrix_fp32(h_C_ref, h_C, M * N, 0.01f);
    }
    
    // Results
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Kernel:           " << get_kernel_name(kernel_num) << std::endl;
    std::cout << "Matrix Size:      " << M << " x " << N << " x " << K << std::endl;
    std::cout << "Avg Time:         " << ms_per_iter << " ms" << std::endl;
    std::cout << "Performance:      " << tflops << " TFLOPS" << std::endl;
    std::cout << "Bandwidth:        " << bandwidth_gbs << " GB/s" << std::endl;
    std::cout << "Verification:     " << (verified ? "PASSED" : "FAILED") << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaFree(d_C_ref));
    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
    std::free(h_C_ref);
    
    return verified ? 0 : 1;
}
