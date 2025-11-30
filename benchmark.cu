#include "src/runner.cuh"
#include "src/kernels.cuh"
#include "src/gemm_cublas.cuh"
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Usage: %s <kernel_num> <M> <N> <K> [iterations]\n", argv[0]);
        printf("  kernel_num: 0=cuBLAS, 1=Naive\n");
        return 1;
    }
    
    int kernel_num = atoi(argv[1]);
    int M = atoi(argv[2]);
    int N = atoi(argv[3]);
    int K = atoi(argv[4]);
    int iterations = (argc > 5) ? atoi(argv[5]) : 100;
    
    CudaDeviceInfo();
    
    printf("Running: %s\n", get_kernel_name(kernel_num));
    printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Iterations: %d\n\n", iterations);
    
    // Allocate host memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_ref = (float*)malloc(size_C);  // For cuBLAS reference
    
    // Initialize matrices
    randomize_matrix_fp32(h_A, M * K);
    randomize_matrix_fp32(h_B, K * N);
    zero_init_matrix_fp32(h_C, M * N);
    zero_init_matrix_fp32(h_C_ref, M * N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_C_ref;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMalloc(&d_C_ref, size_C));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_ref, h_C_ref, size_C, cudaMemcpyHostToDevice));
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Compute cuBLAS reference (always)
    printf("Computing cuBLAS reference...\n");
    run_cublas_fp32(M, N, K, alpha, d_A, d_B, beta, d_C_ref);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Warmup
    printf("Warming up kernel...\n");
    for (int i = 0; i < 5; i++) {
        run_kernel(kernel_num, M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    printf("Benchmarking...\n");
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        run_kernel(kernel_num, M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));
    float ms_per_iter = ms_total / iterations;
    
    // Calculate TFLOPS: 2*M*N*K operations for GEMM
    double tflops = (2.0 * M * N * K) / (ms_per_iter * 1e9);

    // Calculate effective memory bandwidth (GB/s)
    // Reads: A(M*K) + B(K*N), Writes: C (M*N)
    double bytes_accessed = (double)(M * K + K * N + M * N) * sizeof(float);
    double bandwidth_gbs = bytes_accessed / (ms_per_iter * 1e6);
    
    // Verify against cuBLAS (skip if kernel_num == 0)
    bool verified = true;
    if (kernel_num != 0) {
        printf("\nVerifying against cuBLAS...\n");
        CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_C_ref, d_C_ref, size_C, cudaMemcpyDeviceToHost));
        verified = verify_matrix_fp32(h_C_ref, h_C, M * N, 0.01f);
    }
    
    // Results
    printf("\n========================================\n");
    printf("Results\n");
    printf("========================================\n");
    printf("Kernel:           %s\n", get_kernel_name(kernel_num));
    printf("Matrix Size:      %d x %d x %d\n", M, N, K);
    printf("Avg Time:         %.3f ms\n", ms_per_iter);
    printf("Performance:      %.2f TFLOPS\n", tflops);
    printf("Bandwidth:        %.2f GB/s\n", bandwidth_gbs);
    printf("Verification:     %s\n", verified ? "PASSED" : "FAILED");
    printf("========================================\n");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_ref));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    
    return verified ? 0 : 1;
}
