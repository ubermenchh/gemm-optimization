#pragma once

#include <cuda_runtime.h>
#include "../runner.cuh"


// Matrix Shapes (Row-Major):
// A -> M x K
// B -> K x N
// C -> M x N
//
// C = alpha * A @ B + beta * C

template <typename T, size_t BLOCK_M, size_t BLOCK_N, size_t BLOCK_K, size_t THREAD_M, size_t THREAD_N>
__global__ void block_tiling_2d_gemm(
    size_t M, size_t N, size_t K,
    T alpha,
    T const* __restrict__ A,
    T const* __restrict__ B,
    T beta,
    T* __restrict__ C
) {
    constexpr size_t NUM_THREADS = (BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N);
    size_t thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;

    size_t block_row = blockIdx.y * BLOCK_M;
    size_t block_col = blockIdx.x * BLOCK_N;

    __shared__ T A_s[BLOCK_M][BLOCK_K];
    __shared__ T B_s[BLOCK_K][BLOCK_N];

    // Registers for C
    // each thread calculates a TM x TN sub-tile
    T thread_results[THREAD_M][THREAD_N] = {static_cast<T>(0)};

    // registers for A and B fragments
    T reg_A[THREAD_M] = {static_cast<T>(0)};
    T reg_B[THREAD_N] = {static_cast<T>(0)};

    for (size_t k = 0; k < K; k += BLOCK_K) {
        // 1. Load data from Global Memory to Shared Memory

        // load A
        load_tile<T, BLOCK_M, BLOCK_K, NUM_THREADS>(A, K, A_s, block_row, k, M, K, thread_linear_idx);
        // load B
        load_tile<T, BLOCK_K, BLOCK_N, NUM_THREADS>(B, N, B_s, k, block_col, K, N, thread_linear_idx);
        __syncthreads();   

        // 2. Computer Innter Loop

        constexpr size_t threads_per_row = BLOCK_N / THREAD_N;

        const size_t row_idx = (thread_linear_idx / threads_per_row) * THREAD_M; // row offset in A_s
        const size_t col_idx = (thread_linear_idx % threads_per_row) * THREAD_N; // col offset in B_s

        #pragma unroll
        for (size_t dot_idx = 0; dot_idx < BLOCK_K; dot_idx++) {
            // load A fragments into registers
            #pragma unroll
            for (size_t i = 0; i < THREAD_M; i++) {
                reg_A[i] = A_s[row_idx + i][dot_idx];
            }

            // load B fragments into registers
            #pragma unroll
            for (size_t j = 0; j < THREAD_N; j++) {
                reg_B[j] = B_s[dot_idx][col_idx + j];
            }

            // outer product accumulation
            #pragma unroll
            for (size_t i = 0; i < THREAD_M; i++) {
                #pragma unroll
                for (size_t j = 0; j < THREAD_N; j++) {
                    thread_results[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
        __syncthreads();
    }

    // 3. Write results to Global Memory
    constexpr size_t threads_per_row = BLOCK_N / THREAD_N;
    const size_t row_idx = (thread_linear_idx / threads_per_row) * THREAD_M;
    const size_t col_idx = (thread_linear_idx % threads_per_row) * THREAD_N;

    #pragma unroll
    for (size_t i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (size_t j = 0; j < THREAD_N; j++) {
            size_t global_row = block_row + row_idx + i;
            size_t global_col = block_col + col_idx + j;

            if (global_row < M && global_col < N) {
                size_t idx = global_row * N + global_col;
                C[idx] = alpha * thread_results[i][j] + beta * C[idx];
            }
        }
    }
}

template <typename T>
void launch_2d_block_tiling_gemm(
    size_t M, size_t N, size_t K,
    T const* alpha,
    T const* A,
    T const* B,
    T const* beta,
    T* C,
    cudaStream_t stream
) {
    constexpr size_t BM = 128;
    constexpr size_t BN = 128;
    constexpr size_t BK = 8;
    constexpr size_t TM = 8;
    constexpr size_t TN = 8;

    constexpr size_t NUM_THREADS = (BM * BN) / (TM * TN);

    dim3 block_dim(NUM_THREADS, 1, 1);
    dim3 grid_dim(
        (static_cast<unsigned int>(N) + BN - 1) / BN,
        (static_cast<unsigned int>(M) + BM - 1) / BM
    );

    block_tiling_2d_gemm<T, BM, BN, BK, TM, TN><<<grid_dim, block_dim, 0, stream>>>(
        M, N, K, *alpha, A, B, *beta, C
    );
    CHECK_LAST_CUDA_ERROR();
}

template void launch_2d_block_tiling_gemm<float>(
    size_t M, size_t N, size_t K,
    float const* alpha, float const* A,
    float const* B, float const* beta,
    float* C, cudaStream_t stream
);