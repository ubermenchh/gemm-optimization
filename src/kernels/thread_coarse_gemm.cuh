#pragma once

#include <cuda_runtime.h>
#include "../runner.cuh"

// Matrix Shapes (Row-Major):
// A -> M x K
// B -> K x N
// C -> M x N
//
// C = alpha * A @ B + beta * C

template <typename T, size_t BLOCK_M, size_t BLOCK_N, size_t BLOCK_K, size_t THREAD_TILE>
__global__ void thread_coarse_gemm(
    size_t M, size_t N, size_t K,
    T alpha,
    T const* __restrict__ A,
    T const* __restrict__ B,
    T beta,
    T* __restrict__ C
) {
    constexpr size_t NUM_THREADS = BLOCK_M * BLOCK_N / THREAD_TILE;
    size_t thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;

    size_t const block_row = blockIdx.y * BLOCK_M;
    size_t const block_col = blockIdx.x * BLOCK_N;

    __shared__ T A_s[BLOCK_M][BLOCK_K];
    __shared__ T B_s[BLOCK_K][BLOCK_N];

    // Register File Accumulators
    // each thread computes one col but THREAD_TILE rows
    T thread_results[THREAD_TILE] = {static_cast<T>(0)};

    for (size_t k = 0; k < K; k += BLOCK_K) {
        // load A
        load_tile<T, BLOCK_M, BLOCK_K, NUM_THREADS>(A, K, A_s, block_row, k, M, K, thread_linear_idx);
        // load B
        load_tile<T, BLOCK_K, BLOCK_N, NUM_THREADS>(B, K, B_s, k, block_col, K, N, thread_linear_idx);
        __syncthreads();

        size_t t_col = thread_linear_idx % BLOCK_N;
        size_t t_row_start = (thread_linear_idx / BLOCK_N) * THREAD_TILE;

        #pragma unroll
        for (size_t k_i = 0; k_i < BLOCK_K; k_i++) {
            T b_val = B_s[k_i][t_col];

            #pragma unroll
            for (size_t i = 0; i < THREAD_TILE; i++) {
                thread_results[i] += A_s[t_row_start + i][k_i] * b_val; 
            }
        }
        __syncthreads();
    }

    size_t t_col = thread_linear_idx % BLOCK_N;
    size_t t_row_start = (thread_linear_idx / BLOCK_N) * THREAD_TILE;
    size_t global_col = block_col + t_col;

    #pragma unroll
    for (size_t i = 0; i < THREAD_TILE; i++) {
        size_t row_in_block = t_row_start + i;
        size_t global_row = block_row + row_in_block;

        if (global_row < M && global_col < N) {
            size_t idx = global_row * N + global_col;
            C[idx] = alpha * thread_results[i] + beta * C[idx];
        }
    }
}

template <typename T>
void launch_thread_coarse_gemm(
    size_t M, size_t N, size_t K,
    T const* alpha,
    T const* A,
    T const* B,
    T const* beta,
    T* C,
    cudaStream_t stream
) {
    constexpr size_t BM = 64;
    constexpr size_t BN = 64;
    constexpr size_t BK = 8;
    constexpr size_t THREAD_TILE = 8;

    constexpr size_t NUM_THREADS = (BM * BN) / THREAD_TILE;

    dim3 block_dim(NUM_THREADS, 1, 1);
    dim3 grid_dim(
        (static_cast<unsigned int>(N) + BN - 1) / BN,
        (static_cast<unsigned int>(M) + BM - 1) / BM
    );

    thread_coarse_gemm<T, BM, BN, BK, THREAD_TILE><<<grid_dim, block_dim, 0, stream>>>(M, N, K, *alpha, A, B, *beta, C);
    CHECK_LAST_CUDA_ERROR();
}

template void launch_thread_coarse_gemm<float>(
    size_t M, size_t N, size_t K,
    float const* alpha, float const* A,
    float const* B, float const* beta,
    float* C, cudaStream_t stream
);