#pragma once

#include <cuda_runtime.h>
#include "../runner.cuh"

// Matrix Shapes (Row-Major):
// A -> M x K
// B -> K x N
// C -> M x N
//
// C = alpha * A @ B + beta * C

#define TILE_SIZE 32

// template <typename T>
// __global__ void shared_mem_gemm(
//     size_t M, size_t N, size_t K,
//     T alpha,
//     T const* __restrict__ A,
//     T const* __restrict__ B,
//     T beta,
//     T* __restrict__ C
// ) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     __shared__ T A_s[TILE_SIZE][TILE_SIZE];
//     __shared__ T B_s[TILE_SIZE][TILE_SIZE];

//     T temp = static_cast<T>(0);
    
//     for (size_t k = 0; k < K; k += TILE_SIZE) {
//         size_t k_idx_A = k + threadIdx.x;
//         size_t k_idx_B = k + threadIdx.y; 

//         if (row < M && k_idx_A < K) {
//             A_s[threadIdx.y][threadIdx.x] = A[row * K + k_idx_A];
//         } else {
//             A_s[threadIdx.y][threadIdx.x] = static_cast<T>(0);
//         }

//         if (col < N && k_idx_B < K) {
//             B_s[threadIdx.y][threadIdx.x] = B[k_idx_B * N + col];
//         } else {
//             B_s[threadIdx.y][threadIdx.x] = static_cast<T>(0);
//         }
//         __syncthreads();

//         for (size_t k_tile = 0; k_tile < TILE_SIZE; k_tile++) {
//             temp += A_s[threadIdx.y][k_tile] * B_s[k_tile][threadIdx.x];
//         }
//         __syncthreads();
//     }

//     if (row < M && col < N) {
//         C[row * N + col] = alpha * temp + beta * C[row * N + col];
//     }
// }

template <typename T, size_t TILE_H, size_t TILE_W, size_t NUM_THREADS>
__device__ void load_tile(
    T const* __restrict__ src,
    size_t src_stride, 
    T dest[TILE_H][TILE_W],
    size_t tile_row_offset,
    size_t tile_col_offset,
    size_t M_limit,
    size_t N_limit,
    size_t thread_linear_idx
) {
    constexpr size_t NUM_ELEMENTS = TILE_H * TILE_W;
    // the num of elements each thread needs to load
    constexpr size_t LOADS_PER_THREADS = (NUM_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS;

    #pragma unroll
    for (size_t i = 0; i < LOADS_PER_THREADS; i++) {
        // what element to load
        size_t el_idx = thread_linear_idx + i * NUM_THREADS;

        // map element idx to 2d tile coordinates
        size_t row_s = el_idx / TILE_W;
        size_t col_s = el_idx % TILE_W;

        if (row_s < TILE_H && col_s < TILE_W) {
            size_t row_g = tile_row_offset + row_s;
            size_t col_g = tile_col_offset + col_s;

            T val = static_cast<T>(0);
            if (row_g < M_limit && col_g < N_limit) {
                val = src[row_g * src_stride + col_g];
            }

            dest[row_s][col_s] = val;
        }
    }
}

template <typename T, size_t BLOCK_M, size_t BLOCK_N, size_t BLOCK_K>
__global__ void shared_mem_gemm_v2(
    size_t M, size_t N, size_t K,
    T alpha,
    T const* __restrict__ A,
    T const* __restrict__ B,
    T beta,
    T* __restrict__ C
) {
    size_t thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    constexpr size_t NUM_THREADS = BLOCK_M * BLOCK_N;

    size_t const block_row = blockIdx.y * BLOCK_M;
    size_t const block_col = blockIdx.x * BLOCK_N;

    __shared__ T A_s[BLOCK_M][BLOCK_K];
    __shared__ T B_s[BLOCK_K][BLOCK_N];

    T sum = static_cast<T>(0);

    for (size_t k = 0; k < K; k += BLOCK_K) {
        // Load A tile
        load_tile<T, BLOCK_M, BLOCK_K, NUM_THREADS>(A, K, A_s, block_row, k, M, K, thread_linear_idx);
        // Load B tile
        load_tile<T, BLOCK_K, BLOCK_N, NUM_THREADS>(B, N, B_s, k, block_col, K, N, thread_linear_idx);

        __syncthreads();

        if (threadIdx.y < BLOCK_M && threadIdx.x < BLOCK_N) {
            #pragma unroll
            for (size_t i = 0; i < BLOCK_K; i++) {
                sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
            }
        }
        __syncthreads();
    }

    size_t global_row = block_row + threadIdx.y;
    size_t global_col = block_col + threadIdx.x;

    if (global_row < M && global_col < N) {
        size_t idx = global_row * N + global_col;
        C[idx] = alpha * sum + beta * C[idx];
    }
}

template <typename T>
void launch_shared_mem_gemm(
    size_t M, size_t N, size_t K,
    T const* alpha,
    T const* A,
    T const* B,
    T const* beta,
    T* C,
    cudaStream_t stream
) {
    constexpr size_t BM = 32;
    constexpr size_t BN = 32;
    constexpr size_t BK = 32;

    dim3 const block_dim(BN, BM, 1);
    dim3 const grid_dim(
        (static_cast<unsigned int>(N) + BN - 1) / BN,
        (static_cast<unsigned int>(M) + BM - 1) / BM
    );
    
    // shared_mem_gemm<T><<<grid_dim, block_dim, 0U, stream>>>(
        // M, N, K, *alpha, A, B, *beta, C
    // );

    shared_mem_gemm_v2<T, BM, BN, BK><<<grid_dim, block_dim, 0, stream>>>(
        M, N, K, *alpha, A, B, *beta, C
    );
    CHECK_LAST_CUDA_ERROR();
}

template void launch_shared_mem_gemm<float>(
    size_t M, size_t N, size_t K,
    float const* alpha, float const* A,
    float const* B, float const* beta,
    float* C, cudaStream_t stream
);