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

template <typename T>
__global__ void shared_mem_gemm(
    size_t M, size_t N, size_t K,
    T alpha,
    T const* __restrict__ A,
    T const* __restrict__ B,
    T beta,
    T* __restrict__ C
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ T A_s[TILE_SIZE][TILE_SIZE];
    __shared__ T B_s[TILE_SIZE][TILE_SIZE];

    T temp = static_cast<T>(0);
    
    for (size_t k = 0; k < K; k += TILE_SIZE) {
        size_t k_idx_A = k + threadIdx.x;
        size_t k_idx_B = k + threadIdx.y; 

        if (row < M && k_idx_A < K) {
            A_s[threadIdx.y][threadIdx.x] = A[row * K + k_idx_A];
        } else {
            A_s[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }

        if (col < N && k_idx_B < K) {
            B_s[threadIdx.y][threadIdx.x] = B[k_idx_B * N + col];
        } else {
            B_s[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }
        __syncthreads();

        for (size_t k_tile = 0; k_tile < TILE_SIZE; k_tile++) {
            temp += A_s[threadIdx.y][k_tile] * B_s[k_tile][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * temp + beta * C[row * N + col];
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
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(N) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(M) + block_dim.y - 1U) / block_dim.y,
        1U
    };
    
    shared_mem_gemm<T><<<grid_dim, block_dim, 0U, stream>>>(
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