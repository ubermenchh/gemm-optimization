#pragma once

#include <cuda_runtime.h>
#include "../runner.cuh"

// Matrix Shapes (Row-Major):
// A -> M x K
// B -> K x N
// C -> M x N
//
// C = alpha * A @ B + beta * C

template <typename T>
__global__ void naive_gemm_kernel(
    size_t M, size_t N, size_t K,
    T alpha,
    T const* __restrict__ A,
    T const* __restrict__ B,
    T beta,
    T* __restrict__ C
) {
    size_t const row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t const col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T temp = static_cast<T>(0);
        for (size_t i = 0; i < K; ++i) {
            temp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * temp + beta * C[row * N + col];
    }
}

template <typename T>
void launch_naive_gemm(
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
    
    naive_gemm_kernel<T><<<grid_dim, block_dim, 0U, stream>>>(
        M, N, K, *alpha, A, B, *beta, C
    );
    
    CHECK_LAST_CUDA_ERROR();
}

// Explicit template instantiations
template void launch_naive_gemm<float>(
    size_t M, size_t N, size_t K,
    float const* alpha, float const* A,
    float const* B, float const* beta,
    float* C, cudaStream_t stream
);
