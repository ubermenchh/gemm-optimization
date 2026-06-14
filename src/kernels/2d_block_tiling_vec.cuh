#pragma once

#include <cuda_runtime.h>
#include "../runner.cuh"

// Matrix Shapes (Row-Major):
// A -> M x K
// B -> K x N
// C -> M x N
//
// C = alpha * A @ B + beta * C

template <typename T, size_t BLOCK_M, size_t BLOCK_N, size_t BLOCK_K, size_t NUM_THREADS>
__device__ void load_data_to_shared_memory_vectorized(
    T const* A, size_t stride_A,            // A is M x K, 
    T const* B, size_t stride_B,            // B is K x N,
    T A_transposed[BLOCK_K][BLOCK_M],       // transposed
    T B_s[BLOCK_K][BLOCK_N],                // normal layout
    size_t tile_idx,                        // which tile
    size_t thread_linear_idx,               // threadIdx.y * blockDim.x + threadIdx.x
    size_t M, size_t N, size_t K            // matrix dimensions for bounds checking
) {
    // constants
    constexpr size_t VEC_SIZE = 4; // flaots per int4
    constexpr size_t VECS_PER_A_ROW = BLOCK_K / VEC_SIZE;
    constexpr size_t VECS_PER_B_ROW = BLOCK_N / VEC_SIZE;

    constexpr size_t TOTAL_A_VECS = BLOCK_M * VECS_PER_A_ROW;
    constexpr size_t TOTAL_B_VECS = BLOCK_K * VECS_PER_B_ROW;

    constexpr size_t A_LOADS_PER_THREAD = (TOTAL_A_VECS + NUM_THREADS - 1) / NUM_THREADS;
    constexpr size_t B_LOADS_PER_THREAD = (TOTAL_B_VECS + NUM_THREADS - 1) / NUM_THREADS;

    // Load A (transposed)
    #pragma unroll
    for (size_t load_idx = 0; load_idx < A_LOADS_PER_THREAD; load_idx++) {
        size_t vec_idx = thread_linear_idx + load_idx * NUM_THREADS;

        size_t tile_row = vec_idx / VECS_PER_A_ROW;
        size_t tile_col = (vec_idx % VECS_PER_A_ROW) * VEC_SIZE;

        size_t global_row = blockidx.x * BLOCK_M + tile_row;
    }
}