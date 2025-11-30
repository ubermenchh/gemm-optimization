#include <cuda_runtime.h>

// Matrix Shapes:
// A -> M x K
// B -> K x N
// C -> M x N

template <typename T>
__global__ void naive_gemm(
    const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, 
    size_t M, size_t N, size_t K, T alpha, T beta
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float temp = 0.0f;
        for (int i = 0; i < K; i++) {
            temp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * temp + beta * C[row * N + col];
    }
}