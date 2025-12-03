# GEMM Optimization

Optimizing General Matrix Multiplication (GEMM) kernels on NVIDIA GPUs. This project implements progressively optimized CUDA kernels and benchmarks them against cuBLAS.

## Benchmark Results

| Kernel | Matrix Size | Time (ms) | TFLOPS | Bandwidth (GB/s) | GPU |
|--------|-------------|-----------|--------|------------------|-----------|-----|
| cuBLAS SGEMM (Gold Standard) | 4096x4096 | 9.2 | 14.93 | 21.8 | A100 |
| Naive GEMM | 4096x4096 | 470.903 | 0.29 | 0.43 | A100 |
| Memory Coalesced GEMM | 4096x4096 | 46.52 | 2.95 | 4.33 | A100 |
| Shared Memory GEMM | 4096x4096 | 27.95 | 4.92 | 7.20 | A100 |

*A100 FP32 theoretical peak: 19.5 TFLOPS | Memory bandwidth: 2039 GB/s*

## Setup

### Option 1: Run on Modal

Modal provides on-demand access to A100/H100 GPUs without owning hardware.

**Prerequisites:**
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

**Installation:**

```bash
# Clone the repository
git clone https://github.com/yourusername/gemm-optimization.git
cd gemm-optimization

# Install dependencies
uv sync

# Authenticate with Modal (first time only)
uv run modal setup
```

**Running benchmarks:**

```bash
# Using Makefile (recommended)
make naive              # Run naive GEMM kernel
make cublas             # Run cuBLAS reference
make all                # Run all kernels

# With custom parameters
make naive SIZE=2048 ITERS=200

# Using modal directly
uv run modal run run_on_modal.py --kernel 1 --size 4096 --iterations 100

# List available kernels
make list
```

### Option 2: Run Locally (Requires NVIDIA GPU)

**Prerequisites:**
- NVIDIA GPU (Compute Capability 7.0+)
- CUDA Toolkit 11.0+ with nvcc
- cuBLAS library

**Compilation:**

```bash
# Compile the benchmark
nvcc -o gemm_benchmark \
    benchmark.cu \
    src/runner.cu \
    -std=c++17 \
    -arch=sm_80 \
    -O3 \
    -I. \
    -lcublas

# Adjust -arch flag for your GPU:
#   sm_70 = V100
#   sm_75 = T4, RTX 20xx
#   sm_80 = A100
#   sm_86 = RTX 30xx, A10G
#   sm_89 = RTX 40xx, L4, L40S
#   sm_90 = H100
```

## Available Kernels

| ID | Kernel | Description |
|----|--------|-------------|
| 0 | cuBLAS SGEMM | Reference implementation using NVIDIA cuBLAS |
| 1 | Naive GEMM | One thread per output element, no optimization |

## Makefile Targets

```bash
make help       # Show available commands
make list       # List all kernels
make cublas     # Run cuBLAS reference
make naive      # Run naive GEMM
make all        # Run all kernels sequentially
```

**Configuration variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| SIZE | 4096 | Matrix dimension (M=N=K) |
| ITERS | 100 | Benchmark iterations |
| GPU | A100 | Modal GPU type (T4, L4, A10G, A100, H100) |

## References

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
- [CUTLASS](https://github.com/NVIDIA/cutlass)

## License

MIT
