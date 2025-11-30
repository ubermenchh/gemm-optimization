"""
Modal script to benchmark CUDA kernels on Modal GPUs.
Usage:
    modal run run_on_modal.py                             # Run cuBLAS reference
    modal run run_on_modal.py --kernel 1                  # Run naive GEMM
    modal run run_on_modal.py --size 2048                 # Custom matrix size
    modal run run_on_modal.py --list-kernels              # List available kernels
"""

import os
import sys
import modal

app = modal.App("gemm-optimization")
image = modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04",
    add_python="3.11"
).entrypoint([])

KERNEL_NAMES = {
    0: "cuBLAS SGEMM (Gold Standard)",
    1: "Naive GEMM (FP32)"
}

@app.function(gpu="A100", image=image, timeout=600)
def run_benchmark(source_files: dict, kernel_num: int, M: int, N: int, K: int, iterations: int):
    import subprocess
    import os

    os.makedirs("src/kernels", exist_ok=True)

    for filepath, content in source_files.items():
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)

    print("-" * 60)
    print("Source files written:")
    print("-" * 60)
    for filepath in sorted(source_files.keys()):
        print(f" {filepath}")
    print()

    subprocess.run(["nvidia-smi"], check=True)

    print("\n" + "-" * 60)
    print("Compiling...")
    print("-" * 60)

    try:
        result = subprocess.run(
            [
                "nvcc",
                "-o", "gemm_benchmark",
                "benchmark.cu",
                "src/runner.cu",
                "-std=c++17",
                "-arch=sm_80",
                "-O3",
                "-I.",
                "-lcublas"
            ],
            capture_output=True,
            text=True,
            check=True
        )
        print("Compilation successful!")
        if result.stderr:
            print("Warnings:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Compilation failed!")
        print("stdout:", e.stdout)
        print("stderr", e.stderr)
        return

    print("\n" + "-" * 60)
    print("Running benchmark...")
    print("-" * 60)

    try:
        result = subprocess.run(
            ["./gemm_benchmark", str(kernel_num), str(M), str(N), str(K), str(iterations)],
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Execution failed!")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)

@app.local_entrypoint()
def main(kernel: int=1, size: int=4096, iterations: int=100, list_kernels: bool=False):
    if list_kernels:
        print("\nAvailable Kernels:")
        print("--------------------")
        for num, name in KERNEL_NAMES.items():
            print(f"  {num}: {name}")
        return

    if kernel not in KERNEL_NAMES:
        print(f"Error: Invalid kernel number {kernel}")
        print("Use --list-kernels to available kernels")
        sys.exit(1)

    required_files = [
        "benchmark.cu",
        "src/runner.cu",
        "src/runner.cuh",
        "src/kernels.cuh",
        "src/gemm_cublas.cuh",
        "src/kernels/naive_gemm.cuh",
    ]
    
    for filepath in required_files:
        if not os.path.exists(filepath):
            print(f"Error: Required file '{filepath}' not found!")
            sys.exit(1)
    
    print(f"\nKernel:      {KERNEL_NAMES[kernel]}")
    print(f"Matrix Size: {size}x{size}x{size}")
    print(f"Iterations:  {iterations}")
    print("=" * 60)
    
    source_files = {}
    for filepath in required_files:
        with open(filepath) as f:
            source_files[filepath] = f.read()
    
    print("Uploading and running on Modal GPU...")
    run_benchmark.remote(source_files, kernel, size, size, size, iterations)