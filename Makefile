.PHONY: naive mem-coalesce cublas all list clean help

SIZE ?= 4096
ITERS ?= 100
GPU ?= A100

help:
	@echo "GEMM Optimization Benchmark"
	@echo ""
	@echo "Usage: make <target> [SIZE=N] [ITERS=N] [GPU=type]"
	@echo ""
	@echo "Targets:"
	@echo "  cublas      Run cuBLAS reference (kernel 0)"
	@echo "  naive       Run naive GEMM (kernel 1)"
	@echo "  mem-coalesce Run mem coalesced GEMM (kernel 2)"
	@echo "  all         Run all kernels sequentially"
	@echo "  list        List available kernels"
	@echo ""
	@echo "Options:"
	@echo "  SIZE=4096   Matrix size (M=N=K)"
	@echo "  ITERS=100   Benchmark iterations"
	@echo "  GPU=A100    GPU type (T4, L4, A10G, A100, H100)"
	@echo ""
	@echo "Examples:"
	@echo "  make naive"
	@echo "  make naive SIZE=2048"
	@echo "  make cublas SIZE=8192 ITERS=200"
	@echo "  make all GPU=H100"

list:
	@uv run modal run run_on_modal.py --list-kernels

cublas:
	@uv run modal run run_on_modal.py --kernel 0 --size $(SIZE) --iterations $(ITERS)

naive:
	@uv run modal run run_on_modal.py --kernel 1 --size $(SIZE) --iterations $(ITERS)

mem-coalesce:
	@uv run modal run run_on_modal.py --kernel 2 --size $(SIZE) --iterations $(ITERS)

all: cublas naive mem-coalesce