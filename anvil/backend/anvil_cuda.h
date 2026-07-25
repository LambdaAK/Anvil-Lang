// anvil_cuda.h — runtime for programs compiled by `anvil cuda` (Anvil's CUDA backend).
//
// Every tensor lives in CUDA unified memory, so the host can read and write it directly: host code
// (control flow, printing, data loading) synchronizes before it touches memory, and each Anvil
// kernel is a __global__ function launched with one thread per output element.
//
// Compile with nvcc:                     nvcc -O3 -o prog prog.cu -lz
// or, without a GPU, as C++ on the CPU:  c++ -O2 -std=c++17 -DANVIL_EMULATE -x c++ prog.cu -o prog -lz
// (ANVIL_EMULATE runs each launch as a loop over its blocks and threads, one at a time.)
#pragma once
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>

// ---------------------------------------------------------------------------- CUDA or emulation
#ifdef ANVIL_EMULATE
#define __global__
#define __device__
#define __host__
struct anvil_dim3 { unsigned x, y, z; };
static anvil_dim3 blockIdx, threadIdx, blockDim, gridDim;
template <typename T> static inline T atomicAdd(T *p, T v) { T old = *p; *p = old + v; return old; }
static int anvil_alloc_mem(void **p, size_t n) { *p = calloc(1, n); return *p == NULL; }
#define ANVIL_LAUNCH(kernel, blocks, threads, ...) do {                                     \
        gridDim.x = (blocks); blockDim.x = (threads);                                       \
        for (unsigned b_ = 0; b_ < (unsigned)(blocks); b_++)                               \
            for (unsigned t_ = 0; t_ < (unsigned)(threads); t_++) {                        \
                blockIdx.x = b_; threadIdx.x = t_; kernel(__VA_ARGS__);                     \
            }                                                                               \
    } while (0)
static inline void anvil_sync(void) {}
// On the CPU one thread walks every element (the kernels' grid-stride loops allow any launch shape)
#define ANVIL_BLOCKS(n, tpb) 1
#define ANVIL_TPB(tpb) 1
#else
#include <cuda_runtime.h>
static int anvil_alloc_mem(void **p, size_t n) {
    if (cudaMallocManaged(p, n) != cudaSuccess) return 1;
    cudaMemset(*p, 0, n);
    return 0;
}
#define ANVIL_LAUNCH(kernel, blocks, threads, ...) kernel<<<(blocks), (threads)>>>(__VA_ARGS__)
// One thread per element, at most 2^20 blocks (the kernels' grid-stride loops cover the rest)
static inline unsigned anvil_blocks(long n, int tpb) {
    long b = (n + tpb - 1) / tpb;
    return (unsigned)(b < 1 ? 1 : b > (1L << 20) ? (1L << 20) : b);
}
#define ANVIL_BLOCKS(n, tpb) anvil_blocks((n), (tpb))
#define ANVIL_TPB(tpb) (tpb)
static inline void anvil_sync(void) {
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "anvil: CUDA error: %s\n", cudaGetErrorString(e)); exit(1); }
}
#endif

// Helpers that run in kernels and on the host
#define ANVIL_HD __host__ __device__

// ---------------------------------------------------------------------------- device helpers
// A failed bounds check inside a kernel: the first one is kept and reported by the host.
struct anvil_fault { int code; long long value; };
#ifdef ANVIL_EMULATE
static anvil_fault anvil_dev_fault;
#else
__device__ anvil_fault anvil_dev_fault;
#endif
static const char *anvil_fault_messages[256];

__device__ static inline void anvil_fail(int code, long long value) {
#ifdef ANVIL_EMULATE
    if (!anvil_dev_fault.code) { anvil_dev_fault.code = code; anvil_dev_fault.value = value; }
#else
    if (atomicCAS(&anvil_dev_fault.code, 0, code) == 0) anvil_dev_fault.value = value;
#endif
}

#include "anvil_host.h"

static void *anvil_alloc(size_t bytes) {
    void *p = NULL;
    if (bytes < 16) bytes = 16;
    if (anvil_alloc_mem(&p, bytes)) { fprintf(stderr, "anvil: out of memory (%zu bytes)\n", bytes); exit(1); }
    return p;
}

static void anvil_check_faults(void) {
    anvil_sync();
#ifdef ANVIL_EMULATE
    anvil_fault f = anvil_dev_fault;
#else
    anvil_fault f; cudaMemcpyFromSymbol(&f, anvil_dev_fault, sizeof f);
#endif
    if (f.code) {
        fflush(stdout);
        fprintf(stderr, "anvil: runtime error: %s (got %lld)\n", anvil_fault_messages[f.code], f.value);
        exit(1);
    }
}

