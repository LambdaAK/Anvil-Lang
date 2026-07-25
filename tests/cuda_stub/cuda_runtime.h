// Just enough of the CUDA runtime API to type-check generated .cu files with clang's CUDA front
// end (clang -x cuda -nocudainc) on a machine without the CUDA toolkit. clang enforces the
// __host__/__device__ calling rules, so this catches a host function called from a kernel.
#pragma once
#include <stddef.h>
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))

typedef int cudaError_t;
enum { cudaSuccess = 0 };
struct dim3 {
    unsigned x, y, z;
    __host__ __device__ dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};
struct uint3 { unsigned x, y, z; };
__device__ uint3 anvil_stub_index();
#define blockIdx (anvil_stub_index())
#define threadIdx (anvil_stub_index())
#define blockDim (anvil_stub_index())
#define gridDim (anvil_stub_index())

cudaError_t cudaMallocManaged(void **p, size_t n, unsigned flags = 1);
cudaError_t cudaMemset(void *p, int v, size_t n);
cudaError_t cudaDeviceSynchronize(void);
const char *cudaGetErrorString(cudaError_t e);
template <class T> cudaError_t cudaMemcpyFromSymbol(void *dst, const T &sym, size_t n, size_t off = 0, int kind = 2);
extern "C" cudaError_t cudaConfigureCall(dim3 grid, dim3 block, size_t shared = 0, void *stream = 0);

__device__ float atomicAdd(float *p, float v);
__device__ int atomicAdd(int *p, int v);
__device__ int atomicCAS(int *p, int compare, int v);

// the device versions of the math library
__device__ float expf(float); __device__ float logf(float); __device__ float sqrtf(float);
__device__ float tanhf(float); __device__ float sinf(float); __device__ float cosf(float);
__device__ float fabsf(float); __device__ float floorf(float); __device__ float ceilf(float);
__device__ float rintf(float); __device__ float truncf(float); __device__ float powf(float, float);
__device__ float fmodf(float, float); __device__ int abs(int);
