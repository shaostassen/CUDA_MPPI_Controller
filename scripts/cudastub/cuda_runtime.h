// Minimal stand-in for CUDA's cuda_runtime.h, used ONLY to syntax-check the
// project's .cu/.cuh sources with clang on a machine with no CUDA toolkit.
// Signatures mirror the real API, including the C++ template overloads that
// let cudaMalloc(&T_ptr, n) compile. A clean run here is not proof that nvcc
// will accept the code — it is a fast check for our own mistakes.
#pragma once

#include <cstddef>

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ inline

struct __mppi_uint3 {
  unsigned x, y, z;
};
__attribute__((device)) extern __mppi_uint3 threadIdx, blockIdx, blockDim, gridDim;
__attribute__((device)) extern const int warpSize;

// Device intrinsics the project uses. Real CUDA supplies these through its
// device-function headers.
__attribute__((device)) void __syncthreads();
__attribute__((device)) float __shfl_down_sync(unsigned mask, float var, unsigned delta,
                                               int width = 32);
__attribute__((device)) double __shfl_down_sync(unsigned mask, double var, unsigned delta,
                                                int width = 32);
__attribute__((device)) int __shfl_down_sync(unsigned mask, int var, unsigned delta,
                                             int width = 32);

// Device overloads of the math functions the project uses. nvcc supplies
// these via its own math headers; with -nocudainc clang sees only the host
// declarations, so declare device counterparts (CUDA permits host/device
// overloading) to get past the math boundary and reach the real code.
extern "C" {
__attribute__((device)) float sqrtf(float);
__attribute__((device)) float logf(float);
__attribute__((device)) float sinf(float);
__attribute__((device)) float cosf(float);
__attribute__((device)) float expf(float);
__attribute__((device)) float fabsf(float);
__attribute__((device)) double sqrt(double);
__attribute__((device)) double log(double);
__attribute__((device)) double sin(double);
__attribute__((device)) double cos(double);
__attribute__((device)) double exp(double);
__attribute__((device)) double fabs(double);
}
__attribute__((device)) inline float sin(float x) { return sinf(x); }
__attribute__((device)) inline float cos(float x) { return cosf(x); }
__attribute__((device)) inline float sqrt(float x) { return sqrtf(x); }
__attribute__((device)) inline float fabs(float x) { return fabsf(x); }
namespace std {
using ::cos;
using ::fabs;
using ::sin;
using ::sqrt;
}  // namespace std

typedef enum cudaError { cudaSuccess = 0, cudaErrorUnknown = 999 } cudaError_t;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
};

#define cudaHostAllocDefault 0x00
#define cudaHostAllocPortable 0x01
#define cudaHostAllocMapped 0x02

struct dim3 {
  unsigned x, y, z;
  dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

// What clang lowers the <<<...>>> launch syntax into.
extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0,
                                         cudaStream_t stream = 0);
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0,
                                                cudaStream_t stream = 0);
extern "C" cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t offset);
extern "C" cudaError_t cudaLaunch(const void* func);

// Subset of the real cudaDeviceProp — only the fields this project reads.
// clockRate and memoryClockRate are deliberately ABSENT: CUDA 13.0 removed
// them, so leaving them out here makes the stub catch any reintroduced use
// rather than letting it through to fail on the Jetson.
struct cudaDeviceProp {
  char name[256];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  int maxThreadsPerBlock;
  int maxThreadsPerMultiProcessor;
  int multiProcessorCount;
  int l2CacheSize;
  int memoryBusWidth;
  int major;
  int minor;
  int integrated;
  int canMapHostMemory;
  int unifiedAddressing;
  int concurrentKernels;
  int cooperativeLaunch;
};

enum cudaDeviceAttr {
  cudaDevAttrClockRate = 13,
  cudaDevAttrMemoryClockRate = 36,
  cudaDevAttrGlobalMemoryBusWidth = 37,
};

extern "C" {
cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
cudaError_t cudaRuntimeGetVersion(int* version);
cudaError_t cudaDriverGetVersion(int* version);
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags);
cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaGetLastError(void);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
const char* cudaGetErrorString(cudaError_t error);
}

// The templated overloads real cuda_runtime.h provides for C++ callers.
template <class T>
static inline cudaError_t cudaMalloc(T** devPtr, size_t size) {
  return cudaMalloc(reinterpret_cast<void**>(devPtr), size);
}
template <class T>
static inline cudaError_t cudaHostAlloc(T** ptr, size_t size, unsigned int flags) {
  return cudaHostAlloc(reinterpret_cast<void**>(ptr), size, flags);
}
template <class T>
static inline cudaError_t cudaHostGetDevicePointer(T** pDevice, void* pHost, unsigned int flags) {
  return cudaHostGetDevicePointer(reinterpret_cast<void**>(pDevice), pHost, flags);
}
