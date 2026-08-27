// Stand-in for cuBLAS, for syntax checking only. See cuda_runtime.h in this
// directory for the caveats — matching signatures, not the real library.
#pragma once

#include <cuda_runtime.h>

typedef struct cublasContext* cublasHandle_t;

typedef enum {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_ALLOC_FAILED = 3
} cublasStatus_t;

typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 } cublasOperation_t;

extern "C" {
cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId);
cublasStatus_t cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                              const float* alpha, const float* A, int lda, const float* x,
                              int incx, const float* beta, float* y, int incy);
}

#define cublasCreate cublasCreate_v2
#define cublasDestroy cublasDestroy_v2
#define cublasSetStream cublasSetStream_v2
#define cublasSgemv cublasSgemv_v2
