/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/

#ifndef ACC_CUDA_H
#define ACC_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cublas_v2.h>
#include <stdio.h>

#define ACC(x) cuda##x
#define ACC_DRV(x) CU##x
#define ACC_DRV_FUNC_PREFIX(x) cu##x
#define ACC_RTC(x) nvrtc##x
#define ACC_BLAS(x) cublas##x
#define BACKEND "CUDA"

/* Macros for CUDA error handling
 * Wrap calls to CUDA runtime API (CUDART)
 */
#define ACC_API_CALL(func, args) \
  do { \
    cudaError_t result = ACC(func) args; \
    if (result != cudaSuccess) { \
      printf("\nCUDA RUNTIME API error: %s failed with error %s (%s::%d)\n", #func, cudaGetErrorName(result), __FILE__, __LINE__); \
      exit(1); \
    } \
  } while (0)

/* Wrap calls to CUDA driver API */
#define ACC_DRV_CALL(func, args) \
  do { \
    CUresult result = ACC_DRV_FUNC_PREFIX(func) args; \
    if (result != CUDA_SUCCESS) { \
      const char* msg; \
      cuGetErrorName(result, &msg); \
      printf("\nCUDA DRIVER API ERROR: %s failed with error %s (%s::%d)\n", #func, msg, __FILE__, __LINE__); \
      exit(1); \
    } \
  } while (0)

/* Wrap calls to CUDA NVRTC API */
#define ACC_RTC_CALL(func, args) \
  do { \
    nvrtcResult result = ACC_RTC(func) args; \
    if (result != NVRTC_SUCCESS) { \
      printf("\nNVRTC ERROR: %s failed with error %s\n", #func, nvrtcGetErrorString(result)); \
      exit(1); \
    } \
  } while (0)


/* Wrap calls to CUDA CUBLAS API */
#define ACC_BLAS_CALL(func, args) \
  do { \
    cublasStatus_t result = ACC_BLAS(func) args; \
    if (result != CUBLAS_STATUS_SUCCESS) { \
      const char* error_name = "CUBLAS_ERRROR"; \
      if (result == CUBLAS_STATUS_NOT_INITIALIZED) { \
        error_name = "CUBLAS_STATUS_NOT_INITIALIZED"; \
      } \
      else if (result == CUBLAS_STATUS_ALLOC_FAILED) { \
        error_name = "CUBLAS_STATUS_ALLOC_FAILED"; \
      } \
      else if (result == CUBLAS_STATUS_INVALID_VALUE) { \
        error_name = "CUBLAS_STATUS_INVALID_VALUE"; \
      } \
      else if (result == CUBLAS_STATUS_ARCH_MISMATCH) { \
        error_name = "CUBLAS_STATUS_ARCH_MISMATCH"; \
      } \
      else if (result == CUBLAS_STATUS_MAPPING_ERROR) { \
        error_name = "CUBLAS_STATUS_MAPPING_ERROR"; \
      } \
      else if (result == CUBLAS_STATUS_EXECUTION_FAILED) { \
        error_name = "CUBLAS_STATUS_EXECUTION_FAILED"; \
      } \
      else if (result == CUBLAS_STATUS_INTERNAL_ERROR) { \
        error_name = "CUBLAS_STATUS_INTERNAL_ERROR"; \
      } \
      printf("\nCUBLAS ERROR: %s failed with error %s\n", #func, error_name); \
      exit(1); \
    } \
  } while (0)


extern nvrtcResult nvrtcGetLowLevelCode(nvrtcProgram prog, char* code);
extern nvrtcResult nvrtcGetLowLevelCodeSize(nvrtcProgram prog, size_t* codeSizeRet);
extern CUresult cuLaunchJITKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
  unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream stream,
  void** kernelParams, void** extra);

/* CUDA Runtime API: flag values */
extern CUevent_flags CUEventDefault;
extern CUstream_flags CUStreamDefault;
extern CUsharedconfig CUSharedMemBankSizeEightByte;

/* CUBLAS status and operations */
extern cublasStatus_t ACC_BLAS_STATUS_SUCCESS;
extern cublasOperation_t ACC_BLAS_OP_N;
extern cublasOperation_t ACC_BLAS_OP_T;

/* NVRTC error status */
extern nvrtcResult ACC_RTC_SUCCESS;

#endif /*ACC_CUDA_H*/
