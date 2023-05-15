/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/

#ifndef ACC_HIP_H
#define ACC_HIP_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#if __has_include(<hipblas/hipblas.h>)
#  include <hipblas/hipblas.h>
#else
#  include <hipblas.h>
#endif
#include <hip/hiprtc.h>

#define ACC(x) hip##x
#define ACC_DRV(x) ACC(x)
#define ACC_BLAS(x) hipblas##x
#define ACC_RTC(x) hiprtc##x
#define BACKEND "HIP"

/* Macro for HIP error handling
 * Wrap calls to HIP API
 */
#define HIP_API_CALL(func, args) \
  do { \
    hipError_t result = ACC(func) args; \
    if (result != hipSuccess) { \
      printf("\nHIP error: %s failed with error %s (%s::%d)\n", #func, hipGetErrorName(result), __FILE__, __LINE__); \
      exit(1); \
    } \
  } while (0)

/* HIP does not differentiate between "runtime" API and "driver" API */
#define ACC_API_CALL(func, args) HIP_API_CALL(func, args)
#define ACC_DRV_CALL(func, args) HIP_API_CALL(func, args)

/* Wrap calls to HIPRTC API */
#define ACC_RTC_CALL(func, args) \
  do { \
    hiprtcResult result = ACC_RTC(func) args; \
    if (result != HIPRTC_SUCCESS) { \
      printf("\nHIPRTC ERROR: %s failed with error %s (%s::%d)\n", #func, hiprtcGetErrorString(result), __FILE__, __LINE__); \
      exit(1); \
    } \
  } while (0)

/* Wrap calls to HIPBLAS API */
#define ACC_BLAS_CALL(func, args) \
  do { \
    hipblasStatus_t result = ACC_BLAS(func) args; \
    if (result != HIPBLAS_STATUS_SUCCESS) { \
      const char* error_name = "HIPBLAS_ERRROR"; \
      if (result == HIPBLAS_STATUS_NOT_INITIALIZED) { \
        error_name = "HIPBLAS_STATUS_NOT_INITIALIZED "; \
      } \
      else if (result == HIPBLAS_STATUS_ALLOC_FAILED) { \
        error_name = "HIPBLAS_STATUS_ALLOC_FAILED "; \
      } \
      else if (result == HIPBLAS_STATUS_INVALID_VALUE) { \
        error_name = "HIPBLAS_STATUS_INVALID_VALUE "; \
      } \
      else if (result == HIPBLAS_STATUS_MAPPING_ERROR) { \
        error_name = "HIPBLAS_STATUS_MAPPING_ERROR "; \
      } \
      else if (result == HIPBLAS_STATUS_EXECUTION_FAILED) { \
        error_name = "HIPBLAS_STATUS_EXECUTION_FAILED "; \
      } \
      else if (result == HIPBLAS_STATUS_INTERNAL_ERROR) { \
        error_name = "HIPBLAS_STATUS_INTERNAL_ERROR "; \
      } \
      else if (result == HIPBLAS_STATUS_NOT_SUPPORTED) { \
        error_name = "HIPBLAS_STATUS_NOT_SUPPORTED "; \
      } \
      else if (result == HIPBLAS_STATUS_ARCH_MISMATCH) { \
        error_name = "HIPBLAS_STATUS_ARCH_MISMATCH "; \
      } \
      else if (result == HIPBLAS_STATUS_HANDLE_IS_NULLPTR) { \
        error_name = "HIPBLAS_STATUS_HANDLE_IS_NULLPTR "; \
      } \
      printf("\nHIPBLAS ERROR: %s failed with error %s\n", #func, error_name); \
      exit(1); \
    } \
  } while (0)

extern hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags);
extern hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags);
extern unsigned int hipHostAllocDefault;
extern hipError_t hipFreeHost(void* ptr);
extern hiprtcResult hiprtcGetLowLevelCode(hiprtcProgram prog, char* code);
extern hiprtcResult hiprtcGetLowLevelCodeSize(hiprtcProgram prog, size_t* codeSizeRet);
extern hipError_t hipEventCreate(hipEvent_t* event, unsigned flags);
extern hipError_t hipStreamCreate(hipStream_t* stream, unsigned int flags);
extern hipError_t hipLaunchJITKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
  unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream,
  void** kernelParams, void** extra);

/* HIP API: types
 * In the HIP API, there is no difference between runtime API and driver API
 * we therefore remap what the Driver API types would look like back to runtime API
 */
using hipfunction = hipFunction_t;
using hipstream = hipStream_t;
using hipevent = hipEvent_t;
using hipmodule = hipModule_t;
using hipdevice = hipDevice_t;
using hipDeviceProp = hipDeviceProp_t;
using hipcontext = hipCtx_t;

/* HIPBLAS status and operations */
extern hipblasStatus_t ACC_BLAS_STATUS_SUCCESS;
extern hipblasOperation_t ACC_BLAS_OP_N;
extern hipblasOperation_t ACC_BLAS_OP_T;

/* HIPRTC error status */
extern hiprtcResult ACC_RTC_SUCCESS;


#endif /*ACC_HIP_H*/
