/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#ifndef ACC_HIP_H
#define ACC_HIP_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>


#define ACC(x) hip##x
#define ACC_DRV(x) ACC(x)
#define ACC_RTC(x) hiprtc##x
#define BACKEND "HIP"

// Macro for HIP error handling
// Wrap calls to HIP API
#define HIP_API_CALL(func, args)                                  \
  do {                                                            \
    hipError_t result = ACC(func) args;                           \
    if (result != hipSuccess) {                                   \
      printf("\nHIP error: %s failed with error %s\n",            \
             #func, hipGetErrorName(result));                     \
      exit(1);                                                    \
    }                                                             \
  } while(0)

// HIP does not differentiate between "runtime" API and "driver" API
#define ACC_API_CALL(func, args) HIP_API_CALL(func, args)
#define ACC_DRV_CALL(func, args) HIP_API_CALL(func, args)

// Wrap calls to HIPRTC API
#define ACC_RTC_CALL(func, args)                                  \
  do {                                                            \
    hiprtcResult result = ACC_RTC(func) args;                     \
    if (result != HIPRTC_SUCCESS) {                               \
      printf("\nHIPRTC ERROR: %s failed with error %s\n",         \
             #func, hiprtcGetErrorString(result));                \
      exit(1);                                                    \
    }                                                             \
  } while(0)

extern hipError_t hipHostAlloc(void **ptr, size_t size, unsigned int flags);
extern unsigned int hipHostAllocDefault;
extern hipError_t hipFreeHost(void *ptr);
extern hiprtcResult hiprtcGetLowLevelCode(hiprtcProgram prog, char* code);
extern hiprtcResult hiprtcGetLowLevelCodeSize(hiprtcProgram prog, size_t* codeSizeRet);
extern hipError_t hipEventCreate(hipEvent_t *event, unsigned flags);
extern hipError_t hipStreamCreate(hipStream_t *stream, unsigned int flags);
extern hipError_t hipLaunchJITKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                     unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                     unsigned int sharedMemBytes, hipStream_t stream, void **kernelParams, void **extra);

// HIP API: types
// In the HIP API, there is no difference between runtime API and driver API
// we therefore remap what the Driver API types would look like back to runtime API
using hipfunction = hipFunction_t;
using hipstream = hipStream_t;
using hipevent = hipEvent_t;
using hipmodule = hipModule_t;
using hipdevice = hipDevice_t;
using hipDeviceProp = hipDeviceProp_t;
using hipcontext = hipCtx_t;

#endif // ACC_HIP_H
