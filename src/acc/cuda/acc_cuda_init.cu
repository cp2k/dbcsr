/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/acc.h"

#ifdef __CUDA_PROFILING
#include <nvToolsExtCudaRt.h>
#endif

#define CUDA_SAFE_CALL(name, x)                                   \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      printf("\nerror: %s failed with error %s\n",                \
             name, msg);                                          \
      exit(1);                                                    \
    }                                                             \
  } while(0)


/****************************************************************************/
extern "C" int acc_init(){
  // Driver boilerplate
  CUDA_SAFE_CALL("cuInit", cuInit(0));
  CUdevice cuDevice; 
  CUDA_SAFE_CALL("cuDeviceGet", cuDeviceGet(&cuDevice, 0));
  CUcontext ctx;
  CUDA_SAFE_CALL("cuDevicePrimaryCtxRetain", cuDevicePrimaryCtxRetain(&ctx, cuDevice));
  CUDA_SAFE_CALL("cuCtxPushCurrent", cuCtxPushCurrent(ctx));
  return 0;
}

/****************************************************************************/
extern "C" int acc_finalize(){
  // Release driver resources
  CUcontext ctx;
  CUDA_SAFE_CALL("cuCtxGetCurrent", cuCtxGetCurrent(&ctx)); 
  CUDA_SAFE_CALL("cuCtxPopCurrent", cuCtxPopCurrent(&ctx)); 
  CUdevice cuDevice;
  CUDA_SAFE_CALL("cuDeviceGet", cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL("cuDevicePrimaryCtxRelease", cuDevicePrimaryCtxRelease(cuDevice));
  return 0;
}

