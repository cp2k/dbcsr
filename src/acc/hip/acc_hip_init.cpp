/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include <hip/hip_runtime.h>
#include <stdio.h>
#include "../include/acc.h"

#ifdef __CUDA_PROFILING
#include <nvToolsExtCudaRt.h>
#endif

#define HIP_SAFE_CALL(name, x)                                    \
  do {                                                            \
    hipError_t result = x;                                        \
    if (result != hipSuccess) {                                   \
      const char *msg;                                            \
      msg = hipGetErrorName(result);                              \
      printf("\nerror: %s failed with error %s\n",                \
             name, msg);                                          \
      exit(1);                                                    \
    }                                                             \
  } while(0)


/****************************************************************************/
extern "C" int acc_init(){
  // Driver boilerplate
  HIP_SAFE_CALL("hipInit", hipInit(0));
  hipDevice_t hipDevice;
  HIP_SAFE_CALL("hipDeviceGet", hipDeviceGet(&hipDevice, 0));
  hipCtx_t ctx;
  HIP_SAFE_CALL("hipDevicePrimaryCtxRetain", hipDevicePrimaryCtxRetain(&ctx, hipDevice));
  return 0;
}

/****************************************************************************/
extern "C" int acc_finalize(){
  // Release driver resources
  hipCtx_t ctx;
  hipDevice_t hipDevice;
  HIP_SAFE_CALL("hipDeviceGet", hipDeviceGet(&hipDevice, 0));
  HIP_SAFE_CALL("hipDevicePrimaryCtxRelease", hipDevicePrimaryCtxRelease(hipDevice));
  return 0;
}

