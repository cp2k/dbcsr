/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#ifdef __CUDA
#include "cuda/acc_cuda.h"
#else
#include "hip/acc_hip.h"
#endif

#include <stdio.h>
#include "include/acc.h"
#include "include/acc_libsmm.h"

#ifdef __CUDA_PROFILING
#include <nvToolsExtCudaRt.h>
#endif

/****************************************************************************/
extern "C" int acc_init(){
  // Driver boilerplate
  ACC_DRV_CALL(Init, (0));
  ACC_DRV(device) acc_device;
  ACC_DRV_CALL(DeviceGet, (&acc_device, 0));
  ACC_DRV(context) ctx;
  ACC_DRV_CALL(DevicePrimaryCtxRetain, (&ctx, acc_device));

  // Initialize libsmm_acc, DBCSR's GPU backend
  libsmm_acc_init();
  return 0;
}

/****************************************************************************/
extern "C" int acc_finalize(){
  // Release driver resources
  ACC_DRV(device) acc_device;
  ACC_DRV_CALL(DeviceGet, (&acc_device, 0));
  ACC_DRV_CALL(DevicePrimaryCtxRelease, (acc_device));
  return 0;
}

