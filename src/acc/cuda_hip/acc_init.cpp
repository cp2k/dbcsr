/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/

#if defined(__CUDA)
#  include "../cuda/acc_cuda.h"
#elif defined(__HIP)
#  include "../hip/acc_hip.h"
#endif

#include "../acc.h"
#include "../acc_libsmm.h"

#include <stdio.h>

/****************************************************************************/
extern "C" int c_dbcsr_acc_init() {
  int myDevice, runtimeVersion;
  // Driver boilerplate
  ACC_DRV_CALL(Init, (0));
  ACC_DRV(device) acc_device;
  ACC_API_CALL(GetDevice, (&myDevice));
  ACC_DRV_CALL(DeviceGet, (&acc_device, myDevice));
#if defined(__CUDA)
  ACC_DRV(context) ctx;
  ACC_DRV_CALL(DevicePrimaryCtxRetain, (&ctx, acc_device));
#endif
  ACC_API_CALL(RuntimeGetVersion, (&runtimeVersion));

  // Initialize libsmm_acc, DBCSR's GPU backend
  return libsmm_acc_init();
}

/****************************************************************************/
extern "C" int c_dbcsr_acc_finalize() {
  int myDevice;
  // Release driver resources
  ACC_DRV(device) acc_device;
  ACC_API_CALL(GetDevice, (&myDevice));
  ACC_DRV_CALL(DeviceGet, (&acc_device, myDevice));
#if defined(__CUDA)
  ACC_DRV_CALL(DevicePrimaryCtxRelease, (acc_device));
#endif
  return libsmm_acc_finalize();
}
