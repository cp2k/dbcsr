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
#include <math.h>
#include "acc_hip_error.h"
#include "../include/acc.h"


/****************************************************************************/
extern "C" int acc_get_ndevices(int *n_devices){
  hipError_t cErr;

  cErr = hipGetDeviceCount (n_devices);
  if (cuda_error_check (cErr))
    return -1;
  return 0;
}


/****************************************************************************/
extern "C" int acc_set_active_device(int device_id){
  hipError_t cErr;
  int myDevice, runtimeVersion;

  cErr = hipRuntimeGetVersion(&runtimeVersion);
  if (cuda_error_check (cErr))
    return -1;

  cErr = hipSetDevice (device_id);
  if (cuda_error_check (cErr))
    return -1;

  cErr = hipGetDevice (&myDevice);
  if (cuda_error_check (cErr))
    return -1;

  if (myDevice != device_id)
    return -1;

  // establish context
  cErr = hipFree(0);
  if (cuda_error_check (cErr))
    return -1;

  return 0;
}
