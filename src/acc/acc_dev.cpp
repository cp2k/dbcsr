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
#include <math.h>
#include "acc_error.h"
#include "include/acc.h"

// for debug purpose
static const int verbose_print = 1;

/****************************************************************************/
extern "C" int acc_get_ndevices(int *n_devices){
  ACC_API_CALL(GetDeviceCount, (n_devices));
  return 0;
}


/****************************************************************************/
extern "C" int acc_set_active_device(int device_id){
  int myDevice, runtimeVersion;

  ACC_API_CALL(RuntimeGetVersion, (&runtimeVersion));
  ACC_API_CALL(SetDevice, (device_id));
  ACC_API_CALL(GetDevice, (&myDevice));

  if (myDevice != device_id)
    return -1;

  // establish context
  ACC_API_CALL(Free, (0));

#ifdef __HIP_PLATFORM_NVCC__
  if (verbose_print){
    ACC_API_CALL(DeviceSetLimit, (ACC(LimitPrintfFifoSize), (size_t) 1000000000));
  }
#endif

  return 0;
}

