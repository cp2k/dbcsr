/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "acc_cuda_error.h"
#include "../include/acc.h"

// for debug purpose
static const int verbose_print = 1;

/****************************************************************************/
extern "C" int acc_get_ndevices(int *n_devices){
  cudaError_t cErr;

  cErr = cudaGetDeviceCount (n_devices);
  if (cuda_error_check (cErr))
    return -1;
  return 0;
}


/****************************************************************************/
extern "C" int acc_set_active_device(int device_id){
  cudaError_t cErr;
  int myDevice, runtimeVersion;

  cErr = cudaRuntimeGetVersion(&runtimeVersion);
  if (cuda_error_check (cErr))
    return -1;

  cErr = cudaSetDevice (device_id);
  if (cuda_error_check (cErr))
    return -1;

  cErr = cudaGetDevice (&myDevice);
  if (cuda_error_check (cErr))
    return -1;

  if (myDevice != device_id)
    return -1;

  // establish context
  cErr = cudaFree(0);
  if (cuda_error_check (cErr))
    return -1;

  if (verbose_print){
    cErr = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, (size_t) 1000000000);
    if (cuda_error_check (cErr))
      return -1;
  }

  return 0;
}
