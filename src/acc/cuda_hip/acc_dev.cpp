/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* Copyright (C) 2022 Advanced Micro Devices, Inc. - All rights reserved                          */
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

#include "acc_error.h"
#include "../acc.h"

#include <stdio.h>
#include <math.h>

// for debug purpose
#if defined(__HIP_PLATFORM_NVCC__)
static const int verbose_print = 1;
#endif

/****************************************************************************/
extern "C" int c_dbcsr_acc_get_ndevices(int* n_devices) {
  ACC_API_CALL(GetDeviceCount, (n_devices));
  return 0;
}

/****************************************************************************/
extern "C" int c_dbcsr_acc_device_synchronize() {
  ACC_API_CALL(DeviceSynchronize, ());
  return 0;
}

/****************************************************************************/
extern "C" int c_dbcsr_acc_set_active_device(int device_id) {
  int myDevice;

  ACC_API_CALL(SetDevice, (device_id));
  ACC_API_CALL(GetDevice, (&myDevice));

  if (myDevice != device_id) return -1;

  // establish context
  ACC_API_CALL(Free, (0));

#if defined(__HIP_PLATFORM_NVCC__)
  if (verbose_print) {
    ACC_API_CALL(DeviceSetLimit, (ACC(LimitPrintfFifoSize), (size_t)1000000000));
  }
#endif

  return 0;
}
