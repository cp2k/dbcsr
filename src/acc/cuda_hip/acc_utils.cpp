/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/

#include "acc_utils.h"

#if defined(__CUDA)
#  include "../cuda/acc_cuda.h"
#elif defined(__HIP)
#  include "../hip/acc_hip.h"
#endif

//===========================================================================
int acc_get_gpu_warp_size() {
  int device = 0;
  ACC(DeviceProp) prop;
  ACC_API_CALL(GetDevice, (&device));
  ACC_API_CALL(GetDeviceProperties, (&prop, device));
  return prop.warpSize;
}
