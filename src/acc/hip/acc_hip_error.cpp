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

/****************************************************************************/
int cuda_error_check (hipError_t hipError_t){
  if (hipError_t != hipSuccess){
      printf ("CUDA Error: %s\n", hipGetErrorString (hipError_t));
      return -1;
    }
  return 0;
}

extern "C" void acc_clear_errors () {
  hipGetLastError();
}
