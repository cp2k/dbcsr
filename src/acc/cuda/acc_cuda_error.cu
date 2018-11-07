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

/****************************************************************************/
int cuda_error_check (cudaError_t cudaError){
  if (cudaError != cudaSuccess){
      printf ("CUDA Error: %s\n", cudaGetErrorString (cudaError));
      return -1;
    }
  return 0;
}

extern "C" void acc_clear_errors () {
  cudaGetLastError();
}
