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

/****************************************************************************/
int acc_error_check (ACC(Error_t) error){
  if (error != ACC(Success)){
      printf (BACKEND" error: %s\n", ACC(GetErrorString)(error));
      return -1;
    }
  return 0;
}

extern "C" void acc_clear_errors () {
  ACC(GetLastError)();
}
