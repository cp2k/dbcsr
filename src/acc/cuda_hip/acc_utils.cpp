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

#if defined(__parallel)
#  include <mpi.h>
#if defined(OPEN_MPI)
#  include <mpi-ext.h>
#endif
#endif

//===========================================================================
int acc_get_gpu_warp_size() {
  int device = 0;
  ACC(DeviceProp) prop;
  ACC_API_CALL(GetDevice, (&device));
  ACC_API_CALL(GetDeviceProperties, (&prop, device));
  return prop.warpSize;
}

extern "C" {
#if defined(__parallel)
/*
    * Need these bindings since some MPI implementations define these functions
    * only for their C API.
    *  Need to write wrappers for them since they may be defined as a Macro in C.
    */
int dbcsr_mpix_query_cuda_support() {
#  if defined(OPEN_MPI) || defined(MPICH)
  return MPIX_Query_cuda_support();
#  else
  return 0;
#  endif
}
int dbcsr_mpix_query_hip_support() {
#  if defined(OPEN_MPI)
  return MPIX_Query_rocm_support();
#  elif defined(MPICH)
  return MPIX_Query_hip_support();
#  else
  return 0;
#  endif
}
#endif
}