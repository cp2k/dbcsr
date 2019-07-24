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

#ifdef __CUDA_PROFILING
#include <nvToolsExtCudaRt.h>
#endif

static const int verbose_print = 0;


/****************************************************************************/
extern "C" int acc_stream_priority_range(int* least, int* greatest){
  *least = -1;
  *greatest = -1;

#ifndef __HAS_NO_CUDA_STREAM_PRIORITIES
  hipError_t cErr = hipDeviceGetStreamPriorityRange(least, greatest);
  if (cuda_error_check(cErr)) return -1;
  if (cuda_error_check(hipGetLastError())) return -1;
#endif

  return 0;
}


/****************************************************************************/
extern "C" int acc_stream_create(void** stream_p, const char* name, int priority){
  hipError_t cErr;
  *stream_p = malloc(sizeof(hipStream_t));

  hipStream_t* custream = (hipStream_t*) *stream_p;

#ifndef __HAS_NO_CUDA_STREAM_PRIORITIES
  if(priority > 0){
      unsigned int flags = hipStreamNonBlocking;
      cErr =  hipStreamCreateWithPriority(custream, flags, priority);
  }else
#endif
      cErr = hipStreamCreate(custream);


  if (verbose_print) printf("cuda_stream_create: %p -> %d \n", *stream_p, *custream);
  if (cuda_error_check(cErr)) return -1;
  if (cuda_error_check(hipGetLastError())) return -1;

#ifdef __CUDA_PROFILING
  nvtxNameCudaStreamA(*custream, name);
#endif

    return 0;
}


/****************************************************************************/
extern "C" int acc_stream_destroy(void* stream){
    hipStream_t* custream = (hipStream_t*) stream;

    if(verbose_print) printf("cuda_stream_destroy called\n");
    hipError_t cErr = hipStreamDestroy(*custream);
    free(custream);
    if (cuda_error_check (cErr)) return -1;
    if (cuda_error_check(hipGetLastError ()))return -1;
    return 0;
}

/****************************************************************************/
extern "C" int acc_stream_sync(void* stream)
{
  hipStream_t* custream = (hipStream_t*) stream;
  hipError_t cErr = hipStreamSynchronize(*custream);
  if (cuda_error_check (cErr)) return -1;
  return 0;
}
