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

#ifdef __CUDA_PROFILING
#include <nvToolsExtCudaRt.h>
#endif

static const int verbose_print = 0;


/****************************************************************************/
extern "C" int acc_stream_priority_range(int* least, int* greatest){
  *least = -1;
  *greatest = -1;
  ACC_API_CALL(DeviceGetStreamPriorityRange, (least, greatest));

  return 0;
}


/****************************************************************************/
extern "C" int acc_stream_create(void** stream_p, const char* name, int priority){
  ACC(Error_t) cErr;
  *stream_p = malloc(sizeof(ACC(Stream_t)));

  ACC(Stream_t)* acc_stream = (ACC(Stream_t)*) *stream_p;

  if(priority > 0){
      unsigned int flags = ACC(StreamNonBlocking);
      cErr = ACC(StreamCreateWithPriority)(acc_stream, flags, priority);
  } else {
      cErr = ACC(StreamCreate)(acc_stream);
  }

  if (verbose_print) printf("StreamCreate : %p -> %p \n", *stream_p, *acc_stream);
  if (acc_error_check(cErr)) return -1;
  if (acc_error_check(ACC(GetLastError)())) return -1;

#ifdef __CUDA_PROFILING
  nvtxNameCudaStreamA(*custream, name);
#endif

    return 0;
}


/****************************************************************************/
extern "C" int acc_stream_destroy(void* stream){
    ACC(Stream_t)* acc_stream = (ACC(Stream_t)*) stream;

    if(verbose_print) printf("StreamDestroy called\n");
    if (stream == NULL) return 0; /* not an error */
    ACC(Error_t) cErr = ACC(StreamDestroy)(*acc_stream);
    free(acc_stream);
    if (acc_error_check (cErr)) return -1;
    if (acc_error_check(ACC(GetLastError)())) return -1;
    return 0;
}

/****************************************************************************/
extern "C" int acc_stream_sync(void* stream)
{
  ACC(Stream_t)* acc_stream = (ACC(Stream_t)*) stream;
  ACC_API_CALL(StreamSynchronize, (*acc_stream));
  return 0;
}
