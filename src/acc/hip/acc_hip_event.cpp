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

static const int verbose_print = 0;

/****************************************************************************/
extern "C" int acc_event_create(void** event_p){
  *event_p = malloc(sizeof(hipEvent_t));
  hipEvent_t* cuevent = (hipEvent_t*) *event_p;

  hipError_t cErr = hipEventCreate(cuevent);
  if(verbose_print) printf("cuda_event_created:  %p -> %d\n", *event_p, *cuevent);
  if (cuda_error_check(cErr)) return -1;
  if (cuda_error_check(hipGetLastError())) return -1;
  return 0;
}


/****************************************************************************/
extern "C" int acc_event_destroy(void* event){
    hipEvent_t* cuevent = (hipEvent_t*) event;

    if(verbose_print) printf("cuda_event_destroy called\n");
    hipError_t cErr = hipEventDestroy(*cuevent);
    free(cuevent);
    if (cuda_error_check (cErr)) return -1;
    if (cuda_error_check(hipGetLastError ())) return -1;
    return 0;
}


/****************************************************************************/
extern "C" int acc_event_record(void* event, void* stream){
    hipEvent_t* cuevent = (hipEvent_t*) event;
    hipStream_t* custream = (hipStream_t*) stream;

    if(verbose_print) printf("cuda_event_record: %p -> %d,  %p -> %d\n", cuevent, *cuevent,  custream, *custream);
    hipError_t cErr = hipEventRecord (*cuevent, *custream);
    if (cuda_error_check (cErr)) return -1;
    //if (cuda_error_check(hipGetLastError ())) return -1;
    return 0;
}


/****************************************************************************/
extern "C" int acc_event_query(void* event, int* has_occurred){
    if(verbose_print) printf("cuda_event_query called\n");

    hipEvent_t* cuevent = (hipEvent_t*) event;
    hipError_t cErr = hipEventQuery(*cuevent);
    //if(cuda_error_check(hipGetLastError ())) return -1;
    if(cErr==hipSuccess){
         *has_occurred = 1;
         return 0;
    }

    if(cErr==hipErrorNotReady){
        *has_occurred = 0;
        return 0;
    }

    return -1; // something went wrong
}


/****************************************************************************/
extern "C" int acc_stream_wait_event(void* stream, void* event){
    if(verbose_print) printf("cuda_stream_wait_event called\n");

    hipEvent_t* cuevent = (hipEvent_t*) event;
    hipStream_t* custream = (hipStream_t*) stream;

    // flags: Parameters for the operation (must be 0)
    hipError_t cErr = hipStreamWaitEvent(*custream, *cuevent, 0);
    if (cuda_error_check (cErr)) return -1;
    //if (cuda_error_check(hipGetLastError ())) return -1;
    return 0;
}


/****************************************************************************/
extern "C" int acc_event_synchronize(void* event){
    if(verbose_print) printf("cuda_event_synchronize called\n");
    hipEvent_t* cuevent = (hipEvent_t*) event;
    hipError_t cErr = hipEventSynchronize(*cuevent);
    if (cuda_error_check (cErr)) return -1;
    if (cuda_error_check(hipGetLastError ())) return -1;
    return 0;
}
