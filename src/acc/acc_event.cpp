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

static const int verbose_print = 0;

/****************************************************************************/
extern "C" int acc_event_create(void** event_p){
  *event_p = malloc(sizeof(ACC(Event_t)));
  ACC(Event_t)* acc_event = (ACC(Event_t)*) *event_p;

  ACC(Error_t) cErr = ACC(EventCreate)(acc_event);
  if(verbose_print) printf("EventCreate) :  %p -> %d\n", *event_p, *acc_event);
  if (acc_error_check(cErr)) return -1;
  if (acc_error_check(ACC(GetLastError)())) return -1;
  return 0;
}


/****************************************************************************/
extern "C" int acc_event_destroy(void* event){
    ACC(Event_t)* acc_event = (ACC(Event_t*)) event;

    if(verbose_print) printf("EventDestroy, called\n");
    if (event == NULL) return 0; /* not an error */
    ACC(Error_t) cErr = ACC(EventDestroy)(*acc_event);
    free(acc_event);
    if (acc_error_check(cErr)) return -1;
    if (acc_error_check(ACC(GetLastError)())) return -1;
    return 0;
}


/****************************************************************************/
extern "C" int acc_event_record(void* event, void* stream){
    ACC(Event_t)* acc_event = (ACC(Event_t)*) event;
    ACC(Stream_t)* acc_stream = (ACC(Stream_t)*) stream;

    if(verbose_print) printf("EventRecord): %p -> %d,  %p -> %d\n", acc_event, *acc_event, acc_stream, *acc_stream);
    ACC_API_CALL(EventRecord, (*acc_event, *acc_stream));
    return 0;
}


/****************************************************************************/
extern "C" int acc_event_query(void* event, int* has_occurred){
    if(verbose_print) printf("acc_event_query called\n");

    ACC(Event_t)* acc_event = (ACC(Event_t)*) event;
    ACC(Error_t) cErr = ACC(EventQuery)(*acc_event);
    if(cErr==ACC(Success)){
         *has_occurred = 1;
         return 0;
    }

    if(cErr==ACC(ErrorNotReady)){
        *has_occurred = 0;
        return 0;
    }

    return -1; // something went wrong
}


/****************************************************************************/
extern "C" int acc_stream_wait_event(void* stream, void* event){
    if(verbose_print) printf("acc_stream_wait_event called\n");

    ACC(Event_t)* acc_event = (ACC(Event_t)*) event;
    ACC(Stream_t)* acc_stream = (ACC(Stream_t)*) stream;

    // flags: Parameters for the operation (must be 0)
    ACC_API_CALL(StreamWaitEvent, (*acc_stream, *acc_event, 0));
    return 0;
}


/****************************************************************************/
extern "C" int acc_event_synchronize(void* event){
    if(verbose_print) printf("EventSynchronize called\n");
    ACC(Event_t)* acc_event = (ACC(Event_t)*) event;
    ACC_API_CALL(EventSynchronize, (*acc_event));
    return 0;
}
