/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#include "dbcsr_omp.h"
#include <stdlib.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(DBCSR_OMP_EVENT_MAXCOUNT) && (0 < DBCSR_OMP_EVENT_MAXCOUNT)
dbcsr_omp_event_t  dbcsr_omp_events[DBCSR_OMP_EVENT_MAXCOUNT];
dbcsr_omp_event_t* dbcsr_omp_eventp[DBCSR_OMP_EVENT_MAXCOUNT];
#endif
int dbcsr_omp_event_count;


int acc_event_create(acc_event_t** event_p)
{
  dbcsr_omp_event_t* event;
  const int result = dbcsr_omp_alloc((void**)&event,
    sizeof(dbcsr_omp_event_t), &dbcsr_omp_event_count,
#if defined(DBCSR_OMP_EVENT_MAXCOUNT) && (0 < DBCSR_OMP_EVENT_MAXCOUNT)
    DBCSR_OMP_EVENT_MAXCOUNT, dbcsr_omp_events, (void**)dbcsr_omp_eventp);
#else
    0, NULL, NULL);
#endif
  if (EXIT_SUCCESS == result) {
    assert(NULL != event && NULL != event_p);
    event->dependency = NULL;
    *event_p = event;
  }
  DBCSR_OMP_RETURN(result);
}


int acc_event_destroy(acc_event_t* event)
{
  return dbcsr_omp_dealloc(event, sizeof(dbcsr_omp_event_t), &dbcsr_omp_event_count,
#if defined(DBCSR_OMP_EVENT_MAXCOUNT) && (0 < DBCSR_OMP_EVENT_MAXCOUNT)
    DBCSR_OMP_EVENT_MAXCOUNT, dbcsr_omp_events, (void**)dbcsr_omp_eventp);
#else
    0, NULL, NULL);
#endif
}


int acc_event_record(acc_event_t* event, acc_stream_t* stream)
{
  int result;
  if (NULL != stream) {
    dbcsr_omp_stream_t *const s = (dbcsr_omp_stream_t*)stream;
    dbcsr_omp_event_t *const e = (dbcsr_omp_event_t*)event;
#if defined(DBCSR_OMP_OFFLOAD)
    if (0 < dbcsr_omp_ndevices()) {
      dbcsr_omp_depend_t* deps;
      dbcsr_omp_stream_depend(stream, &deps);
      deps->data.args[0].ptr = event;
      deps->data.args[1].logical = 1;
      dbcsr_omp_stream_depend_begin();
      if (NULL != e) { /* reset if reused (re-enqueued) */
        e->dependency = deps->data.out;
      }
#     pragma omp master
      { const int ndepend = dbcsr_omp_stream_depend_get_count();
        int tid = 0;
        for (; tid < ndepend; ++tid) {
          dbcsr_omp_depend_t *const di = &deps[tid];
          dbcsr_omp_event_t *const ei = (dbcsr_omp_event_t*)di->data.args[0].ptr;
          const dbcsr_omp_dependency_t *const id = di->data.in, *const od = di->data.out;
          const acc_bool_t ok = di->data.args[1].logical;
          (void)(id); (void)(od); /* suppress incorrect warning */
#if !defined(NDEBUG)
          if (!ok) break; /* incorrect dependency-count */
#endif
          if (NULL != ei) { /* synchronize event */
            uintptr_t/*const dbcsr_omp_dependency_t**/ volatile* /*const*/ sig = (uintptr_t volatile*)&ei->dependency;
#           pragma omp target depend(in:DBCSR_OMP_DEP(id)) depend(out:DBCSR_OMP_DEP(od)) nowait map(from:sig[0:1])
            *sig = 0/*NULL*/;
          }
          else { /* synchronize entire stream */
            int volatile* /*const*/ sig = (int volatile*)&s->pending;
#           pragma omp target depend(in:DBCSR_OMP_DEP(id)) depend(out:DBCSR_OMP_DEP(od)) nowait map(from:sig[0:1])
            *sig = 0;
          }
        }
      }
      result = dbcsr_omp_stream_depend_end(stream);
    }
    else
#endif
    if (NULL != e) {
      e->dependency = NULL;
      result = EXIT_SUCCESS;
    }
    else {
      s->pending = 0;
      result = EXIT_SUCCESS;
    }
  }
  else if (NULL == event) { /* flush all pending work */
    result = EXIT_SUCCESS;
#if defined(DBCSR_OMP_OFFLOAD) && 0
#   pragma omp master
#   pragma omp task if(0)
    result = EXIT_FAILURE;
#elif defined(_OPENMP)
#   pragma omp barrier
#endif
  }
  else result = EXIT_FAILURE;
  DBCSR_OMP_RETURN(result);
}


int acc_event_query(acc_event_t* event, acc_bool_t* has_occurred)
{
  int result = EXIT_FAILURE;
  if (NULL != has_occurred) {
    if (NULL != event) {
      const dbcsr_omp_event_t *const e = (dbcsr_omp_event_t*)event;
      *has_occurred = (NULL == e->dependency);
      result = EXIT_SUCCESS;
    }
    else *has_occurred = 0;
  }
  DBCSR_OMP_RETURN(result);
}


int acc_event_synchronize(acc_event_t* event)
{ /* Waits on the host-side. */
  const dbcsr_omp_event_t *const e = (dbcsr_omp_event_t*)event;
  int result;
#if defined(DBCSR_OMP_OFFLOAD)
  if (0 < dbcsr_omp_ndevices()) {
    if (NULL != e) {
      dbcsr_omp_depend_t* deps;
      dbcsr_omp_stream_depend(NULL/*stream*/, &deps);
      deps->data.args[0].const_ptr = (const void*)&e->dependency;
      dbcsr_omp_stream_depend_begin();
#     pragma omp master
      { const int ndepend = dbcsr_omp_stream_depend_get_count();
        int tid = 0;
        for (; tid < ndepend; ++tid) {
          dbcsr_omp_depend_t *const di = &deps[tid];
          const dbcsr_omp_dependency_t* *const ptr = (const dbcsr_omp_dependency_t**)di->data.args[0].const_ptr;
          const dbcsr_omp_dependency_t *const id = *ptr;
#         pragma omp task depend(in:DBCSR_OMP_DEP(id)) if(0/*NULL*/ != id)
          *ptr = NULL;
        }
      }
      DBCSR_OMP_WAIT(NULL != e->dependency);
    }
    else { /* branch must participate in barrier */
      dbcsr_omp_stream_depend_begin();
    }
    result = dbcsr_omp_stream_depend_end(NULL/*stream*/);
  }
  else
#endif
  if (NULL != e) {
    DBCSR_OMP_WAIT(NULL != e->dependency);
    result = EXIT_SUCCESS;
  }
  else result = EXIT_FAILURE;
  DBCSR_OMP_RETURN(result);
}

#if defined(__cplusplus)
}
#endif
