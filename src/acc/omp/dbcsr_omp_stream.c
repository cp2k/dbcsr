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
#include <string.h>
#include <assert.h>

#if defined(__cplusplus)
extern "C" {
#endif

dbcsr_omp_depend_t dbcsr_omp_stream_depend_state[DBCSR_OMP_THREADS_MAXCOUNT];
#if defined(DBCSR_OMP_STREAM_MAXCOUNT) && (0 < DBCSR_OMP_STREAM_MAXCOUNT)
dbcsr_omp_stream_t  dbcsr_omp_streams[DBCSR_OMP_STREAM_MAXCOUNT];
dbcsr_omp_stream_t* dbcsr_omp_streamp[DBCSR_OMP_STREAM_MAXCOUNT];
#endif
volatile int dbcsr_omp_stream_depend_counter;
int dbcsr_omp_stream_depend_count;
int dbcsr_omp_stream_count;


void dbcsr_omp_stream_barrier_init(int nthreads)
{ /* boot-strap/initialize custom barrier using OpenMP */
  if (nthreads != dbcsr_omp_stream_depend_count) {
#if defined(_OPENMP)
#   pragma omp barrier
#   pragma omp single /* implied barrier */
#endif
    {
      dbcsr_omp_stream_depend_counter = nthreads;
      dbcsr_omp_stream_depend_count = nthreads;
    }
  }
}


void dbcsr_omp_stream_barrier_wait(void)
{ /* custom barrier wait */
  static volatile int dbcsr_omp_stream_barrier_flag = 0;
#if !defined(_OPENMP)
  dbcsr_omp_depend_t *const di = dbcsr_omp_stream_depend_state;
#else
  dbcsr_omp_depend_t *const di = &dbcsr_omp_stream_depend_state[omp_get_thread_num()];
# pragma omp atomic
#endif
  --dbcsr_omp_stream_depend_counter;
  di->data.counter = !di->data.counter; /* sense reversal */
  if (0 != dbcsr_omp_stream_depend_counter) { /* arrived early */
    DBCSR_OMP_WAIT(di->data.counter != dbcsr_omp_stream_barrier_flag);
  }
  else { /* arrived last */
    dbcsr_omp_stream_depend_counter = dbcsr_omp_stream_depend_count;
    dbcsr_omp_stream_barrier_flag = di->data.counter;
  }
}


void dbcsr_omp_stream_depend(acc_stream_t* stream, dbcsr_omp_depend_t** depend)
{
#if defined(_OPENMP)
  dbcsr_omp_depend_t *const di = &dbcsr_omp_stream_depend_state[omp_get_thread_num()];
  const int nthreads = omp_get_num_threads();
#else
  dbcsr_omp_depend_t *const di = dbcsr_omp_stream_depend_state;
  const int nthreads = 1;
#endif
  dbcsr_omp_stream_t *const s = (dbcsr_omp_stream_t*)stream;
#if defined(DBCSR_OMP_STREAM_MAXCOUNT) && (0 < DBCSR_OMP_STREAM_MAXCOUNT)
  assert(NULL == s || (dbcsr_omp_streams <= s && s < (dbcsr_omp_streams + DBCSR_OMP_STREAM_MAXCOUNT)));
#endif
#if defined(DBCSR_OMP_OFFLOAD) && !defined(NDEBUG)
  assert(NULL == s || omp_get_default_device() == s->device_id);
#endif
  assert(NULL != depend && NULL != di);
  if (NULL != s && EXIT_SUCCESS == s->status) {
    static const dbcsr_omp_dependency_t dummy = 0;
    int index;
#if defined(_OPENMP) && (200805 <= _OPENMP) /* OpenMP 3.0 */
#   pragma omp atomic capture
#elif defined(_OPENMP)
#   pragma omp critical(dbcsr_omp_stream_depend_critical)
#endif
    index = s->pending++;
    di->data.out = s->name + index % DBCSR_OMP_STREAM_MAXPENDING;
    di->data.in = (s->name < di->data.out ? (di->data.out - 1) : &dummy);
  }
#if !defined(NDEBUG)
# if defined(_OPENMP)
# pragma omp master
# endif
  { int tid = 0;
    for (; tid < nthreads; ++tid) {
      /* enable user to assume NULL/0 in case of unset arguments */
      memset(dbcsr_omp_stream_depend_state[tid].data.args, 0,
      DBCSR_OMP_ARGUMENTS_MAXCOUNT * sizeof(dbcsr_omp_any_t));
    }
  }
#endif
  dbcsr_omp_stream_barrier_init(nthreads);
  *depend = di;
}


int dbcsr_omp_stream_depend_get_count(void)
{
#if defined(_OPENMP)
  assert(/*master*/0 == omp_get_thread_num());
#else
  assert(1 == dbcsr_omp_stream_depend_count);
#endif
  return dbcsr_omp_stream_depend_count;
}


void dbcsr_omp_stream_depend_begin(void)
{
  dbcsr_omp_stream_barrier_wait();
}


int dbcsr_omp_stream_depend_end(const acc_stream_t* stream)
{
  const dbcsr_omp_stream_t *const s = (const dbcsr_omp_stream_t*)stream;
  dbcsr_omp_stream_barrier_wait();
  DBCSR_OMP_RETURN(NULL != s ? s->status : EXIT_SUCCESS);
}


int acc_stream_create(acc_stream_t** stream_p, const char* name, int priority)
{
  dbcsr_omp_stream_t* stream;
  const int result = dbcsr_omp_alloc((void**)&stream,
    sizeof(dbcsr_omp_stream_t), &dbcsr_omp_stream_count,
#if defined(DBCSR_OMP_STREAM_MAXCOUNT) && (0 < DBCSR_OMP_STREAM_MAXCOUNT)
    DBCSR_OMP_STREAM_MAXCOUNT, dbcsr_omp_streams, (void**)dbcsr_omp_streamp);
#else
    0, NULL, NULL);
#endif
  assert(NULL != stream_p);
  if (EXIT_SUCCESS == result) {
    assert(NULL != stream);
    strncpy(stream->name, name, DBCSR_OMP_STREAM_MAXPENDING);
    stream->name[DBCSR_OMP_STREAM_MAXPENDING-1] = '\0';
    stream->priority = priority;
    stream->pending = 0;
    stream->status = 0;
#if defined(DBCSR_OMP_OFFLOAD) && !defined(NDEBUG)
    stream->device_id = omp_get_default_device();
#endif
    *stream_p = stream;
  }
  DBCSR_OMP_RETURN(result);
}


int acc_stream_destroy(acc_stream_t* stream)
{
  dbcsr_omp_stream_t *const s = (dbcsr_omp_stream_t*)stream;
  int result = ((NULL != s && 0 < s->pending) ? acc_stream_sync(stream) : EXIT_SUCCESS);
#if defined(DBCSR_OMP_STREAM_MAXCOUNT) && (0 < DBCSR_OMP_STREAM_MAXCOUNT)
  assert(NULL == s || (dbcsr_omp_streams <= s && s < (dbcsr_omp_streams + DBCSR_OMP_STREAM_MAXCOUNT)));
#endif
  if (EXIT_SUCCESS == result) {
    result = dbcsr_omp_dealloc(stream, sizeof(dbcsr_omp_stream_t), &dbcsr_omp_stream_count,
#if defined(DBCSR_OMP_STREAM_MAXCOUNT) && (0 < DBCSR_OMP_STREAM_MAXCOUNT)
      DBCSR_OMP_STREAM_MAXCOUNT, dbcsr_omp_streams, (void**)dbcsr_omp_streamp);
#else
      0, NULL, NULL);
#endif
  }
  DBCSR_OMP_RETURN(result);
}


void dbcsr_omp_stream_clear_errors(void)
{
#if defined(_OPENMP)
# pragma omp critical
#endif
  {
#if defined(DBCSR_OMP_STREAM_MAXCOUNT) && (0 < DBCSR_OMP_STREAM_MAXCOUNT)
    int i = 0;
    for (; i < DBCSR_OMP_STREAM_MAXCOUNT; ++i) {
      dbcsr_omp_streams[i].status = EXIT_SUCCESS;
    }
#endif
  }
}


int acc_stream_priority_range(int* least, int* greatest)
{
  int result;
  if (NULL != least || NULL != greatest) {
    if (NULL != least) {
      *least = -1;
    }
    if (NULL != greatest) {
      *greatest = -1;
    }
    result = EXIT_SUCCESS;
  }
  else {
    result = EXIT_FAILURE;
  }
  DBCSR_OMP_RETURN(result);
}


int acc_stream_sync(acc_stream_t* stream)
{ /* Blocks the host-thread. */
  int result = (NULL != stream ? EXIT_SUCCESS : EXIT_FAILURE);
  if (EXIT_SUCCESS == result) {
    result = acc_event_record(NULL/*event*/, stream);
    if (EXIT_SUCCESS == result) {
      dbcsr_omp_stream_t* const s = (dbcsr_omp_stream_t*)stream;
      DBCSR_OMP_WAIT(s->pending);
    }
  }
  DBCSR_OMP_RETURN(result);
}


int acc_stream_wait_event(acc_stream_t* stream, acc_event_t* event)
{ /* Waits (device-side) for an event (potentially recorded on a different stream). */
  int result;
  if (NULL != stream && NULL != event) {
#if defined(DBCSR_OMP_OFFLOAD)
    if (0 < dbcsr_omp_ndevices()) {
      dbcsr_omp_depend_t* deps;
      dbcsr_omp_stream_depend(stream, &deps);
      deps->data.args[0].const_ptr = event;
      dbcsr_omp_stream_depend_begin();
#     pragma omp master
      { const int ndepend = dbcsr_omp_stream_depend_get_count();
        int tid = 0;
        for (; tid < ndepend; ++tid) {
          const dbcsr_omp_depend_t *const di = &deps[tid];
          const dbcsr_omp_event_t *const ei = (const dbcsr_omp_event_t*)di->data.args[0].const_ptr;
          const char* ie;
#if !defined(NDEBUG)
          if (NULL == ei) break; /* incorrect dependency-count */
#endif
          ie = ei->dependency;
          if (NULL != ie) { /* still pending */
            const char *const id = di->data.in, *const od = di->data.out;
            (void)(id); (void)(od); (void)(ie); /* suppress incorrect warning */
#           pragma omp target depend(in:DBCSR_OMP_DEP(id),DBCSR_OMP_DEP(ie)) depend(out:DBCSR_OMP_DEP(od)) nowait if(0)
            {}
          }
        }
      }
      result = dbcsr_omp_stream_depend_end(stream);
    }
    else
#endif
    result = acc_event_synchronize(event);
  }
  else result = (NULL == event ? EXIT_SUCCESS : EXIT_FAILURE);
  DBCSR_OMP_RETURN(result);
}

#if defined(__cplusplus)
}
#endif
