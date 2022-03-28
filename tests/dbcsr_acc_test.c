/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/
#include "acc/acc.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if !defined(NDEBUG)
#  include <assert.h>
#endif
#if defined(_OPENMP)
#  include <omp.h>
#endif

#if !defined(ACC_STRING_MAXLEN)
#  define ACC_STRING_MAXLEN 32
#endif
#if !defined(ACC_STREAM_MAXCOUNT)
#  define ACC_STREAM_MAXCOUNT 16
#endif
#if !defined(ACC_EVENT_MAXCOUNT)
#  define ACC_EVENT_MAXCOUNT (16 * ACC_STREAM_MAXCOUNT)
#endif
#if !defined(ACC_STREAM_MAXNTH_DESTROY)
#  define ACC_STREAM_MAXNTH_DESTROY 2
#endif
#if !defined(ACC_EVENT_MAXNTH_DESTROY)
#  define ACC_EVENT_MAXNTH_DESTROY 3
#endif

#if defined(NDEBUG)
#  define ACC_CHECK(RESULT) \
    do { \
      const int acc_check_result_ = (RESULT); \
      if (EXIT_SUCCESS != acc_check_result_) exit(acc_check_result_); \
    } while (0)
#  define PRINTF(A, ...)
#else /* debug */
#  define ACC_CHECK(RESULT) assert(EXIT_SUCCESS == (RESULT))
#  define PRINTF(A, ...) printf(A, __VA_ARGS__)
#endif


/**
 * This program tests the ACC interface (include/acc.h) for adhering to expectations.
 * The expected behavior is to match the CUDA based implementation, which was available
 * first. This test program can serve as a specification for other backends such as the
 * OpenMP based backend. It may also be used to stress-test any backend including the
 * CUDA based backend for thread-safety. Thread-safety is an implicit requirement
 * induced by DBCSR (and CP2K). To test any backend (other than the OpenMP backend),
 * the Makefile must be adjusted to link with the desired backend.
 */
int main(int argc, char* argv[]) {
  const int device = (1 < argc ? atoi(argv[1]) : 0);
#if defined(_OPENMP)
  const int max_nthreads = omp_get_max_threads();
#else
  const int max_nthreads = 1;
#endif
  const int cli_nthreads = (2 < argc ? atoi(argv[2]) : max_nthreads);
  const int nthreads = ((0 < cli_nthreads && cli_nthreads <= max_nthreads) ? cli_nthreads : max_nthreads);
  int randnums[ACC_EVENT_MAXCOUNT], ndevices, priomin, priomax, i, nt;
  void *event[ACC_EVENT_MAXCOUNT], *s = NULL;
  const size_t mem_alloc = (16 /*MB*/ << 20);
  size_t mem_free, mem_total, mem_chunk;
  void *host_mem = NULL, *dev_mem = NULL;

  for (i = 0; i < ACC_EVENT_MAXCOUNT; ++i) {
    randnums[i] = rand();
  }

  /* allow get_ndevices/set_active_device before init */
  ACC_CHECK(c_dbcsr_acc_get_ndevices(&ndevices));
  if (0 <= device && device < ndevices) { /* not an error */
    ACC_CHECK(c_dbcsr_acc_set_active_device(device));
  }
  ACC_CHECK(c_dbcsr_acc_init());
  ACC_CHECK(c_dbcsr_acc_get_ndevices(&ndevices));
  PRINTF("ndevices: %i\n", ndevices);
  /* continue tests even with no device */
  if (0 <= device && device < ndevices) { /* not an error */
    ACC_CHECK(c_dbcsr_acc_set_active_device(device));
  }

  if (0 < ndevices) {
    ACC_CHECK(c_dbcsr_acc_dev_mem_info(&mem_free, &mem_total));
    ACC_CHECK(mem_free <= mem_total ? EXIT_SUCCESS : EXIT_FAILURE);
    PRINTF("device memory: free=%i MB total=%i MB\n", (int)(mem_free >> 20), (int)(mem_total >> 20));

    ACC_CHECK(c_dbcsr_acc_stream_priority_range(&priomin, &priomax));
    PRINTF("stream priority: lowest=%i highest=%i\n", priomin, priomax);

    for (i = 0; i < ACC_EVENT_MAXCOUNT; ++i) {
      event[i] = NULL;
    }

    /* create stream with NULL-name and low priority */
    ACC_CHECK(c_dbcsr_acc_stream_create(&s, NULL /*name*/, priomin));
    ACC_CHECK(c_dbcsr_acc_stream_destroy(s));
    /* create stream with empty name and medium priority */
    ACC_CHECK(c_dbcsr_acc_stream_create(&s, "", (priomin + priomax) / 2));
    ACC_CHECK(c_dbcsr_acc_stream_destroy(s));
    /* destroying NULL-stream shall be valid (just like delete/free) */
    ACC_CHECK(c_dbcsr_acc_stream_destroy(NULL));
    ACC_CHECK(c_dbcsr_acc_event_destroy(NULL));

#if defined(_OPENMP)
#  pragma omp parallel for num_threads(nthreads) private(i)
#endif
    for (i = 0; i < ACC_EVENT_MAXCOUNT; ++i) {
      const int r = randnums[i] % ACC_EVENT_MAXCOUNT;
      ACC_CHECK(c_dbcsr_acc_event_create(event + i));
      if (ACC_EVENT_MAXNTH_DESTROY * r < ACC_EVENT_MAXCOUNT) {
        void* const ei = event[i];
        event[i] = NULL;
        ACC_CHECK(c_dbcsr_acc_event_destroy(ei));
      }
    }
#if defined(_OPENMP)
#  pragma omp parallel for num_threads(nthreads) private(i)
#endif
    for (i = 0; i < ACC_EVENT_MAXCOUNT; ++i) {
      if (NULL == event[i]) {
        ACC_CHECK(c_dbcsr_acc_event_create(event + i));
      }
      ACC_CHECK(c_dbcsr_acc_event_destroy(event[i]));
    }
#if defined(_OPENMP)
#  pragma omp parallel for num_threads(nthreads) private(i)
#endif
    for (i = 0; i < ACC_EVENT_MAXCOUNT; ++i) ACC_CHECK(c_dbcsr_acc_event_create(event + i));

    for (i = 0; i < ACC_EVENT_MAXCOUNT; ++i) {
      c_dbcsr_acc_bool_t has_occurred = 0;
      ACC_CHECK(c_dbcsr_acc_event_query(event[i], &has_occurred));
      ACC_CHECK(has_occurred ? EXIT_SUCCESS : EXIT_FAILURE);
    }

    ACC_CHECK(c_dbcsr_acc_stream_create(&s, "stream", priomax));
    if (NULL != s) {
      ACC_CHECK(c_dbcsr_acc_host_mem_allocate(&host_mem, mem_alloc, s));
      ACC_CHECK(c_dbcsr_acc_stream_sync(s)); /* wait for completion */
      memset(host_mem, 0xFF, mem_alloc); /* non-zero pattern */
    }

    ACC_CHECK(c_dbcsr_acc_dev_mem_allocate(&dev_mem, mem_alloc));
    nt = (nthreads < ACC_EVENT_MAXCOUNT ? nthreads : ACC_EVENT_MAXCOUNT);
    mem_chunk = (mem_alloc + nt - 1) / nt;
#if defined(_OPENMP)
#  pragma omp parallel num_threads(nt)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      const size_t offset = tid * mem_chunk, mem_rest = mem_alloc - offset;
      const size_t size = (mem_chunk <= mem_rest ? mem_chunk : mem_rest);
      c_dbcsr_acc_bool_t has_occurred = 0;
      ACC_CHECK(c_dbcsr_acc_memset_zero(dev_mem, offset, size, s));
      /* can enqueue multiple/duplicate copies for the same memory region */
      ACC_CHECK(c_dbcsr_acc_memcpy_d2h(dev_mem, host_mem, mem_alloc, s));
      ACC_CHECK(c_dbcsr_acc_event_query(event[tid], &has_occurred));
      /* unrecorded event has no work to wait for, hence it occurred */
      ACC_CHECK(has_occurred ? EXIT_SUCCESS : EXIT_FAILURE);
      ACC_CHECK(c_dbcsr_acc_event_record(event[tid], s));
      ACC_CHECK(c_dbcsr_acc_stream_wait_event(s, event[tid]));
      ACC_CHECK(c_dbcsr_acc_event_synchronize(event[tid]));
      ACC_CHECK(c_dbcsr_acc_event_query(event[tid], &has_occurred));
      ACC_CHECK(has_occurred ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    /* validate backwards from where the last transfers occurred */
    for (i = (int)(mem_alloc - 1); 0 <= i; --i) {
      ACC_CHECK(0 == ((char*)host_mem)[i] ? EXIT_SUCCESS : EXIT_FAILURE);
    }

#if defined(_OPENMP)
#  pragma omp parallel for num_threads(nthreads) private(i)
#endif
    for (i = 0; i < ACC_EVENT_MAXCOUNT; ++i) ACC_CHECK(c_dbcsr_acc_event_destroy(event[i]));
  }

  ACC_CHECK(c_dbcsr_acc_dev_mem_deallocate(dev_mem));
  if (NULL != s) ACC_CHECK(c_dbcsr_acc_host_mem_deallocate(host_mem, s));
  ACC_CHECK(c_dbcsr_acc_stream_destroy(s));
  c_dbcsr_acc_clear_errors(); /* no result code */
  ACC_CHECK(c_dbcsr_acc_finalize());

  return EXIT_SUCCESS;
}
