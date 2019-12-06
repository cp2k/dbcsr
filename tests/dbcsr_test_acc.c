/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#include "../src/acc/include/acc.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if !defined(NDEBUG)
# include <assert.h>
#endif
#if defined(_OPENMP)
# include <omp.h>
#endif

#if !defined(ACC_STRING_MAXLEN)
# define ACC_STRING_MAXLEN 32
#endif
#if !defined(ACC_STREAM_MAXCOUNT)
# define ACC_STREAM_MAXCOUNT 16
#endif
#if !defined(ACC_EVENT_MAXCOUNT)
# define ACC_EVENT_MAXCOUNT (16*ACC_STREAM_MAXCOUNT)
#endif
#if !defined(ACC_STREAM_MAXNTH_DESTROY)
# define ACC_STREAM_MAXNTH_DESTROY 2
#endif
#if !defined(ACC_EVENT_MAXNTH_DESTROY)
# define ACC_EVENT_MAXNTH_DESTROY 3
#endif

#if defined(NDEBUG)
# define ACC_CHECK(RESULT) do { const int acc_check_result_ = (RESULT); \
    if (EXIT_SUCCESS != acc_check_result_) exit(acc_check_result_); \
  } while(0)
# define PRINTF(A, ...)
#else /* debug */
# define ACC_CHECK(RESULT) assert(EXIT_SUCCESS == (RESULT))
# define PRINTF(A, ...) printf(A, __VA_ARGS__)
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
int main(int argc, char* argv[])
{
  const int device = (1 < argc ? atoi(argv[1]) : 0);
#if defined(_OPENMP)
  const int max_nthreads = omp_get_max_threads();
#else
  const int max_nthreads = 1;
#endif
  const int cli_nthreads = (2 < argc ? atoi(argv[2]) : max_nthreads);
  const int nthreads = ((0 < cli_nthreads && cli_nthreads <= max_nthreads) ? cli_nthreads : max_nthreads);
  int priority[ACC_STREAM_MAXCOUNT], priomin, priomax, priospan;
  int randnums[ACC_EVENT_MAXCOUNT], ndevices, i, n;
  acc_stream_t *stream[ACC_STREAM_MAXCOUNT], *s;
  acc_event_t *event[ACC_EVENT_MAXCOUNT];
  const size_t mem_alloc = (16/*MB*/ << 20);
  const size_t mem_chunk = (mem_alloc + nthreads - 1) / nthreads;
  size_t mem_free, mem_total;
  void *host_mem, *dev_mem;

  for (i = 0; i < ACC_EVENT_MAXCOUNT; ++i) {
    randnums[i] = rand();
  }

  ACC_CHECK(acc_init());
  ACC_CHECK(acc_get_ndevices(&ndevices));
  PRINTF("ndevices: %i\n", ndevices);
  /* continue tests even with no device */
  if (0 <= device && device < ndevices) { /* not an error */
    ACC_CHECK(acc_set_active_device(device));
  }

  ACC_CHECK(acc_dev_mem_info(&mem_free, &mem_total));
  ACC_CHECK(mem_free <= mem_total ? EXIT_SUCCESS : EXIT_FAILURE);
  PRINTF("device memory: free=%i MB total=%i MB\n",
    (int)(mem_free >> 20), (int)(mem_total >> 20));

  ACC_CHECK(acc_stream_priority_range(&priomin, &priomax));
  priospan = 1 + priomax - priomin;
  PRINTF("stream priority: min=%i max=%i%s\n", priomin, priomax,
    0 < priospan ? "" : " <-- WARNING: inconsistent values");

  for (i = 0; i < ACC_STREAM_MAXCOUNT; ++i) {
    priority[i] = priomin + (randnums[i%ACC_STREAM_MAXCOUNT] % priospan);
    stream[i] = NULL;
  }
  for (i = 0; i < ACC_EVENT_MAXCOUNT; ++i) {
    event[i] = NULL;
  }

  ACC_CHECK(acc_stream_destroy(NULL));
#if defined(_OPENMP)
# pragma omp parallel for num_threads(nthreads) private(i)
#endif
  for (i = 0; i < ACC_STREAM_MAXCOUNT; ++i) {
    const int r = randnums[i%ACC_STREAM_MAXCOUNT] % ACC_STREAM_MAXCOUNT;
    char name[ACC_STRING_MAXLEN]; /* thread-local */
    const int n = sprintf(name, "%i", i);
    ACC_CHECK((0 <= n && n < ACC_STRING_MAXLEN) ? EXIT_SUCCESS : EXIT_FAILURE);
    ACC_CHECK(acc_stream_create(stream + i, name, priority[i]));
    if (ACC_STREAM_MAXNTH_DESTROY * r < ACC_STREAM_MAXCOUNT) {
      ACC_CHECK(acc_stream_destroy(stream[i]));
      stream[i] = NULL;
    }
  }

#if defined(_OPENMP)
# pragma omp parallel for num_threads(nthreads) private(i)
#endif
  for (i = 0; i < ACC_STREAM_MAXCOUNT; ++i) {
    if (NULL == stream[i]) {
      char name[ACC_STRING_MAXLEN]; /* thread-local */
      const int n = sprintf(name, "%i", i);
      ACC_CHECK((0 <= n && n < ACC_STRING_MAXLEN) ? EXIT_SUCCESS : EXIT_FAILURE);
      ACC_CHECK(acc_stream_create(stream + i, name, priority[i]));
    }
    ACC_CHECK(acc_stream_destroy(stream[i]));
  }

  ACC_CHECK(acc_event_destroy(NULL));
#if defined(_OPENMP)
# pragma omp parallel for num_threads(nthreads) private(i)
#endif
  for (i = 0; i < ACC_EVENT_MAXCOUNT; ++i) {
    const int r = randnums[i%ACC_EVENT_MAXCOUNT] % ACC_EVENT_MAXCOUNT;
    ACC_CHECK(acc_event_create(event + i));
    if (ACC_EVENT_MAXNTH_DESTROY * r < ACC_EVENT_MAXCOUNT) {
      ACC_CHECK(acc_event_destroy(event[i]));
      event[i] = NULL;
    }
  }

#if defined(_OPENMP)
# pragma omp parallel for num_threads(nthreads) private(i)
#endif
  for (i = 0; i < ACC_EVENT_MAXCOUNT; ++i) {
    if (NULL == event[i]) {
      ACC_CHECK(acc_event_create(event + i));
    }
    ACC_CHECK(acc_event_destroy(event[i]));
  }

  n = (nthreads <= ACC_EVENT_MAXCOUNT ? nthreads : ACC_EVENT_MAXCOUNT);
#if defined(_OPENMP)
# pragma omp parallel for num_threads(n) private(i)
#endif
  for (i = 0; i < n; ++i) ACC_CHECK(acc_event_create(event + i));
  for (i = 0; i < n; ++i) {
    acc_bool_t has_occurred = 0;
    ACC_CHECK(acc_event_query(event[i], &has_occurred));
    ACC_CHECK(has_occurred ? EXIT_SUCCESS : EXIT_FAILURE);
  }

  ACC_CHECK(acc_stream_create(&s, "stream", priomin));
  ACC_CHECK(acc_host_mem_allocate(&host_mem, mem_alloc, s));
  ACC_CHECK(acc_dev_mem_allocate(&dev_mem, mem_alloc));
  ACC_CHECK(acc_stream_sync(s)); /* wait for completion */
  memset(host_mem, 0xFF, mem_alloc); /* non-zero pattern */

#if defined(_OPENMP)
# pragma omp parallel num_threads(n)
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    const size_t offset = tid * mem_chunk, mem_rest = mem_alloc - offset;
    const size_t size = (mem_chunk <= mem_rest ? mem_chunk : mem_rest);
    acc_bool_t has_occurred = 0;
    ACC_CHECK(acc_memset_zero(dev_mem, offset, size, s));
    ACC_CHECK(acc_memcpy_d2h(dev_mem, host_mem, mem_alloc, s));
    ACC_CHECK(acc_event_query(event[tid], &has_occurred));
    /* unrecorded event has no work to wait for, hence it occurred */
    ACC_CHECK(has_occurred ? EXIT_SUCCESS : EXIT_FAILURE);
    ACC_CHECK(acc_event_record(event[tid], s));
    ACC_CHECK(acc_stream_wait_event(s, event[tid]));
    ACC_CHECK(acc_event_query(event[tid], &has_occurred));
    if (!has_occurred) ACC_CHECK(acc_event_synchronize(event[tid]));
    ACC_CHECK(acc_event_query(event[tid], &has_occurred));
    ACC_CHECK(has_occurred ? EXIT_SUCCESS : EXIT_FAILURE);
  }

  for (i = 0; i < (int)mem_alloc; ++i) {
    ACC_CHECK(0 == ((char*)host_mem)[i] ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  ACC_CHECK(acc_dev_mem_deallocate(dev_mem));
  ACC_CHECK(acc_host_mem_deallocate(host_mem, s));
  ACC_CHECK(acc_stream_destroy(s));

#if defined(_OPENMP)
# pragma omp parallel for num_threads(n) private(i)
#endif
  for (i = 0; i < n; ++i) ACC_CHECK(acc_event_destroy(event[i]));

  acc_clear_errors(); /* no result code */
  ACC_CHECK(acc_finalize());

  return EXIT_SUCCESS;
}
