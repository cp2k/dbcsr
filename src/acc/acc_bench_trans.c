/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#include "acc_libsmm.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

#if defined(__LIBXSMM)
# include <libxsmm.h>
# define USE_LIBXSMM
# if !defined(SHUFFLE) && 0
#   define SHUFFLE
# endif
#endif

#if !defined(ELEM_TYPE)
# define ELEM_TYPE double
#endif
#if !defined(MAX_KERNEL_DIM)
# define MAX_KERNEL_DIM 80
#endif
#if !defined(ALIGNMENT)
# define ALIGNMENT 64
#endif
#if !defined(PRIORITY)
# define PRIORITY
#endif
#if !defined(WARMUP)
# define WARMUP 2
#endif

#define MAX(A, B) ((B) < (A) ? (A) : (B))
#define ROUNDUP2(N, NPOT) ((((unsigned long long)N) + ((NPOT) - 1)) & ~((NPOT) - 1))
#define CHECK(EXPR, RPTR) if ((NULL != ((const void*)(RPTR)) && EXIT_SUCCESS != *((const int*)(RPTR))) || \
  EXIT_SUCCESS != (NULL != ((const void*)(RPTR)) ? (*((int*)(RPTR)) = (EXPR)) : (EXPR))) assert(0)


#if defined(_DEBUG) && defined(USE_LIBXSMM)
static void print(FILE* ostream, const char* label, const ELEM_TYPE* mat, int m, int n);
#endif

static void init(int seed, ELEM_TYPE* dst, int m, int n);
static void swap(int* m, int* n) { int tmp = *m; *m = *n; *n = tmp; }


int main(int argc, char* argv[])
{
  const int nrepeat = (1 < argc ? atoi(argv[1]) : 5);
  const int nodd = (0 < nrepeat ? ((nrepeat & 1/*odd*/) ? nrepeat : (nrepeat - 1)) : 1);
  const int stack_size = (2 < argc ? atoi(argv[2]) : 30000);
  const int m = (3 < argc ? atoi(argv[3]) : 23);
  const int n = (4 < argc ? atoi(argv[4]) : m);
  const int offset = (5 < argc ? atoi(argv[5]) : 0);
  const int offset_stack_size = offset + stack_size;
#if defined(ALIGNMENT) && (0 < ALIGNMENT)
  const int mn = (int)ROUNDUP2(sizeof(ELEM_TYPE) * m, ALIGNMENT) * n / sizeof(ELEM_TYPE);
#else
  const int mn = m * n;
#endif
#if defined(SHUFFLE)
  const size_t shuffle = libxsmm_shuffle((unsigned int)offset_stack_size);
#endif
#if defined(WARMUP) && (0 < WARMUP) && !defined(_DEBUG)
  const int warmup = MAX(WARMUP, 2) / 2 * 2;
#else
  const int warmup = 0;
#endif
#if defined(PRIORITY)
  int priomin, priomax;
#endif
  int *stack_hst = NULL, *stack_dev = NULL;
  ELEM_TYPE *mat_hst = NULL, *mat_dev = NULL;
  int result = EXIT_SUCCESS, ndevices = 0, r, i, mm = m, nn = n;
  void *stream = NULL;
#if defined(USE_LIBXSMM)
  libxsmm_timer_tickint start;
  double duration;
#endif
  assert(m <= (mn / n) && 0 == (mn % n));
  printf("%s%s%i %i %i %i\n", 0 < argc ? argv[0] : "", 0 < argc ? " " : "", nrepeat, stack_size, m, n);
  CHECK(acc_init(), &result);
  /* note: libsmm_acc_init() may imply acc_init() */
  CHECK(libsmm_acc_init(), &result);
  CHECK(acc_get_ndevices(&ndevices), &result);
  if (0 < ndevices) {
#if defined(_DEBUG)
    fprintf(stderr, "number of devices found: %i\n", ndevices);
#endif
  }
  else {
#if defined(_DEBUG)
    fprintf(stderr, "Error: no device found!\n");
#endif
#if !defined(__CUDA)
    CHECK(libsmm_acc_finalize(), NULL);
#endif
    CHECK(acc_finalize(), NULL);
    return result;
  }
  printf("typename (id=%i): %s\n", DBCSR_TYPE(ELEM_TYPE), DBCSR_STRINGIFY(ELEM_TYPE));
#if defined(PRIORITY)
  CHECK(acc_stream_priority_range(&priomin, &priomax), &result);
  CHECK(acc_stream_create(&stream, "stream", (priomin + priomax) / 2), &result);
#else
  CHECK(acc_stream_create(&stream, "stream", -1/*default priority*/), &result);
#endif
  CHECK(acc_host_mem_allocate((void**)&mat_hst, sizeof(ELEM_TYPE) * mn * offset_stack_size, stream), &result);
  CHECK(acc_host_mem_allocate((void**)&stack_hst, sizeof(int) * offset_stack_size, stream), &result);
  CHECK(acc_stream_sync(stream), &result); /* ensure host-data is allocated */
  for (i = 0; i < offset_stack_size; ++i) { /* initialize matrices */
    init(i/*seed*/, &mat_hst[i*mn], m, n);
  }
  for (i = 0; i < offset_stack_size; ++i) { /* initialize indexes */
#if defined(SHUFFLE)
    const int j = mn * (int)((shuffle * i) % offset_stack_size);
#else
    const int j = mn * i;
#endif
    stack_hst[i] = j;
  }
  CHECK(acc_dev_mem_allocate((void**)&mat_dev, sizeof(ELEM_TYPE) * mn * offset_stack_size), &result);
  CHECK(acc_dev_mem_allocate((void**)&stack_dev, sizeof(int) * offset_stack_size), &result);
#if defined(USE_LIBXSMM)
  CHECK(acc_stream_sync(stream), &result);
  start = libxsmm_timer_tick();
#endif
  CHECK(acc_memcpy_h2d(mat_hst, mat_dev, sizeof(ELEM_TYPE) * mn * offset_stack_size, stream), &result);
  CHECK(acc_memcpy_h2d(stack_hst, stack_dev, sizeof(int) * offset_stack_size, stream), &result);
#if defined(USE_LIBXSMM)
  CHECK(acc_stream_sync(stream), &result);
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
  printf("copy-in: %.1f ms %.1f GB/s\n", 1000.0 * duration,
    (sizeof(ELEM_TYPE) * mn + sizeof(int))
      * offset_stack_size / (duration * (1ULL << 30)));
#endif
  /* warmup execution and prebuild JIT kernels */
  for (r = 0; r < warmup / 2; ++r) {
    CHECK(libsmm_acc_transpose(stack_dev, offset, stack_size, mat_dev,
      DBCSR_TYPE(ELEM_TYPE), m, n, MAX_KERNEL_DIM, stream), &result);
    CHECK(libsmm_acc_transpose(stack_dev, offset, stack_size, mat_dev,
      DBCSR_TYPE(ELEM_TYPE), n, m, MAX_KERNEL_DIM, stream), &result);
  }
#if defined(USE_LIBXSMM)
  CHECK(acc_stream_sync(stream), &result);
  start = libxsmm_timer_tick();
#endif
  for (r = 0; r < nodd; ++r) {
    CHECK(libsmm_acc_transpose(stack_dev, offset, stack_size, mat_dev,
      DBCSR_TYPE(ELEM_TYPE), mm, nn, MAX_KERNEL_DIM, stream), &result);
    swap(&mm, &nn);
  }
#if defined(USE_LIBXSMM)
  CHECK(acc_stream_sync(stream), &result);
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
  if (EXIT_SUCCESS == result) {
    assert(0 < nodd && (nodd & 1/*odd*/));
    printf("device: %.1f ms %.1f GB/s\n", 1000.0 * duration / nodd,
      (sizeof(ELEM_TYPE) * mn + sizeof(int))
        * offset_stack_size / (duration * (1ULL << 30) / nodd));
    mm = m; nn = n;
    start = libxsmm_timer_tick();
    for (r = 0; r < nodd; ++r) {
      libxsmm_itrans_batch_omp(mat_hst, sizeof(ELEM_TYPE), mm, nn, mm, nn,
        0/*index_base*/, sizeof(int)/*index_stride*/, stack_hst + offset, stack_size);
      swap(&mm, &nn);
    }
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
    printf("host: %.1f ms %.1f GB/s\n", 1000.0 * duration / nodd,
      (sizeof(ELEM_TYPE) * mn + sizeof(int))
        * offset_stack_size / (duration * (1ULL << 30) / nodd));
    /* transfer result from device to host for validation */
    CHECK(acc_memcpy_d2h(mat_dev, mat_hst,
      sizeof(ELEM_TYPE) * mn * offset_stack_size, stream), &result);
    CHECK(acc_stream_sync(stream), &result);
    if (EXIT_SUCCESS == result) {
      unsigned int nerrors = 0;
      for (i = offset; i < offset_stack_size; ++i) {
        ELEM_TYPE gold[MAX_KERNEL_DIM*MAX_KERNEL_DIM];
        const ELEM_TYPE *const test = mat_hst + mn * i;
        init(i/*seed*/, gold, m, n);
        libxsmm_itrans(gold, sizeof(ELEM_TYPE), m, n, m, n);
        for (r = 0; r < (m * n); ++r) {
          if (gold[r] != test[r]) {
            ++nerrors;
# if defined(_DEBUG)
            print(stderr, "gold = ", gold, n, m);
            print(stderr, "test = ", test, n, m);
            init(i/*seed*/, gold, m, n);
            print(stderr, "orig = ", gold, m, n);
            fprintf(stderr, "\n");
# endif
            break;
          }
        }
      }
      printf("errors: %u\n", nerrors);
      if (0 != nerrors) result = EXIT_FAILURE;
    }
  }
#endif
  CHECK(acc_host_mem_deallocate(stack_hst, stream), NULL);
  CHECK(acc_host_mem_deallocate(mat_hst, stream), NULL);
  CHECK(acc_dev_mem_deallocate(stack_dev), NULL);
  CHECK(acc_dev_mem_deallocate(mat_dev), NULL);
  CHECK(acc_stream_destroy(stream), NULL);
#if !defined(__CUDA)
  CHECK(libsmm_acc_finalize(), NULL);
#endif
  CHECK(acc_finalize(), NULL);
  if (EXIT_SUCCESS != result) {
    fprintf(stderr, "FAILED\n");
  }
  return result;
}


static void init(int seed, ELEM_TYPE* dst, int m, int n) {
  int i, j;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      const int k = i * m + j;
      dst[k] = (ELEM_TYPE)((seed + 1) * (k + 1));
    }
  }
}


#if defined(_DEBUG) && defined(USE_LIBXSMM)
static void print(FILE* ostream, const char* label, const ELEM_TYPE* mat, int m, int n)
{
  int i, j;
  const char *const s = (NULL != label ? label : "");
  const int len = (int)strlen(s);
  for (i = 0; i < n; ++i) {
    if (0 < i) fprintf(ostream, "%*s", len, " "); else fprintf(ostream, "%s", s);
    for (j = 0; j < m; ++j) {
      fprintf(ostream, "%.2f ", mat[i*m+j]);
    }
    fprintf(ostream, "\n");
  }
}
#endif
