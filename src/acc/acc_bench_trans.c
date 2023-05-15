/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/
#include "acc_libsmm.h"
#include "acc_bench.h"
#include <string.h>
#include <stdio.h>

#if defined(__LIBXSMM)
#  if defined(LIBXSMM_DEFAULT_CONFIG)
#    include <libxsmm_source.h>
#  else
#    include <libxsmm.h>
#    if !defined(LIBXSMM_TIMER_H)
#      include <utils/libxsmm_timer.h>
#    endif
#    if !defined(LIBXSMM_SYNC_H)
#      include <libxsmm_sync.h>
#    endif
#  endif
#  if defined(LIBXSMM_VERSION_NUMBER) && LIBXSMM_VERSION4(1, 17, 0, 0) < LIBXSMM_VERSION_NUMBER
#    define USE_LIBXSMM
#  endif
#endif

#if defined(USE_LIBXSMM)
#  if defined(_OPENMP)
#    define ACC_BENCH_ITRANSBATCH(A, ...) libxsmm_itrans_batch_omp(A, __VA_ARGS__)
#  else
#    define ACC_BENCH_ITRANSBATCH(A, ...) libxsmm_itrans_batch(A, __VA_ARGS__, 0, 1)
#  endif
#  if !defined(SHUFFLE) && 0
#    define SHUFFLE
#  endif
#endif

#if !defined(ELEM_TYPE)
#  define ELEM_TYPE double
#endif
#if !defined(MAX_KERNEL_DIM)
#  define MAX_KERNEL_DIM 80
#endif
#if !defined(ALIGNMENT)
#  define ALIGNMENT 64
#endif
#if !defined(PRIORITY)
#  define PRIORITY
#endif
#if !defined(WARMUP)
#  define WARMUP 2
#endif

#define MAX(A, B) ((B) < (A) ? (A) : (B))
#define ROUNDUP2(N, NPOT) ((((unsigned long long)N) + ((NPOT)-1)) & ~((NPOT)-1))
#define CHECK(EXPR, RPTR) \
  if ((NULL != ((const void*)(RPTR)) && EXIT_SUCCESS != *((const int*)(RPTR))) || \
      EXIT_SUCCESS != (NULL != ((const void*)(RPTR)) ? (*((int*)(RPTR)) = (EXPR)) : (EXPR))) \
  assert(NULL == ((const void*)(RPTR)) || -1 == *((int*)(RPTR)))


static void swap(int* m, int* n) {
  int tmp = *m;
  *m = *n;
  *n = tmp;
}


int main(int argc, char* argv[]) {
  int arg = 1;
  const char* const x1 = (arg < argc ? strchr(argv[arg], 'x') : NULL);
  const int inr = ((arg < argc && NULL == x1) ? atoi(argv[arg++]) : 0);
  const int iss = ((arg < argc && NULL == x1) ? atoi(argv[arg++]) : 0);
  const char* const x2 = ((arg < argc && NULL == x1) ? strchr(argv[arg], 'x') : NULL);
  const int ism = (arg < argc ? atoi(argv[arg++]) : 0);
  const int isn = ((NULL == x1 && NULL == x2) ? (arg < argc
                                                    /* accept "M N K" as well as "MxNxK" */
                                                    ? atoi(argv[arg++])
                                                    : 0)
                                              : atoi((NULL != x1 ? x1 : x2) + 1));
  const int iof = (arg < argc ? atoi(argv[arg++]) : 0);
  const int nrepeat = (0 < inr ? inr : 5);
  const int nodd = (0 < nrepeat ? ((nrepeat & 1 /*odd*/) ? nrepeat : (nrepeat - 1)) : 1);
  const int stack_size = (0 < iss ? iss : 30000);
  const int m = (0 < ism ? ism : 23);
  const int n = (0 < isn ? isn : m);
  const int offset = (0 < iof ? iof : 0);
  const int offset_stack_size = offset + stack_size;
#if defined(ALIGNMENT) && (0 < ALIGNMENT)
  const int mn = (int)ROUNDUP2(sizeof(ELEM_TYPE) * m, ALIGNMENT) * n / sizeof(ELEM_TYPE);
#else
  const int mn = m * n;
#endif
#if defined(SHUFFLE)
  const size_t shuffle = libxsmm_shuffle((unsigned int)offset_stack_size);
#endif
#if defined(PRIORITY)
  int priomin, priomax;
#endif
#if defined(WARMUP) && (0 < WARMUP) && !defined(_DEBUG)
  const int warmup = MAX(WARMUP, 2) / 2 * 2;
#else
  const int warmup = 0;
#endif
  const char* const env_device = getenv("DEVICE");
  const int device = ((NULL == env_device || '\0' == *env_device) ? 0 : atoi(env_device));
  int *stack_hst = NULL, *stack_dev = NULL;
  ELEM_TYPE *mat_hst = NULL, *mat_dev = NULL;
  int result = EXIT_SUCCESS, ndevices = 0, r, i, mm = m, nn = n;
  void* stream = NULL;
#if defined(USE_LIBXSMM)
  libxsmm_timer_tickint start;
  double duration;
#endif
  assert(m <= (mn / n) && 0 == (mn % n));
  CHECK(c_dbcsr_acc_init(), &result);
  /* note: libsmm_acc_init() may imply acc_init() */
  CHECK(libsmm_acc_init(), &result);
  if (EXIT_SUCCESS == result) {
    result = c_dbcsr_acc_get_ndevices(&ndevices);
    if (0 < ndevices && (0 == device || EXIT_SUCCESS == c_dbcsr_acc_set_active_device(device))) {
#if defined(_DEBUG)
      fprintf(stderr, "Activated device %i of %i (device%i).\n", device + 1, ndevices, device);
#endif
    }
    else {
      if (0 >= ndevices) {
        fprintf(stderr, "No ACC-device found!\n");
      }
      else {
        fprintf(stderr, "Failed to activate device %i of %i!\n", device, ndevices);
      }
#if !defined(__CUDA)
      CHECK(libsmm_acc_finalize(), NULL);
#endif
      CHECK(c_dbcsr_acc_finalize(), NULL);
      return result;
    }
  }
  else {
    fprintf(stderr, "ACC initialization failed!\n");
#if !defined(__CUDA)
    CHECK(libsmm_acc_finalize(), NULL);
#endif
    CHECK(c_dbcsr_acc_finalize(), NULL);
    return result;
  }
  printf("%s%s%i %i %i %i\n", 0 < argc ? argv[0] : "", 0 < argc ? " " : "", nrepeat, stack_size, m, n);
  printf("typename (id=%i): %s\n", DBCSR_TYPE(ELEM_TYPE), DBCSR_STRINGIFY(ELEM_TYPE));
  if (MAX_KERNEL_DIM < m || MAX_KERNEL_DIM < n) {
    fprintf(stderr, "Matrix shape exceeds MAX_KERNEL_DIM!\n");
    result = EXIT_FAILURE;
  }
#if defined(PRIORITY)
  CHECK(c_dbcsr_acc_stream_priority_range(&priomin, &priomax), &result);
  CHECK(c_dbcsr_acc_stream_create(&stream, "stream", (priomin + priomax) / 2), &result);
#else
  CHECK(c_dbcsr_acc_stream_create(&stream, "stream", -1 /*default priority*/), &result);
#endif
  CHECK(c_dbcsr_acc_host_mem_allocate((void**)(void*)&mat_hst, sizeof(ELEM_TYPE) * mn * offset_stack_size, stream), &result);
  CHECK(c_dbcsr_acc_host_mem_allocate((void**)(void*)&stack_hst, sizeof(int) * offset_stack_size, stream), &result);
  CHECK(c_dbcsr_acc_stream_sync(stream), &result); /* ensure host-data is allocated */
  if (NULL != mat_hst && NULL != stack_hst) {
#if defined(_OPENMP)
#  pragma omp parallel for
#endif
    for (i = 0; i < offset_stack_size; ++i) { /* initialize matrices and indexes */
#if defined(SHUFFLE)
      const int j = mn * (int)((shuffle * i) % offset_stack_size);
#else
      const int j = mn * i;
#endif
      INIT_MAT(ELEM_TYPE, i /*seed*/, &mat_hst[i * mn], m, n, 1.0 /*scale*/);
      stack_hst[i] = j;
    }
  }
  CHECK(c_dbcsr_acc_dev_mem_allocate((void**)(void*)&mat_dev, sizeof(ELEM_TYPE) * mn * offset_stack_size), &result);
  CHECK(c_dbcsr_acc_dev_mem_allocate((void**)(void*)&stack_dev, sizeof(int) * offset_stack_size), &result);
#if defined(USE_LIBXSMM)
  CHECK(c_dbcsr_acc_stream_sync(stream), &result);
  start = libxsmm_timer_tick();
#endif
  CHECK(c_dbcsr_acc_memcpy_h2d(mat_hst, mat_dev, sizeof(ELEM_TYPE) * mn * offset_stack_size, stream), &result);
  CHECK(c_dbcsr_acc_memcpy_h2d(stack_hst, stack_dev, sizeof(int) * offset_stack_size, stream), &result);
#if defined(USE_LIBXSMM)
  CHECK(c_dbcsr_acc_stream_sync(stream), &result);
  if (NULL != mat_hst && NULL != stack_hst) {
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
    printf("copy-in: %.2g ms %.1f GB/s\n", 1000.0 * duration,
      (sizeof(ELEM_TYPE) * mn + sizeof(int)) * offset_stack_size / (duration * (1ULL << 30)));
  }
#endif
  /* warmup execution and prebuild JIT kernels */
  for (r = 0; r < warmup / 2; ++r) {
    CHECK(
      libsmm_acc_transpose(stack_dev, offset, stack_size, mat_dev, DBCSR_TYPE(ELEM_TYPE), m, n, MAX_KERNEL_DIM, stream), &result);
    CHECK(
      libsmm_acc_transpose(stack_dev, offset, stack_size, mat_dev, DBCSR_TYPE(ELEM_TYPE), n, m, MAX_KERNEL_DIM, stream), &result);
  }
#if defined(USE_LIBXSMM)
  CHECK(c_dbcsr_acc_stream_sync(stream), &result);
  start = libxsmm_timer_tick();
#endif
  for (r = 0; r < nodd; ++r) {
    CHECK(
      libsmm_acc_transpose(stack_dev, offset, stack_size, mat_dev, DBCSR_TYPE(ELEM_TYPE), mm, nn, MAX_KERNEL_DIM, stream), &result);
    swap(&mm, &nn);
  }
#if defined(USE_LIBXSMM)
  CHECK(c_dbcsr_acc_stream_sync(stream), &result);
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
  if (EXIT_SUCCESS == result) {
    assert(0 < nodd && (nodd & 1 /*odd*/));
    printf("device: %.2g ms %.1f GB/s\n", 1000.0 * duration / nodd,
      (sizeof(ELEM_TYPE) * mn + sizeof(int)) * offset_stack_size / (duration * (1ULL << 30) / nodd));
    mm = m;
    nn = n;
    start = libxsmm_timer_tick();
    for (r = 0; r < nodd; ++r) {
      ACC_BENCH_ITRANSBATCH(
        mat_hst, sizeof(ELEM_TYPE), mm, nn, mm, nn, 0 /*index_base*/, sizeof(int) /*index_stride*/, stack_hst + offset, stack_size);
      swap(&mm, &nn);
    }
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
    printf("host: %.2g ms %.1f GB/s\n", 1000.0 * duration / nodd,
      (sizeof(ELEM_TYPE) * mn + sizeof(int)) * offset_stack_size / (duration * (1ULL << 30) / nodd));
    /* transfer result from device to host for validation */
    CHECK(c_dbcsr_acc_memcpy_d2h(mat_dev, mat_hst, sizeof(ELEM_TYPE) * mn * offset_stack_size, stream), &result);
    CHECK(c_dbcsr_acc_stream_sync(stream), &result);
    if (EXIT_SUCCESS == result) {
      unsigned int nerrors = 0;
      for (i = offset; i < offset_stack_size; ++i) {
        ELEM_TYPE gold[MAX_KERNEL_DIM * MAX_KERNEL_DIM];
        const ELEM_TYPE* const test = mat_hst + mn * i;
        INIT_MAT(ELEM_TYPE, i /*seed*/, gold, m, n, 1.0 /*scale*/);
        libxsmm_itrans(gold, sizeof(ELEM_TYPE), m, n, m, n);
        for (r = 0; r < (m * n); ++r) {
          if (gold[r] != test[r]) {
            ++nerrors;
            break;
          }
        }
      }
      printf("errors: %u\n", nerrors);
      if (0 != nerrors) result = EXIT_FAILURE;
    }
  }
#endif
  CHECK(c_dbcsr_acc_host_mem_deallocate(stack_hst, stream), NULL);
  CHECK(c_dbcsr_acc_host_mem_deallocate(mat_hst, stream), NULL);
  CHECK(c_dbcsr_acc_dev_mem_deallocate(stack_dev), NULL);
  CHECK(c_dbcsr_acc_dev_mem_deallocate(mat_dev), NULL);
  CHECK(c_dbcsr_acc_stream_destroy(stream), NULL);
#if !defined(__CUDA)
  CHECK(libsmm_acc_finalize(), NULL);
#endif
  CHECK(c_dbcsr_acc_finalize(), NULL);
  if (EXIT_SUCCESS != result) {
    if (-1 != result) {
      fprintf(stderr, "FAILED\n");
    }
    else {
      fprintf(stderr, "Kernel not suitable!\n");
      result = EXIT_SUCCESS;
    }
  }
  return result;
}
