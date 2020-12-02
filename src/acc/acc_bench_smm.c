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
#if !defined(WARMUP)
# define WARMUP 2
#endif

#define MIN(A, B) ((A) < (B) ? (A) : (B))
#define MAX(A, B) ((B) < (A) ? (A) : (B))
#define ROUNDUP2(N, NPOT) ((((unsigned long long)N) + ((NPOT) - 1)) & ~((NPOT) - 1))
#define CHECK(EXPR, RPTR) if ((NULL != ((const void*)(RPTR)) && EXIT_SUCCESS != *((const int*)(RPTR))) || \
  EXIT_SUCCESS != (NULL != ((const void*)(RPTR)) ? (*((int*)(RPTR)) = (EXPR)) : (EXPR))) assert(0)


#if defined(_DEBUG) && defined(USE_LIBXSMM)
static void print(FILE* ostream, const char* label, const ELEM_TYPE* mat, int m, int n);
#endif

static void init(int seed, ELEM_TYPE* dst, int m, int n);
/* for comparison, adopt artificial stack-setup from other DBCSR/ACC benchmarks */
static void init_stack(int* stack, int stack_size,
  int mn, int mk, int kn, int nc, int na, int nb);


int main(int argc, char* argv[])
{
  const int nrepeat = (1 < argc ? atoi(argv[1]) : 5);
  const int stack_size = (2 < argc ? atoi(argv[2]) : 30000);
  const int m = (3 < argc ? atoi(argv[3]) : 23);
  const int n = (4 < argc ? atoi(argv[4]) : m);
  const int k = (5 < argc ? atoi(argv[5]) : m);
  const int nc = (6 < argc ? MIN(atoi(argv[6]), stack_size) : MAX(stack_size / 16, 1));
  const int na = (7 < argc ? atoi(argv[7]) : (10 * nc));
  const int nb = (8 < argc ? atoi(argv[8]) : (10 * nc));
#if defined(ALIGNMENT) && (0 < ALIGNMENT)
  const int ma = (int)ROUNDUP2(sizeof(ELEM_TYPE) * m, ALIGNMENT);
  const int ka = (int)ROUNDUP2(sizeof(ELEM_TYPE) * k, ALIGNMENT);
  const int mn = ma * n / (int)sizeof(ELEM_TYPE);
  const int mk = ma * k / (int)sizeof(ELEM_TYPE);
  const int kn = ka * n / (int)sizeof(ELEM_TYPE);
#else
  const int mn = m * n, mk = m * k, kn = k * n;
#endif
#if defined(WARMUP) && (0 < WARMUP) && !defined(_DEBUG)
  const int warmup = WARMUP;
#else
  const int warmup = 0;
#endif
  int *stack_hst = NULL, *stack_dev = NULL;
  ELEM_TYPE *amat_hst = NULL, *bmat_hst = NULL, *cmat_hst = NULL;
  ELEM_TYPE *amat_dev = NULL, *bmat_dev = NULL, *cmat_dev = NULL;
  int result = EXIT_SUCCESS, r, i;
  void *stream = NULL;
#if defined(USE_LIBXSMM)
  libxsmm_timer_tickint start;
  double duration;
#endif
  assert(m <= (mn / n) && 0 == (mn % n) && k <= (mk / k) && 0 == (mk % k) && n <= (kn / n) && 0 == (kn % n));
  printf("%s%s%i %i %i %i %i\n", 0 < argc ? argv[0] : "", 0 < argc ? " " : "", nrepeat, stack_size, m, n, k);
  CHECK(acc_init(), &result);
  CHECK(acc_stream_create(&stream, "stream", -1/*default priority*/), &result);
  CHECK(acc_host_mem_allocate((void**)&amat_hst, sizeof(ELEM_TYPE) * mk * stack_size, stream), &result);
  CHECK(acc_host_mem_allocate((void**)&bmat_hst, sizeof(ELEM_TYPE) * kn * stack_size, stream), &result);
  CHECK(acc_host_mem_allocate((void**)&cmat_hst, sizeof(ELEM_TYPE) * mn * stack_size, stream), &result);
  CHECK(acc_host_mem_allocate((void**)&stack_hst, sizeof(int) * 3 * stack_size, stream), &result);
  CHECK(acc_stream_sync(stream), &result); /* ensure host-data is allocated */
  for (i = 0; i < stack_size; ++i) { /* initialize matrices */
    init(i/*seed*/ + 42, &amat_hst[i*mk], m, k);
    init(i/*seed*/ + 24, &bmat_hst[i*kn], k, n);
  }
  init_stack(stack_hst, stack_size, mn, mk, kn, nc, na, nb);
  CHECK(acc_dev_mem_allocate((void**)&amat_dev, sizeof(ELEM_TYPE) * mk * stack_size), &result);
  CHECK(acc_dev_mem_allocate((void**)&bmat_dev, sizeof(ELEM_TYPE) * kn * stack_size), &result);
  CHECK(acc_dev_mem_allocate((void**)&cmat_dev, sizeof(ELEM_TYPE) * mn * stack_size), &result);
  CHECK(acc_dev_mem_allocate((void**)&stack_dev, sizeof(int) * 3 * stack_size), &result);
  CHECK(acc_memset_zero(cmat_dev, 0/*offset*/, sizeof(ELEM_TYPE) * mn * stack_size, stream), &result);
#if defined(USE_LIBXSMM)
  CHECK(acc_stream_sync(stream), &result);
  start = libxsmm_timer_tick();
#endif
  CHECK(acc_memcpy_h2d(amat_hst, amat_dev, sizeof(ELEM_TYPE) * mk * stack_size, stream), &result);
  CHECK(acc_memcpy_h2d(bmat_hst, bmat_dev, sizeof(ELEM_TYPE) * kn * stack_size, stream), &result);
  CHECK(acc_memcpy_h2d(stack_hst, stack_dev, sizeof(int) * 3 * stack_size, stream), &result);
#if defined(USE_LIBXSMM)
  CHECK(acc_stream_sync(stream), &result);
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
  printf("copy-in: %.1f ms %.1f GB/s\n", 1000.0 * duration,
    (sizeof(ELEM_TYPE) * (mk + kn) + sizeof(int) * 3)
      * stack_size / (duration * (1ULL << 30)));
#endif
  /* warmup execution and prebuild JIT kernels */
  for (r = 0; r < warmup; ++r) {
    CHECK(libsmm_acc_process(stack_hst, stack_dev, stack_size, 3/*nparams*/, DBCSR_TYPE(ELEM_TYPE),
      amat_dev, bmat_dev, cmat_dev, m, n, k, MAX_KERNEL_DIM, 1/*homogeneous*/, stream, stream), &result);
  }
#if defined(USE_LIBXSMM)
  CHECK(acc_stream_sync(stream), &result);
  start = libxsmm_timer_tick();
#endif
  for (r = 0; r < nrepeat; ++r) {
    /* GPU-kernel is limited to C += Ai * Bi^T (i.e., NT, for NN, all Bi must be transposed upfront) */
    CHECK(libsmm_acc_process(stack_hst, stack_dev, stack_size, 3/*nparams*/, DBCSR_TYPE(ELEM_TYPE),
      amat_dev, bmat_dev, cmat_dev, m, n, k, MAX_KERNEL_DIM, 1/*homogeneous*/, stream, stream), &result);
  }
#if defined(USE_LIBXSMM)
  CHECK(acc_stream_sync(stream), &result);
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
  if (EXIT_SUCCESS == result) {
    const char transa = 'N', transb = 'T';
    const ELEM_TYPE alpha = 1, beta = 1;
    printf("device: %.1f ms %.1f GFLOPS/s\n", 1000.0 * duration / nrepeat,
      ((size_t)2 * m * n * k) * stack_size / (duration * (1ULL << 30) / nrepeat));
    memset(cmat_hst, 0, sizeof(ELEM_TYPE) * mn * stack_size);
    start = libxsmm_timer_tick();
    for (r = 0; r < nrepeat; ++r) {
      /* CPU-kernel performs C += Ai * Bi^T to match result of GPU-kernel (NT may perform below NN) */
      libxsmm_gemm_batch_omp(LIBXSMM_GEMM_PRECISION(ELEM_TYPE), LIBXSMM_GEMM_PRECISION(ELEM_TYPE),
        &transa, &transb, m, n, k, &alpha, amat_hst, &m/*lda*/, bmat_hst, &k/*ldb*/,
        &beta, cmat_hst, &m/*ldc*/, 1/*index_base*/, sizeof(int) * 3,
        stack_hst + 0, stack_hst + 1, stack_hst + 2, stack_size);
    }
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
    printf("host: %.1f ms %.1f GFLOPS/s\n", 1000.0 * duration / nrepeat,
      ((size_t)2 * m * n * k) * stack_size / (duration * (1ULL << 30) / nrepeat));
    /* transfer result from device back to host for validation */
    CHECK(acc_memcpy_d2h(cmat_dev, cmat_hst, sizeof(ELEM_TYPE) * mn * stack_size, stream), &result);
    CHECK(acc_stream_sync(stream), &result);
    /* TODO: validation code TBD */
  }
#endif
  CHECK(acc_host_mem_deallocate(stack_hst, stream), NULL);
  CHECK(acc_host_mem_deallocate(amat_hst, stream), NULL);
  CHECK(acc_host_mem_deallocate(bmat_hst, stream), NULL);
  CHECK(acc_host_mem_deallocate(cmat_hst, stream), NULL);
  CHECK(acc_dev_mem_deallocate(stack_dev), NULL);
  CHECK(acc_dev_mem_deallocate(amat_dev), NULL);
  CHECK(acc_dev_mem_deallocate(bmat_dev), NULL);
  CHECK(acc_dev_mem_deallocate(cmat_dev), NULL);
  CHECK(acc_stream_destroy(stream), NULL);
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


static void init_stack(int* stack, int stack_size,
  int mn, int mk, int kn, int nc, int na, int nb)
{
  /* navg matrix products are accumulated into a C-matrix */
  int navg = stack_size / nc;
  int nimb = MAX(1, navg - 4); /* imbalance */
  int i = 0, c = 0, ntop = 0;
  assert(0 < nc && nc <= stack_size);
  while (i < stack_size) {
    const int next = c + 1;
    ntop += navg + (rand() % (2 * nimb) - nimb);
    if (stack_size < ntop) ntop = stack_size;
    for (;i < ntop; ++i) { /* setup one-based indexes */
      const int a = rand() % na, b = rand() % nb;
      *stack++ = a * mk + 1; /* A-index */
      *stack++ = b * kn + 1; /* B-index */
      *stack++ = c * mn + 1; /* C-index */
    }
    if (next < nc) c = next;
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
