/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#include "acc_libsmm.h"
#include "acc_bench.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

#if defined(__LIBXSMM)
# include <libxsmm.h>
# define USE_LIBXSMM
# if defined(_OPENMP)
#   define ACC_BENCH_USEOMP(FUNC) LIBXSMM_USEOMP(FUNC)
# else
#   define ACC_BENCH_USEOMP(FUNC) (FUNC)
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
#if !defined(TRANSPOSE)
# define TRANSPOSE 1
#endif
#if !defined(VALIDATE)
# define VALIDATE 1
#endif
#if !defined(WARMUP)
# define WARMUP 2
#endif

#define ACC_BENCH_SMM_EPSILON(T) DBCSR_CONCATENATE(ACC_BENCH_SMM_EPSILON_, T)
#define ACC_BENCH_SMM_EPSILON_double 5E-6
#define ACC_BENCH_SMM_EPSILON_float 5E-6

#define ROUNDUP2(N, NPOT) ((((unsigned long long)N) + ((NPOT) - 1)) & ~((NPOT) - 1))
#define CHECK(EXPR, RPTR) if ((NULL != ((const void*)(RPTR)) && EXIT_SUCCESS != *((const int*)(RPTR))) || \
  EXIT_SUCCESS != (NULL != ((const void*)(RPTR)) ? (*((int*)(RPTR)) = (EXPR)) : (EXPR))) assert(0)


#if defined(_DEBUG) && defined(USE_LIBXSMM) && defined(VALIDATE) && (0 != VALIDATE)
static void print(FILE* ostream, const char* label, const ELEM_TYPE* mat, int m, int n);
#endif


static int parse_params(int argc, char* argv[], FILE** file,
  int* inr, int* iss, int* ism, int* isn, int* isk, int* inc, int* ina, int* inb)
{
  int result = EXIT_SUCCESS;
  assert(file && inr && iss && ism && isn && isk && inc && ina && inb);
  if (NULL == *file) *file = (1 < argc ? fopen(argv[1], "r") : NULL);
  if (NULL == *file) {
    *inr = (1 < argc ? atoi(argv[1]) : 0);
    *iss = (2 < argc ? atoi(argv[2]) : 0);
    *ism = (3 < argc ? atoi(argv[3]) : 0);
    *isn = (4 < argc ? atoi(argv[4]) : 0);
    *isk = (5 < argc ? atoi(argv[5]) : 0);
    *inc = (6 < argc ? atoi(argv[6]) : 0);
    *ina = (7 < argc ? atoi(argv[7]) : 0);
    *inb = (8 < argc ? atoi(argv[8]) : 0);
  }
  else {
    char buffer[1024];
    *inr = *iss = *ism = *isn = *isk = *inc = *ina = *inb = 0;
    result = ((NULL != fgets(buffer, sizeof(buffer), *file) &&
        0 <= sscanf(buffer, "%i %i %i %i %i %i %i %i",
          inr, iss, ism, isn, isk, inc, ina, inb))
      ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  return result;
}


int main(int argc, char* argv[])
{
  FILE* file = NULL;
  int result = EXIT_SUCCESS;
  do {
    int inr = 0, iss = 0, ism = 0, isn = 0, isk = 0, inc = 0, ina = 0, inb = 0;
    int result = parse_params(argc, argv, &file, &inr, &iss, &ism, &isn, &isk, &inc, &ina, &inb);
    const int nrepeat = (0 < inr ? inr : 3);
    const int stack_size = (0 < iss ? iss : 30000);
    const int m = (0 < ism ? ism : 23);
    const int n = (0 < isn ? isn : m);
    const int k = (0 < isk ? isk : m);
    const int nc = (0 < inc ? MIN(inc, stack_size) : MAX(stack_size / 16, 1));
    const int na = (0 < ina ? ina : (10 * nc));
    const int nb = (0 < inb ? inb : (10 * nc));
    const int nr = nrepeat * nc;
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
    const int warmup = MAX(WARMUP, 2) / 2 * 2;
#else
    const int warmup = 0;
#endif
    const char *const env_device = getenv("DEVICE");
    const int device = ((NULL == env_device || '\0' == *env_device) ? 0 : atoi(env_device));
    int *stack_hst = NULL, *stack_dev = NULL, *trans_hst = NULL, *trans_dev = NULL;
    ELEM_TYPE *amat_hst = NULL, *bmat_hst = NULL, *cmat_hst = NULL;
    ELEM_TYPE *amat_dev = NULL, *bmat_dev = NULL, *cmat_dev = NULL;
    int ndevices = 0, r, i;
    void *stream = NULL;
#if defined(USE_LIBXSMM)
# if defined(VALIDATE) && (0 != VALIDATE)
    const char *const env_check = getenv("CHECK");
    const double check = (NULL == env_check ? -1 : fabs(atof(env_check) * ACC_BENCH_SMM_EPSILON(ELEM_TYPE)));
    ELEM_TYPE *const gold_hst = (ELEM_TYPE*)(0 != check ? libxsmm_malloc(sizeof(ELEM_TYPE) * mn * nc) : NULL);
# endif
    libxsmm_timer_tickint start;
# if defined(TRANSPOSE) && (0 != TRANSPOSE) && defined(VALIDATE) && (0 != VALIDATE)
    double transpose;
# endif
    double duration;
#endif
    assert(m <= (mn / n) && 0 == (mn % n) && k <= (mk / k) && 0 == (mk % k) && n <= (kn / n) && 0 == (kn % n));
    if (EXIT_SUCCESS == result) {
      printf("%s%s%i %i %i %i %i %i %i %i\n", 0 < argc ? argv[0] : "", 0 < argc ? " " : "",
        nrepeat, stack_size, m, n, k, nc, na, nb);
    }
    else break; /* end of input/argument-file reached */
    CHECK(c_dbcsr_acc_init(), &result);
    /* note: libsmm_acc_init() may imply acc_init() */
    CHECK(libsmm_acc_init(), &result);
    CHECK(c_dbcsr_acc_get_ndevices(&ndevices), &result);
    if (0 < ndevices && (0 == device || EXIT_SUCCESS == c_dbcsr_acc_set_active_device(device))) {
#if defined(_DEBUG)
      fprintf(stderr, "Activated device %i of %i.\n", device, ndevices);
#endif
    }
    else {
      if (0 >= ndevices) fprintf(stderr, "No ACC-device found!\n");
      else fprintf(stderr, "Failed to activate device %i of %i!\n", device, ndevices);
#if !defined(__CUDA)
      CHECK(libsmm_acc_finalize(), NULL);
#endif
      CHECK(c_dbcsr_acc_finalize(), NULL);
      return result;
    }
    printf("typename (id=%i): %s\n", DBCSR_TYPE(ELEM_TYPE), DBCSR_STRINGIFY(ELEM_TYPE));
    CHECK(c_dbcsr_acc_stream_create(&stream, "stream", -1/*default priority*/), &result);
    CHECK(c_dbcsr_acc_host_mem_allocate((void**)&amat_hst, sizeof(ELEM_TYPE) * mk * na, stream), &result);
    CHECK(c_dbcsr_acc_host_mem_allocate((void**)&bmat_hst, sizeof(ELEM_TYPE) * kn * nb, stream), &result);
    CHECK(c_dbcsr_acc_host_mem_allocate((void**)&cmat_hst, sizeof(ELEM_TYPE) * mn * nc, stream), &result);
    CHECK(c_dbcsr_acc_host_mem_allocate((void**)&stack_hst, sizeof(int) * 3 * stack_size, stream), &result);
    CHECK(c_dbcsr_acc_host_mem_allocate((void**)&trans_hst, sizeof(int) * nb, stream), &result);
    CHECK(c_dbcsr_acc_stream_sync(stream), &result); /* ensure host-data is allocated */
    /* initialize matrices */
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
#     pragma omp for
#endif
      for (i = 0; i < na; ++i) INIT_MAT(ELEM_TYPE, i/*seed*/ + 42, &amat_hst[i*mk], m, k, 1.0 / (nr * na));
#if defined(_OPENMP)
#     pragma omp for
#endif
      for (i = 0; i < nb; ++i) {
        INIT_MAT(ELEM_TYPE, i/*seed*/ + 24, &bmat_hst[i*kn], k, n, 1.0 / (nr * nb));
        trans_hst[i] = i * kn;
      }
    }
    init_stack(stack_hst, stack_size, mn, mk, kn, nc, na, nb);
    CHECK(c_dbcsr_acc_dev_mem_allocate((void**)&amat_dev, sizeof(ELEM_TYPE) * mk * na), &result);
    CHECK(c_dbcsr_acc_dev_mem_allocate((void**)&bmat_dev, sizeof(ELEM_TYPE) * kn * nb), &result);
    CHECK(c_dbcsr_acc_dev_mem_allocate((void**)&cmat_dev, sizeof(ELEM_TYPE) * mn * nc), &result);
    CHECK(c_dbcsr_acc_dev_mem_allocate((void**)&stack_dev, sizeof(int) * 3 * stack_size), &result);
    CHECK(c_dbcsr_acc_dev_mem_allocate((void**)&trans_dev, sizeof(int) * nb), &result);
    CHECK(c_dbcsr_acc_memset_zero(cmat_dev, 0/*offset*/, sizeof(ELEM_TYPE) * mn * nc, stream), &result);
    CHECK(c_dbcsr_acc_memcpy_h2d(trans_hst, trans_dev, sizeof(int) * nb, stream), &result);
#if defined(USE_LIBXSMM)
    CHECK(c_dbcsr_acc_stream_sync(stream), &result);
    start = libxsmm_timer_tick();
#endif
    CHECK(c_dbcsr_acc_memcpy_h2d(amat_hst, amat_dev, sizeof(ELEM_TYPE) * mk * na, stream), &result);
    CHECK(c_dbcsr_acc_memcpy_h2d(bmat_hst, bmat_dev, sizeof(ELEM_TYPE) * kn * nb, stream), &result);
    CHECK(c_dbcsr_acc_memcpy_h2d(stack_hst, stack_dev, sizeof(int) * 3 * stack_size, stream), &result);
#if defined(USE_LIBXSMM)
    CHECK(c_dbcsr_acc_stream_sync(stream), &result);
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
    printf("copy-in: %.1f ms %.1f GB/s\n", 1000.0 * duration,
      (sizeof(ELEM_TYPE) * (mk + kn) + sizeof(int) * 3)
        * stack_size / (duration * (1ULL << 30)));
#endif
#if defined(TRANSPOSE) && (0 != TRANSPOSE) && defined(VALIDATE) && (0 != VALIDATE)
    /* warmup execution and prebuild transpose-kernel */
    for (r = 0; r < warmup / 2; ++r) {
      CHECK(libsmm_acc_transpose(trans_dev, 0/*offset*/, nb, bmat_dev,
        DBCSR_TYPE(ELEM_TYPE), k, n, MAX_KERNEL_DIM, stream), &result);
      CHECK(libsmm_acc_transpose(trans_dev, 0/*offset*/, nb, bmat_dev,
        DBCSR_TYPE(ELEM_TYPE), n, k, MAX_KERNEL_DIM, stream), &result);
    }
# if defined(USE_LIBXSMM)
    CHECK(c_dbcsr_acc_stream_sync(stream), &result);
    start = libxsmm_timer_tick();
# endif
    /* to perform NN-SMMs on the device, all B-matrices are transposed upfront (SMM-kernel is limited to NT) */
    CHECK(libsmm_acc_transpose(trans_dev, 0/*offset*/, nb, bmat_dev,
      DBCSR_TYPE(ELEM_TYPE), k, n, MAX_KERNEL_DIM, stream), &result);
# if defined(USE_LIBXSMM)
    CHECK(c_dbcsr_acc_stream_sync(stream), &result);
    transpose = libxsmm_timer_duration(start, libxsmm_timer_tick());
# endif
#endif
    /* warmup execution and prebuild SMM-kernel */
    for (r = 0; r < warmup; ++r) {
      CHECK(libsmm_acc_process(stack_hst, stack_dev, stack_size, 3/*nparams*/, DBCSR_TYPE(ELEM_TYPE),
        amat_dev, bmat_dev, cmat_dev, m, n, k, MAX_KERNEL_DIM, 1/*homogeneous*/, stream, stream), &result);
    }
    CHECK(c_dbcsr_acc_memset_zero(cmat_dev, 0/*offset*/, sizeof(ELEM_TYPE) * mn * nc, stream), &result);
#if defined(USE_LIBXSMM)
    CHECK(c_dbcsr_acc_stream_sync(stream), &result);
    start = libxsmm_timer_tick();
#endif
    for (r = 0; r < nrepeat; ++r) {
      /* GPU-kernel is limited to C += Ai * Bi^T, i.e., NT (for NN, all Bi must be transposed upfront) */
      CHECK(libsmm_acc_process(stack_hst, stack_dev, stack_size, 3/*nparams*/, DBCSR_TYPE(ELEM_TYPE),
        amat_dev, bmat_dev, cmat_dev, m, n, k, MAX_KERNEL_DIM, 1/*homogeneous*/, stream, stream), &result);
    }
#if defined(USE_LIBXSMM)
    CHECK(c_dbcsr_acc_stream_sync(stream), &result);
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
    if (EXIT_SUCCESS == result) {
# if defined(TRANSPOSE) && (0 != TRANSPOSE)
      printf("transpose: %.1f ms %.1f GFLOPS/s\n", 1000.0 * (duration + transpose) / nrepeat,
        1E-9 * ((size_t)2 * m * n * k * stack_size * nrepeat) / (duration + transpose));
# endif
      printf("device: %.1f ms %.1f GFLOPS/s\n", 1000.0 * duration / nrepeat,
        1E-9 * ((size_t)2 * m * n * k * stack_size * nrepeat) / duration);
    }
# if defined(VALIDATE) && (0 != VALIDATE)
    /* determine host's performance independent of current result code/status */
    if (NULL != gold_hst) {
      const ELEM_TYPE alpha = 1, beta = 1;
      const char transa = 'N';
#   if defined(TRANSPOSE) && (0 != TRANSPOSE)
      const char transb = 'N';
#   else
      const char transb = 'T';
#   endif
      memset(gold_hst, 0, sizeof(ELEM_TYPE) * mn * nc);
      for (r = 0; r < warmup; ++r) {
        ACC_BENCH_USEOMP(libxsmm_gemm_batch)(
          LIBXSMM_GEMM_PRECISION(ELEM_TYPE), LIBXSMM_GEMM_PRECISION(ELEM_TYPE),
          &transa, &transb, m, n, k, &alpha, amat_hst, &m/*lda*/, bmat_hst, &k/*ldb*/,
          &beta, gold_hst, &m/*ldc*/, 1/*index_base*/, sizeof(int) * 3,
          stack_hst + 0, stack_hst + 1, stack_hst + 2, stack_size);
      }
      memset(gold_hst, 0, sizeof(ELEM_TYPE) * mn * nc);
      start = libxsmm_timer_tick();
      /* CPU-kernel operates on data that is not initialized in NUMA-aware fashion */
      for (r = 0; r < nrepeat; ++r) {
        ACC_BENCH_USEOMP(libxsmm_gemm_batch)(
          LIBXSMM_GEMM_PRECISION(ELEM_TYPE), LIBXSMM_GEMM_PRECISION(ELEM_TYPE),
          &transa, &transb, m, n, k, &alpha, amat_hst, &m/*lda*/, bmat_hst, &k/*ldb*/,
          &beta, gold_hst, &m/*ldc*/, 1/*index_base*/, sizeof(int) * 3,
          stack_hst + 0, stack_hst + 1, stack_hst + 2, stack_size);
      }
      duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      printf("host: %.1f ms %.1f GFLOPS/s\n", 1000.0 * duration / nrepeat,
        1E-9 * ((size_t)2 * m * n * k * stack_size * nrepeat) / duration);
      /* validate correctness in case of successful result code/status */
      if (EXIT_SUCCESS == result) {
        /* transfer result from device to host for validation */
        CHECK(c_dbcsr_acc_memcpy_d2h(cmat_dev, cmat_hst, sizeof(ELEM_TYPE) * mn * nc, stream), &result);
        CHECK(c_dbcsr_acc_stream_sync(stream), &result);
        if (EXIT_SUCCESS == result) {
          double abserror = 0, relerror = 0, a = 0, b = 0;
          for (i = 0; i < nc; ++i) {
            const ELEM_TYPE *const gold = gold_hst + mn * i;
            const ELEM_TYPE *const test = cmat_hst + mn * i;
            double diff = 0, ai = 0, bi = 0;
            for (r = 0; r < (m * n); ++r) {
              const double ar = (double)gold[r];
              const double br = (double)test[r];
              const double d = fabs(ar - br);
              if (diff < d) {
                diff = d; ai = ar; bi = br;
              }
            }
            if (0 < diff) {
              const double rd = fabs(0 != ai ? (diff / ai) : (diff / bi));
#   if defined(_DEBUG)
              print(stderr, "gold = ", gold, m, n);
              print(stderr, "test = ", test, m, n);
              fprintf(stderr, "rel = %g (%g != %g)\n", rd, ai, bi);
#   endif
              if (abserror < diff) {
                abserror = diff;
                relerror = rd;
                a = ai; b = bi;
              }
            }
          }
          printf("max.error: %g", relerror / nr);
          if (0 < abserror) printf(" (%g != %g)\n", a, b); else printf("\n");
          if (0 < check && (nr * check) < relerror) result = EXIT_FAILURE;
        }
      }
      libxsmm_free(gold_hst);
    }
# endif
#endif
    CHECK(c_dbcsr_acc_host_mem_deallocate(stack_hst, stream), NULL);
    CHECK(c_dbcsr_acc_host_mem_deallocate(trans_hst, stream), NULL);
    CHECK(c_dbcsr_acc_host_mem_deallocate(amat_hst, stream), NULL);
    CHECK(c_dbcsr_acc_host_mem_deallocate(bmat_hst, stream), NULL);
    CHECK(c_dbcsr_acc_host_mem_deallocate(cmat_hst, stream), NULL);
    CHECK(c_dbcsr_acc_dev_mem_deallocate(stack_dev), NULL);
    CHECK(c_dbcsr_acc_dev_mem_deallocate(trans_dev), NULL);
    CHECK(c_dbcsr_acc_dev_mem_deallocate(amat_dev), NULL);
    CHECK(c_dbcsr_acc_dev_mem_deallocate(bmat_dev), NULL);
    CHECK(c_dbcsr_acc_dev_mem_deallocate(cmat_dev), NULL);
    CHECK(c_dbcsr_acc_stream_destroy(stream), NULL);
#if !defined(__CUDA)
    CHECK(libsmm_acc_finalize(), NULL);
#endif
    CHECK(c_dbcsr_acc_finalize(), NULL);
    if (EXIT_SUCCESS != result) {
      fprintf(stderr, "FAILED\n");
    }
  } while (NULL != file);
  if (NULL != file) fclose(file);
  return result;
}


#if defined(_DEBUG) && defined(USE_LIBXSMM) && defined(VALIDATE) && (0 != VALIDATE)
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
