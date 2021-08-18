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
#define ACC_BENCH_SMM_EPSILON_double 1E-3
#define ACC_BENCH_SMM_EPSILON_float 1E-3

#define ROUNDUP2(N, NPOT) ((((unsigned long long)N) + ((NPOT) - 1)) & ~((NPOT) - 1))
#define CHECK(EXPR, RPTR) if ((NULL != ((const void*)(RPTR)) && EXIT_SUCCESS != *((const int*)(RPTR))) || \
  EXIT_SUCCESS != (NULL != ((const void*)(RPTR)) ? (*((int*)(RPTR)) = (EXPR)) : (EXPR))) assert(-1 == *((int*)(RPTR)))


static void parse_params(int argc, char* argv[], FILE** file,
  int* inr, int* iss, int* ism, int* isn, int* isk, int* inc, int* ina, int* inb)
{
  char buffer[1024], *args[8];
  int i;
  assert(file && inr && iss && ism && isn && isk && inc && ina && inb);
  --argc;
  if (NULL == *file) *file = (1 <= argc ? fopen(argv[1], "r") : NULL);
  if (NULL == *file) for (i = 0; i < argc; ++i) args[i] = argv[i+1];
  else {
    argc = 0;
    if (NULL != fgets(buffer, sizeof(buffer), *file)) {
      char* arg = strtok(buffer, " \t;,:");
      for (; NULL != arg; arg = strtok(NULL, " \t;,:")) {
        if (argc * sizeof(*args) < sizeof(args)) {
          args[argc++] = arg;
        }
        else { /* malformed command-line */
          fclose(*file);
          *file = NULL;
          break;
        }
      }
    }
    else {
      fclose(*file);
      *file = NULL;
    }
  }
  i = 0;
  if (i < argc) {
    const char* x1 = strchr(args[i], 'x');
    const char* x2 = (NULL == x1 ? NULL : strchr(x1 + 1, 'x'));
    if (NULL == x1 || NULL == x2) { /* accept "M N K"*/
      *inr = atoi(args[i++]);
      *iss = (i < argc ? atoi(args[i++]) : 0);
      if (i < argc) {
        x1 = strchr(args[i], 'x');
        x2 = (NULL == x1 ? NULL : strchr(x1 + 1, 'x'));
        *ism = atoi(args[i++]);
        if (NULL == x1 || NULL == x2) { /* accept "M N K"*/
          *isn = (i < argc ? atoi(args[i++]) : 0);
          *isk = (i < argc ? atoi(args[i++]) : 0);
        }
        else { /* accept "MxNxK" */
          *isn = atoi(x1 + 1);
          *isk = atoi(x2 + 1);
        }
      }
    }
    else { /* accept "MxNxK" */
      *ism = atoi(args[i++]);
      *isn = atoi(x1 + 1);
      *isk = atoi(x2 + 1);
    }
  }
  *inc = (i < argc ? atoi(args[i++]) : 0);
  *ina = (i < argc ? atoi(args[i++]) : 0);
  *inb = (i < argc ? atoi(args[i++]) : 0);
}


int main(int argc, char* argv[])
{
#if defined(USE_LIBXSMM) && defined(VALIDATE) && (0 != VALIDATE)
  double maxerror = 0;
#endif
#if defined(WARMUP) && (0 < WARMUP) && !defined(_DEBUG)
  const int warmup = MAX(WARMUP, 2) / 2 * 2;
#else
  const int warmup = 0;
#endif
  int result = c_dbcsr_acc_init();
  FILE* file = NULL;
  int nok = 0, inr = 0, iss = 0, ism = 0, isn = 0, isk = 0, inc = 0, ina = 0, inb = 0;
  parse_params(argc, argv, &file, &inr, &iss, &ism, &isn, &isk, &inc, &ina, &inb);
  CHECK(libsmm_acc_init(), &result); /* note: libsmm_acc_init() may imply acc_init() */
  if (EXIT_SUCCESS == result) {
    const char *const env_device = getenv("DEVICE");
    const int device = ((NULL == env_device || '\0' == *env_device) ? 0 : atoi(env_device));
    int ndevices = 0;
    result = c_dbcsr_acc_get_ndevices(&ndevices);
    if (0 < ndevices && (0 == device || EXIT_SUCCESS == c_dbcsr_acc_set_active_device(device))) {
#if defined(_DEBUG)
      fprintf(stderr, "Activated device %i of %i.\n", device, ndevices);
#endif
    }
    else {
      if (0 >= ndevices) fprintf(stderr, "No ACC-device found!\n");
      else fprintf(stderr, "Failed to activate device %i of %i!\n", device, ndevices);
      result = EXIT_FAILURE;
    }
  }
  else {
    fprintf(stderr, "ACC initialization failed!\n");
  }
  while (EXIT_SUCCESS == result) {
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
    int *stack_hst = NULL, *stack_dev = NULL, *trans_hst = NULL, *trans_dev = NULL;
    ELEM_TYPE *amat_hst = NULL, *bmat_hst = NULL, *cmat_hst = NULL;
    ELEM_TYPE *amat_dev = NULL, *bmat_dev = NULL, *cmat_dev = NULL;
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
    int r, i;
    assert(m <= (mn / n) && 0 == (mn % n));
    assert(m <= (mk / k) && 0 == (mk % k));
    assert(k <= (kn / n) && 0 == (kn % n));
    printf("%s%s%i %i %i %i %i %i %i %i\n", 0 < argc ? argv[0] : "", 0 < argc ? " " : "",
      nrepeat, stack_size, m, n, k, nc, na, nb);
    printf("typename (id=%i): %s\n", DBCSR_TYPE(ELEM_TYPE), DBCSR_STRINGIFY(ELEM_TYPE));
    if (MAX_KERNEL_DIM < m || MAX_KERNEL_DIM < n || MAX_KERNEL_DIM < k) {
      fprintf(stderr, "Matrix shape exceeds MAX_KERNEL_DIM!\n");
      result = EXIT_FAILURE;
    }
    CHECK(c_dbcsr_acc_stream_create(&stream, "stream", -1/*default priority*/), &result);
    CHECK(c_dbcsr_acc_host_mem_allocate((void**)&amat_hst, sizeof(ELEM_TYPE) * mk * na, stream), &result);
    CHECK(c_dbcsr_acc_host_mem_allocate((void**)&bmat_hst, sizeof(ELEM_TYPE) * kn * nb, stream), &result);
    CHECK(c_dbcsr_acc_host_mem_allocate((void**)&cmat_hst, sizeof(ELEM_TYPE) * mn * nc, stream), &result);
    CHECK(c_dbcsr_acc_host_mem_allocate((void**)&stack_hst, sizeof(int) * 3 * stack_size, stream), &result);
    CHECK(c_dbcsr_acc_host_mem_allocate((void**)&trans_hst, sizeof(int) * nb, stream), &result);
    CHECK(c_dbcsr_acc_stream_sync(stream), &result); /* ensure host-data is allocated */
    if (NULL != amat_hst && NULL != bmat_hst && NULL != trans_hst && NULL != stack_hst) {
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
#       pragma omp for
#endif
        for (i = 0; i < na; ++i) INIT_MAT(ELEM_TYPE, i/*seed*/ + 42, &amat_hst[i*mk], m, k, 1.0 / (nr * na));
#if defined(_OPENMP)
#       pragma omp for
#endif
        for (i = 0; i < nb; ++i) {
          INIT_MAT(ELEM_TYPE, i/*seed*/ + 24, &bmat_hst[i*kn], k, n, 1.0 / (nr * nb));
          trans_hst[i] = i * kn;
        }
      }
      init_stack(stack_hst, stack_size, mn, mk, kn, nc, na, nb);
    }
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
    if (NULL != amat_hst && NULL != bmat_hst && NULL != stack_hst) {
      duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      printf("copy-in: %.1f ms %.1f GB/s\n", 1000.0 * duration,
        (sizeof(ELEM_TYPE) * (mk + kn) + sizeof(int) * 3)
          * stack_size / (duration * (1ULL << 30)));
    }
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
    if (NULL != gold_hst && NULL != amat_hst && NULL != bmat_hst && NULL != stack_hst) {
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
          libxsmm_matdiff_info diff;
#   if (1 < VALIDATE)
          libxsmm_matdiff_info di;
          libxsmm_matdiff_clear(&diff);
          for (i = 0; i < nc; ++i) {
            const ELEM_TYPE *const gold = gold_hst + mn * i;
            const ELEM_TYPE *const test = cmat_hst + mn * i;
            result = libxsmm_matdiff(&di, LIBXSMM_DATATYPE(ELEM_TYPE), m, n, gold, test, &m, &m);
            if (EXIT_SUCCESS == result) libxsmm_matdiff_reduce(&diff, &di);
            else break;
          }
#   else
          /* validate result buffers at once (including excess area/padded space) */
          result = libxsmm_matdiff(&diff, LIBXSMM_DATATYPE(ELEM_TYPE),
            mn, nc, gold_hst, cmat_hst, &mn, &mn);
#   endif
          if (EXIT_SUCCESS == result) {
            const double relerror = 1.0 - diff.rsq;
            printf("rel.error: %g", relerror);
            if (maxerror < relerror && NULL != file) maxerror = relerror;
            if (0 < relerror) {
              if (LIBXSMM_NOTNAN(diff.v_tst)) printf(" (%g != %g)\n", diff.v_ref, diff.v_tst);
              else printf(" (%g)\n", diff.v_tst);
            }
            else printf("\n");
            if (0 < check && check < relerror) result = EXIT_FAILURE;
          }
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
    if (EXIT_SUCCESS == result) {
      ++nok; parse_params(argc, argv, &file, &inr, &iss, &ism, &isn, &isk, &inc, &ina, &inb);
      if (NULL != file) printf("\n"); else break;
    }
  }
#if !defined(__CUDA)
  CHECK(libsmm_acc_finalize(), NULL);
#endif
  CHECK(c_dbcsr_acc_finalize(), NULL);
  if (EXIT_SUCCESS == result) {
#if defined(USE_LIBXSMM) && defined(VALIDATE) && (0 != VALIDATE)
    if (1 < nok) printf("\nmax.error: %g\n", maxerror);
#endif
  }
  else {
    if (NULL != file) fclose(file);
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
