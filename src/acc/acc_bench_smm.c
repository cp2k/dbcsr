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
#include <math.h>

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
#    define ACC_BENCH_USEOMP(FUNC) LIBXSMM_USEOMP(FUNC)
#  else
#    define ACC_BENCH_USEOMP(FUNC) (FUNC)
#  endif
#  define ACC_BENCH_GEMM_BATCH(IPREC, OPREC, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, STRIDE_A, B, LDB, STRIDE_B, BETA, C, LDC, \
    STRIDE_C, INDEX_STRIDE, INDEX_BASE, BATCHSIZE) \
    ACC_BENCH_USEOMP(libxsmm_gemm_batch) \
    (IPREC, OPREC, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, STRIDE_A, B, LDB, STRIDE_B, BETA, C, LDC, STRIDE_C, INDEX_STRIDE, \
      INDEX_BASE, BATCHSIZE)
#  define PRINTF(...) \
    do { \
      const size_t print_buffer_size = sizeof(print_buffer) - print_offset; \
      const int print_buffer_result = LIBXSMM_SNPRINTF(print_buffer + print_offset, print_buffer_size, __VA_ARGS__); \
      assert(0 <= print_buffer_result && print_buffer_result < (int)print_buffer_size); \
      print_offset += print_buffer_result; \
    } while (0)
#else
#  define PRINTF(...) printf(__VA_ARGS__)
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
#if !defined(BATCHGRAIN)
#  define BATCHGRAIN 100
#endif
#if !defined(BATCHSIZE)
#  define BATCHSIZE (300 * BATCHGRAIN)
#endif
#if !defined(NRAND)
#  define NRAND BATCHSIZE
#endif
#if !defined(NREPEAT)
#  define NREPEAT 3
#endif
#if !defined(XREPEAT)
#  define XREPEAT 66
#endif
#if !defined(TRANSPOSE)
#  define TRANSPOSE
#endif
#if !defined(VALIDATE)
#  define VALIDATE
#endif
#if !defined(WARMUP)
#  define WARMUP 2
#endif
#if !defined(DELIMS)
#  define DELIMS ",;:|/\n\t "
#endif

#define ACC_BENCH_SMM_EPSILON(T) DBCSR_CONCATENATE(ACC_BENCH_SMM_EPSILON_, T)
#define ACC_BENCH_SMM_EPSILON_double 1E-3
#define ACC_BENCH_SMM_EPSILON_float 2E-3

#define ROUNDUP2(N, NPOT) ((((unsigned long long)N) + ((NPOT) - 1)) & ~((NPOT) - 1))
#define CHECK(EXPR, RPTR, VALUE) \
  do { \
    if (NULL != ((const void*)(RPTR))) { \
      if (0 == *((const int*)(RPTR)) || (0 > *((const int*)(RPTR)) && 0 == (VALUE))) { \
        const int check_r_ = (EXPR); \
        if (0 != check_r_) { \
          *((int*)(RPTR)) = check_r_; \
          assert(0 > check_r_ && 0 == (VALUE)); \
        } \
      } \
    } \
    else { \
      (EXPR); \
    } \
  } while (0)


static void parse_params(int argc, char* argv[], FILE** file, const char** snr, const char** sss, const char** ssm,
  const char** ssn, const char** ssk, const char** snc, const char** sna, const char** snb) {
  char buffer[1024], *args[8];
  int i;
  assert(file && snr && sss && ssm && ssn && ssk && snc && sna && snb);
  --argc;
  if (NULL == *file) *file = (1 <= argc ? fopen(argv[1], "r") : NULL);
  if (NULL == *file) {
    for (i = 0; i < argc; ++i) args[i] = argv[i + 1];
  }
  else { /* input file is specified */
    argc = 0;
    if (NULL != fgets(buffer, sizeof(buffer), *file)) {
      char* arg = strtok(buffer, DELIMS);
      while (NULL == arg && NULL != fgets(buffer, sizeof(buffer), *file)) {
        arg = strtok(buffer, DELIMS);
      }
      for (; NULL != arg; arg = strtok(NULL, DELIMS)) {
        if (argc * sizeof(*args) < sizeof(args)) args[argc++] = arg;
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
    if (NULL == x1 || NULL == x2) { /* accept "M N K" */
      *snr = args[i++];
      *sss = (i < argc ? args[i++] : NULL);
      if (i < argc) {
        x1 = strchr(args[i], 'x');
        x2 = (NULL == x1 ? NULL : strchr(x1 + 1, 'x'));
        *ssm = args[i++];
        if (NULL == x1 || NULL == x2) { /* accept "M N K" */
          *ssn = (i < argc ? args[i++] : NULL);
          *ssk = (i < argc ? args[i++] : NULL);
        }
        else { /* accept "MxNxK" */
          *ssn = x1 + 1;
          *ssk = x2 + 1;
        }
      }
    }
    else { /* accept "MxNxK" */
      *ssm = args[i++];
      *ssn = x1 + 1;
      *ssk = x2 + 1;
    }
  }
  *snc = (i < argc ? args[i++] : NULL);
  *sna = (i < argc ? args[i++] : NULL);
  *snb = (i < argc ? args[i++] : NULL);
}


static size_t parse_nbytes(const char* nbytes, size_t* nelems) {
  size_t result = 0;
  if (NULL != nelems) *nelems = 0;
  if (NULL != nbytes && '\0' != *nbytes) {
    size_t u = strlen(nbytes) - 1;
    const char units[] = "kmgKMG", *const unit = strchr(units, nbytes[u]);
    char* end = NULL;
    /* take parsed value with increased type-width */
    const long long int ibytes = strtol(nbytes, &end, 10);
    if (NULL != end) { /* no obvious error */
      /* value is given without unit */
      if (NULL == unit && '\0' == *end) {
        result = (size_t)ibytes;
        if (NULL != nelems) *nelems = result;
      }
      /* must match allowed set of units */
      else if (NULL != unit && *unit == *end) {
        result = (size_t)ibytes;
        if (NULL != nelems) *nelems = result;
        u = (unit - units) % 3 + 1;
        result <<= u * 10;
      }
    }
  }
  return result;
}


int main(int argc, char* argv[]) {
  const char* const env_check = getenv("CHECK");
  const double check = (NULL == env_check ? -1 : fabs(atof(env_check) * ACC_BENCH_SMM_EPSILON(ELEM_TYPE)));
#if defined(WARMUP) && (0 < WARMUP) && !defined(_DEBUG)
  const int warmup = MAX(WARMUP, 2) / 2 * 2;
#else
  const int warmup = 0;
#endif
  int result = c_dbcsr_acc_init(), *rnd = NULL, nok = 0, m = 0, n = 0, k = 0, i;
  const char *snr = NULL, *sss = NULL;
  const char *ssm = NULL, *ssn = NULL, *ssk = NULL;
  const char *snc = NULL, *sna = NULL, *snb = NULL;
  FILE* file = NULL;
#if defined(USE_LIBXSMM) && defined(VALIDATE)
  double maxdiff = 0;
#else
  DBCSR_MARK_USED(check);
#endif
  CHECK(libsmm_acc_init(), &result, check); /* note: libsmm_acc_init() may imply acc_init() */
  if (EXIT_SUCCESS == result) {
    const char* const env_device = getenv("DEVICE");
    const int device = ((NULL == env_device || '\0' == *env_device) ? 0 : atoi(env_device));
    int ndevices = 0;
    result = c_dbcsr_acc_get_ndevices(&ndevices);
    if (0 < ndevices && (0 == device || EXIT_SUCCESS == c_dbcsr_acc_set_active_device(device))) {
      printf("Activated device%i (ndevices=%i)\n", device, ndevices);
    }
    else {
      if (0 >= ndevices) {
        fprintf(stderr, "ERROR: No ACC-device found!\n");
      }
      else {
        fprintf(stderr, "ERROR: Failed to activate device %i of %i!\n", device, ndevices);
      }
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      rnd = (int*)malloc(sizeof(int) * NRAND);
      if (NULL != rnd) {
        srand(25071975); /* seed rng */
        for (i = 0; i < NRAND; ++i) rnd[i] = rand();
        parse_params(argc, argv, &file, &snr, &sss, &ssm, &ssn, &ssk, &snc, &sna, &snb);
      }
      else result = EXIT_FAILURE;
    }
  }
  else {
    fprintf(stderr, "ERROR: ACC initialization failed!\n");
  }
  while (EXIT_SUCCESS == result || (NULL != file && 0 > result && 0 == check)) {
    const int inr = (NULL != snr ? atoi(snr) : 0);
    const int ism = (NULL != ssm ? atoi(ssm) : 0);
    const int isn = (NULL != ssn ? atoi(ssn) : 0);
    const int isk = (NULL != ssk ? atoi(ssk) : 0);
    const int inc = (NULL != snc ? atoi(snc) : 0);
    const int ina = (NULL != sna ? atoi(sna) : 0);
    const int inb = (NULL != snb ? atoi(snb) : 0);
    if (NULL != file && 0 > result && 0 == check) result = EXIT_SUCCESS;
    m = (0 < ism ? ism : 23);
    n = (0 < isn ? isn : m);
    k = (0 < isk ? isk : m);
    {
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
      void* stream = NULL;
#if defined(USE_LIBXSMM)
      libxsmm_timer_tickint start;
      int print_offset = 0;
      char print_buffer[1024];
#  if defined(__OPENCL)
      const char* const env_smm_repeat = getenv("SMM_NREPEAT");
      const int smm_nrepeat = (NULL == env_smm_repeat ? 1 : MAX(atoi(env_smm_repeat), 1));
#  else
      const int smm_nrepeat = 1;
#  endif
#  if defined(TRANSPOSE) && defined(VALIDATE)
      double transpose = 0;
#  endif
      double duration = 0;
#endif
      const char* const env_stack_size = getenv("SMM_BATCHSIZE");
      const int xrepeat = (0 != check ? NREPEAT : XREPEAT);
      int nrepeat = (0 < inr ? inr : xrepeat);
      int stack_size, na, nb, nc, nr, r;
      if (NULL == env_stack_size) {
        stack_size = 0;
        if (NULL != sss) {
          size_t nelems, s;
          const size_t nbytes = parse_nbytes(sss, &nelems);
          if (nbytes != nelems) {
            while (1) {
              nc = (0 < inc ? MIN(inc, stack_size) : MAX(stack_size / 16, 1));
              na = (0 < ina ? ina : (10 * nc));
              nb = (0 < inb ? inb : (10 * nc));
              s = sizeof(ELEM_TYPE) * (mk * na + kn * nb) + sizeof(int) * 3 * stack_size;
              if (s < nbytes) ++stack_size;
              else break;
            }
          }
          else stack_size = (int)nelems;
        }
      }
      else { /* parse SMM_BATCHSIZE=batchsize,nrepfactor */
        i = strcspn(env_stack_size, DELIMS);
        if (i < (int)sizeof(DELIMS)) {
          r = atoi(env_stack_size + i + 1);
          if (0 < r) nrepeat = r;
        }
        stack_size = atoi(env_stack_size);
      }
      if (0 >= stack_size) { /* trigger default */
        if (0 > stack_size) { /* randomize batchsize */
          const int r = rnd[nok % NRAND], ss = -stack_size, bs = (1 < ss ? ss : BATCHSIZE);
          const int limit = (BATCHGRAIN < ss ? ((bs + BATCHGRAIN - 1) / BATCHGRAIN) : ss);
          stack_size = (r % limit + 1) * BATCHGRAIN;
          nrepeat = MAX((BATCHSIZE * nrepeat + stack_size - 1) / stack_size, xrepeat);
        }
        else stack_size = BATCHSIZE; /* plain default */
      }
      nc = (0 < inc ? MIN(inc, stack_size) : MAX(stack_size / 16, 1));
      na = (0 < ina ? ina : (10 * nc));
      nb = (0 < inb ? inb : (10 * nc));
      nr = nrepeat * nc;
      assert(m <= (mn / n) && 0 == (mn % n));
      assert(m <= (mk / k) && 0 == (mk % k));
      assert(k <= (kn / n) && 0 == (kn % n));
      PRINTF(
        "%s%s%i %i %i %i %i %i %i %i\n", 0 < argc ? argv[0] : "", 0 < argc ? " " : "", nrepeat, stack_size, m, n, k, nc, na, nb);
      PRINTF("typename (id=%i): %s\n", DBCSR_TYPE(ELEM_TYPE), DBCSR_STRINGIFY(ELEM_TYPE));
      if (MAX_KERNEL_DIM < m || MAX_KERNEL_DIM < n || MAX_KERNEL_DIM < k) {
        fprintf(stderr, "ERROR: Matrix shape exceeds MAX_KERNEL_DIM!\n");
        result = EXIT_FAILURE;
      }
      CHECK(c_dbcsr_acc_stream_create(&stream, "stream", -1 /*default priority*/), &result, check);
      CHECK(c_dbcsr_acc_host_mem_allocate((void**)(void*)&amat_hst, sizeof(ELEM_TYPE) * mk * na, stream), &result, check);
      CHECK(c_dbcsr_acc_host_mem_allocate((void**)(void*)&bmat_hst, sizeof(ELEM_TYPE) * kn * nb, stream), &result, check);
      CHECK(c_dbcsr_acc_host_mem_allocate((void**)(void*)&cmat_hst, sizeof(ELEM_TYPE) * mn * nc, stream), &result, check);
      CHECK(c_dbcsr_acc_host_mem_allocate((void**)(void*)&stack_hst, sizeof(int) * 3 * stack_size, stream), &result, check);
      CHECK(c_dbcsr_acc_host_mem_allocate((void**)(void*)&trans_hst, sizeof(int) * nb, stream), &result, check);
      CHECK(c_dbcsr_acc_stream_sync(stream), &result, check); /* ensure host-data is allocated */
      if (NULL != amat_hst && NULL != bmat_hst && NULL != trans_hst && NULL != stack_hst) {
        init_stack(stack_hst, stack_size, NRAND, rnd, mn, mk, kn, nc, na, nb);
#if defined(_OPENMP)
#  pragma omp parallel
#endif
        {
#if defined(_OPENMP)
#  pragma omp for
#endif
          for (i = 0; i < na; ++i) INIT_MAT(ELEM_TYPE, i /*seed*/ + 42, &amat_hst[i * mk], m, k, 1.0 / (nr * na));
#if defined(_OPENMP)
#  pragma omp for
#endif
          for (i = 0; i < nb; ++i) {
            INIT_MAT(ELEM_TYPE, i /*seed*/ + 24, &bmat_hst[i * kn], k, n, 1.0 / (nr * nb));
            trans_hst[i] = i * kn;
          }
        }
      }
      CHECK(c_dbcsr_acc_dev_mem_allocate((void**)(void*)&amat_dev, sizeof(ELEM_TYPE) * mk * na), &result, check);
      CHECK(c_dbcsr_acc_dev_mem_allocate((void**)(void*)&bmat_dev, sizeof(ELEM_TYPE) * kn * nb), &result, check);
      CHECK(c_dbcsr_acc_dev_mem_allocate((void**)(void*)&cmat_dev, sizeof(ELEM_TYPE) * mn * nc), &result, check);
      CHECK(c_dbcsr_acc_dev_mem_allocate((void**)(void*)&stack_dev, sizeof(int) * 3 * stack_size), &result, check);
      CHECK(c_dbcsr_acc_dev_mem_allocate((void**)(void*)&trans_dev, sizeof(int) * nb), &result, check);
      CHECK(c_dbcsr_acc_memset_zero(cmat_dev, 0 /*offset*/, sizeof(ELEM_TYPE) * mn * nc, stream), &result, check);
      CHECK(c_dbcsr_acc_memcpy_h2d(trans_hst, trans_dev, sizeof(int) * nb, stream), &result, check);
#if defined(USE_LIBXSMM)
      CHECK(c_dbcsr_acc_stream_sync(stream), &result, check);
      start = libxsmm_timer_tick();
#endif
      CHECK(c_dbcsr_acc_memcpy_h2d(amat_hst, amat_dev, sizeof(ELEM_TYPE) * mk * na, stream), &result, check);
      CHECK(c_dbcsr_acc_memcpy_h2d(bmat_hst, bmat_dev, sizeof(ELEM_TYPE) * kn * nb, stream), &result, check);
      CHECK(c_dbcsr_acc_memcpy_h2d(stack_hst, stack_dev, sizeof(int) * 3 * stack_size, stream), &result, check);
#if defined(USE_LIBXSMM)
      CHECK(c_dbcsr_acc_stream_sync(stream), &result, check);
      if (NULL != amat_hst && NULL != bmat_hst && NULL != stack_hst) {
        const size_t size = sizeof(ELEM_TYPE) * (mk * na + kn * nb) + sizeof(int) * 3 * stack_size;
        duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
        PRINTF("copy-in (%i MB): %.2g ms %.1f GB/s\n", (int)((size + (1 << 19)) >> 20), 1000.0 * duration,
          size / (duration * (1ULL << 30)));
      }
#endif
#if defined(TRANSPOSE) && defined(VALIDATE)
      /* warmup execution and prebuild transpose-kernel */
      for (r = 0; r < warmup / 2; ++r) {
        CHECK(libsmm_acc_transpose(trans_dev, 0 /*offset*/, nb, bmat_dev, DBCSR_TYPE(ELEM_TYPE), k, n, MAX_KERNEL_DIM, stream),
          &result, check);
        CHECK(libsmm_acc_transpose(trans_dev, 0 /*offset*/, nb, bmat_dev, DBCSR_TYPE(ELEM_TYPE), n, k, MAX_KERNEL_DIM, stream),
          &result, check);
      }
#  if defined(USE_LIBXSMM)
      CHECK(c_dbcsr_acc_stream_sync(stream), &result, check);
      start = libxsmm_timer_tick();
#  endif
      /* to perform NN-SMMs on the device, all B-matrices are transposed upfront (SMM-kernel is limited to NT) */
      CHECK(libsmm_acc_transpose(trans_dev, 0 /*offset*/, nb, bmat_dev, DBCSR_TYPE(ELEM_TYPE), k, n, MAX_KERNEL_DIM, stream),
        &result, check);
#  if defined(USE_LIBXSMM)
      CHECK(c_dbcsr_acc_stream_sync(stream), &result, check);
      transpose = libxsmm_timer_duration(start, libxsmm_timer_tick());
#  endif
#endif
      /* warmup execution and prebuild SMM-kernel */
      for (r = 0; r < warmup; ++r) {
        CHECK(libsmm_acc_process(stack_hst, stack_dev, stack_size, DBCSR_TYPE(ELEM_TYPE), amat_dev, bmat_dev, cmat_dev, m, n, k,
                MAX_KERNEL_DIM, 1 /*homogeneous*/, stream, stream),
          &result, check);
      }
      CHECK(c_dbcsr_acc_memset_zero(cmat_dev, 0 /*offset*/, sizeof(ELEM_TYPE) * mn * nc, stream), &result, check);
#if defined(USE_LIBXSMM)
      CHECK(c_dbcsr_acc_stream_sync(stream), &result, check);
      start = libxsmm_timer_tick();
#endif
      for (r = 0; r < nrepeat; ++r) {
        /* GPU-kernel is limited to C += Ai * Bi^T, i.e., NT (for NN, all Bi must be transposed upfront) */
        CHECK(libsmm_acc_process(stack_hst, stack_dev, stack_size, DBCSR_TYPE(ELEM_TYPE), amat_dev, bmat_dev, cmat_dev, m, n, k,
                MAX_KERNEL_DIM, 1 /*homogeneous*/, stream, stream),
          &result, check);
      }
#if defined(USE_LIBXSMM)
      CHECK(c_dbcsr_acc_stream_sync(stream), &result, check);
      duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration && EXIT_SUCCESS == result) {
#  if defined(TRANSPOSE) && defined(VALIDATE)
        PRINTF("transpose: %.2g ms %.1f GFLOPS/s\n", 1000.0 * (duration + transpose) / (nrepeat * smm_nrepeat),
          1E-9 * ((size_t)2 * m * n * k * stack_size * nrepeat * smm_nrepeat) / (duration + transpose));
#  endif
        PRINTF("device: %.2g ms %.1f GFLOPS/s\n", 1000.0 * duration / (nrepeat * smm_nrepeat),
          1E-9 * ((size_t)2 * m * n * k * stack_size * nrepeat * smm_nrepeat) / duration);
      }
      else {
#  if defined(TRANSPOSE)
        PRINTF("transpose: 0 ms 0 GFLOPS/s\n");
#  endif
        PRINTF("device: 0 ms 0 GFLOPS/s\n");
      }
#  if defined(VALIDATE)
      {
        ELEM_TYPE* const gold_hst = (ELEM_TYPE*)(0 != check ? libxsmm_malloc(sizeof(ELEM_TYPE) * mn * nc) : NULL);
        /* determine host's performance independent of current result code/status */
        if (NULL != gold_hst && NULL != amat_hst && NULL != bmat_hst && NULL != stack_hst) {
          const ELEM_TYPE alpha = 1, beta = 1;
          const char transa = 'N';
#    if defined(TRANSPOSE)
          const char transb = 'N';
#    else
          const char transb = 'T';
#    endif
          memset(gold_hst, 0, sizeof(ELEM_TYPE) * mn * nc);
          for (r = 0; r < warmup; ++r) {
            ACC_BENCH_GEMM_BATCH(LIBXSMM_DATATYPE(ELEM_TYPE), LIBXSMM_DATATYPE(ELEM_TYPE), &transa, &transb, m, n, k, &alpha,
              amat_hst, &m /*lda*/, stack_hst + 0 /*stride_a*/, bmat_hst, &k /*ldb*/, stack_hst + 1 /*stride_b*/, &beta, gold_hst,
              &m /*ldc*/, stack_hst + 2 /*stride_c*/, sizeof(int) * 3, 1 /*index_base*/, stack_size);
          }
          memset(gold_hst, 0, sizeof(ELEM_TYPE) * mn * nc);
          start = libxsmm_timer_tick();
          /* CPU-kernel operates on data that is not initialized in NUMA-aware fashion */
          for (r = 0; r < (nrepeat * smm_nrepeat); ++r) {
            ACC_BENCH_GEMM_BATCH(LIBXSMM_DATATYPE(ELEM_TYPE), LIBXSMM_DATATYPE(ELEM_TYPE), &transa, &transb, m, n, k, &alpha,
              amat_hst, &m /*lda*/, stack_hst + 0 /*stride_a*/, bmat_hst, &k /*ldb*/, stack_hst + 1 /*stride_b*/, &beta, gold_hst,
              &m /*ldc*/, stack_hst + 2 /*stride_c*/, sizeof(int) * 3, 1 /*index_base*/, stack_size);
          }
          duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
          PRINTF("host: %.2g ms %.1f GFLOPS/s\n", 1000.0 * duration / (nrepeat * smm_nrepeat),
            1E-9 * ((size_t)2 * m * n * k * stack_size * nrepeat * smm_nrepeat) / duration);
          /* validate correctness in case of successful result code/status */
          if (EXIT_SUCCESS == result) {
            /* transfer result from device to host for validation */
            CHECK(c_dbcsr_acc_memcpy_d2h(cmat_dev, cmat_hst, sizeof(ELEM_TYPE) * mn * nc, stream), &result, check);
            CHECK(c_dbcsr_acc_stream_sync(stream), &result, check);
#    if defined(USE_LIBXSMM)
            if (EXIT_SUCCESS == result) {
              libxsmm_matdiff_info diff;
              /* validate result buffers at once (including excess/padded space) */
              result = libxsmm_matdiff(&diff, LIBXSMM_DATATYPE(ELEM_TYPE), mn, nc, gold_hst, cmat_hst, &mn, &mn);
              if (EXIT_SUCCESS == result) {
#      if defined(USE_LIBXSMM) && LIBXSMM_VERSION4(1, 17, 0, 0) < LIBXSMM_VERSION_NUMBER
                const double epsilon = libxsmm_matdiff_epsilon(&diff); /* 1.0 - diff.rsq */
#      else
                const double epsilon = diff.normf_rel;
#      endif
                PRINTF("diff.cur: %g", epsilon);
                if (maxdiff < epsilon && NULL != file) maxdiff = epsilon;
                if (0 < epsilon) {
                  if (LIBXSMM_NOTNAN(diff.v_tst)) {
                    PRINTF(" (|%g-%g|=%g)\n", diff.v_ref, diff.v_tst, fabs(diff.v_ref - diff.v_tst));
                  }
                  else {
                    PRINTF(" (%g)\n", diff.v_tst);
                  }
                }
                else {
                  PRINTF("\n");
                }
                if (0 < check && check < epsilon) result = EXIT_FAILURE;
              }
            }
#    endif
          }
        }
        libxsmm_free(gold_hst);
      }
#  endif
#endif
      CHECK(c_dbcsr_acc_host_mem_deallocate(stack_hst, stream), NULL, check);
      CHECK(c_dbcsr_acc_host_mem_deallocate(trans_hst, stream), NULL, check);
      CHECK(c_dbcsr_acc_host_mem_deallocate(amat_hst, stream), NULL, check);
      CHECK(c_dbcsr_acc_host_mem_deallocate(bmat_hst, stream), NULL, check);
      CHECK(c_dbcsr_acc_host_mem_deallocate(cmat_hst, stream), NULL, check);
      CHECK(c_dbcsr_acc_dev_mem_deallocate(stack_dev), NULL, check);
      CHECK(c_dbcsr_acc_dev_mem_deallocate(trans_dev), NULL, check);
      CHECK(c_dbcsr_acc_dev_mem_deallocate(amat_dev), NULL, check);
      CHECK(c_dbcsr_acc_dev_mem_deallocate(bmat_dev), NULL, check);
      CHECK(c_dbcsr_acc_dev_mem_deallocate(cmat_dev), NULL, check);
      CHECK(c_dbcsr_acc_stream_destroy(stream), NULL, check);
      if (0 == result || (0 > result && 0 == check)) {
        parse_params(argc, argv, &file, &snr, &sss, &ssm, &ssn, &ssk, &snc, &sna, &snb);
        if (NULL != file) PRINTF("\n");
        ++nok;
      }
#if defined(USE_LIBXSMM)
      if (0 == result) {
        LIBXSMM_STDIO_ACQUIRE();
        fputs(print_buffer, stdout);
        LIBXSMM_STDIO_RELEASE();
      }
      else
#endif
      {
        if (0 > result) fprintf(stderr, "WARNING: %ix%ix%i-kernel not available or unsuitable!\n", m, n, k);
      }
      if (NULL == file && (0 == result || (0 > result && 0 == check))) break;
    }
  }
  free(rnd); /* release array of random numbers */
#if !defined(__CUDA)
  CHECK(libsmm_acc_finalize(), NULL, check);
#endif
  CHECK(c_dbcsr_acc_finalize(), NULL, check);
#if defined(USE_LIBXSMM) && defined(VALIDATE)
  if (1 < nok) printf("\ndiff.max: %g\n", maxdiff);
#endif
  if (EXIT_SUCCESS != result) {
    if (NULL != file) fclose(file);
    if (0 < result) fprintf(stderr, "\nFAILED\n\n");
    else if (0 == check) result = EXIT_SUCCESS;
  }
  return result;
}
