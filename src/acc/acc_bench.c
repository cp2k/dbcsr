/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: BSD-3-Clause                                                          */
/*------------------------------------------------------------------------------------------------*/
#include "acc_libsmm.h"
#include "acc_bench.h"

#if defined(__LIBXS)
#  include <libxs_malloc.h>
#  include <libxs_timer.h>
#  include <libxs_math.h>
#  include <libxs_gemm.h>
#else /* code depends on LIBXS */
#  include <libxs_source.h>
#  define __LIBXS
#endif

#if defined(_OPENMP)
#  include <omp.h>
#endif

#define PRINTF(...) \
  do { \
    const size_t print_buffer_size = sizeof(print_buffer) - print_offset; \
    const int print_buffer_result = LIBXS_SNPRINTF(print_buffer + print_offset, print_buffer_size, __VA_ARGS__); \
    assert(0 <= print_buffer_result && print_buffer_result < (int)print_buffer_size); \
    print_offset += print_buffer_result; \
  } while (0)

#if !defined(ALIGNMENT)
#  define ALIGNMENT LIBXS_ALIGNMENT
#endif
#if !defined(ELEM_TYPE)
#  define ELEM_TYPE double
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
#if !defined(DEDUPLICATE) && 0
#  define DEDUPLICATE
#endif
#if !defined(NREPEAT)
#  define NREPEAT 3
#endif
#if !defined(XREPEAT)
#  define XREPEAT 66
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

#define CHECK(EXPR, RPTR, VALUE) \
  do { \
    if (NULL != ((const void*)(RPTR))) { \
      if (0 == *((const int*)(RPTR)) || (0 > *((const int*)(RPTR)) && 0 == (VALUE))) { \
        const int check_r_ = (EXPR); \
        if (0 != check_r_) { \
          *((int*)(RPTR)) = check_r_; \
          if (!(0 > check_r_ && 0 == (VALUE))) { \
            fprintf(stderr, "ERROR ACC/BENCH: %s failed (code=%i)\n", #EXPR, check_r_); \
          } \
          assert(0 > check_r_ && 0 == (VALUE)); \
        } \
      } \
    } \
    else { \
      (EXPR); \
    } \
  } while (0)


static void parse_params(int argc, char* argv[], FILE** file, const char** snr, const char** sss, const char** ssm,
  const char** ssn, const char** ssk, const char** snc, const char** sna, const char** snb);

static size_t parse_nbytes(const char* nbytes, size_t* nelems);


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
#if !defined(__OFFLOAD_UNIFIED_MEMORY) || !defined(DEDUPLICATE)
  const char* const env_nrepeat_h2d = getenv("NREPEAT_H2D");
  const int nrepeat_h2d = (NULL == env_nrepeat_h2d ? 1 : MAX(atoi(env_nrepeat_h2d), 1));
#endif
  double maxdiff = 0, perf_h2d = 0, perf_dev = 0, perf_hst = 0;
  const char* const env_check_h2d = getenv("CHECK_H2D");
  const char* const env_check_dev = getenv("CHECK_DEV");
  const char* const env_check_hst = getenv("CHECK_HST");
#if defined(__OPENCL)
  const char* const env_nrepeat_smm = getenv("NREPEAT_SMM");
  const int nrepeat_smm = (NULL == env_nrepeat_smm ? 1 : MAX(atoi(env_nrepeat_smm), 1));
#else
  const int nrepeat_smm = 1;
#endif
  CHECK(libsmm_acc_init(), &result, check);
  if (EXIT_SUCCESS == result) {
    int ndevices = 0;
    result = c_dbcsr_acc_get_ndevices(&ndevices);
    if (EXIT_SUCCESS == result && 0 < ndevices) {
      const char* const env_device = getenv("DEVICE");
      const int nranks = libxs_nranks(), nrank = libxs_nrank();
      int device = ((NULL == env_device || '\0' == *env_device) ? 0 : atoi(env_device));
      device = ((0 <= device && device < ndevices) ? (1 < nranks ? (nrank % ndevices) : device) : -1);
      result = c_dbcsr_acc_set_active_device(device);
      if (EXIT_SUCCESS == result) {
        printf("Activated device%i (ndevices=%i)\n", device, ndevices);
      }
      else {
        fprintf(stderr, "ERROR: Failed to activate device!\n");
      }
    }
    else {
      fprintf(stderr, "ERROR: No ACC-device found!\n");
      if (EXIT_SUCCESS == result) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      rnd = (int*)malloc(sizeof(int) * NRAND);
      if (NULL != rnd) {
        srand(25071975);
        for (i = 0; i < NRAND; ++i) rnd[i] = rand();
        parse_params(argc, argv, &file, &snr, &sss, &ssm, &ssn, &ssk, &snc, &sna, &snb);
      }
      else {
        result = EXIT_FAILURE;
      }
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
      const int ma = (int)LIBXS_UP2(sizeof(ELEM_TYPE) * m, ALIGNMENT);
      const int ka = (int)LIBXS_UP2(sizeof(ELEM_TYPE) * k, ALIGNMENT);
      const int mn = ma * n / (int)sizeof(ELEM_TYPE);
      const int mk = ma * k / (int)sizeof(ELEM_TYPE);
      const int kn = ka * n / (int)sizeof(ELEM_TYPE);
#else
      const int mn = m * n, mk = m * k, kn = k * n;
#endif
      const int max_kernel_dim = ceil(sqrt(m * n));
      int *stack_hst = NULL, *stack_dev = NULL, *trans_hst = NULL, *trans_dev = NULL;
      ELEM_TYPE *amat_hst = NULL, *bmat_hst = NULL, *cmat_hst = NULL;
      ELEM_TYPE *amat_dev = NULL, *bmat_dev = NULL, *cmat_dev = NULL;
      void* stream = NULL;
      libxs_timer_tick_t start;
      int print_offset = 0;
      char print_buffer[1024] = "";
      double transpose = 0, duration = 0;
      const char* const env_batchsize_smm = getenv("BATCHSIZE_SMM");
      const int xrepeat = (0 != check ? NREPEAT : XREPEAT);
      int nrepeat = (0 < inr ? inr : xrepeat);
      int stack_size, na, nb, nc, nr, r;
      if (NULL == env_batchsize_smm) {
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
              if (s < nbytes) {
                ++stack_size;
              }
              else {
                break;
              }
            }
          }
          else {
            stack_size = (int)nelems;
          }
        }
      }
      else { /* parse SMM_BATCHSIZE=batchsize,nrepfactor */
        i = strcspn(env_batchsize_smm, DELIMS);
        if (i < (int)sizeof(DELIMS)) {
          r = atoi(env_batchsize_smm + i + 1);
          if (0 < r) nrepeat = r;
        }
        stack_size = atoi(env_batchsize_smm);
      }
      if (0 >= stack_size) { /* trigger default */
        if (0 > stack_size) { /* randomize batchsize */
          const int rr = rnd[nok % NRAND], ss = -stack_size, bs = (1 < ss ? ss : BATCHSIZE);
          const int limit = (BATCHGRAIN < ss ? LIBXS_UPDIV(bs, BATCHGRAIN) : ss);
          stack_size = (rr % limit + 1) * BATCHGRAIN;
          nrepeat = MAX(LIBXS_UPDIV(BATCHSIZE * nrepeat, stack_size), xrepeat);
        }
        else {
          stack_size = BATCHSIZE; /* plain default */
        }
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
      if (MAX_KERNEL_DIM < max_kernel_dim) {
        fprintf(stderr, "ERROR: Matrix shape (%i) exceeds MAX_KERNEL_DIM=%i\n", max_kernel_dim, MAX_KERNEL_DIM);
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
        INIT_STACK(stack_hst, stack_size, NRAND, rnd, mn, mk, kn, nc, na, nb);
#if defined(_OPENMP)
#  pragma omp parallel
#endif
        {
#if defined(_OPENMP)
#  pragma omp for nowait
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
#if !defined(__OFFLOAD_UNIFIED_MEMORY) || !defined(DEDUPLICATE)
      CHECK(c_dbcsr_acc_dev_mem_allocate((void**)(void*)&amat_dev, sizeof(ELEM_TYPE) * mk * na), &result, check);
      CHECK(c_dbcsr_acc_dev_mem_allocate((void**)(void*)&bmat_dev, sizeof(ELEM_TYPE) * kn * nb), &result, check);
      CHECK(c_dbcsr_acc_dev_mem_allocate((void**)(void*)&cmat_dev, sizeof(ELEM_TYPE) * mn * nc), &result, check);
      CHECK(c_dbcsr_acc_dev_mem_allocate((void**)(void*)&stack_dev, sizeof(int) * 3 * stack_size), &result, check);
      CHECK(c_dbcsr_acc_dev_mem_allocate((void**)(void*)&trans_dev, sizeof(int) * nb), &result, check);
      CHECK(c_dbcsr_acc_memset_zero(cmat_dev, 0 /*offset*/, sizeof(ELEM_TYPE) * mn * nc, stream), &result, check);
      CHECK(c_dbcsr_acc_memcpy_h2d(trans_hst, trans_dev, sizeof(int) * nb, stream), &result, check);
      CHECK(c_dbcsr_acc_stream_sync(stream), &result, check);
      start = libxs_timer_tick();
      CHECK(c_dbcsr_acc_memcpy_h2d(amat_hst, amat_dev, sizeof(ELEM_TYPE) * mk * na, stream), &result, check);
      CHECK(c_dbcsr_acc_memcpy_h2d(bmat_hst, bmat_dev, sizeof(ELEM_TYPE) * kn * nb, stream), &result, check);
      CHECK(c_dbcsr_acc_memcpy_h2d(stack_hst, stack_dev, sizeof(int) * 3 * stack_size, stream), &result, check);
      if (1 < nrepeat_h2d) {
        CHECK(c_dbcsr_acc_stream_sync(stream), &result, check);
        start = libxs_timer_tick();
        for (r = 0; r < nrepeat_h2d; ++r) {
          CHECK(c_dbcsr_acc_memcpy_h2d(amat_hst, amat_dev, sizeof(ELEM_TYPE) * mk * na, stream), &result, check);
          CHECK(c_dbcsr_acc_memcpy_h2d(bmat_hst, bmat_dev, sizeof(ELEM_TYPE) * kn * nb, stream), &result, check);
          CHECK(c_dbcsr_acc_memcpy_h2d(stack_hst, stack_dev, sizeof(int) * 3 * stack_size, stream), &result, check);
        }
      }
      CHECK(c_dbcsr_acc_stream_sync(stream), &result, check);
      if (NULL != amat_hst && NULL != bmat_hst && NULL != stack_hst && EXIT_SUCCESS == result) {
        const size_t size = (sizeof(ELEM_TYPE) * (mk * na + kn * nb) + sizeof(int) * 3 * stack_size);
        duration = libxs_timer_duration(start, libxs_timer_tick()) / nrepeat_h2d;
        perf_h2d = size / (duration * (1ULL << 30));
        PRINTF("copy-in (%i MB): %.2g ms %.1f GB/s\n", (int)((size + (1 << 19)) >> 20), 1000.0 * duration, perf_h2d);
      }
#else
      amat_dev = amat_hst;
      bmat_dev = bmat_hst;
      cmat_dev = cmat_hst;
      stack_dev = stack_hst;
      trans_dev = trans_hst;
      CHECK(c_dbcsr_acc_memset_zero(cmat_dev, 0 /*offset*/, sizeof(ELEM_TYPE) * mn * nc, stream), &result, check);
#endif
      /* warmup execution and prebuild transpose-kernel */
      for (r = 0; r < warmup / 2; ++r) {
        CHECK(libsmm_acc_transpose(trans_dev, 0 /*offset*/, nb, bmat_dev, DBCSR_TYPE(ELEM_TYPE), k, n, MAX_KERNEL_DIM, stream),
          &result, check);
        CHECK(libsmm_acc_transpose(trans_dev, 0 /*offset*/, nb, bmat_dev, DBCSR_TYPE(ELEM_TYPE), n, k, MAX_KERNEL_DIM, stream),
          &result, check);
      }
      CHECK(c_dbcsr_acc_stream_sync(stream), &result, check);
      start = libxs_timer_tick();
      CHECK(libsmm_acc_transpose(trans_dev, 0 /*offset*/, nb, bmat_dev, DBCSR_TYPE(ELEM_TYPE), k, n, MAX_KERNEL_DIM, stream),
        &result, check);
      CHECK(c_dbcsr_acc_stream_sync(stream), &result, check);
      transpose = libxs_timer_duration(start, libxs_timer_tick());
      /* warmup execution and prebuild SMM-kernel */
      for (r = 0; r < warmup; ++r) {
        CHECK(libsmm_acc_process(stack_hst, stack_dev, stack_size, DBCSR_TYPE(ELEM_TYPE), amat_dev, bmat_dev, cmat_dev, m, n, k,
                MAX_KERNEL_DIM, 1 /*homogeneous*/, stream, stream),
          &result, check);
      }
      CHECK(c_dbcsr_acc_memset_zero(cmat_dev, 0 /*offset*/, sizeof(ELEM_TYPE) * mn * nc, stream), &result, check);
      CHECK(c_dbcsr_acc_stream_sync(stream), &result, check);
      start = libxs_timer_tick();
      for (r = 0; r < nrepeat; ++r) {
        CHECK(libsmm_acc_process(stack_hst, stack_dev, stack_size, DBCSR_TYPE(ELEM_TYPE), amat_dev, bmat_dev, cmat_dev, m, n, k,
                MAX_KERNEL_DIM, 1 /*homogeneous*/, stream, stream),
          &result, check);
      }
      CHECK(c_dbcsr_acc_stream_sync(stream), &result, check);
      duration = libxs_timer_duration(start, libxs_timer_tick());
      if (EXIT_SUCCESS == result) {
        if (0 < duration) {
          PRINTF("transpose: %.2g ms %.1f GFLOPS/s\n", 1000.0 * (duration + transpose) / (nrepeat * nrepeat_smm),
            1E-9 * ((size_t)2 * m * n * k * stack_size * nrepeat * nrepeat_smm) / (duration + transpose));
          perf_dev = 1E-9 * ((size_t)2 * m * n * k * stack_size * nrepeat * nrepeat_smm) / duration;
          PRINTF("device: %.2g ms %.1f GFLOPS/s\n", 1000.0 * duration / (nrepeat * nrepeat_smm), perf_dev);
        }
        else {
          PRINTF("transpose: 0 ms 0 GFLOPS/s\n");
          PRINTF("device: 0 ms 0 GFLOPS/s\n");
        }
      }
      if (EXIT_SUCCESS == result) {
        ELEM_TYPE* const gold_hst = (ELEM_TYPE*)(0 != check ? libxs_malloc(NULL, sizeof(ELEM_TYPE) * mn * nc, 0) : NULL);
        if (NULL != gold_hst && NULL != amat_hst && NULL != bmat_hst && NULL != stack_hst) {
          const ELEM_TYPE alpha = 1, beta = 1;
          const char transa = 'N', transb = 'N';
          libxs_registry_t* host_registry = libxs_registry_create();
          const libxs_gemm_config_t* const host_config = libxs_gemm_dispatch(
            LIBXS_DATATYPE(ELEM_TYPE), transa, transb, m, n, k, m, k, m, &alpha, &beta, host_registry);
          memset(gold_hst, 0, sizeof(ELEM_TYPE) * mn * nc);
          for (r = 0; r < warmup; ++r) {
#if defined(_OPENMP)
#  pragma omp parallel
            libxs_gemm_index_task(amat_hst, stack_hst + 0 /*stride_a*/, bmat_hst, stack_hst + 1 /*stride_b*/, gold_hst,
              stack_hst + 2 /*stride_c*/, sizeof(int) * 3, 1 /*index_base*/, stack_size, host_config, omp_get_thread_num(),
              omp_get_num_threads());
#else
            libxs_gemm_index(amat_hst, stack_hst + 0 /*stride_a*/, bmat_hst, stack_hst + 1 /*stride_b*/, gold_hst,
              stack_hst + 2 /*stride_c*/, sizeof(int) * 3, 1 /*index_base*/, stack_size, host_config);
#endif
          }
          memset(gold_hst, 0, sizeof(ELEM_TYPE) * mn * nc);
          start = libxs_timer_tick();
          for (r = 0; r < (nrepeat * nrepeat_smm); ++r) {
#if defined(_OPENMP)
#  pragma omp parallel
            libxs_gemm_index_task(amat_hst, stack_hst + 0 /*stride_a*/, bmat_hst, stack_hst + 1 /*stride_b*/, gold_hst,
              stack_hst + 2 /*stride_c*/, sizeof(int) * 3, 1 /*index_base*/, stack_size, host_config, omp_get_thread_num(),
              omp_get_num_threads());
#else
            libxs_gemm_index(amat_hst, stack_hst + 0 /*stride_a*/, bmat_hst, stack_hst + 1 /*stride_b*/, gold_hst,
              stack_hst + 2 /*stride_c*/, sizeof(int) * 3, 1 /*index_base*/, stack_size, host_config);
#endif
          }
          libxs_gemm_release_registry(host_registry);
          duration = libxs_timer_duration(start, libxs_timer_tick());
          perf_hst = 1E-9 * ((size_t)2 * m * n * k * stack_size * nrepeat * nrepeat_smm) / duration;
          PRINTF("host: %.2g ms %.1f GFLOPS/s\n", 1000.0 * duration / (nrepeat * nrepeat_smm), perf_hst);
          if (EXIT_SUCCESS == result) {
#if !defined(__OFFLOAD_UNIFIED_MEMORY) || !defined(DEDUPLICATE)
            CHECK(c_dbcsr_acc_memcpy_d2h(cmat_dev, cmat_hst, sizeof(ELEM_TYPE) * mn * nc, stream), &result, check);
            CHECK(c_dbcsr_acc_stream_sync(stream), &result, check);
#endif
            if (EXIT_SUCCESS == result) {
              libxs_matdiff_t diff;
              result = libxs_matdiff(&diff, LIBXS_DATATYPE(ELEM_TYPE), mn, nc, gold_hst, cmat_hst, &mn, &mn);
              if (EXIT_SUCCESS == result) {
                const double epsilon = libxs_matdiff_epsilon(&diff);
                PRINTF("diff.cur: %g", epsilon);
                if (maxdiff < epsilon && NULL != file) maxdiff = epsilon;
                if (0 < epsilon) {
                  if (LIBXS_NOTNAN(diff.v_tst)) {
                    PRINTF(" (|%g-%g|=%g)\n", diff.v_ref, diff.v_tst, diff.linf_abs);
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
              else {
                fprintf(stderr, "ERROR: failed to validate!\n");
              }
            }
          }
        }
        libxs_free(gold_hst);
      }
      CHECK(c_dbcsr_acc_host_mem_deallocate(stack_hst, stream), NULL, check);
      CHECK(c_dbcsr_acc_host_mem_deallocate(trans_hst, stream), NULL, check);
      CHECK(c_dbcsr_acc_host_mem_deallocate(amat_hst, stream), NULL, check);
      CHECK(c_dbcsr_acc_host_mem_deallocate(bmat_hst, stream), NULL, check);
      CHECK(c_dbcsr_acc_host_mem_deallocate(cmat_hst, stream), NULL, check);
#if !defined(__OFFLOAD_UNIFIED_MEMORY) || !defined(DEDUPLICATE)
      CHECK(c_dbcsr_acc_dev_mem_deallocate(stack_dev), NULL, check);
      CHECK(c_dbcsr_acc_dev_mem_deallocate(trans_dev), NULL, check);
      CHECK(c_dbcsr_acc_dev_mem_deallocate(amat_dev), NULL, check);
      CHECK(c_dbcsr_acc_dev_mem_deallocate(bmat_dev), NULL, check);
      CHECK(c_dbcsr_acc_dev_mem_deallocate(cmat_dev), NULL, check);
#endif
      CHECK(c_dbcsr_acc_stream_destroy(stream), NULL, check);
      if (0 == result || (0 > result && 0 == check)) {
        parse_params(argc, argv, &file, &snr, &sss, &ssm, &ssn, &ssk, &snc, &sna, &snb);
        if (NULL != file) PRINTF("\n");
        ++nok;
      }
      if (0 == result) {
        LIBXS_STDIO_ACQUIRE();
        fputs(print_buffer, stdout);
        LIBXS_STDIO_RELEASE();
      }
      else {
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
  if (1 < nok) printf("\ndiff.max: %g\n", maxdiff);
  if (EXIT_SUCCESS == result) {
    if (NULL != env_check_h2d && 0 < perf_h2d) {
      const double check_h2d = atof(env_check_h2d);
      if (perf_h2d < check_h2d) result = EXIT_FAILURE;
    }
    if (NULL != env_check_dev && 0 < perf_dev) {
      const double check_dev = atof(env_check_dev);
      if (perf_dev < check_dev) result = EXIT_FAILURE;
    }
    if (NULL != env_check_hst && 0 < perf_hst) {
      const double check_hst = atof(env_check_hst);
      if (perf_hst < check_hst) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS != result) fprintf(stderr, "\nFAILED\n\n");
  }
  else {
    if (NULL != file) fclose(file);
    if (0 < result) {
      fprintf(stderr, "\nFAILED\n\n");
    }
    else if (0 == check) {
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


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
    const long long int ibytes = strtol(nbytes, &end, 10);
    if (NULL != end) {
      if (NULL == unit && '\0' == *end) {
        result = (size_t)ibytes;
        if (NULL != nelems) *nelems = result;
      }
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
