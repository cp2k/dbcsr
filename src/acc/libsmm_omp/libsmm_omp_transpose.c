/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#include "libsmm_omp.h"
#include <assert.h>

#if !defined(LIBSMM_TRANSPOSE_BLOCKDIM_MAX)
# define LIBSMM_TRANSPOSE_BLOCKDIM_MAX 80
#endif

/** Naive implementation */
#define LIBSMM_TRANSPOSE(TYPE, DEPEND_IN, DEPEND_OUT, INDEX, OFFSET, SIZE, M, N, DATA) { \
  int libsmm_transpose_s_; \
  DBCSR_OMP_PRAGMA(omp target teams distribute parallel for simd \
    depend(in:DBCSR_OMP_DEP(DEPEND_IN)) depend(out:DBCSR_OMP_DEP(DEPEND_OUT)) \
    nowait is_device_ptr(INDEX,DATA)) \
  for (libsmm_transpose_s_ = 0; libsmm_transpose_s_ < (SIZE); ++libsmm_transpose_s_) { \
    TYPE libsmm_transpose_tmp_[LIBSMM_TRANSPOSE_BLOCKDIM_MAX*LIBSMM_TRANSPOSE_BLOCKDIM_MAX]; \
    const int libsmm_transpose_idx_ = (INDEX)[(OFFSET)+libsmm_transpose_s_]; \
    TYPE *const libsmm_transpose_mat_ = &((DATA)[libsmm_transpose_idx_]); \
    int libsmm_transpose_i_, libsmm_transpose_j_; \
    for (libsmm_transpose_i_ = 0; libsmm_transpose_i_ < (M); ++libsmm_transpose_i_) { \
      for (libsmm_transpose_j_ = 0; libsmm_transpose_j_ < (N); ++libsmm_transpose_j_) { \
        libsmm_transpose_tmp_[libsmm_transpose_i_*(N)+libsmm_transpose_j_] = \
        libsmm_transpose_mat_[libsmm_transpose_j_*(M)+libsmm_transpose_i_]; \
      } \
    } \
    for (libsmm_transpose_i_ = 0; libsmm_transpose_i_ < (M); ++libsmm_transpose_i_) { \
      for (libsmm_transpose_j_ = 0; libsmm_transpose_j_ < (N); ++libsmm_transpose_j_) { \
        libsmm_transpose_mat_[libsmm_transpose_i_*(N)+libsmm_transpose_j_] = \
        libsmm_transpose_tmp_[libsmm_transpose_i_*(N)+libsmm_transpose_j_]; \
      } \
    } \
  } \
}


#if defined(__cplusplus)
extern "C" {
#endif

int libsmm_acc_transpose_d(const dbcsr_omp_dependency_t* in, const dbcsr_omp_dependency_t* out,
  const int* dev_trs_stack, int offset, int nblks, double* dev_data, int m, int n)
{
  int result;
  (void)(in); (void)(out); /* suppress incorrect warning */
#if defined(LIBSMM_TRANSPOSE_BLOCKDIM_MAX)
  if (LIBSMM_TRANSPOSE_BLOCKDIM_MAX >= m && LIBSMM_TRANSPOSE_BLOCKDIM_MAX >= n) {
    LIBSMM_TRANSPOSE(double, in, out, dev_trs_stack, offset, nblks, m, n, dev_data);
    result = EXIT_SUCCESS;
  }
  else
#endif
  { /* TODO: well-performing library based implementation */
    result = EXIT_FAILURE;
  }
  return result;
}


int libsmm_acc_transpose_s(const dbcsr_omp_dependency_t* in, const dbcsr_omp_dependency_t* out,
  const int* dev_trs_stack, int offset, int nblks, float* dev_data, int m, int n)
{
  int result;
  (void)(in); (void)(out); /* suppress incorrect warning */
#if defined(LIBSMM_TRANSPOSE_BLOCKDIM_MAX)
  if (LIBSMM_TRANSPOSE_BLOCKDIM_MAX >= m && LIBSMM_TRANSPOSE_BLOCKDIM_MAX >= n) {
    LIBSMM_TRANSPOSE(float, in, out, dev_trs_stack, offset, nblks, m, n, dev_data);
    result = EXIT_SUCCESS;
  }
  else
#endif
  { /* TODO: well-performing library based implementation */
    result = EXIT_FAILURE;
  }
  return result;
}

#if defined(__cplusplus)
}
#endif
