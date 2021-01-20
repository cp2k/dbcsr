/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#ifndef OPENCL_LIBSMM_H
#define OPENCL_LIBSMM_H

#include "../../acc_libsmm.h"
#include "../acc_opencl.h"

#if defined(__LIBXSMM)
# include <libxsmm.h>
#else
# error OpenCL backend currently depends on LIBXSMM!
#endif

#if !defined(OPENCL_LIBSMM_TRANS_INPLACE) && 0
# define OPENCL_LIBSMM_TRANS_INPLACE
#endif
#if !defined(OPENCL_LIBSMM_PARAMS_DELIMS)
# define OPENCL_LIBSMM_PARAMS_DELIMS ";, \t|/"
#endif
#if !defined(OPENCL_LIBSMM_DEBUG) && 0
# define OPENCL_LIBSMM_DEBUG
#endif
#if !defined(OPENCL_LIBSMM_SYNC) && 0
# define OPENCL_LIBSMM_SYNC
#endif
#if !defined(OPENCL_LIBSMM_CMEM) && 1
# define OPENCL_LIBSMM_CMEM
#endif
#if !defined(OPENCL_LIBSMM_F32) /*&& !defined(__DBCSR_ACC)*/
# define OPENCL_LIBSMM_F32
#endif
#if !defined(OPENCL_LIBSMM_F64) && 1
# define OPENCL_LIBSMM_F64
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/** Type for querying transpose kernel/configuration. */
typedef struct opencl_libsmm_transkey_t {
  libsmm_acc_data_t type;
  int m, n;
} opencl_libsmm_transkey_t;

/** Type for transpose kernel/configuration. */
typedef struct opencl_libsmm_trans_t {
  cl_kernel kernel;
  size_t wgsize;
} opencl_libsmm_trans_t;

/** Type for querying SMM-kernel/configuration. */
typedef struct opencl_libsmm_smmkey_t {
  libsmm_acc_data_t type;
  int m, n, k;
} opencl_libsmm_smmkey_t;

/** Type for SMM-kernel/configuration. */
typedef struct opencl_libsmm_smm_t {
  cl_kernel kernel;
  size_t wgsize;
  /* tuned parameters for SMM-kernels */
  int bs, bm, bn;
} opencl_libsmm_smm_t;

/** If buffers are hinted for non-concurrent writes aka "OpenCL constant". */
int opencl_libsmm_use_cmem(cl_device_id device);

/* Tokenize parambuf and initialize key/value pair. */
int opencl_libsmm_read_params(char* parambuf,
  opencl_libsmm_smmkey_t* key, opencl_libsmm_smm_t* value);

#if defined(OPENCL_LIBSMM_DEBUG) && defined(_DEBUG)
void opencl_libsmm_print_matrix(FILE* ostream, const char* label,
  libsmm_acc_data_t type, const void* mat, int m, int n);
#endif

#if defined(__cplusplus)
}
#endif

#endif /*OPENCL_LIBSMM_H*/
