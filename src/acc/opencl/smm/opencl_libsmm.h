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
/* OpenCL backend depends on LIBXSMM */
# include <libxsmm_source.h>
# define __LIBXSMM
#endif

#if !defined(OPENCL_LIBSMM_TRANS_INPLACE) && 0
# define OPENCL_LIBSMM_TRANS_INPLACE
#endif
#if !defined(OPENCL_LIBSMM_PARAMS_DELIMS)
# define OPENCL_LIBSMM_PARAMS_DELIMS ";,\t|/"
#endif
#if !defined(OPENCL_LIBSMM_SUITABLE) && 0
# define OPENCL_LIBSMM_SUITABLE
#endif
#if !defined(OPENCL_LIBSMM_DEVMATCH) && 0
# define OPENCL_LIBSMM_DEVMATCH
#endif
#if !defined(OPENCL_LIBSMM_DEBUG) && 0
# define OPENCL_LIBSMM_DEBUG 1
#endif
#if !defined(OPENCL_LIBSMM_CMEM) && 1
# define OPENCL_LIBSMM_CMEM
#endif
#if !defined(OPENCL_LIBSMM_F32) && !defined(__DBCSR_ACC)
# define OPENCL_LIBSMM_F32
#endif
#if !defined(OPENCL_LIBSMM_F64) && 1
# define OPENCL_LIBSMM_F64
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/** Type for querying transpose kernel configuration. */
typedef struct opencl_libsmm_transkey_t {
  libsmm_acc_data_t type; /* must be the 1st data member */
  int m, n;
} opencl_libsmm_transkey_t;

/** Type for transpose kernel configuration. */
typedef struct opencl_libsmm_trans_t {
  cl_kernel kernel; /* must be the 1st data member */
  size_t wgsize;
  /* ACC_OPENCL_VERBOSE: perf. counters */
  double membw_sumlog, membw_comp;
  size_t nexec;
  int size[7];
} opencl_libsmm_trans_t;

/** Type for querying SMM-kernel configuration. */
typedef struct opencl_libsmm_smmkey_t {
  libsmm_acc_data_t type; /* must be the 1st data member */
  int m, n, k;
  /* device that matches configuration (parameters) */
  const char* device; /* must be the last data member */
} opencl_libsmm_smmkey_t;

/** Type for SMM-kernel configuration. */
typedef struct opencl_libsmm_smm_t {
  cl_kernel kernel; /* must be the 1st data member */
  size_t wgsize;
  /* parameters (either pretuned or determined) */
  int bs, bm, bn, bk, ws, wg, lu, nz, al, tb, tc, ap, aa, ab, ac;
  /* ACC_OPENCL_VERBOSE: perf. counters */
  double gflops_sumlog, gflops_comp;
  size_t nexec;
  int size[7];
} opencl_libsmm_smm_t;

/** Type to collect statistics about tuned SMM-kernels */
typedef struct opencl_libsmm_perfest_t {
  double gf_ai_sratio_max, gf_ai_sratio_sumlog, gf_ai_sratio_kahan;
  double gf_ai_dratio_max, gf_ai_dratio_sumlog, gf_ai_dratio_kahan;
  size_t scount, dcount;
} opencl_libsmm_perfest_t;

/** If buffers are hinted for non-concurrent writes aka "OpenCL constant". */
int opencl_libsmm_use_cmem(cl_device_id device);

/** Tokenize parambuf and initialize key/value pair. */
int opencl_libsmm_read_params(char* parambuf,
  opencl_libsmm_smmkey_t* key, opencl_libsmm_smm_t* value,
  opencl_libsmm_perfest_t* perfest,
  char* const* device);

/** Get active device and configuration name. */
int opencl_libsmm_device(void* stream, cl_device_id* device, const char** config);

#if defined(OPENCL_LIBSMM_DEBUG) && defined(_DEBUG)
void opencl_libsmm_print_matrix(FILE* ostream, const char* label,
  libsmm_acc_data_t type, const void* mat, int m, int n);
#endif

c_dbcsr_acc_bool_t libsmm_acc_process_suitable(
  c_dbcsr_acc_bool_t def_mnk, libsmm_acc_data_t datatype,
  int stack_size, int m_max, int n_max, int k_max,
  int max_kernel_dim);

#if defined(__cplusplus)
}
#endif

#endif /*OPENCL_LIBSMM_H*/
