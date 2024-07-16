/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: BSD-3-Clause                                                          */
/*------------------------------------------------------------------------------------------------*/
#ifndef OPENCL_LIBSMM_H
#define OPENCL_LIBSMM_H

#include "../../acc_libsmm.h"
#include "../acc_opencl.h"

/* Inplace-transpose by default (corresponding environment variable exists also) */
#if !defined(OPENCL_LIBSMM_TRANS_INPLACE) && 0
#  define OPENCL_LIBSMM_TRANS_INPLACE
#endif
/* Validate kernels (1: OPENCL_LIBSMM_VALIDATE_SMM, 2: OPENCL_LIBSMM_VALIDATE_TRANS) */
#if !defined(OPENCL_LIBSMM_VALIDATE) && 0
#  define OPENCL_LIBSMM_VALIDATE 1
#endif
#if !defined(OPENCL_LIBSMM_F32_OFF) && defined(__DBCSR_ACC) && 0
#  define OPENCL_LIBSMM_F32_OFF
#endif
#if !defined(OPENCL_LIBSMM_F32) && !defined(OPENCL_LIBSMM_F32_OFF)
#  define OPENCL_LIBSMM_F32
#endif
#if !defined(OPENCL_LIBSMM_F64) && !defined(OPENCL_LIBSMM_F64_OFF)
#  define OPENCL_LIBSMM_F64
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
} opencl_libsmm_trans_t;

/** Type for querying SMM-kernel configuration. */
typedef struct opencl_libsmm_smmkey_t {
  libsmm_acc_data_t type; /* must be the 1st data member */
  int m, n, k;
  /* device matching configuration (parameters) */
  unsigned int devuid;
} opencl_libsmm_smmkey_t;

/** Type for SMM-kernel configuration. */
typedef struct opencl_libsmm_smm_t {
  cl_kernel kernel[2]; /* must be the 1st data member */
  size_t wgsize[2];
  double gflops;
  /* (pseudo-)parameters (either pretuned or determined) */
  int s, bs, bm, bn, bk, ws, wg, lu, nz, al, tb, tc, ap, aa, ab, ac, flags;
} opencl_libsmm_smm_t;

/** Type to collect statistics about tuned SMM-kernels */
typedef struct opencl_libsmm_perfest_t {
  double gf_ai_sratio_max, gf_ai_sratio_sumlog, gf_ai_sratio_kahan;
  double gf_ai_dratio_max, gf_ai_dratio_sumlog, gf_ai_dratio_kahan;
  size_t scount, dcount;
} opencl_libsmm_perfest_t;

/** If buffers are hinted for non-concurrent writes aka "OpenCL constant". */
int opencl_libsmm_use_cmem(cl_device_id device);

/**
 * TRANS-kernel: write key and tunables into a (file-)stream.
 * If config=NULL, key/parameter names are written. The arguments
 * delim, begin, and close are optional as well (can be NULL).
 * With only the key being written the config still controls
 * if values or names are written.
 * Returns the number of characters written (negative if error).
 */
int opencl_libsmm_write_trans_params(FILE* stream, int only_key, const opencl_libsmm_transkey_t* key,
  const opencl_libsmm_trans_t* config, const char* delim, const char* begin, const char* close);

/**
 * SMM-kernel: write key and tunables into a (file-)stream.
 * The environment variable OPENCL_LIBSMM_SMM_PARAMS="<output>"
 * reproduces a configuration. If config=NULL, key/parameter
 * names are written. The arguments delim, begin, and close
 * are optional as well (can be NULL).
 * With only the key being written the config still controls
 * if values or names are written.
 * Returns the number of characters written (negative if error).
 */
int opencl_libsmm_write_smm_params(FILE* stream, int only_key, const opencl_libsmm_smmkey_t* key, const opencl_libsmm_smm_t* config,
  const char* delim, const char* begin, const char* close);

/** Tokenize parambuf and initialize key/value pair. */
int opencl_libsmm_read_smm_params(char* parambuf, opencl_libsmm_smmkey_t* key, opencl_libsmm_smm_t* value,
  opencl_libsmm_perfest_t* perfest, char* device, int* key_ok);

#if defined(OPENCL_LIBSMM_VALIDATE) && defined(_DEBUG)
void opencl_libsmm_print_matrix(FILE* ostream, const char* label, libsmm_acc_data_t type, const void* mat, int m, int n);
#endif

c_dbcsr_acc_bool_t libsmm_acc_process_suitable(
  c_dbcsr_acc_bool_t def_mnk, libsmm_acc_data_t datatype, int stack_size, int m_max, int n_max, int k_max, int max_kernel_dim);

#if defined(__cplusplus)
}
#endif

#endif /*OPENCL_LIBSMM_H*/
