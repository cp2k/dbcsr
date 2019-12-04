/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#ifndef LIBSMM_ACC_H
#define LIBSMM_ACC_H

#include "../include/acc_libsmm.h"
#include "../omp/dbcsr_omp.h"
#include <stdlib.h>

#if !defined(ACC_EXIT_FALLBACK)
/** signals fall-back code; not an error/failure. */
#define ACC_EXIT_FALLBACK (EXIT_FAILURE + 1)
#endif

#ifdef __cplusplus
extern "C" {
#endif

int libsmm_acc_transpose_d(const dbcsr_omp_dependency_t* in, const dbcsr_omp_dependency_t* out,
  const int* dev_trs_stack, int offset, int nblks, double* dev_data, int m, int n);
int libsmm_acc_transpose_s(const dbcsr_omp_dependency_t* in, const dbcsr_omp_dependency_t* out,
  const int* dev_trs_stack, int offset, int nblks, float* dev_data, int m, int n);

int libsmm_acc_process_d(const dbcsr_omp_dependency_t* in, const dbcsr_omp_dependency_t* out,
  const libsmm_acc_stack_descriptor_type* dev_param_stack, int stack_size, int nparams,
  const double* dev_a_data, const double* dev_b_data, double* dev_c_data,
  int m_max, int n_max, int k_max);
int libsmm_acc_process_s(const dbcsr_omp_dependency_t* in, const dbcsr_omp_dependency_t* out,
  const libsmm_acc_stack_descriptor_type* dev_param_stack, int stack_size, int nparams,
  const float* dev_a_data, const float* dev_b_data, float* dev_c_data,
  int m_max, int n_max, int k_max);

#ifdef __cplusplus
}
#endif

#endif /*LIBSMM_ACC_H*/
