/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#include "libsmm_omp.h"
#include <stdlib.h>
#include <assert.h>


#if defined(__cplusplus)
extern "C" {
#endif

int libsmm_acc_process_d(const dbcsr_omp_dependency_t* in, const dbcsr_omp_dependency_t* out,
  const libsmm_acc_stack_descriptor_type* dev_param_stack, int stack_size, int nparams,
  const double* dev_a_data, const double* dev_b_data, double* dev_c_data,
  int m_max, int n_max, int k_max)
{
  int result = EXIT_FAILURE; /* TODO */
  (void)(in); (void)(out); /* suppress incorrect warning */
/*pragma omp target depend(in:DBCSR_OMP_DEP(id)) depend(out:DBCSR_OMP_DEP(od)) nowait is_device_ptr(params,a_data,b_data,c_data)*/
  return result;
}


int libsmm_acc_process_s(const dbcsr_omp_dependency_t* in, const dbcsr_omp_dependency_t* out,
  const libsmm_acc_stack_descriptor_type* dev_param_stack, int stack_size, int nparams,
  const float* dev_a_data, const float* dev_b_data, float* dev_c_data,
  int m_max, int n_max, int k_max)
{
  int result = EXIT_FAILURE; /* TODO */
  (void)(in); (void)(out); /* suppress incorrect warning */
/*pragma omp target depend(in:DBCSR_OMP_DEP(id)) depend(out:DBCSR_OMP_DEP(od)) nowait is_device_ptr(params,a_data,b_data,c_data)*/
  return result;
}

#if defined(__cplusplus)
}
#endif
