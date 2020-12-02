/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#ifndef DBCSR_ACC_LIBSMM_H
#define DBCSR_ACC_LIBSMM_H

#include "acc.h"

#define DBCSR_TYPE(T) DBCSR_CONCATENATE(DBCSR_TYPE_, T)
#define DBCSR_TYPE_double dbcsr_type_real_8
#define DBCSR_TYPE_float dbcsr_type_real_4


#if defined(__cplusplus)
extern "C" {
#endif

typedef enum libsmm_acc_data_t {
  dbcsr_type_real_4 = 1,
  dbcsr_type_real_8 = 3,
  dbcsr_type_complex_4 = 5,
  dbcsr_type_complex_8 = 7
} libsmm_acc_data_t;

int libsmm_acc_init(void);
int libsmm_acc_finalize(void);
acc_bool_t libsmm_acc_is_thread_safe(void);

int libsmm_acc_transpose(const int* dev_trs_stack, int offset, int stack_size,
  void* dev_data, libsmm_acc_data_t datatype, int m, int n, int max_kernel_dim, void* stream);

int libsmm_acc_process(const int* host_param_stack, const int* dev_param_stack, int stack_size,
  int nparams, libsmm_acc_data_t datatype, const void* dev_a_data, const void* dev_b_data, void* dev_c_data,
  int m_max, int n_max, int k_max, int max_kernel_dim, acc_bool_t def_mnk, void* stack_stream, void* c_stream);

#if defined(__cplusplus)
}
#endif

#endif /*DBCSR_ACC_LIBSMM_H*/
