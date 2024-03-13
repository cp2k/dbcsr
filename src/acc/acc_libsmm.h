/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/
#ifndef DBCSR_ACC_LIBSMM_H
#define DBCSR_ACC_LIBSMM_H

#include "acc.h"

#define DBCSR_TYPE(T) DBCSR_CONCATENATE(DBCSR_TYPE_, T)
#define DBCSR_TYPE_double dbcsr_type_real_8
#define DBCSR_TYPE_float dbcsr_type_real_4

#define LIBSMM_ACC_TRANSPOSE_ROUTINE_NAME_STRPTR ((const char**)&libsmm_acc_transpose_routine_name_ptr)
#define LIBSMM_ACC_TRANSPOSE_ROUTINE_NAME_LENPTR (&libsmm_acc_transpose_routine_name_len)
#define LIBSMM_ACC_TRANSPOSE_ROUTINE_NAME_STR (libsmm_acc_transpose_routine_name_str)

#define LIBSMM_ACC_PROCESS_ROUTINE_NAME_STRPTR ((const char**)&libsmm_acc_process_routine_name_ptr)
#define LIBSMM_ACC_PROCESS_ROUTINE_NAME_LENPTR (&libsmm_acc_process_routine_name_len)
#define LIBSMM_ACC_PROCESS_ROUTINE_NAME_STR (libsmm_acc_process_routine_name_str)


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
c_dbcsr_acc_bool_t libsmm_acc_is_thread_safe(void);

int libsmm_acc_transpose(const int* dev_trs_stack, int offset, int stack_size, void* dev_data, libsmm_acc_data_t datatype, int m,
  int n, int max_kernel_dim, void* stream);

int libsmm_acc_process(const int* host_param_stack, const int* dev_param_stack, int stack_size, libsmm_acc_data_t datatype,
  const void* dev_a_data, const void* dev_b_data, void* dev_c_data, int m_max, int n_max, int k_max, int max_kernel_dim,
  c_dbcsr_acc_bool_t def_mnk, void* stack_stream, void* c_stream);

int c_calculate_norms(const double* mat, int nblks, const int* offsets, const int* nelems, float* norms, void* stream_ptr);

static const char libsmm_acc_transpose_routine_name_str[] = "jit_kernel_transpose";
static const char* const libsmm_acc_transpose_routine_name_ptr = libsmm_acc_transpose_routine_name_str;
static const int libsmm_acc_transpose_routine_name_len = (int)sizeof(libsmm_acc_transpose_routine_name_str) - 1;

static const char libsmm_acc_process_routine_name_str[] = "jit_kernel_multiply";
static const char* const libsmm_acc_process_routine_name_ptr = libsmm_acc_process_routine_name_str;
static const int libsmm_acc_process_routine_name_len = (int)sizeof(libsmm_acc_process_routine_name_str) - 1;

#if defined(__cplusplus)
}
#endif

#endif /*DBCSR_ACC_LIBSMM_H*/
