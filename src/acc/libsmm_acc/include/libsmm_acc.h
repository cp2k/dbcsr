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

#include "../../include/acc.h"

#ifdef __cplusplus
extern "C" {
#endif

int libsmm_acc_process(void* param_stack, int stack_size,
    int nparams, int datatype, void* a_data, void* b_data, void* c_data,
    int m_max, int n_max, int k_max, int def_mnk, acc_stream_t stream);

int libsmm_acc_transpose(void* trs_stack, int offset, int nblks,
    void* buffer, int datatype, int m, int n, acc_stream_t stream);

int libsmm_acc_libcusmm_is_thread_safe(void);

#ifdef __cplusplus
}
#endif

#endif /*DBCSR_ACC_LIBSMM_H*/
