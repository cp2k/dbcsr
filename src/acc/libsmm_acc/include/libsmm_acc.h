/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#ifdef __cplusplus
 extern "C" {
#endif

int libsmm_acc_process (void *param_stack, int stack_size,
    int nparams, int datatype, void *a_data, void *b_data, void *c_data,
    int m_max, int n_max, int k_max, int def_mnk, void* stream);

int libsmm_acc_transpose (void *trs_stack, int offset, int nblks,
    void *buffer, int datatype, int m, int n, void* stream);

#ifdef __cplusplus
 }
#endif
