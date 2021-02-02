/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#ifndef LIBSMM_ACC_INIT_H
#define LIBSMM_ACC_INIT_H

#include "../cuda/acc_blas.h"

#include <vector>
#include <string>

#if !defined(NO_DBCSR_TIMESET)
extern "C" void c_dbcsr_timeset(const char** routineN, int* routineN_len, int* handle);
void timeset(const std::string& routine_name, int& handle);

extern "C" void c_dbcsr_timestop(int* handle);
void timestop(int handle);
#endif

extern "C" int libsmm_acc_init (void);
extern "C" int libsmm_acc_finalize (void);

int libsmm_acc_gpu_blas_init();

int libsmm_acc_check_gpu_warp_size_consistency (void);

int acc_get_gpu_warp_size (void);

extern "C" int libsmm_acc_is_thread_safe (void);

extern std::vector<ACC_BLAS(Handle_t)*> acc_blashandles;

#endif /*LIBSMM_ACC_INIT_H*/
