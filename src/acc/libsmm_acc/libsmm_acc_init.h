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

#ifdef __CUDA
# include "../cuda/acc_cuda.h"
#else
# include "../hip/acc_hip.h"
#endif

extern "C" int libsmm_acc_init (void);

int libsmm_acc_gpu_blas_init();

int libsmm_acc_check_gpu_warp_size_consistency (void);

int acc_get_gpu_warp_size (void);

extern "C" int libsmm_acc_is_thread_safe (void);

extern cublasHandle_t* cublas_handle;

#endif /*LIBSMM_ACC_INIT_H*/
