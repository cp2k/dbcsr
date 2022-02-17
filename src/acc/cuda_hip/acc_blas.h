/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/

#ifndef ACC_BLAS_H
#define ACC_BLAS_H

#if defined(__CUDA)
#  include "../cuda/acc_cuda.h"
#elif defined(__HIP)
#  include "../hip/acc_hip.h"
#endif

#include <stdio.h>
#include "acc_error.h"

/****************************************************************************/
int acc_blas_create(ACC_BLAS(Handle_t) * *handle);

/****************************************************************************/
int acc_blas_destroy(ACC_BLAS(Handle_t) * handle);

/****************************************************************************/
int acc_blas_dgemm(ACC_BLAS(Handle_t) * handle, char transa, char transb, int m, int n, int k, int a_offset, int b_offset,
  int c_offset, const double* a_data, const double* b_data, double* c_data, double alpha, double beta, ACC(Stream_t) * stream);

#endif /* ACC_BLAS_H */
