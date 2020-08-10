/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include "acc_cuda.h"
#include "../acc_error.h"


/****************************************************************************/
int cublas_create(cublasHandle_t **handle);

/****************************************************************************/
int cublas_destroy(cublasHandle_t *handle);

/****************************************************************************/
int cublas_dgemm(cublasHandle_t *handle, char transa, char transb,
		         int m, int n, int k,
			     int a_offset, int b_offset, int c_offset,
			     const double *a_data, const double *b_data, double *c_data,
			     double alpha, double beta, cudaStream_t *stream);
