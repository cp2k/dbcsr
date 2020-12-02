/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include <stdio.h>

#include "acc_blas.h"
#include "../cuda/acc_error.h"


/****************************************************************************/
int acc_blas_create(ACC_BLAS(Handle_t) **handle)
{
  *handle = (ACC_BLAS(Handle_t)*)malloc(sizeof(ACC_BLAS(Handle_t)));
  ACC_BLAS_CALL(Create, (*handle));
  return(0);
}

/****************************************************************************/
int acc_blas_destroy(ACC_BLAS(Handle_t) *handle)
{
  ACC_BLAS(Status_t) cStatus = ACC_BLAS(Destroy)(*handle);
  free(handle);
  if (cStatus != ACC_BLAS_STATUS_SUCCESS) {
    printf ("ACC BLAS finalization failed\n");
    return(-1);
  }
  return(0);
}

/****************************************************************************/
int acc_blas_dgemm(ACC_BLAS(Handle_t) *handle, char transa, char transb,
                   int m, int n, int k,
                   int a_offset, int b_offset, int c_offset,
                   const double *a_data, const double *b_data, double *c_data,
                   double alpha, double beta, ACC(Stream_t) *stream)
{

  ACC_BLAS(Operation_t) cTransa = transa=='N' ? ACC_BLAS_OP_N : ACC_BLAS_OP_T;
  ACC_BLAS(Operation_t) cTransb = transb=='N' ? ACC_BLAS_OP_N : ACC_BLAS_OP_T;
  int &lda = transa=='N' ? m : k;
  int &ldb = transb=='N' ? k : n;

  ACC_BLAS_CALL(SetStream, (*handle, *stream));

  ACC_BLAS_CALL(Dgemm, (*handle, cTransa, cTransb,
				    m, n, k,
				    &alpha, &a_data[a_offset], lda,
				    &b_data[ b_offset], ldb,
				    &beta, &c_data[ c_offset], lda));

  return(0);
}

/****************************************************************************/
int acc_blas_transpose(ACC_BLAS(Handle_t) *handle, int m, int n, int offset,
                       double *data, ACC(Stream_t) *stream)
{

  ACC_BLAS_CALL(SetStream, (*handle, *stream));

  const double alpha = 1.;
  const double beta = 0.;

  // cu/hip blas does not have in-place transpose:
  // we duplicate the matrix into a temporary buffer before transposing
  double *data_temp;
  ACC_API_CALL(Malloc, (&data_temp, m * n * sizeof(double)));
  ACC_API_CALL(Memcpy, (data, data_temp, m * n * sizeof(double), ACC(MemcpyDeviceToDevice)));

  ACC_BLAS_CALL(Dgeam, (*handle, ACC_BLAS_OP_T, ACC_BLAS_OP_N,
				m, n,
				&alpha, data_temp, n,
				&beta, data_temp, m,
                &data[offset], m));

  // free temporary buffer
  ACC_API_CALL(Free, (data_temp));

  return(0);
}
