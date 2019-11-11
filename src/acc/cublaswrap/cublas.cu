/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#if (__DBCSR_ACC == 2)

#include <stdio.h>
#include "cublas_v2.h"
#include "../acc_error.h"


/****************************************************************************/
extern "C" int cublas_create(cublasHandle_t **handle)
{
  *handle = (cublasHandle_t*)malloc(sizeof(cublasHandle_t));
  cublasStatus_t cStatus = cublasCreate(*handle);
  if (cStatus != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS initialization failed\n");
    return(-1);
  }
  if (acc_error_check(cudaGetLastError())) return(-1);
  return(0);
}

/****************************************************************************/
extern "C" int cublas_destroy(cublasHandle_t *handle)
{
  cublasStatus_t cStatus = cublasDestroy(*handle);
  free(handle);
  if (cStatus != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS finalization failed\n");
    return(-1);
  }
  if (acc_error_check(cudaGetLastError())) return(-1);
  return(0);
}

/****************************************************************************/
extern "C" int cublas_dgemm(cublasHandle_t *handle, char transa, char transb,
		            int m, int n, int k,
			    int a_offset, int b_offset, int c_offset,
			    double *a_data, double *b_data, double *c_data,
			    double alpha, double beta, cudaStream_t *stream)
{
  cublasStatus_t cStatus = cublasSetStream(*handle, *stream);
  if (cStatus != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS SetStream failed\n");
    return(-1);
  }
  cublasOperation_t cTransa = transa=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cTransb = transb=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;
  int &lda = transa=='N' ? m : k;
  int &ldb = transb=='N' ? k : n;

  cublasStatus_t stat = cublasDgemm(*handle, cTransa, cTransb,
				    m, n, k,
				    &alpha, &a_data[ a_offset ], lda,
				    &b_data[ b_offset], ldb,
				    &beta, &c_data[ c_offset], lda);
  if (stat != CUBLAS_STATUS_SUCCESS) return(-1);
  if (acc_error_check(cudaGetLastError())) return(-1);
  return(0);
}

#endif
