/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#if (__DBCSR_ACC == 3)

#include <stdio.h>
#include "hipblas_v2.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "../acc_error.h"


/****************************************************************************/
extern "C" int hipblas_create(hipblasHandle_t **handle)
{
  *handle = (hipblasHandle_t*)malloc(sizeof(hipblasHandle_t));
  hipblasStatus_t cStatus = hipblasCreate(*handle);
  if (cStatus != HIPBLAS_STATUS_SUCCESS) {
    printf ("HIPBLAS initialization failed\n");
    return(-1);
  }
  if (acc_error_check(hipGetLastError)()) return(-1);
  return(0);
}

/****************************************************************************/
extern "C" int hipblas_destroy(hipblasHandle_t *handle)
{
  hipblasStatus_t cStatus = hipblasDestroy(*handle);
  free(handle);
  if (cStatus != HIPBLAS_STATUS_SUCCESS) {
    printf ("HIPBLAS finalization failed\n");
    return(-1);
  }
  if (acc_error_check(hipGetLastError())) return(-1);
  return(0);
}

/****************************************************************************/
extern "C" int hipblas_dgemm(hipblasHandle_t *handle, char transa, char transb,
		             int m, int n, int k,
			     int a_offset, int b_offset, int c_offset,
			     double *a_data, double *b_data, double *c_data,
			     double alpha, double beta, hipdaStream_t *stream)
{
  hipblasStatus_t cStatus = hipblasSetStream(*handle, *stream);
  if (cStatus != HIPBLAS_STATUS_SUCCESS) {
    printf ("HIPBLAS SetStream failed\n");
    return(-1);
  }
  hipblasOperation_t cTransa = transa=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;
  hipblasOperation_t cTransb = transb=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;
  int &lda = transa=='N' ? m : k;
  int &ldb = transb=='N' ? k : n;

  hipblasStatus_t stat = hipblasDgemm(*handle, cTransa, cTransb,
	   			      m, n, k,
				      &alpha, &a_data[ a_offset ], lda,
				      &b_data[ b_offset], ldb,
				      &beta, &c_data[ c_offset], lda);
  if (stat != HIPBLAS_STATUS_SUCCESS) return(-1);
  if (acc_error_check(hipGetLastError())) return(-1);
  return(0);
}

#endif
