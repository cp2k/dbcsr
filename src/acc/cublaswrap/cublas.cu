#if (__DBCSR_ACC == 2)

#include <stdio.h>
#include "cublas_v2.h"
#include "../cuda/acc_cuda_error.h"

/****************************************************************************/
extern "C" int cublas_create(cublasHandle_t **handle)
{
  *handle = (cublasHandle_t*)malloc(sizeof(cublasHandle_t));
  cublasStatus_t cStatus = cublasCreate(*handle);
  if (cStatus != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS initialization failed\n");
    return(-1);
  }
  if (cuda_error_check(cudaGetLastError())) return(-1);
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
  if (cuda_error_check(cudaGetLastError())) return(-1);
  return(0);
}

/****************************************************************************/
extern "C" int cublas_dgemm_loop(cublasHandle_t *handle, char transa, char transb, 
				 int *stack_params, int ps_width, int stack_size,
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
  int m, n, k;
  int &lda = transa=='N' ? m : k;
  int &ldb = transb=='N' ? k : n;

  for (int ii = 0; ii < stack_size; ii++) {
    // get mnk from stack data
    m = stack_params[ ps_width * ii ];
    n = stack_params[ ps_width * ii + 1];
    k = stack_params[ ps_width * ii + 2];

    // get first element of data, index - 1 becasue data comes from fortran
    double *a_mat = &a_data[ stack_params[ ps_width * ii + 3 ] - 1 ];
    double *b_mat = &b_data[ stack_params[ ps_width * ii + 4 ] - 1 ];
    double *c_mat = &c_data[ stack_params[ ps_width * ii + 5 ] - 1 ];

    cublasStatus_t stat = cublasDgemm(*handle, cTransa, cTransb, m, n, k, &alpha, a_mat, lda, b_mat, ldb, &beta, c_mat, lda);
    if (stat != CUBLAS_STATUS_SUCCESS) return(-1);
  }
  if (cuda_error_check(cudaGetLastError())) return(-1);
  return(0);
}

#endif

