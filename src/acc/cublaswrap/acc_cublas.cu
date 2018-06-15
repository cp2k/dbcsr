#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cublas_v2.h>
#include <stdio.h>

#ifdef __ACC_CUBLAS

#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

extern "C" int acc_dgemm(cublasHandle_t *handle, void *stack_params, int stack_size, int mnk, int transa, int transb, 
                          void *a_data, void *b_data, void *c_data, double alpha, double beta, void *stream)
{
    cudaStream_t* custream = (cudaStream_t*) stream;
    double *a = (double*)a_data;
    double *b = (double*)b_data;
    double *c = (double*)c_data;
    int *stack = (int*)stack_params;
    
    // TODO: something smarter
    int stack_width = 7;
 
    /*  
    double *host;
    int sz = m*k > n*k ? n*k : m*k;
    int print_sz = 50;
 
    host = (double*)malloc(sizeof(double)*sz);
    
    cudaMemcpy(host,a,sizeof(double)*sz,cudaMemcpyDeviceToHost);
    for (int i=0; i < (print_sz < sz ? print_sz: sz) ; i++) printf("%f ",host[i]);
    printf("\n");
     
    cudaMemcpy(host,b,sizeof(double)*sz,cudaMemcpyDeviceToHost);
    for (int i=0; i < (print_sz < sz ? print_sz: sz) ; i++) printf("%f ",host[i]);
    printf("\n");
    
    free(host);
 
    */
    
      
  
    //printf("in C-cublas");
 
 
    cublasSetStream(*handle, *custream);  
  
    auto check_transposity = [] (int transa, int transb, int m, int n, int k, cublasOperation_t& trana, cublasOperation_t& tranb, int& lda, int& ldb)
    {
        trana = CUBLAS_OP_N;
        lda = m;
    
        if (transa!=0) {
          trana = CUBLAS_OP_T;
          lda = k;
        }
    
        tranb = CUBLAS_OP_N;
        ldb = k;
    
        if (transb!=0) {
          tranb = CUBLAS_OP_T;
          ldb = n;
        }
    };


    int lda, ldb;
    cublasOperation_t trana, tranb;
/*
    if (mnk == 1)
    {
        int m = stack[ 0 ];
        int n = stack[ 1 ];
        int k = stack[ 2 ];
         
        thrust::host_vector<double*> a_ptrs_h(stack_size), b_ptrs_h(stack_size), c_ptrs_h(stack_size);
 
        for (int i = 0; i < stack_size; i++)
        {
            a_ptrs_h[i] = &a[ stack[ stack_width * i + 3 ] - 1 ];
            b_ptrs_h[i] = &b[ stack[ stack_width * i + 4 ] - 1 ];
            c_ptrs_h[i] = &c[ stack[ stack_width * i + 5 ] - 1 ];
        }

        thrust::device_vector<double*> a_ptrs, b_ptrs, c_ptrs;

        a_ptrs = a_ptrs_h;
        b_ptrs = b_ptrs_h;
        c_ptrs = c_ptrs_h;

        check_transposity(transa, transb, m, n, k, trana, tranb, lda, ldb);
       
        //printf("batched");

        cublasStatus_t stat = cublasDgemmBatched(*handle, trana, tranb, m, n, k, &alpha, (const double**)thrust::raw_pointer_cast(a_ptrs.data()), lda, 
                                                 (const double**)thrust::raw_pointer_cast(b_ptrs.data()), ldb, &beta, (double**)thrust::raw_pointer_cast(c_ptrs.data()), m, stack_size);
    
        if (stat != CUBLAS_STATUS_SUCCESS) return(-1);
        
//        cudaError_t custat = cudaThreadSynchronize();   
//        if (custat != cudaSuccess) return(-1);
        
        return 0;
    }
*/    
    
    for (int i = 0; i < stack_size; i++)
    {
        // get mnk from stack data
        int m = stack[ stack_width * i ];
        int n = stack[ stack_width * i + 1];
        int k = stack[ stack_width * i + 2];
 
        // get first element of data, index - 1 becasue data comes from fortran
        double * a_buf = &a[ stack[ stack_width * i + 3 ] - 1 ];
        double * b_buf = &b[ stack[ stack_width * i + 4 ] - 1];
        double * c_buf = &c[ stack[ stack_width * i + 5 ] - 1 ];
 
        check_transposity(transa, transb, m, n, k, trana, tranb, lda, ldb);
        
        //printf("m n k = %d %d %d, alpha beta = %f %f, a_lb b_lb c_lb = %d %d %d\n",m,n,k,alpha,beta,stack[ stack_width * i + 3], stack[ stack_width * i + 4 ], stack[ stack_width * i + 5 ]);
        cublasStatus_t stat = cublasDgemm(*handle, trana, tranb, m, n, k, &alpha, a_buf, lda, b_buf, ldb, &beta, c_buf, m);
 
        if (stat != CUBLAS_STATUS_SUCCESS) return(-1);
    }
//  cudaDeviceSynchronize();
//    cudaError_t custat = cudaThreadSynchronize();   
//   if (custat != cudaSuccess) return(-1);

//  printf("cublas OK");
  return(0); 
}



// cublas interface ------------------
 
extern "C" int f_cublasCreate(cublasHandle_t **handle)
{
    *handle = (cublasHandle_t*)malloc(sizeof(cublasHandle_t));
    return cublasCreate(*handle);
}
 
extern "C" int f_cublasDgemm(cublasHandle_t *handle,
               cublasOperation_t transa, cublasOperation_t transb,
              int m, int n, int k,
              const double *alpha,
              const double *A, int lda,
              const double *B, int ldb,
              const double *beta,
              double *C, int ldc)
{
    return cublasDgemm(*handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}
 
extern "C" int f_cublasDgemmBatched(cublasHandle_t *handle,
               cublasOperation_t transa, cublasOperation_t transb,
              int m, int n, int k,
              const double *alpha,
              const double **A, int lda,
              const double **B, int ldb,
              const double *beta,
              double **C, int ldc,
              int batch_count)
{
    return cublasDgemmBatched(*handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,batch_count);
}
 
extern "C" void f_cublasDestroy(cublasHandle_t *handle)
{
    cublasDestroy(*handle);
    free(handle);
}
 
extern "C" int f_cudaStreamCreate(cudaStream_t **stream)
{
    *stream = (cudaStream_t *) malloc(sizeof(cudaStream_t));
    return cudaStreamCreate(*stream);
}
 
extern "C" int f_cublasSetStream(cublasHandle_t *handle, cudaStream_t *streamid)
{
    return cublasSetStream(*handle, *streamid);
}
 
extern "C" void f_cudaStreamDestroy(cudaStream_t *stream)
{
    cudaStreamDestroy(*stream);
}

#endif

