/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <algorithm>
#include "libcusmm_benchmark.h"
#include "parameters.h"
#include "parameters_utils.h"

//===========================================================================
// Allocate memory and cuda events
void libcusmm_benchmark_init(libcusmm_benchmark_t** handle, benchmark_mode mode,
                             int max_m, int max_n, int max_k){

    libcusmm_benchmark_t* h = (libcusmm_benchmark_t*) malloc(sizeof(libcusmm_benchmark_t));
    *handle = h;
    h->mode = mode;

    switch(mode) {
        case tune: 
        case timing: 
            h->n_a = 10000;
            h->n_b = 10000;
            h->n_c = 1000;
            h->n_stack = 16005;
            h->n_stack_trs_a = 0;
            h->n_stack_trs_b = 0;
            break; 
        case test: 
            h->n_a = 100;
            h->n_b = 100;
            h->n_c = 10;
            h->n_stack = 100;
            h->n_stack_trs_a = h->n_a;
            h->n_stack_trs_b = h->n_b;
            break; 
    }

    h->max_m = max_m;
    h->max_n = max_n;
    h->max_k = max_k;

    h->mat_a =     (double*) malloc(h->n_a * max_m * max_k * sizeof(double));
    h->mat_trs_a = (double*) malloc(h->n_a * max_m * max_k * sizeof(double));
    h->mat_b =     (double*) malloc(h->n_b * max_k * max_n * sizeof(double));
    h->mat_trs_b = (double*) malloc(h->n_b * max_k * max_n * sizeof(double));
    h->mat_c =     (double*) malloc(h->n_c * max_m * max_n * sizeof(double));
    h->stack =        (int*) malloc(h->n_stack * 3 * sizeof(int));
    h->stack_trs_a =  (int*) malloc(h->n_stack_trs_a * sizeof(int));
    h->stack_trs_b =  (int*) malloc(h->n_stack_trs_b * sizeof(int));

    cudaMalloc(&h->d_mat_a, h->n_a * max_m * max_k * sizeof(double));
    cudaMalloc(&h->d_mat_b, h->n_b * max_k * max_n * sizeof(double));
    cudaMalloc(&h->d_mat_c, h->n_c * max_m * max_n * sizeof(double));
    cudaMalloc(&h->d_stack, h->n_stack * 3 * sizeof(int));
    cudaMalloc(&h->d_stack_trs_a, h->n_stack_trs_a * sizeof(int));
    cudaMalloc(&h->d_stack_trs_b, h->n_stack_trs_b * sizeof(int));

    cuEventCreate(&h->t_start, CU_EVENT_DEFAULT);
    cuEventCreate(&h->t_stop, CU_EVENT_DEFAULT);

    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess){
      printf("libcusmm_benchmark_init: %s\n", cudaGetErrorString(cudaError));
      exit(1);
    }
}


//===========================================================================
// Free memory and cuda events
void libcusmm_benchmark_finalize(libcusmm_benchmark_t* handle){
    cudaEventDestroy(handle->t_stop);
    cudaEventDestroy(handle->t_start);
    cudaFree(handle->d_stack_trs_b);
    cudaFree(handle->d_stack_trs_a);
    cudaFree(handle->d_stack);
    cudaFree(handle->d_mat_c);
    cudaFree(handle->d_mat_b);
    cudaFree(handle->d_mat_a);
    free(handle->stack_trs_b);
    free(handle->stack_trs_a);
    free(handle->stack);
    free(handle->mat_c);
    free(handle->mat_trs_b);
    free(handle->mat_b);
    free(handle->mat_trs_a);
    free(handle->mat_a);
    free(handle);
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess){
      printf("libcusmm_benchmark_finalize: %s\n", cudaGetErrorString(cudaError));
      exit(1);
    }
}


//===========================================================================
// initialize matrix
void matInit(double* mat, int mat_n, int x, int y, int seed){

 double *m = mat;

 for(int n=0; n<mat_n; n++)
   for(int j=0; j<y; j++)
     for(int i=0; i<x; i++, m++)
       *m = (double) j*x + i + n + seed;

}


//===========================================================================
// initialize the task list ("stack" in CP2K lingo)
// for each of the result matrices we have a random number
void stackInit(int *stack, int n_stack, int n_c, double* mat_c,
               int n_a, double * mat_a, int n_b, double* mat_b,
               int mat_m, int mat_n, int mat_k){

  if(n_stack < n_c){
    printf("Error: n_stack < n_c\n");
    exit(1);
  }

  // on average, we have n_avg matrix products contributing to a result mat_c
  int n_avg = n_stack / n_c;

  int n_imbalance = std::max(1, n_avg-4);

  int c = 0;
  int n_top = 0;
  int p = 0;

 int* s = stack;
  while( p < n_stack ){
    if(c >= n_c) c = n_c-1;

    n_top += n_avg + (rand() % (2*n_imbalance) - n_imbalance);
    if(n_top > n_stack) n_top = n_stack;

    for(;p < n_top; p++){
     int a = rand() % n_a;
     int b = rand() % n_b;

     *s++ =  a * mat_m * mat_k + 1;        // A_src
     *s++ =  b * mat_k * mat_n + 1;        // B_src
     *s++ =  c * mat_m * mat_n + 1;        // C_dst
    }
    c++;
 }
}


//===========================================================================
// initialize the task list ("stack" in CP2K lingo)
void stackInitTransp(int *stack, int n_stack, int mat_m, int mat_n){

  int* s = stack;
  for(int p=0; p<n_stack; p++)
     *s++ =  p * mat_m * mat_n;
}


//===========================================================================
void stackCalc(int* stack, int n_stack, double* mat_c, double *mat_a, double* mat_b,
               int mat_m, int mat_n, int mat_k){

  for(int s=0; s<n_stack; s++){
     int a_base = stack[3 * s    ] - 1;
     int b_base = stack[3 * s + 1] - 1;
     int c_base = stack[3 * s + 2] - 1;

     for(int n=0; n<mat_n; n++){
       for(int m=0; m<mat_m; m++){
         double res = 0.;
         for(int k=0; k<mat_k; k++){
          int a_ind = k * mat_m + m;
          int b_ind = k * mat_n + n;
          res += mat_a[a_base + a_ind] * mat_b[b_base + b_ind];
         }
         int c_ind = n * mat_m +  m;
         mat_c[c_base + c_ind] += res;
       }
     }
  }

}


//===========================================================================
void stackTransp(int* stack, int n_stack, double *mat, double* mat_trs,
                 int mat_m, int mat_n){
    for(int s=0; s < n_stack; s++){
        int offset = stack[s];
        for(int m=0; m < mat_m; m++){
            for(int n=0; n < mat_n; n++){
                int i = n * mat_m + m;
                int r_out = i % mat_n;
                int c_out = i / mat_n;
                int it = r_out * mat_m + c_out;
                mat_trs[offset + i] = mat[offset + it];
            }
        }
    }
}


//===========================================================================
double checkSum(double* mat_c, int n_c, int mat_m, int mat_n){
   double res = 0;
   for(int i=0; i<n_c * mat_m * mat_n; i++){
     res += mat_c[i];
   }
   return res;
}


//===========================================================================
double checkSumTransp(double* mat, int n, int n_stack, int mat_m, int mat_n){
    // for transposition, a regular checkSum does not inform about the 
    // transpose's correctness. Instead, we perform a checkSum on a 
    // sample of elements.
    double res = 0;
    int size = mat_m * mat_n;
    int n_samples = size / 3;
    int step = size / n_samples; 
    for(int s=0; s < n_stack; s++){
        int offset = s * size; 
        for(int idx=s%step; idx < size; idx+=step)
            res += mat[offset + idx]; 
    } 
    return res;
}


//===========================================================================
//Removes special symbols so that the output is usefull for awk and gnuplot.
static void clean_string(char* str_in, char* str_out){
    for(int i=0; i<1000 ; i++){
        if(str_in[i] == '=' || str_in[i] == ',' || str_in[i] == '(' || str_in[i] == ')'){
            str_out[i] = ' ';
         }else{
             str_out[i] = str_in[i];
         }
         if(str_in[i] == 0)
             break;
    }
}


//===========================================================================
int libcusmm_benchmark(libcusmm_benchmark_t* h,
                       int mat_m, int mat_n, int mat_k,
                       int nkernels, KernelLauncher* launchers, char ** kernel_descr){

 if(mat_m > h->max_m || mat_n > h->max_n || mat_k > h->max_k){
     printf("libcusmm_benchmark: got handle with too few resources\n");
     exit(1);
 }
 std::vector<Triplet> blocksizes;
 get_libcusmm_triplets(blocksizes, ht); 
 auto it = std::find(std::begin(blocksizes), std::end(blocksizes), Triplet({ mat_m, mat_n, mat_k }));
 if(it == std::end(blocksizes) && h->mode != tune){
     printf("Triplet %i x %i x %i is not defined in libcusmm\n", mat_m, mat_n, mat_k);
     exit(1);
 }


 int n_iter, n_warm; 
 switch(h->mode) {
   case tune:
   case timing: // for larger matrices few iteration give enough statistics
     n_iter = max(3, 12500/(mat_m * mat_n * mat_k));
     n_warm = min(3, n_iter);
     break;
   case test:
     n_iter = 1;
     n_warm = 1;
     break;
    }

 CUstream stream; 
 cuStreamCreate(&stream, CU_STREAM_DEFAULT);

 int error_counter = 0;
 int best_kernel = -1;
 double best_gflops = 0.0;
 double sumCPU, sumGPU;
 float t_duration;
 char descr[1000], msg_prefix[100]="";
 cudaError_t cudaError;

 memset(h->mat_c, 0, h->n_c * mat_m * mat_n * sizeof(double));
 matInit(h->mat_a, h->n_a, mat_m, mat_k, 42);
 matInit(h->mat_b, h->n_b, mat_k, mat_n, 24);

 if(h->mode == tune)
     printf("Initializing ...\n");
 stackInit(h->stack, h->n_stack, h->n_c, h->mat_c, h->n_a, h->mat_a, h->n_b, h->mat_b, mat_m, mat_n, mat_k);

 // Actually, we would have to calculate the stack n_iter times.
 // We cheat by simply scaling the results of a single stack calulcation.
 stackCalc(h->stack, h->n_stack, h->mat_c, h->mat_a, h->mat_b, mat_m, mat_n, mat_k);
 for(int i=0 ; i < h->n_c*mat_m*mat_n ; i++)
     h->mat_c[i] *= n_iter;

 sumCPU =  checkSum(h->mat_c, h->n_c, mat_m, mat_n);

 cudaMemcpy(h->d_mat_a, h->mat_a, h->n_a * mat_m * mat_k * sizeof(double), cudaMemcpyHostToDevice);
 cudaMemcpy(h->d_mat_b, h->mat_b, h->n_b * mat_k * mat_n * sizeof(double), cudaMemcpyHostToDevice);
 cudaMemcpy(h->d_stack, h->stack, h->n_stack * 3 * sizeof(int), cudaMemcpyHostToDevice);
 // d_mat_c gets zeroed after warmup run

 for(int ikern=0; ikern < nkernels; ikern++){

    // Warmup run (more often if n_iter is small)
    for(int i=0; i<n_warm; i++)
        launchers[ikern](h->d_stack, h->n_stack, stream, mat_m, mat_n, mat_k, h->d_mat_a, h->d_mat_b, h->d_mat_c);
    cudaMemset(h->d_mat_c, 0, h->n_c * mat_m * mat_n * sizeof(double));

    cuEventRecord(h->t_start, stream);

    for(int i=0; i<n_iter; i++)
        launchers[ikern](h->d_stack, h->n_stack, stream, mat_m, mat_n, mat_k, h->d_mat_a, h->d_mat_b, h->d_mat_c);

    cuEventRecord(h->t_stop, stream);
    cuEventSynchronize(h->t_stop);
    cuEventElapsedTime(&t_duration, h->t_start, h->t_stop);

    cudaMemcpy(h->mat_c, h->d_mat_c, h->n_c * mat_m * mat_n * sizeof(double), cudaMemcpyDeviceToHost);

    clean_string(kernel_descr[ikern], descr);

    if(h->mode == tune)
        sprintf(msg_prefix, "params %d / %d\n",ikern+1, nkernels);

    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess){
      printf("%sERROR %s cuda_error: %s\n", msg_prefix, descr, cudaGetErrorString(cudaError));
      error_counter++;
      continue;
    }

    sumGPU =  checkSum(h->mat_c, h->n_c, mat_m, mat_n);
    if(sumGPU != sumCPU){
        printf("%sERROR %s checksum_diff: %g\n",msg_prefix, descr, sumGPU-sumCPU);
        error_counter++;
        continue;
    }

    if(h->mode == tune || h->mode == timing){
       double gflops = ((double) n_iter * h->n_stack * mat_m * mat_n * mat_k * 2 / (1e9))/(t_duration * 1e-3);
       printf("%sOK %s GFlop/s %g\n", msg_prefix, descr, gflops);
       if(best_gflops < gflops){
           best_gflops = gflops;
           best_kernel = ikern;
       }
    }else{
       printf("%sOK %s\n", msg_prefix, descr);
    }
 }

 if(h->mode == tune){
    printf("\n\n");
    if(best_kernel > -1){
        printf("WINNER: %d %s , # %g GFlop/s \n", best_kernel+1, kernel_descr[best_kernel], best_gflops);
    }else{
       printf("WINNER: None\n");
    }
    printf("Number of errors: %d\n", error_counter);
    cudaDeviceReset();
 }

 return(error_counter);
}


//===========================================================================
int libcusmm_benchmark_transpose_(int n_stack, int* stack, int* d_stack,
                                  double* mat, double* mat_trs, double* d_mat, 
                                  int n, int mat_m, int mat_n, 
                                  CUevent start, CUevent stop, char** kernel_descr,
                                  TransposeLauncher* launcher){
 if(mat_m > MAX_BLOCK_DIM || mat_n > MAX_BLOCK_DIM){
     printf("Cannot transpose matrices with dimensions above %i, got (%i x %i)\n",
            MAX_BLOCK_DIM, mat_m, mat_n);
     exit(1);
 }

 CUstream stream;
 cuStreamCreate(&stream, CU_STREAM_DEFAULT);

 int offset = 0;
 int n_warm = 0;
 int n_iter = 1;
 int error_counter = 0;
 double sumCPU, sumGPU;
 float t_duration;
 char descr[1000], msg_prefix[100]="";
 cudaError_t cudaError;

 // Matrix and stack initialization
 matInit(mat, n, mat_m, mat_n, 42);
 memset(mat_trs, 0, n * mat_m * mat_n * sizeof(double));
 stackInitTransp(stack, n_stack, mat_m, mat_n);

 // Reference result on CPU
 stackTransp(stack, n_stack, mat, mat_trs, mat_m, mat_n);
 sumCPU = checkSumTransp(mat_trs, n, n_stack, mat_m, mat_n);

 // Compute on GPU
 cudaMemcpy(d_mat, mat, n * mat_m * mat_n * sizeof(double), cudaMemcpyHostToDevice);
 cudaMemcpy(d_stack, stack, n_stack * sizeof(int), cudaMemcpyHostToDevice);
 cudaError = cudaGetLastError();
 if (cudaError != cudaSuccess){
   printf("%sERROR %s cuda_error: %s\n", msg_prefix, descr, cudaGetErrorString(cudaError));
   error_counter++;
 }

 // Warmup run
 for(int i=0; i<n_warm; i++)
   launcher[0](d_stack, offset, n_stack, d_mat, mat_m, mat_n, stream);

 // Real runs
 cuEventRecord(start, stream);

 for(int i=0; i<n_iter; i++)
   launcher[0](d_stack, offset, n_stack, d_mat, mat_m, mat_n, stream);

 cuEventRecord(stop, stream);
 cuEventSynchronize(stop);
 cuEventElapsedTime(&t_duration, start, stop);

 cudaError = cudaGetLastError();

 // Check for errors and compare libcusmm result on GPU to reference
 cudaMemcpy(mat_trs, d_mat, n * mat_m * mat_n * sizeof(double), cudaMemcpyDeviceToHost);
 clean_string(kernel_descr[0], descr);
 cudaError = cudaGetLastError();
 if (cudaError != cudaSuccess){
   printf("%sERROR %s cuda_error: %s\n", msg_prefix, descr, cudaGetErrorString(cudaError));
   error_counter++;
 }

 sumGPU = checkSumTransp(mat_trs, n, n_stack, mat_m, mat_n);
 if(sumGPU != sumCPU){
     printf("%sERROR %s checksum_diff: %g\n", msg_prefix, descr, sumGPU-sumCPU);
     error_counter++;
 } else {
     printf("%sOK %s\n", msg_prefix, descr);
 }

 return error_counter;

}


//===========================================================================
int libcusmm_benchmark_transpose(libcusmm_benchmark_t* handle,
                                 int mat_m, int mat_n,
                                 TransposeLauncher* launcher, char** kernel_descr){

 if(mat_m > handle->max_m || mat_n > handle->max_n){
     printf("libcusmm_benchmark_transpose: got handle with too few resources\n");
     exit(1);
 }
 if(handle->mode == tune){
     printf("Tune mode not supported for benchmarking of transpose"); 
     exit(1); 
 }

 int errors;
 errors += libcusmm_benchmark_transpose_(handle->n_stack_trs_a, handle->stack_trs_a, handle->d_stack_trs_a,
                                         handle->mat_a, handle->mat_trs_a, handle->d_mat_a,
                                         handle->n_a, mat_m, mat_n,
                                         handle->t_start, handle->t_stop, kernel_descr, launcher);
 return errors;

}
