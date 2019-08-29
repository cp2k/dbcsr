/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "libsmm_acc_benchmark.h"
#include "parameters.h"
#include "parameters_utils.h"


//===========================================================================
// Allocate memory and accelerator events
void libsmm_acc_benchmark_init(libsmm_acc_benchmark_t** handle, benchmark_mode mode,
                               int max_m, int max_n, int max_k){

    libsmm_acc_benchmark_t* h = (libsmm_acc_benchmark_t*) malloc(sizeof(libsmm_acc_benchmark_t));
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

    ACC_API_CALL(Malloc, (&h->d_mat_a, h->n_a * max_m * max_k * sizeof(double)));
    ACC_API_CALL(Malloc, (&h->d_mat_b, h->n_b * max_k * max_n * sizeof(double)));
    ACC_API_CALL(Malloc, (&h->d_mat_c, h->n_c * max_m * max_n * sizeof(double)));
    ACC_API_CALL(Malloc, (&h->d_stack, h->n_stack * 3 * sizeof(int)));
    ACC_API_CALL(Malloc, (&h->d_stack_trs_a, h->n_stack_trs_a * sizeof(int)));
    ACC_API_CALL(Malloc, (&h->d_stack_trs_b, h->n_stack_trs_b * sizeof(int)));

    ACC_DRV_CALL(EventCreate, (&h->t_start, ACC_DRV(EventDefault)));
    ACC_DRV_CALL(EventCreate, (&h->t_stop, ACC_DRV(EventDefault)));

}


//===========================================================================
// Free memory and accelerator events
void libsmm_acc_benchmark_finalize(libsmm_acc_benchmark_t* handle){
    ACC_DRV_CALL(EventDestroy, (handle->t_start));
    ACC_DRV_CALL(EventDestroy, (handle->t_stop));
    ACC_API_CALL(Free, (handle->d_stack_trs_b));
    ACC_API_CALL(Free, (handle->d_stack_trs_a));
    ACC_API_CALL(Free, (handle->d_stack));
    ACC_API_CALL(Free, (handle->d_mat_c));
    ACC_API_CALL(Free, (handle->d_mat_b));
    ACC_API_CALL(Free, (handle->d_mat_a));
    free(handle->stack_trs_b);
    free(handle->stack_trs_a);
    free(handle->stack);
    free(handle->mat_c);
    free(handle->mat_trs_b);
    free(handle->mat_b);
    free(handle->mat_trs_a);
    free(handle->mat_a);
    free(handle);
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
// initialize the task list ("stack" in DBCSR lingo)
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
// initialize the task list ("stack" in DBCSR lingo)
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
//Removes special symbols so that the output is useful for awk and gnuplot.
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
int libsmm_acc_benchmark(libsmm_acc_benchmark_t* h,
                         int mat_m, int mat_n, int mat_k,
                         int nkernels, KernelLauncher* launchers, char ** kernel_descr){

 if(mat_m > h->max_m || mat_n > h->max_n || mat_k > h->max_k){
     printf("libsmm_acc_benchmark: got handle with too few resources\n");
     exit(1);
 }
 std::vector<Triplet> blocksizes;
 get_libsmm_acc_triplets(blocksizes, ht);
 auto it = std::find(std::begin(blocksizes), std::end(blocksizes), Triplet({ mat_m, mat_n, mat_k }));
 if(it == std::end(blocksizes) && h->mode != tune){
     printf("Triplet %i x %i x %i is not defined in libsmm_acc\n", mat_m, mat_n, mat_k);
     exit(1);
 }


 int n_iter, n_warm;
 switch(h->mode){
   case tune:
   case timing: // for larger matrices few iteration give enough statistics
     n_iter = std::max(3, 12500/(mat_m * mat_n * mat_k));
     n_warm = std::min(3, n_iter);
     break;
   case test:
     n_iter = 1;
     n_warm = 1;
     break;
    }

 ACC_DRV(stream) stream;
 ACC_DRV_CALL(StreamCreate, (&stream, ACC_DRV(StreamDefault)));

 int error_counter = 0;
 int best_kernel = -1;
 double best_gflops = 0.0;
 double sumCPU, sumGPU;
 float t_duration;
 char descr[1000], msg_prefix[100]="";

 memset(h->mat_c, 0, h->n_c * mat_m * mat_n * sizeof(double));
 matInit(h->mat_a, h->n_a, mat_m, mat_k, 42);
 matInit(h->mat_b, h->n_b, mat_k, mat_n, 24);

 if(h->mode == tune)
     printf("Initializing ...\n");
 stackInit(h->stack, h->n_stack, h->n_c, h->mat_c, h->n_a, h->mat_a, h->n_b, h->mat_b, mat_m, mat_n, mat_k);

 // Actually, we would have to calculate the stack n_iter times.
 // We cheat by simply scaling the results of a single stack calculation.
 stackCalc(h->stack, h->n_stack, h->mat_c, h->mat_a, h->mat_b, mat_m, mat_n, mat_k);
 for(int i=0 ; i < h->n_c*mat_m*mat_n ; i++)
     h->mat_c[i] *= n_iter;

 sumCPU =  checkSum(h->mat_c, h->n_c, mat_m, mat_n);

 ACC_API_CALL(Memcpy, (h->d_mat_a, h->mat_a, h->n_a * mat_m * mat_k * sizeof(double), ACC(MemcpyHostToDevice)));
 ACC_API_CALL(Memcpy, (h->d_mat_b, h->mat_b, h->n_b * mat_k * mat_n * sizeof(double), ACC(MemcpyHostToDevice)));
 ACC_API_CALL(Memcpy, (h->d_stack, h->stack, h->n_stack * 3 * sizeof(int), ACC(MemcpyHostToDevice)));
 // d_mat_c gets zeroed after warmup run

 for(int ikern=0; ikern < nkernels; ikern++){

    // Warmup run (more often if n_iter is small)
    for(int i=0; i<n_warm; i++)
        launchers[ikern](h->d_stack, h->n_stack, stream, mat_m, mat_n, mat_k, h->d_mat_a, h->d_mat_b, h->d_mat_c);
    ACC_API_CALL(Memset, (h->d_mat_c, 0, h->n_c * mat_m * mat_n * sizeof(double)));

    ACC_DRV_CALL(EventRecord, (h->t_start, stream));

    for(int i=0; i<n_iter; i++)
        launchers[ikern](h->d_stack, h->n_stack, stream, mat_m, mat_n, mat_k, h->d_mat_a, h->d_mat_b, h->d_mat_c);

    ACC_DRV_CALL(EventRecord, (h->t_stop, stream));
    ACC_DRV_CALL(EventSynchronize, (h->t_stop));
    ACC_DRV_CALL(EventElapsedTime, (&t_duration, h->t_start, h->t_stop));

    ACC_API_CALL(Memcpy, (h->mat_c, h->d_mat_c, h->n_c * mat_m * mat_n * sizeof(double), ACC(MemcpyDeviceToHost)));

    clean_string(kernel_descr[ikern], descr);

    if(h->mode == tune)
        sprintf(msg_prefix, "params %d / %d\n",ikern+1, nkernels);

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
    } else {
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
 }

 return(error_counter);
}


//===========================================================================
int libsmm_acc_benchmark_transpose_(int n_stack, int* stack, int* d_stack,
                                    double* mat, double* mat_trs, double* d_mat,
                                    int n, int mat_m, int mat_n,
                                    ACC_DRV(event) start, ACC_DRV(event) stop, char** kernel_descr,
                                    TransposeLauncher* launcher){
 if(mat_m > MAX_BLOCK_DIM || mat_n > MAX_BLOCK_DIM){
     printf("Cannot transpose matrices with dimensions above %i, got (%i x %i)\n",
            MAX_BLOCK_DIM, mat_m, mat_n);
     exit(1);
 }

 ACC_DRV(stream) stream;
 ACC_DRV_CALL(StreamCreate, (&stream, ACC_DRV(StreamDefault)));

 int offset = 0;
 int n_warm = 0;
 int n_iter = 1;
 int error_counter = 0;
 double sumCPU, sumGPU;
 float t_duration;
 char descr[1000], msg_prefix[100]="";

 // Matrix and stack initialization
 matInit(mat, n, mat_m, mat_n, 42);
 memset(mat_trs, 0, n * mat_m * mat_n * sizeof(double));
 stackInitTransp(stack, n_stack, mat_m, mat_n);

 // Reference result on CPU
 stackTransp(stack, n_stack, mat, mat_trs, mat_m, mat_n);
 sumCPU = checkSumTransp(mat_trs, n, n_stack, mat_m, mat_n);

 // Compute on GPU
 ACC_API_CALL(Memcpy, (d_mat, mat, n * mat_m * mat_n * sizeof(double), ACC(MemcpyHostToDevice)));
 ACC_API_CALL(Memcpy, (d_stack, stack, n_stack * sizeof(int), ACC(MemcpyHostToDevice)));

 // Warmup run
 for(int i=0; i<n_warm; i++)
   launcher[0](d_stack, offset, n_stack, d_mat, mat_m, mat_n, stream);

 // Real runs
 ACC_DRV_CALL(EventRecord, (start, stream));

 for(int i=0; i<n_iter; i++)
   launcher[0](d_stack, offset, n_stack, d_mat, mat_m, mat_n, stream);

 ACC_DRV_CALL(EventRecord, (stop, stream));
 ACC_DRV_CALL(EventSynchronize, (stop));
 ACC_DRV_CALL(EventElapsedTime, (&t_duration, start, stop));

 // Check for errors and compare libsmm_acc result on GPU to reference
 ACC_API_CALL(Memcpy, (mat_trs, d_mat, n * mat_m * mat_n * sizeof(double), ACC(MemcpyDeviceToHost)));
 clean_string(kernel_descr[0], descr);

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
int libsmm_acc_benchmark_transpose(libsmm_acc_benchmark_t* handle,
                                   int mat_m, int mat_n,
                                   TransposeLauncher* launcher, char** kernel_descr){

 if(mat_m > handle->max_m || mat_n > handle->max_n){
     printf("libsmm_acc_benchmark_transpose: got handle with too few resources\n");
     exit(1);
 }
 if(handle->mode == tune){
     printf("Tune mode not supported for benchmarking of transpose");
     exit(1);
 }

 int errors = 0;
 errors += libsmm_acc_benchmark_transpose_(handle->n_stack_trs_a, handle->stack_trs_a, handle->d_stack_trs_a,
                                       handle->mat_a, handle->mat_trs_a, handle->d_mat_a,
                                       handle->n_a, mat_m, mat_n,
                                       handle->t_start, handle->t_stop, kernel_descr, launcher);
 return errors;

}
