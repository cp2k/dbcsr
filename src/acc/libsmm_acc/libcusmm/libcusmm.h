/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2018  CP2K developers group                         *
 *****************************************************************************/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <unordered_map>

#define NVRTC_SAFE_CALL(name, x)                                  \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      printf("\nerror: %s failed with error %s",                  \
             name, nvrtcGetErrorString(result));                  \
      exit(1);                                                    \
    }                                                             \
  } while(0)
#define CUDA_SAFE_CALL(name, x)                                   \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      printf("\nerror: %s failed with error %s",                  \
             name, msg);                                          \
      exit(1);                                                    \
    }                                                             \
  } while(0)

enum libcusmm_algo {
    largeDB1 = 1,
    largeDB2 = 2,
    medium = 3,
    small = 4,
    tiny = 5
};

//===========================================================================
// initialize matrix
static void matInit(double* mat, int mat_n, int x, int y, int seed){

 double *m = mat;

 for(int n=0; n<mat_n; n++){
   for(int j=0; j<y; j++) {
     for(int i=0; i<x; i++, m++) {
     *m = (double) j*x + i + n + seed;
     //printf("matrix [%d, %d]=%g\n", i, j, *m);
     }
   }
 }

}


//===========================================================================
// initialize the task list ("stack" in CP2K lingo)
// for each of the result matrices we have a random number
static void stackInit(int *stack, int n_stack, int n_c, double* mat_c,
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
static void stackCalc(int* stack, int n_stack, double* mat_c, double *mat_a, double* mat_b,
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


//         // initialize with non-transpose matrix
//         int b_ind = n * mat_k + k;
//         res += mat_a[a_base + a_ind] * mat_b[b_base + b_ind];

          // initialize with transpose matrix
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
static double checkSum(double* mat_c, int n_c, int mat_m, int mat_n){
   double res = 0;
   for(int i=0; i<n_c * mat_m * mat_n; i++){
     res += mat_c[i];
   }
   return res;
}


static std::unordered_map<int, CUfunction> kernel_handles;

int libcusmm_process_d(int *param_stack, int stack_size,
    CUstream stream, int m, int n, int k,
    double * a_data, double * b_data, double * c_data);

static std::unordered_map<int, CUfunction> transpose_handles;

int libcusmm_transpose_d(int *trs_stack, int offset, int nblks, double *buffer,
                         int m, int n, cudaStream_t * stream);

//EOF
