/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2018  CP2K developers group                         *
 *****************************************************************************/
#ifndef LIBCUSMM_BENCHMARK_H
#define LIBCUSMM_BENCHMARK_H

#include <cuda.h>

#define MAX_BLOCK_DIM 80

typedef int (*KernelLauncher)(int *param_stack, int stack_size, CUstream stream,
                              int m_max, int n_max, int k_max,
                              double *a_data, double *b_data, double *c_data);

typedef int (*TransposeLauncher)(int *param_stack, int offset, int nblks, 
                                 double *buffer, int m, int n, CUstream stream);

enum benchmark_mode {test, tune, timing};

typedef struct {
    benchmark_mode mode;
    // max block-sizes to expect
    int max_m, max_n, max_k;
    // number of blocks to allocate in each panel
    int n_a, n_b, n_c;
    // length of stack (multiplication, transpose a, transpose b) 
    int n_stack, n_stack_trs_a, n_stack_trs_b;
    // host-buffers
    double *mat_a, *mat_b, *mat_c;
    double *mat_trs_a, *mat_trs_b;
    int    *stack, *stack_trs_a, *stack_trs_b;
    // device-buffers
    double *d_mat_a, *d_mat_b, *d_mat_c;
    int    *d_stack, *d_stack_trs_a, *d_stack_trs_b; 
    // events for measuring the runtime
    CUevent t_start, t_stop;
} libcusmm_benchmark_t;

void matInit(double* mat, int mat_n, int x, int y, int seed);

void stackInit(int *stack, int n_stack, int n_c, double* mat_c,
               int n_a, double * mat_a, int n_b, double* mat_b,
               int mat_m, int mat_n, int mat_k);
void stackInitTransp(int *stack, int n_stack, int mat_m, int mat_n);

void stackCalc(int* stack, int n_stack, double* mat_c, double *mat_a, double* mat_b,
               int mat_m, int mat_n, int mat_k);
void stackTransp(int* stack, int n_stack, double *mat_a, double* mat_atrs,
                 int mat_m, int mat_n);

double checkSum(double* mat_c, int n_c, int mat_m, int mat_n);
double checkSumTransp(double* mat, int n, int n_stack, int mat_m, int mat_n);

void libcusmm_benchmark_init(libcusmm_benchmark_t** handle, benchmark_mode mode,
                             int max_m, int max_n, int max_k);

void libcusmm_benchmark_finalize(libcusmm_benchmark_t*);

int libcusmm_benchmark(libcusmm_benchmark_t* handle,
              int mat_m, int mat_n, int mat_k, int nkernel,
              KernelLauncher* launchers, char** kernel_descr);
int libcusmm_benchmark_transpose(libcusmm_benchmark_t* handle, int mat_m, int mat_n, 
                                 TransposeLauncher* launcher, char** kernel_descr);
int libcusmm_benchmark_transpose_(int n_stack, int* stack, int* d_stack,
                                  double* mat, double* mat_trs, double* d_mat,
                                  int n, int mat_m, int mat_n,
                                  CUevent start, CUevent stop, char** kernel_descr,
                                  TransposeLauncher* launcher);
#endif
//EOF
