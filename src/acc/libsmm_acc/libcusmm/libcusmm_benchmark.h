/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2017  CP2K developers group                         *
 *****************************************************************************/
#include <cuda.h>

typedef int (*KernelLauncher)(int *param_stack, int stack_size, CUstream stream,
                              int m_max, int n_max, int k_max,
                              double *a_data, double *b_data, double *c_data);

typedef struct {
    bool tune_mode;
    // max block-sizes to expect
    int max_m, max_n, max_k;
    // number of blocks to allocate in each panel
    int n_a, n_b, n_c;
    // length of stack
    int n_stack;
    // host-buffers
    double* mat_a;
    double* mat_b;
    double* mat_c;
    int*    stack;
    // device-buffers
    double* d_mat_a;
    double* d_mat_b;
    double* d_mat_c;
    int*    d_stack;
    // events for measuring the runtime
    cudaEvent_t t_start, t_stop;
} libcusmm_benchmark_t;

void matInit(double* mat, int mat_n, int x, int y, int seed);

void stackInit(int *stack, int n_stack, int n_c, double* mat_c,
               int n_a, double * mat_a, int n_b, double* mat_b,
               int mat_m, int mat_n, int mat_k);

void stackCalc(int* stack, int n_stack, double* mat_c, double *mat_a, double* mat_b,
               int mat_m, int mat_n, int mat_k);

double checkSum(double* mat_c, int n_c, int mat_m, int mat_n);

void libcusmm_benchmark_init(libcusmm_benchmark_t** handle, bool tune_mode,
                             int max_m, int max_n, int max_k);

void libcusmm_benchmark_finalize(libcusmm_benchmark_t*);

int libcusmm_benchmark(libcusmm_benchmark_t* handle,
              int mat_m, int mat_n, int mat_k, int nkernel,
              KernelLauncher* launchers, char** kernel_descr);

//EOF
