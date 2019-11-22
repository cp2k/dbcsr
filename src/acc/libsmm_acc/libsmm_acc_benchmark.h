/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#ifndef LIBSMM_ACC_BENCHMARK_H
#define LIBSMM_ACC_BENCHMARK_H

#ifdef __CUDA
#include "../cuda/acc_cuda.h"
#else
#include "../hip/acc_hip.h"
#endif

#define MAX_BLOCK_DIM 80

typedef int (*KernelLauncher)(const int *param_stack, int stack_size, ACC_DRV(stream) stream,
                              int m_max, int n_max, int k_max,
                              const double *a_data, const double *b_data, double *c_data);

typedef int (*TransposeLauncher)(const int *param_stack, int offset, int nblks,
                                 double *buffer, int m, int n, ACC_DRV(stream) stream);

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
    ACC_DRV(event) t_start, t_stop;
} libsmm_acc_benchmark_t;

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

void libsmm_acc_benchmark_init(libsmm_acc_benchmark_t** handle, benchmark_mode mode,
                               int max_m, int max_n, int max_k);

void libsmm_acc_benchmark_finalize(libsmm_acc_benchmark_t*);

int libsmm_acc_benchmark(libsmm_acc_benchmark_t* handle,
                         int mat_m, int mat_n, int mat_k, int nkernel,
                         KernelLauncher* launchers, char** kernel_descr);
int libsmm_acc_benchmark_transpose(libsmm_acc_benchmark_t* handle, int mat_m, int mat_n,
                                   TransposeLauncher* launcher, char** kernel_descr);
int libsmm_acc_benchmark_transpose_(int n_stack, int* stack, int* d_stack,
                                    double* mat, double* mat_trs, double* d_mat,
                                    int n, int mat_m, int mat_n,
                                    ACC_DRV(event) start, ACC_DRV(event) stop, char** kernel_descr,
                                    TransposeLauncher* launcher);

#endif // LIBSMM_ACC_BENCHMARK_H
