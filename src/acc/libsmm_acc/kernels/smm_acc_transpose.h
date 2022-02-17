/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/

/*****************************************************************************
 *  Authors: Peter Messmer <pmessmer@nvidia.com>,                            *
 *           Nikolay Markovskiy <nmarkovskiy@nvidia.com>                     *
 *****************************************************************************/

#include "smm_acc_common.h"

/*
 * Execution configuration:
 * gridDim.x = number of matrix blocks in this batched matrix transpose
 *           = length of the batched transpose stack
 * blockIdx.x = {0, ..., gridDim.x-1}
 * blockDim.x = 128 or the smallest multiple of warp_size larger or equal to m*n
 * threadIdx.x = {0, ..., blockDim.x-1}

 * Execute batched matrix transpose in place

 * Template parameters
 * --- m, n: pair of integers characterising the dimensions of the matrix to transpose

 * Function arguments
 * --- trs_stack: transpose stack (pointer to global memory):
 *     array of stack entries (indices), indicating where each matrix to transpose starts in the "mat" array
 * --- mat (pointer to global memory):
 *     arrays containing the values of the matrix to be transposed
 *     mat is column major: m = number of rows, n = number of columns

 * Algorithm specificities:
 * - the temporary buffer (of size m * n * 8 bytes) in which matrix elements are stored has to fit entirely into shared memory. Therefore, this kernel cannot be run for matrix sizes such that m * n * 8 bytes > available shared memory per block.
 */

template<int m, int n> __global__ void transpose_d(int* trs_stack, double* mat) {
  __shared__ double buf[m * n];

  /* Get the offset in the transpose-stack that this block ID should handle */
  int offset = trs_stack[blockIdx.x];

  /* Loop over m*n matrix elements */
  for (int i = threadIdx.x; i < m * n; i += blockDim.x) {
    /* Load matrix elements into a temporary buffer */
    buf[i] = mat[offset + i];
  }
  syncthreads();

  /* Loop over elements of the matrix to be overwritten */
  for (int i = threadIdx.x; i < m * n; i += blockDim.x) {
    /* Compute old row and column index of matrix element */
    int r_out = i % n;
    int c_out = i / n;
    /* Compute the corresponding old 1D index of matrix element */
    int idx = r_out * m + c_out;
    /* Overwrite the matrix element */
    mat[offset + i] = buf[idx];
  }
}
