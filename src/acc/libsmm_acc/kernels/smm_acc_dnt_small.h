/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

/*****************************************************************************
 *  Authors: Peter Messmer <pmessmer@nvidia.com>,                            *
 *           Nikolay Markovskiy <nmarkovskiy@nvidia.com>                     *
 *  Simplified, cleanup & refacture: Based on 'cusmm_dnt_small.h'            *
 *           Andreas Gloess <andreas.gloess@chem.uzh.ch>                     *
 *                                                                           *
 *****************************************************************************/

#include "smm_acc_common.h"

/*
 * Execution configuration:
 * gridDim.x = (stack_size + (grouping-1))/grouping
 * blockIdx.x = {0, ..., gridDim.x-1}
 * blockDim.x = threads
 * threadIdx.x = {0, ..., threads-1}

 * Execute sparse matrix-matrix multiplication C += A*B
 * according to a stack parameter list, decomposed into mutliple block
 * multiplications c_block += a_block * b_block
 *     c_block, dimension (m x n)
 *     a_block, dimension (m x k)
 *     b_block, dimension (k x n)

 * Template parameters
 * --- m, n, k: triplet of integers characterising the block multiplication dimensions
 * --- M, N: dimensions of the tile T (submatrix of c_block) to compute with one thread
 * --- threads: number of CUDA threads (in HIP linguo, "work items") this kernel is run with
 * --- grouping: number of stack parameter entries to process per thread block
 * --- minblocks: the desired minimum number of resident blocks (in HIP linguo, "workgroup") per multiprocessor (in HIP linguo, "compute units") (used in __launch_bounds__)

 * Function arguments
 * --- param_stack: parameter stack array (pointers to global memory):
 *     array of stack entries (index triplets), indicating which elements of
 *     a_data, b_data to multiply and to which element of c_data to add them to
 * --- stack_size: number of entries (3 integer triplets) in param_stack,
 *     corresponds to the number of block-matrix multiplications to run in total
 * --- a_data, b_data, c_data (pointers to global memory):
 *     arrays containing the values of matrices A, B, C

 * Algorithm specificities:
 * - each thread computes a tile T (of size tile_m * tile_n) of c_block, in order to increase
 *   ILP per thread and given the large number of registers allowed per thread.
 * - the result tiles do not need to be shared between threads, it is therefore stored in registers
 * - T get decompressed to shared memory buffer before being written back to global memory via an atomic add
 */
template <int m, int n, int k, int M, int N, int threads, int grouping, int minblocks>
__global__ void
__launch_bounds__(threads, minblocks)
smm_acc_dnt_small(const int* __restrict__ param_stack, int stack_size,
     const double* __restrict__ a_data, const double* __restrict__ b_data, double* c_data){

  /* Total number of elements in block matrices */
  const int mn = m * n; /* c_block */
  const int mk = m * k; /* a_block */
  const int kn = n * k; /* b_block */

  /* Block and thread index */
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;

  /* Number of columns and rows used to divide c_block into tiles */
  const int cmax = (n % N == 0)?  n / N: n / N + 1;
  const int rmax = (m % M == 0)?  m / M: m / M + 1;

  // buff_l and buff_r can overlap in the multiplication step
  // (still better than 'if' in inner loop, see ^ref1)
  const int buf_tmp = (mk + k * N * cmax < M * rmax * k + 1)? M * rmax * k + 1: mk + k * N * cmax;
  const int buf_sz = (buf_tmp < mn)? mn: buf_tmp;

  /* Column and row starting index of the tile T this thread will compute
   * Each thread computes all elements of one tile T */
  const int c = tidx / rmax;
  const int r = tidx - c * rmax;

  /* Number of parameters per stack entry in parameter stack */
  const int  npar = 3;

  /* If multiple warps (in HIP linguo, "wavefronts") are running a single block multiplication,
   * synchronization is needed */
  const bool need_sync = (mn > warpSize || mk > warpSize || kn > warpSize || threads > warpSize);

  /* psp: parameter stack position:
   *      index in the parameter stack of the first parameter stack entry to be processed by this thread
   * nrun: number of runs: number of stack entries to process in this thread */
  int    nrun, psp;

  /* Partial sum of the tile T of block_c that this thread will compute */
  double myc[M * N];

  /* Arrays in shared memory
   *(common to all threads in a thread block, but different from thread-block to thread-block):
   * param_stack_s: shared memory buffer containing the stack entries this thread should process
   *                number of stack entries in param_stack_s = grouping,
   *                number of integers per stack entry: 3 */
  __shared__ int    param_stack_s[npar * grouping];

  /* buff: shared memory buffer containing a_block (block submatrix of A) and b_block to multiply
   * and to which c-results are decompressed */
  __shared__ double buff[buf_sz];
  double *buff_l = buff; /* pointer to the beginning of a_block in buffer */
  double *buff_r = &buff[mk]; /* pointer to the beginning of b_block in buffer */

  /* Set the number of runs (i.e. how many stack entries to process in this thread)
   * If the current block is the last block set the number of stack entries to process in
   * this thread to the remainder of stack_size / grouping */
  nrun = grouping;
  if (((bidx + 1) * grouping) > stack_size) nrun = stack_size % grouping;

  /* Set the partial sum (tile T) to zero */
#pragma unroll
  for (int i = 0; i < M * N; i++)
    myc[i] = 0.0;

  /* Load and pack stack data for current block from global memory into shared memory
   * Get parameter stack entries from index "psp" to "psp + (nrun-1)*npar + 2"
   * Each triplet indicates the beginning of a submatrix to multiply */
  psp = bidx * npar * grouping;
#pragma unroll
  for (int i = tidx; i < nrun; i += threads){
    param_stack_s[i * npar    ] = __ldg(&param_stack[psp + i * npar    ]) - 1; /* value = index in a_data */
    param_stack_s[i * npar + 1] = __ldg(&param_stack[psp + i * npar + 1]) - 1; /* value = index in b_data */
    param_stack_s[i * npar + 2] = __ldg(&param_stack[psp + i * npar + 2]) - 1; /* value = index in c_data */
  }

  /* Wait until all the data has been loaded */
  if (need_sync) syncthreads ();

  /* In each run, we process one stack entry from param_stack_s */
  for (int run = 0; run < nrun; run++){

    /* Index in shared memory buffers to read from */
    psp = run * npar;

    /* Index in a_data, b_data and c_data arrays
     * indicating where to fetch resp. write back matrix elements for this run
     * srcA, B, C corresponding to the strting indices of block submatrices to multiply */
    int srcA = param_stack_s[psp];
    int srcB = param_stack_s[psp + 1];
    int srcC = param_stack_s[psp + 2];

    /* Load block matrices a_block and b_block for current block and stack into smem */
    if (m == n){
#pragma unroll
      for (int i = tidx; i < mk; i += threads){
         buff_l[i] = __ldg(&a_data[srcA + i]);
         buff_r[i] = __ldg(&b_data[srcB + i]);
      }
    } else {
#pragma unroll
      for (int i = tidx; i < mk; i += threads){
         buff_l[i] = __ldg(&a_data[srcA + i]);
      }
#pragma unroll
      for (int i = tidx; i < kn; i += threads){
         buff_r[i] = __ldg(&b_data[srcB + i]);
      }
    }

    /* Wait until all the data has been loaded */
    if (need_sync) syncthreads ();

    /* Do actual multiplication. */
    if (c < cmax && r < rmax){
      for (int l = 0; l < k; l++){
        /* Loop over all elements c_ij of tile T */
#pragma unroll
        for (int i = 0; i < N; i++){
#pragma unroll
          for (int j = 0; j < M; j++){
               /* Compute c_ij = sum_k (a_ik * b_kj) in shared memory */
               myc[M * i + j] += buff_l[l * m + M * r + j] * buff_r[l * n + N * c + i];
          }
        }
      }
    }

    /* last loop or C_idx for next stack entry is different */
    if ((run == (nrun - 1)) || (srcC != param_stack_s[psp + 2 + npar])){
      if ((M > 1) || (N > 1)){
        if (need_sync) syncthreads ();

        /* Decompress results to buffer and set tile elements back to 0 */
        if (c < cmax && r < rmax){
#pragma unroll
          for (int i = 0; i < N; i++){
#pragma unroll
            for (int j = 0; j < M; j++){
              if (M * r + j < m && N * c + i < n){
                 buff[(N * c + i) * m + M * r + j] = myc[M * i + j];
                 myc[M * i + j] = 0.0;
              }
            }
          }
        }

        if (need_sync) syncthreads ();

        /* Add results from shared memory buffer to global C block. */
#pragma unroll
        for (int i = tidx; i < mn; i += threads)
          atomicAdd (&c_data[srcC + i], buff[i]);
      } else {
        /* Add results from registers to global C block. */
#pragma unroll
        for (int i = tidx; i < mn; i += threads)
          atomicAdd (&c_data[srcC + i], myc[0]);
        myc[0]= 0.0;
      }
    }

    if (need_sync) syncthreads ();
  }
}
