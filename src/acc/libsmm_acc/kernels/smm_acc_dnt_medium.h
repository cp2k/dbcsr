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
 * - a_block and b_block are not copied directly from global memory to shared memory,
 *   but rather via registers
 */
template < int m,  int n,  int k, int M, int N, int threads, int grouping, int minblocks>
__global__ void
__launch_bounds__(threads, minblocks)
smm_acc_dnt_medium(const int* __restrict__ param_stack, int stack_size,
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
  const int r = tidx - rmax * c;

  /* Number of loads to perform by each thread in order to load matrices a_block, b_block into shared memory */
  const int load_unroll_factor_1 = mk / threads + 1;
  const int load_unroll_factor_2 = kn / threads + 1;

  const int n_mkloads = mk / (load_unroll_factor_1 * threads);
  const int n_knloads = kn / (load_unroll_factor_2 * threads);

  /* Remainder */
  const int mkloads_remained = (mk - n_mkloads * load_unroll_factor_1 * threads) / threads;
  const int knloads_remained = (kn - n_knloads * load_unroll_factor_2 * threads) / threads;

  /* ... */
  const int mkloads_tail = ((mkloads_remained + n_mkloads * load_unroll_factor_1) * threads == mk)? 0: 1;
  const int knloads_tail = ((knloads_remained + n_knloads * load_unroll_factor_2) * threads == kn)? 0: 1;

  /* ... */
  const int m_loads_to_finish = n_mkloads * (load_unroll_factor_1 * threads) + mkloads_remained * threads;
  const int n_loads_to_finish = n_knloads * (load_unroll_factor_2 * threads) + knloads_remained * threads;
  const int left_to_finish_1 = m_loads_to_finish + tidx;
  const int left_to_finish_2 = n_loads_to_finish + tidx;

  /* Number of parameters per stack entry in parameter stack */
  const int  npar = 3;

  /* If multiple warps (in HIP linguo, "wavefronts") are running a single block multiplication,
   * synchronization is needed */
  const bool need_sync = (mn > warpSize || mk > warpSize || kn > warpSize || threads > warpSize);

  /* Partial result of the tile T of c_block that this thread will compute */
  /* Registers */
  double myc[N * M];

  /* ... */
  double lba_r[load_unroll_factor_1 + mkloads_remained + mkloads_tail];
  double lbb_r[load_unroll_factor_2 + knloads_remained + knloads_tail];

  /* ... */
  const double* __restrict__ buff_l;
  const double* __restrict__ buff_r;

  /* psp: parameter stack position:
   *      index in the parameter stack of the first parameter stack entry to be processed by this thread
   * nrun: number of runs: number of stack entries to process in this thread */
  int  psp, nrun;
  bool is_loaded = false;

  /* Arrays in shared memory
   *(common to all threads in a thread block, but different from thread-block to thread-block):
   * param_stack_s: shared memory buffer containing the stack entries this thread should process
   *                number of stack entries in param_stack_s = grouping,
   *                number of integers per stack entry: 3 */
  __shared__ int    param_stack_s[npar * grouping];

  /* buff: shared memory buffer containing a_block (block submatrix of A) and b_block to multiply */
  __shared__ double buff[buf_sz];
  buff_l = buff; /* pointer to the beginning of a_block in buffer */
  buff_r = &(buff[mk]); /* pointer to the beginning of b_block in buffer */

  /* Set the number of runs (i.e. how many stack entries to process in this thread)
   * If the current block is the last block set the number of stack entries to process in
   * this thread to the remainder of stack_size / grouping */
  nrun = grouping;
  if (((bidx + 1) * grouping) > stack_size) nrun = stack_size % grouping;

  /* Set the partial sum (tile T) to zero */
  // WHY NO PRAGMA UNROLL ????
  for (int i = 0; i < M * N; i++)
    myc[i] = 0.0;

  /* Load and pack stack data for current block from global memory into smem
   * Get parameter stack entries from index "psp" to "psp + (nrun-1)*npar + 2"
   * Each triplet indicates the beginning of a submatrix to multiply */
  psp = bidx * npar * grouping;
#pragma unroll 3
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

    int srcA, srcB;

    if (!is_loaded){ /* If a_block, b_block are not loaded into registers yet */

      /* Index in a_data, b_data and c_data arrays
       * indicating where to fetch resp. write back matrix elements for this run
       * srcA, B, C corresponding to the starting indices (i.e. offsets) of block submatrices to multiply */
      srcA = param_stack_s[psp    ];
      srcB = param_stack_s[psp + 1];

    /* Copy a_block, b_block from global memory to registers */
    if (m == n){
#pragma unroll 3
      for (int i = tidx; i < n_mkloads * (load_unroll_factor_1 * threads); i += load_unroll_factor_1 * threads){
#pragma unroll
        for (int l = 0; l < load_unroll_factor_1; l++){
          lba_r[l] = __ldg(&a_data[srcA + i + l * threads]);
          lbb_r[l] = __ldg(&b_data[srcB + i + l * threads]);
        }
      }

#pragma unroll 3
      for (int i = n_mkloads * (load_unroll_factor_1 * threads) + tidx; i < m_loads_to_finish; i += mkloads_remained * threads){
#pragma unroll
        for (int l = 0; l < mkloads_remained; l++){
          lba_r[l] = __ldg(&a_data[srcA + i + l * threads]);
          lbb_r[l] = __ldg(&b_data[srcB + i + l * threads]);
        }
      }

      if (left_to_finish_1 < mk){
        lba_r[load_unroll_factor_1 + mkloads_remained] = __ldg(&a_data[srcA + left_to_finish_1]);
        lbb_r[load_unroll_factor_2 + knloads_remained] = __ldg(&b_data[srcB + left_to_finish_2]);
      }
    } else {
#pragma unroll 3
      for (int i = tidx; i < n_mkloads * (load_unroll_factor_1 * threads); i += load_unroll_factor_1 * threads){
#pragma unroll
        for (int l = 0; l < load_unroll_factor_1; l++){
          lba_r[l] = __ldg(&a_data[srcA + i + l * threads]);
        }
      }

#pragma unroll 3
      for (int i = tidx; i < n_knloads * (load_unroll_factor_2 * threads); i += load_unroll_factor_2 * threads){
#pragma unroll
        for (int l = 0; l < load_unroll_factor_2; l++){
          lbb_r[l] = __ldg(&b_data[srcB + i + l * threads]);
        }
      }

#pragma unroll 3
      for (int i = n_mkloads * (load_unroll_factor_1 * threads) + tidx; i < m_loads_to_finish; i += mkloads_remained * threads){
#pragma unroll
        for (int l = 0; l < mkloads_remained; l++){
          lba_r[l] = __ldg(&a_data[srcA + i + l * threads]);
        }
      }

#pragma unroll 3
      for (int i = n_knloads * (load_unroll_factor_2 * threads) + tidx; i < n_loads_to_finish; i += knloads_remained * threads){
#pragma unroll
        for (int l = 0; l < knloads_remained; l++){
          lbb_r[l] = __ldg(&b_data[srcB + i + l * threads]);
        }
      }

      if (left_to_finish_1 < mk) lba_r[load_unroll_factor_1 + mkloads_remained] = __ldg(&a_data[srcA + left_to_finish_1]);
      if (left_to_finish_2 < kn) lbb_r[load_unroll_factor_2 + knloads_remained] = __ldg(&b_data[srcB + left_to_finish_2]);
    }

      /* Wait until all the data has been loaded to registers */
      syncthreads ();
    } else {
       is_loaded = false;
    }

    /* Copy a_block, b_block from registers to shared memory */
    if (m == n){
#pragma unroll 3
      for (int i = n_mkloads * (load_unroll_factor_1 * threads) + tidx; i < m_loads_to_finish; i += mkloads_remained * threads){
#pragma unroll
        for (int l = 0; l < mkloads_remained; l++){
          buff[i + l * threads] = lba_r[l];
          buff[i + mk + l * threads] = lbb_r[l];
        }
      }

#pragma unroll 3
      for (int i = tidx; i < n_mkloads * (load_unroll_factor_1 * threads); i += load_unroll_factor_1 * threads){
#pragma unroll
        for (int l = 0; l < load_unroll_factor_1; l++){
          buff[i + l * threads] = lba_r[l];
          buff[i + mk + l * threads] = lbb_r[l];
        }
      }

      if (left_to_finish_1 < mk){
        buff[left_to_finish_1] = lba_r[load_unroll_factor_1 + mkloads_remained];
        buff[mk + left_to_finish_2] = lbb_r[load_unroll_factor_2 + knloads_remained];
      }
    } else {
#pragma unroll 3
      for (int i = n_mkloads * (load_unroll_factor_1 * threads) + tidx; i < m_loads_to_finish; i += mkloads_remained * threads){
#pragma unroll
        for (int l = 0; l < mkloads_remained; l++){
          buff[i + l * threads] = lba_r[l];
        }
      }

#pragma unroll 3
      for (int i = n_knloads * (load_unroll_factor_2 * threads) + tidx; i < n_loads_to_finish; i += knloads_remained * threads){
#pragma unroll
        for (int l = 0; l < knloads_remained; l++){
          buff[i + mk + l * threads] = lbb_r[l];
        }
      }

#pragma unroll 3
      for (int i = tidx; i < n_mkloads * (load_unroll_factor_1 * threads); i += load_unroll_factor_1 * threads){
#pragma unroll
        for (int l = 0; l < load_unroll_factor_1; l++){
          buff[i + l * threads] = lba_r[l];
        }
      }

#pragma unroll 3
      for (int i = tidx; i < n_knloads * (load_unroll_factor_2 * threads); i += load_unroll_factor_2 * threads){
#pragma unroll
        for (int l = 0; l < load_unroll_factor_2; l++){
          buff[i + mk + l * threads] = lbb_r[l];
        }
      }

      if (left_to_finish_1 < mk) buff[left_to_finish_1] = lba_r[load_unroll_factor_1 + mkloads_remained];
      if (left_to_finish_2 < kn) buff[mk + left_to_finish_2] = lbb_r[load_unroll_factor_2 + knloads_remained];
    }

    /* Wait until all the data has been loaded to shared memory */
    if (need_sync) syncthreads ();


    int next_run = run + 1;
    if (next_run >= nrun) is_loaded = true;

    if (!is_loaded || (run == 0 && nrun > 1)){
      /* Next parameter stack position */
      int next_psp = next_run * npar;

      /* Index in a_data, b_data and c_data arrays
       * indicating where to fetch resp. write back matrix elements for this run
       * srcA, B, C corresponding to the strting indices of block submatrices to multiply */
      srcA = param_stack_s[next_psp    ];
      srcB = param_stack_s[next_psp + 1];

      /* Buffering: copy the input data for the next run from global memory to registers */
      if (m == n){
#pragma unroll 3
        for (int i = tidx; i < n_mkloads * (load_unroll_factor_1 * threads); i += load_unroll_factor_1 * threads){
#pragma unroll
          for (int l = 0; l < load_unroll_factor_1; l++){
            lba_r[l] = __ldg(&a_data[srcA + i + l * threads]);
            lbb_r[l] = __ldg(&b_data[srcB + i + l * threads]);
          }
        }

#pragma unroll 3
        for (int i = n_mkloads * (load_unroll_factor_1 * threads) + tidx; i < m_loads_to_finish; i += mkloads_remained * threads){
#pragma unroll
          for (int l = 0; l < mkloads_remained; l++){
            lba_r[l] = __ldg(&a_data[srcA + i + l * threads]);
            lbb_r[l] = __ldg(&b_data[srcB + i +l * threads]);
          }
        }

        if (left_to_finish_1 < mk){
          lba_r[load_unroll_factor_1 + mkloads_remained] = __ldg(&a_data[srcA + left_to_finish_1]);
          lbb_r[load_unroll_factor_2 + knloads_remained] = __ldg(&b_data[srcB + left_to_finish_2]);
        }
      } else {
#pragma unroll 3
        for (int i = tidx; i < n_mkloads * (load_unroll_factor_1 * threads); i += load_unroll_factor_1 * threads){
#pragma unroll
          for (int l = 0; l < load_unroll_factor_1; l++){
            lba_r[l] = __ldg(&a_data[srcA + i + l * threads]);
          }
        }

#pragma unroll 3
        for (int i = tidx; i < n_knloads * (load_unroll_factor_2 * threads); i += load_unroll_factor_2 * threads){
#pragma unroll
          for (int l = 0; l < load_unroll_factor_2; l++){
            lbb_r[l] = __ldg(&b_data[srcB + i + l * threads]);
          }
        }

#pragma unroll 3
        for (int i = n_mkloads * (load_unroll_factor_1 * threads) + tidx; i < m_loads_to_finish; i += mkloads_remained * threads){
#pragma unroll
          for (int l = 0; l < mkloads_remained; l++){
            lba_r[l] = __ldg(&a_data[srcA + i + l * threads]);
          }
        }

#pragma unroll 3
        for (int i = n_knloads * (load_unroll_factor_2 * threads) + tidx; i < n_loads_to_finish; i += knloads_remained * threads){
#pragma unroll
          for (int l = 0; l < knloads_remained; l++){
            lbb_r[l] = __ldg(&b_data[srcB + i + l * threads]);
          }
        }

        if (left_to_finish_1 < mk) lba_r[load_unroll_factor_1 + mkloads_remained] = __ldg(&a_data[srcA + left_to_finish_1]);
        if (left_to_finish_2 < kn) lbb_r[load_unroll_factor_2 + knloads_remained] = __ldg(&b_data[srcB + left_to_finish_2]);
      }

      is_loaded = true;
    }

    /* Do actual multiplication. */
    if (c < cmax  && r < rmax){
      for (int l = 0; l < k; l++){
        /* Loop over all elements c_ij of tile T */
        for (int i = 0; i < M; i++){
          for (int j = 0; j < N; j++){
            /* Compute c_ij = sum_k (a_ik * b_kj) in shared memory */
            myc[N * i + j] += buff_l[l * m + M * r + i] * buff_r[l * n + N * c + j];
          }
        }
      }
    }


    if (run == nrun - 1 || param_stack_s[psp + 2] != param_stack_s[psp + 2 + npar]){

      /* Index in c_data indicating where to write back matrix elements for this run */
      int srcC = param_stack_s[psp + 2];

      if (M > 1 || N > 1){
        if (need_sync) syncthreads();
        /* Decompress results to buffer and set tile elements back to 0 */
        if (c < cmax  && r < rmax){
          for (int i = 0; i < M; i++){
            for (int j = 0; j < N; j++){
              if (M * r + i < m && N * c + j < n){
                buff[(N * c + j) * m + M * r + i] = myc[N * i + j];
                myc[N * i + j] = 0.0;
              }
            }
          }
        }
        if (need_sync) syncthreads();

        /* Add results from shared memory buffer to global C block. */
#pragma unroll
        for (int i = tidx; i < mn; i += threads){
          atomicAdd (&c_data[srcC + i], buff[i]);
        }
      } else {
        /* Add results from registers to global C block. */
#pragma unroll
        for (int i = tidx; i < mn; i += threads){
          atomicAdd (&c_data[srcC + i], myc[0]);
        }
        myc[0] = 0.0;
      }
    }
    if (need_sync) syncthreads ();
  }
}
