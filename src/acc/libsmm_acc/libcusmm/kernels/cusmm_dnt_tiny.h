/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2018  CP2K developers group                         *
 *****************************************************************************/

/*****************************************************************************
 *  Authors: Peter Messmer <pmessmer@nvidia.com>,                            *
 *           Nikolay Markovskiy <nmarkovskiy@nvidia.com>                     *
 *  Simplified, cleanup & refacture: Based on 'cusmm_dnt_tiny.h'             *
 *           Andreas Gloess <andreas.gloess@chem.uzh.ch>                     *
 *                                                                           *
 *****************************************************************************/

#include "cusmm_common.h"

/*
 * Execution configuration:
 * gridDim.x = (stack_size + (grouping-1))/grouping
 * blockIdx.x = {0, ..., gridDim.x-1}
 * blockDim.x = threads
 * threadIdx.x = {0, ..., threads-1}

 * Execute sparse matrix-matrix multiplication: C += A*B
 * according to stack parameter list,
 * decomposed into mutliple block multiplications c_block += a_block * b_block
 *     c_block, dimension (m x n)
 *     a_block, dimension (m x k)
 *     b_block, dimension (k x n)
 * (optimized for block matrices that fit into shared memory)

 * Template parameters
 * --- m, n, k: triplet of integers characterising the block multiplication dimensions
 * --- threads: number of CUDA threads this kernel is run with
 * --- grouping: number of stack parameter entries to process per thread block
 * --- minblocks: (used in __launch_bounds__)

 * Function arguments
 * --- param_stack: parameter stack array
 *     array of stayk entries (index triplets), indicating which elements of
 *     a_data, b_data to multiply and to which element of c_data to add them to
 * --- stack_size: number of entries (3 integer triplets) in param_stack
 *     corresponds to the number of block-matrix multiplications to run
 * --- a_data, b_data, c_data
 *     arrays containing the non-zero values of matrices A, B, C
 */
template <int m, int n, int k, int threads, int grouping, int minblocks>
__global__ void
__launch_bounds__(threads, minblocks)
cusmm_dnt_tiny(const int* __restrict__ param_stack, int stack_size,
     const double* __restrict__ a_data, const double* __restrict__ b_data, double* c_data){

  /* Total number of elements in block matrices ... */
  const int mn = m * n; /* c_block */
  const int mk = m * k; /* a_block */
  const int kn = n * k; /* b_block */

  /* Block and thread index */
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;

  /* Column and row index of the block matrix c_block this thread will compute
   * Each thread computes exactly one element of block_c */
  const int c = tidx / m;
  const int r = tidx - c * m;

  /* Number of parameters per stack entry in parameter stack */
  const int  npar      = 3;
  const int  warp_size = 32;

  /* If multiple warps are running a single block multiplication,
   * synchronization is needed */
  const bool need_sync = (mn > warp_size || mk > warp_size || kn > warp_size || threads > warp_size);

  /* psp: parameter stack position:
   *      index in the parameter stack of the first parameter stack entry to be processed by this thread
   * nrun: number of runs: number of stack entries to process in this thread */
  int    psp, nrun;

  /* Partial sum of the element of block_c that this thread will compute */
  double myc;

  /* Arrays in shared memory
   *(common to all threads in a thread block, but different from thread-block to thread-block):
   * param_stack_s: shared memory buffer containing the stack entries this thread should process
   *                number of stack entries in param_stack_s = grouping,
   *                number of integers per stack entry: 3 */
  __shared__ int    param_stack_s[npar * grouping];
  /* buff_a, buff_b: shared memory buffer containing a_block (block submatrix of A),
   * resp. b_block to multiply */
  __shared__ double buff_a[mk];
  __shared__ double buff_b[kn];

  /* Set the number of runs (i.e. how many stack entries to process in this thread) */
  nrun = grouping;
  if (((bidx + 1) * grouping) > stack_size) nrun = stack_size % grouping;

  /* Set the partial sum to zero */
  myc = 0.0;

  /* Load and pack stack data for current block from global memory into smem
   * Get parameter stack entries from index "psp" to "psp + (nrun-1)*npar + 2"
   * Each triplet indicates the beginning of a submatrix to multiply*/
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
  for (int run = 0; run < nrun; run++)
  {
    /* Index in shared memory buffers to read from */
    psp = npar * run;

    /* Index in a_data, b_data and c_data arrays
     * indicating where to fetch resp. write back matrix elements for this run
     * srcA, B, C corresponding to the strting indices of block submatrices to multiply */
    int srcA = param_stack_s[psp    ];
    int srcB = param_stack_s[psp + 1];
    int srcC = param_stack_s[psp + 2];

    /* Load block matrices a_block and b_block for current block and stack into smem
     * (no computation/load overlap!)
     * once an element s loaded into shared memory, it is available for all threads of the thread block to use */
    if (m == n) {
#pragma unroll
      for (int i = tidx; i < mk; i += threads){
        buff_a[i] = __ldg(&a_data[srcA + i]);
        buff_b[i] = __ldg(&b_data[srcB + i]);
      }
    } else {
#pragma unroll
      for (int i = tidx; i < mk; i += threads){
        buff_a[i] = __ldg(&a_data[srcA + i]);
      }
#pragma unroll
      for (int i = tidx; i < kn; i += threads){
        buff_b[i] = __ldg(&b_data[srcB + i]);
      }
    }

    /* Wait until all the data has been loaded */
    if (need_sync) syncthreads ();

    if (tidx < mn) {
      /* Compute c_ij = sum_k (a_ik * b_kj) in shared memory */
#pragma unroll
      for (int l = 0; l < k; l++) {
        myc += buff_a[l * m + r] * buff_b[l * n + c];
      }
      /* Store result in global memory */
      atomicAdd (&c_data[srcC + tidx], myc);
      myc = 0.0;
    }

    if (need_sync) syncthreads ();
  }

}
