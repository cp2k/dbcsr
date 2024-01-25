/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* Copyright (C) 2022 Advanced Micro Devices, Inc. - All rights reserved                          */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/

/*****************************************************************************
 *  Authors: Gina Sitaraman <gina.sitaraman@amd.com>                         *
 *****************************************************************************/

/*
 * Execution configuration:
 * gridDim.x = number of matrix blocks in this batched norms calculation
 *           = length of the batched norms calculation stack
 * blockIdx.x = {0, ..., gridDim.x-1}
 * blockDim.x = warp size (for now, assuming warp size is going to be 64 or 32)
 * threadIdx.x = {0, ..., blockDim.x-1}

 * Execute batched norms calculation

 * Function arguments
 * --- norms: (pointer to global memory):
 *     output array of norms, one per matrix in the stack
 * --- offsets: (pointer to global memory):
 *     array of offsets, indicating where each block starts in the "mat" buffer
 * --- nelems: (pointer to global memory):
 *     array of integers, indicating the number of elements in each matrix/block
 * --- mat (pointer to global memory):
 *     arrays containing the matrices for which norms are calculated

 * Algorithm specificities:
 * --- warp level primitives are used to reduce within a warp/wavefront, and
 *     shared memory is used if more than one warp/wavefront is detected
 */

#if defined(__CUDA)
#  include "../cuda/acc_cuda.h"
#elif defined(__HIP)
#  include "../hip/acc_hip.h"
#endif
#include "acc_utils.h"

template<int warpsz, int blocksz>
__global__ void calculate_norms_d(
  float* __restrict__ norms, const int* __restrict__ offsets, const int* __restrict__ nelems, const double* __restrict__ mat) {
  __shared__ double buf[(blocksz + warpsz - 1) / warpsz];
  double d, sum = 0.0;

  /* Get the offset in the stack that this thread block should handle */
  int blkoffset = offsets[blockIdx.x];

  /* Get the number of elements in this matrix */
  int nelem = nelems[blockIdx.x];

  /* Loop over nelem matrix elements for this block */
  for (int i = threadIdx.x; i < nelem; i += blockDim.x) {
    /* Load matrix elements, reduce in register */
    d = mat[blkoffset + i];
    sum += d * d;
  }
  __syncthreads();

  /* reduce in warp to one value using warp level primitives */
#if defined(__CUDA)
  unsigned mask = 0xffffffff;
  for (int offset = warpsz / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(mask, sum, offset);
  }
#elif defined(__HIP)
  for (int offset = warpsz / 2; offset > 0; offset /= 2) {
    sum += __shfl_down(sum, offset);
  }
#endif

  /* reduce between warps if needed */
  if (blocksz > warpsz) {
    if (threadIdx.x % warpsz == 0) {
      int warpid = threadIdx.x / warpsz;
      buf[warpid] = sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int i = 1; i < blocksz / warpsz; ++i) {
        sum += buf[i];
      }
    }
  }
  if (threadIdx.x == 0) {
    /* write out this stack's dot product */
    norms[blockIdx.x] = sum;
  }
}

extern "C" int c_calculate_norms(double* mat, int nblks, int* offsets, int* nelems, float* norms, void* stream_ptr) {
  int warp_size = acc_get_gpu_warp_size();

  dim3 grid(nblks);
  dim3 block(warp_size);

  ACC_DRV(stream) stream = *((ACC_DRV(stream)*)stream_ptr);
  /* block size may be a multiple of warp_size as well */
  if (warp_size == 64) {
    calculate_norms_d<64, 64><<<grid, block, 0, stream>>>(norms, offsets, nelems, mat);
  }
  else if (warp_size == 32) {
    calculate_norms_d<32, 32><<<grid, block, 0, stream>>>(norms, offsets, nelems, mat);
  }
  else {
    fprintf(stderr, "Found warp size other than 64 or 32, aborting..\n");
    return -1;
  }
  return 0;
}
