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
 *****************************************************************************/

#include "smm_acc_common.h"

template < int m, int n>
__global__ void transpose_d(int *trs_stack, double* mat){
 __shared__ double buf[m*n];
 int offset = trs_stack[blockIdx.x];
 for(int i=threadIdx.x; i < m*n; i+=blockDim.x){
     buf[i] = mat[offset + i];
 }
 syncthreads();

 for(int i=threadIdx.x; i < m*n; i+=blockDim.x){
     int r_out = i % n;
     int c_out = i / n;
     int idx = r_out * m + c_out;
     mat[offset + i] = buf[idx];
 }

}
