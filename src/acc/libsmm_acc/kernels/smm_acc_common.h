/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#ifdef __HIP
#if not defined(__HIP_PLATFORM_NVCC__)
#include <hip/hip_runtime.h>
#endif
#endif

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/******************************************************************************
 * There is no native support for atomicAdd on doubles in Cuda 5.0. However   *
 * the following implementation is provided in the CUDA C Programing guide.   *
 ******************************************************************************/
#ifdef __CUDA
#if (__CUDACC_VER_MAJOR__<8) || ( defined(__CUDA_ARCH__) && (__CUDA_ARCH__<600) )
static __device__ double atomicAdd(double *address, double val) {
    unsigned long long int *address_as_ull =
        (unsigned long long int *) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif
#endif

/******************************************************************************
 * A simple __ldg replacement for older cuda devices.                         *
 ******************************************************************************/
#ifdef __CUDA
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 350)
#define __ldg(x)  (*(x))
#endif
#endif

/******************************************************************************
 * syncthreads macro                                                          *
 ******************************************************************************/

#if (defined(__HIP) && not defined(__HIP_PLATFORM_NVCC__)) || (defined(__CUDA) && ((__CUDACC_VER_MAJOR__ >= 8) || ( defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600) ) ) )
#define syncthreads(x) __syncthreads(x)
#endif
