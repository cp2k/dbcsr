/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/

// work around an issue where -D flags are not propagated in hiprtcCompileProgram (tested on 3.9.0)
#if defined(__HIP_ROCclr__)
#  if !defined(__HIP)
#    define __HIP
#  endif
#endif
#if defined(__HIP) && !defined(__HIP_PLATFORM_NVCC__) && !defined(__HIPCC_RTC__)
#  include <hip/hip_runtime.h>
#endif

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/******************************************************************************
 * There is no native support for atomicAdd on doubles in Cuda 5.0. However   *
 * the following implementation is provided in the CUDA C Programing guide.   *
 ******************************************************************************/
#if defined(__CUDA)
#  if (__CUDACC_VER_MAJOR__ < 8) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600))
static __device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#  endif
#endif

/******************************************************************************
 * A simple __ldg replacement for older cuda devices.                         *
 ******************************************************************************/
#if defined(__CUDA)
#  if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 350)
#    define __ldg(x) (*(x))
#  endif
#endif

/******************************************************************************
 * syncthreads macro                                                          *
 ******************************************************************************/
// clang-format off
#if (defined(__HIP) && not defined(__HIP_PLATFORM_NVCC__)) || (defined(__CUDA) && ((__CUDACC_VER_MAJOR__ >= 8) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600))))
#  define syncthreads(x) __syncthreads(x)
#endif
// clang-format on
