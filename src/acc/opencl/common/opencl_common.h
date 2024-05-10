/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/
#ifndef OPENCL_COMMON_H
#define OPENCL_COMMON_H

#if !defined(ACC_OPENCL_C_VERSION)
#  define ACC_OPENCL_C_VERSION __OPENCL_C_VERSION__
#endif
#if !defined(ACC_OPENCL_VERSION)
#  define ACC_OPENCL_VERSION __OPENCL_VERSION__
#endif

#if (200 /*CL_VERSION_2_0*/ <= ACC_OPENCL_C_VERSION) || defined(__NV_CL_C_VERSION)
#  define UNROLL_FORCE(N) __attribute__((opencl_unroll_hint(N)))
#  define UNROLL_AUTO __attribute__((opencl_unroll_hint))
#else
#  define UNROLL_FORCE(N)
#  define UNROLL_AUTO
#endif

#if !defined(LU) || (-1 == LU)
#  define UNROLL_OUTER(N)
#  define UNROLL(N)
#else /* (-2) full, (-1) no hints, (0) inner, (1) outer-dehint, (2) block-m */
#  if (1 <= LU) /* outer-dehint */
#    define UNROLL_OUTER(N) UNROLL_FORCE(1)
#  elif (-1 > LU) /* full */
#    define UNROLL_OUTER(N) UNROLL_FORCE(N)
#  else /* inner */
#    define UNROLL_OUTER(N)
#  endif
#  define UNROLL(N) UNROLL_FORCE(N)
#endif

#if !defined(MIN)
#  define MIN(A, B) ((A) < (B) ? (A) : (B))
#endif
#if !defined(MAX)
#  define MAX(A, B) ((A) < (B) ? (B) : (A))
#endif
#if !defined(MAD)
#  define MAD fma
#endif

#define DIVUP(A, B) (((A) + (B) - 1) / (B))
#define NUP(N, UP) (DIVUP(N, UP) * (UP))
#define BLR(N, BN) (NUP(N, BN) - (N))

#define IDX(I, J, M, N) ((int)(I) * (N) + (J))
#define IDT(I, J, M, N) IDX(J, I, N, M)

#endif /*OPENCL_COMMON_H*/
