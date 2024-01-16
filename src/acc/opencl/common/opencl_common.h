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

#if (200 /*CL_VERSION_2_0*/ <= __OPENCL_VERSION__) || defined(__NV_CL_C_VERSION)
#  define UNROLL_FORCE(N) __attribute__((opencl_unroll_hint(N)))
#else
#  define UNROLL_FORCE(N)
#endif

#define MIN(A, B) ((A) < (B) ? (A) : (B))
#define MAX(A, B) ((A) < (B) ? (B) : (A))

#if !defined(LU) || (-1 == LU)
#  define UNROLL_OUTER(N)
#  define UNROLL(N)
#else
#  if (1 <= LU)
#    define UNROLL_OUTER(N) UNROLL_FORCE(1)
#  else
#    define UNROLL_OUTER(N) UNROLL_FORCE(N)
#  endif
#  define UNROLL(N) UNROLL_FORCE(N)
#endif

#endif /*OPENCL_COMMON_H*/
