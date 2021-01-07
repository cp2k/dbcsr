/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#define NBN ((SN + BN - 1) / BN)


__attribute__((always_inline))
inline void atomic_add_global_cmpxchg(global volatile T* dst, T inc)
{
  union { TA a; T f; } old_val, try_val, new_val = { .f = *dst };
  do {
    old_val.a = new_val.a;
    try_val.f = old_val.f + inc;
    new_val.a = CMPXCHG((global volatile TA*)dst, old_val.a, try_val.a);
  } while (old_val.a != new_val.a);
}


__attribute__((always_inline))
inline void atomic_add_global_xchg(global volatile T* dst, T inc)
{
  union { TA a; T f; } old_val = { .f = inc }, try_val, new_val = { .f = 0 };
  do {
    try_val.a = XCHG((global volatile TA*)dst, new_val.a);
    try_val.f += old_val.f;
    old_val.a = XCHG((global volatile TA*)dst, try_val.a);
  } while (old_val.a != new_val.a);
}


kernel void FN(GLOBAL const int *restrict param_stack,
  GLOBAL const T *restrict amat, GLOBAL const T *restrict bmat,
  global T *restrict cmat)
{
  const int gid = get_group_id(0), idx = get_local_id(0);
  GLOBAL const int *const restrict params = param_stack + gid * 3;
  /* indexes given by param_stack are one-based */
  const int ai = params[0] - 1, bi = params[1] - 1, ci = params[2] - 1;
  global T *const restrict cwg = cmat + ci;
  local T a[SM*SK];

#if (1 != BM) || (SN != BN)
  const int im = idx / NBN;
  const int m0 = im * BM, m1 = min(m0 + BM, SM);
  const int n0 = (idx - im * NBN) * BN;
  const int n1 = min(n0 + BN, SN);
  local T b[SK*SN];
# if (1 < BS)
  T c[BM*BN] = { 0 };
# endif
#else
  const int m0 = idx * BM, m1 = min(m0 + BM, SM);
  const int n = idx;
  T b[SK];
# if (1 < BS)
  T c[SM] = { 0 }
# endif
#endif

  { /* transpose A-matrix into local buffer */
    GLOBAL const T *const restrict awg = amat + ai;
    for (int m = m0; m < m1; ++m) {
      for (int k = 0; k < SK; ++k) a[SK*m+k] = awg[SM*k+m];
    }
  }

  { /* copy B-matrix into local buffer */
    GLOBAL const T *const restrict bwg = bmat + bi;
    for (int k = 0; k < SK; ++k) {
#if (1 != BM) || (SN != BN)
      for (int n = n0; n < n1; ++n) b[SN*k+n] = bwg[SN*k+n];
#else
      b[k] = bwg[SN*k+n];
#endif
    }
  }

  { /* calculate private result-tile */
    barrier(CLK_LOCAL_MEM_FENCE);
#if (1 != BM) || (SN != BN)
    for (int m = m0; m < m1; ++m) for (int n = n0; n < n1; ++n) {
# if (1 < BS)
      T *const restrict r = c + BN * (m-m0) + n-n0;
      for (int k = 0; k < SK; ++k) *r = FMA(a[SK*m+k], b[SN*k+n], *r);
# else
      T r = 0;
      for (int k = 0; k < SK; ++k) r = FMA(a[SK*m+k], b[SN*k+n], r);
      ATOMIC_ADD_GLOBAL(&cwg[SM*n+m], r);
# endif
    }
#else
    for (int m = 0; m < SM; ++m) {
# if (1 < BS)
      T *const restrict r = c + m;
      for (int k = 0; k < SK; ++k) *r = FMA(a[SK*m+k], b[k], *r);
# else
      T r = 0;
      for (int k = 0; k < SK; ++k) r = FMA(a[SK*m+k], b[k], r);
      ATOMIC_ADD_GLOBAL(&cwg[SM*n+m], r);
# endif
    }
#endif
  }

#if (1 < BS)
  { /* copy private tile to global memory */
# if (1 != BM) || (SN != BN)
    for (int m = m0; m < m1; ++m) for (int n = n0; n < n1; ++n) {
      ATOMIC_ADD_GLOBAL(&cwg[SM*n+m], c[BN*(m-m0)+n-n0]);
    }
# else
    for (int m = 0; m < SM; ++m) {
      ATOMIC_ADD_GLOBAL(&cwg[SM*n+m], c[m]);
    }
# endif
  }
#endif
}
