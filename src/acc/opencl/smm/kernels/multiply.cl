/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

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


__attribute__((reqd_work_group_size(SN, 1, 1)))
kernel void FN(GLOBAL const int *restrict param_stack,
  GLOBAL const T *restrict amat, GLOBAL const T *restrict bmat,
  global T *restrict cmat)
{
  const int gid = get_group_id(0);
  GLOBAL const int *const restrict param_base = param_stack + gid * 3;
  /* indexes given by param_stack are one-based */
  const int ai = param_base[0] - 1, bi = param_base[1] - 1, ci = param_base[2] - 1;
  GLOBAL const T *const restrict awg = amat + ai, *const restrict bwg = bmat + bi;
  global T *const restrict cwg = cmat + ci;
  local T a[SM*SK];
  T b[SK];

  const int n = get_local_id(0);
  /* assume SN == get_local_size(0) */
  const int msize = (SM + SN - 1) / SN;
  const int m0 = n * msize, m1 = min(m0 + msize, SM);
  /* split work among WG (a[m,k] does not depend on WG-index) */
  for (int m = m0; m < m1; ++m) { /* transpose A-matrix */
    for (int k = 0; k < SK; ++k) a[SK*m+k] = awg[SM*k+m];
  }
  /* gather/transpose B-matrix (strided load) */
  for (int k = 0; k < SK; ++k) b[k] = bwg[SN*k+n];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int m = 0; m < SM; ++m) {
    T r = 0;
    for (int k = 0; k < SK; ++k) r += a[SK*m+k] * b[k];
    barrier(CLK_LOCAL_MEM_FENCE);
    ATOMIC_ADD_GLOBAL(&cwg[SM*n+m], r);
  }
}
