/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

__attribute__((always_inline))
inline void atomic_add_global(global volatile T* dst, T inc)
{
  union { TA a; T f; } old_val, try_val, new_val = { .f = *dst };
  do {
    old_val.a = new_val.a;
    try_val.f = old_val.f + inc;
    new_val.a = FA((global volatile TA*)dst, old_val.a, try_val.a);
  } while (old_val.a != new_val.a);
}


__attribute__((reqd_work_group_size(SN, 1, 1)))
kernel void FN(global const int *restrict param_stack,
  global const T *restrict amat, global const T *restrict bmat, global T *restrict cmat)
{
  const int gid = get_group_id(0);
  global const int *const restrict param_base = param_stack + gid * 3;
  /* indexes given by param_stack are one-based */
  const int ai = param_base[0] - 1, bi = param_base[1] - 1, ci = param_base[2] - 1;
  global const T *const restrict awg = amat + ai, *const restrict bwg = bmat + bi;
  global T *const restrict cwg = cmat + ci;
  local T a[SM*SK];
  T b[SK];

  const int n = get_local_id(0);
  /* assume SN == get_local_size(0) */
  const int msize = ((SM - 1) + SN) / SN;
  const int m0 = n * msize, m1 = min(m0 + msize, SM);
  /* split work among WG (a[m,k] does not depend on WG-index) */
  for (int m = m0; m < m1; ++m) {
    for (int k = 0; k < SK; ++k) a[SK*m+k] = awg[SM*k+m];
  }
  for (int k = 0; k < SK; ++k) b[k] = bwg[SN*k+n];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int m = 0; m < SM; ++m) {
    T r = 0;
    for (int k = 0; k < SK; ++k) r += a[SK*m+k] * b[k];
    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_add_global(&cwg[SM*n+m], r);
  }
}
