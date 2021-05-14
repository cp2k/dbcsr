/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

/* number of M-blocks */
#define NBM ((SM + BM - 1) / BM)
/* number of N-blocks */
#define NBN ((SN + BN - 1) / BN)
/* size of workgroup (WG) */
#define SWG (NBM * NBN)


#if !defined(cl_intel_global_float_atomics)

__attribute__((always_inline))
inline void atomic_add_global_cmpxchg(global volatile T* dst, T inc)
{
  union { T f; TA a; } old_val, try_val, new_val = { .f = *dst };
  do {
    old_val.a = new_val.a;
    try_val.f = old_val.f + inc;
    new_val.a = CMPXCHG((global volatile TA*)dst, old_val.a, try_val.a);
  } while (old_val.a != new_val.a);
}

__attribute__((always_inline))
inline void atomic_add_global_xchg(global volatile T* dst, T inc)
{
  union { T f; TA a; } old_val = { .f = inc }, try_val, new_val = { .f = 0 };
  do {
    try_val.a = XCHG((global volatile TA*)dst, new_val.a);
    try_val.f += old_val.f;
    old_val.a = XCHG((global volatile TA*)dst, try_val.a);
  } while (old_val.a != new_val.a);
}

#if defined(ATOMIC_ADD2_GLOBAL)
__attribute__((always_inline))
inline void atomic_add_global_cmpxchg2(global volatile float2* dst, float2 inc)
{
  union { float2 f; long a; } old_val, try_val, new_val = { .f = *dst };
  do {
    old_val.a = new_val.a;
    try_val.f = old_val.f + inc;
    new_val.a = atom_cmpxchg((global volatile long*)dst, old_val.a, try_val.a);
  } while (old_val.a != new_val.a);
}
#endif

#endif


kernel void FN(global T *restrict cdata,
  GLOBAL const T *restrict adata, GLOBAL const T *restrict bdata,
#if (1 < BS)
  GLOBAL const int *restrict param_stack, int stack_size)
#else
  GLOBAL const int *restrict param_stack)
#endif
{
  const int gid = get_group_id(0), idx = get_local_id(0);
  GLOBAL const int *const restrict params = param_stack + gid * (3 * BS);
  /* indexes given by param_stack are one-based */
  int c0 = params[2] - 1;
#if (1 < BS)
  int b1 = -1;
#endif
  global T *restrict c = cdata + c0;

  T amk[SK];
#if (SWG != SN)
  T bkn[SK][BN];
# if (1 < BS)
  T cmn[BM][BN] = {{ 0 }};
# endif
#else
  local T awg[SM][SK];
  T bkn[SK];
# if (1 < BS)
  T cmn[SM] = { 0 };
# endif
#endif

  /* intra-kernel mini-batch of SMMs */
#if (1 < BS)
  const int batchsize = min(BS, stack_size - BS * gid);
  for (int i = 0; i < batchsize; ++i)
#endif
  {
#if (SWG != SN)
    const int im = idx / NBN;
    const int m0 = im * BM, m1 = min(m0 + BM, SM);
    const int n0 = (idx - im * NBN) * BN;
    const int n1 = min(n0 + BN, SN);
#else
    const int bm = (SM + SN - 1) / SN;
    const int m0 = idx * bm, m1 = min(m0 + bm, SM);
    const int n = idx;
#endif
#if (1 < BS)
    const int c1 = (i < (batchsize - 1) ? (params[3*i+5] - 1) : -1);
    const int a0 = params[3*i+0] - 1, b0 = params[3*i+1] - 1;
#else
    const int a0 = params[0] - 1, b0 = params[1] - 1;
#endif
    GLOBAL const T *const restrict a = adata + a0;

#if (SWG == SN)
    /* transpose A-matrix into local buffer */
    for (int m = m0; m < m1; ++m) {
      for (int k = 0; k < SK; ++k) awg[m][k] = a[SM*k+m];
    }
#endif

    /* avoiding to load same B-tile seems to be not beneficial */
#if (1 < BS) && 0
    if (b0 != b1)
#endif
    { /* copy B-matrix into private buffer */
      GLOBAL const T *const restrict b = bdata + b0;
      for (int k = 0; k < SK; ++k) {
#if (SWG != SN)
        for (int n = n0; n < n1; ++n) bkn[k][n-n0] = b[SN*k+n];
#else
        bkn[k] = b[SN*k+n];
#endif
      }
#if (1 < BS)
      b1 = b0;
#endif
    }

    /* calculate private result-tile */
#if (SWG != SN)
    for (int m = m0; m < m1; ++m) {
      for (int k = 0; k < SK; ++k) amk[k] = a[SM*k+m];
      for (int n = n0; n < n1; ++n) {
        T r = 0;
        for (int k = 0; k < SK; ++k) r = FMA(amk[k], bkn[k][n-n0], r);
# if (1 < BS)
        cmn[m-m0][n-n0] += r;
# else
        if (0 != r) ATOMIC_ADD_GLOBAL(&c[SM*n+m], r);
# endif
      }
    }
#else
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int m = 0; m < SM; ++m) {
      T r = 0;
      for (int k = 0; k < SK; ++k) amk[k] = awg[m][k];
      for (int k = 0; k < SK; ++k) r = FMA(amk[k], bkn[k], r);
# if (1 < BS)
      cmn[m] += r;
# else
      if (0 != r) ATOMIC_ADD_GLOBAL(&c[SM*n+m], r);
# endif
    }
#endif

#if (1 < BS)
    if (c0 != c1) { /* copy private tile to global memory */
# if (SWG != SN)
      for (int m = 0; m < BM; ++m) for (int n = 0; n < BN; ++n) {
        const int gm = m + m0, gn = n + n0;
        if (gm < SM && gn < SN && 0 != cmn[m][n]) {
          ATOMIC_ADD_GLOBAL(&c[SM*gn+gm], cmn[m][n]);
          cmn[m][n] = 0; /* reset */
        }
      }
# else
#   if defined(ATOMIC_ADD2_GLOBAL)
      for (int m = 0; m < SM; m += 2) {
        float2 *const restrict r = (float2*)(cmn + m);
        if (0 != r) {
          ATOMIC_ADD2_GLOBAL((global volatile float2*)(c + SM * n + m), *r);
          *r = 0; /* reset */
        }
      }
#   else
      for (int m = 0; m < SM; ++m) {
        if (0 != cmn[m]) {
          ATOMIC_ADD_GLOBAL(&c[SM*n+m], cmn[m]);
          cmn[m] = 0; /* reset */
        }
      }
#   endif
# endif
      /* next iteration */
      c = cdata + c1;
      c0 = c1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
  }
}
