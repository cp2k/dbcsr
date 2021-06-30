/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#if (200/*CL_VERSION_2_0*/ <= __OPENCL_VERSION__)
# define UNROLL(N) __attribute__((opencl_unroll_hint(N)))
#else
# define UNROLL(N)
#endif

#define BMN ((SM + SN - 1) / SN)
/* number of M-blocks */
#define NBM ((SM + BM - 1) / BM)
/* number of N-blocks */
#define NBN ((SN + BN - 1) / BN)
/* size of workgroup (WG) */
#define SWG (NBM * NBN)

#if !defined(PRIVATE_A) && (SWG != SN)
# define PRIVATE_A
#endif
#if !defined(PRIVATE_B) && 0
# define PRIVATE_B
#endif
#if !defined(SHARED_A) && !defined(PRIVATE_A)
# define SHARED_A
#endif
#if !defined(TRACK_B) && 0
# define TRACK_B
#endif
#if !defined(TRACK_C) && 1
# define TRACK_C
#endif


#if !defined(cl_intel_global_float_atomics)
#if defined(CMPXCHG)
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
#endif

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

#if defined(__NV_CL_C_VERSION) || defined(XCHG)
__attribute__((always_inline))
inline void atomic_add_global_xchg(global volatile T* dst, T inc)
{
# if (defined(__NV_CL_C_VERSION) && !defined(XCHG)) && (1 == TN)
  asm("{ .reg .f32 t; atom.global.add.f32 t, [%0], %1; }" :: "l"(dst), "f"(inc));
# elif (defined(__NV_CL_C_VERSION) && !defined(XCHG)) && (3 == TN)
  asm("{ .reg .f64 t; atom.global.add.f64 t, [%0], %1; }" :: "l"(dst), "d"(inc));
# else
  union { T f; TA a; } old_val = { .f = inc }, try_val, new_val = { .f = 0 };
  do {
    try_val.a = XCHG((global volatile TA*)dst, new_val.a);
    try_val.f += old_val.f;
    old_val.a = XCHG((global volatile TA*)dst, try_val.a);
  } while (old_val.a != new_val.a);
# endif
}
#endif
#endif


__attribute__((reqd_work_group_size(SWG, 1, 1)))
kernel void FN(global T *restrict cdata,
  GLOBAL const T *restrict adata, GLOBAL const T *restrict bdata,
#if (1 < BS)
  GLOBAL const int *restrict param_stack, int stack_size)
#else
  GLOBAL const int *restrict param_stack)
#endif
{
  const int gid = get_group_id(0), idx = get_local_id(0);
  /* indexes given by param_stack are one-based */
  GLOBAL const int *restrict params = param_stack + gid * (3 * BS);
#if (1 < BS)
# if defined(TRACK_B)
  int b1 = -1;
# endif
#else
  GLOBAL const T *const restrict a = adata + params[0] - 1;
  GLOBAL const T *const restrict b = bdata + params[1] - 1;
#endif
  int c0 = params[2] - 1;
  global T *restrict c = cdata + c0;

#if defined(SHARED_A)
  local T awg[SM][SK+BC];
#endif
#if (SWG != SN)
# if defined(PRIVATE_A)
  T amk[SK];
# endif
# if defined(PRIVATE_B)
  T bkn[SK][BN];
# endif
# if (1 < BS)
  T cmn[BM][BN] = {{ 0 }};
# endif
  const int m0 = (idx / NBN) * BM, m1 = min(m0 + BM, SM);
  const int n0 = (idx % NBN) * BN, n1 = min(n0 + BN, SN);
#else
# if defined(PRIVATE_B)
  T bkn[SK];
# endif
# if (1 < BS)
  T cmn[SM] = { 0 };
# endif
# if (SM != SN)
  const int m0 = idx * BMN, m1 = min(m0 + BMN, SM);
# endif
#endif

  /* intra-kernel mini-batch of SMMs */
#if (1 < BS)
  const int batchsize = min(BS, stack_size - BS * gid);
  UNROLL(1)
  for (int i = 0; i < batchsize; ++i) {
    const int c1 = (i < (batchsize - 1) ? (params[5] - 1) : -1);
    const int a0 = params[0] - 1, b0 = params[1] - 1;
    GLOBAL const T *const restrict a = adata + a0;
    GLOBAL const T *const restrict b = bdata + b0;
    params += 3; /* next set of indexes */
#else
  {
#endif
    /* transpose A-matrix into local/shared buffer */
#if defined(SHARED_A)
# if (SM != SN || SWG != SN)
    UNROLL(BMN)
    for (int m = m0; m < m1; ++m) {
# else
    { const int m = idx;
# endif
      UNROLL(SK)
      for (int k = 0; k < SK; ++k) awg[m][k] = a[SM*k+m];
    }
#endif

#if defined(PRIVATE_B)
# if defined(TRACK_B) && (1 < BS)
    if (b0 != b1) {
      b1 = b0;
# else
    { /* copy B-matrix into private buffer */
# endif
      UNROLL(SK)
      for (int k = 0; k < SK; ++k) {
# if (SWG != SN)
#   if defined(__NV_CL_C_VERSION)
        UNROLL(BN)
#   endif
        for (int n = n0; n < n1; ++n) bkn[k][n-n0] = b[SN*k+n];
# else
        bkn[k] = b[SN*k+idx];
# endif
      }
    }
#endif

#if defined(SHARED_A) && (1 < SWG)
    /* finish copy-transpose */
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    /* calculate private result-tile */
#if (SWG != SN)
    /*UNROLL(BM)*/
    for (int m = m0; m < m1; ++m) {
# if defined(PRIVATE_A)
      UNROLL(SK)
      for (int k = 0; k < SK; ++k) amk[k] = a[SM*k+m];
# endif
      /*UNROLL(BN)*/
      for (int n = n0; n < n1; ++n) {
        T r = 0;
        UNROLL(SK)
# if defined(PRIVATE_B)
#   if defined(PRIVATE_A)
        for (int k = 0; k < SK; ++k) r = FMA(amk[k], bkn[k][n-n0], r);
#   else
        for (int k = 0; k < SK; ++k) r = FMA(awg[m][k], bkn[k][n-n0], r);
#   endif
# else
#   if defined(PRIVATE_A)
        for (int k = 0; k < SK; ++k) r = FMA(amk[k], b[SN*k+n], r);
#   else
        for (int k = 0; k < SK; ++k) r = FMA(awg[m][k], b[SN*k+n], r);
#   endif
# endif
# if (1 < BS)
        cmn[m-m0][n-n0] += r;
# else
        if (0 != r) ATOMIC_ADD_GLOBAL(&c[SM*n+m], r);
# endif
      }
    }
#else
    UNROLL(SM)
    for (int m = 0; m < SM; ++m) {
# if (1 < BS)
      T r = cmn[m];
# else
      T r = 0;
# endif
      UNROLL(SK)
# if defined(PRIVATE_B)
#   if defined(PRIVATE_A) || !defined(SHARED_A)
      for (int k = 0; k < SK; ++k) r = FMA(a[SM*k+m], bkn[k], r);
#   else
      for (int k = 0; k < SK; ++k) r = FMA(awg[m][k], bkn[k], r);
#   endif
# else
#   if defined(PRIVATE_A) || !defined(SHARED_A)
      for (int k = 0; k < SK; ++k) r = FMA(a[SM*k+m], b[SN*k+idx], r);
#   else
      for (int k = 0; k < SK; ++k) r = FMA(awg[m][k], b[SN*k+idx], r);
#   endif
# endif
# if (1 < BS)
      cmn[m] = r;
# else
      if (0 != r) ATOMIC_ADD_GLOBAL(&c[SM*idx+m], r);
# endif
    }
#endif

#if (1 < BS)
# if defined(TRACK_C)
    if (c0 != c1)
# endif
    { /* atomically commit private C-tile to global memory */
# if (SWG != SN)
      UNROLL(1)
      for (int m = 0; m < BM; ++m) {
        UNROLL(BN)
        for (int n = 0; n < BN; ++n) {
          const int gm = m + m0, gn = n + n0;
          if (gm < SM && gn < SN && 0 != cmn[m][n]) {
            ATOMIC_ADD_GLOBAL(&c[SM*gn+gm], cmn[m][n]);
            cmn[m][n] = 0; /* reset */
          }
        }
      }
# else
#   if defined(ATOMIC_ADD2_GLOBAL)
      UNROLL(SM)
      for (int m = 0; m < SM; m += 2) {
        /*if (0 != cmn[m] && 0 != cmn[m+1])*/ {
          const float2 r2 = (float2)(cmn[m], cmn[m+1]);
          ATOMIC_ADD2_GLOBAL((global volatile float2*)(c + SM * idx + m), r2);
          cmn[m] = cmn[m+1] = 0; /* reset */
        }
      }
#   else
      UNROLL(SM)
      for (int m = 0; m < SM; ++m) {
        if (0 != cmn[m]) {
          ATOMIC_ADD_GLOBAL(&c[SM*idx+m], cmn[m]);
          cmn[m] = 0; /* reset */
        }
      }
#   endif
# endif
      /* next iteration */
      c = cdata + c1;
      c0 = c1;
    }
#endif
  }
}
