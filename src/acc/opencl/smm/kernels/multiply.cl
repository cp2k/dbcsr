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
#if defined(INTEL)
# define BC 1
#else
# define BC 2
#endif
#if (1 == TN)
# define ZERO 0.f
#elif (3 == TN)
# define ZERO 0.0
#else
# define ZERO 0
#endif

/* number of M-blocks */
#define NBM ((SM + BM - 1) / BM)
/* number of N-blocks */
#define NBN ((SN + BN - 1) / BN)
/* size of workgroup (WG) */
#define SWG (NBM * NBN)

#if !defined(SHARED_A) && 1
# define SHARED_A ((SK % 16) ? 1 : BC)
#endif
#if !defined(SHARED_B) && !defined(INTEL) && 1
# define SHARED_B ((SN % 16) ? 1 : BC)
#endif
#if !defined(SHARED_C) && 0
# define SHARED_C ((SN % 16) ? 1 : BC)
#endif
#if !defined(SHARED_S) && !defined(INTEL) && 1
# define SHARED_S
#endif
#if !defined(PRIVATE_A) && !defined(SHARED_A)
# define PRIVATE_A
#endif
#if !defined(PRIVATE_B) && !defined(SHARED_B)
# define PRIVATE_B
#endif
#if !defined(PRIVATE_C) && !defined(SHARED_C)
# define PRIVATE_C
#endif
#if !defined(TRACK_B) && (1 < BS) && 0
# if defined(PRIVATE_B) && !defined(SHARED_B)
#   define TRACK_B
# endif
#endif
#if !defined(TRACK_C) && (1 < BS) && 1
# define TRACK_C
#endif
#if defined(SHARED_S) && (1 < BS)
# define IDXBASE 0
#else
# define IDXBASE 1
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
  /* indexes given by param_stack are one-based (Fortran) */
#if (1 < BS)
  GLOBAL const int *restrict param_stack, int stack_size)
#else
  GLOBAL const int *restrict param_stack)
#endif
{
  const int gid = get_group_id(0), idx = get_local_id(0);
  GLOBAL const int *restrict pbase = param_stack + gid * (3 * BS);
#if defined(SHARED_S) && (1 < BS)
  local int params[3*BS];
#else
  GLOBAL const int *restrict params = pbase;
#endif
#if defined(SHARED_A)
# if (1 < SHARED_A)
  local T amk[SM][SK+1];
# else
  local T amk[SM][SK];
# endif
#endif
#if defined(SHARED_B)
# if (1 < SHARED_B)
  local T bkn[SK][SN+1];
# else
  local T bkn[SK][SN];
# endif
#endif
#if (SWG != SN)
# if defined(PRIVATE_A) && !defined(SHARED_A)
  T amk[SK];
# endif
# if defined(PRIVATE_B) && !defined(SHARED_B)
  T bkn[SK][BN];
# endif
# if defined(PRIVATE_C) && !defined(SHARED_C) && (1 < BS)
  T cmn[BM][BN] = {{ 0 }};
# endif
  const int m0 = (idx / NBN) * BM, m1 = min(m0 + BM, SM);
  const int n0 = (idx % NBN) * BN, n1 = min(n0 + BN, SN);
#else
# if defined(PRIVATE_B) && !defined(SHARED_B)
  T bkn[SK];
# endif
# if defined(PRIVATE_C) && !defined(SHARED_C) && (1 < BS)
  T cmn[SM] = { 0 };
# endif
#endif
#if defined(TRACK_B)
  int b1 = -1;
#endif

  /* intra-kernel mini-batch of SMMs */
#if (1 < BS)
  const int batchsize = min(BS, stack_size - BS * gid);
  global T *restrict c;
  int c0, i;
# if defined(SHARED_C)
#   if (1 < SHARED_C)
  local T cmn[SM][SN+1];
#   else
  local T cmn[SM][SN];
#   endif
  for (int m = idx; m < SM; m += SWG) {
    UNROLL(SN)
    for (int n = 0; n < SN; ++n) cmn[m][n] = ZERO;
  }
# endif
# if defined(SHARED_S)
  for (i = idx; i < (3 * batchsize); i += SWG) params[i] = pbase[i] - 1;
# endif
# if defined(SHARED_C) || defined(SHARED_S)
  barrier(CLK_LOCAL_MEM_FENCE);
# endif
  c0 = params[2] - IDXBASE;
  c = cdata + c0;
  UNROLL(1)
  for (i = 0; i < batchsize; ++i) {
    const int a0 = params[3*i] - IDXBASE, b0 = params[3*i+1] - IDXBASE;
    const int c1 = ((i + 1) < batchsize ? (params[3*i+5] - IDXBASE) : -1);
#else
  {
    const int a0 = params[0] - IDXBASE, b0 = params[1] - IDXBASE;
    global T *restrict c = cdata + params[2] - IDXBASE;
#endif
    GLOBAL const T *const restrict a = adata + a0;
    GLOBAL const T *const restrict b = bdata + b0;

#if defined(SHARED_A)
    { /* transpose A-matrix into local/shared buffer */
      int m = idx;
# if (SM != SN || SWG != SN)
      for (; m < SM; m += SWG)
# endif
      { UNROLL(SK)
        for (int k = 0; k < SK; ++k) amk[m][k] = a[SM*k+m];
      }
    }
#endif

#if defined(SHARED_B)
    UNROLL(SK)
    for (int k = 0; k < SK; ++k) {
      int n = idx;
# if (SM != SN || SWG != SN)
      for (; n < SN; n += SWG)
# endif
      bkn[k][n] = b[SN*k+n];
    }
#elif defined(PRIVATE_B)
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

#if (defined(SHARED_A) || defined(SHARED_B)) && (1 < SWG)
    /* finish transpose/copy */
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    /* calculate private result-tile */
#if (SWG != SN)
    /*UNROLL(BM)*/
    for (int m = m0; m < m1; ++m) {
# if defined(PRIVATE_A) && !defined(SHARED_A)
      UNROLL(SK)
      for (int k = 0; k < SK; ++k) amk[k] = a[SM*k+m];
# endif
      /*UNROLL(BN)*/
      for (int n = n0; n < n1; ++n) {
        T r = ZERO;
        UNROLL(SK)
        for (int k = 0; k < SK; ++k) r = FMA(
# if defined(SHARED_A)
          amk[m][k],
# elif defined(PRIVATE_A)
          amk[k],
# else
          a[SM*k+m],
# endif
# if defined(SHARED_B)
          bkn[k][n],
# elif defined(PRIVATE_B)
          bkn[k][n-n0],
# else
          b[SN*k+n],
# endif
          r);
# if (1 < BS)
#   if defined(SHARED_C)
        cmn[m][n] += r;
#   else
        cmn[m-m0][n-n0] += r;
#   endif
# else
        if (ZERO != r) ATOMIC_ADD_GLOBAL(&c[SM*n+m], r);
# endif
      }
    }
#else
    UNROLL(SM)
    for (int m = 0; m < SM; ++m) {
# if (1 < BS)
#   if defined(SHARED_C)
      T r = cmn[m][idx];
#   else
      T r = cmn[m];
#   endif
# else
      T r = ZERO;
# endif
      UNROLL(SK)
      for (int k = 0; k < SK; ++k) r = FMA(
# if defined(SHARED_A)
        amk[m][k],
# else
        a[SM*k+m],
# endif
# if defined(SHARED_B)
        bkn[k][idx],
# elif defined(PRIVATE_B)
        bkn[k],
# else
        b[SN*k+idx],
# endif
        r);
# if (1 < BS)
#   if defined(SHARED_C)
      cmn[m][idx] = r;
#   else
      cmn[m] = r;
#   endif
# else
      if (ZERO != r) ATOMIC_ADD_GLOBAL(&c[SM*idx+m], r);
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
#   if defined(SHARED_C)
          local T *restrict ci = &cmn[gm][gn];
#   else
          private T *restrict ci = &cmn[m][n];
#   endif
          if (gm < SM && gn < SN && ZERO != *ci) {
            ATOMIC_ADD_GLOBAL(&c[SM*gn+gm], *ci);
            *ci = ZERO; /* reset */
          }
        }
      }
# else
#   if defined(ATOMIC_ADD2_GLOBAL)
      UNROLL(SM)
      for (int m = 0; m < SM; m += 2) {
#     if defined(SHARED_C)
        local T *restrict c0 = &cmn[m+0][idx];
        local T *restrict c1 = &cmn[m+1][idx];
#     else
        private T *restrict c0 = &cmn[m+0];
        private T *restrict c1 = &cmn[m+1];
#     endif
        /*if (ZERO != *c0 && ZERO != *c1)*/ {
          const float2 r2 = (float2)(*c0, *c1);
          ATOMIC_ADD2_GLOBAL((global volatile float2*)(c + SM * idx + m), r2);
          *c0 = *c1 = ZERO; /* reset */
        }
      }
#   else
      UNROLL(SM)
      for (int m = 0; m < SM; ++m) {
#     if defined(SHARED_C)
        local T *restrict ci = &cmn[m][idx];
#     else
        private T *restrict ci = cmn + m;
#     endif
        if (ZERO != *ci) {
          ATOMIC_ADD_GLOBAL(&c[SM*idx+m], *ci);
          *ci = ZERO; /* reset */
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
