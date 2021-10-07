/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#if (200/*CL_VERSION_2_0*/ <= __OPENCL_VERSION__) || defined(__NV_CL_C_VERSION)
# define UNROLL_FORCE(N) __attribute__((opencl_unroll_hint(N)))
#else
# define UNROLL_FORCE(N)
#endif
#if !defined(UNROLL)
# define UNROLL(N)
#endif
#if !defined(UNROLL_SM)
# define UNROLL_SM UNROLL(SM)
#endif

#if (1 == TN)
# define ZERO 0.f
#elif (3 == TN)
# define ZERO 0.0
#else
# define ZERO 0
#endif

#if !defined(TRACK_B) && (1 < BS) && 0
# if defined(PRIVATE_B) && !defined(SHARED_B)
#   define TRACK_B
# endif
#endif
#if !defined(TRACK_C) && (1 < BS) && 1
# define TRACK_C
#endif

#if defined(SHARED_P) && (1 < BS)
# define IDXBASE 0
#else
# define IDXBASE 1
#endif

#define NBM ((SM + BM - 1) / BM)
#define NBN ((SN + BN - 1) / BN)
#define NBK (NBM * NBN)


#if !defined(cl_intel_global_float_atomics) || (1 != TN)
#if defined(CMPXCHG)
__attribute__((always_inline))
inline void atomic_add_global_cmpxchg(global volatile T* dst, T inc)
{
  union { T f; TA a; } exp_val, try_val, cur_val = { .f = *dst };
  do {
    exp_val.a = cur_val.a; try_val.f = exp_val.f + inc;
# if defined(TM)
    if (0 == atomic_compare_exchange_weak_explicit((global volatile TM*)dst, &cur_val.a, try_val.a,
      memory_order_relaxed, memory_order_relaxed, memory_scope_work_group)) continue;
# else
    cur_val.a = CMPXCHG((global volatile TA*)dst, exp_val.a, try_val.a);
# endif
  } while (cur_val.a != exp_val.a);
}
#endif

#if defined(ATOMIC_ADD2_GLOBAL)
__attribute__((always_inline))
inline void atomic_add_global_cmpxchg2(global volatile float* dst, float2 inc)
{
  union { float2 f; long a; } exp_val, try_val, cur_val = { .f = (float2)(dst[0], dst[1]) };
  do {
    exp_val.a = cur_val.a; try_val.f = exp_val.f + inc;
# if defined(TM)
    if (0 == atomic_compare_exchange_weak_explicit((global volatile atomic_long*)dst, &cur_val.a, try_val.a,
      memory_order_relaxed, memory_order_relaxed, memory_scope_work_group)) continue;
# else
    cur_val.a = atom_cmpxchg((global volatile long*)dst, exp_val.a, try_val.a);
# endif
  } while (cur_val.a != exp_val.a);
}
#endif

#if defined(XCHG) || (defined(__NV_CL_C_VERSION) && !defined(CMPXCHG))
__attribute__((always_inline))
inline void atomic_add_global_xchg(global volatile T* dst, T inc)
{
# if (defined(__NV_CL_C_VERSION) && !defined(XCHG)) && (1 == TN)
  asm("{ .reg .f32 t; atom.global.add.f32 t, [%0], %1; }" :: "l"(dst), "f"(inc));
# elif (defined(__NV_CL_C_VERSION) && !defined(XCHG)) && (3 == TN)
  asm("{ .reg .f64 t; atom.global.add.f64 t, [%0], %1; }" :: "l"(dst), "d"(inc));
# else
  union { T f; TA a; } exp_val = { .f = inc }, try_val, cur_val = { .f = 0 };
  do {
    try_val.a = XCHG((global volatile TA*)dst, cur_val.a);
    try_val.f += exp_val.f;
    exp_val.a = XCHG((global volatile TA*)dst, try_val.a);
  } while (cur_val.a != exp_val.a);
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
#if defined(SHARED_P) && (1 < BS)
  local int params[3*BS];
#else
  GLOBAL const int *restrict params = pbase;
#endif
#if defined(SHARED_A)
  local T amk[SM][SK+SHARED_A-1];
#endif
#if defined(SHARED_B)
  local T bkn[SK][SN+SHARED_B-1];
#endif
#if (BM < SM || 1 != BN)
# if defined(PRIVATE_A) && !defined(SHARED_A)
  T amk[SK];
# endif
# if defined(PRIVATE_B) && !defined(SHARED_B)
  T bkn[SK][BN];
# endif
# if !defined(SHARED_C) && (1 < BS)
  T cmn[BM][BN] = {{ 0 }};
# endif
  const int m0 = (idx / NBN) * BM, n0 = (idx % NBN) * BN;
#else
# if defined(PRIVATE_B) && !defined(SHARED_B)
  T bkn[SK];
# endif
# if !defined(SHARED_C) && (1 < BS)
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
  local T cmn[SM][SN+SHARED_C-1];
  for (int m = idx; m < SM; m += SWG) {
    UNROLL(SN)
    for (int n = 0; n < SN; ++n) cmn[m][n] = ZERO;
  }
# endif
# if defined(SHARED_P)
  for (i = idx; i < (3 * batchsize); i += SWG) params[i] = pbase[i] - 1;
# endif
# if (defined(SHARED_C) || defined(SHARED_P)) && !defined(NOBARRIER)
  barrier(CLK_LOCAL_MEM_FENCE);
# endif
# if (defined(SHARED_A) || defined(SHARED_B)) && (NBK < SWG)
  if (NBK <= idx) return;
# endif
  c0 = params[2] - IDXBASE;
  c = cdata + c0;
  UNROLL_FORCE(1)
  for (i = 0; i < batchsize; ++i) {
    const int a0 = params[3*i] - IDXBASE, b0 = params[3*i+1] - IDXBASE;
    const int c1 = ((i + 1) < batchsize ? (params[3*i+5] - IDXBASE) : -1);
#else
# if (defined(SHARED_A) || defined(SHARED_B)) && (NBK < SWG)
  if (NBK > idx)
# endif
  {
    const int a0 = params[0] - IDXBASE, b0 = params[1] - IDXBASE;
    global T *restrict c = cdata + params[2] - IDXBASE;
#endif
    GLOBAL const T *const restrict a = adata + a0;
    GLOBAL const T *const restrict b = bdata + b0;

#if defined(SHARED_A)
    /* transpose A-matrix into local/shared buffer */
    int m = idx;
# if (NBK != SM)
    for (; m < SM; m += NBK)
# endif
    { UNROLL(SK)
      for (int k = 0; k < SK; ++k) amk[m][k] = a[SM*k+m];
    }
#endif

#if defined(SHARED_B)
    UNROLL(SK)
    for (int k = 0; k < SK; ++k) {
      int n = idx;
# if (NBK != SN)
      for (; n < SN; n += NBK)
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
# if (BM < SM || 1 != BN)
        UNROLL(BN)
        for (int bn = 0; bn < BN; ++bn) {
#   if (SN % BN)
          const int n = min(bn + n0, SN - 1);
#   else
          const int n = bn + n0;
#   endif
          bkn[k][bn] = b[SN*k+n];
        }
# else
        bkn[k] = b[SN*k+idx];
# endif
      }
    }
#endif

#if (defined(SHARED_A) || defined(SHARED_B)) && (1 < NBK) && !defined(NOBARRIER)
    /* finish transpose/copy */
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if (BM < SM || 1 != BN)
    /* calculate result-tile using general tiles */
    UNROLL(BM)
    for (int bm = 0; bm < BM; ++bm) {
      const int m = bm + m0;
# if (SM % BM)
      if (m < SM)
# endif
      {
# if defined(PRIVATE_A) && !defined(SHARED_A)
        UNROLL(SK)
        for (int k = 0; k < SK; ++k) amk[k] = a[SM*k+m];
# endif
        UNROLL(BN)
        for (int bn = 0; bn < BN; ++bn) {
          const int n = bn + n0;
# if (SN % BN)
          if (n < SN)
# endif
          {
# if (1 < BS)
#   if defined(SHARED_C)
            T r = cmn[m][n];
#   else
            T r = cmn[bm][bn];
#   endif
# else
            T r = ZERO;
# endif
            UNROLL_FORCE(SK)
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
              bkn[k][bn],
# else
              b[SN*k+n],
# endif
              r);
# if (1 < BS)
#   if defined(SHARED_C)
            cmn[m][n] = r;
#   else
            cmn[bm][bn] = r;
#   endif
# else
#   if defined(ATOMIC_INC_NZ)
            if (ZERO != r)
#   endif
            ATOMIC_ADD_GLOBAL(&c[SM*n+m], r);
# endif
          }
        }
      }
    }
#else
    /* calculate result-tile using columns */
    UNROLL_SM
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
      UNROLL_FORCE(SK)
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
#   if defined(ATOMIC_INC_NZ)
      if (ZERO != r)
#   endif
      ATOMIC_ADD_GLOBAL(&c[SM*idx+m], r);
# endif
    }
#endif

#if (1 < BS)
# if defined(TRACK_C)
    if (c0 != c1)
# endif
# if (BM < SM || 1 != BN)
    { /* atomically commit C-tile to global memory */
      UNROLL(BM)
      for (int bm = 0; bm < BM; ++bm) {
        const int m = bm + m0;
#   if (SM % BM)
        if (m < SM)
#   endif
        {
          UNROLL(BN)
          for (int bn = 0; bn < BN; ++bn) {
            const int n = bn + n0;
#   if (SN % BN)
            if (n < SN)
#   endif
            {
#   if defined(SHARED_C)
              local T *restrict r = &cmn[m][n];
#   else
              private T *restrict r = &cmn[bm][bn];
#   endif
#   if defined(ATOMIC_INC_NZ)
              if (ZERO != *r)
#   endif
              {
                ATOMIC_ADD_GLOBAL(&c[SM*n+m], *r);
                *r = ZERO; /* reset */
              }
            }
          }
        }
      }
# else
    { /* atomically commit C-column to global memory */
      int m = 0;
#   if defined(ATOMIC_ADD2_GLOBAL)
      for (; m < (SM - 1); m += 2) {
#     if defined(SHARED_C)
        local T *restrict r0 = &cmn[m+0][idx];
        local T *restrict r1 = &cmn[m+1][idx];
#     else
        private T *restrict r0 = &cmn[m+0];
        private T *restrict r1 = &cmn[m+1];
#     endif
#     if defined(ATOMIC_INC_NZ)
        if (ZERO != *r0 && ZERO != *r1)
#     endif
        {
          const float2 r2 = (float2)(*r0, *r1);
          ATOMIC_ADD2_GLOBAL(&c[SM*idx+m], r2);
          *r0 = *r1 = ZERO; /* reset */
        }
      }
#   endif
#   if !defined(ATOMIC_ADD2_GLOBAL) || (SM & 1)
      for (; m < SM; ++m) {
#     if defined(SHARED_C)
        local T *restrict r = &cmn[m][idx];
#     else
        private T *restrict r = cmn + m;
#     endif
#     if defined(ATOMIC_INC_NZ)
        if (ZERO != *r)
#     endif
        {
          ATOMIC_ADD_GLOBAL(&c[SM*idx+m], *r);
          *r = ZERO; /* reset */
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
