/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/
#include "../../common/opencl_atomics.h"

#if !defined(AL) || (SM != SN) || (SM != BM) || (SN != SK) || (1 == BS)
#  define ADX(M, K) adata[SM * K + M + a0] /* transposed */
#  define BDX(K, N) bdata[SN * K + N + b0] /* linear */
#  define CDX(M, N) cdata[SM * N + M + c0] /* transposed */
#else
#  define ADX(M, K) adata[SK * M + K + a0] /* linear */
#  define BDX(K, N) bdata[SK * N + K + b0] /* transposed */
#  define CDX(M, N) cdata[SN * M + N + c0] /* linear */
#endif

#if defined(SLM_A)
#  if (1 != BK || BM < SM || 1 != BN)
#    define AMK(M, K) amk[M][K]
#  else
#    define AMK(M, K) amk[M]
#  endif
#elif defined(REG_A)
#  if (1 != BK)
#    define AMK(M, K) amk[K]
#  else
#    define AMK(M, K) amk[M]
#  endif
#else
#  define AMK(M, K) ADX(M, K)
#endif
#if defined(SLM_B)
#  define BNK(N, K) bnk[N][K]
#elif defined(REG_B)
#  if (BM < SM && 1 != BN)
#    define BNK(N, K) bnk[N][K]
#  else
#    define BNK(N, K) bnk[K]
#  endif
#else
#  define BNK(N, K) BDX(K, N)
#endif
#if (1 < BS) && (defined(SLM_C) || (BM < SM && 1 != BN))
#  define CNM(N, M) cnm[N][M]
#else
#  define CNM(N, M) cnm[M]
#endif

#if !defined(SINT) /* covers matrix shape */
#  define SINT signed char
#endif

#if defined(SLM_P) && (1 < BS)
#  define IDXBASE 0
#else
#  define IDXBASE 1
#endif
#if !defined(REPEAT)
#  define REPEAT 1
#endif

#define NBM ((SM + BM - 1) / BM)
#define NBN ((SN + BN - 1) / BN)
#define WRK (NBM * NBN)

#define UM (SM / BK)
#define VM (SM % UM)


__attribute__((reqd_work_group_size(SWG, 1, 1)))
#if (0 < SGS)
__attribute__((intel_reqd_sub_group_size(SGS)))
#endif
kernel void
FN(global T* restrict cdata, GLOBAL const T* restrict adata, GLOBAL const T* restrict bdata,
#if (1 < BS)
  GLOBAL const int* restrict param_stack, int stack_size, int bs) {
  const int gid = get_group_id(0), idx = get_local_id(0);
#else
  GLOBAL const int* restrict param_stack) {
  const int gid = get_group_id(0), idx = get_local_id(0), bs = 1;
#endif
  /* indexes given by param_stack are one-based (Fortran) */
  GLOBAL const int* restrict pbase = param_stack + gid * (3 * bs);
#if defined(SLM_P) && (1 < BS)
  local int params[3 * BS]; /* bs <= BS */
#else
  GLOBAL const int* restrict params = pbase;
#endif
#if defined(SLM_A)
#  if (1 != BK || BM < SM || 1 != BN)
  local T amk[SM][SK + SLM_A - 1];
#  else
  local T amk[SM];
#  endif
#elif defined(REG_A) && (1 != BK)
  T amk[SK];
#endif
#if defined(SLM_B)
  local T bnk[SN][SK + SLM_B - 1]; /* tile */
#endif
#if (BM < SM || 1 != BN)
#  if defined(REG_A) && !defined(SLM_A) && (1 == BK)
  T amk[BM];
#  endif
#  if defined(REG_B) && !defined(SLM_B)
#    if (1 != BN)
  T bnk[BN][SK]; /* rows */
#    else
  T bnk[SK]; /* row */
#    endif
#  endif
#  if !defined(SLM_C) && (1 < BS)
#    if (1 != BN)
  T cnm[BN][BM]; /* general tile */
#    else
  T cnm[BM]; /* column-block */
#    endif
#  endif
  const int m0 = (idx / NBN) * BM, n0 = (idx % NBN) * BN;
#else
#  if defined(REG_A) && !defined(SLM_A) && (1 == BK)
  T amk[SM];
#  endif
#  if defined(REG_B) && !defined(SLM_B)
  T bnk[SK]; /* row */
#  endif
#  if !defined(SLM_C) && (1 < BS)
  T cnm[SM]; /* column */
#  endif
#endif
#if defined(TRACK_B) && (1 < BS) && defined(REG_B) && !defined(SLM_B)
  int b1 = -1;
#endif

#if (1 < BS)
  /* intra-kernel mini-batch of SMMs */
  const int batchsize = min(bs, stack_size - bs * gid);
  int c0;
#  if defined(SLM_C)
  local T cnm[SN][SM + SLM_C - 1]; /* tile in SLM */
  for (SINT n = (SINT)idx; n < SN; n += SWG) {
    UNROLL_FORCE(SM)
    for (SINT m = 0; m < SM; ++m) cnm[n][m] = ZERO;
  }
#  elif (BM < SM || 1 != BN)
#    if (1 != BN)
  UNROLL(BN)
  for (SINT bn = 0; bn < BN; ++bn)
#    endif
  {
    UNROLL_FORCE(BM)
    for (SINT bm = 0; bm < BM; ++bm) CNM(bn, bm) = ZERO;
  }
#  else
  UNROLL_FORCE(SM)
  for (SINT m = 0; m < SM; ++m) cnm[m] = ZERO;
#  endif
#  if defined(SLM_P)
  UNROLL_FORCE(3 * BS)
  for (int i = idx; i < (3 * batchsize); i += SWG) params[i] = pbase[i] - 1;
#  endif
#  if defined(BARRIER) && (MAX(1, SGS) < SWG) && (defined(SLM_C) || defined(SLM_P))
  BARRIER(CLK_LOCAL_MEM_FENCE);
#  endif
#  if (WRK < SWG)
  if (WRK <= idx) return; /* WRK <= idx */
#  endif
  c0 = params[2] - IDXBASE;
#  if defined(BSC) && (1 != BK) && (1 != UM)
  UNROLL_OUTER(REPEAT * BS)
#  else
  UNROLL_FORCE(1)
#  endif
#  if (1 < REPEAT)
  for (int item = 0; item < (REPEAT * batchsize); ++item) {
    const int i = item % batchsize;
#  else
  for (int item = 0; item < (REPEAT * batchsize); ++item) {
    const int i = item;
#  endif
    const int a0 = params[3 * i] - IDXBASE, b0 = params[3 * i + 1] - IDXBASE;
    const int c1 = ((i + 1) < batchsize ? (params[3 * i + 5] - IDXBASE) : -1);
#else
#  if (WRK < SWG)
  if (WRK > idx) /* WRK > idx */
#  endif
  {
    const int a0 = params[0] - IDXBASE, b0 = params[1] - IDXBASE, c0 = params[2] - IDXBASE;
#endif

#if defined(SLM_A) && (1 != BK || BM < SM || 1 != BN)
    { /* copy or transpose A-matrix into SLM */
      int m = idx;
#  if (WRK != SM)
      for (; m < SM; m += WRK)
#  endif
      {
        UNROLL_FORCE(SK)
        for (SINT k = 0; k < SK; ++k) amk[m][k] = ADX(m, k);
      }
    }
#endif

#if defined(SLM_B)
    { /* copy or transpose B-matrix into SLM */
      int n = idx;
#  if (WRK != SN)
      for (; n < SN; n += WRK)
#  endif
      {
        UNROLL(SK)
        for (SINT k = 0; k < SK; ++k) bnk[n][k] = BDX(k, n);
      }
    }
#elif defined(REG_B)
#  if defined(TRACK_B) && (1 < BS)
    if (b0 != b1) {
      b1 = b0;
#  else
    { /* copy or transpose B-matrix into registers */
#  endif
      UNROLL(SK)
      for (SINT k = 0; k < SK; ++k) {
#  if (BM < SM || 1 != BN)
        SINT bn = 0;
#    if (1 != BN)
        UNROLL_FORCE(BN)
        for (; bn < BN; ++bn)
#    endif
        {
#    if (SN % BN)
          const int n = min(bn + n0, SN - 1);
#    else
          const int n = bn + n0;
#    endif
          BNK(bn, k) = BDX(k, n);
        }
#  else
        bnk[k] = BDX(k, idx);
#  endif
      }
    }
#endif

#if defined(BARRIER) && (MAX(1, SGS) < SWG) && (defined(SLM_B) || ((1 != BK || BM < SM || 1 != BN) && defined(SLM_A)))
    /* finish transpose/copy */
    BARRIER(CLK_LOCAL_MEM_FENCE);
#endif

#if (BM < SM || 1 != BN)
    { /* calculate result-tile using general tiles */
#  if defined(REG_A) && !defined(SLM_A) && (1 != BK)
#    if (1 == BS)
      T cnm[BN] = {ZERO}; /* row */
#    endif
      UNROLL(BM)
#    if (SM % BM)
      for (SINT bm = 0, m = m0; bm < BM && m < SM; m = ++bm + m0)
#    else
      for (SINT bm = 0, m = m0; bm < BM; m = ++bm + m0)
#    endif
      { /* general BK, A in registers */
        SINT bn = 0;
        UNROLL_FORCE(SK)
        for (SINT k = 0; k < SK; ++k) amk[k] = ADX(m, k);
#    if (1 != BN)
        UNROLL(BN)
        for (; bn < BN; ++bn)
#    endif
        {
#    if (SN % BN) || (defined(SLM_C) && (1 < BS)) || !defined(REG_B)
          const int n = bn + n0;
#    endif
#    if (SN % BN)
          if (n < SN) /* n < SN */
#    endif
          {
#    if defined(SLM_C) && (1 < BS)
            const int mc = m, nc = n;
#    elif (1 < BS)
            const int mc = bm, nc = bn;
#    else
            const int mc = bn, nc = idx;
#    endif
            UNROLL_FORCE(SK)
            for (SINT k = 0; k < SK; ++k) {
              CNM(nc, mc) = MAD(AMK(m, k),
#    if defined(REG_B)
                BNK(bn, k),
#    else
                BNK(n, k),
#    endif
                CNM(nc, mc));
            }
          }
        }
#    if (1 == BS)
        bn = 0;
#      if (1 != BN)
        UNROLL(BN)
        for (; bn < BN; ++bn)
#      endif
        {
#      if defined(ATOMIC_INC_NZ)
          if (ZERO != CNM(idx, bn))
#      endif
          {
            ACCUMULATE(&CDX(m, bn + n0), CNM(idx, bn));
            CNM(idx, bn) = ZERO; /* reset */
          }
        }
#    endif
      }
#  elif (1 == BK)
#    if (1 == BS)
      T cnm[BM] = {ZERO}; /* column-block */
#    endif
      UNROLL(SK)
      for (SINT k = 0; k < SK; ++k) {
#    if (SN % BN) || !defined(REG_B) || (defined(SLM_C) && (1 < BS)) || (1 == BS) || (1 != BN)
        SINT bn = 0;
#    endif
#    if defined(REG_A) && !defined(SLM_A)
        UNROLL_FORCE(BM)
        for (SINT bm = 0; bm < BM; ++bm) amk[bm] = ADX(bm + m0, k);
#    endif
#    if (1 != BN)
        UNROLL(BN)
        for (; bn < BN; ++bn)
#    endif
        { /* BK=1 */
#    if (SN % BN) || !defined(REG_B) || (defined(SLM_C) && (1 < BS)) || (1 == BS)
          const int n = bn + n0;
#    endif
#    if (SN % BN)
          if (n < SN) /* n < SN */
#    endif
          {
#    if defined(REG_B)
            const T b = BNK(bn, k);
#    else
            const T b = BNK(n, k);
#    endif
            UNROLL_FORCE(BM)
#    if (SM % BM)
            for (SINT bm = 0, m = m0; bm < BM && m < SM; m = ++bm + m0)
#    else
            for (SINT bm = 0, m = m0; bm < BM; m = ++bm + m0)
#    endif
            {
#    if defined(REG_A) && !defined(SLM_A)
              const T a = AMK(bm, k);
#    else
              const T a = AMK(m, k);
#    endif
#    if defined(SLM_C) && (1 < BS)
              CNM(n, m) = MAD(a, b, CNM(n, m));
#    else
              CNM(bn, bm) = MAD(a, b, CNM(bn, bm));
#    endif
            }
#    if (1 == BS)
            UNROLL(BM)
            for (SINT bm = 0; bm < BM; ++bm) {
#      if defined(ATOMIC_INC_NZ)
              if (ZERO != CNM(idx, bm))
#      endif
              {
                ACCUMULATE(&CDX(bm + m0, n), CNM(idx, bm));
                CNM(idx, bm) = ZERO; /* reset */
              }
            }
#    endif
          }
        }
      }
#  else /* general BK */
    SINT bn = 0;
#    if (1 != BN)
    UNROLL(BN)
    for (; bn < BN; ++bn)
#    endif
    {
#    if (SN % BN) || !defined(REG_B) || (defined(SLM_C) && (1 < BS)) || (1 == BS)
      const int n = bn + n0;
#    endif
#    if (SN % BN)
      if (n < SN) /* n < SN */
#    endif
      { /* general BK */
#    if (1 == BS)
        T cnm[BM] = {ZERO}; /* column-block */
#    endif
        UNROLL(BM)
#    if (SM % BM)
        for (SINT bm = 0, m = m0; bm < BM && m < SM; m = ++bm + m0)
#    else
        for (SINT bm = 0, m = m0; bm < BM; m = ++bm + m0)
#    endif
        {
#    if defined(SLM_C) && (1 < BS)
#      if (1 < BS) && (defined(SLM_C) || (BM < SM && 1 != BN))
          const int mc = m, nc = n;
#      else
          const int mc = m;
#      endif
#    else
#      if (1 < BS) && (defined(SLM_C) || (BM < SM && 1 != BN))
          const int mc = bm, nc = bn;
#      else
          const int mc = bm;
#      endif
#    endif
#    if defined(REG_B)
          const int nb = bn;
#    else
          const int nb = n;
#    endif
          UNROLL_FORCE(SK)
          for (SINT k = 0; k < SK; ++k) CNM(nc, mc) = MAD(AMK(m, k), BNK(nb, k), CNM(nc, mc));
        }
#    if (1 == BS)
        UNROLL(BM)
        for (SINT bm = 0; bm < BM; ++bm) {
#      if defined(ATOMIC_INC_NZ)
          if (ZERO != CNM(idx, bm))
#      endif
          {
            ACCUMULATE(&CDX(bm + m0, n), CNM(idx, bm));
            CNM(idx, bm) = ZERO; /* reset */
          }
        }
#    endif
      }
    }
#  endif
    }
#else /* BM == SM && 1 == BN */
    { /* calculate result-tile using columns */
#  if (1 == BS)
      T cnm[UM] = {ZERO}; /* column-block */
#  endif
#  if (1 == BK)
      UNROLL_OUTER(SK)
      for (SINT k = 0; k < SK; ++k) {
        const T b = BNK(idx, k);
#    if defined(SLM_A)
#      if (WRK != SM)
        for (SINT m = (SINT)idx; m < SM; m += WRK) amk[m] = ADX(m, k);
#      else
        amk[idx] = ADX(idx, k);
#      endif
#    elif defined(REG_A)
        UNROLL_FORCE(SM)
        for (SINT m = 0; m < SM; ++m) amk[m] = ADX(m, k);
#    endif
#    if defined(BARRIER) && (MAX(1, SGS) < SWG) && defined(SLM_A)
        BARRIER(CLK_LOCAL_MEM_FENCE);
#    endif
#    if (WRK == SM) && (SM <= SGS || SM <= SWG) && !defined(SLM_A) && !defined(REG_A)
        const T a = AMK(idx, k);
#    endif
        UNROLL_FORCE(SM)
        for (SINT m = 0; m < SM; ++m) {
#    if (200 /*CL_VERSION_2_0*/ <= __OPENCL_VERSION__) && !defined(SLM_A) && !defined(REG_A) && (WRK == SM) && \
      (SM <= SGS || SM <= SWG)
#      if (SM <= SGS)
          /* size of subgroup is sufficient */
          CNM(idx, m) = MAD(sub_group_broadcast(a, m), b, CNM(idx, m));
#      else
          /* size of workgroup is sufficient */
          CNM(idx, m) = MAD(work_group_broadcast(a, m), b, CNM(idx, m));
#      endif
#    else
          CNM(idx, m) = MAD(AMK(m, k), b, CNM(idx, m)); /* fallback */
#    endif
        }
#    if defined(BARRIER) && (MAX(1, SGS) < SWG) && defined(SLM_A)
        BARRIER(CLK_LOCAL_MEM_FENCE);
#    endif
      }
#    if (1 == BS)
      UNROLL(SM)
      for (SINT m = 0; m < SM; ++m) {
#      if defined(ATOMIC_INC_NZ)
        if (ZERO != CNM(idx, m))
#      endif
        {
          ACCUMULATE(&CDX(m, idx), CNM(idx, m));
          CNM(idx, m) = ZERO; /* reset */
        }
      }
#    endif
#  else
      SINT m = 0, u;
#    if (1 == UM)
      UNROLL_OUTER(SM)
#    endif
      for (; m < (SM - UM + 1); m += UM) {
        u = 0;
#    if (1 < UM)
        UNROLL(UM)
        for (; u < UM; ++u)
#    endif
        {
          const int um = u + m;
#    if (1 < BS)
          const int vm = um;
#    else
          const int vm = u;
#    endif
#    if defined(REG_A) && !defined(SLM_A)
          UNROLL_FORCE(SK)
          for (SINT k = 0; k < SK; ++k) amk[k] = ADX(um, k);
#    endif
          UNROLL_FORCE(SK)
          for (SINT k = 0; k < SK; ++k) {
            CNM(idx, vm) = MAD(AMK(um, k), BNK(idx, k), CNM(idx, vm));
          }
        }
#    if (1 == BS)
        u = 0;
#      if (1 < UM)
        UNROLL(UM)
        for (; u < UM; ++u)
#      endif
#      if defined(ATOMIC_INC_NZ)
          if (ZERO != CNM(idx, u))
#      endif
          {
            ACCUMULATE(&CDX(u + m, idx), CNM(idx, u));
            CNM(idx, u) = ZERO; /* reset */
          }
#    endif
      }
#    if (0 < VM)
      /* calculate remainder */
      u = 0;
#      if (1 < VM)
      UNROLL(VM)
      for (; u < VM; ++u)
#      endif
      {
        const int um = u + m;
#      if (1 < BS)
        const int vm = um;
#      else
        const int vm = u;
#      endif
#      if defined(REG_A) && !defined(SLM_A)
        UNROLL_FORCE(SK)
        for (SINT k = 0; k < SK; ++k) amk[k] = ADX(um, k);
#      endif
        UNROLL_FORCE(SK)
        for (SINT k = 0; k < SK; ++k) {
          CNM(idx, vm) = MAD(AMK(um, k), BNK(idx, k), CNM(idx, vm));
        }
      }
#      if (1 == BS)
      u = 0;
#        if (1 < VM)
      UNROLL(VM)
      for (; u < VM; ++u)
#        endif
#        if defined(ATOMIC_INC_NZ)
        if (ZERO != CNM(idx, u))
#        endif
        {
          ACCUMULATE(&CDX(u + m, idx), CNM(idx, u));
          CNM(idx, u) = ZERO; /* reset */
        }
#      endif
#    endif
#  endif
    }
#endif

#if (1 < BS)
#  if defined(TRACK_C)
    if (c0 != c1)
#  endif
#  if (BM < SM || 1 != BN)
    { /* atomically commit C-tile to global memory */
      SINT bn = 0;
#    if (1 != BN)
      UNROLL(BN)
      for (; bn < BN; ++bn)
#    endif
      {
        const int n = bn + n0;
#    if (SN % BN)
        if (n < SN) /* n < SN */
#    endif
        {
          UNROLL_FORCE(BM)
#    if (SM % BM)
          for (SINT bm = 0, m = m0; bm < BM && m < SM; m = ++bm + m0)
#    else
          for (SINT bm = 0, m = m0; bm < BM; m = ++bm + m0)
#    endif
          {
#    if defined(SLM_C)
#      if (1 < BS) && (defined(SLM_C) || (BM < SM && 1 != BN))
            const int mc = m, nc = n;
#      else
            const int mc = m;
#      endif
#    else
#      if (1 < BS) && (defined(SLM_C) || (BM < SM && 1 != BN))
            const int mc = bm, nc = bn;
#      else
            const int mc = bm;
#      endif
#    endif
#    if defined(ATOMIC_INC_NZ)
            if (ZERO != CNM(nc, mc))
#    endif
            {
              ACCUMULATE(&CDX(m, n), CNM(nc, mc));
              CNM(nc, mc) = ZERO; /* reset */
            }
          }
        }
      }
#  else
    { /* atomically commit C-column to global memory */
      SINT m = 0;
#    if defined(ATOMIC_ADD2_GLOBAL)
      for (; m < (SM - 1); m += 2) {
#      if defined(ATOMIC_INC_NZ)
        if (ZERO != CNM(idx, m) && ZERO != CNM(idx, m + 1))
#      endif
        {
          const float2 r2 = (float2)(CNM(idx, m), CNM(idx, m + 1));
          ATOMIC_ADD2_GLOBAL(&CDX(m, idx), r2);
          CNM(idx, m) = CNM(idx, m + 1) = ZERO; /* reset */
        }
      }
#    else
      UNROLL(SM)
#    endif
#    if !defined(ATOMIC_ADD2_GLOBAL) || (SM & 1)
      for (; m < SM; ++m) {
#      if defined(ATOMIC_INC_NZ)
        if (ZERO != CNM(idx, m))
#      endif
        {
          ACCUMULATE(&CDX(m, idx), CNM(idx, m));
          CNM(idx, m) = ZERO; /* reset */
        }
      }
#    endif
#  endif
      /* next iteration */
      c0 = c1;
    }
#endif
#if defined(BARRIER) && (MAX(1, SGS) < SWG) && defined(SLM_A) && (BM <= SM || 1 != BN || 1 != BK)
    BARRIER(CLK_LOCAL_MEM_FENCE);
#endif
  }
}
