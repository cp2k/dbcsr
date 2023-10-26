/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/
#if (200 /*CL_VERSION_2_0*/ <= __OPENCL_VERSION__) || defined(__NV_CL_C_VERSION)
#  define UNROLL_FORCE(N) __attribute__((opencl_unroll_hint(N)))
#else
#  define UNROLL_FORCE(N)
#endif

#define MIN(A, B) ((A) < (B) ? (A) : (B))
#define MAX(A, B) ((A) < (B) ? (B) : (A))

#if !defined(LU) || (-1 == LU) || (1 < LU)
#  define UNROLL_OUTER(N)
#  define UNROLL(N)
#else
#  if (1 == LU)
#    define UNROLL_OUTER(N) UNROLL_FORCE(1)
#  else
#    define UNROLL_OUTER(N) UNROLL_FORCE(N)
#  endif
#  define UNROLL(N) UNROLL_FORCE(N)
#endif

#if !defined(AL) || (SM != SN) || (SM != BM) || (SN != SK) || (1 == BS)
#  define ADX(M, K) adata[SM * K + M + a0]
#  define BDX(K, N) bdata[SN * K + N + b0]
#  define CDX(M, N) cdata[SM * N + M + c0]
#else
#  define ADX(M, K) adata[SK * M + K + a0]
#  define BDX(K, N) bdata[SK * N + K + b0]
#  define CDX(M, N) cdata[SN * M + N + c0]
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

#if (1 == TN)
#  define ZERO 0.f
#elif (3 == TN)
#  define ZERO 0.0
#else
#  define ZERO 0
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

#define GLOBAL_VOLATILE(A) global volatile A
#if defined(ATOMIC_PROTOTYPES) || defined(__opencl_c_ext_fp64_global_atomic_add)
#  if defined(__opencl_c_ext_fp64_global_atomic_add)
#    undef ATOMIC_ADD_GLOBAL
#    if defined(TF)
#      define ATOMIC_ADD_GLOBAL(A, B) \
        atomic_fetch_add_explicit((GLOBAL_VOLATILE(TF)*)A, B, memory_order_relaxed, memory_scope_work_group)
#    else
#      define ATOMIC_ADD_GLOBAL(A, B) atomic_add(A, B)
#    endif
#  elif (2 < ATOMIC_PROTOTYPES) && defined(TF)
#    undef ATOMIC_ADD_GLOBAL
#    define ATOMIC_ADD_GLOBAL(A, B) \
      __opencl_atomic_fetch_add((GLOBAL_VOLATILE(TF)*)A, B, memory_order_relaxed, memory_scope_work_group)
#  else
#    if defined(TF) && (!defined(ATOMIC_PROTOTYPES) || 1 < ATOMIC_PROTOTYPES)
__attribute__((overloadable)) T atomic_fetch_add_explicit(GLOBAL_VOLATILE(TF) *, T, memory_order, memory_scope);
#    else
__attribute__((overloadable)) T atomic_add(GLOBAL_VOLATILE(T) *, T);
#    endif
#  endif
#endif
#define ACCUMULATE(A, B) ATOMIC_ADD_GLOBAL(A, B)

#if !defined(cl_intel_global_float_atomics) || (1 != TN)
#  if defined(ATOMIC32_ADD64)
__attribute__((always_inline)) inline void atomic32_add64_global(GLOBAL_VOLATILE(double) * dst, double inc) {
  *dst += inc; /* TODO */
}
#  endif

#  if defined(CMPXCHG)
__attribute__((always_inline)) inline void atomic_add_global_cmpxchg(GLOBAL_VOLATILE(T) * dst, T inc) {
#    if !defined(ATOMIC32_ADD64)
  union {
    T f;
    TA a;
  } exp_val, try_val, cur_val = {.f = *dst};
  do {
    exp_val.a = cur_val.a;
    try_val.f = exp_val.f + inc;
#      if defined(TA2)
    if (0 == atomic_compare_exchange_weak_explicit((GLOBAL_VOLATILE(TA2)*)dst, &cur_val.a, try_val.a, memory_order_relaxed,
               memory_order_relaxed, memory_scope_work_group))
      continue;
#      else
    cur_val.a = CMPXCHG((GLOBAL_VOLATILE(TA)*)dst, exp_val.a, try_val.a);
#      endif
  } while (cur_val.a != exp_val.a);
#    else
  atomic32_add64_global(dst, inc);
#    endif
}
#  endif

#  if defined(ATOMIC_ADD2_GLOBAL) && (1 == TN)
__attribute__((always_inline)) inline void atomic_add_global_cmpxchg2(GLOBAL_VOLATILE(float) * dst, float2 inc) {
  union {
    float2 f;
    long a;
  } exp_val, try_val, cur_val = {.f = (float2)(dst[0], dst[1])};
  do {
    exp_val.a = cur_val.a;
    try_val.f = exp_val.f + inc;
#    if defined(TA2)
    if (0 == atomic_compare_exchange_weak_explicit((GLOBAL_VOLATILE(atomic_long)*)dst, &cur_val.a, try_val.a, memory_order_relaxed,
               memory_order_relaxed, memory_scope_work_group))
      continue;
#    else
    cur_val.a = atom_cmpxchg((GLOBAL_VOLATILE(long)*)dst, exp_val.a, try_val.a);
#    endif
  } while (cur_val.a != exp_val.a);
}
#  endif

#  if defined(XCHG) || (defined(__NV_CL_C_VERSION) && !defined(CMPXCHG) && !defined(ATOMIC_PROTOTYPES))
__attribute__((always_inline)) inline void atomic_add_global_xchg(GLOBAL_VOLATILE(T) * dst, T inc) {
#    if !defined(ATOMIC32_ADD64)
#      if (defined(__NV_CL_C_VERSION) && !defined(XCHG)) && (1 == TN)
  asm("{ .reg .f32 t; atom.global.add.f32 t, [%0], %1; }" ::"l"(dst), "f"(inc));
#      elif (defined(__NV_CL_C_VERSION) && !defined(XCHG)) && (3 == TN)
  asm("{ .reg .f64 t; atom.global.add.f64 t, [%0], %1; }" ::"l"(dst), "d"(inc));
#      else
  union {
    T f;
    TA a;
  } exp_val = {.f = inc}, try_val, cur_val = {/*.f = ZERO*/ .a = 0};
  do {
#        if defined(TA2)
    try_val.a = atomic_exchange_explicit((GLOBAL_VOLATILE(TA2)*)dst, cur_val.a, memory_order_relaxed, memory_scope_work_group);
#        else
    try_val.a = XCHG((GLOBAL_VOLATILE(TA)*)dst, cur_val.a);
#        endif
    try_val.f += exp_val.f;
#        if defined(TA2)
    exp_val.a = atomic_exchange_explicit((GLOBAL_VOLATILE(TA2)*)dst, try_val.a, memory_order_relaxed, memory_scope_work_group);
#        else
    exp_val.a = XCHG((GLOBAL_VOLATILE(TA)*)dst, try_val.a);
#        endif
  } while (cur_val.a != exp_val.a);
#      endif
#    else
  atomic32_add64_global(dst, inc);
#    endif
}
#  endif
#endif


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
  for (int n = idx; n < SN; n += SWG) {
    UNROLL_FORCE(SM)
    for (int m = 0; m < SM; ++m) cnm[n][m] = ZERO;
  }
#  elif (BM < SM || 1 != BN)
#    if (1 != BN)
  UNROLL(BN)
  for (int bn = 0; bn < BN; ++bn)
#    endif
  {
    UNROLL_FORCE(BM)
    for (int bm = 0; bm < BM; ++bm) CNM(bn, bm) = ZERO;
  }
#  else
  UNROLL_FORCE(SM)
  for (int m = 0; m < SM; ++m) cnm[m] = ZERO;
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
        for (int k = 0; k < SK; ++k) amk[m][k] = ADX(m, k);
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
        for (int k = 0; k < SK; ++k) bnk[n][k] = BDX(k, n);
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
      for (int k = 0; k < SK; ++k) {
#  if (BM < SM || 1 != BN)
        int bn = 0;
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
      for (int bm = 0, m = m0; bm < BM && m < SM; m = ++bm + m0)
#    else
      for (int bm = 0, m = m0; bm < BM; m = ++bm + m0)
#    endif
      { /* general BK, A in registers */
        int bn = 0;
        UNROLL_FORCE(SK)
        for (int k = 0; k < SK; ++k) amk[k] = ADX(m, k);
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
            for (int k = 0; k < SK; ++k) {
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
      for (int k = 0; k < SK; ++k) {
#    if (SN % BN) || !defined(REG_B) || (defined(SLM_C) && (1 < BS)) || (1 == BS) || (1 != BN)
        int bn = 0;
#    endif
#    if defined(REG_A) && !defined(SLM_A)
        UNROLL_FORCE(BM)
        for (int bm = 0; bm < BM; ++bm) amk[bm] = ADX(bm + m0, k);
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
            for (int bm = 0, m = m0; bm < BM && m < SM; m = ++bm + m0)
#    else
            for (int bm = 0, m = m0; bm < BM; m = ++bm + m0)
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
            for (int bm = 0; bm < BM; ++bm) {
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
    int bn = 0;
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
        for (int bm = 0, m = m0; bm < BM && m < SM; m = ++bm + m0)
#    else
        for (int bm = 0, m = m0; bm < BM; m = ++bm + m0)
#    endif
        {
#    if defined(SLM_C) && (1 < BS)
          const int mc = m, nc = n;
#    else
          const int mc = bm, nc = bn;
#    endif
#    if defined(REG_B)
          const int nb = bn;
#    else
          const int nb = n;
#    endif
          UNROLL_FORCE(SK)
          for (int k = 0; k < SK; ++k) CNM(nc, mc) = MAD(AMK(m, k), BNK(nb, k), CNM(nc, mc));
        }
#    if (1 == BS)
        UNROLL(BM)
        for (int bm = 0; bm < BM; ++bm) {
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
#else
    { /* calculate result-tile using columns */
#  if (1 == BS)
      T cnm[UM] = {ZERO}; /* column-block */
#  endif
#  if (1 == BK)
      UNROLL_OUTER(SK)
      for (int k = 0; k < SK; ++k) {
        const T b = BNK(idx, k);
#    if defined(SLM_A)
#      if (WRK != SM)
        for (int m = idx; m < SM; m += WRK) amk[m] = ADX(m, k);
#      else
        amk[idx] = ADX(idx, k);
#      endif
#    elif defined(REG_A)
        UNROLL_FORCE(SM)
        for (int m = 0; m < SM; ++m) amk[m] = ADX(m, k);
#    endif
#    if defined(BARRIER) && (MAX(1, SGS) < SWG) && defined(SLM_A)
        BARRIER(CLK_LOCAL_MEM_FENCE);
#    endif
#    if (WRK == SM) && (SM <= SGS || SM <= SWG) && !defined(SLM_A) && !defined(REG_A)
        const T a = AMK(idx, k);
#    endif
        UNROLL_FORCE(SM)
        for (int m = 0; m < SM; ++m) {
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
      for (int m = 0; m < SM; ++m) {
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
      int m = 0, u;
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
          for (int k = 0; k < SK; ++k) amk[k] = ADX(um, k);
#    endif
          UNROLL_FORCE(SK)
          for (int k = 0; k < SK; ++k) {
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
        for (int k = 0; k < SK; ++k) amk[k] = ADX(um, k);
#      endif
        UNROLL_FORCE(SK)
        for (int k = 0; k < SK; ++k) {
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
      int bn = 0;
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
          for (int bm = 0, m = m0; bm < BM && m < SM; m = ++bm + m0)
#    else
          for (int bm = 0, m = m0; bm < BM; m = ++bm + m0)
#    endif
          {
#    if defined(SLM_C)
            const int mc = m, nc = n;
#    else
            const int mc = bm, nc = bn;
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
      int m = 0;
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
  }
}
