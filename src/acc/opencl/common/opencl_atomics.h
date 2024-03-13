/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/
#ifndef OPENCL_ATOMICS_H
#define OPENCL_ATOMICS_H

#include "opencl_common.h"

#if (2 == TAN /*c_dbcsr_acc_opencl_atomic_fp_64*/)
#  if !defined(T)
#    define T double
#  endif
#  define ZERO 0.0
#elif (1 == TAN /*c_dbcsr_acc_opencl_atomic_fp_32*/)
#  if !defined(T)
#    define T float
#  endif
#  define ZERO 0.f
#elif defined(T) /*c_dbcsr_acc_opencl_atomic_fp_no*/
#  define ZERO 0
#endif

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


#if !defined(cl_intel_global_float_atomics) || (2 == TAN /*c_dbcsr_acc_opencl_atomic_fp_64*/)
#  if defined(ATOMIC32_ADD64)
__attribute__((always_inline)) inline void atomic32_add64_global(GLOBAL_VOLATILE(double) * dst, double inc) {
  *dst += inc; /* TODO */
}
#  endif
#endif


#if !defined(cl_intel_global_float_atomics) || (2 == TAN /*c_dbcsr_acc_opencl_atomic_fp_64*/)
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
#endif


#if !defined(cl_intel_global_float_atomics) || (2 == TAN /*c_dbcsr_acc_opencl_atomic_fp_64*/)
#  if defined(ATOMIC_ADD2_GLOBAL) && (1 == TAN /*c_dbcsr_acc_opencl_atomic_fp_32*/)
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
#endif


#if !defined(cl_intel_global_float_atomics) || (2 == TAN /*c_dbcsr_acc_opencl_atomic_fp_64*/)
#  if defined(XCHG) || (defined(__NV_CL_C_VERSION) && !defined(CMPXCHG) && !defined(ATOMIC_PROTOTYPES))
__attribute__((always_inline)) inline void atomic_add_global_xchg(GLOBAL_VOLATILE(T) * dst, T inc) {
#    if !defined(ATOMIC32_ADD64)
#      if (defined(__NV_CL_C_VERSION) && !defined(XCHG)) && (1 == TAN /*c_dbcsr_acc_opencl_atomic_fp_32*/)
  asm("{ .reg .f32 t; atom.global.add.f32 t, [%0], %1; }" ::"l"(dst), "f"(inc));
#      elif (defined(__NV_CL_C_VERSION) && !defined(XCHG)) && (2 == TAN /*c_dbcsr_acc_opencl_atomic_fp_64*/)
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

#endif /*OPENCL_ATOMICS_H*/
