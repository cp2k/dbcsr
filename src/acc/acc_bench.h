/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/
#ifndef DBCSR_ACC_BENCH_H
#define DBCSR_ACC_BENCH_H

#include <stdlib.h>
#include <assert.h>

#if !defined(MIN)
#  define MIN(A, B) ((A) < (B) ? (A) : (B))
#endif
#if !defined(MAX)
#  define MAX(A, B) ((B) < (A) ? (A) : (B))
#endif

#if !defined(INLINE) && (defined(__cplusplus) || (defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__) /*C99*/))
#  define INLINE inline
#else
#  define INLINE
#endif

#if !defined(MAX_KERNEL_DIM)
#  define MAX_KERNEL_DIM 80
#endif

#define INIT_MAT(ELEM_TYPE, SEED, MAT, M, N, SCALE) \
  do { \
    const double init_mat_seed1_ = (SCALE) * (SEED) + (SCALE); \
    int init_mat_i_, init_mat_j_; \
    for (init_mat_i_ = 0; init_mat_i_ < (N); ++init_mat_i_) { \
      for (init_mat_j_ = 0; init_mat_j_ < (M); ++init_mat_j_) { \
        const int init_mat_k_ = init_mat_i_ * (M) + init_mat_j_; \
        ((ELEM_TYPE*)(MAT))[init_mat_k_] = (ELEM_TYPE)(init_mat_seed1_ * (init_mat_k_ + 1)); \
      } \
    } \
  } while (0)


/**
 * Artificial stack-setup for DBCSR/ACC benchmarks.
 * The arguments rnd and rnd_size optionally allow
 * to supply an array of (pseudo-)random-numbers.
 */
static INLINE void init_stack(
  int* stack, int stack_size, int rnd_size, const int* rnd, int mn, int mk, int kn, int nc, int na, int nb) {
  /* navg matrix products are accumulated into a C-matrix */
  const int navg = stack_size / nc;
  const int nimb = MAX(1, navg - 4); /* imbalance */
  int i = 0, c = 0, ntop = 0;
  assert(0 < nc && nc <= stack_size);
  while (i < stack_size) {
    const int r = ((NULL == rnd || 0 >= rnd_size) ? rand() : rnd[i % rnd_size]), next = c + 1;
    ntop += navg + (r % (2 * nimb) - nimb);
    if (stack_size < ntop) ntop = stack_size;
    for (; i < ntop; ++i) { /* setup one-based indexes */
      int a, b;
      if (NULL != rnd && 0 < rnd_size) {
        a = rnd[(2 * i + 0) % rnd_size] % na;
        b = rnd[(2 * i + 1) % rnd_size] % nb;
      }
      else {
        a = rand() % na;
        b = rand() % nb;
      }
      *stack++ = a * mk + 1; /* A-index */
      *stack++ = b * kn + 1; /* B-index */
      *stack++ = c * mn + 1; /* C-index */
    }
    if (next < nc) c = next;
  }
}

#endif /*DBCSR_ACC_BENCH_H*/
