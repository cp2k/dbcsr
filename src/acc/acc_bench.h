/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: BSD-3-Clause                                                          */
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

#if !defined(MAX_KERNEL_DIM)
#  define MAX_KERNEL_DIM 80
#endif

/**
 * Initialize a column-major M-by-N matrix with
 * deterministic values derived from seed and scale.
 */
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
#define INIT_STACK(STACK, STACK_SIZE, RND_SIZE, RND, MN, MK, KN, NC, NA, NB) \
  do { \
    const int* init_stack_rnd_ = (const int*)(RND); \
    const int init_stack_rnd_size_ = MAX(1, (RND_SIZE)); \
    const int init_stack_navg_ = (STACK_SIZE) / (NC); \
    const int init_stack_nimb_ = MAX(1, init_stack_navg_ - 4); \
    int init_stack_i_ = 0, init_stack_c_ = 0, init_stack_ntop_ = 0; \
    int* init_stack_p_ = (STACK); \
    assert(0 < (NC) && (NC) <= (STACK_SIZE)); \
    while (init_stack_i_ < (STACK_SIZE)) { \
      const int init_stack_r_ = \
        ((NULL == init_stack_rnd_ || 0 >= (RND_SIZE)) ? rand() : init_stack_rnd_[init_stack_i_ % init_stack_rnd_size_]); \
      const int init_stack_next_ = init_stack_c_ + 1; \
      init_stack_ntop_ += init_stack_navg_ + (init_stack_r_ % (2 * init_stack_nimb_) - init_stack_nimb_); \
      if ((STACK_SIZE) < init_stack_ntop_) init_stack_ntop_ = (STACK_SIZE); \
      for (; init_stack_i_ < init_stack_ntop_; ++init_stack_i_) { \
        int init_stack_a_, init_stack_b_; \
        if (NULL != init_stack_rnd_ && 0 < (RND_SIZE)) { \
          init_stack_a_ = init_stack_rnd_[(2 * init_stack_i_ + 0) % init_stack_rnd_size_] % (NA); \
          init_stack_b_ = init_stack_rnd_[(2 * init_stack_i_ + 1) % init_stack_rnd_size_] % (NB); \
        } \
        else { \
          init_stack_a_ = rand() % (NA); \
          init_stack_b_ = rand() % (NB); \
        } \
        *init_stack_p_++ = init_stack_a_ * (MK) + 1; \
        *init_stack_p_++ = init_stack_b_ * (KN) + 1; \
        *init_stack_p_++ = init_stack_c_ * (MN) + 1; \
      } \
      if (init_stack_next_ < (NC)) init_stack_c_ = init_stack_next_; \
    } \
  } while (0)

#endif /*DBCSR_ACC_BENCH_H*/
