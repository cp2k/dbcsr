/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/

#ifndef LIBSMM_ACC_H
#define LIBSMM_ACC_H

#if defined(__CUDA)
#  include "../cuda/acc_cuda.h"
#elif defined(__HIP)
#  include "../hip/acc_hip.h"
#endif

#include "../acc_libsmm.h"
#include "parameters_utils.h"
#include "libsmm_acc_init.h"

#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <vector>

enum libsmm_acc_algo { largeDB1 = 1, largeDB2 = 2, medium = 3, small = 4, tiny = 5 };

struct kernel_launcher {
  ACC_DRV(function) kernel_function;
  int threads;
  int grouping;
  kernel_launcher(ACC_DRV(function) const& kf, int th, int gp) : kernel_function(kf), threads(th), grouping(gp) {}
};

typedef std::unordered_map<Triplet, kernel_launcher>::iterator kernel_map_iterator;

static std::unordered_map<Triplet, kernel_launcher> kernel_handles;
static std::unordered_set<Triplet> failed_acc_kernels;

int libsmm_acc_process_blas(const int* param_stack_host, int stack_size, ACC_DRV(stream) stream, int m, int n, int k,
  const double* a_data, const double* b_data, double* c_data);

int libsmm_acc_process_d(const int* param_stack_dev, int stack_size, ACC_DRV(stream) stream, int m, int n, int k,
  const double* a_data, const double* b_data, double* c_data);

static std::unordered_map<Triplet, ACC_DRV(function)> transpose_handles;

int libsmm_acc_transpose_d(const int* trs_stack, int offset, int nblks, double* buffer, int m, int n, ACC_DRV(stream) stream);

#endif /*LIBSMM_ACC_H*/
