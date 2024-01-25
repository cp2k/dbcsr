/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/

#include "libsmm_acc_init.h"
#include "../acc_libsmm.h"
#include "../cuda_hip/acc_utils.h"

#if defined(_OPENMP)
#  include <omp.h>
#endif


std::vector<ACC_BLAS(Handle_t)*> acc_blashandles;

//===========================================================================
#if defined(__DBCSR_ACC)
void timeset(const std::string& routine_name, int& handle) {
  const char* routine_name_ = routine_name.c_str();
  int routine_name_length = routine_name.length();
  c_dbcsr_timeset(&routine_name_, &routine_name_length, &handle);
}
void timestop(int handle) { c_dbcsr_timestop(&handle); }
#else
void timeset(const std::string& routine_name, int& handle) {
  (void)(routine_name);
  (void)(handle);
}
void timestop(int handle) { (void)(handle); }
#endif

//===========================================================================
int libsmm_acc_gpu_blas_init() {
  // allocate memory for acc_blas handles
#if defined _OPENMP
  const int nthreads = omp_get_num_threads();
#else
  const int nthreads = 1;
#endif
  const int size = static_cast<int>(acc_blashandles.size());

  if (size < nthreads) {
    acc_blashandles.resize(nthreads);
    // initialize acc_blas and store acc_blas handles
    for (int i = size; i < nthreads; i++) {
      ACC_BLAS(Handle_t) * c_handle;
      acc_blas_create(&c_handle);
      acc_blashandles[i] = c_handle;
    }
  }
  return 0;
}

//===========================================================================
extern "C" int libsmm_acc_init() {
  std::string routineN = "libsmm_acc_init";
  int handle;

  timeset(routineN, handle);

  // check warp size consistency
  libsmm_acc_check_gpu_warp_size_consistency();
  libsmm_acc_gpu_blas_init();

  timestop(handle);

  return 0;
}

//===========================================================================
extern "C" int libsmm_acc_finalize() {
  std::string routineN = "libsmm_acc_finalize";
  int handle;

  timeset(routineN, handle);

  // free acc_blas handle resources; one handle per thread
  for (size_t i = 0; i < acc_blashandles.size(); i++) {
    acc_blas_destroy(acc_blashandles[i]);
  }
  acc_blashandles.clear();

  timestop(handle);

  return 0;
}

//===========================================================================
int libsmm_acc_check_gpu_warp_size_consistency() {
  int acc_warp_size = acc_get_gpu_warp_size();
  extern const int warp_size;
  if (warp_size != acc_warp_size) {
    printf("Inconsistency in warp sizes: Cuda/Hip indicates warp size = %d, while the gpu_properties files indicates warp_size = "
           "%d.\nPlease check whether src/acc/libsmm_acc/kernels/gpu_properties.json contains the correct data about the GPU you "
           "are using.",
      warp_size, acc_warp_size);
  }
  return 0;
}

//===========================================================================
extern "C" int libsmm_acc_is_thread_safe() {
#if defined(_OPENMP)
  return 1; // i.e. true, libsmm_acc is threaded
#else
  return 0; // i.e. false, libsmm_acc is not threaded
#endif
}
