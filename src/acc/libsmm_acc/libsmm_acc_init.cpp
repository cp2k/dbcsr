/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include "libsmm_acc_init.h"
#include "parameters.h"

#if defined(_OPENMP)
#include <omp.h>
#endif


std::vector<ACC_BLAS(Handle_t)*> acc_blashandles;

//===========================================================================
int libsmm_acc_gpu_blas_init(){
    // allocate memory for acc_blas handles
#if defined _OPENMP
    int nthreads = omp_get_num_threads();
#else
    int nthreads = 1;
#endif
    acc_blashandles.resize(nthreads);

    // initialize acc_blas and store acc_blas handles
    // one handle per thread!
    for(int i = 0; i < nthreads; i++){
        ACC_BLAS(Handle_t)* c_handle;
        acc_blas_create(&c_handle);
        acc_blashandles[i] = c_handle;
    }

    return 0;
}


//===========================================================================
int libsmm_acc_init() {

    // check warp size consistency
    libsmm_acc_check_gpu_warp_size_consistency();
    libsmm_acc_gpu_blas_init();
    return 0;
}


//===========================================================================
int libsmm_acc_finalize() {

    // deallocate memory for acc_blas handles
#if defined _OPENMP
    int nthreads = omp_get_num_threads();
#else
    int nthreads = 1;
#endif

    // free acc_blas handle resources
    // one handle per thread!
    for(int i = 0; i < nthreads; i++){
        acc_blas_destroy(acc_blashandles[i]);
    }

    return 0;
}


//===========================================================================
int libsmm_acc_check_gpu_warp_size_consistency() {
    int acc_warp_size = acc_get_gpu_warp_size();
    if (warp_size != acc_warp_size){
        printf("Inconsistency in warp sizes: Cuda/Hip indicates warp size = %d, while the gpu_properties files indicates warp_size = %d.\nPlease check whether src/acc/libsmm_acc/kernels/gpu_properties.json contains the correct data about the GPU you are using.", warp_size, acc_warp_size);
    }
    return 0;
}

//===========================================================================
int acc_get_gpu_warp_size() {
    int device = 0;
    ACC(DeviceProp) prop;
    ACC_API_CALL(GetDevice, (&device));
    ACC_API_CALL(GetDeviceProperties, (&prop, device));
    return prop.warpSize;
}

//===========================================================================
extern "C" int libsmm_acc_is_thread_safe() {
#if defined(_OPENMP)
    return 1;  // i.e. true, libsmm_acc is threaded
#else
    return 0;  // i.e. false, libsmm_acc is not threaded
#endif
}

