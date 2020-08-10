/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include "libsmm_acc_init.h"
//#include "libsmm_acc.h"
#include "parameters.h"
#include "../cuda/cublas.h"

#ifdef __CUDA
# include "../cuda/acc_cuda.h"
#else
# include "../hip/acc_hip.h"
#endif

#if defined _OPENMP
#include <omp.h>
#endif


cublasHandle_t* cublas_handle;

//===========================================================================
int libsmm_acc_gpu_blas_init(){
#if defined _OPENMP
    // allocate memory for cublas handles
    int nthreads = 0;
#pragma omp master
{
    nthreads = omp_get_num_threads();
    cublas_handles.resize(nthreads);
}
#pragma omp barrier
    int ithread = omp_get_thread_num();
    // initialize cublas and store cublas handles
    // one handle per thread!
    cublasHandle_t* c_handle;
    cublas_create(&c_handle);
    cublas_handles.emplace(ithread, c_handle);
#else
    cublas_create(&cublas_handle);
#endif

    return 0;
}


//===========================================================================
int libsmm_acc_init() {

    // check warp size consistency
    libsmm_acc_check_gpu_warp_size_consistency();

    libsmm_acc_gpu_blas_init();
    //printf("[libsmm_acc_init] exit with cublas_handles size = %i\n", int(cublas_handles.size()));
    return 0;
}


//===========================================================================
int libsmm_acc_finalize() {

    // deallocate memory for cublas handles
#if defined _OPENMP
    int ithread = omp_get_thread_num();
#else
    //int ithread = 0;
#endif
    // initialize cublas and store cublas handles
    // one handle per thread!
    //FIXME cublas_destroy(cublas_handle);
#if defined _OPENMP
#pragma omp barrier
#endif

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
#if defined _OPENMP
    return 1;  // i.e. true, libsmm_acc is threaded
#else
    return 0;  // i.e. false, libsmm_acc is not threaded
#endif
}

