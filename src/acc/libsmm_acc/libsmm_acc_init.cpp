/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include "libsmm_acc.h"
#include "libsmm_acc_init.h"
#include "parameters.h"

#if defined _OPENMP
#include <omp.h>
#endif

//===========================================================================
int libsmm_acc_init() {
    libsmm_acc_check_gpu_warp_size_consistency();
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

