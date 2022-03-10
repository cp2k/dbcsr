/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/

#include "acc_hip.h"

#if !defined(__HIP_PLATFORM_NVCC__)
hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags) { return hipHostMalloc(ptr, size, flags); }

hipError_t hipFreeHost(void* ptr) { return hipHostFree(ptr); }
#endif

unsigned int hipHostAllocDefault = hipHostMallocDefault;

hiprtcResult hiprtcGetLowLevelCode(hiprtcProgram prog, char* code) { return hiprtcGetCode(prog, code); }

hiprtcResult hiprtcGetLowLevelCodeSize(hiprtcProgram prog, size_t* codeSizeRet) { return hiprtcGetCodeSize(prog, codeSizeRet); }

hipError_t hipEventCreate(hipEvent_t* event, unsigned flags) { return hipEventCreateWithFlags(event, flags); }

hipError_t hipStreamCreate(hipStream_t* stream, unsigned int flags) { return hipStreamCreateWithFlags(stream, flags); }

hipError_t hipLaunchJITKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
  unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream,
  void** kernelParams, void** extra) {
  return hipModuleLaunchKernel(
    f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra);
}

hipblasStatus_t ACC_BLAS_STATUS_SUCCESS = HIPBLAS_STATUS_SUCCESS;
hipblasOperation_t ACC_BLAS_OP_N = HIPBLAS_OP_N;
hipblasOperation_t ACC_BLAS_OP_T = HIPBLAS_OP_T;
hiprtcResult ACC_RTC_SUCCESS = HIPRTC_SUCCESS;
