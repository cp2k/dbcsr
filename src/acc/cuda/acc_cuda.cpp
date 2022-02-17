/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/

#include "acc_cuda.h"

nvrtcResult nvrtcGetLowLevelCode(nvrtcProgram prog, char* code) { return nvrtcGetPTX(prog, code); }

nvrtcResult nvrtcGetLowLevelCodeSize(nvrtcProgram prog, size_t* codeSizeRet) { return nvrtcGetPTXSize(prog, codeSizeRet); }

CUresult cuLaunchJITKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
  unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream stream,
  void** kernelParams, void** extra) {
  return cuLaunchKernel(
    f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra);
}

// CUDA Runtime API: flag values
CUevent_flags CUEventDefault = CU_EVENT_DEFAULT;
CUstream_flags CUStreamDefault = CU_STREAM_DEFAULT;
CUsharedconfig CUSharedMemBankSizeEightByte = CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE;

cublasStatus_t ACC_BLAS_STATUS_SUCCESS = CUBLAS_STATUS_SUCCESS;
cublasOperation_t ACC_BLAS_OP_N = CUBLAS_OP_N;
cublasOperation_t ACC_BLAS_OP_T = CUBLAS_OP_T;
nvrtcResult ACC_RTC_SUCCESS = NVRTC_SUCCESS;
