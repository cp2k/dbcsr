/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2017  CP2K developers group                         *
 *****************************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/acc.h"

#ifdef __CUDA_PROFILING
#include <nvToolsExtCudaRt.h>
#endif

#define CUDA_SAFE_CALL(name, x)                                   \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      printf("\nerror: %s failed with error %s",                  \
             name, msg);                                          \
      exit(1);                                                    \
    }                                                             \
  } while(0)


/****************************************************************************/
extern "C" int acc_drv_init(){
  // Driver boilerplate
  CUDA_SAFE_CALL("cuInit", cuInit(0));
  CUDA_SAFE_CALL("cuDeviceGet", cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL("cuCtxGetCurrent", cuCtxGetCurrent(&context));
  if(context == NULL){
    printf("cuCtxGetCurrent error: no context is bound to the calling CPU thread");
    return -1;
  }
  return 0;
}


