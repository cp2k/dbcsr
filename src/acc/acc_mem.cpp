/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#ifdef __CUDA
#include "cuda/acc_cuda.h"
#else
#include "hip/acc_hip.h"
#endif

#include <stdio.h>
#include <math.h>
#include "acc_error.h"
#include "include/acc.h"

static const int verbose_print = 0;


/****************************************************************************/
extern "C" int acc_dev_mem_allocate(void **dev_mem, size_t n){
  ACC_API_CALL(Malloc, ((void **) dev_mem, (size_t) n));
  if (dev_mem == NULL)
    return -2;
  if (verbose_print)
    printf ("Device allocation address %p, size %ld\n", *dev_mem, (long) n);

  return 0;
}


/****************************************************************************/
extern "C" int acc_dev_mem_deallocate(void *dev_mem){
  if (verbose_print)
    printf ("Device deallocation address %p\n", dev_mem);
  ACC_API_CALL(Free, ((void *) dev_mem));

  return 0;
}


/****************************************************************************/
extern "C" int acc_host_mem_allocate(void **host_mem, size_t n, void *stream){
  unsigned int flag = ACC(HostAllocDefault);

  ACC_API_CALL(HostAlloc, ((void **) host_mem, (size_t) n, flag));
  if (host_mem == NULL)
    return -2;
  if (verbose_print)
    printf ("Allocating %zd bytes of host pinned memory at %p\n", n, *host_mem);

  return 0;
}


/****************************************************************************/
extern "C" int acc_host_mem_deallocate(void *host_mem, void *stream){
  if (verbose_print)
    printf ("Host pinned deallocation address %p\n", host_mem);
  ACC_API_CALL(FreeHost, ((void *) host_mem));

  return 0;
}

/****************************************************************************/
extern "C" int acc_dev_mem_set_ptr(void **dev_mem, void *other, size_t lb){

  (*dev_mem) = ((char *) other) + lb;

  return 0;
}

/****************************************************************************/
extern "C" int acc_memcpy_h2d(const void *host_mem, void *dev_mem, size_t count, void* stream){
  ACC(Stream_t)* acc_stream = (ACC(Stream_t)*) stream;
  if (verbose_print)
      printf ("Copying %zd bytes from host address %p to device address %p \n", count, host_mem, dev_mem);

  ACC_API_CALL(MemcpyAsync, (dev_mem, host_mem, count, ACC(MemcpyHostToDevice), *acc_stream));

  return 0;
}


/****************************************************************************/
extern "C" int acc_memcpy_d2h(const void *dev_mem, void *host_mem, size_t count, void* stream){
  ACC(Stream_t)* acc_stream = (ACC(Stream_t)*) stream;
  if (verbose_print)
      printf ("Copying %zd bytes from device address %p to host address %p\n", count, dev_mem, host_mem);

  ACC_API_CALL(MemcpyAsync, (host_mem, dev_mem, count, ACC(MemcpyDeviceToHost), *acc_stream));

  if (verbose_print)
    printf ("d2h %f\n", *((double *) host_mem));

  return 0;
}


/****************************************************************************/
extern "C" int acc_memcpy_d2d(const void *devmem_src, void *devmem_dst, size_t count, void* stream){
  ACC(Stream_t)* acc_stream = (ACC(Stream_t)*) stream;
  if (verbose_print)
      printf ("Copying %zd bytes from device address %p to device address %p \n", count, devmem_src, devmem_dst);


  if(stream == NULL){
      ACC_API_CALL(Memcpy, (devmem_dst, devmem_src, count, ACC(MemcpyDeviceToDevice)));
  } else {
      ACC_API_CALL(MemcpyAsync, (devmem_dst, devmem_src, count, ACC(MemcpyDeviceToDevice), *acc_stream));
  }

  return 0;
}


/****************************************************************************/
extern "C" int acc_memset_zero(void *dev_mem, size_t offset, size_t length, void* stream){
  ACC(Error_t) cErr;
  ACC(Stream_t)* acc_stream = (ACC(Stream_t)*) stream;
  if(stream == NULL){
      cErr = ACC(Memset)((void *) (((char *) dev_mem) + offset), (int) 0, length);
  } else {
      cErr = ACC(MemsetAsync)((void *) (((char *) dev_mem) + offset), (int) 0, length, *acc_stream);
  }

  if (verbose_print)
    printf ("Zero at device address %p, offset %d, len %d\n",
     dev_mem, (int) offset, (int) length);
  if (acc_error_check(cErr))
    return -1;
  if (acc_error_check(ACC(GetLastError)()))
    return -1;

  return 0;
}


/****************************************************************************/
extern "C" int acc_dev_mem_info(size_t* free, size_t* avail){
  ACC_API_CALL(MemGetInfo, (free, avail));
  return 0;
}
