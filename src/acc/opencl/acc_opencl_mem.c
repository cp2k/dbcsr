/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#if defined(__OPENCL)
#include "acc_opencl.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#if defined(_WIN32)
# include <Windows.h>
#else
# if !defined(__linux__)
#   include <sys/types.h>
#   include <sys/sysctl.h>
# endif
# include <unistd.h>
#endif

#if !defined(ACC_OPENCL_MEM_MAPMULTI) && 0
# define ACC_OPENCL_MEM_MAPMULTI
#endif
#if !defined(ACC_OPENCL_MEM_ALIGNSCALE)
# define ACC_OPENCL_MEM_ALIGNSCALE 8
#endif


#if defined(__cplusplus)
extern "C" {
#endif

int acc_opencl_memalignment(size_t /*size*/);
int acc_opencl_memalignment(size_t size)
{
  int result;
  if ((ACC_OPENCL_MEM_ALIGNSCALE * ACC_OPENCL_MAXALIGN_NBYTES) <= size) {
    result = ACC_OPENCL_MAXALIGN_NBYTES;
  }
  else if ((ACC_OPENCL_MEM_ALIGNSCALE * ACC_OPENCL_CACHELINE_NBYTES) <= size) {
    result = ACC_OPENCL_CACHELINE_NBYTES;
  }
  else {
    result = sizeof(void*);
  }
  return result;
}


acc_opencl_info_hostptr_t* acc_opencl_info_hostptr(void* memory)
{
  assert(NULL == memory || sizeof(acc_opencl_info_hostptr_t) <= (uintptr_t)memory);
  return (NULL != memory
    ? (acc_opencl_info_hostptr_t*)((uintptr_t)memory - sizeof(acc_opencl_info_hostptr_t))
    : (acc_opencl_info_hostptr_t*)NULL);
}


void* acc_opencl_get_hostptr(cl_mem memory)
{
  void* result = NULL;
  if (NULL != memory) {
    ACC_OPENCL_EXPECT(CL_SUCCESS, clGetMemObjectInfo(result, CL_MEM_HOST_PTR, sizeof(void*), &result, NULL));
    assert(NULL != result);
  }
  return result;
}


int acc_host_mem_allocate(void** host_mem, size_t nbytes, void* stream)
{
  cl_int result;
  const int alignment = acc_opencl_memalignment(nbytes);
  const size_t size_meminfo = sizeof(acc_opencl_info_hostptr_t);
  const size_t size = nbytes + alignment + size_meminfo - 1;
  const cl_mem buffer = (acc_opencl_options.svm_interop
    ? clCreateBuffer(acc_opencl_context, CL_MEM_USE_HOST_PTR, size, clSVMAlloc(
        acc_opencl_context, CL_MEM_READ_WRITE, size, sizeof(void*)/*minimal alignment*/), &result)
    : clCreateBuffer(acc_opencl_context, CL_MEM_ALLOC_HOST_PTR, size, NULL/*host_ptr*/, &result));
  assert(NULL != host_mem && NULL != stream);
  if (NULL != buffer) {
    const cl_command_queue queue = *ACC_OPENCL_STREAM(stream);
    const uintptr_t address = (uintptr_t)clEnqueueMapBuffer(queue, buffer,
      !acc_opencl_options.async_memops, CL_MAP_READ | CL_MAP_WRITE,
      0/*offset*/, size, 0, NULL, NULL, &result);
    if (0 != address) {
      const uintptr_t aligned = ACC_OPENCL_UP2(address + size_meminfo, alignment);
      acc_opencl_info_hostptr_t* meminfo;
      assert(address + size_meminfo <= aligned);
      assert(CL_SUCCESS == result);
#if defined(ACC_OPENCL_MEM_MAPMULTI)
      assert(0 < aligned - address - size_meminfo);
      meminfo = (acc_opencl_info_hostptr_t*)clEnqueueMapBuffer(queue, buffer,
        CL_TRUE/*blocking*/, CL_MAP_READ | CL_MAP_WRITE,
        aligned - address - size_meminfo, size_meminfo, 0, NULL, NULL, &result);
#else
      meminfo = (acc_opencl_info_hostptr_t*)(aligned - size_meminfo);
#endif
      if (NULL != meminfo) {
        meminfo->buffer = buffer;
        meminfo->mapped = (void*)address;
        *host_mem = (void*)aligned;
      }
      else {
        ACC_OPENCL_ERROR("map buffer info", result);
        *host_mem = NULL;
      }
    }
    else {
      assert(CL_SUCCESS != result);
      ACC_OPENCL_ERROR("map host buffer", result);
      *host_mem = NULL;
    }
  }
  else {
    assert(CL_SUCCESS != result);
    ACC_OPENCL_ERROR("create host buffer", result);
    *host_mem = NULL;
  }
  ACC_OPENCL_RETURN(result);
}


int acc_host_mem_deallocate(void* host_mem, void* stream)
{
  int result = EXIT_SUCCESS;
  assert(NULL != stream);
  if (NULL != host_mem) {
    acc_opencl_info_hostptr_t *const meminfo = acc_opencl_info_hostptr(host_mem);
    const acc_opencl_info_hostptr_t info = *meminfo; /* copy meminfo prior to unmap */
    const cl_command_queue queue = *ACC_OPENCL_STREAM(stream);
    if (NULL != meminfo->buffer) {
#if defined(ACC_OPENCL_MEM_MAPMULTI)
      ACC_OPENCL_CHECK(clEnqueueUnmapMemObject(queue, meminfo->buffer, meminfo,
        0, NULL, NULL), "unmap memory info", result);
#endif
      ACC_OPENCL_CHECK(clEnqueueUnmapMemObject(queue, info.buffer, info.mapped,
        0, NULL, NULL), "unmap host memory", result);
      ACC_OPENCL_CHECK(clReleaseMemObject(info.buffer),
        "release host memory buffer", result);
      if (acc_opencl_options.svm_interop) {
        clSVMFree(acc_opencl_context, info.mapped);
      }
    }
  }
  ACC_OPENCL_RETURN(result);
}


int acc_dev_mem_allocate(void** dev_mem, size_t nbytes)
{
  cl_int result;
  const cl_mem buffer = (acc_opencl_options.svm_interop
    ? clCreateBuffer(acc_opencl_context, CL_MEM_USE_HOST_PTR, nbytes, clSVMAlloc(
        acc_opencl_context, CL_MEM_READ_WRITE, nbytes, 0/*default alignment*/), &result)
    : clCreateBuffer(acc_opencl_context, CL_MEM_READ_WRITE, nbytes, NULL/*host_ptr*/, &result));
  assert(NULL != dev_mem);
  if (NULL != buffer) {
#if defined(ACC_OPENCL_MEM_NOALLOC)
    assert(sizeof(void*) >= sizeof(cl_mem));
    *dev_mem = (void*)buffer;
#else
    *dev_mem = malloc(sizeof(cl_mem));
    if (NULL != *dev_mem) {
      *(cl_mem*)*dev_mem = buffer;
      result = EXIT_SUCCESS;
    }
    else {
      void *const ptr = (acc_opencl_options.svm_interop
        ? acc_opencl_get_hostptr(buffer) : NULL);
      clReleaseMemObject(buffer);
      /*if (NULL != ptr)*/ clSVMFree(acc_opencl_context, ptr);
      result = EXIT_FAILURE;
    }
#endif
  }
  else {
    assert(CL_SUCCESS != result);
    ACC_OPENCL_ERROR("create device buffer", result);
    *dev_mem = NULL;
  }
  ACC_OPENCL_RETURN(result);
}


int acc_dev_mem_deallocate(void* dev_mem)
{
  int result = EXIT_SUCCESS;
  if (NULL != dev_mem) {
    const cl_mem buffer = *ACC_OPENCL_MEM(dev_mem);
    void *const ptr = (acc_opencl_options.svm_interop
      ? acc_opencl_get_hostptr(buffer) : NULL);
    ACC_OPENCL_CHECK(clReleaseMemObject(buffer),
      "release device memory buffer", result);
#if defined(ACC_OPENCL_MEM_NOALLOC)
    assert(sizeof(void*) >= sizeof(cl_mem));
#else
    free(dev_mem);
#endif
    /*if (NULL != ptr)*/ clSVMFree(acc_opencl_context, ptr);
  }
  ACC_OPENCL_RETURN(result);
}


int acc_dev_mem_set_ptr(void** dev_mem, void* other, size_t lb)
{
  int result;
  assert(NULL != dev_mem);
  if (NULL != other || 0 == lb) {
    *dev_mem = (char*)other + lb;
    result = EXIT_SUCCESS;
  }
  else result = EXIT_FAILURE;
  ACC_OPENCL_RETURN(result);
}


int acc_memcpy_h2d(const void* host_mem, void* dev_mem, size_t nbytes, void* stream)
{
  int result = EXIT_SUCCESS;
  assert((NULL != host_mem || 0 == nbytes) && (NULL != dev_mem || 0 == nbytes) && NULL != stream);
  if (NULL != host_mem && NULL != dev_mem && 0 != nbytes) {
    ACC_OPENCL_CHECK(clEnqueueWriteBuffer(*ACC_OPENCL_STREAM(stream), *ACC_OPENCL_MEM(dev_mem),
      !acc_opencl_options.async_memops, 0/*offset*/, nbytes, host_mem, 0, NULL, NULL),
      "enqueue h2d copy", result);
  }
  ACC_OPENCL_RETURN(result);
}


int acc_memcpy_d2h(const void* dev_mem, void* host_mem, size_t nbytes, void* stream)
{
  int result = EXIT_SUCCESS;
  assert((NULL != dev_mem || 0 == nbytes) && (NULL != host_mem || 0 == nbytes) && NULL != stream);
  if (NULL != host_mem && NULL != dev_mem && 0 != nbytes) {
    ACC_OPENCL_CHECK(clEnqueueReadBuffer(*ACC_OPENCL_STREAM(stream), *ACC_OPENCL_MEM(dev_mem),
      !acc_opencl_options.async_memops, 0/*offset*/, nbytes, host_mem, 0, NULL, NULL),
      "enqueue d2h copy", result);
  }
  ACC_OPENCL_RETURN(result);
}


int acc_memcpy_d2d(const void* devmem_src, void* devmem_dst, size_t nbytes, void* stream)
{
  int result = EXIT_SUCCESS;
  assert((NULL != devmem_src || 0 == nbytes) && (NULL != devmem_dst || 0 == nbytes) && NULL != stream);
  if (NULL != devmem_src && NULL != devmem_dst && 0 != nbytes) {
    ACC_OPENCL_CHECK(clEnqueueCopyBuffer(*ACC_OPENCL_STREAM(stream),
      *ACC_OPENCL_MEM(devmem_src), *ACC_OPENCL_MEM(devmem_dst),
      0/*src_offset*/, 0/*dst_offset*/, nbytes, 0, NULL, NULL),
      "enqueue d2d copy", result);
  }
  ACC_OPENCL_RETURN(result);
}


int acc_memset_zero(void* dev_mem, size_t offset, size_t nbytes, void* stream)
{
  int result = EXIT_SUCCESS;
  assert((NULL != dev_mem || 0 == nbytes) && NULL != stream);
  if (0 != nbytes) {
    const cl_uchar pattern = 0; /* fill with zeros */
    ACC_OPENCL_CHECK(clEnqueueFillBuffer(*ACC_OPENCL_STREAM(stream), *ACC_OPENCL_MEM(dev_mem),
      &pattern, sizeof(pattern), offset, nbytes, 0, NULL, NULL),
      "enqueue zeroing kernel", result);
  }
  ACC_OPENCL_RETURN(result);
}


int acc_opencl_info_devmem(cl_device_id device, size_t* mem_free, size_t* mem_total)
{
  int result = EXIT_SUCCESS;
  size_t size_free = 0, size_total = 0;
#if defined(_WIN32)
  MEMORYSTATUSEX mem_status;
  mem_status.dwLength = sizeof(mem_status);
  if (GlobalMemoryStatusEx(&mem_status)) {
    size_total = (size_t)mem_status.ullTotalPhys;
    size_free  = (size_t)mem_status.ullAvailPhys;
  }
#else
# if defined(_SC_PAGE_SIZE)
  const long page_size = sysconf(_SC_PAGE_SIZE);
# else
  const long page_size = 4096;
# endif
# if defined(__linux__)
#   if defined(_SC_PHYS_PAGES)
  const long pages_total = sysconf(_SC_PHYS_PAGES);
#   else
  const long pages_total = 0;
#   endif
#   if defined(_SC_AVPHYS_PAGES)
  const long pages_free = sysconf(_SC_AVPHYS_PAGES);
#   else
  const long pages_free = pages_total;
#   endif
# else
  /*const*/ size_t size_pages_free = sizeof(const long), size_pages_total = sizeof(const long);
  long pages_free = 0, pages_total = 0;
  ACC_OPENCL_EXPECT(0, sysctlbyname("hw.memsize", &pages_total, &size_pages_total, NULL, 0));
  if (0 < page_size) pages_total /= page_size;
  if (0 != sysctlbyname("vm.page_free_count", &pages_free, &size_pages_free, NULL, 0)) {
    pages_free = pages_total;
  }
# endif
  if (0 < page_size && 0 <= pages_free && 0 <= pages_total) {
    const size_t size_page = (size_t)page_size;
    size_total = size_page * (size_t)pages_total;
    size_free  = size_page * (size_t)pages_free;
  }
#endif
  if (NULL != device) {
    cl_ulong cl_size_total = 0;
    ACC_OPENCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
      sizeof(cl_ulong), &cl_size_total, NULL), "retrieve amount of device memory", result);
    assert(0 < acc_opencl_ndevices);
    size_total /= acc_opencl_ndevices;
    size_free  /= acc_opencl_ndevices;
    if (EXIT_SUCCESS == result) {
      if (cl_size_total < size_total) size_total = cl_size_total;
      if (size_total < size_free) size_free = size_total;
    }
  }
  result = (size_free <= size_total ? EXIT_SUCCESS : EXIT_FAILURE);
  assert(NULL != mem_free || NULL != mem_total);
  if (NULL != mem_total) *mem_total = size_total;
  if (NULL != mem_free)  *mem_free  = size_free;
  ACC_OPENCL_RETURN(result);
}


int acc_dev_mem_info(size_t* mem_free, size_t* mem_total)
{
  int result = EXIT_SUCCESS;
  cl_device_id active_id = NULL;
  if (NULL != acc_opencl_context) {
    result = acc_opencl_device(NULL/*stream*/, &active_id);
  }
  if (EXIT_SUCCESS == result) {
    result = acc_opencl_info_devmem(active_id, mem_free, mem_total);
  }
  ACC_OPENCL_RETURN(result);
}

#if defined(__cplusplus)
}
#endif

#endif /*__OPENCL*/
