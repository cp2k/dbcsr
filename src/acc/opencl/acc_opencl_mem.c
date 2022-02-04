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
#include <libxsmm_sync.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(_WIN32)
# include <Windows.h>
#else
# if !defined(__linux__) && defined(__APPLE__) && defined(__MACH__)
#   include <sys/types.h>
#   include <sys/sysctl.h>
# endif
# include <unistd.h>
#endif

#if !defined(ACC_OPENCL_MEM_ZERO_KERNEL) && 0
# define ACC_OPENCL_MEM_ZERO_KERNEL
#endif
#if !defined(ACC_OPENCL_MEM_NLOCKS)
# define ACC_OPENCL_MEM_NLOCKS ACC_OPENCL_NLOCKS
#endif
#if !defined(ACC_OPENCL_MEM_ALIGNSCALE)
# define ACC_OPENCL_MEM_ALIGNSCALE 8
#endif


#if defined(__cplusplus)
extern "C" {
#endif

int c_dbcsr_acc_opencl_memalignment(size_t /*size*/);
int c_dbcsr_acc_opencl_memalignment(size_t size)
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


void* c_dbcsr_acc_opencl_get_hostptr(cl_mem memory)
{
  void* result = NULL;
  ACC_OPENCL_EXPECT(CL_SUCCESS, clGetMemObjectInfo(memory, CL_MEM_HOST_PTR, sizeof(void*), &result, NULL));
  return result;
}


c_dbcsr_acc_opencl_info_hostptr_t* c_dbcsr_acc_opencl_info_hostptr(void* memory)
{
  assert(NULL == memory || sizeof(c_dbcsr_acc_opencl_info_hostptr_t) <= (uintptr_t)memory);
  return (NULL != memory
    ? (c_dbcsr_acc_opencl_info_hostptr_t*)((uintptr_t)memory - sizeof(c_dbcsr_acc_opencl_info_hostptr_t))
    : (c_dbcsr_acc_opencl_info_hostptr_t*)NULL);
}


int c_dbcsr_acc_host_mem_allocate(void** host_mem, size_t nbytes, void* stream)
{
  cl_context context = NULL;
  cl_command_queue queue;
  cl_mem memory = NULL;
  cl_int result;
  const size_t size_meminfo = sizeof(c_dbcsr_acc_opencl_info_hostptr_t);
  const int alignment = c_dbcsr_acc_opencl_memalignment(nbytes);
  nbytes += alignment + size_meminfo - 1;
  assert(NULL != host_mem && NULL != stream);
  queue = *ACC_OPENCL_STREAM(stream);
  ACC_OPENCL_DEBUG_IF(EXIT_SUCCESS != c_dbcsr_acc_opencl_stream_is_thread_specific(
    ACC_OPENCL_OMP_TID(), queue))
  {
    ACC_OPENCL_DEBUG_FPRINTF(stderr, "WARNING ACC/OpenCL: "
      "c_dbcsr_acc_host_mem_allocate called by foreign thread!\n");
  }
  result = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL);
  if (CL_SUCCESS == result) {
    memory = (
#if defined(ACC_OPENCL_SVM)
      0 != c_dbcsr_acc_opencl_config.devinfo.svm_interop ? clCreateBuffer(context,
          CL_MEM_USE_HOST_PTR, nbytes,
        clSVMAlloc(context, CL_MEM_READ_WRITE, nbytes, sizeof(void*)/*minimal alignment*/), &result) :
#elif defined(ACC_OPENCL_MALLOC_LIBXSMM)
      clCreateBuffer(context, CL_MEM_USE_HOST_PTR, nbytes,
        libxsmm_aligned_malloc(nbytes, sizeof(void*)/*minimal alignment*/), &result));
#else
      clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, nbytes, NULL/*host_ptr*/, &result));
#endif
  }
  assert(CL_SUCCESS == result || NULL == memory);
  if (CL_SUCCESS == result) {
    void *const mapped = clEnqueueMapBuffer(queue, memory,
      !c_dbcsr_acc_opencl_config.devinfo.async, CL_MAP_READ | CL_MAP_WRITE,
      0/*offset*/, nbytes, 0, NULL, NULL, &result);
    assert(CL_SUCCESS == result || NULL == mapped);
    if (CL_SUCCESS == result) {
      const uintptr_t address = (uintptr_t)mapped;
      const uintptr_t aligned = LIBXSMM_UP2(address + size_meminfo, alignment);
      c_dbcsr_acc_opencl_info_hostptr_t* meminfo;
      assert(address + size_meminfo <= aligned);
      meminfo = (c_dbcsr_acc_opencl_info_hostptr_t*)(aligned - size_meminfo);
      if (NULL != meminfo) {
        meminfo->memory = memory;
        meminfo->mapped = mapped;
        *host_mem = (void*)aligned;
      }
      else {
        result = EXIT_FAILURE;
        ACC_OPENCL_ERROR("map buffer info", result);
        *host_mem = NULL;
      }
    }
    else {
      ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseMemObject(memory));
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


int c_dbcsr_acc_host_mem_deallocate(void* host_mem, void* stream)
{
  int result;
  assert(NULL != stream);
  if (NULL != host_mem) {
    c_dbcsr_acc_opencl_info_hostptr_t *const meminfo = c_dbcsr_acc_opencl_info_hostptr(host_mem);
    if (NULL != meminfo->memory) {
      const c_dbcsr_acc_opencl_info_hostptr_t info = *meminfo; /* copy meminfo prior to unmap */
      const cl_command_queue queue = *ACC_OPENCL_STREAM(stream);
      int result_release;
      result = clEnqueueUnmapMemObject(queue, info.memory, info.mapped, 0, NULL, NULL);
# if defined(ACC_OPENCL_SVM)
      /* svm_interop shall be part of c_dbcsr_acc_opencl_info_hostptr_t */
      assert(0 != c_dbcsr_acc_opencl_config.devinfo.svm_interop);
      { cl_context context;
        result = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT,
          sizeof(cl_context), &context, NULL);
        if (CL_SUCCESS == result) clSVMFree(context, info.mapped);
      }
# elif defined(ACC_OPENCL_MALLOC_LIBXSMM)
      libxsmm_free(info.mapped);
# endif
      result_release = clReleaseMemObject(info.memory);
      if (EXIT_SUCCESS == result) result = result_release;
    }
    else result = EXIT_FAILURE;
  }
  else result = EXIT_FAILURE;
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_dev_mem_allocate(void** dev_mem, size_t nbytes)
{
  cl_int result;
  const int try_flag = ((0 != c_dbcsr_acc_opencl_config.devinfo.unified
      ||  (0 == c_dbcsr_acc_opencl_config.devinfo.intel_id)
      ||  (0x4905 != c_dbcsr_acc_opencl_config.devinfo.intel_id
        && 0x020a != c_dbcsr_acc_opencl_config.devinfo.intel_id
        && 0x0bd5 != c_dbcsr_acc_opencl_config.devinfo.intel_id))
    ? 0 : (1u << 22));
  const cl_context context = c_dbcsr_acc_opencl_context();
  cl_mem buffer;
  assert(NULL != dev_mem && 0 <= ACC_OPENCL_OVERMALLOC);
  assert(NULL != context);
  buffer = (
#if defined(ACC_OPENCL_SVM)
    0 != c_dbcsr_acc_opencl_config.devinfo.svm_interop ? clCreateBuffer(context, CL_MEM_USE_HOST_PTR,
      nbytes + ACC_OPENCL_OVERMALLOC, clSVMAlloc(context, (cl_mem_flags)(CL_MEM_READ_WRITE | try_flag),
      nbytes + ACC_OPENCL_OVERMALLOC, 0/*default alignment*/), &result) :
#endif
    clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE | try_flag),
      nbytes + ACC_OPENCL_OVERMALLOC, NULL/*host_ptr*/, &result));
  if (0 != try_flag && CL_SUCCESS != result) { /* retry without try_flag */
    buffer = (
#if defined(ACC_OPENCL_SVM)
      0 != c_dbcsr_acc_opencl_config.devinfo.svm_interop ? clCreateBuffer(context, CL_MEM_USE_HOST_PTR,
        nbytes + ACC_OPENCL_OVERMALLOC, clSVMAlloc(context, CL_MEM_READ_WRITE,
        nbytes + ACC_OPENCL_OVERMALLOC, 0/*default alignment*/), &result) :
#endif
      clCreateBuffer(context, CL_MEM_READ_WRITE,
        nbytes + ACC_OPENCL_OVERMALLOC, NULL/*host_ptr*/, &result));
  }
  if (EXIT_SUCCESS == result) {
    assert(NULL != buffer);
#if defined(ACC_OPENCL_MEM_NOALLOC)
    assert(sizeof(void*) >= sizeof(cl_mem));
    *dev_mem = (void*)buffer;
#else
    *dev_mem = malloc(sizeof(cl_mem));
    if (NULL != *dev_mem) {
      *(cl_mem*)*dev_mem = buffer;
    }
    else {
# if defined(ACC_OPENCL_SVM)
      void *const ptr = (0 != c_dbcsr_acc_opencl_config.devinfo.svm_interop
        ? c_dbcsr_acc_opencl_get_hostptr(buffer) : NULL);
# endif
      clReleaseMemObject(buffer);
# if defined(ACC_OPENCL_SVM)
      /*if (NULL != ptr)*/ clSVMFree(context, ptr);
# endif
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


int c_dbcsr_acc_dev_mem_deallocate(void* dev_mem)
{
  int result = EXIT_SUCCESS;
  if (NULL != dev_mem) {
    const cl_mem buffer = *ACC_OPENCL_MEM(dev_mem);
#if defined(ACC_OPENCL_SVM)
    /* svm_interop shall be part of a device memory info */
    void *const ptr = (0 != c_dbcsr_acc_opencl_config.devinfo.svm_interop
      ? c_dbcsr_acc_opencl_get_hostptr(buffer) : NULL);
#endif
    ACC_OPENCL_CHECK(clReleaseMemObject(buffer),
      "release device memory buffer", result);
#if defined(ACC_OPENCL_MEM_NOALLOC)
    assert(sizeof(void*) >= sizeof(cl_mem));
#else
    free(dev_mem);
#endif
#if defined(ACC_OPENCL_SVM)
    /*if (NULL != ptr)*/
    {
      const cl_context context = c_dbcsr_acc_opencl_context();
      assert(NULL != context);
      clSVMFree(context, ptr);
    }
#endif
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_dev_mem_set_ptr(void** dev_mem, void* other, size_t lb)
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


int c_dbcsr_acc_memcpy_h2d(const void* host_mem, void* dev_mem, size_t nbytes, void* stream)
{
  int result = EXIT_SUCCESS;
  assert((NULL != host_mem || 0 == nbytes) && (NULL != dev_mem || 0 == nbytes) && NULL != stream);
  if (NULL != host_mem && NULL != dev_mem && 0 != nbytes) {
    const cl_command_queue queue = *ACC_OPENCL_STREAM(stream);
    ACC_OPENCL_DEBUG_IF(EXIT_SUCCESS != c_dbcsr_acc_opencl_stream_is_thread_specific(
      ACC_OPENCL_OMP_TID(), queue))
    {
      ACC_OPENCL_DEBUG_FPRINTF(stderr, "WARNING ACC/OpenCL: "
        "c_dbcsr_acc_memcpy_h2d called by foreign thread!\n");
    }
    result = clEnqueueWriteBuffer(queue, *ACC_OPENCL_MEM(dev_mem),
      CL_FALSE, 0/*offset*/, nbytes, host_mem, 0, NULL, NULL);
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_memcpy_d2h(const void* dev_mem, void* host_mem, size_t nbytes, void* stream)
{
  int result = EXIT_SUCCESS;
  assert((NULL != dev_mem || 0 == nbytes) && (NULL != host_mem || 0 == nbytes) && NULL != stream);
  if (NULL != host_mem && NULL != dev_mem && 0 != nbytes) {
    const cl_command_queue queue = *ACC_OPENCL_STREAM(stream);
    ACC_OPENCL_DEBUG_IF(EXIT_SUCCESS != c_dbcsr_acc_opencl_stream_is_thread_specific(
      ACC_OPENCL_OMP_TID(), queue))
    {
      ACC_OPENCL_DEBUG_FPRINTF(stderr, "WARNING ACC/OpenCL: "
        "c_dbcsr_acc_memcpy_d2h called by foreign thread!\n");
    }
    result = clEnqueueReadBuffer(queue, *ACC_OPENCL_MEM(dev_mem),
      !c_dbcsr_acc_opencl_config.devinfo.async, 0/*offset*/, nbytes,
      host_mem, 0, NULL, NULL);
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_memcpy_d2d(const void* devmem_src, void* devmem_dst, size_t nbytes, void* stream)
{
  int result = EXIT_SUCCESS;
  assert((NULL != devmem_src || 0 == nbytes) && (NULL != devmem_dst || 0 == nbytes) && NULL != stream);
  if (NULL != devmem_src && NULL != devmem_dst && 0 != nbytes) {
    const cl_command_queue queue = *ACC_OPENCL_STREAM(stream);
    ACC_OPENCL_DEBUG_IF(EXIT_SUCCESS != c_dbcsr_acc_opencl_stream_is_thread_specific(
      ACC_OPENCL_OMP_TID(), queue))
    {
      ACC_OPENCL_DEBUG_FPRINTF(stderr, "WARNING ACC/OpenCL: "
        "c_dbcsr_acc_memcpy_d2d called by foreign thread!\n");
    }
    result = clEnqueueCopyBuffer(queue,
      *ACC_OPENCL_MEM(devmem_src), *ACC_OPENCL_MEM(devmem_dst),
      0/*src_offset*/, 0/*dst_offset*/, nbytes, 0, NULL, NULL);
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_memset_zero(void* dev_mem, size_t offset, size_t nbytes, void* stream)
{
  int result = EXIT_SUCCESS;
  assert((NULL != dev_mem || 0 == nbytes) && NULL != stream);
  if (0 != nbytes) {
    const cl_command_queue queue = *ACC_OPENCL_STREAM(stream);
    ACC_OPENCL_DEBUG_IF(EXIT_SUCCESS != c_dbcsr_acc_opencl_stream_is_thread_specific(
      ACC_OPENCL_OMP_TID(), queue))
    {
      ACC_OPENCL_DEBUG_FPRINTF(stderr, "WARNING ACC/OpenCL: "
        "c_dbcsr_acc_memset_zero called by foreign thread!\n");
    }
    { const cl_mem *const buffer = ACC_OPENCL_MEM(dev_mem);
#if defined(ACC_OPENCL_MEM_ZERO_KERNEL)
      /* creating cl_kernel and calling clSetKernelArg must be synchronized */
      static volatile int lock;
      static cl_kernel kernel = NULL;
      LIBXSMM_ATOMIC_ACQUIRE(&lock, LIBXSMM_SYNC_NPAUSE, LIBXSMM_ATOMIC_RELAXED);
      if (NULL == kernel) { /* generate kernel */
        const char source[] =
          "kernel void memset_zero(global uchar *restrict buffer) {\n"
          "  const size_t i = get_global_id(0);\n"
          "  const uchar pattern = 0;\n"
          "  buffer[i] = pattern;\n"
          "}\n";
        result = c_dbcsr_acc_opencl_kernel(source, "memset_zero"/*kernel_name*/,
          NULL/*build_params*/, NULL/*build_options*/,
          NULL/*try_build_options*/, NULL/*try_ok*/,
          NULL/*extnames*/, 0/*num_exts*/,
          &kernel);
      }
      if (EXIT_SUCCESS == result) {
        assert(NULL != kernel);
        ACC_OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), buffer),
          "set buffer argument of memset_zero kernel", result);
        ACC_OPENCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1/*work_dim*/,
          &offset, &nbytes, NULL/*local_work_size*/, 0, NULL, NULL),
          "launch memset_zero kernel", result);
      }
      LIBXSMM_ATOMIC_RELEASE(&lock, LIBXSMM_ATOMIC_RELAXED);
#else
      static const cl_uchar pattern = 0; /* fill with zeros */
      result = clEnqueueFillBuffer(queue, *buffer,
        &pattern, sizeof(pattern), offset, nbytes, 0, NULL, NULL);
#endif
    }
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_info_devmem(cl_device_id device,
  size_t* mem_free, size_t* mem_total, size_t* mem_local,
  int* mem_unified)
{
  int result = EXIT_SUCCESS, unified = 0;
  size_t size_free = 0, size_total = 0, size_local = 0;
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
  long pages_free = 0, pages_total = 0;
# if defined(__linux__)
#   if defined(_SC_PHYS_PAGES)
  pages_total = sysconf(_SC_PHYS_PAGES);
#   else
  pages_total = 0;
#   endif
#   if defined(_SC_AVPHYS_PAGES)
  pages_free = sysconf(_SC_AVPHYS_PAGES);
#   else
  pages_free = pages_total;
#   endif
# elif defined(__APPLE__) && defined(__MACH__)
  /*const*/ size_t size_pages_free = sizeof(const long), size_pages_total = sizeof(const long);
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
    cl_device_local_mem_type cl_local_type = CL_GLOBAL;
    cl_ulong cl_size_total = 0, cl_size_local = 0;
    cl_bool cl_unified = CL_FALSE;
    ACC_OPENCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
      sizeof(cl_ulong), &cl_size_total, NULL),
      "retrieve amount of global memory", result);
    ACC_OPENCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE,
      sizeof(cl_device_local_mem_type), &cl_local_type, NULL),
      "retrieve kind of local memory", result);
    if (CL_LOCAL == cl_local_type) {
      ACC_OPENCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
        sizeof(cl_ulong), &cl_size_local, NULL),
        "retrieve amount of local memory", result);
    }
    ACC_OPENCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY,
      sizeof(cl_bool), &cl_unified, NULL),
      "retrieve if host memory is unified", result);
    if (EXIT_SUCCESS == result) {
      if (cl_size_total < size_total) size_total = cl_size_total;
      if (size_total < size_free) size_free = size_total;
      size_local = cl_size_local;
      unified = cl_unified;
    }
  }
  result = (size_free <= size_total ? EXIT_SUCCESS : EXIT_FAILURE);
  assert(NULL != mem_local || NULL != mem_total || NULL != mem_free || NULL != mem_unified);
  if (NULL != mem_unified) *mem_unified = unified;
  if (NULL != mem_local) *mem_local = size_local;
  if (NULL != mem_total) *mem_total = size_total;
  if (NULL != mem_free)  *mem_free  = size_free;
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_dev_mem_info(size_t* mem_free, size_t* mem_total)
{
  cl_device_id active_id = NULL;
  int result = 0 < c_dbcsr_acc_opencl_config.ndevices
    ? c_dbcsr_acc_opencl_device(ACC_OPENCL_OMP_TID(), &active_id)
    : EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    result = c_dbcsr_acc_opencl_info_devmem(
      active_id, mem_free, mem_total,
      NULL/*mem_local*/, NULL/*mem_unified*/);
  }
  ACC_OPENCL_RETURN(result);
}

#if defined(__cplusplus)
}
#endif

#endif /*__OPENCL*/
