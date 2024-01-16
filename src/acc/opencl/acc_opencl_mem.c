/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/
#if defined(__OPENCL)
#  include "acc_opencl.h"
#  include <string.h>
#  if defined(_WIN32)
#    include <Windows.h>
#  else
#    if !defined(__linux__) && defined(__APPLE__) && defined(__MACH__)
#      include <sys/types.h>
#      include <sys/sysctl.h>
#    endif
#    include <unistd.h>
#  endif

#  if !defined(ACC_OPENCL_MEM_DEBUG) && !defined(NDEBUG) && 0
#    define ACC_OPENCL_MEM_DEBUG
#  endif
#  if !defined(ACC_OPENCL_MEM_ALIGNSCALE)
#    define ACC_OPENCL_MEM_ALIGNSCALE 8
#  endif


#  if defined(__cplusplus)
extern "C" {
#  endif

int c_dbcsr_acc_opencl_memalignment(size_t /*size*/);
int c_dbcsr_acc_opencl_memalignment(size_t size) {
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


void* c_dbcsr_acc_opencl_get_hostptr(cl_mem memory) {
  void* result = NULL;
  ACC_OPENCL_EXPECT(CL_SUCCESS == clGetMemObjectInfo(memory, CL_MEM_HOST_PTR, sizeof(void*), &result, NULL));
  return result;
}


c_dbcsr_acc_opencl_info_hostptr_t* c_dbcsr_acc_opencl_info_hostptr(void* memory) {
  assert(NULL == memory || sizeof(c_dbcsr_acc_opencl_info_hostptr_t) <= (uintptr_t)memory);
  return (NULL != memory ? (c_dbcsr_acc_opencl_info_hostptr_t*)((uintptr_t)memory - sizeof(c_dbcsr_acc_opencl_info_hostptr_t))
                         : (c_dbcsr_acc_opencl_info_hostptr_t*)NULL);
}


void* c_dbcsr_acc_opencl_info_devptr(const void* memory, size_t elsize, const size_t* amount, size_t* offset) {
  void* result = NULL;
#  if defined(ACC_OPENCL_MEM_OFFSET) && LIBXSMM_VERSION4(1, 17, 0, 0) < LIBXSMM_VERSION_NUMBER && \
    defined(ACC_OPENCL_HANDLES_MAXCOUNT) && (0 < ACC_OPENCL_HANDLES_MAXCOUNT)
  if (NULL != c_dbcsr_acc_opencl_config.clmems && NULL != memory && 0 < elsize) {
    const char* const buffer = (const char*)memory;
    const size_t n = ACC_OPENCL_HANDLES_MAXCOUNT * c_dbcsr_acc_opencl_config.nthreads;
    size_t i = c_dbcsr_acc_opencl_config.nclmems, hit = (size_t)-1;
    for (; i < n; ++i) {
      void** const handle = c_dbcsr_acc_opencl_config.clmems[i];
      char* const mem = (char*)(NULL != handle ? *handle : NULL);
      if (mem == buffer) { /* fast-path */
        if (NULL != offset) *offset = 0;
        assert(NULL != mem);
        result = handle;
        break;
      }
      else if (NULL != mem && mem < buffer && NULL != offset) {
        size_t d = buffer - mem, s = 0;
        if (d < hit && CL_SUCCESS == clGetMemObjectInfo((cl_mem)mem, CL_MEM_SIZE, sizeof(size_t), &s, NULL) &&
            (1 == elsize || (0 == (d % elsize) && 0 == (s % elsize))) && (NULL == amount || (*amount + d) <= s))
        {
          *offset = (1 == elsize ? d : (d / elsize));
          result = handle;
          hit = d;
        }
      }
    }
  }
#  else
  LIBXSMM_UNUSED(memory);
  LIBXSMM_UNUSED(elsize);
  LIBXSMM_UNUSED(amount);
  LIBXSMM_UNUSED(offset);
#  endif
  return result;
}


int c_dbcsr_acc_host_mem_allocate(void** host_mem, size_t nbytes, void* stream) {
  c_dbcsr_acc_opencl_info_stream_t* const info = c_dbcsr_acc_opencl_info_stream(stream);
  const size_t size_meminfo = sizeof(c_dbcsr_acc_opencl_info_hostptr_t);
  const int alignment = c_dbcsr_acc_opencl_memalignment(nbytes);
  void* host_ptr = NULL;
  cl_mem memory = NULL;
  cl_int result;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  nbytes += alignment + size_meminfo - 1;
  assert(NULL != host_mem && NULL != info);
#  if defined(CL_VERSION_2_0)
  if (0 != c_dbcsr_acc_opencl_config.device[info->tid].svm_interop) {
    host_ptr = clSVMAlloc(
      c_dbcsr_acc_opencl_config.device[info->tid].context, CL_MEM_READ_WRITE, nbytes, sizeof(void*) /*minimal alignment*/);
    if (NULL == host_ptr) c_dbcsr_acc_opencl_config.device[info->tid].svm_interop = 0; /* sanitize */
  }
#  endif
  memory = clCreateBuffer(c_dbcsr_acc_opencl_config.device[info->tid].context,
    NULL == host_ptr ? CL_MEM_ALLOC_HOST_PTR : CL_MEM_USE_HOST_PTR, nbytes, host_ptr, &result);
  assert(CL_SUCCESS == result || NULL == memory);
  if (CL_SUCCESS == result) {
    /*const*/ cl_command_queue queue = *ACC_OPENCL_STREAM(
#  if defined(ACC_OPENCL_STREAM_NULL)
      NULL == stream ? c_dbcsr_acc_opencl_stream_default() :
#  endif
                     stream);
    void* const mapped = clEnqueueMapBuffer(
      queue, memory, CL_TRUE /*blocking*/, CL_MAP_READ | CL_MAP_WRITE, 0 /*offset*/, nbytes, 0, NULL, NULL, &result);
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
      else { /* error: buffer info */
        result = EXIT_FAILURE;
        *host_mem = NULL;
      }
#  if defined(ACC_OPENCL_STREAM_NULL)
      if (NULL == stream && EXIT_SUCCESS == result) {
        result = c_dbcsr_acc_stream_sync(&queue);
      }
#  endif
    }
    else { /* error: mapping host buffer */
      ACC_OPENCL_EXPECT(CL_SUCCESS == clReleaseMemObject(memory));
      *host_mem = NULL;
    }
  }
  else {
    *host_mem = NULL; /* error: creating host buffer */
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_host_mem_deallocate(void* host_mem, void* stream) {
  int result;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  if (NULL != host_mem) {
    c_dbcsr_acc_opencl_info_hostptr_t* const meminfo = c_dbcsr_acc_opencl_info_hostptr(host_mem);
    if (NULL != meminfo->memory) {
      const c_dbcsr_acc_opencl_info_hostptr_t info = *meminfo; /* copy meminfo prior to unmap */
      /*const*/ cl_command_queue queue = *ACC_OPENCL_STREAM(
#  if defined(ACC_OPENCL_STREAM_NULL)
        NULL == stream ? c_dbcsr_acc_opencl_stream_default() :
#  endif
                       stream);
      int result_release;
      result = clEnqueueUnmapMemObject(queue, info.memory, info.mapped, 0, NULL, NULL);
#  if defined(CL_VERSION_2_0)
      {
        const c_dbcsr_acc_opencl_info_stream_t* const qinfo = c_dbcsr_acc_opencl_info_stream(stream);
        assert(NULL != qinfo);
        if (0 != c_dbcsr_acc_opencl_config.device[qinfo->tid].svm_interop) {
          clSVMFree(c_dbcsr_acc_opencl_config.device[qinfo->tid].context, info.mapped);
        }
      }
#  endif
#  if defined(ACC_OPENCL_STREAM_NULL)
      if (NULL == stream && EXIT_SUCCESS == result) {
        result = c_dbcsr_acc_stream_sync(&queue);
      }
#  endif
      result_release = clReleaseMemObject(info.memory);
      if (EXIT_SUCCESS == result) result = result_release;
    }
    else {
      result = EXIT_FAILURE;
    }
  }
  else {
    result = EXIT_FAILURE;
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_dev_mem_allocate(void** dev_mem, size_t nbytes) {
  cl_int result;
  int tid = 0;
  const cl_context context = c_dbcsr_acc_opencl_context(&tid);
  const int devuid = c_dbcsr_acc_opencl_config.device[tid].uid,
            try_flag = ((0 != c_dbcsr_acc_opencl_config.device[tid].unified || 0 == c_dbcsr_acc_opencl_config.device[tid].intel ||
                          (0x4905 != devuid && 0x020a != devuid && (0x0bd0 > devuid || 0x0bdb < devuid)))
                          ? 0
                          : (1u << 22));
  cl_mem buffer;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(NULL != dev_mem && 0 <= ACC_OPENCL_OVERMALLOC);
  assert(sizeof(void*) >= sizeof(cl_mem));
  assert(NULL != context);
  buffer = (
#  if defined(CL_VERSION_2_0)
    0 != c_dbcsr_acc_opencl_config.device[tid].svm_interop
      ? clCreateBuffer(context, CL_MEM_USE_HOST_PTR, nbytes + ACC_OPENCL_OVERMALLOC,
          clSVMAlloc(
            context, (cl_mem_flags)(CL_MEM_READ_WRITE | try_flag), nbytes + ACC_OPENCL_OVERMALLOC, 0 /*default alignment*/),
          &result)
      :
#  endif
      clCreateBuffer(
        context, (cl_mem_flags)(CL_MEM_READ_WRITE | try_flag), nbytes + ACC_OPENCL_OVERMALLOC, NULL /*host_ptr*/, &result));
  if (0 != try_flag && CL_SUCCESS != result) { /* retry without try_flag */
    buffer = (
#  if defined(CL_VERSION_2_0)
      0 != c_dbcsr_acc_opencl_config.device[tid].svm_interop
        ? clCreateBuffer(context, CL_MEM_USE_HOST_PTR, nbytes + ACC_OPENCL_OVERMALLOC,
            clSVMAlloc(context, CL_MEM_READ_WRITE, nbytes + ACC_OPENCL_OVERMALLOC, 0 /*default alignment*/), &result)
        :
#  endif
        clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes + ACC_OPENCL_OVERMALLOC, NULL /*host_ptr*/, &result));
  }
  if (EXIT_SUCCESS == result) {
    assert(NULL != buffer);
#  if defined(ACC_OPENCL_MEM_OFFSET) && LIBXSMM_VERSION4(1, 17, 0, 0) < LIBXSMM_VERSION_NUMBER && \
    defined(ACC_OPENCL_HANDLES_MAXCOUNT) && (0 < ACC_OPENCL_HANDLES_MAXCOUNT)
    assert(NULL != c_dbcsr_acc_opencl_config.clmems);
    {
      void** handle = libxsmm_pmalloc(c_dbcsr_acc_opencl_config.clmems, &c_dbcsr_acc_opencl_config.nclmems);
      if (NULL != handle) {
        *handle = buffer;
#    if defined(ACC_OPENCL_MEM_DEBUG)
        printf("c_dbcsr_acc_dev_mem_allocate: %p size=%llu\n", buffer, (unsigned long long)nbytes);
#    endif
      }
      else result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS != result) {
      *dev_mem = NULL; /* TODO: clReleaseMemObject */
    }
    else
#  endif
    {
      *dev_mem = (void*)buffer;
    }
  }
  else {
    *dev_mem = NULL; /* error: creating device buffer */
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_dev_mem_deallocate(void* dev_mem) {
  int result = EXIT_SUCCESS;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  if (NULL != dev_mem) {
    const cl_mem buffer = (cl_mem)dev_mem;
    assert(sizeof(void*) >= sizeof(cl_mem));
#  if defined(ACC_OPENCL_MEM_OFFSET) && LIBXSMM_VERSION4(1, 17, 0, 0) < LIBXSMM_VERSION_NUMBER && \
    defined(ACC_OPENCL_HANDLES_MAXCOUNT) && (0 < ACC_OPENCL_HANDLES_MAXCOUNT)
    assert(NULL != c_dbcsr_acc_opencl_config.clmems);
#    if defined(_OPENMP)
#      pragma omp critical(c_dbcsr_acc_dev_mem_deallocate)
#    endif
    {
      void** handle = c_dbcsr_acc_opencl_info_devptr(dev_mem, 1 /*elsize*/, NULL /*amount*/, NULL /*offset*/);
      if (NULL != handle) {
        void** const pfree = c_dbcsr_acc_opencl_config.clmems[c_dbcsr_acc_opencl_config.nclmems];
        libxsmm_pfree(pfree, c_dbcsr_acc_opencl_config.clmems, &c_dbcsr_acc_opencl_config.nclmems);
        *handle = *pfree;
#    if defined(ACC_OPENCL_MEM_DEBUG)
        printf("c_dbcsr_acc_dev_mem_deallocate: %p\n", buffer);
#    endif
      }
#    if !defined(NDEBUG) || defined(ACC_OPENCL_MEM_DEBUG)
      else result = EXIT_FAILURE;
#    endif
    }
#  endif
#  if defined(CL_VERSION_2_0)
    {
      const int tid = ACC_OPENCL_OMP_TID();
      if (0 != c_dbcsr_acc_opencl_config.device[tid].svm_interop) {
        void* const ptr = (0 != c_dbcsr_acc_opencl_config.device[tid].svm_interop ? c_dbcsr_acc_opencl_get_hostptr(buffer) : NULL);
        const cl_context context = c_dbcsr_acc_opencl_context(NULL /*thread_id*/);
        clSVMFree(context, ptr);
      }
    }
#  endif
    ACC_OPENCL_CHECK(clReleaseMemObject(buffer), "release device memory buffer", result);
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_dev_mem_set_ptr(void** dev_mem, void* other, size_t lb) {
  int result;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(NULL != dev_mem);
  if (NULL != other || 0 == lb) {
    *dev_mem = (char*)other + lb;
    result = EXIT_SUCCESS;
  }
  else {
    result = EXIT_FAILURE;
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_memcpy_h2d(const void* host_mem, void* dev_mem, size_t nbytes, void* stream) {
  int result = EXIT_SUCCESS;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert((NULL != host_mem || 0 == nbytes) && (NULL != dev_mem || 0 == nbytes));
  if (NULL != host_mem && NULL != dev_mem && 0 != nbytes) {
    cl_mem buffer = (cl_mem)dev_mem;
    size_t offset = 0;
#  if defined(ACC_OPENCL_MEM_OFFSET) && LIBXSMM_VERSION4(1, 17, 0, 0) < LIBXSMM_VERSION_NUMBER && \
    defined(ACC_OPENCL_HANDLES_MAXCOUNT) && (0 < ACC_OPENCL_HANDLES_MAXCOUNT)
    assert(NULL != c_dbcsr_acc_opencl_config.clmems);
    {
      void* const handle = c_dbcsr_acc_opencl_info_devptr(dev_mem, 1 /*elsize*/, &nbytes, &offset);
      if (NULL != handle) buffer = *(cl_mem*)handle;
#    if !defined(NDEBUG) || defined(ACC_OPENCL_MEM_DEBUG)
      else result = EXIT_FAILURE;
#    endif
    }
#    if !defined(NDEBUG) || defined(ACC_OPENCL_MEM_DEBUG)
    if (EXIT_SUCCESS == result)
#    endif
#  endif
    {
      /*const*/ cl_command_queue queue = *ACC_OPENCL_STREAM(
#  if defined(ACC_OPENCL_STREAM_NULL)
        NULL == stream ? c_dbcsr_acc_opencl_stream_default() :
#  endif
                       stream);
      result = clEnqueueWriteBuffer(
        queue, buffer, 0 == (1 & c_dbcsr_acc_opencl_config.async), offset, nbytes, host_mem, 0, NULL, NULL);
#  if defined(ACC_OPENCL_STREAM_NULL)
      if (NULL == stream && EXIT_SUCCESS == result) {
        result = c_dbcsr_acc_stream_sync(&queue);
      }
#  endif
    }
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_memcpy_d2h(const void* dev_mem, void* host_mem, size_t nbytes, void* stream) {
  int result = EXIT_SUCCESS;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert((NULL != dev_mem || 0 == nbytes) && (NULL != host_mem || 0 == nbytes));
  if (NULL != host_mem && NULL != dev_mem && 0 != nbytes) {
    cl_mem buffer = NULL;
    size_t offset = 0;
    LIBXSMM_ASSIGN127(&buffer, &dev_mem);
#  if defined(ACC_OPENCL_MEM_OFFSET) && LIBXSMM_VERSION4(1, 17, 0, 0) < LIBXSMM_VERSION_NUMBER && \
    defined(ACC_OPENCL_HANDLES_MAXCOUNT) && (0 < ACC_OPENCL_HANDLES_MAXCOUNT)
    assert(NULL != c_dbcsr_acc_opencl_config.clmems);
    {
      void* const handle = c_dbcsr_acc_opencl_info_devptr(dev_mem, 1 /*elsize*/, &nbytes, &offset);
      if (NULL != handle) buffer = *(cl_mem*)handle;
#    if !defined(NDEBUG) || defined(ACC_OPENCL_MEM_DEBUG)
      else result = EXIT_FAILURE;
#    endif
    }
#    if !defined(NDEBUG) || defined(ACC_OPENCL_MEM_DEBUG)
    if (EXIT_SUCCESS == result)
#    endif
#  endif
    {
      /*const*/ cl_command_queue queue = *ACC_OPENCL_STREAM(
#  if defined(ACC_OPENCL_STREAM_NULL)
        NULL == stream ? c_dbcsr_acc_opencl_stream_default() :
#  endif
                       stream);
      result = clEnqueueReadBuffer(
        queue, buffer, 0 == (2 & c_dbcsr_acc_opencl_config.async), offset, nbytes, host_mem, 0, NULL, NULL);
      if (CL_SUCCESS == result) {
#  if defined(ACC_OPENCL_STREAM_NULL)
        if (NULL == stream) result = c_dbcsr_acc_stream_sync(&queue);
#  endif
      }
      else { /* synchronous */
        const int result_sync = clEnqueueReadBuffer(queue, buffer, CL_TRUE, offset, nbytes, host_mem, 0, NULL, NULL);
        c_dbcsr_acc_opencl_config.async |= 2; /* retract feature */
        if (0 != c_dbcsr_acc_opencl_config.verbosity) {
          fprintf(stderr, "WARN ACC/OpenCL: falling back to synchronous readback (code=%i).\n", result);
        }
        result = result_sync;
      }
    }
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_memcpy_d2d(const void* devmem_src, void* devmem_dst, size_t nbytes, void* stream) {
  int result = EXIT_SUCCESS;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert((NULL != devmem_src || 0 == nbytes) && (NULL != devmem_dst || 0 == nbytes));
  if (NULL != devmem_src && NULL != devmem_dst && 0 != nbytes) {
    cl_mem src = NULL, dst = (cl_mem)devmem_dst;
    size_t src_offset = 0, dst_offset = 0;
    LIBXSMM_ASSIGN127(&src, &devmem_src);
#  if defined(ACC_OPENCL_MEM_OFFSET) && LIBXSMM_VERSION4(1, 17, 0, 0) < LIBXSMM_VERSION_NUMBER && \
    defined(ACC_OPENCL_HANDLES_MAXCOUNT) && (0 < ACC_OPENCL_HANDLES_MAXCOUNT)
    assert(NULL != c_dbcsr_acc_opencl_config.clmems);
    {
      void* const handle_src = c_dbcsr_acc_opencl_info_devptr(devmem_src, 1 /*elsize*/, &nbytes, &src_offset);
      void* const handle_dst = c_dbcsr_acc_opencl_info_devptr(devmem_dst, 1 /*elsize*/, &nbytes, &dst_offset);
      if (NULL != handle_src) src = *(cl_mem*)handle_src;
#    if !defined(NDEBUG) || defined(ACC_OPENCL_MEM_DEBUG)
      else result = EXIT_FAILURE;
#    endif
      if (NULL != handle_dst) dst = *(cl_mem*)handle_dst;
#    if !defined(NDEBUG) || defined(ACC_OPENCL_MEM_DEBUG)
      else result = EXIT_FAILURE;
#    endif
    }
#    if !defined(NDEBUG) || defined(ACC_OPENCL_MEM_DEBUG)
    if (EXIT_SUCCESS == result)
#    endif
#  endif
    {
      /*const*/ cl_command_queue queue = *ACC_OPENCL_STREAM(
#  if defined(ACC_OPENCL_STREAM_NULL)
        NULL == stream ? c_dbcsr_acc_opencl_stream_default() :
#  endif
                       stream);
      if (0 == (2 & c_dbcsr_acc_opencl_config.devcopy)) {
        result = clEnqueueCopyBuffer(queue, src, dst, src_offset, dst_offset, nbytes, 0, NULL, NULL);
      }
      else {
        static volatile int lock; /* creating cl_kernel and clSetKernelArg must be synchronized */
        static cl_kernel kernel = NULL;
        LIBXSMM_ATOMIC_ACQUIRE(&lock, LIBXSMM_SYNC_NPAUSE, LIBXSMM_ATOMIC_RELAXED);
        if (NULL == kernel) { /* generate kernel */
          const char source[] = "kernel void memcpy_d2d(\n"
                                "  global uchar *restrict dst, size_t dst_offset,\n"
                                "  global uchar *restrict src, size_t src_offset)\n"
                                "{\n"
                                "  const size_t i = get_global_id(0);\n"
                                "  dst[i+dst_offset] = src[i+src_offset];\n"
                                "}\n";
          result = c_dbcsr_acc_opencl_kernel(0 /*source_is_file*/, source, "memcpy_d2d" /*kernel_name*/, NULL /*build_params*/,
            NULL /*build_options*/, NULL /*try_build_options*/, NULL /*try_ok*/, NULL /*extnames*/, 0 /*num_exts*/, &kernel);
        }
        if (EXIT_SUCCESS == result) {
          ACC_OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &dst), "set src argument of memcpy_d2d kernel", result);
          ACC_OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_offset), "set dst-offset of memcpy_d2d kernel", result);
          ACC_OPENCL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &src), "set dst argument of memcpy_d2d kernel", result);
          ACC_OPENCL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &src_offset), "set src-offset of memcpy_d2d kernel", result);
          ACC_OPENCL_CHECK(clEnqueueNDRangeKernel(
                             queue, kernel, 1 /*work_dim*/, NULL /*offset*/, &nbytes, NULL /*local_work_size*/, 0, NULL, NULL),
            "launch memcpy_d2d kernel", result);
        }
        LIBXSMM_ATOMIC_RELEASE(&lock, LIBXSMM_ATOMIC_RELAXED);
      }
#  if defined(ACC_OPENCL_STREAM_NULL)
      if (NULL == stream && EXIT_SUCCESS == result) {
        result = c_dbcsr_acc_stream_sync(&queue);
      }
#  endif
    }
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_memset(void* dev_mem, int value, size_t offset, size_t nbytes, void* stream) {
  int result = EXIT_SUCCESS;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(NULL != dev_mem || 0 == nbytes);
  if (0 != nbytes) {
    cl_mem buffer = (cl_mem)dev_mem;
#  if defined(ACC_OPENCL_MEM_OFFSET) && LIBXSMM_VERSION4(1, 17, 0, 0) < LIBXSMM_VERSION_NUMBER && \
    defined(ACC_OPENCL_HANDLES_MAXCOUNT) && (0 < ACC_OPENCL_HANDLES_MAXCOUNT)
    if (0 == offset && NULL != c_dbcsr_acc_opencl_config.clmems) {
      void* const handle = c_dbcsr_acc_opencl_info_devptr(dev_mem, 1 /*elsize*/, &nbytes, &offset);
      if (NULL != handle) buffer = *(cl_mem*)handle;
#    if !defined(NDEBUG) || defined(ACC_OPENCL_MEM_DEBUG)
      else result = EXIT_FAILURE;
#    endif
    }
#    if !defined(NDEBUG) || defined(ACC_OPENCL_MEM_DEBUG)
    if (EXIT_SUCCESS == result)
#    endif
#  endif
    {
      /*const*/ cl_command_queue queue = *ACC_OPENCL_STREAM(
#  if defined(ACC_OPENCL_STREAM_NULL)
        NULL == stream ? c_dbcsr_acc_opencl_stream_default() :
#  endif
                       stream);
      if (0 == (1 & c_dbcsr_acc_opencl_config.devcopy)) {
        static LIBXSMM_TLS cl_long pattern = 0;
        size_t size_of_pattern = 1;
        pattern = value; /* fill with value */
        if (0 == LIBXSMM_MOD2(nbytes, sizeof(cl_long))) size_of_pattern = sizeof(cl_long);
        else if (0 == LIBXSMM_MOD2(nbytes, 4)) size_of_pattern = 4;
        else if (0 == LIBXSMM_MOD2(nbytes, 2)) size_of_pattern = 2;
        result = clEnqueueFillBuffer(queue, buffer, &pattern, size_of_pattern, offset, nbytes, 0, NULL, NULL);
      }
      else {
        static volatile int lock; /* creating cl_kernel and clSetKernelArg must be synchronized */
        static cl_kernel kernel = NULL;
        LIBXSMM_ATOMIC_ACQUIRE(&lock, LIBXSMM_SYNC_NPAUSE, LIBXSMM_ATOMIC_RELAXED);
        if (NULL == kernel) { /* generate kernel */
          const char source[] = "kernel void memset(global uchar *restrict buffer, uchar value) {\n"
                                "  buffer[get_global_id(0)] = value;\n"
                                "}\n";
          result = c_dbcsr_acc_opencl_kernel(0 /*source_is_file*/, source, "memset" /*kernel_name*/, NULL /*build_params*/,
            NULL /*build_options*/, NULL /*try_build_options*/, NULL /*try_ok*/, NULL /*extnames*/, 0 /*num_exts*/, &kernel);
        }
        if (EXIT_SUCCESS == result) {
          ACC_OPENCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer), "set buffer argument of memset-kernel", result);
          ACC_OPENCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_uchar), &value), "set value argument of memset-kernel", result);
          ACC_OPENCL_CHECK(
            clEnqueueNDRangeKernel(queue, kernel, 1 /*work_dim*/, &offset, &nbytes, NULL /*local_work_size*/, 0, NULL, NULL),
            "launch memset-kernel", result);
        }
        LIBXSMM_ATOMIC_RELEASE(&lock, LIBXSMM_ATOMIC_RELAXED);
      }
#  if defined(ACC_OPENCL_STREAM_NULL)
      if (NULL == stream && EXIT_SUCCESS == result) {
        result = c_dbcsr_acc_stream_sync(&queue);
      }
#  endif
    }
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_memset_zero(void* dev_mem, size_t offset, size_t nbytes, void* stream) {
  return c_dbcsr_acc_opencl_memset(dev_mem, 0 /*value*/, offset, nbytes, stream);
}


int c_dbcsr_acc_opencl_info_devmem(cl_device_id device, size_t* mem_free, size_t* mem_total, size_t* mem_local, int* mem_unified) {
  int result = EXIT_SUCCESS, unified = 0;
  size_t size_free = 0, size_total = 0, size_local = 0;
#  if defined(_WIN32)
  MEMORYSTATUSEX mem_status;
  mem_status.dwLength = sizeof(mem_status);
  if (GlobalMemoryStatusEx(&mem_status)) {
    size_total = (size_t)mem_status.ullTotalPhys;
    size_free = (size_t)mem_status.ullAvailPhys;
  }
#  else
#    if defined(_SC_PAGE_SIZE)
  const long page_size = sysconf(_SC_PAGE_SIZE);
#    else
  const long page_size = 4096;
#    endif
  long pages_free = 0, pages_total = 0;
#    if defined(__linux__)
#      if defined(_SC_PHYS_PAGES)
  pages_total = sysconf(_SC_PHYS_PAGES);
#      else
  pages_total = 0;
#      endif
#      if defined(_SC_AVPHYS_PAGES)
  pages_free = sysconf(_SC_AVPHYS_PAGES);
#      else
  pages_free = pages_total;
#      endif
#    elif defined(__APPLE__) && defined(__MACH__)
  /*const*/ size_t size_pages_free = sizeof(const long), size_pages_total = sizeof(const long);
  ACC_OPENCL_EXPECT(0 == sysctlbyname("hw.memsize", &pages_total, &size_pages_total, NULL, 0));
  if (0 < page_size) pages_total /= page_size;
  if (0 != sysctlbyname("vm.page_free_count", &pages_free, &size_pages_free, NULL, 0)) {
    pages_free = pages_total;
  }
#    endif
  if (0 < page_size && 0 <= pages_free && 0 <= pages_total) {
    const size_t size_page = (size_t)page_size;
    size_total = size_page * (size_t)pages_total;
    size_free = size_page * (size_t)pages_free;
  }
#  endif
  if (NULL != device) {
    cl_device_local_mem_type cl_local_type = CL_GLOBAL;
    cl_ulong cl_size_total = 0, cl_size_local = 0;
    cl_bool cl_unified = CL_FALSE;
    ACC_OPENCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &cl_size_total, NULL),
      "retrieve amount of global memory", result);
    ACC_OPENCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(cl_device_local_mem_type), &cl_local_type, NULL),
      "retrieve kind of local memory", result);
    if (CL_LOCAL == cl_local_type) {
      ACC_OPENCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &cl_size_local, NULL),
        "retrieve amount of local memory", result);
    }
    ACC_OPENCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &cl_unified, NULL),
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
  if (NULL != mem_free) *mem_free = size_free;
  return result;
}


int c_dbcsr_acc_dev_mem_info(size_t* mem_free, size_t* mem_total) {
  cl_device_id active_id = NULL;
  int result = 0 < c_dbcsr_acc_opencl_config.ndevices ? c_dbcsr_acc_opencl_device(ACC_OPENCL_OMP_TID(), &active_id) : EXIT_FAILURE;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  if (EXIT_SUCCESS == result) {
    result = c_dbcsr_acc_opencl_info_devmem(active_id, mem_free, mem_total, NULL /*mem_local*/, NULL /*mem_unified*/);
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}

#  if defined(__cplusplus)
}
#  endif

#endif /*__OPENCL*/
