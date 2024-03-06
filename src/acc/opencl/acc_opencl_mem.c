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

#  if !defined(ACC_OPENCL_MEM_ALIGNSCALE)
#    define ACC_OPENCL_MEM_ALIGNSCALE 8
#  endif


#  if defined(__cplusplus)
extern "C" {
#  endif

void c_dbcsr_acc_opencl_pmalloc_init(ACC_OPENCL_LOCKTYPE* lock, size_t size, size_t* num, void* pool[], void* storage) {
  char* p = (char*)storage;
  size_t n, i = 0;
  assert(0 < size && NULL != num && NULL != pool && NULL != storage);
  if (NULL != lock) ACC_OPENCL_ACQUIRE(lock);
  for (n = *num; i < n; ++i, p += size) pool[i] = p;
  if (NULL != lock) ACC_OPENCL_RELEASE(lock);
}


void* c_dbcsr_acc_opencl_pmalloc(ACC_OPENCL_LOCKTYPE* lock, void* pool[], size_t* i) {
  void* pointer;
  assert(NULL != pool && NULL != i);
  if (NULL != lock) ACC_OPENCL_ACQUIRE(lock);
  pointer = (0 < *i ? pool[--(*i)] : NULL);
  if (NULL != lock) ACC_OPENCL_RELEASE(lock);
  assert(NULL != pointer);
  return pointer;
}


void c_dbcsr_acc_opencl_pfree(ACC_OPENCL_LOCKTYPE* lock, const void* pointer, void* pool[], size_t* i) {
  assert(NULL != pool && NULL != i);
  if (NULL != pointer) {
    if (NULL != lock) ACC_OPENCL_ACQUIRE(lock);
    LIBXSMM_ASSIGN127(pool + *i, &pointer);
    ++(*i);
    if (NULL != lock) ACC_OPENCL_RELEASE(lock);
  }
}


c_dbcsr_acc_opencl_info_memptr_t* c_dbcsr_acc_opencl_info_hostptr(void* memory) {
  assert(NULL == memory || sizeof(c_dbcsr_acc_opencl_info_memptr_t) <= (uintptr_t)memory);
  return (NULL != memory ? (c_dbcsr_acc_opencl_info_memptr_t*)((uintptr_t)memory - sizeof(c_dbcsr_acc_opencl_info_memptr_t))
                         : (c_dbcsr_acc_opencl_info_memptr_t*)NULL);
}


c_dbcsr_acc_opencl_info_memptr_t* c_dbcsr_acc_opencl_info_devptr_modify(
  ACC_OPENCL_LOCKTYPE* lock, void* memory, size_t elsize, const size_t* amount, size_t* offset) {
  c_dbcsr_acc_opencl_info_memptr_t* result = NULL;
  assert(0 < elsize);
  if (NULL != memory) {
#  if defined(ACC_OPENCL_MEM_DEVPTR)
    if (NULL == c_dbcsr_acc_opencl_config.device.clSetKernelArgMemPointerINTEL) {
      const char* const pointer = (const char*)memory;
      const size_t n = ACC_OPENCL_MAXNITEMS * c_dbcsr_acc_opencl_config.nthreads;
      size_t hit = (size_t)-1, i;
      assert(NULL != c_dbcsr_acc_opencl_config.memptrs);
      if (NULL != lock) ACC_OPENCL_ACQUIRE(lock);
      for (i = c_dbcsr_acc_opencl_config.nmemptrs; i < n; ++i) {
        c_dbcsr_acc_opencl_info_memptr_t* const info = c_dbcsr_acc_opencl_config.memptrs[i];
        if (NULL != info) {
          char* const memptr = (char*)info->memptr;
          assert(NULL != memptr);
          if (memptr == pointer) { /* fast-path */
            if (NULL != offset) *offset = 0;
            result = info;
            break;
          }
          else if (memptr < pointer && NULL != offset) {
            size_t d = pointer - memptr, s = d;
            assert(0 != d);
            if (d < hit &&
#    if !defined(NDEBUG)
                (EXIT_SUCCESS == clGetMemObjectInfo(info->memory, CL_MEM_SIZE, sizeof(size_t), &s, NULL)) &&
                (NULL == amount || (*amount * elsize + d) <= s) &&
#    endif
                (1 == elsize || (0 == (d % elsize) && 0 == (s % elsize))) && d <= s)
            {
              *offset = (1 == elsize ? d : (d / elsize));
              result = info;
              hit = d;
            }
          }
        }
        else break;
      }
      if (NULL != lock) ACC_OPENCL_RELEASE(lock);
    }
    else
#  endif
    {
#  if defined(ACC_OPENCL_MEM_DEVPTR)
#    if defined(NDEBUG)
      LIBXSMM_UNUSED(amount);
#    endif
#  else
      LIBXSMM_UNUSED(lock);
      LIBXSMM_UNUSED(elsize);
      LIBXSMM_UNUSED(amount);
#  endif
      /* assume only first item of c_dbcsr_acc_opencl_info_memptr_t is accessed */
      result = (c_dbcsr_acc_opencl_info_memptr_t*)memory;
      if (NULL != offset) *offset = 0;
    }
  }
  return result;
}


int c_dbcsr_acc_opencl_info_devptr_lock(c_dbcsr_acc_opencl_info_memptr_t* info, ACC_OPENCL_LOCKTYPE* lock, const void* memory,
  size_t elsize, const size_t* amount, size_t* offset) {
  c_dbcsr_acc_opencl_info_memptr_t* devptr = NULL;
  int result = EXIT_SUCCESS;
  void* non_const;
  LIBXSMM_ASSIGN127(&non_const, &memory);
  assert(NULL != info);
  devptr = c_dbcsr_acc_opencl_info_devptr_modify(lock, non_const, elsize, amount, offset);
  if (NULL != devptr) {
#  if defined(ACC_OPENCL_MEM_DEVPTR)
    if (NULL == c_dbcsr_acc_opencl_config.device.clSetKernelArgMemPointerINTEL) {
      LIBXSMM_ASSIGN127(info, devptr);
    }
    else
#  endif
    {
#  if !defined(NDEBUG)
      LIBXSMM_MEMZERO127(info);
#  endif
      info->memory = (cl_mem)devptr;
    }
  }
  else result = EXIT_FAILURE;
  return result;
}


int c_dbcsr_acc_opencl_info_devptr(
  c_dbcsr_acc_opencl_info_memptr_t* info, const void* memory, size_t elsize, const size_t* amount, size_t* offset) {
  return c_dbcsr_acc_opencl_info_devptr_lock(info, c_dbcsr_acc_opencl_config.lock_memory, memory, elsize, amount, offset);
}


int c_dbcsr_acc_host_mem_allocate(void** host_mem, size_t nbytes, void* stream) {
  const size_t size_meminfo = sizeof(c_dbcsr_acc_opencl_info_memptr_t);
  int result = EXIT_SUCCESS, alignment = sizeof(void*);
  cl_mem memory = NULL;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  if ((ACC_OPENCL_MEM_ALIGNSCALE * ACC_OPENCL_CACHELINE) <= nbytes) {
    alignment = ((ACC_OPENCL_MEM_ALIGNSCALE * ACC_OPENCL_MAXALIGN) <= nbytes ? ACC_OPENCL_MAXALIGN : ACC_OPENCL_CACHELINE);
  }
  nbytes += alignment + size_meminfo - 1;
  assert(NULL != host_mem);
  memory = clCreateBuffer(c_dbcsr_acc_opencl_config.device.context, CL_MEM_ALLOC_HOST_PTR, nbytes, NULL /*host_ptr*/, &result);
  if (EXIT_SUCCESS == result) {
    const c_dbcsr_acc_opencl_stream_t* const str = (NULL != stream ? ACC_OPENCL_STREAM(stream)
                                                                   : c_dbcsr_acc_opencl_stream_default());
    void* const mapped = clEnqueueMapBuffer(
      str->queue, memory, CL_TRUE /*always block*/, CL_MAP_READ | CL_MAP_WRITE, 0 /*offset*/, nbytes, 0, NULL, NULL, &result);
    assert(EXIT_SUCCESS == result || NULL == mapped);
    if (EXIT_SUCCESS == result) {
      const uintptr_t address = (uintptr_t)mapped;
      const uintptr_t aligned = LIBXSMM_UP2(address + size_meminfo, alignment);
      c_dbcsr_acc_opencl_info_memptr_t* meminfo;
      assert(address + size_meminfo <= aligned);
      meminfo = (c_dbcsr_acc_opencl_info_memptr_t*)(aligned - size_meminfo);
      if (NULL != meminfo) {
        meminfo->memory = memory;
        meminfo->memptr = mapped;
        *host_mem = (void*)aligned;
      }
      else result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS != result) {
    if (NULL != memory) ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseMemObject(memory));
    *host_mem = NULL;
  }
  assert(EXIT_SUCCESS == result || NULL == *host_mem);
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_host_mem_deallocate(void* host_mem, void* stream) {
  int result = EXIT_SUCCESS;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  if (NULL != host_mem) {
    c_dbcsr_acc_opencl_info_memptr_t* const meminfo = c_dbcsr_acc_opencl_info_hostptr(host_mem);
    if (NULL != meminfo->memory) {
      const c_dbcsr_acc_opencl_info_memptr_t info = *meminfo; /* copy meminfo prior to unmap */
      const c_dbcsr_acc_opencl_stream_t* const str = (NULL != stream ? ACC_OPENCL_STREAM(stream)
                                                                     : c_dbcsr_acc_opencl_stream_default());
      int result_release;
      cl_event event;
      assert(NULL != str && NULL != str->queue);
      result = clEnqueueUnmapMemObject(str->queue, info.memory, info.memptr, 0, NULL, &event);
      if (NULL == stream && EXIT_SUCCESS == result) result = clWaitForEvents(1, &event);
      result_release = clReleaseMemObject(info.memory);
      if (EXIT_SUCCESS == result) result = result_release;
    }
    else result = EXIT_FAILURE;
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


/* like c_dbcsr_acc_memcpy_d2h, but apply some async workaround. */
int c_dbcsr_acc_opencl_memcpy_d2h(
  cl_mem /*dev_mem*/, void* /*host_mem*/, size_t /*offset*/, size_t /*nbytes*/, cl_command_queue /*queue*/, int /*blocking*/);
int c_dbcsr_acc_opencl_memcpy_d2h(
  cl_mem dev_mem, void* host_mem, size_t offset, size_t nbytes, cl_command_queue queue, int blocking) {
#  if defined(ACC_OPENCL_ASYNC)
  const cl_bool finish = (0 != blocking || 0 == (2 & c_dbcsr_acc_opencl_config.async) ||
                          (0 != c_dbcsr_acc_opencl_config.device.nv && NULL == (ACC_OPENCL_ASYNC)));
#  else
  const cl_bool finish = CL_TRUE;
#  endif
  int result = EXIT_SUCCESS;
#  if defined(ACC_OPENCL_MEM_DEVPTR)
  if (NULL != c_dbcsr_acc_opencl_config.device.clEnqueueMemcpyINTEL) {
    result = c_dbcsr_acc_opencl_config.device.clEnqueueMemcpyINTEL(queue, finish, host_mem, dev_mem, nbytes, 0, NULL, NULL);
  }
  else
#  endif
  {
    result = clEnqueueReadBuffer(queue, dev_mem, finish, offset, nbytes, host_mem, 0, NULL, NULL);
  }
  if (EXIT_SUCCESS != result && !finish) { /* retry synchronously */
    int result_sync = EXIT_SUCCESS;
#  if defined(ACC_OPENCL_MEM_DEVPTR)
    if (NULL != c_dbcsr_acc_opencl_config.device.clEnqueueMemcpyINTEL) {
      result_sync = c_dbcsr_acc_opencl_config.device.clEnqueueMemcpyINTEL(queue, CL_TRUE, host_mem, dev_mem, nbytes, 0, NULL, NULL);
    }
    else
#  endif
    {
      result_sync = clEnqueueReadBuffer(queue, dev_mem, CL_TRUE, offset, nbytes, host_mem, 0, NULL, NULL);
    }
    if (EXIT_SUCCESS == result_sync) {
      c_dbcsr_acc_opencl_config.async &= ~2; /* retract async feature */
      if (0 != c_dbcsr_acc_opencl_config.verbosity) {
        fprintf(stderr, "WARN ACC/OpenCL: falling back to synchronous readback (code=%i).\n", result);
      }
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


int c_dbcsr_acc_dev_mem_allocate(void** dev_mem, size_t nbytes) {
  int result = EXIT_SUCCESS;
  /* assume no lock is needed to protect against context/device changes */
  const cl_context context = c_dbcsr_acc_opencl_config.device.context;
  cl_mem memory = NULL;
  void* memptr = NULL;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(NULL != dev_mem && NULL != context);
#  if defined(ACC_OPENCL_MEM_DEVPTR)
  if (NULL != c_dbcsr_acc_opencl_config.device.clDeviceMemAllocINTEL) {
    cl_device_id device = NULL;
    result = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);
    if (EXIT_SUCCESS == result) {
      *dev_mem = memptr = c_dbcsr_acc_opencl_config.device.clDeviceMemAllocINTEL(
        context, device, NULL /*properties*/, nbytes, 0 /*alignment*/, &result);
    }
    if (EXIT_SUCCESS != result) *dev_mem = NULL;
  }
  else
#  endif
  {
    const int devuid = c_dbcsr_acc_opencl_config.device.uid;
    const int try_flag = ((0 != c_dbcsr_acc_opencl_config.device.unified || 0 == c_dbcsr_acc_opencl_config.device.intel ||
                            (0x4905 != devuid && 0x020a != devuid && (0x0bd0 > devuid || 0x0bdb < devuid)))
                            ? 0
                            : (1u << 22));
    memory = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE | try_flag), nbytes, NULL /*host_ptr*/, &result);
    if (0 != try_flag && EXIT_SUCCESS != result) { /* retry without try_flag */
      memory = clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL /*host_ptr*/, &result);
    }
    if (EXIT_SUCCESS == result) {
#  if defined(ACC_OPENCL_MEM_DEVPTR)
      const c_dbcsr_acc_opencl_stream_t* str = NULL;
      static cl_kernel kernel = NULL;
      const size_t size = 1;
      ACC_OPENCL_ACQUIRE(c_dbcsr_acc_opencl_config.lock_memory);
      str = c_dbcsr_acc_opencl_stream(NULL /*lock*/, ACC_OPENCL_OMP_TID());
      /* determine device-side value of device-memory object by running some kernel */
      assert(NULL != memory && NULL != str && NULL != str->queue);
      if (NULL == kernel) { /* generate kernel */
        const char source[] = "kernel void memptr(global unsigned long* ptr) {\n"
                              "  const union { global unsigned long* p; unsigned long u; } cast = { ptr };\n"
                              "  const size_t i = get_global_id(0);\n"
                              "  ptr[i] = cast.u + i;\n"
                              "}\n";
        assert(sizeof(size_t) == sizeof(cl_ulong));
        result = c_dbcsr_acc_opencl_kernel(0 /*source_is_file*/, source, "memptr" /*kernel_name*/, NULL /*build_params*/,
          NULL /*build_options*/, NULL /*try_build_options*/, NULL /*try_ok*/, NULL /*extnames*/, 0 /*num_exts*/, &kernel);
      }
      /* TODO: backup/restore memory */
      if (EXIT_SUCCESS == result) result = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memory);
      if (EXIT_SUCCESS == result) {
        result = clEnqueueNDRangeKernel(
          str->queue, kernel, 1 /*work_dim*/, NULL /*offset*/, &size, NULL /*local_work_size*/, 0, NULL, NULL);
      }
      if (EXIT_SUCCESS == result) {
        result = c_dbcsr_acc_opencl_memcpy_d2h(memory, &memptr, 0, sizeof(void*), str->queue, 1 /*blocking*/);
      }
      assert(EXIT_SUCCESS != result || NULL != memptr);
      if (EXIT_SUCCESS == result) {
        c_dbcsr_acc_opencl_info_memptr_t* const info = (c_dbcsr_acc_opencl_info_memptr_t*)c_dbcsr_acc_opencl_pmalloc(
          NULL /*lock*/, (void**)c_dbcsr_acc_opencl_config.memptrs, &c_dbcsr_acc_opencl_config.nmemptrs);
        assert(NULL != memptr);
        if (NULL != info) {
          info->memory = memory;
          info->memptr = memptr;
          *dev_mem = memptr;
        }
        else result = EXIT_FAILURE;
      }
      ACC_OPENCL_RELEASE(c_dbcsr_acc_opencl_config.lock_memory);
#  else
      *dev_mem = memptr = memory;
#  endif
    }
    if (EXIT_SUCCESS != result) {
      if (NULL != memory) ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseMemObject(memory));
      *dev_mem = NULL;
    }
  }
  if (0 != c_dbcsr_acc_opencl_config.debug && 0 != c_dbcsr_acc_opencl_config.verbosity && EXIT_SUCCESS == result) {
    fprintf(stderr, "INFO ACC/OpenCL: memory=%p pointer=%p size=%llu allocated\n", (const void*)memory, memptr,
      (unsigned long long)nbytes);
  }
  assert(EXIT_SUCCESS == result || NULL == *dev_mem);
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
    cl_mem memory = NULL;
#  if !defined(ACC_OPENCL_MEM_DEVPTR)
    memory = (cl_mem)dev_mem;
#  else
    if (NULL != c_dbcsr_acc_opencl_config.device.clMemFreeINTEL) {
      cl_device_id device = NULL;
      assert(NULL != c_dbcsr_acc_opencl_config.device.context);
      result = clGetContextInfo(c_dbcsr_acc_opencl_config.device.context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);
      if (EXIT_SUCCESS == result) {
        result = c_dbcsr_acc_opencl_config.device.clMemFreeINTEL(c_dbcsr_acc_opencl_config.device.context, dev_mem);
      }
    }
    else {
      c_dbcsr_acc_opencl_info_memptr_t* info = NULL;
      ACC_OPENCL_ACQUIRE(c_dbcsr_acc_opencl_config.lock_memory);
      info = c_dbcsr_acc_opencl_info_devptr_modify(NULL /*lock*/, dev_mem, 1 /*elsize*/, NULL /*amount*/, NULL /*offset*/);
      if (NULL != info && info->memptr == dev_mem && NULL != info->memory) {
        c_dbcsr_acc_opencl_info_memptr_t* const pfree = c_dbcsr_acc_opencl_config.memptrs[c_dbcsr_acc_opencl_config.nmemptrs];
        memory = info->memory;
#  endif
    result = clReleaseMemObject(memory);
#  if defined(ACC_OPENCL_MEM_DEVPTR)
    c_dbcsr_acc_opencl_pfree(NULL /*lock*/, pfree, (void**)c_dbcsr_acc_opencl_config.memptrs, &c_dbcsr_acc_opencl_config.nmemptrs);
    *info = *pfree;
#    if !defined(NDEBUG)
    LIBXSMM_MEMZERO127(pfree);
#    endif
  }
  else result = EXIT_FAILURE;
  ACC_OPENCL_RELEASE(c_dbcsr_acc_opencl_config.lock_memory);
}
#  endif
if (0 != c_dbcsr_acc_opencl_config.debug && 0 != c_dbcsr_acc_opencl_config.verbosity && EXIT_SUCCESS == result) {
  fprintf(stderr, "INFO ACC/OpenCL: memory=%p pointer=%p deallocated\n", (const void*)memory, dev_mem);
}
}
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
c_dbcsr_timestop(&routine_handle);
#  endif
ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_dev_mem_set_ptr(void** dev_mem, void* other, size_t offset) {
  int result = EXIT_SUCCESS;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(NULL != dev_mem);
  if (NULL != other || 0 == offset) {
    *dev_mem = (char*)other + offset;
  }
  else {
    result = EXIT_FAILURE;
    *dev_mem = NULL;
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
  assert((NULL != host_mem && NULL != dev_mem) || 0 == nbytes);
  if (NULL != host_mem && NULL != dev_mem && 0 != nbytes) {
    const c_dbcsr_acc_opencl_stream_t* const str =
      (NULL != stream ? ACC_OPENCL_STREAM(stream) : c_dbcsr_acc_opencl_stream(NULL /*lock*/, ACC_OPENCL_OMP_TID()));
#  if defined(ACC_OPENCL_ASYNC)
    const cl_bool finish = (0 == (1 & c_dbcsr_acc_opencl_config.async) || NULL == stream ||
                            (0 != c_dbcsr_acc_opencl_config.device.nv && NULL == (ACC_OPENCL_ASYNC)));
#  else
        const cl_bool finish = CL_TRUE;
#  endif
    assert(NULL != str && NULL != str->queue);
#  if defined(ACC_OPENCL_MEM_DEVPTR)
    if (NULL != c_dbcsr_acc_opencl_config.device.clEnqueueMemcpyINTEL) {
      result = c_dbcsr_acc_opencl_config.device.clEnqueueMemcpyINTEL(str->queue, finish, dev_mem, host_mem, nbytes, 0, NULL, NULL);
    }
    else
#  endif
    {
      c_dbcsr_acc_opencl_info_memptr_t info;
      size_t offset = 0;
#  if defined(ACC_OPENCL_MEM_DEVPTR)
      ACC_OPENCL_ACQUIRE(c_dbcsr_acc_opencl_config.lock_memory);
#  endif
      result = c_dbcsr_acc_opencl_info_devptr_lock(&info, NULL /*lock*/, dev_mem, 1 /*elsize*/, &nbytes, &offset);
      if (EXIT_SUCCESS == result) {
        assert(NULL != info.memory);
        result = clEnqueueWriteBuffer(str->queue, info.memory, finish, offset, nbytes, host_mem, 0, NULL, NULL);
      }
#  if defined(ACC_OPENCL_MEM_DEVPTR)
      ACC_OPENCL_RELEASE(c_dbcsr_acc_opencl_config.lock_memory);
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
  assert((NULL != dev_mem && NULL != host_mem) || 0 == nbytes);
  if (NULL != host_mem && NULL != dev_mem && 0 != nbytes) {
    const c_dbcsr_acc_opencl_stream_t* const str =
      (NULL != stream ? ACC_OPENCL_STREAM(stream) : c_dbcsr_acc_opencl_stream(NULL /*lock*/, ACC_OPENCL_OMP_TID()));
    const cl_bool finish = (NULL != stream ? CL_FALSE : CL_TRUE);
    c_dbcsr_acc_opencl_info_memptr_t info;
    size_t offset = 0;
    assert(NULL != str && NULL != str->queue);
#  if defined(ACC_OPENCL_MEM_DEVPTR)
    ACC_OPENCL_ACQUIRE(c_dbcsr_acc_opencl_config.lock_memory);
#  endif
    result = c_dbcsr_acc_opencl_info_devptr_lock(&info, NULL /*lock*/, dev_mem, 1 /*elsize*/, &nbytes, &offset);
    if (EXIT_SUCCESS == result) {
      assert(NULL != info.memory);
      result = c_dbcsr_acc_opencl_memcpy_d2h(info.memory, host_mem, offset, nbytes, str->queue, finish);
    }
#  if defined(ACC_OPENCL_MEM_DEVPTR)
    ACC_OPENCL_RELEASE(c_dbcsr_acc_opencl_config.lock_memory);
#  endif
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
  assert((NULL != devmem_src && NULL != devmem_dst) || 0 == nbytes);
  if (NULL != devmem_src && NULL != devmem_dst && 0 != nbytes) {
    const c_dbcsr_acc_opencl_stream_t* const str =
      (NULL != stream ? ACC_OPENCL_STREAM(stream) : c_dbcsr_acc_opencl_stream(NULL /*lock*/, ACC_OPENCL_OMP_TID()));
    cl_event event = NULL;
    assert(NULL != str && NULL != str->queue);
#  if defined(ACC_OPENCL_MEM_DEVPTR)
    if (NULL != c_dbcsr_acc_opencl_config.device.clEnqueueMemcpyINTEL) {
      result = c_dbcsr_acc_opencl_config.device.clEnqueueMemcpyINTEL(
        str->queue, CL_FALSE /*blocking*/, devmem_dst, devmem_src, nbytes, 0, NULL, &event);
    }
    else
#  endif
    {
      c_dbcsr_acc_opencl_info_memptr_t info_src, info_dst;
      size_t offset_src = 0, offset_dst = 0;
#  if defined(ACC_OPENCL_MEM_DEVPTR)
      ACC_OPENCL_ACQUIRE(c_dbcsr_acc_opencl_config.lock_memory);
#  endif
      result |= c_dbcsr_acc_opencl_info_devptr_lock(&info_src, NULL /*lock*/, devmem_src, 1 /*elsize*/, &nbytes, &offset_src);
      result |= c_dbcsr_acc_opencl_info_devptr_lock(&info_dst, NULL /*lock*/, devmem_dst, 1 /*elsize*/, &nbytes, &offset_dst);
      if (EXIT_SUCCESS == result) {
        assert(NULL != info_src.memory && NULL != info_dst.memory);
        result = clEnqueueCopyBuffer(str->queue, info_src.memory, info_dst.memory, offset_src, offset_dst, nbytes, 0, NULL, &event);
      }
#  if defined(ACC_OPENCL_MEM_DEVPTR)
      ACC_OPENCL_RELEASE(c_dbcsr_acc_opencl_config.lock_memory);
#  endif
    }
    if (NULL == stream && EXIT_SUCCESS == result) result = clWaitForEvents(1, &event);
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
    const c_dbcsr_acc_opencl_stream_t* const str =
      (NULL != stream ? ACC_OPENCL_STREAM(stream) : c_dbcsr_acc_opencl_stream(NULL /*lock*/, ACC_OPENCL_OMP_TID()));
    size_t size_of_value = 1;
    cl_event event;
    if (0 == LIBXSMM_MOD2(nbytes, 4)) size_of_value = 4;
    else if (0 == LIBXSMM_MOD2(nbytes, 2)) size_of_value = 2;
    assert(NULL != str && NULL != str->queue);
#  if defined(ACC_OPENCL_MEM_DEVPTR)
    if (NULL != c_dbcsr_acc_opencl_config.device.clEnqueueMemFillINTEL) {
      cl_device_id device = NULL;
      assert(NULL != c_dbcsr_acc_opencl_config.device.context);
      result = clGetContextInfo(c_dbcsr_acc_opencl_config.device.context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);
      if (EXIT_SUCCESS == result) {
        result = c_dbcsr_acc_opencl_config.device.clEnqueueMemFillINTEL(
          str->queue, (char*)dev_mem + offset, &value, size_of_value, nbytes, 0, NULL, &event);
      }
    }
    else
#  endif
    {
      size_t offset_info = 0;
      c_dbcsr_acc_opencl_info_memptr_t info;
#  if defined(ACC_OPENCL_MEM_DEVPTR)
      ACC_OPENCL_ACQUIRE(c_dbcsr_acc_opencl_config.lock_memory);
#  endif
      result = c_dbcsr_acc_opencl_info_devptr_lock(&info, NULL /*lock*/, dev_mem, 1 /*elsize*/, &nbytes, &offset_info);
      if (EXIT_SUCCESS == result) {
        assert(NULL != info.memory);
        offset_info += offset;
        result = clEnqueueFillBuffer(str->queue, info.memory, &value, size_of_value, offset_info, nbytes, 0, NULL, &event);
      }
#  if defined(ACC_OPENCL_MEM_DEVPTR)
      ACC_OPENCL_RELEASE(c_dbcsr_acc_opencl_config.lock_memory);
#  endif
    }
    if (NULL == stream && EXIT_SUCCESS == result) result = clWaitForEvents(1, &event);
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
  cl_device_id device = NULL;
  int result = (0 < c_dbcsr_acc_opencl_config.ndevices ? clGetContextInfo(c_dbcsr_acc_opencl_config.device.context,
                                                           CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL)
                                                       : EXIT_FAILURE);
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  if (EXIT_SUCCESS == result) {
    result = c_dbcsr_acc_opencl_info_devmem(device, mem_free, mem_total, NULL /*mem_local*/, NULL /*mem_unified*/);
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
