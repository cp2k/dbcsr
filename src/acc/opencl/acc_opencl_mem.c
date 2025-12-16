/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: BSD-3-Clause                                                          */
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

#  if !defined(ACC_OPENCL_MEM_ALLOC)
#    if 1
#      define ACC_OPENCL_MEM_ALLOC(SIZE, ALIGNMENT) libxsmm_aligned_malloc(SIZE, ALIGNMENT)
#      define ACC_OPENCL_MEM_FREE(PTR) libxsmm_free(PTR)
#    else
#      define ACC_OPENCL_MEM_ALLOC(SIZE, ALIGNMENT) aligned_alloc(ALIGNMENT, SIZE)
#      define ACC_OPENCL_MEM_FREE(PTR) free(PTR)
#    endif
#  endif
#  if !defined(ACC_OPENCL_MEM_ALIGNSCALE)
#    define ACC_OPENCL_MEM_ALIGNSCALE 8
#  endif
#  if !defined(ACC_OPENCL_MEM_SVM_INTEL) && 0
#    define ACC_OPENCL_MEM_SVM_INTEL
#  endif
#  if !defined(ACC_OPENCL_MEM_HST_INTEL) && 0
#    define ACC_OPENCL_MEM_HST_INTEL
#  endif
#  if !defined(ACC_OPENCL_MEM_SVM_USM) && 0
#    define ACC_OPENCL_MEM_SVM_USM
#  endif
#  if !defined(ACC_OPENCL_MEM_DEBUG) && 0
#    if && !defined(NDEBUG)
#      define ACC_OPENCL_MEM_DEBUG
#    endif
#  endif


#  if defined(__cplusplus)
extern "C" {
#  endif

void c_dbcsr_acc_opencl_pmalloc_init(size_t size, size_t* num, void* pool[], void* storage) {
  char* p = (char*)storage;
  size_t i = 0;
  assert(0 < size && NULL != num && NULL != pool && NULL != storage);
  for (; i < *num; ++i, p += size) pool[i] = p;
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


void c_dbcsr_acc_opencl_pfree(const void* pointer, void* pool[], size_t* i) {
  assert(NULL != pool && NULL != i);
  if (NULL != pointer) {
    LIBXSMM_ASSIGN127(pool + *i, &pointer);
    ++(*i);
  }
}


c_dbcsr_acc_opencl_info_memptr_t* c_dbcsr_acc_opencl_info_hostptr(const void* memory) {
  c_dbcsr_acc_opencl_info_memptr_t* result = NULL;
  if (NULL == c_dbcsr_acc_opencl_config.device.clHostMemAllocINTEL &&
#  if (0 != ACC_OPENCL_USM)
      0 == c_dbcsr_acc_opencl_config.device.usm &&
#  endif
      NULL != memory)
  {
    assert(sizeof(c_dbcsr_acc_opencl_info_memptr_t) < (uintptr_t)memory);
    result = (c_dbcsr_acc_opencl_info_memptr_t*)((uintptr_t)memory - sizeof(c_dbcsr_acc_opencl_info_memptr_t));
  }
  return result;
}


c_dbcsr_acc_opencl_info_memptr_t* c_dbcsr_acc_opencl_info_devptr_modify(
  ACC_OPENCL_LOCKTYPE* lock, void* memory, size_t elsize, const size_t* amount, size_t* offset) {
  c_dbcsr_acc_opencl_info_memptr_t* result = NULL;
#  if !defined(ACC_OPENCL_MEM_DEBUG)
  LIBXSMM_UNUSED(amount);
#  endif
  if (NULL != memory) {
    assert(NULL != c_dbcsr_acc_opencl_config.device.context);
    if (/* USM-pointer */
#  if (0 != ACC_OPENCL_USM)
      0 != c_dbcsr_acc_opencl_config.device.usm ||
#  endif
      NULL != c_dbcsr_acc_opencl_config.device.clDeviceMemAllocINTEL)
    { /* assume only first item of c_dbcsr_acc_opencl_info_memptr_t is accessed */
      assert(0 != c_dbcsr_acc_opencl_config.device.usm || NULL != c_dbcsr_acc_opencl_config.device.clDeviceMemAllocINTEL);
      result = NULL; /*(c_dbcsr_acc_opencl_info_memptr_t*)memory*/
      if (NULL != offset) *offset = 0;
    }
    else { /* info-augmented pointer */
      const char* const pointer = (const char*)memory;
      const size_t n = ACC_OPENCL_MAXNITEMS * c_dbcsr_acc_opencl_config.nthreads;
      size_t hit = (size_t)-1, i;
      assert(0 == c_dbcsr_acc_opencl_config.device.usm && NULL == c_dbcsr_acc_opencl_config.device.clDeviceMemAllocINTEL);
      assert(NULL != c_dbcsr_acc_opencl_config.memptrs);
      if (NULL != lock) ACC_OPENCL_ACQUIRE(lock);
      for (i = c_dbcsr_acc_opencl_config.nmemptrs; i < n; ++i) {
        c_dbcsr_acc_opencl_info_memptr_t* const info = c_dbcsr_acc_opencl_config.memptrs[i];
        if (NULL != info) {
          char* const memptr = (char*)info->memptr;
          if (memptr == pointer) { /* fast-path */
            if (NULL != offset) *offset = 0;
            result = info;
            break;
          }
          else if (memptr < pointer && NULL != offset) {
            size_t d = pointer - memptr, s = d;
            assert(0 < elsize && 0 != d);
            if (d < hit && (1 == elsize || 0 == (d % elsize)) &&
#  if defined(ACC_OPENCL_MEM_DEBUG) /* TODO: verify enclosed conditions */
                (EXIT_SUCCESS == clGetMemObjectInfo(info->memory, CL_MEM_SIZE, sizeof(size_t), &s, NULL)) &&
                (NULL == amount || (*amount * elsize + d) <= s) &&
#  endif
                (1 == elsize || 0 == (s % elsize)) && d <= s)
            {
              *offset = (1 == elsize ? d : (d / elsize));
              result = info;
              hit = d;
            }
#  if defined(ACC_OPENCL_MEM_DEBUG)
            else if (d < hit && 0 != c_dbcsr_acc_opencl_config.debug && 0 != c_dbcsr_acc_opencl_config.verbosity) {
              fprintf(stderr, "ERROR ACC/OpenCL: memory=%p pointer=%p size=%llu offset=%llu info failed\n",
                (const void*)info->memory, info->memptr, (unsigned long long)s,
                (unsigned long long)(1 == elsize ? d : (d / elsize)));
            }
#  endif
          }
        }
        else break;
      }
      if (NULL != lock) ACC_OPENCL_RELEASE(lock);
      assert(NULL != result);
    }
  }
  return result;
}


int c_dbcsr_acc_opencl_info_devptr_lock(c_dbcsr_acc_opencl_info_memptr_t* info, ACC_OPENCL_LOCKTYPE* lock, const void* memory,
  size_t elsize, const size_t* amount, size_t* offset) {
  const c_dbcsr_acc_opencl_info_memptr_t* meminfo = NULL;
  int result = EXIT_SUCCESS;
  void* non_const;
  LIBXSMM_ASSIGN127(&non_const, &memory);
  meminfo = c_dbcsr_acc_opencl_info_devptr_modify(lock, non_const, elsize, amount, offset);
  assert(NULL != info);
  if (NULL == meminfo) { /* USM-pointer */
    LIBXSMM_MEMZERO127(info);
    info->memory = (cl_mem)non_const;
  }
  else { /* info-augmented pointer */
    assert(NULL != c_dbcsr_acc_opencl_config.device.context);
    LIBXSMM_ASSIGN127(info, meminfo);
  }
  return result;
}


int c_dbcsr_acc_opencl_info_devptr(
  c_dbcsr_acc_opencl_info_memptr_t* info, const void* memory, size_t elsize, const size_t* amount, size_t* offset) {
  ACC_OPENCL_LOCKTYPE* const lock_memory = ((
#  if (0 != ACC_OPENCL_USM)
                                              0 != c_dbcsr_acc_opencl_config.device.usm ||
#  endif
                                              NULL != c_dbcsr_acc_opencl_config.device.clSetKernelArgMemPointerINTEL)
                                              ? NULL /* no lock required */
                                              : c_dbcsr_acc_opencl_config.lock_memory);
  return c_dbcsr_acc_opencl_info_devptr_lock(info, lock_memory, memory, elsize, amount, offset);
}


int c_dbcsr_acc_host_mem_deallocate_internal(void* /*host_ptr*/, cl_command_queue /*queue*/);
int c_dbcsr_acc_host_mem_deallocate_internal(void* host_ptr, cl_command_queue queue) {
  const c_dbcsr_acc_opencl_device_t* const devinfo = &c_dbcsr_acc_opencl_config.device;
  int result = EXIT_FAILURE;
#  if (1 >= ACC_OPENCL_USM)
  if (NULL != devinfo->clMemFreeINTEL) {
#    if defined(ACC_OPENCL_MEM_SVM_INTEL) || defined(ACC_OPENCL_MEM_HST_INTEL)
    result = devinfo->clMemFreeINTEL(devinfo->context, host_ptr);
#    else
    ACC_OPENCL_MEM_FREE(host_ptr);
    result = EXIT_SUCCESS;
#    endif
  }
  else
#  endif
#  if (0 != ACC_OPENCL_USM) && ((1 >= ACC_OPENCL_USM) || defined(ACC_OPENCL_MEM_SVM_USM))
    if (0 != devinfo->usm && 0 != devinfo->unified)
  {
    if (0 == ((CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) & devinfo->usm)) {
      result = clEnqueueSVMUnmap(queue, host_ptr, 0, NULL, NULL); /* clSVMFree below synchronizes */
    }
    else result = EXIT_SUCCESS;
    clSVMFree(devinfo->context, host_ptr);
  }
  else
#  endif
  {
    LIBXSMM_UNUSED(queue);
    ACC_OPENCL_MEM_FREE(host_ptr);
    result = EXIT_SUCCESS;
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_host_mem_allocate(void** host_mem, size_t nbytes, void* stream) {
  int result = EXIT_SUCCESS;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
  assert(NULL != host_mem);
  if (0 != nbytes) {
    const c_dbcsr_acc_opencl_device_t* const devinfo = &c_dbcsr_acc_opencl_config.device;
    const c_dbcsr_acc_opencl_stream_t* const str = (NULL != stream ? ACC_OPENCL_STREAM(stream)
                                                                   : c_dbcsr_acc_opencl_stream_default());
    int alignment = LIBXSMM_MAX(0x10000, sizeof(void*));
    void* host_ptr = NULL;
    cl_mem memory = NULL;
    assert(NULL != str);
    if ((ACC_OPENCL_MEM_ALIGNSCALE * ACC_OPENCL_CACHELINE) <= nbytes) {
      const int a = ((ACC_OPENCL_MEM_ALIGNSCALE * ACC_OPENCL_MAXALIGN) <= nbytes ? ACC_OPENCL_MAXALIGN : ACC_OPENCL_CACHELINE);
      if (alignment < a) alignment = a;
    }
#  if !defined(ACC_OPENCL_ACTIVATE)
    if (NULL == devinfo->context) {
      ACC_OPENCL_EXPECT(EXIT_SUCCESS == c_dbcsr_acc_opencl_set_active_device(
                                          c_dbcsr_acc_opencl_config.lock_main, c_dbcsr_acc_opencl_config.device_id));
    }
#  endif
#  if (1 >= ACC_OPENCL_USM)
    if (NULL != devinfo->clMemFreeINTEL) {
#    if defined(ACC_OPENCL_MEM_SVM_INTEL)
      const cl_device_id device_id = c_dbcsr_acc_opencl_config.devices[c_dbcsr_acc_opencl_config.device_id];
      host_ptr = devinfo->clSharedMemAllocINTEL(devinfo->context, device_id, NULL /*properties*/, nbytes, 0 /*alignment*/, &result);
#    elif defined(ACC_OPENCL_MEM_HST_INTEL)
      host_ptr = devinfo->clHostMemAllocINTEL(devinfo->context, NULL /*properties*/, nbytes, 0 /*alignment*/, &result);
#    else
      host_ptr = ACC_OPENCL_MEM_ALLOC(nbytes, alignment);
#    endif
      assert(NULL != host_ptr || EXIT_SUCCESS != result);
      /*if (NULL != host_ptr)*/ *host_mem = host_ptr;
    }
    else
#  endif
      if (0 != devinfo->usm)
    {
#  if (0 != ACC_OPENCL_USM)
#    if ((1 >= ACC_OPENCL_USM) || defined(ACC_OPENCL_MEM_SVM_USM))
      if (0 != devinfo->unified) {
        const int svmmem_fine = (0 != ((CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) & devinfo->usm)
                                   ? CL_MEM_SVM_FINE_GRAIN_BUFFER
                                   : 0);
        host_ptr = clSVMAlloc(devinfo->context, CL_MEM_READ_WRITE | svmmem_fine, nbytes, 0 /*alignment*/);
        if (NULL != host_ptr) {
          if (0 == svmmem_fine) {
            result = clEnqueueSVMMap(
              str->queue, CL_TRUE /*always block*/, CL_MAP_READ | CL_MAP_WRITE, host_ptr, nbytes, 0, NULL, NULL);
          }
          *host_mem = host_ptr;
        }
        else result = EXIT_FAILURE;
      }
      else
#    endif
      {
        host_ptr = ACC_OPENCL_MEM_ALLOC(nbytes, alignment);
        if (NULL != host_ptr) *host_mem = host_ptr;
        else result = EXIT_FAILURE;
      }
#  endif
    }
    else {
      const size_t size_meminfo = sizeof(c_dbcsr_acc_opencl_info_memptr_t);
      int memflags = CL_MEM_ALLOC_HOST_PTR;
      nbytes += alignment + size_meminfo - 1;
#  if defined(ACC_OPENCL_XHINTS)
      if (0 != (4 & c_dbcsr_acc_opencl_config.xhints) && (0 != devinfo->nv || NULL != (ACC_OPENCL_XHINTS))) {
        host_ptr = ACC_OPENCL_MEM_ALLOC(nbytes, alignment);
        if (NULL != host_ptr) memflags = CL_MEM_USE_HOST_PTR;
      }
#  endif
      memory = clCreateBuffer(devinfo->context, (cl_mem_flags)(CL_MEM_READ_WRITE | memflags), nbytes, host_ptr, &result);
      if (EXIT_SUCCESS == result) {
        void* mapped = host_ptr;
        if (NULL == host_ptr) {
          mapped = clEnqueueMapBuffer(str->queue, memory, CL_TRUE /*always block*/,
#  if defined(ACC_OPENCL_XHINTS) && (defined(CL_VERSION_1_2) || defined(CL_MAP_WRITE_INVALIDATE_REGION))
            (16 & c_dbcsr_acc_opencl_config.xhints) ? CL_MAP_WRITE_INVALIDATE_REGION :
#  endif
                                                    (CL_MAP_READ | CL_MAP_WRITE),
            0 /*offset*/, nbytes, 0, NULL, NULL, &result);
        }
        assert(EXIT_SUCCESS == result || NULL == mapped);
        if (EXIT_SUCCESS == result) {
          const uintptr_t address = (uintptr_t)mapped;
          const uintptr_t aligned = LIBXSMM_UP2(address + size_meminfo, alignment);
          c_dbcsr_acc_opencl_info_memptr_t* const meminfo = (c_dbcsr_acc_opencl_info_memptr_t*)(aligned - size_meminfo);
          assert(address + size_meminfo <= aligned && NULL != meminfo);
          meminfo->memory = memory;
          meminfo->memptr = mapped;
          *host_mem = (void*)aligned;
          assert(meminfo == c_dbcsr_acc_opencl_info_hostptr(*host_mem));
        }
      }
    }
    if (EXIT_SUCCESS != result) {
      if (NULL != memory) ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseMemObject(memory));
      if (NULL != host_ptr) {
        ACC_OPENCL_EXPECT(EXIT_SUCCESS == c_dbcsr_acc_host_mem_deallocate_internal(host_ptr, str->queue));
      }
      *host_mem = NULL;
    }
  }
  else *host_mem = NULL; /* consider warning */
  assert(EXIT_SUCCESS == result || NULL == *host_mem);
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_host_mem_deallocate(void* host_mem, void* stream) {
  int result = EXIT_SUCCESS;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
  if (NULL != host_mem) {
    const c_dbcsr_acc_opencl_stream_t* const str = (NULL != stream ? ACC_OPENCL_STREAM(stream)
                                                                   : c_dbcsr_acc_opencl_stream_default());
    const c_dbcsr_acc_opencl_info_memptr_t* const meminfo = c_dbcsr_acc_opencl_info_hostptr(host_mem);
    assert(NULL != str);
    if (NULL == meminfo || NULL == meminfo->memory) { /* USM-pointer */
      assert(0 != c_dbcsr_acc_opencl_config.device.usm || NULL != c_dbcsr_acc_opencl_config.device.clMemFreeINTEL);
      result = c_dbcsr_acc_host_mem_deallocate_internal(host_mem, str->queue);
    }
    else { /* info-augmented pointer */
      const c_dbcsr_acc_opencl_info_memptr_t info = *meminfo; /* copy meminfo prior to unmap */
      int result_release = EXIT_SUCCESS;
      void* host_ptr = NULL;
      assert(0 == c_dbcsr_acc_opencl_config.device.usm && NULL == c_dbcsr_acc_opencl_config.device.clMemFreeINTEL);
      if (EXIT_SUCCESS == clGetMemObjectInfo(info.memory, CL_MEM_HOST_PTR, sizeof(void*), &host_ptr, NULL) && NULL != host_ptr) {
        result = c_dbcsr_acc_host_mem_deallocate_internal(host_ptr, str->queue);
      }
      else { /* clReleaseMemObject later on synchronizes */
        result = clEnqueueUnmapMemObject(str->queue, info.memory, info.memptr, 0, NULL, NULL);
      }
      result_release = clReleaseMemObject(info.memory);
      if (EXIT_SUCCESS == result) result = result_release;
    }
  }
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


void CL_CALLBACK c_dbcsr_acc_memcpy_notify(cl_event /*event*/, cl_int /*event_status*/, void* /*data*/);
void CL_CALLBACK c_dbcsr_acc_memcpy_notify(cl_event event, cl_int event_status, void* data) {
  int result = EXIT_SUCCESS;
  const double durdev = c_dbcsr_acc_opencl_duration(event, &result);
  c_dbcsr_acc_opencl_info_memptr_t info;
  cl_command_type type;
  size_t size = 0, offset = 0;
  LIBXSMM_UNUSED(event_status);
  assert(CL_COMPLETE == event_status && NULL != data);
  if (EXIT_SUCCESS == result && EXIT_SUCCESS == clGetEventInfo(event, CL_EVENT_COMMAND_TYPE, sizeof(type), &type, NULL) &&
      EXIT_SUCCESS == c_dbcsr_acc_opencl_info_devptr_lock(&info, NULL, data, 1 /*elsize*/, NULL /*amount*/, &offset) &&
      EXIT_SUCCESS == clGetMemObjectInfo(info.memory, CL_MEM_SIZE, sizeof(size_t), &size, NULL) && offset <= size)
  {
    /*const double durhst = libxsmm_timer_duration((libxsmm_timer_tickint)info.data, libxsmm_timer_tick());
    const double durtot = durdev - LIBXSMM_MIN(durdev, durhst);*/
    const size_t amount = size - offset;
    const double vals[] = {(double)amount, durdev};
    const int mb = (int)((amount + (1 << 19)) >> 20);
    switch (type) {
      case CL_COMMAND_WRITE_BUFFER: {
        assert(NULL != c_dbcsr_acc_opencl_config.hist_h2d);
        c_dbcsr_acc_opencl_hist_set(c_dbcsr_acc_opencl_config.lock_memory, c_dbcsr_acc_opencl_config.hist_h2d, vals);
        if (0 > c_dbcsr_acc_opencl_config.profile) fprintf(stderr, "PROF ACC/OpenCL: H2D mb=%i us=%.0f\n", mb, durdev * 1E6);
      } break;
      case CL_COMMAND_READ_BUFFER: {
        assert(NULL != c_dbcsr_acc_opencl_config.hist_d2h);
        c_dbcsr_acc_opencl_hist_set(c_dbcsr_acc_opencl_config.lock_memory, c_dbcsr_acc_opencl_config.hist_d2h, vals);
        if (0 > c_dbcsr_acc_opencl_config.profile) fprintf(stderr, "PROF ACC/OpenCL: D2H mb=%i us=%.0f\n", mb, durdev * 1E6);
      } break;
      case CL_COMMAND_COPY_BUFFER: {
        assert(NULL != c_dbcsr_acc_opencl_config.hist_d2d);
        c_dbcsr_acc_opencl_hist_set(c_dbcsr_acc_opencl_config.lock_memory, c_dbcsr_acc_opencl_config.hist_d2d, vals);
        if (0 > c_dbcsr_acc_opencl_config.profile) fprintf(stderr, "PROF ACC/OpenCL: D2D mb=%i us=%.0f\n", mb, durdev * 1E6);
      } break;
    }
  }
  if (NULL != event) ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseEvent(event));
}


int c_dbcsr_acc_dev_mem_allocate(void** dev_mem, size_t nbytes) {
  /* assume no lock is needed to protect against context/device changes */
  const c_dbcsr_acc_opencl_device_t* const devinfo = &c_dbcsr_acc_opencl_config.device;
  int result = EXIT_SUCCESS;
  void* memptr = NULL;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
#  if !defined(ACC_OPENCL_ACTIVATE)
  if (NULL == devinfo->context) {
    ACC_OPENCL_EXPECT(EXIT_SUCCESS == c_dbcsr_acc_opencl_set_active_device(
                                        c_dbcsr_acc_opencl_config.lock_main, c_dbcsr_acc_opencl_config.device_id));
  }
#  endif
  assert(NULL != dev_mem && NULL != devinfo->context);
  if (0 != nbytes) {
    cl_mem memory = NULL;
#  if (1 >= ACC_OPENCL_USM)
    if (NULL != devinfo->clMemFreeINTEL) {
      const cl_device_id device_id = c_dbcsr_acc_opencl_config.devices[c_dbcsr_acc_opencl_config.device_id];
#    if defined(ACC_OPENCL_MEM_SVM_INTEL)
      memptr = devinfo->clSharedMemAllocINTEL(devinfo->context, device_id, NULL /*properties*/, nbytes, 0 /*alignment*/, &result);
#    else
      memptr = devinfo->clDeviceMemAllocINTEL(devinfo->context, device_id, NULL /*properties*/, nbytes, 0 /*alignment*/, &result);
#    endif
      *dev_mem = memptr;
    }
    else
#  endif
#  if (0 != ACC_OPENCL_USM)
      if (0 != devinfo->usm)
    {
#    if (1 >= ACC_OPENCL_USM) || defined(ACC_OPENCL_MEM_SVM_USM)
      const int svmflags = (0 != ((CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) & devinfo->usm)
                              ? CL_MEM_SVM_FINE_GRAIN_BUFFER
                              : 0);
      memptr = clSVMAlloc(devinfo->context, (cl_svm_mem_flags)(CL_MEM_READ_WRITE | svmflags), nbytes, 0 /*alignment*/);
#    else
      int alignment = LIBXSMM_MAX(0x10000, sizeof(void*));
      if ((ACC_OPENCL_MEM_ALIGNSCALE * ACC_OPENCL_CACHELINE) <= nbytes) {
        const int a = ((ACC_OPENCL_MEM_ALIGNSCALE * ACC_OPENCL_MAXALIGN) <= nbytes ? ACC_OPENCL_MAXALIGN : ACC_OPENCL_CACHELINE);
        if (alignment < a) alignment = a;
      }
      memptr = ACC_OPENCL_MEM_ALLOC(nbytes, alignment);
#    endif
      *dev_mem = memptr;
    }
    else
#  endif
    {
#  if defined(ACC_OPENCL_XHINTS)
      const int devuid = devinfo->uid, devuids = (0x4905 == devuid || 0x020a == devuid || (0x0bd0 <= devuid && 0x0bdb >= devuid));
      const int try_flag = ((0 != (8 & c_dbcsr_acc_opencl_config.xhints) && 0 != devinfo->intel && 0 == devinfo->unified &&
                              (devuids || NULL != (ACC_OPENCL_XHINTS)))
                              ? (1u << 22)
                              : 0);
      memory = clCreateBuffer(devinfo->context, (cl_mem_flags)(CL_MEM_READ_WRITE | try_flag), nbytes, NULL /*host_ptr*/, &result);
      if (0 != try_flag && EXIT_SUCCESS != result) /* retry without try_flag */
#  endif
      {
        memory = clCreateBuffer(devinfo->context, CL_MEM_READ_WRITE, nbytes, NULL /*host_ptr*/, &result);
      }
      if (EXIT_SUCCESS == result) {
        const c_dbcsr_acc_opencl_stream_t* str = NULL;
        static cl_kernel kernel = NULL;
        const size_t size = 1;
        ACC_OPENCL_ACQUIRE(c_dbcsr_acc_opencl_config.lock_memory);
        str = c_dbcsr_acc_opencl_stream(NULL /*lock*/, ACC_OPENCL_OMP_TID());
        assert(NULL != str && NULL != memory);
        /* determine device-side value of device-memory object by running some kernel */
        if (NULL == kernel) { /* generate kernel */
          const char source[] = "kernel void memptr(global unsigned long* ptr) {\n"
                                "  const union { global unsigned long* p; unsigned long u; } cast = { ptr };\n"
                                "  const size_t i = get_global_id(0);\n"
                                "  ptr[i] = cast.u + i;\n"
                                "}\n";
          assert(sizeof(size_t) == sizeof(cl_ulong));
          result = c_dbcsr_acc_opencl_kernel(0 /*source_kind*/, source, "memptr" /*kernel_name*/, NULL /*build_params*/,
            NULL /*build_options*/, NULL /*try_build_options*/, NULL /*try_ok*/, NULL /*extnames*/, 0 /*num_exts*/, &kernel);
        }
        /* TODO: backup/restore memory */
        if (EXIT_SUCCESS == result) result = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memory);
        if (EXIT_SUCCESS == result) {
          result = clEnqueueNDRangeKernel(
            str->queue, kernel, 1 /*work_dim*/, NULL /*offset*/, &size, NULL /*local_work_size*/, 0, NULL, NULL);
        }
        ACC_OPENCL_RELEASE(c_dbcsr_acc_opencl_config.lock_memory);
        if (EXIT_SUCCESS == result) {
          result = clEnqueueReadBuffer(
            str->queue, memory, CL_TRUE /*blocking*/, 0 /*offset*/, sizeof(void*), &memptr, 0, NULL, NULL /*event*/);
        }
        assert(EXIT_SUCCESS != result || NULL != memptr);
        if (EXIT_SUCCESS == result) {
          c_dbcsr_acc_opencl_info_memptr_t* const info = (c_dbcsr_acc_opencl_info_memptr_t*)c_dbcsr_acc_opencl_pmalloc(
            c_dbcsr_acc_opencl_config.lock_memory, (void**)c_dbcsr_acc_opencl_config.memptrs, &c_dbcsr_acc_opencl_config.nmemptrs);
          if (NULL != info) {
            info->memory = memory;
            info->memptr = memptr;
            *dev_mem = memptr;
          }
          else result = EXIT_FAILURE;
        }
      }
    }
    if (EXIT_SUCCESS != result) {
      if (0 != c_dbcsr_acc_opencl_config.verbosity) {
        fprintf(stderr, "ERROR ACC/OpenCL: memory=%p pointer=%p size=%llu failed to allocate\n", (const void*)memory, memptr,
          (unsigned long long)nbytes);
      }
      if (NULL != memory) ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseMemObject(memory));
      *dev_mem = NULL;
    }
  }
  else *dev_mem = NULL;
  assert(EXIT_SUCCESS == result || NULL == *dev_mem);
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_dev_mem_deallocate(void* dev_mem) {
  int result = EXIT_SUCCESS;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
  if (NULL != dev_mem) {
    assert(NULL != c_dbcsr_acc_opencl_config.device.context);
#  if (1 >= ACC_OPENCL_USM)
    if (NULL != c_dbcsr_acc_opencl_config.device.clMemFreeINTEL) {
      result = c_dbcsr_acc_opencl_config.device.clMemFreeINTEL(c_dbcsr_acc_opencl_config.device.context, dev_mem);
    }
    else
#  endif
#  if (0 != ACC_OPENCL_USM)
      if (0 != c_dbcsr_acc_opencl_config.device.usm)
    {
#    if (1 >= ACC_OPENCL_USM) || defined(ACC_OPENCL_MEM_SVM_USM)
      clSVMFree(c_dbcsr_acc_opencl_config.device.context, dev_mem);
#    else
      ACC_OPENCL_MEM_FREE(dev_mem);
#    endif
    }
    else
#  endif
    {
      c_dbcsr_acc_opencl_info_memptr_t* info = NULL;
      ACC_OPENCL_ACQUIRE(c_dbcsr_acc_opencl_config.lock_memory);
      info = c_dbcsr_acc_opencl_info_devptr_modify(NULL, dev_mem, 1 /*elsize*/, NULL /*amount*/, NULL /*offset*/);
      if (NULL != info && info->memptr == dev_mem && NULL != info->memory) {
        c_dbcsr_acc_opencl_info_memptr_t* const pfree = c_dbcsr_acc_opencl_config.memptrs[c_dbcsr_acc_opencl_config.nmemptrs];
        result = clReleaseMemObject(info->memory);
        c_dbcsr_acc_opencl_pfree(pfree, (void**)c_dbcsr_acc_opencl_config.memptrs, &c_dbcsr_acc_opencl_config.nmemptrs);
        *info = *pfree;
        LIBXSMM_MEMZERO127(pfree);
      }
      else result = EXIT_FAILURE;
      ACC_OPENCL_RELEASE(c_dbcsr_acc_opencl_config.lock_memory);
    }
  }
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_dev_mem_set_ptr(void** dev_mem, void* other, size_t offset) {
  int result = EXIT_SUCCESS;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
  assert(NULL != dev_mem);
  if (NULL != other || 0 == offset) {
    *dev_mem = (char*)other + offset;
  }
  else {
    result = EXIT_FAILURE;
    *dev_mem = NULL;
  }
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_memcpy_h2d(const void* host_mem, void* dev_mem, size_t nbytes, void* stream) {
  const c_dbcsr_acc_opencl_device_t* const devinfo = &c_dbcsr_acc_opencl_config.device;
  int result = EXIT_SUCCESS;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
  assert((NULL != host_mem && NULL != dev_mem) || 0 == nbytes);
  assert(NULL != devinfo->context);
  if (
#  if (0 != ACC_OPENCL_USM)
    host_mem != dev_mem && /* fast-path only sensible without offsets */
#  endif
    NULL != host_mem && NULL != dev_mem && 0 != nbytes)
  {
#  if defined(ACC_OPENCL_ASYNC)
    const cl_bool finish = (0 == (1 & c_dbcsr_acc_opencl_config.async) || NULL == stream ||
                            (0 != (8 & c_dbcsr_acc_opencl_config.wa) && 0 != devinfo->intel && 0 != devinfo->unified));
#  else
    const cl_bool finish = CL_TRUE;
#  endif
    const c_dbcsr_acc_opencl_stream_t* str;
    cl_event event = NULL;
    ACC_OPENCL_ACQUIRE(c_dbcsr_acc_opencl_config.lock_memory);
    str = (NULL != stream ? ACC_OPENCL_STREAM(stream) : c_dbcsr_acc_opencl_stream(NULL, ACC_OPENCL_OMP_TID()));
    assert(NULL != str);
#  if (1 >= ACC_OPENCL_USM)
    if (NULL != devinfo->clEnqueueMemcpyINTEL) {
      result = devinfo->clEnqueueMemcpyINTEL(str->queue, finish, dev_mem, host_mem, nbytes, 0, NULL, NULL);
    }
    else
#  endif
#  if (0 != ACC_OPENCL_USM)
      if (0 != devinfo->usm)
    {
#    if (1 >= ACC_OPENCL_USM) || defined(ACC_OPENCL_MEM_SVM_USM)
      result = clEnqueueSVMMemcpy(
        str->queue, finish, dev_mem, host_mem, nbytes, 0, NULL, NULL == c_dbcsr_acc_opencl_config.hist_h2d ? NULL : &event);
#    else
      memcpy(dev_mem, host_mem, nbytes);
#    endif
    }
    else
#  endif
    {
      size_t offset = 0;
      c_dbcsr_acc_opencl_info_memptr_t* const info = c_dbcsr_acc_opencl_info_devptr_modify(
        NULL, dev_mem, 1 /*elsize*/, &nbytes, &offset);
      if (NULL != info) {
        result = clEnqueueWriteBuffer(str->queue, info->memory, finish, offset, nbytes, host_mem, 0, NULL,
          NULL == c_dbcsr_acc_opencl_config.hist_h2d ? NULL : &event);
        /*if (NULL != event && EXIT_SUCCESS == result) info->data = (void*)libxsmm_timer_tick();*/
      }
      else result = EXIT_FAILURE;
    }
    ACC_OPENCL_RELEASE(c_dbcsr_acc_opencl_config.lock_memory);
    if (NULL != event) { /* c_dbcsr_acc_memcpy_notify must be outside of locked region */
      if (EXIT_SUCCESS == result) {
        assert(NULL != c_dbcsr_acc_opencl_config.hist_h2d);
        if (!finish) { /* asynchronous */
          result = clSetEventCallback(event, CL_COMPLETE, c_dbcsr_acc_memcpy_notify, dev_mem);
        }
        else c_dbcsr_acc_memcpy_notify(event, CL_COMPLETE, dev_mem); /* synchronous */
      }
      else ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseEvent(event));
    }
  }
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


/* like c_dbcsr_acc_memcpy_d2h, but apply some async workaround. */
int c_dbcsr_acc_opencl_memcpy_d2h(const void* /*dev_mem*/, void* /*host_mem*/, size_t /*offset*/, size_t /*nbytes*/,
  cl_command_queue /*queue*/, int /*blocking*/, cl_event* /*event*/);
int c_dbcsr_acc_opencl_memcpy_d2h(
  const void* dev_mem, void* host_mem, size_t offset, size_t nbytes, cl_command_queue queue, int blocking, cl_event* event) {
  const c_dbcsr_acc_opencl_device_t* const devinfo = &c_dbcsr_acc_opencl_config.device;
#  if defined(ACC_OPENCL_ASYNC)
  const cl_bool finish = (0 != blocking || 0 == (2 & c_dbcsr_acc_opencl_config.async) ||
                          (0 != (8 & c_dbcsr_acc_opencl_config.wa) && 0 != devinfo->intel && 0 != devinfo->unified));
#  else
  const cl_bool finish = CL_TRUE;
#  endif
  int result = EXIT_SUCCESS;
  assert(NULL != dev_mem);
#  if (1 >= ACC_OPENCL_USM)
  if (NULL != devinfo->clEnqueueMemcpyINTEL) {
    result = devinfo->clEnqueueMemcpyINTEL(queue, finish, host_mem, (const char*)dev_mem + offset, nbytes, 0, NULL, event);
  }
  else
#  endif
#  if (0 != ACC_OPENCL_USM)
    if (0 != devinfo->usm)
  {
#    if (1 >= ACC_OPENCL_USM) || defined(ACC_OPENCL_MEM_SVM_USM)
    result = clEnqueueSVMMemcpy(queue, finish, host_mem, (const char*)dev_mem + offset, nbytes, 0, NULL, event);
#    else
    memcpy(host_mem, (const char*)dev_mem + offset, nbytes);
#    endif
  }
  else
#  endif
  {
    result = clEnqueueReadBuffer(queue, (cl_mem)(uintptr_t)dev_mem, finish, offset, nbytes, host_mem, 0, NULL, event);
  }
  if (EXIT_SUCCESS != result && !finish) { /* retry synchronously */
    int result_sync = EXIT_FAILURE;
#  if (1 >= ACC_OPENCL_USM)
    if (NULL != devinfo->clEnqueueMemcpyINTEL) {
      result_sync = devinfo->clEnqueueMemcpyINTEL(queue, CL_TRUE, host_mem, (const char*)dev_mem + offset, nbytes, 0, NULL, event);
    }
    else
#  endif
#  if (0 != ACC_OPENCL_USM)
      if (0 != devinfo->usm)
    {
#    if (1 >= ACC_OPENCL_USM) || defined(ACC_OPENCL_MEM_SVM_USM)
      result_sync = clEnqueueSVMMemcpy(queue, CL_TRUE, host_mem, (const char*)dev_mem + offset, nbytes, 0, NULL, event);
#    else
      memcpy(host_mem, (const char*)dev_mem + offset, nbytes);
#    endif
    }
    else
#  endif
    {
      result_sync = clEnqueueReadBuffer(queue, (cl_mem)(uintptr_t)dev_mem, CL_TRUE, offset, nbytes, host_mem, 0, NULL, event);
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


int c_dbcsr_acc_memcpy_d2h(const void* dev_mem, void* host_mem, size_t nbytes, void* stream) {
  int result = EXIT_SUCCESS;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
  assert((NULL != dev_mem && NULL != host_mem) || 0 == nbytes);
  if (
#  if (0 != ACC_OPENCL_USM)
    host_mem != dev_mem && /* fast-path only sensible without offsets */
#  endif
    NULL != host_mem && NULL != dev_mem && 0 != nbytes)
  {
    const cl_bool finish = (NULL != stream ? CL_FALSE : CL_TRUE);
    c_dbcsr_acc_opencl_info_memptr_t* info = NULL;
    cl_event event = NULL;
    size_t offset = 0;
    union {
      const void* input;
      void* ptr;
    } nconst = {dev_mem};
    const c_dbcsr_acc_opencl_stream_t* str;
    ACC_OPENCL_ACQUIRE(c_dbcsr_acc_opencl_config.lock_memory);
    str = (NULL != stream ? ACC_OPENCL_STREAM(stream) : c_dbcsr_acc_opencl_stream(NULL, ACC_OPENCL_OMP_TID()));
    assert(NULL != str);
    info = c_dbcsr_acc_opencl_info_devptr_modify(NULL, nconst.ptr, 1 /*elsize*/, &nbytes, &offset);
    if (NULL == info) {
      result = c_dbcsr_acc_opencl_memcpy_d2h(
        dev_mem, host_mem, offset, nbytes, str->queue, finish, NULL == c_dbcsr_acc_opencl_config.hist_d2h ? NULL : &event);
    }
    else {
      result = c_dbcsr_acc_opencl_memcpy_d2h(
        info->memory, host_mem, offset, nbytes, str->queue, finish, NULL == c_dbcsr_acc_opencl_config.hist_d2h ? NULL : &event);
      /*if (NULL != event && EXIT_SUCCESS == result) info->data = (void*)libxsmm_timer_tick();*/
    }
    ACC_OPENCL_RELEASE(c_dbcsr_acc_opencl_config.lock_memory);
    if (NULL != event) { /* c_dbcsr_acc_memcpy_notify must be outside of locked region */
      if (EXIT_SUCCESS == result) {
        assert(NULL != c_dbcsr_acc_opencl_config.hist_d2h /*&& NULL == c_dbcsr_acc_opencl_config.device.clEnqueueMemcpyINTEL*/);
        if (!finish) { /* asynchronous */
          result = clSetEventCallback(event, CL_COMPLETE, c_dbcsr_acc_memcpy_notify, nconst.ptr);
        }
        else c_dbcsr_acc_memcpy_notify(event, CL_COMPLETE, nconst.ptr); /* synchronous */
      }
      else ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseEvent(event));
    }
  }
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_memcpy_d2d(const void* devmem_src, void* devmem_dst, size_t nbytes, void* stream) {
  int result = EXIT_SUCCESS;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
  assert((NULL != devmem_src && NULL != devmem_dst) || 0 == nbytes);
  if (NULL != devmem_src && devmem_src != devmem_dst && 0 != nbytes) {
#  if defined(ACC_OPENCL_ASYNC)
    cl_event event = NULL, *const pevent = (0 == (4 & c_dbcsr_acc_opencl_config.async) || NULL == stream) ? &event : NULL;
#  else
    cl_event event = NULL, *const pevent = NULL;
#  endif
    const c_dbcsr_acc_opencl_device_t* const devinfo = &c_dbcsr_acc_opencl_config.device;
    union {
      const void* input;
      void* ptr;
    } nconst = {devmem_src};
    const c_dbcsr_acc_opencl_stream_t* str;
    ACC_OPENCL_ACQUIRE(c_dbcsr_acc_opencl_config.lock_memory);
    str = (NULL != stream ? ACC_OPENCL_STREAM(stream) : c_dbcsr_acc_opencl_stream(NULL, ACC_OPENCL_OMP_TID()));
    assert(NULL != str && NULL != devinfo->context);
#  if (1 >= ACC_OPENCL_USM)
    if (NULL != devinfo->clEnqueueMemcpyINTEL) {
      result = devinfo->clEnqueueMemcpyINTEL(str->queue, CL_FALSE /*blocking*/, devmem_dst, devmem_src, nbytes, 0, NULL, pevent);
    }
    else
#  endif
#  if (0 != ACC_OPENCL_USM)
      if (0 != devinfo->usm)
    {
#    if (1 >= ACC_OPENCL_USM) || defined(ACC_OPENCL_MEM_SVM_USM)
      result = clEnqueueSVMMemcpy(str->queue, CL_FALSE /*blocking*/, devmem_dst, devmem_src, nbytes, 0, NULL,
        NULL == c_dbcsr_acc_opencl_config.hist_d2d ? pevent : &event);
#    else
      memcpy(devmem_dst, devmem_src, nbytes);
#    endif
    }
    else
#  endif
    {
      size_t offset_src = 0, offset_dst = 0;
      c_dbcsr_acc_opencl_info_memptr_t* const info_src = c_dbcsr_acc_opencl_info_devptr_modify(
        NULL, nconst.ptr, 1 /*elsize*/, &nbytes, &offset_src);
      c_dbcsr_acc_opencl_info_memptr_t* const info_dst = c_dbcsr_acc_opencl_info_devptr_modify(
        NULL, devmem_dst, 1 /*elsize*/, &nbytes, &offset_dst);
      if (NULL != info_src && NULL != info_dst) {
        result = clEnqueueCopyBuffer(str->queue, info_src->memory, info_dst->memory, offset_src, offset_dst, nbytes, 0, NULL,
          NULL == c_dbcsr_acc_opencl_config.hist_d2d ? pevent : &event);
        /*if (NULL != event && EXIT_SUCCESS == result && NULL != c_dbcsr_acc_opencl_config.hist_d2d) {
          info_src->data = (void*)libxsmm_timer_tick();
        }*/
      }
      else result = EXIT_FAILURE;
    }
    ACC_OPENCL_RELEASE(c_dbcsr_acc_opencl_config.lock_memory);
    if (NULL != event) { /* c_dbcsr_acc_memcpy_notify must be outside of locked region */
      if (EXIT_SUCCESS == result) {
        if (NULL == pevent) { /* asynchronous */
          assert(NULL == devinfo->clEnqueueMemcpyINTEL);
          assert(NULL != c_dbcsr_acc_opencl_config.hist_d2d);
          result = clSetEventCallback(event, CL_COMPLETE, c_dbcsr_acc_memcpy_notify, nconst.ptr);
        }
        else { /* synchronous */
          result = clWaitForEvents(1, &event);
          if (EXIT_SUCCESS == result) {
            if (NULL != c_dbcsr_acc_opencl_config.hist_d2d) {
              assert(NULL == devinfo->clEnqueueMemcpyINTEL);
              c_dbcsr_acc_memcpy_notify(event, CL_COMPLETE, nconst.ptr);
            }
            else result = clReleaseEvent(event);
          }
          else ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseEvent(event));
        }
      }
      else ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseEvent(event));
    }
  }
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_memset(void* dev_mem, int value, size_t offset, size_t nbytes, void* stream) {
  int result = EXIT_SUCCESS;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
  assert(NULL != dev_mem || 0 == nbytes);
  if (0 != nbytes) {
#  if defined(ACC_OPENCL_ASYNC)
    cl_event event = NULL, *const pevent = (0 == (8 & c_dbcsr_acc_opencl_config.async) || NULL == stream) ? &event : NULL;
#  else
    cl_event event = NULL, *const pevent = NULL;
#  endif
    const c_dbcsr_acc_opencl_device_t* const devinfo = &c_dbcsr_acc_opencl_config.device;
    const c_dbcsr_acc_opencl_stream_t* str;
    size_t base = 0, vsize = 1;
    if (0 == LIBXSMM_MOD2(nbytes, 4)) vsize = 4;
    else if (0 == LIBXSMM_MOD2(nbytes, 2)) vsize = 2;
    ACC_OPENCL_ACQUIRE(c_dbcsr_acc_opencl_config.lock_memory);
    str = (NULL != stream ? ACC_OPENCL_STREAM(stream) : c_dbcsr_acc_opencl_stream(NULL, ACC_OPENCL_OMP_TID()));
    assert(NULL != str && NULL != devinfo->context);
#  if (1 >= ACC_OPENCL_USM)
    if (NULL != devinfo->clEnqueueMemFillINTEL) {
      result = devinfo->clEnqueueMemFillINTEL(str->queue, (char*)dev_mem + offset, &value, vsize, nbytes, 0, NULL, pevent);
    }
    else
#  endif
#  if (0 != ACC_OPENCL_USM)
      if (0 != devinfo->usm)
    {
#    if (1 >= ACC_OPENCL_USM) || defined(ACC_OPENCL_MEM_SVM_USM)
      result = clEnqueueSVMMemFill(str->queue, (char*)dev_mem + offset, &value, vsize, nbytes, 0, NULL, pevent);
#    else
      memset((char*)dev_mem + offset, value, nbytes);
#    endif
    }
    else
#  endif
    {
      const c_dbcsr_acc_opencl_info_memptr_t* const info = c_dbcsr_acc_opencl_info_devptr_modify(
        NULL, dev_mem, 1 /*elsize*/, &nbytes, &base);
      if (NULL != info) {
        result = clEnqueueFillBuffer(str->queue, info->memory, &value, vsize, base + offset, nbytes, 0, NULL, pevent);
        dev_mem = info->memptr;
      }
      else result = EXIT_FAILURE;
    }
    ACC_OPENCL_RELEASE(c_dbcsr_acc_opencl_config.lock_memory);
    if (NULL != event) {
      int result_release;
      if (EXIT_SUCCESS == result) result = clWaitForEvents(1, &event);
      result_release = clReleaseEvent(event);
      if (EXIT_SUCCESS == result) result = result_release;
    }
  }
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_memset_zero(void* dev_mem, size_t offset, size_t nbytes, void* stream) {
  return c_dbcsr_acc_opencl_memset(dev_mem, 0 /*value*/, offset, nbytes, stream);
}


int c_dbcsr_acc_opencl_info_devmem(cl_device_id device, size_t* mem_free, size_t* mem_total, size_t* mem_local, int* mem_unified) {
  int result = EXIT_SUCCESS, unified = 0;
  size_t size_free = 0, size_total = 0, size_local = 0;
  cl_device_local_mem_type cl_local_type = CL_GLOBAL;
  cl_ulong cl_size_total = 0, cl_size_local = 0;
  cl_bool cl_unified = CL_FALSE;
#  if !defined(_WIN32)
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
#  else
  MEMORYSTATUSEX mem_status;
  mem_status.dwLength = sizeof(mem_status);
  if (GlobalMemoryStatusEx(&mem_status)) {
    size_total = (size_t)mem_status.ullTotalPhys;
    size_free = (size_t)mem_status.ullAvailPhys;
  }
#  endif
  ACC_OPENCL_CHECK(result, clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &cl_size_total, NULL),
    "retrieve amount of global memory");
  ACC_OPENCL_CHECK(result,
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(cl_device_local_mem_type), &cl_local_type, NULL),
    "retrieve kind of local memory");
  if (CL_LOCAL == cl_local_type) {
    ACC_OPENCL_CHECK(result, clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &cl_size_local, NULL),
      "retrieve amount of local memory");
  }
  ACC_OPENCL_CHECK(result, clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &cl_unified, NULL),
    "retrieve if host memory is unified");
  if (EXIT_SUCCESS == result) {
    if (cl_size_total < size_total) size_total = cl_size_total;
    if (size_total < size_free) size_free = size_total;
    size_local = cl_size_local;
    unified = cl_unified;
    assert(size_free <= size_total);
  }
  assert(NULL != mem_local || NULL != mem_total || NULL != mem_free || NULL != mem_unified);
  if (NULL != mem_unified) *mem_unified = unified;
  if (NULL != mem_local) *mem_local = size_local;
  if (NULL != mem_total) *mem_total = size_total;
  if (NULL != mem_free) *mem_free = size_free;
  return result;
}


int c_dbcsr_acc_dev_mem_info(size_t* mem_free, size_t* mem_total) {
  const cl_device_id device_id = c_dbcsr_acc_opencl_config.devices[c_dbcsr_acc_opencl_config.device_id];
  int result;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
  result = c_dbcsr_acc_opencl_info_devmem(device_id, mem_free, mem_total, NULL /*mem_local*/, NULL /*mem_unified*/);
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}

#  if defined(__cplusplus)
}
#  endif

#endif /*__OPENCL*/
