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
#  include <libxsmm_sync.h>
#  include <stdlib.h>
#  include <string.h>
#  include <assert.h>
#  if defined(_OPENMP)
#    include <omp.h>
#  endif

#  if defined(CL_VERSION_2_0)
#    define ACC_OPENCL_CREATE_COMMAND_QUEUE(CTX, DEV, PROPS, RESULT) clCreateCommandQueueWithProperties(CTX, DEV, PROPS, RESULT)
#  else
#    define ACC_OPENCL_CREATE_COMMAND_QUEUE(CTX, DEV, PROPS, RESULT) \
      clCreateCommandQueue(CTX, DEV, (cl_command_queue_properties)(NULL != (PROPS) ? ((PROPS)[1]) : 0), RESULT)
#  endif


#  if defined(__cplusplus)
extern "C" {
#  endif

int c_dbcsr_acc_opencl_stream_counter;


c_dbcsr_acc_opencl_info_stream_t* c_dbcsr_acc_opencl_info_stream(void* stream) {
  c_dbcsr_acc_opencl_info_stream_t* result;
#  if defined(ACC_OPENCL_STREAM_NOALLOC)
  LIBXSMM_UNUSED(stream);
#  else
  assert(NULL == stream || sizeof(c_dbcsr_acc_opencl_info_stream_t) <= (uintptr_t)stream);
  if (NULL != stream) {
    result = (c_dbcsr_acc_opencl_info_stream_t*)((uintptr_t)stream - sizeof(c_dbcsr_acc_opencl_info_stream_t));
  }
  else
#  endif
  result = NULL;
  return result;
}


const int* c_dbcsr_acc_opencl_stream_priority(const void* stream) {
  const int* result;
#  if !defined(ACC_OPENCL_STREAM_PRIORITIES)
  LIBXSMM_UNUSED(stream);
#  else
  const c_dbcsr_acc_opencl_info_stream_t* const info = c_dbcsr_acc_opencl_info_stream((void*)stream);
  if (NULL != info) {
    result = &info->priority;
  }
  else
#  endif
  result = NULL;
  return result;
}


int c_dbcsr_acc_opencl_stream_is_thread_specific(int thread_id, const void* stream) {
  void** const streams = c_dbcsr_acc_opencl_config.streams + ACC_OPENCL_STREAMS_MAXCOUNT * thread_id;
  assert(0 <= thread_id && thread_id < c_dbcsr_acc_opencl_config.nthreads);
  assert(NULL != c_dbcsr_acc_opencl_config.streams);
  if (NULL != stream) {
    int i = 0;
    for (; i < ACC_OPENCL_STREAMS_MAXCOUNT; ++i) {
      if (stream == streams[i]) return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}


int c_dbcsr_acc_stream_create(void** stream_p, const char* name, int priority) {
  void** streams = NULL;
  int result, tid, i;
  ACC_OPENCL_COMMAND_QUEUE_PROPERTIES properties[8] = {
    CL_QUEUE_PROPERTIES, 0 /*placeholder*/, 0 /* terminator */
  };
  cl_command_queue queue = NULL;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
#  if !defined(ACC_OPENCL_STREAM_PRIORITIES)
  LIBXSMM_UNUSED(priority);
#  else
  if (CL_QUEUE_PRIORITY_HIGH_KHR <= priority && CL_QUEUE_PRIORITY_LOW_KHR >= priority) {
    properties[3] = priority;
  }
  else {
    int least = -1, greatest = -1;
    if (0 != (1 & c_dbcsr_acc_opencl_config.priority) && EXIT_SUCCESS == c_dbcsr_acc_stream_priority_range(&least, &greatest) &&
        least != greatest) {
      properties[3] = (0 != (2 & c_dbcsr_acc_opencl_config.priority) &&
                        (NULL != libxsmm_stristr(name, "calc") || (NULL != strstr(name, "priority"))))
                        ? CL_QUEUE_PRIORITY_HIGH_KHR
                        : CL_QUEUE_PRIORITY_MED_KHR;
    }
    else {
      properties[3] = least;
    }
  }
  if (CL_QUEUE_PRIORITY_HIGH_KHR <= properties[3] && CL_QUEUE_PRIORITY_LOW_KHR >= properties[3]) {
    priority = properties[3]; /* sanitize */
    properties[2] = CL_QUEUE_PRIORITY_KHR;
    properties[4] = 0; /* terminator */
  }
#  endif
  if (3 <= c_dbcsr_acc_opencl_config.verbosity || 0 > c_dbcsr_acc_opencl_config.verbosity) {
    properties[1] = CL_QUEUE_PROFILING_ENABLE;
  }
  ACC_OPENCL_DEBUG_IF(NULL == c_dbcsr_acc_opencl_config.contexts || NULL == c_dbcsr_acc_opencl_config.contexts[/*master*/ 0]) {
    ACC_OPENCL_DEBUG_FPRINTF(stderr,
      "ERROR ACC/OpenCL: "
      "pid=%u not initialized!\n",
      libxsmm_get_pid());
  }
#  if defined(_OPENMP)
  if (1 < omp_get_num_threads()) {
    assert(0 < c_dbcsr_acc_opencl_config.nthreads);
#    if (201107 /*v3.1*/ <= _OPENMP)
#      pragma omp atomic capture
#    else
#      pragma omp critical(c_dbcsr_acc_opencl_stream)
#    endif
    i = c_dbcsr_acc_opencl_stream_counter++;
    tid = (i < c_dbcsr_acc_opencl_config.nthreads ? i : (i % c_dbcsr_acc_opencl_config.nthreads));
    /* inherit master's context if current context is NULL */
    LIBXSMM_ATOMIC_CMPSWP(
      c_dbcsr_acc_opencl_config.contexts + tid, NULL, c_dbcsr_acc_opencl_config.contexts[/*master*/ 0], LIBXSMM_ATOMIC_RELAXED);
  }
  else
#  endif
  {
    tid = 0; /*master*/
  }
  {
    const cl_context context = c_dbcsr_acc_opencl_config.contexts[tid];
    if (NULL != context) {
      cl_device_id device = NULL;
      result = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);
      if (CL_SUCCESS == result) {
        const int s =
          (0 >= c_dbcsr_acc_opencl_config.share ? 0 : (1 < c_dbcsr_acc_opencl_config.share ? c_dbcsr_acc_opencl_config.share : 2));
        if (1 >= s || s >= c_dbcsr_acc_opencl_config.nthreads || 0 == (tid % s)) {
          queue = ACC_OPENCL_CREATE_COMMAND_QUEUE(context, device, properties, &result);
        }
        else {
          int n = 0;
          assert(0 < tid);
          for (i = 0; n < c_dbcsr_acc_opencl_config.nthreads; n += s) {
            const int j = n + tid - 1, t = j < c_dbcsr_acc_opencl_config.nthreads ? j : (j - c_dbcsr_acc_opencl_config.nthreads);
            if (0 != t) { /* avoid cloning master's streams */
              streams = c_dbcsr_acc_opencl_config.streams + ACC_OPENCL_STREAMS_MAXCOUNT * t;
              for (i = 0; i < ACC_OPENCL_STREAMS_MAXCOUNT; ++i) {
                if (NULL != streams[i]) {
                  n = c_dbcsr_acc_opencl_config.nthreads;
                  break;
                }
              }
            }
          }
          assert(i == ACC_OPENCL_STREAMS_MAXCOUNT || c_dbcsr_acc_opencl_config.streams <= streams);
          if (i < ACC_OPENCL_STREAMS_MAXCOUNT) { /* clone existing stream (share) */
            cl_command_queue stream;
            assert(NULL != streams);
            stream = *ACC_OPENCL_STREAM(streams[i]);
            result = clRetainCommandQueue(stream);
            if (CL_SUCCESS == result) queue = stream;
          }
          else {
            queue = ACC_OPENCL_CREATE_COMMAND_QUEUE(context, device, properties, &result);
          }
        }
      }
    }
    else {
      result = EXIT_FAILURE;
    }
  }
  assert(NULL != stream_p);
  if (EXIT_SUCCESS == result) {
    const int base = ACC_OPENCL_STREAMS_MAXCOUNT * tid;
    cl_command_queue* const stats = c_dbcsr_acc_opencl_config.stats + base;
    streams = c_dbcsr_acc_opencl_config.streams + base;
    for (i = 0; i < ACC_OPENCL_STREAMS_MAXCOUNT; ++i) {
      if (NULL == streams[i]) break;
    }
    if (i < ACC_OPENCL_STREAMS_MAXCOUNT) { /* register stream */
#  if defined(ACC_OPENCL_STREAM_NOALLOC)
      assert(sizeof(void*) >= sizeof(cl_command_queue) && NULL != queue);
      streams[i] = *stream_p = (void*)queue;
      stats[i] = queue;
#  else
      const size_t size_info = sizeof(c_dbcsr_acc_opencl_info_stream_t);
      const size_t size = sizeof(cl_command_queue) + sizeof(void*) + size_info - 1;
      void* const handle = malloc(size);
      assert(NULL != queue);
      if (NULL != handle) {
        const uintptr_t address = (uintptr_t)handle;
        const uintptr_t aligned = LIBXSMM_UP2(address + size_info, sizeof(void*));
        c_dbcsr_acc_opencl_info_stream_t* const info = (c_dbcsr_acc_opencl_info_stream_t*)(aligned - size_info);
        assert(address + size_info <= aligned && NULL != info);
        info->pointer = (void*)address;
        info->priority = priority;
        stats[i] = *(cl_command_queue*)aligned = queue;
        streams[i] = *stream_p = (void*)aligned;
        assert(queue == *ACC_OPENCL_STREAM(streams[i]));
        assert(queue == *ACC_OPENCL_STREAM(*stream_p));
      }
      else {
        clReleaseCommandQueue(queue);
        result = EXIT_FAILURE;
        *stream_p = NULL;
      }
#  endif
    }
    else {
      clReleaseCommandQueue(queue);
      result = EXIT_FAILURE;
      *stream_p = NULL;
    }
    ACC_OPENCL_DEBUG_IF(EXIT_SUCCESS == result) {
      ACC_OPENCL_DEBUG_FPRINTF(stderr, "INFO ACC/OpenCL: create stream \"%s\" (tid=%i", NULL != name ? name : "unknown", tid);
#  if defined(ACC_OPENCL_STREAM_PRIORITIES)
      ACC_OPENCL_DEBUG_FPRINTF(stderr, ", priority=%i [%i->%i->%i]", priority, CL_QUEUE_PRIORITY_LOW_KHR, CL_QUEUE_PRIORITY_MED_KHR,
        CL_QUEUE_PRIORITY_HIGH_KHR);
#  endif
      ACC_OPENCL_DEBUG_FPRINTF(stderr, ").\n");
    }
  }
  else {
    *stream_p = NULL;
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN_CAUSE(result, name);
}


int c_dbcsr_acc_stream_destroy(void* stream) {
  int result = EXIT_SUCCESS;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  if (NULL != stream) {
    result = clReleaseCommandQueue(*ACC_OPENCL_STREAM(stream));
    if (EXIT_SUCCESS == result) {
      int tid = 0, i = ACC_OPENCL_STREAMS_MAXCOUNT;
      void** streams = NULL;
      for (; tid < c_dbcsr_acc_opencl_config.nthreads; ++tid) { /* unregister */
        streams = c_dbcsr_acc_opencl_config.streams + ACC_OPENCL_STREAMS_MAXCOUNT * tid;
        for (i = 0; i < ACC_OPENCL_STREAMS_MAXCOUNT; ++i) {
          if (stream == streams[i]) {
            tid = c_dbcsr_acc_opencl_config.nthreads; /* break outer loop */
            break;
          }
        }
      }
      if (i < ACC_OPENCL_STREAMS_MAXCOUNT) {
        const int j = i + 1;
        assert(NULL != streams);
        streams[i] = NULL;
        /* compacting streams is not thread-safe */
        if (j < ACC_OPENCL_STREAMS_MAXCOUNT && NULL != streams[j]) {
          memmove(streams + i, streams + j, sizeof(cl_command_queue) * (ACC_OPENCL_STREAMS_MAXCOUNT - j));
        }
      }
      else {
        result = EXIT_FAILURE;
      }
    }
#  if defined(_OPENMP)
#    if (201107 /*v3.1*/ <= _OPENMP)
#      pragma omp atomic write
#    else
#      pragma omp critical(c_dbcsr_acc_opencl_stream)
#    endif
#  endif
    c_dbcsr_acc_opencl_stream_counter = 0; /* reset */
#  if defined(ACC_OPENCL_STREAM_NOALLOC)
    assert(sizeof(void*) >= sizeof(cl_command_queue));
#  else
    free(c_dbcsr_acc_opencl_info_stream(stream)->pointer);
#  endif
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_stream_priority_range(int* least, int* greatest) {
  int result = ((NULL != least || NULL != greatest) ? EXIT_SUCCESS : EXIT_FAILURE);
  int priohi = -1, priolo = -1;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(least != greatest); /* no alias */
#  if defined(ACC_OPENCL_STREAM_PRIORITIES)
  if (0 < c_dbcsr_acc_opencl_config.ndevices) {
    char buffer[ACC_OPENCL_BUFFERSIZE];
    cl_platform_id platform = NULL;
    cl_device_id active_id = NULL;
    if (EXIT_SUCCESS == result) {
      result = c_dbcsr_acc_opencl_device(ACC_OPENCL_OMP_TID(), &active_id);
    }
    ACC_OPENCL_CHECK(clGetDeviceInfo(active_id, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL),
      "retrieve platform associated with active device", result);
    ACC_OPENCL_CHECK(clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, ACC_OPENCL_BUFFERSIZE, buffer, NULL),
      "retrieve platform extensions", result);
    if (EXIT_SUCCESS == result) {
      if (NULL != strstr(buffer, "cl_khr_priority_hints") ||
          EXIT_SUCCESS == c_dbcsr_acc_opencl_device_vendor(active_id, "nvidia")) {
        priohi = CL_QUEUE_PRIORITY_HIGH_KHR;
        priolo = CL_QUEUE_PRIORITY_LOW_KHR;
      }
    }
  }
#  endif
  if (NULL != greatest) *greatest = priohi;
  if (NULL != least) *least = priolo;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_stream_sync(void* stream) {
  int result = EXIT_SUCCESS;
#  if defined(ACC_OPENCL_STREAM_PRIORITIES)
  const int* const priority = (0 == (1 & c_dbcsr_acc_opencl_config.flush) ? c_dbcsr_acc_opencl_stream_priority(stream) : NULL);
#  endif
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(NULL != stream);
  ACC_OPENCL_DEBUG_IF(EXIT_SUCCESS != c_dbcsr_acc_opencl_stream_is_thread_specific(ACC_OPENCL_OMP_TID(), stream)) {
    ACC_OPENCL_DEBUG_FPRINTF(stderr, "WARNING ACC/OpenCL: "
                                     "c_dbcsr_acc_stream_sync called by foreign thread!\n");
  }
#  if defined(ACC_OPENCL_STREAM_PRIORITIES)
  if (NULL != priority && CL_QUEUE_PRIORITY_HIGH_KHR <= *priority && CL_QUEUE_PRIORITY_MED_KHR > *priority) {
    if (0 != (2 & c_dbcsr_acc_opencl_config.flush)) {
      result = clFlush(*ACC_OPENCL_STREAM(stream));
    }
  }
  else
#  endif
  {
    result = clFinish(*ACC_OPENCL_STREAM(stream));
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
