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

#  if defined(CL_VERSION_2_0)
#    define ACC_OPENCL_STREAM_PROPERTIES_TYPE cl_queue_properties
#    define ACC_OPENCL_CREATE_COMMAND_QUEUE(CTX, DEV, PROPS, RESULT) clCreateCommandQueueWithProperties(CTX, DEV, PROPS, RESULT)
#  else
#    define ACC_OPENCL_STREAM_PROPERTIES_TYPE cl_int
#    define ACC_OPENCL_CREATE_COMMAND_QUEUE(CTX, DEV, PROPS, RESULT) \
      clCreateCommandQueue(CTX, DEV, (cl_command_queue_properties)(NULL != (PROPS) ? ((PROPS)[1]) : 0), RESULT)
#  endif


#  if defined(__cplusplus)
extern "C" {
#  endif

int c_dbcsr_acc_opencl_stream_counter_base;
int c_dbcsr_acc_opencl_stream_counter;


c_dbcsr_acc_opencl_info_stream_t* c_dbcsr_acc_opencl_info_stream(void* stream) {
  assert(NULL == stream || sizeof(c_dbcsr_acc_opencl_info_stream_t) <= (uintptr_t)stream);
  return (
    NULL != stream ? ((c_dbcsr_acc_opencl_info_stream_t*)((uintptr_t)stream - sizeof(c_dbcsr_acc_opencl_info_stream_t))) : NULL);
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


int c_dbcsr_acc_stream_create(void** stream_p, const char* name, int priority) {
  ACC_OPENCL_STREAM_PROPERTIES_TYPE properties[8] = {
    CL_QUEUE_PROPERTIES, 0 /*placeholder*/, 0 /* terminator */
  };
  int result, i, tid = 0, offset = 0;
  cl_command_queue queue = NULL;
  cl_context context = NULL;
  void** streams = NULL;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(NULL != stream_p);
#  if !defined(ACC_OPENCL_STREAM_PRIORITIES)
  LIBXSMM_UNUSED(priority);
#  else
  if (CL_QUEUE_PRIORITY_HIGH_KHR <= priority && CL_QUEUE_PRIORITY_LOW_KHR >= priority) {
    properties[3] = priority;
  }
  else {
    int least = -1, greatest = -1;
    if (0 != (1 & c_dbcsr_acc_opencl_config.priority) && EXIT_SUCCESS == c_dbcsr_acc_stream_priority_range(&least, &greatest) &&
        least != greatest)
    {
      properties[3] = (0 != (2 & c_dbcsr_acc_opencl_config.priority) &&
                        (NULL != LIBXSMM_STRISTR(name, "calc") || (NULL != strstr(name, "priority"))))
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
    if (NULL != c_dbcsr_acc_opencl_config.device) { /* inherit master's context if current context is NULL */
      LIBXSMM_ATOMIC_CMPSWP(&c_dbcsr_acc_opencl_config.device[tid].context, NULL,
        c_dbcsr_acc_opencl_config.device[/*main*/ 0].context, LIBXSMM_ATOMIC_RELAXED);
    }
  }
  else offset = c_dbcsr_acc_opencl_stream_counter_base++;
#  endif
  if (NULL != c_dbcsr_acc_opencl_config.device) context = c_dbcsr_acc_opencl_config.device[tid].context;
  if (NULL != context) {
    cl_device_id device = NULL;
    result = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);
    if (CL_SUCCESS == result) {
      const int s = c_dbcsr_acc_opencl_config.share;
      if (0 == s || 0 == (tid % s)) {
        if (0 != c_dbcsr_acc_opencl_config.device[tid].intel) {
          const int xhints = ((1 == c_dbcsr_acc_opencl_config.xhints || 0 > c_dbcsr_acc_opencl_config.xhints)
                                ? (0 != c_dbcsr_acc_opencl_config.device[tid].intel ? 1 : 0)
                                : (c_dbcsr_acc_opencl_config.xhints >> 1));
          if (0 != (1 & xhints)) { /* attempt to enable command aggregation */
            const ACC_OPENCL_STREAM_PROPERTIES_TYPE props[4] = {
              CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0 /* terminator */
            };
            const cl_command_queue q = ACC_OPENCL_CREATE_COMMAND_QUEUE(context, device, props, &result);
            if (CL_SUCCESS == result) {
              c_dbcsr_acc_opencl_config.timer = c_dbcsr_acc_opencl_timer_host; /* force host-timer */
              clReleaseCommandQueue(q);
            }
            else result = CL_SUCCESS;
          }
          if (0 != (2 & xhints)) { /* attempt to enable queue families */
            struct {
              cl_command_queue_properties properties;
              cl_bitfield capabilities;
              cl_uint count;
              char name[64 /*CL_QUEUE_FAMILY_MAX_NAME_SIZE_INTEL*/];
            } intel_qfprops[16];
            size_t nbytes = 0, i;
            if (CL_SUCCESS == clGetDeviceInfo(device, 0x418B /*CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL*/, sizeof(intel_qfprops),
                                intel_qfprops, &nbytes))
            {
              for (i = 0; (i * sizeof(*intel_qfprops)) < nbytes; ++i) {
                if (0 /*CL_QUEUE_DEFAULT_CAPABILITIES_INTEL*/ == intel_qfprops[i].capabilities && 1 < intel_qfprops[i].count) {
                  const int j = (0 /*terminator*/ == properties[2] ? 2 : 4);
                  properties[j + 0] = 0x418C; /* CL_QUEUE_FAMILY_INTEL */
                  properties[j + 1] = (int)i;
                  properties[j + 2] = 0x418D; /* CL_QUEUE_INDEX_INTEL */
                  properties[j + 3] = (i + offset) % intel_qfprops[i].count;
                  properties[j + 4] = 0; /* terminator */
                  break;
                }
              }
            }
          }
        }
        if ((c_dbcsr_acc_opencl_timer_device == c_dbcsr_acc_opencl_config.timer) &&
            (3 <= c_dbcsr_acc_opencl_config.verbosity || 0 > c_dbcsr_acc_opencl_config.verbosity))
        {
          properties[1] = CL_QUEUE_PROFILING_ENABLE;
        }
        queue = ACC_OPENCL_CREATE_COMMAND_QUEUE(context, device, properties, &result);
      }
      else { /* attempt to share existing stream */
        const int maxn = c_dbcsr_acc_opencl_config.nthreads;
        cl_command_queue stream = NULL;
        assert(0 < tid);
        for (i = 0; i < maxn; ++i) {
          int j = i + tid - 1, t = (j < maxn ? j : (j - maxn));
          if (0 < t && t != tid) { /* avoid cloning master's and own streams */
            streams = c_dbcsr_acc_opencl_config.streams + ACC_OPENCL_STREAMS_MAXCOUNT * t;
            for (j = 0; j < ACC_OPENCL_STREAMS_MAXCOUNT; ++j) {
              if (NULL != streams[j]) {
                stream = *ACC_OPENCL_STREAM(streams[j]);
                i = maxn; /* break outer loop */
                break;
              }
            }
          }
        }
        if (NULL != stream) { /* clone existing stream */
          result = clRetainCommandQueue(stream);
        }
        else {
          for (i = 0; i < ACC_OPENCL_STREAMS_MAXCOUNT; ++i) { /* adopt master's stream created last */
            if (NULL != c_dbcsr_acc_opencl_config.streams[i]) {
              stream = *ACC_OPENCL_STREAM(c_dbcsr_acc_opencl_config.streams[i]);
              result = clRetainCommandQueue(stream);
            }
            else break;
          }
        }
        if (EXIT_SUCCESS == result) queue = stream;
      }
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    const int base = ACC_OPENCL_STREAMS_MAXCOUNT * tid;
    cl_command_queue* const stats = c_dbcsr_acc_opencl_config.stats + base;
    streams = c_dbcsr_acc_opencl_config.streams + base;
    for (i = 0; i < ACC_OPENCL_STREAMS_MAXCOUNT; ++i) {
      if (NULL == streams[i]) break;
    }
    if (i < ACC_OPENCL_STREAMS_MAXCOUNT) { /* register stream */
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
        info->tid = tid;
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
    }
    else {
      clReleaseCommandQueue(queue);
      result = EXIT_FAILURE;
      *stream_p = NULL;
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
#  if defined(_OPENMP)
#    pragma omp critical(c_dbcsr_acc_opencl_stream)
#  endif
  if (NULL != stream) {
    const cl_command_queue queue = *ACC_OPENCL_STREAM(stream);
    assert(NULL != c_dbcsr_acc_opencl_config.streams);
    if (NULL != queue) {
      int tid = 0, i = ACC_OPENCL_STREAMS_MAXCOUNT;
      void** streams = NULL;
      for (; tid < c_dbcsr_acc_opencl_config.nthreads; ++tid) { /* unregister */
        streams = c_dbcsr_acc_opencl_config.streams + ACC_OPENCL_STREAMS_MAXCOUNT * tid;
        for (i = 0; i < ACC_OPENCL_STREAMS_MAXCOUNT; ++i) {
          if (stream == streams[i]) {
            const int j = i + 1, result_release = clReleaseCommandQueue(queue); /* soft-error */
            if (j < ACC_OPENCL_STREAMS_MAXCOUNT && NULL != streams[j]) { /* compacting streams is not thread-safe */
              memmove(streams + i, streams + j, sizeof(void*) * (ACC_OPENCL_STREAMS_MAXCOUNT - j));
            }
            streams[ACC_OPENCL_STREAMS_MAXCOUNT - j] = NULL;
            /* consider breaking outer loop */
            if (0 == c_dbcsr_acc_opencl_config.share) {
              tid = c_dbcsr_acc_opencl_config.nthreads;
              result = result_release; /* promote */
            }
            else if (EXIT_SUCCESS != result_release) {
              tid = c_dbcsr_acc_opencl_config.nthreads;
            }
            break;
          }
          else if (NULL == streams[i]) { /* compact streams */
            break;
          }
        }
      }
    }
    c_dbcsr_acc_opencl_stream_counter_base = c_dbcsr_acc_opencl_stream_counter = 0; /* reset */
    free(c_dbcsr_acc_opencl_info_stream(stream)->pointer);
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
      if (NULL != strstr(buffer, "cl_khr_priority_hints") || EXIT_SUCCESS == c_dbcsr_acc_opencl_device_vendor(active_id, "nvidia"))
      {
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
  const int* const priority = (0 == (1 & c_dbcsr_acc_opencl_config.flush) ? NULL : c_dbcsr_acc_opencl_stream_priority(stream));
#  endif
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(NULL != stream);
#  if defined(ACC_OPENCL_STREAM_PRIORITIES)
  if (NULL != priority && CL_QUEUE_PRIORITY_HIGH_KHR <= *priority && CL_QUEUE_PRIORITY_MED_KHR > *priority) {
    if (0 != (2 & c_dbcsr_acc_opencl_config.flush)) {
      result = clFlush(*ACC_OPENCL_STREAM(stream));
    }
  }
  else
#  endif
  {
    const cl_command_queue queue = *ACC_OPENCL_STREAM(stream);
    result = clFinish(queue);
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
