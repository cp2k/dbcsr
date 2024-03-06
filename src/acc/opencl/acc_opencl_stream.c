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


#  if defined(__cplusplus)
extern "C" {
#  endif

int c_dbcsr_acc_opencl_stream_counter_base;
int c_dbcsr_acc_opencl_stream_counter;


const c_dbcsr_acc_opencl_stream_t* c_dbcsr_acc_opencl_stream(ACC_OPENCL_LOCKTYPE* lock, int thread_id) {
  const c_dbcsr_acc_opencl_stream_t *result = NULL, *result_main = NULL;
  const size_t n = ACC_OPENCL_MAXNITEMS * c_dbcsr_acc_opencl_config.nthreads;
  size_t i;
  assert(NULL != c_dbcsr_acc_opencl_config.streams);
  assert(thread_id < c_dbcsr_acc_opencl_config.nthreads);
  if (NULL != lock) ACC_OPENCL_ACQUIRE(lock);
  for (i = c_dbcsr_acc_opencl_config.nstreams; i < n; ++i) {
    const c_dbcsr_acc_opencl_stream_t* const str = c_dbcsr_acc_opencl_config.streams[i];
    if (NULL != str && NULL != str->queue) {
      if (str->tid == thread_id || 0 > thread_id) { /* hit */
        result = str;
        break;
      }
      else if (NULL == result_main && 0 == str->tid) {
        result_main = str;
      }
    }
    else break; /* error */
  }
  if (NULL == result) { /* fallback */
    result = (NULL != result_main ? result_main : &c_dbcsr_acc_opencl_config.device.stream);
  }
  if (NULL != lock) ACC_OPENCL_RELEASE(lock);
  return result;
}


const c_dbcsr_acc_opencl_stream_t* c_dbcsr_acc_opencl_stream_default(void) {
  const c_dbcsr_acc_opencl_stream_t* result = NULL;
  result = c_dbcsr_acc_opencl_stream(c_dbcsr_acc_opencl_config.lock_stream, ACC_OPENCL_OMP_TID());
  assert(NULL != result);
  return result;
}


int c_dbcsr_acc_stream_create(void** stream_p, const char* name, int priority) {
  ACC_OPENCL_STREAM_PROPERTIES_TYPE properties[8] = {
    CL_QUEUE_PROPERTIES, 0 /*placeholder*/, 0 /* terminator */
  };
  int result, tid = 0, offset = 0;
  cl_command_queue queue = NULL;
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
  ACC_OPENCL_ACQUIRE(c_dbcsr_acc_opencl_config.lock_stream);
#  if defined(_OPENMP)
  if (1 < omp_get_num_threads()) {
    const int i = c_dbcsr_acc_opencl_stream_counter++;
    assert(0 < c_dbcsr_acc_opencl_config.nthreads);
    tid = (i < c_dbcsr_acc_opencl_config.nthreads ? i : (i % c_dbcsr_acc_opencl_config.nthreads));
  }
  else offset = c_dbcsr_acc_opencl_stream_counter_base++;
#  endif
  if (NULL != c_dbcsr_acc_opencl_config.device.context) {
    cl_device_id device = NULL;
    result = clGetContextInfo(c_dbcsr_acc_opencl_config.device.context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);
    if (EXIT_SUCCESS == result) {
      if (0 != (2 & c_dbcsr_acc_opencl_config.xhints) && 0 != c_dbcsr_acc_opencl_config.device.intel) { /* enable queue families */
        struct {
          cl_command_queue_properties properties;
          cl_bitfield capabilities;
          cl_uint count;
          char name[64 /*CL_QUEUE_FAMILY_MAX_NAME_SIZE_INTEL*/];
        } intel_qfprops[16];
        size_t nbytes = 0, i;
        if (EXIT_SUCCESS == clGetDeviceInfo(device, 0x418B /*CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL*/, sizeof(intel_qfprops),
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
      if ((c_dbcsr_acc_opencl_timer_device == c_dbcsr_acc_opencl_config.timer) &&
          (3 <= c_dbcsr_acc_opencl_config.verbosity || 0 > c_dbcsr_acc_opencl_config.verbosity))
      {
        properties[1] = CL_QUEUE_PROFILING_ENABLE;
      }
      queue = ACC_OPENCL_CREATE_COMMAND_QUEUE(c_dbcsr_acc_opencl_config.device.context, device, properties, &result);
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) { /* register stream */
    assert(NULL != c_dbcsr_acc_opencl_config.streams && NULL != queue);
    *stream_p = c_dbcsr_acc_opencl_pmalloc(
      NULL /*lock*/, (void**)c_dbcsr_acc_opencl_config.streams, &c_dbcsr_acc_opencl_config.nstreams);
    if (NULL != *stream_p) {
      c_dbcsr_acc_opencl_stream_t* const str = (c_dbcsr_acc_opencl_stream_t*)*stream_p;
#  if !defined(NDEBUG)
      LIBXSMM_MEMZERO127(str);
#  endif
      str->queue = queue;
      str->tid = tid;
#  if defined(ACC_OPENCL_STREAM_PRIORITIES)
      str->priority = priority;
#  endif
    }
    else result = EXIT_FAILURE;
  }
  ACC_OPENCL_RELEASE(c_dbcsr_acc_opencl_config.lock_stream);
  if (EXIT_SUCCESS != result && NULL != queue) {
    clReleaseCommandQueue(queue);
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
    const c_dbcsr_acc_opencl_stream_t* const str = ACC_OPENCL_STREAM(stream);
    const cl_command_queue queue = str->queue;
    assert(NULL != c_dbcsr_acc_opencl_config.streams);
    ACC_OPENCL_ACQUIRE(c_dbcsr_acc_opencl_config.lock_stream);
    c_dbcsr_acc_opencl_pfree(NULL /*lock*/, stream, (void**)c_dbcsr_acc_opencl_config.streams, &c_dbcsr_acc_opencl_config.nstreams);
    if (NULL != queue) {
      result = clReleaseCommandQueue(queue);
#  if !defined(NDEBUG)
      LIBXSMM_MEMZERO127((c_dbcsr_acc_opencl_stream_t*)stream);
#  endif
    }
    ACC_OPENCL_RELEASE(c_dbcsr_acc_opencl_config.lock_stream);
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
      result = clGetContextInfo(
        c_dbcsr_acc_opencl_config.device.context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &active_id, NULL);
    }
    ACC_OPENCL_CHECK(clGetDeviceInfo(active_id, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL),
      "retrieve platform associated with active device", result);
    ACC_OPENCL_CHECK(clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, ACC_OPENCL_BUFFERSIZE, buffer, NULL),
      "retrieve platform extensions", result);
    if (EXIT_SUCCESS == result) {
      if (NULL != strstr(buffer, "cl_khr_priority_hints") ||
          EXIT_SUCCESS == c_dbcsr_acc_opencl_device_vendor(active_id, "nvidia", 0 /*use_platform_name*/))
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
  const c_dbcsr_acc_opencl_stream_t* str = NULL;
  int result = EXIT_SUCCESS;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  str = (NULL != stream ? ACC_OPENCL_STREAM(stream) : c_dbcsr_acc_opencl_stream_default());
  assert(NULL != str && NULL != str->queue);
  result = clFinish(str->queue);
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_device_synchronize(ACC_OPENCL_LOCKTYPE* lock, int thread_id) {
  int result = EXIT_SUCCESS;
  const size_t n = ACC_OPENCL_MAXNITEMS * c_dbcsr_acc_opencl_config.nthreads;
  size_t i;
  assert(thread_id < c_dbcsr_acc_opencl_config.nthreads);
  assert(NULL != c_dbcsr_acc_opencl_config.streams);
  if (NULL != lock) ACC_OPENCL_ACQUIRE(lock);
  for (i = c_dbcsr_acc_opencl_config.nstreams; i < n; ++i) {
    const c_dbcsr_acc_opencl_stream_t* const str = c_dbcsr_acc_opencl_config.streams[i];
    if (NULL != str && NULL != str->queue) {
      if (str->tid == thread_id || 0 > thread_id) { /* hit */
        result = clFinish(str->queue);
        if (EXIT_SUCCESS != result) break;
      }
    }
    else { /* error */
      result = EXIT_FAILURE;
      break;
    }
  }
  if (NULL != lock) ACC_OPENCL_RELEASE(lock);
  return result;
}


int c_dbcsr_acc_device_synchronize(void) {
  int result = EXIT_SUCCESS;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
#  if defined(_OPENMP)
  if (1 == omp_get_num_threads()) {
    result = c_dbcsr_acc_opencl_device_synchronize(c_dbcsr_acc_opencl_config.lock_stream, -1 /*all*/);
  }
  else {
    result = c_dbcsr_acc_opencl_device_synchronize(NULL /*lock*/, omp_get_thread_num());
  }
#  else
  result = c_dbcsr_acc_opencl_device_synchronize(NULL /*lock*/, /*main*/ 0);
#  endif
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}

#  if defined(__cplusplus)
}
#  endif

#endif /*__OPENCL*/
