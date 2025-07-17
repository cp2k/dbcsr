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


#  if defined(__cplusplus)
extern "C" {
#  endif

int c_dbcsr_acc_event_create(void** event_p) {
  int result = EXIT_SUCCESS;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
  assert(NULL != c_dbcsr_acc_opencl_config.events && NULL != event_p);
  *event_p = c_dbcsr_acc_opencl_pmalloc(
    c_dbcsr_acc_opencl_config.lock_event, (void**)c_dbcsr_acc_opencl_config.events, &c_dbcsr_acc_opencl_config.nevents);
  if (NULL != *event_p) *(cl_event*)*event_p = NULL;
  else result = EXIT_FAILURE;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_destroy(void* event) {
  int result = EXIT_SUCCESS;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
  if (NULL != event) {
    const cl_event clevent = *ACC_OPENCL_EVENT(event);
    assert(NULL != c_dbcsr_acc_opencl_config.events);
    ACC_OPENCL_ACQUIRE(c_dbcsr_acc_opencl_config.lock_event);
    c_dbcsr_acc_opencl_pfree(event, (void**)c_dbcsr_acc_opencl_config.events, &c_dbcsr_acc_opencl_config.nevents);
    if (NULL != clevent) {
      result = clReleaseEvent(clevent);
#  if !defined(NDEBUG)
      *(cl_event*)event = NULL;
#  endif
    }
    ACC_OPENCL_RELEASE(c_dbcsr_acc_opencl_config.lock_event);
  }
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_stream_wait_event(void* stream, void* event) { /* wait for an event (device-side) */
  int result = EXIT_SUCCESS;
  const c_dbcsr_acc_opencl_stream_t* str = NULL;
  cl_event clevent = NULL;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
  str = (NULL != stream ? ACC_OPENCL_STREAM(stream) : c_dbcsr_acc_opencl_stream_default());
  assert(NULL != str && NULL != str->queue && NULL != event);
  clevent = *ACC_OPENCL_EVENT(event);
  if (NULL != clevent) {
#  if defined(CL_VERSION_1_2)
    result = clEnqueueBarrierWithWaitList(str->queue, 1, &clevent, NULL);
#  else
    result = clEnqueueWaitForEvents(str->queue, 1, &clevent);
#  endif
    if (EXIT_SUCCESS != result) {
      ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseEvent(clevent));
      *(cl_event*)event = NULL;
    }
  }
  else if (3 <= c_dbcsr_acc_opencl_config.verbosity || 0 > c_dbcsr_acc_opencl_config.verbosity) {
    fprintf(stderr, "WARN ACC/OpenCL: c_dbcsr_acc_stream_wait_event discovered an empty event.\n");
  }
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_record(void* event, void* stream) {
  int result = EXIT_SUCCESS;
  const c_dbcsr_acc_opencl_stream_t* str = NULL;
  cl_event clevent = NULL, clevent_result = NULL;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
  str = (NULL != stream ? ACC_OPENCL_STREAM(stream) : c_dbcsr_acc_opencl_stream_default());
  assert(NULL != str && NULL != str->queue && NULL != event);
  clevent = *ACC_OPENCL_EVENT(event);
#  if defined(CL_VERSION_1_2)
  result = clEnqueueMarkerWithWaitList(str->queue, 0, NULL, &clevent_result);
#  else
  result = clEnqueueMarker(str->queue, &clevent_result);
#  endif
  if (NULL != clevent) {
    const int result_release = clReleaseEvent(clevent);
    if (EXIT_SUCCESS == result) result = result_release;
  }
  if (EXIT_SUCCESS == result) {
    assert(NULL != clevent_result);
    *(cl_event*)event = clevent_result;
  }
  else {
    if (NULL != clevent_result) ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseEvent(clevent_result));
    *(cl_event*)event = NULL;
  }
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_query(void* event, c_dbcsr_acc_bool_t* has_occurred) {
  cl_int status = CL_COMPLETE;
  int result;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
  assert(NULL != event && NULL != has_occurred);
  result = clGetEventInfo(*ACC_OPENCL_EVENT(event), CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, NULL);
  if (EXIT_SUCCESS == result && 0 <= status) *has_occurred = (CL_COMPLETE == status ? 1 : 0);
  else { /* error state */
    result = EXIT_SUCCESS; /* soft-error */
    *has_occurred = 1;
  }
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_synchronize(void* event) { /* waits on the host-side */
  int result = EXIT_SUCCESS;
  cl_event clevent;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
  assert(NULL != event);
  clevent = *ACC_OPENCL_EVENT(event);
  if (NULL != clevent) {
    if (0 == (32 & c_dbcsr_acc_opencl_config.wa)) result = clWaitForEvents(1, &clevent);
    else {
      cl_command_queue queue = NULL;
      result = clGetEventInfo(clevent, CL_EVENT_COMMAND_QUEUE, sizeof(cl_command_queue), &queue, NULL);
      if (EXIT_SUCCESS == result) result = clFinish(queue);
    }
  }
  else if (3 <= c_dbcsr_acc_opencl_config.verbosity || 0 > c_dbcsr_acc_opencl_config.verbosity) {
    fprintf(stderr, "WARN ACC/OpenCL: c_dbcsr_acc_event_synchronize discovered an empty event.\n");
  }
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}

#  if defined(__cplusplus)
}
#  endif

#endif /*__OPENCL*/
