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

#  if defined(CL_VERSION_1_2)
#    define ACC_OPENCL_WAIT_EVENT(QUEUE, EVENT) clEnqueueMarkerWithWaitList(QUEUE, 1, EVENT, NULL)
#  else
#    define ACC_OPENCL_WAIT_EVENT(QUEUE, EVENT) clEnqueueWaitForEvents(QUEUE, 1, EVENT)
#  endif

#  if !defined(ACC_OPENCL_EVENT_BARRIER) && 0
#    define ACC_OPENCL_EVENT_BARRIER
#  endif
#  if !defined(ACC_OPENCL_EVENT_CREATE) && 0
#    define ACC_OPENCL_EVENT_CREATE
#  endif


#  if defined(__cplusplus)
extern "C" {
#  endif

int c_dbcsr_acc_opencl_event_create(cl_event* event_p) {
  int result;
  assert(NULL != event_p);
  if (NULL != *event_p) result = EXIT_SUCCESS;
  else {
    *event_p = clCreateUserEvent(c_dbcsr_acc_opencl_context(NULL /*tid*/), &result);
  }
  if (CL_SUCCESS == result) {
    assert(NULL != *event_p);
    /* an empty event (unrecorded) has no work to wait for; hence it is
     * considered occurred and c_dbcsr_acc_event_synchronize must not block
     */
    result = clSetUserEventStatus(*event_p, CL_COMPLETE);
    if (CL_SUCCESS != result) { /* error: setting initial event state */
      ACC_OPENCL_EXPECT(CL_SUCCESS == clReleaseEvent(*event_p));
      *event_p = NULL;
    }
  }
  else {
    *event_p = NULL; /* error: creating user-defined event */
  }
  return result;
}


int c_dbcsr_acc_event_create(void** event_p) {
  int result = EXIT_SUCCESS;
  cl_event event = NULL;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(NULL != event_p);
#  if defined(ACC_OPENCL_EVENT_CREATE)
  result = c_dbcsr_acc_opencl_event_create(&event);
  assert(NULL != event || EXIT_SUCCESS != result);
  if (EXIT_SUCCESS == result)
#  endif
  {
    assert(NULL == c_dbcsr_acc_opencl_config.events || sizeof(void*) >= sizeof(cl_event));
    *event_p = (
#  if LIBXSMM_VERSION4(1, 17, 0, 0) < LIBXSMM_VERSION_NUMBER && defined(ACC_OPENCL_HANDLES_MAXCOUNT) && \
    (0 < ACC_OPENCL_HANDLES_MAXCOUNT)
      NULL != c_dbcsr_acc_opencl_config.events
        ? libxsmm_pmalloc(c_dbcsr_acc_opencl_config.events, &c_dbcsr_acc_opencl_config.nevents)
        :
#  endif
        malloc(sizeof(cl_event)));
    if (NULL != *event_p) {
      *(cl_event*)*event_p = event;
    }
    else {
      if (NULL != event) ACC_OPENCL_EXPECT(CL_SUCCESS == clReleaseEvent(event));
      result = EXIT_FAILURE;
    }
  }
#  if defined(ACC_OPENCL_EVENT_CREATE)
  else {
    *event_p = NULL; /* error: creating user-defined event */
  }
#  endif
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_destroy(void* event) {
  int result = EXIT_SUCCESS;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  if (NULL != event) {
    const cl_event clevent = *ACC_OPENCL_EVENT(event);
    if (NULL != clevent) result = clReleaseEvent(clevent);
#  if LIBXSMM_VERSION4(1, 17, 0, 0) < LIBXSMM_VERSION_NUMBER && defined(ACC_OPENCL_HANDLES_MAXCOUNT) && \
    (0 < ACC_OPENCL_HANDLES_MAXCOUNT)
    if (NULL != c_dbcsr_acc_opencl_config.events) {
      /**(cl_event*)event = NULL; assert(NULL == *ACC_OPENCL_EVENT(event));*/
      libxsmm_pfree(event, c_dbcsr_acc_opencl_config.events, &c_dbcsr_acc_opencl_config.nevents);
    }
    else
#  endif
    {
      free(event);
    }
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_stream_wait_event(void* stream, void* event) { /* wait for an event (device-side) */
  int result = EXIT_SUCCESS;
  cl_event clevent;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(NULL != stream && NULL != event);
  clevent = *ACC_OPENCL_EVENT(event);
#  if defined(ACC_OPENCL_EVENT_CREATE)
  assert(NULL != clevent);
#  else
  if (NULL != clevent)
#  endif
  {
    result = ACC_OPENCL_WAIT_EVENT(*ACC_OPENCL_STREAM(stream), &clevent);
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_record(void* event, void* stream) {
  int result;
  cl_event clevent = NULL;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(NULL != event && NULL != stream);
#  if defined(ACC_OPENCL_EVENT_BARRIER) && defined(CL_VERSION_1_2)
  result = clEnqueueBarrierWithWaitList(*ACC_OPENCL_STREAM(stream), 0, NULL, &clevent);
#  elif defined(CL_VERSION_1_2)
  result = clEnqueueMarkerWithWaitList(*ACC_OPENCL_STREAM(stream), 0, NULL, &clevent);
#  else
  result = clEnqueueMarker(*ACC_OPENCL_STREAM(stream), &clevent);
#  endif
  if (CL_SUCCESS == result) {
    assert(NULL != clevent);
    *(cl_event*)event = clevent;
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_query(void* event, c_dbcsr_acc_bool_t* has_occurred) {
  cl_int status = CL_COMPLETE;
  int result;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(NULL != event && NULL != has_occurred);
  result = clGetEventInfo(*ACC_OPENCL_EVENT(event), CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, NULL);
  if (CL_SUCCESS == result && 0 <= status) *has_occurred = (CL_COMPLETE == status ? 1 : 0);
  else { /* error state */
#  if defined(ACC_OPENCL_EVENT_CREATE)
    if (CL_SUCCESS == result) result = EXIT_FAILURE;
#  else
    result = EXIT_SUCCESS;
#  endif
    *has_occurred = 1;
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_synchronize(void* event) { /* waits on the host-side */
  int result;
  cl_event clevent;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(NULL != event);
  clevent = *ACC_OPENCL_EVENT(event);
#  if !defined(ACC_OPENCL_EVENT_CREATE)
  if (NULL == clevent) {
    result = EXIT_SUCCESS;
  }
  else
#  endif
  {
    result = clWaitForEvents(1, &clevent);
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
