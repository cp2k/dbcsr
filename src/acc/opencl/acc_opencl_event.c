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
#if defined(_OPENMP)
# include <omp.h>
#endif

#if !defined(ACC_OPENCL_EVENT_BARRIER) && 0
# define ACC_OPENCL_EVENT_BARRIER
#endif


#if defined(__cplusplus)
extern "C" {
#endif

int c_dbcsr_acc_event_create(void** event_p)
{
  cl_int result = EXIT_SUCCESS;
  const cl_context context = c_dbcsr_acc_opencl_context();
  cl_event event;
#if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char *const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#endif
  assert(NULL != event_p && NULL != context);
  event  = clCreateUserEvent(context, &result);
  if (CL_SUCCESS == result) {
    cl_int status = CL_COMPLETE;
    assert(NULL != event);
    /* an empty event (unrecorded) has no work to wait for; hence it is
     * considered occurred and c_dbcsr_acc_event_synchronize must not block
     */
    result = clSetUserEventStatus(event, status);
    if (CL_SUCCESS == result) {
#if defined(ACC_OPENCL_EVENT_NOALLOC)
      assert(sizeof(void*) >= sizeof(cl_event));
      *event_p = (void*)event;
#else
      *event_p = malloc(sizeof(cl_event));
      if (NULL != *event_p) {
        *(cl_event*)*event_p = event;
      }
      else {
        ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseEvent(event));
        result = EXIT_FAILURE;
      }
#endif
    }
    else { /* error: setting initial event state */
      ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseEvent(event));
      *event_p = NULL;
    }
  }
  else *event_p = NULL; /* error: creating user-defined event */
#if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_destroy(void* event)
{
  int result = EXIT_SUCCESS;
#if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char *const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#endif
  if (NULL != event) {
    result = clReleaseEvent(*ACC_OPENCL_EVENT(event));
#if defined(ACC_OPENCL_EVENT_NOALLOC)
    assert(sizeof(void*) >= sizeof(cl_event));
#else
    free(event);
#endif
  }
#if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_record(void* event, void* stream)
{
  int result;
  cl_event clevent;
#if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char *const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#endif
  assert(NULL != event && NULL != stream);
  ACC_OPENCL_DEBUG_IF(EXIT_SUCCESS != c_dbcsr_acc_opencl_stream_is_thread_specific(
    ACC_OPENCL_OMP_TID(), stream))
  {
    ACC_OPENCL_DEBUG_FPRINTF(stderr, "WARNING ACC/OpenCL: "
      "c_dbcsr_acc_event_record called by foreign thread!\n");
  }
#if defined(ACC_OPENCL_EVENT_BARRIER) && defined(CL_VERSION_1_2)
  result = clEnqueueBarrierWithWaitList(*ACC_OPENCL_STREAM(stream), 0, NULL, &clevent);
#elif defined(CL_VERSION_1_2)
  result = clEnqueueMarkerWithWaitList(*ACC_OPENCL_STREAM(stream), 0, NULL, &clevent);
#else
  result = clEnqueueMarker(*ACC_OPENCL_STREAM(stream), &clevent);
#endif
  if (CL_SUCCESS == result) {
#if defined(ACC_OPENCL_EVENT_NOALLOC)
    assert(!"ACC_OPENCL_EVENT_NOALLOC not supported");
    result = EXIT_FAILURE;
#else
    *(cl_event*)event = clevent;
#endif
  }
#if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_query(void* event, c_dbcsr_acc_bool_t* has_occurred)
{
  cl_int status = CL_COMPLETE, result = clGetEventInfo(
    NULL != event ? *ACC_OPENCL_EVENT(event) : NULL,
    CL_EVENT_COMMAND_EXECUTION_STATUS,
    sizeof(cl_int), &status, NULL);
#if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char *const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#endif
  assert(NULL != has_occurred);
  if (0 <= status) {
    *has_occurred = ((CL_COMPLETE == status || CL_SUCCESS != result) ? 1 : 0);
    if (0 == (8 & c_dbcsr_acc_opencl_config.flush) && 0 == *has_occurred) {
      result = c_dbcsr_acc_opencl_device_synchronize(ACC_OPENCL_OMP_TID());
    }
  }
  else { /* error state */
    if (CL_SUCCESS == result) result = EXIT_FAILURE;
    *has_occurred = 1;
  }
#if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_synchronize(void* event)
{ /* waits on the host-side */
  int result;
#if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char *const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#endif
  result = clWaitForEvents(1, ACC_OPENCL_EVENT(event));
#if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#endif
  ACC_OPENCL_RETURN(result);
}

#if defined(__cplusplus)
}
#endif

#endif /*__OPENCL*/
