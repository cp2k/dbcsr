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


#if defined(__cplusplus)
extern "C" {
#endif

int c_dbcsr_acc_event_create(void** event_p)
{
  cl_int result = EXIT_SUCCESS;
  const cl_event event = clCreateUserEvent(c_dbcsr_acc_opencl_context, &result);
  assert(NULL != event_p);
  if (NULL != event) {
    cl_int status = CL_COMPLETE;
    assert(CL_SUCCESS == result);
    /* an empty event (unrecorded) has no work to wait for; hence it is
     * considered occurred and c_dbcsr_acc_event_synchronize must not block
     */
    if (CL_SUCCESS == clSetUserEventStatus(event, status)) {
#if defined(ACC_OPENCL_EVENT_NOALLOC)
      assert(sizeof(void*) >= sizeof(cl_event));
      *event_p = (void*)event;
#else
      *event_p = malloc(sizeof(cl_event));
      if (NULL != *event_p) {
        *(cl_event*)*event_p = event;
        result = EXIT_SUCCESS;
      }
      else {
        clReleaseEvent(event);
        result = EXIT_FAILURE;
      }
#endif
    }
    else {
      ACC_OPENCL_ERROR("set initial event state", result);
      clReleaseEvent(event);
      *event_p = NULL;
    }
  }
  else {
    assert(CL_SUCCESS != result);
    ACC_OPENCL_ERROR("create user-defined event", result);
    *event_p = NULL;
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_destroy(void* event)
{
  int result = EXIT_SUCCESS;
  if (NULL != event) {
    ACC_OPENCL_CHECK(clReleaseEvent(*ACC_OPENCL_EVENT(event)),
      "release user-defined event", result);
#if defined(ACC_OPENCL_EVENT_NOALLOC)
    assert(sizeof(void*) >= sizeof(cl_event));
#else
    free(event);
#endif
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_enqueue_barrier(void* event, void* stream)
{
  int result = EXIT_SUCCESS;
  assert(NULL != event && NULL != stream);
#if defined(CL_VERSION_1_2)
  ACC_OPENCL_CHECK(clEnqueueBarrierWithWaitList(*ACC_OPENCL_STREAM(stream),
    0, NULL, ACC_OPENCL_EVENT(event)), "record event", result);
#else
  result = EXIT_FAILURE;
#endif
  return result;
}


int c_dbcsr_acc_opencl_enqueue_marker(void* event, void* stream)
{
  int result = EXIT_SUCCESS;
  assert(NULL != event && NULL != stream);
#if defined(CL_VERSION_1_2)
  ACC_OPENCL_CHECK(clEnqueueMarkerWithWaitList(*ACC_OPENCL_STREAM(stream),
    0, NULL, ACC_OPENCL_EVENT(event)), "record event", result);
#else
  ACC_OPENCL_CHECK(clEnqueueMarker(*ACC_OPENCL_STREAM(stream),
    ACC_OPENCL_EVENT(event)), "record event", result);
#endif
  return result;
}


int c_dbcsr_acc_event_record(void* event, void* stream)
{
  int result = EXIT_SUCCESS;
  assert(NULL != c_dbcsr_acc_opencl_config.record_event);
  result = c_dbcsr_acc_opencl_config.record_event(event, stream);
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_query(void* event, c_dbcsr_acc_bool_t* has_occurred)
{
  int result = EXIT_SUCCESS;
  cl_int status = CL_COMPLETE;
  if (NULL != event) {
    ACC_OPENCL_CHECK(clGetEventInfo(*ACC_OPENCL_EVENT(event), CL_EVENT_COMMAND_EXECUTION_STATUS,
      sizeof(cl_int), &status, NULL), "retrieve event status", result);
  }
  assert(NULL != has_occurred);
  *has_occurred = (CL_COMPLETE == status || 0 > status);
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_event_synchronize(void* event)
{ /* Waits on the host-side. */
  int result = EXIT_SUCCESS;
  assert(NULL != event);
  ACC_OPENCL_CHECK(clWaitForEvents(1, ACC_OPENCL_EVENT(event)),
    "synchronize event", result);
  ACC_OPENCL_RETURN(result);
}

#if defined(__cplusplus)
}
#endif

#endif /*__OPENCL*/
