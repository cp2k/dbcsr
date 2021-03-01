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
#include <string.h>
#include <assert.h>

#if defined(CL_VERSION_2_0)
# define ACC_OPENCL_CREATE_COMMAND_QUEUE(CTX, DEV, PROPS, RESULT) \
    clCreateCommandQueueWithProperties(CTX, DEV, PROPS, RESULT)
#else
# define ACC_OPENCL_CREATE_COMMAND_QUEUE(CTX, DEV, PROPS, RESULT) \
    clCreateCommandQueue(CTX, DEV, (cl_command_queue_properties) \
      (NULL != (PROPS) ? ((PROPS)[1]) : 0), RESULT)
#endif

#if defined(CL_VERSION_1_2)
# define ACC_OPENCL_WAIT_EVENT(QUEUE, EVENT) clEnqueueMarkerWithWaitList(QUEUE, 1, EVENT, NULL)
#else
# define ACC_OPENCL_WAIT_EVENT(QUEUE, EVENT) clEnqueueWaitForEvents(QUEUE, 1, EVENT)
#endif


#if defined(__cplusplus)
extern "C" {
#endif

int c_dbcsr_acc_opencl_stream_create(cl_command_queue* stream_p, const char* name,
  const ACC_OPENCL_COMMAND_QUEUE_PROPERTIES* properties)
{
  cl_int result = EXIT_SUCCESS;
  assert(NULL != stream_p);
  if (NULL != c_dbcsr_acc_opencl_context) {
    cl_device_id device_id = NULL;
    result = c_dbcsr_acc_opencl_device(NULL/*stream*/, &device_id);
    if (EXIT_SUCCESS == result) {
      *stream_p = ACC_OPENCL_CREATE_COMMAND_QUEUE(c_dbcsr_acc_opencl_context, device_id, properties, &result);
    }
    else {
      ACC_OPENCL_ERROR("create command queue", result);
    }
  }
  ACC_OPENCL_RETURN_CAUSE(result, name);
}


int c_dbcsr_acc_stream_create(void** stream_p, const char* name, int priority)
{
  cl_int result = EXIT_SUCCESS;
  if (NULL != c_dbcsr_acc_opencl_context) {
    ACC_OPENCL_COMMAND_QUEUE_PROPERTIES properties[8] = {
      CL_QUEUE_PROPERTIES, 0/*placeholder*/,
      0 /* terminator */
    };
    cl_command_queue queue = NULL;
#if !defined(ACC_OPENCL_STREAM_PRIORITIES) || !defined(CL_QUEUE_PRIORITY_KHR)
    ACC_OPENCL_UNUSED(priority);
#else
    if (0 <= priority) {
      properties[2] = CL_QUEUE_PRIORITY_KHR;
      properties[3] = ((  CL_QUEUE_PRIORITY_HIGH_KHR <= priority
                        && CL_QUEUE_PRIORITY_LOW_KHR >= priority)
        ? priority : CL_QUEUE_PRIORITY_MED_KHR);
      properties[4] = 0; /* terminator */
    }
#endif
    if (3 <= c_dbcsr_acc_opencl_options.verbosity || 0 > c_dbcsr_acc_opencl_options.verbosity) {
      properties[1] = CL_QUEUE_PROFILING_ENABLE;
    }
    result = c_dbcsr_acc_opencl_stream_create(&queue, name, properties);
    assert(NULL != stream_p);
    if (EXIT_SUCCESS == result) {
      assert(NULL != queue);
#if defined(ACC_OPENCL_STREAM_NOALLOC)
      assert(sizeof(void*) >= sizeof(cl_command_queue));
      *stream_p = (void*)queue;
#else
      *stream_p = malloc(sizeof(cl_command_queue));
      if (NULL != *stream_p) {
        *(cl_command_queue*)*stream_p = queue;
      }
      else {
        clReleaseCommandQueue(queue);
        result = EXIT_FAILURE;
      }
#endif
    }
    else {
      *stream_p = NULL;
    }
  }
  ACC_OPENCL_RETURN_CAUSE(result, name);
}


int c_dbcsr_acc_stream_destroy(void* stream)
{
  int result = EXIT_SUCCESS;
  if (NULL != stream) {
    ACC_OPENCL_CHECK(clReleaseCommandQueue(*ACC_OPENCL_STREAM(stream)),
      "release command queue", result);
#if defined(ACC_OPENCL_STREAM_NOALLOC)
    assert(sizeof(void*) >= sizeof(cl_command_queue));
#else
    free(stream);
#endif
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_stream_priority_range(int* least, int* greatest)
{
  int result = ((NULL != least || NULL != greatest) ? EXIT_SUCCESS : EXIT_FAILURE);
  if (NULL != c_dbcsr_acc_opencl_context) {
#if defined(ACC_OPENCL_STREAM_PRIORITIES) && defined(CL_QUEUE_PRIORITY_KHR)
    char buffer[ACC_OPENCL_BUFFERSIZE];
    cl_platform_id platform = NULL;
    cl_device_id active_id = NULL;
    assert(0 < c_dbcsr_acc_opencl_ndevices);
    if (EXIT_SUCCESS == result) result = c_dbcsr_acc_opencl_device(NULL/*stream*/, &active_id);
    ACC_OPENCL_CHECK(clGetDeviceInfo(active_id, CL_DEVICE_PLATFORM,
      sizeof(cl_platform_id), &platform, NULL),
      "retrieve platform associated with active device", result);
    ACC_OPENCL_CHECK(clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS,
      ACC_OPENCL_BUFFERSIZE, buffer, NULL),
      "retrieve platform extensions", result);
    if (EXIT_SUCCESS == result) {
      if (NULL != strstr(buffer, "cl_khr_priority_hints")) {
        if (NULL != least) *least = CL_QUEUE_PRIORITY_LOW_KHR;
        if (NULL != greatest) *greatest = CL_QUEUE_PRIORITY_HIGH_KHR;
      }
      else
#endif
      {
        if (NULL != least) *least = -1;
        if (NULL != greatest) *greatest = -1;
      }
#if defined(ACC_OPENCL_STREAM_PRIORITIES) && defined(CL_QUEUE_PRIORITY_KHR)
    }
#endif
  }
  else {
    if (NULL != least) *least = -1;
    if (NULL != greatest) *greatest = -1;
  }
  assert(least != greatest); /* no alias */
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_stream_sync(void* stream)
{ /* Blocks the host-thread. */
  int result = EXIT_SUCCESS;
  assert(NULL != stream);
  ACC_OPENCL_CHECK(clFinish(*ACC_OPENCL_STREAM(stream)),
    "synchronize stream", result);
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_stream_wait_event(void* stream, void* event)
{ /* Wait for an event (device-side). */
  int result = EXIT_SUCCESS;
  assert(NULL != stream && NULL != event);
  ACC_OPENCL_CHECK(ACC_OPENCL_WAIT_EVENT(*ACC_OPENCL_STREAM(stream), ACC_OPENCL_EVENT(event)),
    "wait for an event", result);
  ACC_OPENCL_RETURN(result);
}

#if defined(__cplusplus)
}
#endif

#endif /*__OPENCL*/
