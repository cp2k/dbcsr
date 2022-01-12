/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#ifndef ACC_OPENCL_H
#define ACC_OPENCL_H

#if !defined(CL_TARGET_OPENCL_VERSION)
# define CL_TARGET_OPENCL_VERSION 220
#endif

#if defined(__OPENCL)
# if defined(__APPLE__)
#   include <OpenCL/cl.h>
# else
#   include <CL/cl.h>
# endif
#else
# error Definition of __OPENCL preprocessor symbol is missing!
#endif

#if !defined(ACC_OPENCL_NOEXT)
# if defined(__APPLE__)
#   include <OpenCL/cl_ext.h>
# else
#   include <CL/cl_ext.h>
# endif
#endif

#if defined(__LIBXSMM)
# include <libxsmm.h>
#else
/* OpenCL backend depends on LIBXSMM */
# include <libxsmm_source.h>
# define __LIBXSMM
#endif

#include "../acc.h"
#if !defined(NDEBUG)
# include <assert.h>
#endif
#include <stdio.h>

#if !defined(ACC_OPENCL_CACHELINE_NBYTES)
# define ACC_OPENCL_CACHELINE_NBYTES LIBXSMM_CACHELINE
#endif
#if !defined(ACC_OPENCL_MAXALIGN_NBYTES)
# define ACC_OPENCL_MAXALIGN_NBYTES (2 << 20/*2MB*/)
#endif
#if !defined(ACC_OPENCL_BUFFERSIZE)
# define ACC_OPENCL_BUFFERSIZE (8 << 10/*8KB*/)
#endif
#if !defined(ACC_OPENCL_MAXSTRLEN)
# define ACC_OPENCL_MAXSTRLEN 48
#endif
#if !defined(ACC_OPENCL_DEVICES_MAXCOUNT)
# define ACC_OPENCL_DEVICES_MAXCOUNT 256
#endif
#if !defined(ACC_OPENCL_STREAMS_MAXCOUNT)
# define ACC_OPENCL_STREAMS_MAXCOUNT 128
#endif
#if !defined(ACC_OPENCL_OVERMALLOC)
# if defined(__DBCSR_ACC) || 1
#   define ACC_OPENCL_OVERMALLOC 0
# else
#   define ACC_OPENCL_OVERMALLOC 8192
# endif
#endif

#if !defined(ACC_OPENCL_DEBUG) && (defined(_DEBUG) || 0)
# define ACC_OPENCL_DEBUG
#endif
#if !defined(ACC_OPENCL_STREAM_PRIORITIES) && 1
# if defined(CL_QUEUE_PRIORITY_KHR)
#   define ACC_OPENCL_STREAM_PRIORITIES
# endif
#endif
#if !defined(ACC_OPENCL_MALLOC_LIBXSMM) && 0
# define ACC_OPENCL_MALLOC_LIBXSMM
#elif !defined(ACC_OPENCL_SVM) && 0
# if defined(CL_VERSION_2_0)
#   define ACC_OPENCL_SVM
# endif
#endif
#if !defined(ACC_OPENCL_DEVMATCH) && 0
# define ACC_OPENCL_DEVMATCH
#endif

/* can depend on OpenCL implementation (unlikely) */
#if !defined(ACC_OPENCL_MEM_NOALLOC) && 1
# define ACC_OPENCL_MEM_NOALLOC
# define ACC_OPENCL_MEM(A) ((cl_mem*)&(A))
#else
# define ACC_OPENCL_MEM(A) ((cl_mem*)(A))
#endif
/* can depend on OpenCL implementation (unlikely) */
#if !defined(ACC_OPENCL_STREAM_NOALLOC) && \
    !defined(ACC_OPENCL_STREAM_PRIORITIES) && 1
# define ACC_OPENCL_STREAM_NOALLOC
# define ACC_OPENCL_STREAM(A) ((cl_command_queue*)&(A))
#else
# define ACC_OPENCL_STREAM(A) ((cl_command_queue*)(A))
#endif
/* incompatible with c_dbcsr_acc_event_record */
#if !defined(ACC_OPENCL_EVENT_NOALLOC) && 0
# define ACC_OPENCL_EVENT_NOALLOC
# define ACC_OPENCL_EVENT(A) ((cl_event*)&(A))
#else
# define ACC_OPENCL_EVENT(A) ((cl_event*)(A))
#endif

#if defined(CL_VERSION_2_0)
# define ACC_OPENCL_COMMAND_QUEUE_PROPERTIES cl_queue_properties
#else
# define ACC_OPENCL_COMMAND_QUEUE_PROPERTIES cl_int
#endif

#if defined(ACC_OPENCL_DEBUG)
# define ACC_OPENCL_DEBUG_FPRINTF(STREAM, ...) fprintf(STREAM, __VA_ARGS__)
# define ACC_OPENCL_DEBUG_IF(CONDITION) if (CONDITION)
# define ACC_OPENCL_DEBUG_ELSE else
#else
# define ACC_OPENCL_DEBUG_FPRINTF(STREAM, ...)
# define ACC_OPENCL_DEBUG_IF(CONDITION)
# define ACC_OPENCL_DEBUG_ELSE
#endif

#if !defined(NDEBUG) || defined(ACC_OPENCL_DEBUG)
# define ACC_OPENCL_EXPECT(EXPECTED, EXPR) assert((EXPECTED) == (EXPR))
# define ACC_OPENCL_ERROR(MSG, RESULT) do { \
    assert(CL_SUCCESS == EXIT_SUCCESS); \
    if (-1001 != (RESULT)) { \
      fprintf(stderr, "ERROR ACC/OpenCL: " MSG); \
      if (EXIT_FAILURE != (RESULT)) { \
        fprintf(stderr, " (code=%i)", RESULT); \
      } \
      fprintf(stderr, ".\n"); \
      assert(CL_SUCCESS != (RESULT)); \
    } \
    else { \
      fprintf(stderr, "ERROR ACC/OpenCL: incomplete installation (" MSG ").\n"); \
    } \
    assert(!MSG); \
  } while (0)
# define ACC_OPENCL_RETURN_CAUSE(RESULT, CAUSE) do { \
    const int acc_opencl_return_cause_result_ = (RESULT); \
    if (EXIT_SUCCESS != acc_opencl_return_cause_result_) { \
      fprintf(stderr, "ERROR ACC/OpenCL: failed for %s!\n", \
        (NULL != (CAUSE) && '\0' != *(const char*)(CAUSE)) \
          ? ((const char*)CAUSE) : (LIBXSMM_FUNCNAME)); \
      assert(!"SUCCESS"); \
    } \
    return acc_opencl_return_cause_result_; \
  } while (0)
#else
# define ACC_OPENCL_EXPECT(EXPECTED, EXPR) (EXPR)
# define ACC_OPENCL_ERROR(MSG, RESULT)
# define ACC_OPENCL_RETURN_CAUSE(RESULT, CAUSE) LIBXSMM_UNUSED(CAUSE); return RESULT
#endif
#define ACC_OPENCL_RETURN(RESULT) ACC_OPENCL_RETURN_CAUSE(RESULT, NULL)

#define ACC_OPENCL_CHECK(EXPR, MSG, RESULT) do { \
  if (EXIT_SUCCESS == (RESULT)) { \
    (RESULT) = (EXPR); assert((MSG) && *(MSG)); \
    if (CL_SUCCESS != (RESULT)) { \
      ACC_OPENCL_ERROR(MSG, RESULT); \
    } \
  } \
} while (0)

#if defined(__cplusplus)
extern "C" {
#endif

/** Settings updated during c_dbcsr_acc_set_active_device. */
typedef struct c_dbcsr_acc_opencl_devinfo_t {
  /** Function used to record an event. */
  int (*record_event)(void* /*event*/, void* /*stream*/);
#if defined(ACC_OPENCL_SVM)
  /** Runtime SVM support (needs ACC_OPENCL_SVM at compile-time). */
  cl_bool svm_interop;
#endif
  /** Intel device ID (zero if non-Intel). */
  cl_int intel_id;
  /** Whether host memory is unified or not. */
  cl_bool unified;
  /** Asynchronous memory operations (may crash for some OpenCL implementations). */
  cl_bool async;
  /** Flush level. */
  cl_int flush;
} c_dbcsr_acc_opencl_devinfo_t;

/**
 * Settings discovered/setup during c_dbcsr_acc_init (independent of the device)
 * and settings updated during c_dbcsr_acc_set_active_device (devinfo).
 */
typedef struct c_dbcsr_acc_opencl_config_t {
  /** Table of ordered viable/discovered devices (matching criterion). */
  cl_device_id devices[ACC_OPENCL_DEVICES_MAXCOUNT];
  /** Settings updated during c_dbcsr_acc_set_active_device. */
  c_dbcsr_acc_opencl_devinfo_t devinfo;
  /** Kernel-parameters are matched against device's UID */
  cl_bool devmatch;
  /** Table of activated device contexts (thread-specific). */
  cl_context* contexts;
  /** All created streams partitioned by thread-ID (thread-local slots). */
  cl_command_queue* streams;
  /** Counts number of streams created (thread-local). */
  int* stream_stats;
  /** Verbosity level (output on stderr). */
  cl_int verbosity;
  /** Non-zero if library is initialized; negative in case of no device. */
  cl_int ndevices;
  /** Maximum number of threads (omp_get_max_threads). */
  cl_int nthreads;
  /** Dump level. */
  cl_int dump;
} c_dbcsr_acc_opencl_config_t;

/** Global configuration setup in c_dbcsr_acc_init. */
extern c_dbcsr_acc_opencl_config_t c_dbcsr_acc_opencl_config;

/** Contexts implement 1:1 relation with device. */
cl_context c_dbcsr_acc_opencl_context(void);
/** Share context for given device (start searching at optional thread_id), or return NULL). */
cl_context c_dbcsr_acc_opencl_device_context(cl_device_id device, const int* thread_id);

/** Information about host-memory pointer (c_dbcsr_acc_host_mem_allocate). */
typedef struct c_dbcsr_acc_opencl_info_hostptr_t {
  cl_mem buffer;
  void* mapped;
} c_dbcsr_acc_opencl_info_hostptr_t;
c_dbcsr_acc_opencl_info_hostptr_t* c_dbcsr_acc_opencl_info_hostptr(void* memory);

/** Information about streams (c_dbcsr_acc_stream_create). */
typedef struct c_dbcsr_acc_opencl_info_stream_t {
  void* pointer;
  int priority;
} c_dbcsr_acc_opencl_info_stream_t;
c_dbcsr_acc_opencl_info_stream_t* c_dbcsr_acc_opencl_info_stream(void* stream);

/** Get host-pointer associated with device-memory (c_dbcsr_acc_dev_mem_allocate). */
void* c_dbcsr_acc_opencl_get_hostptr(cl_mem memory);
/** Amount of device memory; local memory is only non-zero if separate from global. */
int c_dbcsr_acc_opencl_info_devmem(cl_device_id device,
  size_t* mem_free, size_t* mem_total, size_t* mem_local,
  int* mem_unified);
/** Get device associated with thread-ID. */
int c_dbcsr_acc_opencl_device(int thread_id, cl_device_id* device);
/** Get device-ID for given device, and optionally global device-ID. */
int c_dbcsr_acc_opencl_device_id(cl_device_id device, int* device_id, int* global_id);
/** Confirm the vendor of the given device. */
int c_dbcsr_acc_opencl_device_vendor(cl_device_id device, const char vendor[]);
/** Confirm that match is matching the name of the given device. */
int c_dbcsr_acc_opencl_device_name(cl_device_id device, const char match[]);
/** Capture or calculate UID based on the device-name. */
int c_dbcsr_acc_opencl_devuid(const char devname[], int* uid);
/** Capture or calculate UID based on the device-ID. */
int c_dbcsr_acc_opencl_device_uid(cl_device_id device, int* uid);
/** Return the OpenCL support level for the given device. */
int c_dbcsr_acc_opencl_device_level(cl_device_id device,
  int* level_major, int* level_minor, char cl_std[16], cl_device_type* type);
/** Check if given device supports the extensions. */
int c_dbcsr_acc_opencl_device_ext(cl_device_id device,
  const char *const extnames[], int num_exts);
/** Create context for given thread-ID and device. */
int c_dbcsr_acc_opencl_create_context(int thread_id, cl_device_id device_id);
/** Internal variant of c_dbcsr_acc_set_active_device. */
int c_dbcsr_acc_opencl_set_active_device(int thread_id, int device_id);
/** Get preferred multiple and max. size of workgroup (kernel- or device-specific). */
int c_dbcsr_acc_opencl_wgsize(cl_device_id device, cl_kernel kernel,
  int* max_value, int* preferred_multiple);
/**
 * Build kernel from source with given kernel_name, build_params and build_options.
 * The build_params are meant to instantiate the kernel (-D) whereas build_options
 * are are meant to be compiler-flags.
 */
int c_dbcsr_acc_opencl_kernel(const char source[], const char kernel_name[],
  const char build_params[], const char build_options[],
  const char try_build_options[], int* try_ok,
  const char *const extnames[], int num_exts,
  cl_kernel* kernel);
/** Create command queue (stream). */
int c_dbcsr_acc_opencl_stream_create(int thread_id, cl_command_queue* stream_p,
  const char name[], const ACC_OPENCL_COMMAND_QUEUE_PROPERTIES* properties);
/** Enqueue barrier (c_dbcsr_acc_opencl_config.devinfo.record_event). */
int c_dbcsr_acc_opencl_enqueue_barrier(void* event, void* stream);
/** Enqueue marker (c_dbcsr_acc_opencl_config.devinfo.record_event). */
int c_dbcsr_acc_opencl_enqueue_marker(void* event, void* stream);
/** Per-thread variant of c_dbcsr_acc_device_synchronize. */
int c_dbcsr_acc_opencl_device_synchronize(int thread_id);

#if defined(__cplusplus)
}
#endif

#endif /*ACC_OPENCL_H*/
