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

#include "../acc.h"
#if !defined(NDEBUG)
# include <assert.h>
#endif
#include <stdio.h>

#if !defined(ACC_OPENCL_CACHELINE_NBYTES)
# define ACC_OPENCL_CACHELINE_NBYTES 64
#endif
#if !defined(ACC_OPENCL_MAXALIGN_NBYTES)
# define ACC_OPENCL_MAXALIGN_NBYTES (2 << 20/*2MB*/)
#endif
#if !defined(ACC_OPENCL_MAXLINELEN)
# define ACC_OPENCL_MAXLINELEN 128
#endif
#if !defined(ACC_OPENCL_BUFFER_MAXSIZE)
# define ACC_OPENCL_BUFFER_MAXSIZE (64 * ACC_OPENCL_MAXLINELEN)
#endif
#if !defined(ACC_OPENCL_DEVICES_MAXCOUNT)
# define ACC_OPENCL_DEVICES_MAXCOUNT 32
#endif
#if defined(_WIN32)
# define ACC_OPENCL_PATHSEP "\\"
#else
# define ACC_OPENCL_PATHSEP "/"
#endif
#if !defined(ACC_OPENCL_SRCEXT)
# define ACC_OPENCL_SRCEXT "cl"
#endif

/* can depend on OpenCL implementation */
#if !defined(ACC_OPENCL_MEM_NOALLOC) && 1
# define ACC_OPENCL_MEM_NOALLOC
# define ACC_OPENCL_MEM(A) ((cl_mem*)&(A))
#else
# define ACC_OPENCL_MEM(A) ((cl_mem*)(A))
#endif
#if !defined(ACC_OPENCL_STREAM_NOALLOC) && 1
# define ACC_OPENCL_STREAM_NOALLOC
# define ACC_OPENCL_STREAM(A) ((cl_command_queue*)&(A))
#else
# define ACC_OPENCL_STREAM(A) ((cl_command_queue*)(A))
#endif
#if !defined(ACC_OPENCL_EVENT_NOALLOC) && 0
/* incompatible with acc_event_record */
# define ACC_OPENCL_EVENT_NOALLOC
# define ACC_OPENCL_EVENT(A) ((cl_event*)&(A))
#else
# define ACC_OPENCL_EVENT(A) ((cl_event*)(A))
#endif

#if !defined(ACC_OPENCL_THREADLOCAL_CONTEXT) && 1
# define ACC_OPENCL_THREADLOCAL_CONTEXT
#endif
#if !defined(ACC_OPENCL_STREAM_PRIORITIES) && 1
# define ACC_OPENCL_STREAM_PRIORITIES
#endif
#if !defined(ACC_OPENCL_STREAM_SYNCFLUSH) && 0
# define ACC_OPENCL_STREAM_FINISH
#endif
#if !defined(ACC_OPENCL_STREAM_OOOEXEC) && 0
# define ACC_OPENCL_STREAM_OOOEXEC
#endif
#if !defined(ACC_OPENCL_EVENT_BARRIER) && 0
# define ACC_OPENCL_EVENT_BARRIER
#endif
#if !defined(ACC_OPENCL_MEM_ASYNC) && 1
# define ACC_OPENCL_MEM_ASYNC
#endif
#if !defined(ACC_OPENCL_VERBOSE) && 0
# define ACC_OPENCL_VERBOSE
#endif
#if !defined(ACC_OPENCL_SVM) && 1
# if defined(CL_VERSION_2_0)
#   define ACC_OPENCL_SVM
# endif
#endif

#if defined(CL_VERSION_2_0)
# define ACC_OPENCL_COMMAND_QUEUE_PROPERTIES cl_queue_properties
#else
# define ACC_OPENCL_COMMAND_QUEUE_PROPERTIES cl_int
#endif

#define ACC_OPENCL_UP2(N, NPOT) ((((uint64_t)N) + ((NPOT) - 1)) & ~((NPOT) - 1))
#define ACC_OPENCL_UNUSED(VAR) (void)(VAR)

#if defined(__cplusplus)
# if defined(__GNUC__) || defined(_CRAYC)
#   define ACC_OPENCL_FUNCNAME __PRETTY_FUNCTION__
# elif defined(_MSC_VER)
#   define ACC_OPENCL_FUNCNAME __FUNCDNAME__
# else
#   define ACC_OPENCL_FUNCNAME __FUNCNAME__
# endif
#else
# if defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__) /*C99*/
#   define ACC_OPENCL_FUNCNAME __func__
# elif defined(_MSC_VER)
#   define ACC_OPENCL_FUNCNAME __FUNCDNAME__/*__FUNCTION__*/
# elif defined(__GNUC__) && !defined(__STRICT_ANSI__)
#   define ACC_OPENCL_FUNCNAME __PRETTY_FUNCTION__
# else
#   define ACC_OPENCL_FUNCNAME ""
# endif
#endif

#if defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__ || defined(__GNUC__))
# define ACC_OPENCL_SNPRINTF(S, N, ...) snprintf(S, N, __VA_ARGS__)
#else
# define ACC_OPENCL_SNPRINTF(S, N, ...) sprintf((S) + /*unused*/(N) * 0, __VA_ARGS__)
#endif

#if defined(_DEBUG)
# define ACC_OPENCL_DEBUG_PRINTF(A, ...) printf(A, __VA_ARGS__)
#else
# define ACC_OPENCL_DEBUG_PRINTF(A, ...)
#endif

#if defined(NDEBUG)
# define ACC_OPENCL_EXPECT(EXPECTED, EXPR) (EXPR)
# define ACC_OPENCL_ERROR(MSG, RESULT) (RESULT) = EXIT_FAILURE
# define ACC_OPENCL_RETURN_CAUSE(RESULT, CAUSE) ACC_OPENCL_UNUSED(CAUSE); return RESULT
#else
# define ACC_OPENCL_EXPECT(EXPECTED, EXPR) assert((EXPECTED) == (EXPR))
# define ACC_OPENCL_ERROR(MSG, RESULT) do { \
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
    (RESULT) = EXIT_FAILURE; \
  } while (0)
# define ACC_OPENCL_RETURN_CAUSE(RESULT, CAUSE) do { \
    const int acc_opencl_return_cause_result_ = (RESULT); \
    if (EXIT_SUCCESS != acc_opencl_return_cause_result_) { \
      fprintf(stderr, "ERROR ACC/OpenCL: failed for %s!\n", \
        (NULL != (CAUSE) && '\0' != *(const char*)(CAUSE)) \
          ? ((const char*)CAUSE) \
          : (ACC_OPENCL_FUNCNAME)); \
      assert(!"SUCCESS"); \
    } \
    return acc_opencl_return_cause_result_; \
  } while (0)
#endif
#define ACC_OPENCL_RETURN(RESULT) ACC_OPENCL_RETURN_CAUSE(RESULT, NULL)

#define ACC_OPENCL_CHECK(EXPR, MSG, RESULT) do { \
  if (EXIT_SUCCESS == (RESULT)) { \
    (RESULT) = (EXPR); assert((MSG) && *(MSG)); \
    if (CL_SUCCESS == (RESULT)) { \
      (RESULT) = EXIT_SUCCESS; \
    } \
    else { \
      ACC_OPENCL_ERROR(MSG, RESULT); \
    } \
  } \
} while (0)

#if defined(__cplusplus)
extern "C" {
#endif

/** Settings depending on OpenCL vendor or standard level (discovered/setup in acc_init). */
typedef struct acc_opencl_options_t {
  /** Asynchronous memory operations may crash for some OpenCL implementations. */
  cl_bool async_memops;
  cl_bool svm_interop;
} acc_opencl_options_t;

extern acc_opencl_options_t acc_opencl_options;

/* non-zero if library is initialized, zero devices is signaled by nagative value */
extern int acc_opencl_ndevices;
/* allow a context per each OpenMP thread */
extern cl_context acc_opencl_context;
#if defined(_OPENMP) && defined(ACC_OPENCL_THREADLOCAL_CONTEXT)
# pragma omp threadprivate(acc_opencl_context)
#endif

typedef struct acc_opencl_info_hostptr_t {
  cl_mem buffer;
  void* mapped;
} acc_opencl_info_hostptr_t;

/** Information about host-memory pointer (acc_host_mem_allocate). */
acc_opencl_info_hostptr_t* acc_opencl_info_hostptr(void* memory);
/** Get host-pointer associated with device-memory (acc_dev_mem_allocate). */
void* acc_opencl_get_hostptr(cl_mem memory);
/** Information about amount of device memory. */
int acc_opencl_info_devmem(cl_device_id device,
  size_t* mem_free, size_t* mem_total);

/** Return the pointer to the 1st match of "b" in "a", or NULL (no match). */
const char* acc_opencl_stristr(const char* a, const char* b);
/** Get active device (can be thread/queue-specific). */
int acc_opencl_device(void* stream, cl_device_id* device);
/** Confirm the vendor of the given device. */
int acc_opencl_device_vendor(cl_device_id device, const char* vendor);
/** Return the OpenCL support level for the given device. */
int acc_opencl_device_level(cl_device_id device,
  int* level_major, int* level_minor);
/** Check if given device supports the extensions. */
int acc_opencl_device_ext(cl_device_id device,
  const char *const extnames[], int num_exts);
/** Internal flavor of acc_set_active_device; yields cl_device_id. */
int acc_opencl_set_active_device(int device_id, cl_device_id* device);

/** Get directory path to load source files from. */
const char* acc_opencl_source_path(const char* fileext);
/** Opens filename (read-only) in source path (if not NULL) or dirpath otherwise. */
FILE* acc_opencl_source_open(const char* filename,
  const char *const dirpaths[], int ndirpaths);
/**
 * Reads source file or lines[0] (if source is NULL), and builds an array of strings
 * with line-wise content (lines). Returns the number of processed lines, and when
 * non-zero, lines[0] shall be released by the caller (free). Optionally applies
 * extensions (if not NULL).
 */
int acc_opencl_source(FILE* source, char* lines[], const char* extensions, int max_nlines, int cleanup);
/** Get preferred multiple of the size of the workgroup (kernel-specific). */
int acc_opencl_wgsize(cl_kernel kernel, int* preferred_multiple, int* max_value);
/** Build kernel function with given name from source using given build_options. */
int acc_opencl_kernel(const char *const source[], int nlines, const char* build_options,
  const char* kernel_name, cl_kernel* kernel);
/** Create command queue (stream). */
int acc_opencl_stream_create(cl_command_queue* stream_p, const char* name,
  const ACC_OPENCL_COMMAND_QUEUE_PROPERTIES* properties);

#if defined(__cplusplus)
}
#endif

#endif /*ACC_OPENCL_H*/
