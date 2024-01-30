/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/
#ifndef ACC_OPENCL_H
#define ACC_OPENCL_H

#if defined(__OFFLOAD_OPENCL) && !defined(__OPENCL)
#  define __OPENCL
#endif

#if defined(__OPENCL)
#  if !defined(CL_TARGET_OPENCL_VERSION)
#    define CL_TARGET_OPENCL_VERSION 220
#  endif
#  if defined(__APPLE__)
#    include <OpenCL/cl.h>
#  else
#    include <CL/cl.h>
#  endif
#else
#  error Definition of __OPENCL preprocessor symbol is missing!
#endif

#if !defined(ACC_OPENCL_NOEXT)
#  if defined(__APPLE__)
#    include <OpenCL/cl_ext.h>
#  else
#    include <CL/cl_ext.h>
#  endif
#endif

#if !defined(LIBXSMM_SYNC_NPAUSE)
#  define LIBXSMM_SYNC_NPAUSE 0
#endif

#if defined(__LIBXSMM) && !defined(LIBXSMM_DEFAULT_CONFIG)
#  include <libxsmm.h>
#  if !defined(LIBXSMM_TIMER_H)
#    include <utils/libxsmm_timer.h>
#  endif
#  if !defined(LIBXSMM_SYNC_H)
#    include <libxsmm_sync.h>
#  endif
#else
/* OpenCL backend depends on LIBXSMM */
#  include <libxsmm_source.h>
#  if !defined(__LIBXSMM)
#    define __LIBXSMM
#  endif
#endif

#if !defined(LIBXSMM_VERSION_NUMBER)
#  define LIBXSMM_VERSION_NUMBER \
    LIBXSMM_VERSION4(LIBXSMM_VERSION_MAJOR, LIBXSMM_VERSION_MINOR, LIBXSMM_VERSION_UPDATE, LIBXSMM_VERSION_PATCH)
#endif

#include "../acc.h"
#if !defined(NDEBUG)
#  include <assert.h>
#endif
#include <stdlib.h>
#include <stdio.h>

#if !defined(ACC_OPENCL_CACHELINE)
#  define ACC_OPENCL_CACHELINE LIBXSMM_CACHELINE
#endif
#if !defined(ACC_OPENCL_ATOMIC)
#  define ACC_OPENCL_ATOMIC LIBXSMM_ATOMIC_SEQ_CST
#endif
#if defined(LIBXSMM_ATOMIC_LOCKTYPE)
#  define ACC_OPENCL_ATOMIC_LOCKTYPE volatile LIBXSMM_ATOMIC_LOCKTYPE
#else
#  define ACC_OPENCL_ATOMIC_LOCKTYPE volatile int
#endif
#if !defined(ACC_OPENCL_MAXALIGN)
#  define ACC_OPENCL_MAXALIGN (2 << 20 /*2MB*/)
#endif
#if !defined(ACC_OPENCL_BUFFERSIZE)
#  define ACC_OPENCL_BUFFERSIZE (8 << 10 /*8KB*/)
#endif
#if !defined(ACC_OPENCL_MAXSTRLEN)
#  define ACC_OPENCL_MAXSTRLEN 48
#endif
#if !defined(ACC_OPENCL_MAXNDEVS)
#  define ACC_OPENCL_MAXNDEVS 64
#endif
/* Counted on a per-thread basis! */
#if !defined(ACC_OPENCL_MAXNITEMS)
#  define ACC_OPENCL_MAXNITEMS 1024
#endif
/* First char is CSV-separator by default (w/o spaces) */
#if !defined(ACC_OPENCL_DELIMS)
#  define ACC_OPENCL_DELIMS ",;"
#endif
#if !defined(ACC_OPENCL_LAZYINIT) && (defined(__DBCSR_ACC) || 1)
#  define ACC_OPENCL_LAZYINIT
#endif
#if !defined(ACC_OPENCL_ASYNC) && 1
#  define ACC_OPENCL_ASYNC getenv("ACC_OPENCL_ASYNC")
#endif
#if !defined(ACC_OPENCL_STREAM_PRIORITIES) && 0
#  if defined(CL_QUEUE_PRIORITY_KHR)
#    define ACC_OPENCL_STREAM_PRIORITIES
#  endif
#endif
/* Stream-argument (ACC-interface) can be NULL (synchronous) */
#if !defined(ACC_OPENCL_STREAM_NULL) && 1
#  define ACC_OPENCL_STREAM_NULL
#endif
/* Support arithmetic for device-pointers (DBM) */
#if !defined(ACC_OPENCL_MEM_DEVPTR) && 1
#  define ACC_OPENCL_MEM_DEVPTR
#endif
#if !defined(ACC_OPENCL_OMPLOCKS) && 1
#  define ACC_OPENCL_OMPLOCKS
#endif
/* Use DBCSR's profile for detailed timings */
#if !defined(ACC_OPENCL_PROFILE) && 0
#  define ACC_OPENCL_PROFILE
#endif

/* attaching c_dbcsr_acc_opencl_stream_t is needed */
#define ACC_OPENCL_STREAM(A) ((const c_dbcsr_acc_opencl_stream_t*)(A))
/* incompatible with c_dbcsr_acc_event_record */
#define ACC_OPENCL_EVENT(A) ((const cl_event*)(A))

#if defined(_OPENMP)
#  include <omp.h>
#  define ACC_OPENCL_OMP_TID() omp_get_thread_num()
#else
#  define ACC_OPENCL_OMP_TID() (/*main*/ 0)
#  undef ACC_OPENCL_OMPLOCKS
#endif

#define ACC_OPENCL_ATOMIC_ACQUIRE(LOCK) \
  do { \
    LIBXSMM_ATOMIC_ACQUIRE(LOCK, LIBXSMM_SYNC_NPAUSE, ACC_OPENCL_ATOMIC); \
  } while (0)
#define ACC_OPENCL_ATOMIC_RELEASE(LOCK) \
  do { \
    LIBXSMM_ATOMIC_RELEASE(LOCK, ACC_OPENCL_ATOMIC); \
  } while (0)

#if defined(ACC_OPENCL_OMPLOCKS)
#  define ACC_OPENCL_INIT(LOCK) omp_init_lock(LOCK)
#  define ACC_OPENCL_DESTROY(LOCK) omp_destroy_lock(LOCK)
#  define ACC_OPENCL_ACQUIRE(LOCK) omp_set_lock(LOCK)
#  define ACC_OPENCL_RELEASE(LOCK) omp_unset_lock(LOCK)
#  define ACC_OPENCL_LOCKTYPE omp_lock_t
#else
#  define ACC_OPENCL_INIT(LOCK) (*(LOCK) = 0)
#  define ACC_OPENCL_DESTROY(LOCK)
#  define ACC_OPENCL_ACQUIRE(LOCK) ACC_OPENCL_ATOMIC_ACQUIRE(LOCK)
#  define ACC_OPENCL_RELEASE(LOCK) ACC_OPENCL_ATOMIC_RELEASE(LOCK)
#  define ACC_OPENCL_LOCKTYPE ACC_OPENCL_ATOMIC_LOCKTYPE
#endif

#if defined(CL_VERSION_2_0)
#  define ACC_OPENCL_STREAM_PROPERTIES_TYPE cl_queue_properties
#  define ACC_OPENCL_CREATE_COMMAND_QUEUE(CTX, DEV, PROPS, RESULT) clCreateCommandQueueWithProperties(CTX, DEV, PROPS, RESULT)
#else
#  define ACC_OPENCL_STREAM_PROPERTIES_TYPE cl_int
#  define ACC_OPENCL_CREATE_COMMAND_QUEUE(CTX, DEV, PROPS, RESULT) \
    clCreateCommandQueue(CTX, DEV, (cl_command_queue_properties)(NULL != (PROPS) ? ((PROPS)[1]) : 0), RESULT)
#endif

#if LIBXSMM_VERSION4(1, 17, 0, 0) < LIBXSMM_VERSION_NUMBER
#  define ACC_OPENCL_EXPECT(EXPR) LIBXSMM_EXPECT(EXPR)
#  define LIBXSMM_STRISTR libxsmm_stristr
#else
#  define ACC_OPENCL_EXPECT(EXPR) \
    if (0 == (EXPR)) assert(0)
#  define LIBXSMM_STRISTR strstr
#endif

#if !defined(NDEBUG) && 1
#  define ACC_OPENCL_CHECK(EXPR, MSG, RESULT) \
    do { \
      if (EXIT_SUCCESS == (RESULT)) { \
        (RESULT) = (EXPR); \
        assert((MSG) && *(MSG)); \
        if (EXIT_SUCCESS != (RESULT)) { \
          assert(EXIT_SUCCESS == EXIT_SUCCESS); \
          if (-1001 != (RESULT)) { \
            fprintf(stderr, "ERROR ACC/OpenCL: " MSG); \
            if (EXIT_FAILURE != (RESULT)) { \
              fprintf(stderr, " (code=%i)", RESULT); \
            } \
            fprintf(stderr, ".\n"); \
            assert(EXIT_SUCCESS != (RESULT)); \
          } \
          else { \
            fprintf(stderr, "ERROR ACC/OpenCL: incomplete installation (" MSG ").\n"); \
          } \
          assert(!MSG); \
        } \
      } \
    } while (0)
#  define ACC_OPENCL_RETURN_CAUSE(RESULT, CAUSE) \
    do { \
      const int acc_opencl_return_cause_result_ = (RESULT); \
      if (EXIT_SUCCESS != acc_opencl_return_cause_result_) { \
        if (NULL != (CAUSE) && '\0' != *(const char*)(CAUSE)) { \
          fprintf(stderr, "ERROR ACC/OpenCL: failed for %s!\n", (const char*)CAUSE); \
          assert(!"SUCCESS"); \
        } \
        else if (NULL != (LIBXSMM_FUNCNAME) && '\0' != *(const char*)(LIBXSMM_FUNCNAME)) { \
          fprintf(stderr, "ERROR ACC/OpenCL: failed for %s!\n", (const char*)LIBXSMM_FUNCNAME); \
          assert(!"SUCCESS"); \
        } \
        else { \
          fprintf(stderr, "ERROR ACC/OpenCL: failure!\n"); \
          assert(!"SUCCESS"); \
        } \
      } \
      return acc_opencl_return_cause_result_; \
    } while (0)
#else
#  define ACC_OPENCL_CHECK(EXPR, MSG, RESULT) \
    do { \
      if (EXIT_SUCCESS == (RESULT)) { \
        (RESULT) = (EXPR); \
        assert((MSG) && *(MSG)); \
      } \
    } while (0)
#  define ACC_OPENCL_RETURN_CAUSE(RESULT, CAUSE) \
    LIBXSMM_UNUSED(CAUSE); \
    return RESULT
#endif
#define ACC_OPENCL_RETURN(RESULT) ACC_OPENCL_RETURN_CAUSE(RESULT, "")


#if defined(__cplusplus)
extern "C" {
#endif

/** Information about streams (c_dbcsr_acc_stream_create). */
typedef struct c_dbcsr_acc_opencl_stream_t {
  cl_command_queue queue;
  int tid;
#if defined(ACC_OPENCL_STREAM_PRIORITIES)
  int priority;
#endif
} c_dbcsr_acc_opencl_stream_t;

/** Settings updated during c_dbcsr_acc_set_active_device. */
typedef struct c_dbcsr_acc_opencl_device_t {
  /** Activated device context. */
  cl_context context;
  /** Stream for internal purpose. */
  c_dbcsr_acc_opencl_stream_t stream;
  /** OpenCL compiler flag (language standard). */
  char std_flag[16];
  /** OpenCL support-level of device. */
  cl_int std_level[2], std_clevel[2];
  /** Kind of device (GPU, CPU, or other). */
  cl_device_type type;
  /** Whether host memory is unified. */
  cl_int unified;
  /** Device-ID. */
  cl_uint uid;
  /** Main vendor? */
  cl_int intel, amd, nv;
  /* USM support functions */
  cl_int (*clSetKernelArgMemPointerINTEL)(cl_kernel, cl_uint, const void*);
  cl_int (*clEnqueueMemFillINTEL)(cl_command_queue, void*, const void*, size_t, size_t, cl_uint, const cl_event*, cl_event*);
  cl_int (*clEnqueueMemcpyINTEL)(cl_command_queue, cl_bool, void*, const void*, size_t, cl_uint, const cl_event*, cl_event*);
  void* (*clDeviceMemAllocINTEL)(cl_context, cl_device_id, const void/*cl_mem_properties_intel*/*, size_t, cl_uint, cl_int*);
  cl_int (*clMemFreeINTEL)(cl_context, void*);
} c_dbcsr_acc_opencl_device_t;

/** Information about host/device-memory pointer. */
typedef struct c_dbcsr_acc_opencl_info_memptr_t {
  cl_mem memory; /* first item! */
  void* memptr;
} c_dbcsr_acc_opencl_info_memptr_t;

/** Enumeration of timer kinds used for built-in execution-profile. */
typedef enum c_dbcsr_acc_opencl_timer_t {
  c_dbcsr_acc_opencl_timer_device,
  c_dbcsr_acc_opencl_timer_host
} c_dbcsr_acc_opencl_timer_t;

/** Enumeration of FP-atomic kinds. */
typedef enum c_dbcsr_acc_opencl_atomic_fp_t {
  c_dbcsr_acc_opencl_atomic_fp_no = 0,
  c_dbcsr_acc_opencl_atomic_fp_32 = 1,
  c_dbcsr_acc_opencl_atomic_fp_64 = 2
} c_dbcsr_acc_opencl_atomic_fp_t;

/**
 * Settings discovered/setup during c_dbcsr_acc_init (independent of the device)
 * and settings updated during c_dbcsr_acc_set_active_device (devinfo).
 */
typedef struct c_dbcsr_acc_opencl_config_t {
  /** Table of ordered viable/discovered devices (matching criterion). */
  cl_device_id devices[ACC_OPENCL_MAXNDEVS];
  /** Table of devices (thread-specific). */
  c_dbcsr_acc_opencl_device_t device;
  /** Locks used by domain. */
  ACC_OPENCL_LOCKTYPE *lock_main, *lock_stream, *lock_event, *lock_memory;
#if defined(ACC_OPENCL_MEM_DEVPTR)
  /** All memptrs and related storage/counter. */
  c_dbcsr_acc_opencl_info_memptr_t **memptrs, *memptr_data;
  size_t nmemptrs; /* counter */
#endif
  /** Handle-counter. */
  size_t nstreams, nevents;
  /** All streams and related storage. */
  c_dbcsr_acc_opencl_stream_t **streams, *stream_data;
  /** All events and related storage. */
  cl_event **events, *event_data;
  /** Kind of timer used for built-in execution-profile. */
  c_dbcsr_acc_opencl_timer_t timer; /* c_dbcsr_acc_opencl_device_t? */
  /** Kernel-parameters are matched against device's UID */
  cl_uint devmatch;
  /** Verbosity level (output on stderr). */
  cl_int verbosity;
  /** Non-zero if library is initialized (negative: no device). */
  cl_int ndevices;
  /** Maximum number of threads (omp_get_max_threads). */
  cl_int nthreads;
  /** How to apply/use stream priorities. */
  cl_int priority;
  /** Configuration and execution-hints. */
  cl_int xhints;
  /** Asynchronous memory operations. */
  cl_int async;
  /** Debug (output/symbols, etc.). */
  cl_int debug;
  /** Dump level. */
  cl_int dump;
} c_dbcsr_acc_opencl_config_t;

/** Global configuration setup in c_dbcsr_acc_init. */
extern c_dbcsr_acc_opencl_config_t c_dbcsr_acc_opencl_config;

/** Determines host-pointer registration for modification. */
c_dbcsr_acc_opencl_info_memptr_t* c_dbcsr_acc_opencl_info_hostptr(void* memory);
/** Determines device-pointer registration for modification (internal). */
c_dbcsr_acc_opencl_info_memptr_t* c_dbcsr_acc_opencl_info_devptr_modify(
  ACC_OPENCL_LOCKTYPE* lock, void* memory, size_t elsize, const size_t* amount, size_t* offset);
/** Determines device-pointer registration for information (lock-control). */
int c_dbcsr_acc_opencl_info_devptr_lock(c_dbcsr_acc_opencl_info_memptr_t* info, ACC_OPENCL_LOCKTYPE* lock, const void* memory,
  size_t elsize, const size_t* amount, size_t* offset);
/** Determines device-pointer registration for information. */
int c_dbcsr_acc_opencl_info_devptr(
  c_dbcsr_acc_opencl_info_memptr_t* info, const void* memory, size_t elsize, const size_t* amount, size_t* offset);
/** Finds an existing stream for the given thread-ID (or NULL). */
const c_dbcsr_acc_opencl_stream_t* c_dbcsr_acc_opencl_stream(ACC_OPENCL_LOCKTYPE* lock, int thread_id);
/** Determines default-stream (see ACC_OPENCL_STREAM_NULL). */
const c_dbcsr_acc_opencl_stream_t* c_dbcsr_acc_opencl_stream_default(void);
/** Like c_dbcsr_acc_memset_zero, but supporting an arbitrary value used as initialization pattern. */
int c_dbcsr_acc_opencl_memset(void* dev_mem, int value, size_t offset, size_t nbytes, void* stream);
/** Amount of device memory; local memory is only non-zero if separate from global. */
int c_dbcsr_acc_opencl_info_devmem(cl_device_id device, size_t* mem_free, size_t* mem_total, size_t* mem_local, int* mem_unified);
/** Get device-ID for given device, and optionally global device-ID. */
int c_dbcsr_acc_opencl_device_id(cl_device_id device, int* device_id, int* global_id);
/** Confirm the vendor of the given device. */
int c_dbcsr_acc_opencl_device_vendor(cl_device_id device, const char vendor[], int use_platform_name);
/** Capture or calculate UID based on the device-name. */
int c_dbcsr_acc_opencl_device_uid(cl_device_id device, const char devname[], unsigned int* uid);
/** Based on the device-ID, return the device's UID (capture or calculate), device name, and platform name. */
int c_dbcsr_acc_opencl_device_name(
  cl_device_id device, char name[], size_t name_maxlen, char platform[], size_t platform_maxlen, int cleanup);
/** Return the OpenCL support-level for the given device. */
int c_dbcsr_acc_opencl_device_level(
  cl_device_id device, int std_clevel[2], int std_level[2], char std_flag[16], cl_device_type* type);
/** Check if given device supports the extensions. */
int c_dbcsr_acc_opencl_device_ext(cl_device_id device, const char* const extnames[], int num_exts);
/** Create context for given device. */
int c_dbcsr_acc_opencl_create_context(cl_device_id device_id, cl_context* context);
/** Internal variant of c_dbcsr_acc_set_active_device. */
int c_dbcsr_acc_opencl_set_active_device(ACC_OPENCL_LOCKTYPE* lock, int device_id);
/** Get preferred multiple and max. size of workgroup (kernel- or device-specific). */
int c_dbcsr_acc_opencl_wgsize(cl_device_id device, cl_kernel kernel, size_t* max_value, size_t* preferred_multiple);
/**
 * Build kernel from source with given kernel_name, build_params and build_options.
 * The build_params are meant to instantiate the kernel (-D) whereas build_options
 * are are meant to be compiler-flags.
 */
int c_dbcsr_acc_opencl_kernel(int source_is_file, const char source[], const char kernel_name[], const char build_params[],
  const char build_options[], const char try_build_options[], int* try_ok, const char* const extnames[], int num_exts,
  cl_kernel* kernel);
/** Per-thread variant of c_dbcsr_acc_device_synchronize. */
int c_dbcsr_acc_opencl_device_synchronize(ACC_OPENCL_LOCKTYPE* lock, int thread_id);
/** Assemble flags to support atomic operations. */
int c_dbcsr_acc_opencl_flags_atomics(const c_dbcsr_acc_opencl_device_t* devinfo, c_dbcsr_acc_opencl_atomic_fp_t kind,
  const char* exts[], int exts_maxlen, char flags[], size_t flags_maxlen);
/** Combines build-params and build-options, optional flags (try_build_options). */
int c_dbcsr_acc_opencl_flags(
  const char build_params[], const char build_options[], const char try_build_options[], char buffer[], size_t buffer_size);
/** To support USM, call this function for pointer arguments instead of clSetKernelArg. */
int c_dbcsr_acc_opencl_set_kernel_ptr(cl_kernel kernel, cl_uint arg_index, const void* arg_value);

/** Support older LIBXSMM (libxsmm_pmalloc_init). */
void c_dbcsr_acc_opencl_pmalloc_init(ACC_OPENCL_LOCKTYPE* lock, size_t size, size_t* num, void* pool[], void* storage);
/** Support older LIBXSMM (libxsmm_pmalloc). */
void* c_dbcsr_acc_opencl_pmalloc(ACC_OPENCL_LOCKTYPE* lock, void* pool[], size_t* i);
/** Support older LIBXSMM (libxsmm_pfree). */
void c_dbcsr_acc_opencl_pfree(ACC_OPENCL_LOCKTYPE* lock, const void* pointer, void* pool[], size_t* i);

#if defined(__cplusplus)
}
#endif

#endif /*ACC_OPENCL_H*/
