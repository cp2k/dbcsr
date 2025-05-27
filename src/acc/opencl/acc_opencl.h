/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: BSD-3-Clause                                                          */
/*------------------------------------------------------------------------------------------------*/
#ifndef ACC_OPENCL_H
#define ACC_OPENCL_H

/* Support for other libraries, e.g., CP2K's DBM/DBT */
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
#if !defined(ACC_OPENCL_TLS)
#  define ACC_OPENCL_TLS LIBXSMM_TLS
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
#if !defined(ACC_OPENCL_ASYNC) && 1
#  define ACC_OPENCL_ASYNC getenv("ACC_OPENCL_ASYNC")
#endif
#if !defined(ACC_OPENCL_XHINTS) && 1
#  define ACC_OPENCL_XHINTS getenv("ACC_OPENCL_XHINTS")
#endif
#if !defined(ACC_OPENCL_STREAM_PRIORITIES) && 0
#  if defined(CL_QUEUE_PRIORITY_KHR)
#    define ACC_OPENCL_STREAM_PRIORITIES
#  endif
#endif
/* Activate device by default */
#if !defined(ACC_OPENCL_ACTIVATE) && 0
#  define ACC_OPENCL_ACTIVATE 0
#endif
/* Use DBCSR's profile for detailed timings (function name prefix-offset) */
#if !defined(ACC_OPENCL_PROFILE_DBCSR) && (defined(__OFFLOAD_PROFILING) || 1)
#  if defined(__DBCSR_ACC)
#    define ACC_OPENCL_PROFILE_DBCSR 8
#  endif
#endif

/* attaching c_dbcsr_acc_opencl_stream_t is needed */
#define ACC_OPENCL_STREAM(A) ((const c_dbcsr_acc_opencl_stream_t*)(A))
/* incompatible with c_dbcsr_acc_event_record */
#define ACC_OPENCL_EVENT(A) ((const cl_event*)(A))

#define ACC_OPENCL_ATOMIC_ACQUIRE(LOCK) \
  do { \
    LIBXSMM_ATOMIC_ACQUIRE(LOCK, LIBXSMM_SYNC_NPAUSE, LIBXSMM_ATOMIC_SEQ_CST); \
  } while (0)
#define ACC_OPENCL_ATOMIC_RELEASE(LOCK) \
  do { \
    LIBXSMM_ATOMIC_RELEASE(LOCK, LIBXSMM_ATOMIC_SEQ_CST); \
  } while (0)

#if defined(LIBXSMM_ATOMIC_LOCKTYPE)
#  define ACC_OPENCL_ATOMIC_LOCKTYPE volatile LIBXSMM_ATOMIC_LOCKTYPE
#else
#  define ACC_OPENCL_ATOMIC_LOCKTYPE volatile int
#endif

#if defined(_OPENMP)
#  include <omp.h>
#  define ACC_OPENCL_OMP_TID() omp_get_thread_num()
#  define ACC_OPENCL_INIT(LOCK) omp_init_lock(LOCK)
#  define ACC_OPENCL_DESTROY(LOCK) omp_destroy_lock(LOCK)
#  define ACC_OPENCL_ACQUIRE(LOCK) omp_set_lock(LOCK)
#  define ACC_OPENCL_RELEASE(LOCK) omp_unset_lock(LOCK)
#  define ACC_OPENCL_LOCKTYPE omp_lock_t
#else
#  define ACC_OPENCL_OMP_TID() (/*main*/ 0)
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

#define ACC_OPENCL_ERROR() c_dbcsr_acc_opencl_config.device.error.code
#define ACC_OPENCL_ERROR_NAME(CODE) \
  ((EXIT_SUCCESS != c_dbcsr_acc_opencl_config.device.error.code && (CODE) == c_dbcsr_acc_opencl_config.device.error.code) \
      ? c_dbcsr_acc_opencl_config.device.error.name \
      : "")

#define ACC_OPENCL_ERROR_REPORT(NAME) \
  do { \
    const char* const acc_opencl_error_report_name_ = (const char*)('\0' != *#NAME ? (uintptr_t)(NAME + 0) : 0); \
    if (0 != c_dbcsr_acc_opencl_config.verbosity) { \
      if (NULL != acc_opencl_error_report_name_ && '\0' != *acc_opencl_error_report_name_) { \
        fprintf(stderr, "ERROR ACC/OpenCL: failed for %s!\n", acc_opencl_error_report_name_); \
      } \
      else if (0 != c_dbcsr_acc_opencl_config.device.error.code) { \
        if (NULL != c_dbcsr_acc_opencl_config.device.error.name && '\0' != *c_dbcsr_acc_opencl_config.device.error.name) { \
          fprintf(stderr, "ERROR ACC/OpenCL: %s (code=%i)\n", c_dbcsr_acc_opencl_config.device.error.name, \
            c_dbcsr_acc_opencl_config.device.error.code); \
        } \
        else if (-1001 == c_dbcsr_acc_opencl_config.device.error.code) { \
          fprintf(stderr, "ERROR ACC/OpenCL: incomplete OpenCL installation?\n"); \
        } \
        else { \
          fprintf(stderr, "ERROR ACC/OpenCL: unknown error (code=%i)\n", c_dbcsr_acc_opencl_config.device.error.code); \
        } \
      } \
      memset(&c_dbcsr_acc_opencl_config.device.error, 0, sizeof(c_dbcsr_acc_opencl_config.device.error)); \
    } \
    assert(!"SUCCESS"); \
  } while (0)

#define ACC_OPENCL_CHECK(RESULT, CMD, MSG) \
  do { \
    if (EXIT_SUCCESS == (RESULT)) { \
      (RESULT) = (CMD); /* update result given code from cmd */ \
      c_dbcsr_acc_opencl_config.device.error.name = (MSG); \
      c_dbcsr_acc_opencl_config.device.error.code = (RESULT); \
      assert(EXIT_SUCCESS == (RESULT)); \
    } \
    else ACC_OPENCL_ERROR_REPORT(); \
  } while (0)

#define ACC_OPENCL_RETURN(RESULT, ...) \
  do { \
    if (EXIT_SUCCESS == (RESULT)) { \
      assert(EXIT_SUCCESS == c_dbcsr_acc_opencl_config.device.error.code); \
      memset(&c_dbcsr_acc_opencl_config.device.error, 0, sizeof(c_dbcsr_acc_opencl_config.device.error)); \
    } \
    else ACC_OPENCL_ERROR_REPORT(__VA_ARGS__); \
    return (RESULT); \
  } while (0)


#if defined(__cplusplus)
extern "C" {
#endif

/** Rich type denoting an error. */
typedef struct c_dbcsr_acc_opencl_error_t {
  const char* name;
  int code;
} c_dbcsr_acc_opencl_error_t;

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
  /**
   * Stream for internal purpose, e.g., stream-argument
   * (ACC-interface) can be NULL (synchronous)
   */
  c_dbcsr_acc_opencl_stream_t stream;
  /** Last error (not necessarily thread-safe/specific). */
  c_dbcsr_acc_opencl_error_t error;
  /** OpenCL compiler flag (language standard). */
  char std_flag[16];
  /** OpenCL support-level (major and minor). */
  cl_int std_level[2], std_clevel[2];
  /**
   * Maximum size of workgroup (WG), preferred multiple of WG-size (PM),
   * and size of subgoup (SG) only if larger-equal than PM. SG is signaled
   * smaller if an alternative SG-size exists (SG is zero if no support).
   */
  size_t wgsize[3];
  /** Kind of device (GPU, CPU, or other). */
  cl_device_type type;
  /** Whether host memory is unified. */
  cl_int unified;
  /** Device-UID. */
  cl_uint uid;
  /** Main vendor? */
  cl_int intel, amd, nv;
  /* USM support functions */
  cl_int (*clSetKernelArgMemPointerINTEL)(cl_kernel, cl_uint, const void*);
  cl_int (*clEnqueueMemFillINTEL)(cl_command_queue, void*, const void*, size_t, size_t, cl_uint, const cl_event*, cl_event*);
  cl_int (*clEnqueueMemcpyINTEL)(cl_command_queue, cl_bool, void*, const void*, size_t, cl_uint, const cl_event*, cl_event*);
  void* (*clDeviceMemAllocINTEL)(cl_context, cl_device_id, const void /*cl_mem_properties_intel*/*, size_t, cl_uint, cl_int*);
  cl_int (*clMemFreeINTEL)(cl_context, void*);
} c_dbcsr_acc_opencl_device_t;

/** Information about host/device-memory pointer. */
typedef struct c_dbcsr_acc_opencl_info_memptr_t {
  cl_mem memory; /* first item! */
  void* memptr;
  /*void *data;*/
} c_dbcsr_acc_opencl_info_memptr_t;

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
  /** Active device (per process). */
  c_dbcsr_acc_opencl_device_t device;
  /** Locks used by domain. */
  ACC_OPENCL_LOCKTYPE *lock_main, *lock_stream, *lock_event, *lock_memory;
  /** All memptrs and related storage/counter. */
  c_dbcsr_acc_opencl_info_memptr_t **memptrs, *memptr_data;
  size_t nmemptrs; /* counter */
  /** Handle-counter. */
  size_t nstreams, nevents;
  /** All streams and related storage. */
  c_dbcsr_acc_opencl_stream_t **streams, *stream_data;
  /** All events and related storage. */
  cl_event **events, *event_data;
  /** Device-ID to lookup devices-array. */
  cl_int device_id;
  /** Kernel-parameters are matched against device's UID */
  cl_uint devmatch;
  /** Split devices into sub-devices (if possible) */
  cl_int devsplit;
  /** Verbosity level (output on stderr). */
  cl_int verbosity;
  /** Guessed number of ranks per node (local), and rank-ID. */
  cl_int nranks, nrank;
  /** Non-zero if library is initialized (negative: no device). */
  cl_int ndevices;
  /** Maximum number of threads (omp_get_max_threads). */
  cl_int nthreads;
#if defined(ACC_OPENCL_STREAM_PRIORITIES)
  /** Runtime-adjust ACC_OPENCL_STREAM_PRIORITIES. */
  cl_int priority;
#endif
  /** Runtime-enable ACC_OPENCL_PROFILE_DBCSR. */
  cl_int profile;
  /** Detailed/optional insight. */
  void *hist_h2d, *hist_d2h, *hist_d2d;
  /** Configuration and execution-hints. */
  cl_int xhints;
  /** Asynchronous memory operations. */
  cl_int async;
  /** Debug (output/symbols, etc.). */
  cl_int debug;
  /** Dump level. */
  cl_int dump;
  /** WA level */
  cl_int wa;
} c_dbcsr_acc_opencl_config_t;

/** Global configuration setup in c_dbcsr_acc_init. */
extern c_dbcsr_acc_opencl_config_t c_dbcsr_acc_opencl_config;

/** Determines host-pointer registration for modification. */
c_dbcsr_acc_opencl_info_memptr_t* c_dbcsr_acc_opencl_info_hostptr(const void* memory);
/** Determines device-pointer registration for modification (internal); offset is measured in elsize. */
c_dbcsr_acc_opencl_info_memptr_t* c_dbcsr_acc_opencl_info_devptr_modify(
  ACC_OPENCL_LOCKTYPE* lock, void* memory, size_t elsize, const size_t* amount, size_t* offset);
/** Determines device-pointer registration for info/ro (lock-control); offset is measured in elsize. */
int c_dbcsr_acc_opencl_info_devptr_lock(c_dbcsr_acc_opencl_info_memptr_t* info, ACC_OPENCL_LOCKTYPE* lock, const void* memory,
  size_t elsize, const size_t* amount, size_t* offset);
/** Determines device-pointer registration for info/ro; offset is measured in elsize. */
int c_dbcsr_acc_opencl_info_devptr(
  c_dbcsr_acc_opencl_info_memptr_t* info, const void* memory, size_t elsize, const size_t* amount, size_t* offset);
/** Finds an existing stream for the given thread-ID (or NULL). */
const c_dbcsr_acc_opencl_stream_t* c_dbcsr_acc_opencl_stream(ACC_OPENCL_LOCKTYPE* lock, int thread_id);
/** Determines default-stream (see c_dbcsr_acc_opencl_device_t::stream). */
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
/** Assemble flags to support atomic operations. */
int c_dbcsr_acc_opencl_flags_atomics(const c_dbcsr_acc_opencl_device_t* devinfo, c_dbcsr_acc_opencl_atomic_fp_t kind,
  const char* exts[], size_t* exts_maxlen, char flags[], size_t flags_maxlen);
/** Assemble given defines and internal definitions. */
int c_dbcsr_acc_opencl_defines(const char defines[], char buffer[], size_t buffer_size, int cleanup);
/** Combines build-params, build-options, and extra flags. */
int c_dbcsr_acc_opencl_kernel_flags(const char build_params[], const char build_options[], const char try_options[],
  cl_program program, char buffer[], size_t buffer_size);
/**
 * Build kernel from source with given kernel_name, build_params and build_options.
 * The build_params are meant to instantiate the kernel (-D) whereas build_options
 * are are meant to be compiler-flags.
 */
int c_dbcsr_acc_opencl_kernel(int source_is_file, const char source[], const char kernel_name[], const char build_params[],
  const char build_options[], const char try_build_options[], int* try_ok, const char* const extnames[], size_t num_exts,
  cl_kernel* kernel);
/** Per-thread variant of c_dbcsr_acc_device_synchronize. */
int c_dbcsr_acc_opencl_device_synchronize(ACC_OPENCL_LOCKTYPE* lock, int thread_id);
/** To support USM, call this function for pointer arguments instead of clSetKernelArg. */
int c_dbcsr_acc_opencl_set_kernel_ptr(cl_kernel kernel, cl_uint arg_index, const void* arg_value);

/** Support older LIBXSMM (libxsmm_pmalloc_init). */
void c_dbcsr_acc_opencl_pmalloc_init(size_t size, size_t* num, void* pool[], void* storage);
/** Support older LIBXSMM (libxsmm_pmalloc). */
void* c_dbcsr_acc_opencl_pmalloc(ACC_OPENCL_LOCKTYPE* lock, void* pool[], size_t* i);
/** Support older LIBXSMM (libxsmm_pfree). */
void c_dbcsr_acc_opencl_pfree(const void* pointer, void* pool[], size_t* i);

/** Measure time in seconds for the given event. */
double c_dbcsr_acc_opencl_duration(cl_event event, int* result_code);

typedef void (*c_dbcsr_acc_opencl_hist_update_fn)(double* /*dst*/, const double* /*src*/);
typedef double (*c_dbcsr_acc_opencl_hist_adjust_fn)(double /*value*/, int count);
void c_dbcsr_acc_opencl_hist_create(
  void** hist, int nbuckets, int nqueue, int nvals, const c_dbcsr_acc_opencl_hist_update_fn update[]);
void c_dbcsr_acc_opencl_hist_avg(double* dst, const double* src);
void c_dbcsr_acc_opencl_hist_add(double* dst, const double* src);
void c_dbcsr_acc_opencl_hist_set(ACC_OPENCL_LOCKTYPE* lock, void* hist, const double vals[]);
void c_dbcsr_acc_opencl_hist_get(
  ACC_OPENCL_LOCKTYPE* lock, void* hist, const int** buckets, int* nbuckets, double range[2], const double** vals, int* nvals);
void c_dbcsr_acc_opencl_hist_print(
  FILE* stream, void* hist, const char title[], const int prec[], const c_dbcsr_acc_opencl_hist_adjust_fn adjust[]);
void c_dbcsr_acc_opencl_hist_free(void* hist);

#if defined(__cplusplus)
}
#endif

#endif /*ACC_OPENCL_H*/
