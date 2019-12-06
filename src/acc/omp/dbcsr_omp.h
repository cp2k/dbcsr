/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#ifndef DBCSR_OMP_H
#define DBCSR_OMP_H

#if !defined(DBCSR_OMP_THREADS_MAXCOUNT)
# define DBCSR_OMP_THREADS_MAXCOUNT 8192
#endif
#if !defined(DBCSR_OMP_CACHELINE_NBYTES)
# define DBCSR_OMP_CACHELINE_NBYTES 64
#endif
#if !defined(DBCSR_OMP_ARGUMENTS_MAXCOUNT)
# define DBCSR_OMP_ARGUMENTS_MAXCOUNT 13
#endif
#if !defined(DBCSR_OMP_STREAM_MAXPENDING)
# define DBCSR_OMP_STREAM_MAXPENDING 256
#endif
#if !defined(DBCSR_OMP_STREAM_MAXCOUNT)
# define DBCSR_OMP_STREAM_MAXCOUNT 32
#endif
#if !defined(DBCSR_OMP_EVENT_MAXCOUNT)
# define DBCSR_OMP_EVENT_MAXCOUNT ((DBCSR_OMP_THREADS_MAXCOUNT) > (32*(DBCSR_OMP_STREAM_MAXCOUNT)) \
    ? (DBCSR_OMP_THREADS_MAXCOUNT) : (32*(DBCSR_OMP_STREAM_MAXCOUNT)))
#endif
#if !defined(DBCSR_OMP_PAUSE_MAXCOUNT)
# define DBCSR_OMP_PAUSE_MAXCOUNT 4096
#endif

#if defined(__INTEL_COMPILER)
# if !defined(__INTEL_COMPILER_UPDATE)
#   define DBCSR_OMP_INTEL_COMPILER __INTEL_COMPILER
# else
#   define DBCSR_OMP_INTEL_COMPILER (__INTEL_COMPILER + __INTEL_COMPILER_UPDATE)
# endif
#elif defined(__INTEL_COMPILER_BUILD_DATE)
# define DBCSR_OMP_INTEL_COMPILER ((__INTEL_COMPILER_BUILD_DATE / 10000 - 2000) * 100)
#endif

#define DBCSR_OMP_EXPAND(SYMBOL) SYMBOL
#define DBCSR_OMP_STRINGIFY2(SYMBOL) #SYMBOL
#define DBCSR_OMP_STRINGIFY(SYMBOL) DBCSR_OMP_STRINGIFY2(SYMBOL)
#define DBCSR_OMP_UP2(N, NPOT) ((((uint64_t)N) + ((NPOT) - 1)) & ~((NPOT) - 1))

#if defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__) /*C99*/
# define DBCSR_OMP_PRAGMA(DIRECTIVE) _Pragma(DBCSR_OMP_STRINGIFY(DIRECTIVE))
#else
# if defined(DBCSR_OMP_INTEL_COMPILER) || defined(_MSC_VER)
#   define DBCSR_OMP_PRAGMA(DIRECTIVE) __pragma(DBCSR_OMP_EXPAND(DIRECTIVE))
# else
#   define DBCSR_OMP_PRAGMA(DIRECTIVE)
# endif
#endif
#if defined(DBCSR_OMP_INTEL_COMPILER)
# define DBCSR_OMP_DEP(DEP) *DEP
#else
# define DBCSR_OMP_DEP(DEP) DEP[0]
#endif

#if defined(_OPENMP)
# define DBCSR_OMP_PAUSE { DBCSR_OMP_PRAGMA(omp flush) }
#elif (defined(__GNUC__) && ( \
    (defined(__x86_64__) && 0 != (__x86_64__)) || \
    (defined(__amd64__) && 0 != (__amd64__)) || \
    (defined(_M_X64) || defined(_M_AMD64)) || \
    (defined(__i386__) && 0 != (__i386__)) || \
    (defined(_M_IX86))))
# define DBCSR_OMP_PAUSE __asm__ __volatile__("pause" ::: "memory")
#else
# define DBCSR_OMP_PAUSE __asm__ __volatile__("" ::: "memory")
#endif
#define DBCSR_OMP_WAIT(CONDITION) do { int npause = 0; \
  while (CONDITION) { int counter = 0; \
    for (; counter <= npause; ++counter) DBCSR_OMP_PAUSE; \
    if (npause < DBCSR_OMP_PAUSE_MAXCOUNT) { \
      npause = (0 < npause ? (2 * npause) : 1); \
    } else { /* yield? */ } \
  } \
} while (0)

#if defined(NDEBUG)
# define DBCSR_OMP_EXPECT(EXPECTED, EXPR) EXPR
# define DBCSR_OMP_RETURN(RESULT) return RESULT
#else
# define DBCSR_OMP_EXPECT(EXPECTED, EXPR) assert((EXPECTED) == (EXPR))
# define DBCSR_OMP_RETURN(RESULT) do { \
    const int dbcsr_omp_return_result_ = (RESULT); \
    assert(EXIT_SUCCESS == dbcsr_omp_return_result_); \
    return dbcsr_omp_return_result_; \
  } while (0)
#endif

#if defined(__cplusplus)
# define DBCSR_OMP_EXTERN extern "C"
# define DBCSR_OMP_EXPORT DBCSR_OMP_EXTERN
#else
# define DBCSR_OMP_EXTERN extern
# define DBCSR_OMP_EXPORT
#endif

#if !defined(DBCSR_OMP_BASELINE)
# define DBCSR_OMP_BASELINE 45
#endif

#if defined(_OPENMP)
# if   (201811 <= _OPENMP/*v5.0*/)
#   define DBCSR_OMP_VERSION 50
# elif (201511 <= _OPENMP/*v4.5*/)
#   define DBCSR_OMP_VERSION 45
# elif (201307 <= _OPENMP/*v4.0*/)
#   define DBCSR_OMP_VERSION 40
# endif
# if defined(DBCSR_OMP_VERSION) && (DBCSR_OMP_BASELINE <= DBCSR_OMP_VERSION)
#   if !defined(_CRAYC) /* CRAY: control manually, i.e., define DBCSR_OMP_OFFLOAD */
#     define DBCSR_OMP_OFFLOAD
#   endif
# elif !defined(DBCSR_OMP_VERSION) && defined(__ibmxl__)
#   define DBCSR_OMP_VERSION DBCSR_OMP_BASELINE
#   define DBCSR_OMP_OFFLOAD
# endif
#endif

#include "../include/acc.h"
#include <stdint.h>
#if !defined(NDEBUG)
# include <assert.h>
#endif
#if defined(_OPENMP)
# include <omp.h>
#endif


DBCSR_OMP_EXPORT typedef struct dbcsr_omp_stream_t {
  /* address of each character is (side-)used to form OpenMP task dependencies */
  char name[DBCSR_OMP_STREAM_MAXPENDING];
  volatile int pending, status;
  int priority;
#if defined(DBCSR_OMP_OFFLOAD) && !defined(NDEBUG)
  int device_id; /* should match active device as set by acc_set_active_device */
#endif
} dbcsr_omp_stream_t;

typedef char dbcsr_omp_dependency_t;

DBCSR_OMP_EXPORT typedef struct dbcsr_omp_event_t {
  const dbcsr_omp_dependency_t *volatile dependency;
  acc_bool_t has_occurred;
} dbcsr_omp_event_t;

DBCSR_OMP_EXPORT typedef union dbcsr_omp_any_t {
  const void* const_ptr; void* ptr;
  uint64_t u64;
  int64_t i64;
  size_t size;
  uint32_t u32;
  int32_t i32;
  acc_bool_t logical;
} dbcsr_omp_any_t;

DBCSR_OMP_EXPORT typedef struct dbcsr_omp_depend_data_t {
  /** Used to record the arguments/signature of each OpenMP-offload/call on a per-thread basis. */
  dbcsr_omp_any_t args[DBCSR_OMP_ARGUMENTS_MAXCOUNT];
  /** The in/out-pointer must be dereferenced (depend clause expects value; due to syntax issues use in[0]/out[0]). */
  const dbcsr_omp_dependency_t *in, *out;
  int counter;
} dbcsr_omp_depend_data_t;

DBCSR_OMP_EXPORT typedef union dbcsr_omp_depend_t {
  char pad[DBCSR_OMP_UP2(sizeof(dbcsr_omp_depend_data_t),DBCSR_OMP_CACHELINE_NBYTES)];
  dbcsr_omp_depend_data_t data;
} dbcsr_omp_depend_t;

DBCSR_OMP_EXPORT int dbcsr_omp_ndevices(void);
/** Helper function for lock-free allocation of preallocated items such as streams or events. */
DBCSR_OMP_EXPORT int dbcsr_omp_alloc(void** item, int typesize, int* counter, int maxcount, void* storage, void** pointer);
/** Helper function for lock-free deallocation (companion of dbcsr_omp_alloc). */
DBCSR_OMP_EXPORT int dbcsr_omp_dealloc(void* item, int typesize, int* counter, int maxcount, void* storage, void** pointer);
/** Generate dependency for given stream. If a dependency is not consumed, acc_event_record(NULL, NULL) shall be called. */
DBCSR_OMP_EXPORT void dbcsr_omp_stream_depend(acc_stream_t* stream, dbcsr_omp_depend_t** depend);
/** Get the number of tasks to be issued. */
DBCSR_OMP_EXPORT int dbcsr_omp_stream_depend_get_count(void);
/** Commits the data filled into "depend" (as given by dbcsr_omp_stream_depend of each thread). */
DBCSR_OMP_EXPORT void dbcsr_omp_stream_depend_begin(void);
/** Signals the end of the reduction and returns an error code. */
DBCSR_OMP_EXPORT int dbcsr_omp_stream_depend_end(const acc_stream_t* stream);
/** Clears status of all streams (if possible). */
DBCSR_OMP_EXPORT void dbcsr_omp_stream_clear_errors(void);

#endif /*DBCSR_OMP_H*/
