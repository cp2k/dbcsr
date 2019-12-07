/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#include "dbcsr_omp.h"
#include <stdlib.h>
#include <assert.h>

#if defined(__cplusplus)
extern "C" {
#endif

int dbcsr_omp_initialized;
const char* dbcsr_omp_device;


int dbcsr_omp_alloc(void** item, int typesize, int* counter, int maxcount, void* storage, void** pointer)
{
  int result, i;
  assert(NULL != item && 0 < typesize && NULL != counter);
#if defined(_OPENMP) && (200805 <= _OPENMP) /* OpenMP 3.0 */
# pragma omp atomic capture
#elif defined(_OPENMP)
# pragma omp critical(dbcsr_omp_alloc_critical)
#endif
  i = (*counter)++;
  if (0 < maxcount) { /* fast allocation */
    if (i < maxcount) {
      assert(NULL != storage && NULL != pointer);
      *item = pointer[i];
      if (NULL == *item) {
        *item = &((char*)storage)[i*typesize];
      }
      assert(((const char*)storage) <= ((const char*)*item) && ((const char*)*item) < &((const char*)storage)[maxcount*typesize]);
      result = EXIT_SUCCESS;
    }
    else { /* out of space */
      result = EXIT_FAILURE;
#if defined(_OPENMP)
#     pragma omp atomic
#endif
      --(*counter);
      *item = NULL;
    }
  }
  else {
    *item = malloc(typesize);
    if (NULL != *item) {
      result = EXIT_SUCCESS;
    }
    else {
      result = EXIT_FAILURE;
#if defined(_OPENMP)
#     pragma omp atomic
#endif
      --(*counter);
    }
  }
  DBCSR_OMP_RETURN(result);
}


int dbcsr_omp_dealloc(void* item, int typesize, int* counter, int maxcount, void* storage, void** pointer)
{
  int result;
  assert(0 < typesize && NULL != counter);
  if (NULL != item) {
    int i;
#if defined(_OPENMP) && (200805 <= _OPENMP) /* OpenMP 3.0 */
#   pragma omp atomic capture
#elif defined(_OPENMP)
#   pragma omp critical(dbcsr_omp_alloc_critical)
#endif
    i = (*counter)--;
    assert(0 <= i);
    if (0 < maxcount) { /* fast allocation */
      assert(NULL != storage && NULL != pointer);
      result = ((0 <= i && i < maxcount && storage <= item && /* check if item came from storage */
          ((const char*)item) < &((const char*)storage)[maxcount*typesize])
        ? EXIT_SUCCESS : EXIT_FAILURE);
      pointer[i] = item;
    }
    else {
      result = EXIT_SUCCESS;
      free(item);
    }
  }
  else {
    result = EXIT_SUCCESS;
  }
  DBCSR_OMP_RETURN(result);
}


int acc_init(void)
{
  extern int dbcsr_omp_stream_count;
  extern int dbcsr_omp_event_count;
  dbcsr_omp_device = getenv("DBCSR_OMP_DEVICE");
#if defined(_OPENMP)
  assert(/*master*/0 == omp_get_thread_num());
#endif
#if defined(DBCSR_OMP_OFFLOAD)
# pragma omp target map(tofrom:dbcsr_omp_initialized) if(0/*NULL*/ == dbcsr_omp_device)
#endif
#if defined(_OPENMP)
# pragma omp parallel
# pragma omp master
#endif
  ++dbcsr_omp_initialized;
  DBCSR_OMP_RETURN((1 == dbcsr_omp_initialized
    && 0 == dbcsr_omp_stream_count
    && 0 == dbcsr_omp_event_count)
  ? EXIT_SUCCESS : EXIT_FAILURE);
}


int acc_finalize(void)
{
  extern int dbcsr_omp_stream_count;
  extern int dbcsr_omp_event_count;
#if defined(_OPENMP)
  assert(/*master*/0 == omp_get_thread_num());
# pragma omp atomic
#endif
  --dbcsr_omp_initialized;
  DBCSR_OMP_RETURN((0 == dbcsr_omp_initialized
#if 0
    && 0 == dbcsr_omp_stream_count
    && 0 == dbcsr_omp_event_count
#endif
  ) ? EXIT_SUCCESS : EXIT_FAILURE);
}


void acc_clear_errors(void)
{ assert(0 < dbcsr_omp_initialized);
  /* flush all pending work */
  DBCSR_OMP_EXPECT(EXIT_SUCCESS, acc_event_record(NULL/*event*/, NULL/*stream*/));
  dbcsr_omp_stream_clear_errors();
}


int dbcsr_omp_ndevices(void)
{
#if defined(DBCSR_OMP_OFFLOAD)
  const int ndevices = omp_get_num_devices();
#else
  const int ndevices = 0;
#endif
  return (0 != ndevices ? ndevices : ((NULL != dbcsr_omp_device && 0 != *dbcsr_omp_device) ? 1 : 0));
}


int acc_get_ndevices(int* n_devices)
{
  int result;
  if (NULL != n_devices) {
    *n_devices = dbcsr_omp_ndevices();
    assert(0 <= *n_devices);
    result = EXIT_SUCCESS;
  }
  else {
    result = EXIT_FAILURE;
  }
  assert(0 < dbcsr_omp_initialized);
  DBCSR_OMP_RETURN(result);
}


int acc_set_active_device(int device_id)
{
  const int device = ((NULL == dbcsr_omp_device || 0 == *dbcsr_omp_device)
    ? device_id : (device_id + atoi(dbcsr_omp_device)));
  int result = (0 <= device ? EXIT_SUCCESS : EXIT_FAILURE);
#if defined(_OPENMP)
# pragma omp master
#endif
  if (EXIT_SUCCESS == result) {
#if !defined(NDEBUG)
    if (device_id < dbcsr_omp_ndevices())
#endif
    {
#if defined(DBCSR_OMP_OFFLOAD)
      omp_set_default_device(device);
#endif
      result = EXIT_SUCCESS;
    }
#if !defined(NDEBUG)
    else {
      result = EXIT_FAILURE;
    }
#endif
  }
  assert(0 < dbcsr_omp_initialized);
  DBCSR_OMP_RETURN(result);
}

#if defined(__cplusplus)
}
#endif
