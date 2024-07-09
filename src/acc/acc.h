/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: BSD-3-Clause                                                          */
/*------------------------------------------------------------------------------------------------*/
#ifndef DBCSR_ACC_H
#define DBCSR_ACC_H

#include <stddef.h>

#define DBCSR_STRINGIFY_AUX(SYMBOL) #SYMBOL
#define DBCSR_STRINGIFY(SYMBOL) DBCSR_STRINGIFY_AUX(SYMBOL)
#define DBCSR_CONCATENATE2(A, B) A##B
#define DBCSR_CONCATENATE(A, B) DBCSR_CONCATENATE2(A, B)

/** used to mark variables used */
#define DBCSR_MARK_USED(x) (void)(x)

#if defined(__cplusplus)
extern "C" {
#endif

/** types */
typedef int c_dbcsr_acc_bool_t;

/** initialization and finalization */
int c_dbcsr_acc_init(void);
int c_dbcsr_acc_finalize(void);
void c_dbcsr_acc_clear_errors(void);

/** devices */
int c_dbcsr_acc_get_ndevices(int* ndevices);
int c_dbcsr_acc_set_active_device(int device_id);
int c_dbcsr_acc_device_synchronize(void);

/** streams */
int c_dbcsr_acc_stream_priority_range(int* least, int* greatest);
int c_dbcsr_acc_stream_create(void** stream_p, const char* name,
  /** lower number is higher priority */
  int priority);
int c_dbcsr_acc_stream_destroy(void* stream);
int c_dbcsr_acc_stream_sync(void* stream);
int c_dbcsr_acc_stream_wait_event(void* stream, void* event);

/** events */
int c_dbcsr_acc_event_create(void** event_p);
int c_dbcsr_acc_event_destroy(void* event);
int c_dbcsr_acc_event_record(void* event, void* stream);
int c_dbcsr_acc_event_query(void* event, c_dbcsr_acc_bool_t* has_occurred);
int c_dbcsr_acc_event_synchronize(void* event);

/** memory */
int c_dbcsr_acc_dev_mem_allocate(void** dev_mem, size_t nbytes);
int c_dbcsr_acc_dev_mem_deallocate(void* dev_mem);
int c_dbcsr_acc_dev_mem_set_ptr(void** dev_mem, void* other, size_t lb);
int c_dbcsr_acc_host_mem_allocate(void** host_mem, size_t nbytes, void* stream);
int c_dbcsr_acc_host_mem_deallocate(void* host_mem, void* stream);
int c_dbcsr_acc_memcpy_h2d(const void* host_mem, void* dev_mem, size_t nbytes, void* stream);
int c_dbcsr_acc_memcpy_d2h(const void* dev_mem, void* host_mem, size_t nbytes, void* stream);
int c_dbcsr_acc_memcpy_d2d(const void* devmem_src, void* devmem_dst, size_t nbytes, void* stream);
int c_dbcsr_acc_memset_zero(void* dev_mem, size_t offset, size_t nbytes, void* stream);
int c_dbcsr_acc_dev_mem_info(size_t* mem_free, size_t* mem_total);

void c_dbcsr_timeset(const char** routineN, const int* routineN_len, int* handle);
void c_dbcsr_timestop(const int* handle);

#if defined(__cplusplus)
}
#endif

#endif /*DBCSR_ACC_H*/
