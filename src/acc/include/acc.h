/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#ifndef DBCSR_ACC_H
#define DBCSR_ACC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** types */
typedef void acc_stream_t;
typedef void acc_event_t;
typedef int acc_bool_t;

typedef enum acc_data_t {
  ACC_DATA_F64 = 3,
  ACC_DATA_F32 = 1,
  ACC_DATA_C64 = 7,
  ACC_DATA_C32 = 5,
  ACC_DATA_UNKNOWN = -1
} acc_data_t;

/** accelerator driver initialization and finalization */
int acc_init(void);
int acc_finalize(void);
int acc_clear_errors(void);

/** devices */
int acc_get_ndevices(int* n_devices);
int acc_set_active_device(int device_id);

/** streams */
int acc_stream_priority_range(int* least, int* greatest);
int acc_stream_create(acc_stream_t** stream_p, const char* name, int priority);
int acc_stream_destroy(acc_stream_t* stream);
int acc_stream_sync(acc_stream_t* stream);
int acc_stream_wait_event(acc_stream_t* stream, acc_event_t* event);

/** events */
int acc_event_create(acc_event_t** event_p);
int acc_event_destroy(acc_event_t* event);
int acc_event_record(acc_event_t* event, acc_stream_t* stream);
int acc_event_query(acc_event_t* event, acc_bool_t* has_occurred);
int acc_event_synchronize(acc_event_t* event);

/** memory */
int acc_dev_mem_allocate(void** dev_mem, size_t n);
int acc_dev_mem_deallocate(void* dev_mem);
int acc_dev_mem_set_ptr(void** dev_mem, void* other, size_t lb);
int acc_host_mem_allocate(void** host_mem, size_t n, acc_stream_t* stream);
int acc_host_mem_deallocate(void* host_mem, acc_stream_t* stream);
int acc_memcpy_h2d(const void* host_mem, void* dev_mem, size_t count, acc_stream_t* stream);
int acc_memcpy_d2h(const void* dev_mem, void* host_mem, size_t count, acc_stream_t* stream);
int acc_memcpy_d2d(const void* devmem_src, void* devmem_dst, size_t count, acc_stream_t* stream);
int acc_memset_zero(void* dev_mem, size_t offset, size_t length, acc_stream_t* stream);
int acc_dev_mem_info(size_t* mem_free, size_t* mem_total);

#ifdef __cplusplus
}
#endif

#endif /*DBCSR_ACC_H*/
