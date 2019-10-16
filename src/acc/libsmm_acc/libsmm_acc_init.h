/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#ifndef LIBSMM_ACC_INIT_H
#define LIBSMM_ACC_INIT_H

int libsmm_acc_init (void);

int libsmm_acc_check_gpu_warp_size_consistency (void);

int acc_get_gpu_warp_size (void);

int libsmm_acc_is_thread_safe (void);

#endif // LIBSMM_ACC_INIT_H
