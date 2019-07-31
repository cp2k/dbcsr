/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#ifndef CUSMM_COMMON_H
#define CUSMM_COMMON_H

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#include "hip/hip_runtime.h"

/******************************************************************************
 * There IS native support for atomicAdd on doubles in HIP so no need for the following
 ******************************************************************************/

/******************************************************************************
 * A simple __ldg replacement for HIP
 ******************************************************************************/
#define __ldg(x)  (*(x))

/******************************************************************************
 * syncthreads macro                                                          *
 ******************************************************************************/
#define syncthreads(x) __syncthreads(x)

#endif
