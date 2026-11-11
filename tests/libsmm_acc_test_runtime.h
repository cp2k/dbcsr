/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/

#ifndef LIBSMM_ACC_TEST_RUNTIME_H
#define LIBSMM_ACC_TEST_RUNTIME_H

#include <stdio.h>

#include "acc/acc.h"

#if defined(DBCSR_LIBSMM_ACC_TEST_USE_MPI)
#  include <mpi.h>
#endif

struct libsmm_acc_test_runtime {
  int rank;
  int nranks;
};

inline int libsmm_acc_test_runtime_init(int* argc, char*** argv, libsmm_acc_test_runtime* runtime) {
  runtime->rank = 0;
  runtime->nranks = 1;
  int local_rank = 0;

#if defined(DBCSR_LIBSMM_ACC_TEST_USE_MPI)
  if (MPI_Init(argc, argv) != MPI_SUCCESS) return 1;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &runtime->rank) != MPI_SUCCESS ||
      MPI_Comm_size(MPI_COMM_WORLD, &runtime->nranks) != MPI_SUCCESS)
  {
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  MPI_Comm local_comm = MPI_COMM_NULL;
  if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, runtime->rank, MPI_INFO_NULL, &local_comm) != MPI_SUCCESS ||
      MPI_Comm_rank(local_comm, &local_rank) != MPI_SUCCESS || MPI_Comm_free(&local_comm) != MPI_SUCCESS)
  {
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
#else
  DBCSR_MARK_USED(argc);
  DBCSR_MARK_USED(argv);
#endif

  int ndevices = 0;
  int status = 0;
  if (c_dbcsr_acc_get_ndevices(&ndevices) != 0 || ndevices < 1) {
    fprintf(stderr, "ERROR: No accelerator device found.\n");
    status = 1;
  }

  int device = -1;
  if (status == 0) {
    device = local_rank % ndevices;
    if (c_dbcsr_acc_set_active_device(device) != 0 || c_dbcsr_acc_init() != 0) {
      fprintf(stderr, "ERROR: Rank %d failed to initialize accelerator device %d.\n", runtime->rank, device);
      status = 1;
    }
  }

  if (status != 0) {
#if defined(DBCSR_LIBSMM_ACC_TEST_USE_MPI)
    MPI_Abort(MPI_COMM_WORLD, status);
#endif
    return 1;
  }

  printf("# Rank %d/%d activated device %d/%d.\n", runtime->rank, runtime->nranks, device, ndevices);
  return 0;
}

inline int libsmm_acc_test_runtime_finalize(int errors) {
  if (c_dbcsr_acc_finalize() != 0) errors += 1;

#if defined(DBCSR_LIBSMM_ACC_TEST_USE_MPI)
  int total_errors = 0;
  if (MPI_Allreduce(&errors, &total_errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) total_errors += 1;
  if (MPI_Finalize() != MPI_SUCCESS) total_errors += 1;
#else
  const int total_errors = errors;
#endif
  return total_errors;
}

#endif /* LIBSMM_ACC_TEST_RUNTIME_H */
