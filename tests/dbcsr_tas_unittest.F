!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

PROGRAM dbcsr_tas_unittest

   !! unit testing for tall-and-skinny matrices
   USE dbcsr_api, ONLY: dbcsr_finalize_lib, &
                        dbcsr_init_lib, &
                        dbcsr_print_statistics
   USE dbcsr_tas_base, ONLY: dbcsr_tas_destroy, &
                             dbcsr_tas_info, &
                             dbcsr_tas_nblkcols_total, &
                             dbcsr_tas_nblkrows_total, &
                             dbcsr_tas_create
   USE dbcsr_tas_types, ONLY: dbcsr_tas_type
   USE dbcsr_tas_test, ONLY: dbcsr_tas_random_bsizes, &
                             dbcsr_tas_setup_test_matrix, &
                             dbcsr_tas_test_mm
   USE dbcsr_kinds, ONLY: int_8, &
                          real_8
   USE dbcsr_machine, ONLY: default_output_unit
   USE dbcsr_mpiwrap, ONLY: mp_comm_free, &
                            mp_environ, &
                            mp_world_finalize, &
                            mp_world_init
   USE dbcsr_tas_io, ONLY: dbcsr_tas_write_split_info
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   INTEGER(KIND=int_8), PARAMETER :: m = 100, k = 20, n = 10
   TYPE(dbcsr_tas_type)             :: A, B, C, At, Bt, Ct, A_out, B_out, C_out, At_out, Bt_out, Ct_out
   INTEGER, DIMENSION(m)          :: bsize_m
   INTEGER, DIMENSION(n)          :: bsize_n
   INTEGER, DIMENSION(k)          :: bsize_k
   REAL(KIND=real_8), PARAMETER   :: sparsity = 0.1
   INTEGER                        :: mp_comm, numnodes, mynode, io_unit
   INTEGER                        :: mp_comm_A, mp_comm_At, mp_comm_B, mp_comm_Bt, mp_comm_C, mp_comm_Ct
   REAL(KIND=real_8), PARAMETER   :: filter_eps = 1.0E-08

   CALL mp_world_init(mp_comm)

   CALL mp_environ(numnodes, mynode, mp_comm)

   io_unit = 0
   IF (mynode .EQ. 0) io_unit = default_output_unit

   CALL dbcsr_init_lib(mp_comm, io_unit)

   CALL dbcsr_tas_random_bsizes([13, 8, 5, 25, 12], 2, bsize_m)
   CALL dbcsr_tas_random_bsizes([3, 78, 33, 12, 3, 15], 1, bsize_n)
   CALL dbcsr_tas_random_bsizes([9, 64, 23, 2], 3, bsize_k)

   CALL mp_environ(numnodes, mynode, mp_comm)

   CALL dbcsr_tas_setup_test_matrix(A, mp_comm_A, mp_comm, m, k, bsize_m, bsize_k, [5, 1], "A", sparsity)
   CALL dbcsr_tas_setup_test_matrix(At, mp_comm_At, mp_comm, k, m, bsize_k, bsize_m, [3, 8], "A^t", sparsity)
   CALL dbcsr_tas_setup_test_matrix(B, mp_comm_B, mp_comm, n, m, bsize_n, bsize_m, [3, 2], "B", sparsity)
   CALL dbcsr_tas_setup_test_matrix(Bt, mp_comm_Bt, mp_comm, m, n, bsize_m, bsize_n, [1, 3], "B^t", sparsity)
   CALL dbcsr_tas_setup_test_matrix(C, mp_comm_C, mp_comm, k, n, bsize_k, bsize_n, [5, 7], "C", sparsity)
   CALL dbcsr_tas_setup_test_matrix(Ct, mp_comm_Ct, mp_comm, n, k, bsize_n, bsize_k, [1, 1], "C^t", sparsity)

   CALL dbcsr_tas_create(A, A_out)
   CALL dbcsr_tas_create(At, At_out)
   CALL dbcsr_tas_create(B, B_out)
   CALL dbcsr_tas_create(Bt, Bt_out)
   CALL dbcsr_tas_create(C, C_out)
   CALL dbcsr_tas_create(Ct, Ct_out)

   IF (mynode == 0) WRITE (io_unit, '(A)') "DBCSR TALL-AND-SKINNY MATRICES"
   IF (mynode == 0) WRITE (io_unit, '(1X, A, 1X, A, I10, 1X, A, 1X, I10)') "Split info for matrix", &
      TRIM(A%matrix%name), dbcsr_tas_nblkrows_total(A), 'X', dbcsr_tas_nblkcols_total(A)
   CALL dbcsr_tas_write_split_info(dbcsr_tas_info(A), io_unit, name="A")
   IF (mynode == 0) WRITE (io_unit, '(1X, A, 1X, A, I10, 1X, A, 1X, I10)') "Split info for matrix", &
      TRIM(At%matrix%name), dbcsr_tas_nblkrows_total(At), 'X', dbcsr_tas_nblkcols_total(At)
   CALL dbcsr_tas_write_split_info(dbcsr_tas_info(At), io_unit, name="At")
   IF (mynode == 0) WRITE (io_unit, '(1X, A, 1X, A, I10, 1X, A, 1X, I10)') "Split info for matrix", &
      TRIM(B%matrix%name), dbcsr_tas_nblkrows_total(B), 'X', dbcsr_tas_nblkcols_total(B)
   CALL dbcsr_tas_write_split_info(dbcsr_tas_info(B), io_unit, name="B")
   IF (mynode == 0) WRITE (io_unit, '(1X, A, 1X, A, I10, 1X, A, 1X, I10)') "Split info for matrix", &
      TRIM(Bt%matrix%name), dbcsr_tas_nblkrows_total(Bt), 'X', dbcsr_tas_nblkcols_total(Bt)
   CALL dbcsr_tas_write_split_info(dbcsr_tas_info(Bt), io_unit, name="Bt")
   IF (mynode == 0) WRITE (io_unit, '(1X, A, 1X, A, I10, 1X, A, 1X, I10)') "Split info for matrix", &
      TRIM(C%matrix%name), dbcsr_tas_nblkrows_total(C), 'X', dbcsr_tas_nblkcols_total(C)
   CALL dbcsr_tas_write_split_info(dbcsr_tas_info(C), io_unit, name="C")
   IF (mynode == 0) WRITE (io_unit, '(1X, A, 1X, A, I10, 1X, A, 1X, I10)') "Split info for matrix", &
      TRIM(Ct%matrix%name), dbcsr_tas_nblkrows_total(Ct), 'X', dbcsr_tas_nblkcols_total(Ct)
   CALL dbcsr_tas_write_split_info(dbcsr_tas_info(Ct), io_unit, name="Ct")

   CALL dbcsr_tas_test_mm('N', 'N', 'N', B, A, Ct_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('T', 'N', 'N', Bt, A, Ct_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('N', 'T', 'N', B, At, Ct_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('T', 'T', 'N', Bt, At, Ct_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('N', 'N', 'T', B, A, C_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('T', 'N', 'T', Bt, A, C_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('N', 'T', 'T', B, At, C_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('T', 'T', 'T', Bt, At, C_out, unit_nr=default_output_unit, filter_eps=filter_eps)

   CALL dbcsr_tas_test_mm('N', 'N', 'N', A, C, Bt_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('T', 'N', 'N', At, C, Bt_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('N', 'T', 'N', A, Ct, Bt_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('T', 'T', 'N', At, Ct, Bt_out, unit_nr=default_output_unit, filter_eps=filter_eps)

   CALL dbcsr_tas_test_mm('N', 'N', 'T', A, C, B_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('T', 'N', 'T', At, C, B_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('N', 'T', 'T', A, Ct, B_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('T', 'T', 'T', At, Ct, B_out, unit_nr=default_output_unit, filter_eps=filter_eps)

   CALL dbcsr_tas_test_mm('N', 'N', 'N', C, B, At_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('T', 'N', 'N', Ct, B, At_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('N', 'T', 'N', C, Bt, At_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('T', 'T', 'N', Ct, Bt, At_out, unit_nr=default_output_unit, filter_eps=filter_eps)

   CALL dbcsr_tas_test_mm('N', 'N', 'T', C, B, A_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('T', 'N', 'T', Ct, B, A_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('N', 'T', 'T', C, Bt, A_out, unit_nr=default_output_unit, filter_eps=filter_eps)
   CALL dbcsr_tas_test_mm('T', 'T', 'T', Ct, Bt, A_out, unit_nr=default_output_unit, filter_eps=filter_eps)

   CALL dbcsr_tas_destroy(A)
   CALL dbcsr_tas_destroy(At)
   CALL dbcsr_tas_destroy(B)
   CALL dbcsr_tas_destroy(Bt)
   CALL dbcsr_tas_destroy(C)
   CALL dbcsr_tas_destroy(Ct)
   CALL dbcsr_tas_destroy(A_out)
   CALL dbcsr_tas_destroy(At_out)
   CALL dbcsr_tas_destroy(B_out)
   CALL dbcsr_tas_destroy(Bt_out)
   CALL dbcsr_tas_destroy(C_out)
   CALL dbcsr_tas_destroy(Ct_out)

   CALL mp_comm_free(mp_comm_A)
   CALL mp_comm_free(mp_comm_At)
   CALL mp_comm_free(mp_comm_B)
   CALL mp_comm_free(mp_comm_Bt)
   CALL mp_comm_free(mp_comm_C)
   CALL mp_comm_free(mp_comm_Ct)

   call dbcsr_print_statistics(.true.)
   CALL dbcsr_finalize_lib()

   CALL mp_world_finalize()

END PROGRAM
