!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tas_test

   !! testing infrastructure for tall-and-skinny matrices
   USE dbcsr_types, ONLY: dbcsr_type_real_8
   USE dbcsr_data_methods, ONLY: dbcsr_scalar
   USE dbcsr_methods, ONLY: &
      dbcsr_release, dbcsr_nblkcols_total, dbcsr_nblkrows_total, dbcsr_row_block_sizes, dbcsr_col_block_sizes, &
      dbcsr_mp_release, dbcsr_distribution_release
   USE dbcsr_multiply_api, ONLY: dbcsr_multiply
   USE dbcsr_tas_base, ONLY: &
      dbcsr_tas_convert_to_dbcsr, dbcsr_tas_create, dbcsr_tas_distribution_new, &
      dbcsr_tas_finalize, dbcsr_tas_get_stored_coordinates, dbcsr_tas_nblkcols_total, &
      dbcsr_tas_nblkrows_total, dbcsr_tas_put_block, dbcsr_tas_info
   USE dbcsr_tas_types, ONLY: dbcsr_tas_distribution_type, &
                              dbcsr_tas_type
   USE dbcsr_tas_global, ONLY: dbcsr_tas_blk_size_arb, &
                               dbcsr_tas_dist_cyclic, &
                               dbcsr_tas_default_distvec
   USE dbcsr_tas_mm, ONLY: dbcsr_tas_multiply
   USE dbcsr_tas_split, ONLY: dbcsr_tas_mp_comm, &
                              dbcsr_tas_get_split_info
   USE dbcsr_tas_util, ONLY: dbcsr_mp_environ, &
                             invert_transpose_flag
   USE dbcsr_types, ONLY: &
      dbcsr_type, dbcsr_distribution_obj, dbcsr_mp_obj, dbcsr_no_transpose, dbcsr_transpose, &
      dbcsr_type_no_symmetry
   USE dbcsr_kinds, ONLY: int_8, &
                          real_8
   USE dbcsr_mpiwrap, ONLY: mp_environ, &
                            mp_cart_create, &
                            mp_comm_free, mp_comm_type
   USE dbcsr_dist_methods, ONLY: dbcsr_distribution_new
   USE dbcsr_work_operations, ONLY: dbcsr_create, &
                                    dbcsr_finalize
   USE dbcsr_dist_util, ONLY: dbcsr_checksum
   USE dbcsr_operations, ONLY: dbcsr_maxabs, &
                               dbcsr_add
   USE dbcsr_transformations, ONLY: dbcsr_complete_redistribute
   USE dbcsr_blas_operations, ONLY: &
      set_larnv_seed
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   PUBLIC :: &
      dbcsr_tas_benchmark_mm, &
      dbcsr_tas_checksum, &
      dbcsr_tas_random_bsizes, &
      dbcsr_tas_setup_test_matrix, &
      dbcsr_tas_test_mm, &
      dbcsr_tas_reset_randmat_seed

   INTEGER, SAVE :: randmat_counter = 0
   INTEGER, PARAMETER, PRIVATE :: rand_seed_init = 12341313

CONTAINS
   SUBROUTINE dbcsr_tas_setup_test_matrix(matrix, mp_comm_out, mp_comm, nrows, ncols, rbsizes, cbsizes, &
      !! Setup tall-and-skinny matrix for testing
                                          dist_splitsize, name, sparsity, reuse_comm)

      TYPE(dbcsr_tas_type), INTENT(OUT)                    :: matrix
      TYPE(mp_comm_type), INTENT(OUT)                               :: mp_comm_out
      TYPE(mp_comm_type), INTENT(IN)                                :: mp_comm
      INTEGER(KIND=int_8), INTENT(IN)                    :: nrows, ncols
      INTEGER, DIMENSION(nrows), INTENT(IN)              :: rbsizes
      INTEGER, DIMENSION(ncols), INTENT(IN)              :: cbsizes
      INTEGER, DIMENSION(2), INTENT(IN)                  :: dist_splitsize
      CHARACTER(len=*), INTENT(IN)                       :: name
      REAL(KIND=real_8), INTENT(IN)                      :: sparsity
      LOGICAL, INTENT(IN), OPTIONAL                      :: reuse_comm

      INTEGER                                            :: col_size, max_col_size, max_nze, &
                                                            max_row_size, mynode, node_holds_blk, &
                                                            numnodes, nze, row_size
      INTEGER(KIND=int_8)                                :: col, col_s, row, row_s, nrow, ncol
      INTEGER, DIMENSION(2)                              :: pcoord, pdims
      LOGICAL                                            :: reuse_comm_prv, tr
      REAL(KIND=real_8), DIMENSION(1)                    :: rn
      REAL(KIND=real_8), ALLOCATABLE, DIMENSION(:, :)    :: values
      TYPE(dbcsr_tas_blk_size_arb)                         :: cbsize_obj, rbsize_obj
      TYPE(dbcsr_tas_dist_cyclic)                          :: col_dist_obj, row_dist_obj
      TYPE(dbcsr_tas_distribution_type)                    :: dist
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_setup_test_matrix'
      INTEGER :: handle
      INTEGER, DIMENSION(4)                              :: iseed, jseed

      ! we don't reserve blocks prior to putting them, so this time is meaningless and should not
      ! be considered in benchmark!
      CALL timeset(routineN, handle)

      ! Check that the counter was initialised (or has not overflowed)
      DBCSR_ASSERT(randmat_counter .NE. 0)
      ! the counter goes into the seed. Every new call gives a new random matrix
      randmat_counter = randmat_counter + 1

      IF (PRESENT(reuse_comm)) THEN
         reuse_comm_prv = reuse_comm
      ELSE
         reuse_comm_prv = .FALSE.
      END IF

      IF (reuse_comm_prv) THEN
         mp_comm_out = mp_comm
      ELSE
         mp_comm_out = dbcsr_tas_mp_comm(mp_comm, nrows, ncols)
      END IF

      CALL mp_environ(numnodes, mynode, mp_comm_out)
      CALL mp_environ(numnodes, pdims, pcoord, mp_comm_out)

      row_dist_obj = dbcsr_tas_dist_cyclic(dist_splitsize(1), pdims(1), nrows)
      col_dist_obj = dbcsr_tas_dist_cyclic(dist_splitsize(2), pdims(2), ncols)

      rbsize_obj = dbcsr_tas_blk_size_arb(rbsizes)
      cbsize_obj = dbcsr_tas_blk_size_arb(cbsizes)

      CALL dbcsr_tas_distribution_new(dist, mp_comm_out, row_dist_obj, col_dist_obj)
      CALL dbcsr_tas_create(matrix, name, dist=dist, data_type=dbcsr_type_real_8, &
                            row_blk_size=rbsize_obj, col_blk_size=cbsize_obj, own_dist=.TRUE.)

      max_row_size = MAXVAL(rbsizes)
      max_col_size = MAXVAL(cbsizes)
      max_nze = max_row_size*max_col_size

      nrow = dbcsr_tas_nblkrows_total(matrix)
      ncol = dbcsr_tas_nblkcols_total(matrix)

      ALLOCATE (values(max_row_size, max_col_size))

      CALL set_larnv_seed(7, 42, 3, 42, randmat_counter, jseed)

      DO row = 1, dbcsr_tas_nblkrows_total(matrix)
         DO col = 1, dbcsr_tas_nblkcols_total(matrix)
            CALL dlarnv(1, jseed, 1, rn)
            IF (rn(1) .LT. sparsity) THEN
               tr = .FALSE.
               row_s = row; col_s = col
               CALL dbcsr_tas_get_stored_coordinates(matrix, row_s, col_s, node_holds_blk)

               IF (node_holds_blk .EQ. mynode) THEN
                  row_size = rbsize_obj%data(row_s)
                  col_size = cbsize_obj%data(col_s)
                  nze = row_size*col_size
                  CALL set_larnv_seed(INT(row_s), INT(nrow), INT(col_s), INT(ncol), randmat_counter, iseed)
                  CALL dlarnv(1, iseed, max_nze, values)
                  CALL dbcsr_tas_put_block(matrix, row_s, col_s, values(1:row_size, 1:col_size))
               END IF
            END IF
         END DO
      END DO

      CALL dbcsr_tas_finalize(matrix)

      CALL timestop(handle)

   END SUBROUTINE

   SUBROUTINE dbcsr_tas_benchmark_mm(transa, transb, transc, matrix_a, matrix_b, matrix_c, compare_dbcsr, filter_eps, io_unit)
      !! Benchmark routine. Due to random sparsity (as opposed to structured sparsity pattern), this
      !! may not be representative for actual applications.

      CHARACTER(LEN=1), INTENT(IN)                       :: transa, transb, transc
      TYPE(dbcsr_tas_type), INTENT(INOUT)                  :: matrix_a, matrix_b, matrix_c

      REAL(KIND=real_8), INTENT(IN), OPTIONAL            :: filter_eps
      INTEGER, INTENT(IN), OPTIONAL                      :: io_unit
      LOGICAL, INTENT(IN) :: compare_dbcsr

      INTEGER                                            :: handle1, handle2
      TYPE(dbcsr_type)                                   :: dbcsr_a, dbcsr_b, dbcsr_c, &
                                                            dbcsr_a_mm, dbcsr_b_mm, dbcsr_c_mm
      TYPE(mp_comm_type)                                 :: mp_comm, comm_dbcsr
      TYPE(dbcsr_distribution_obj)                       :: dist_a, dist_b, dist_c
      INTEGER, DIMENSION(2)                              :: npdims, myploc
      INTEGER, DIMENSION(:), POINTER, CONTIGUOUS         :: cd_a, cd_b, cd_c, &
                                                            rd_a, rd_b, rd_c
      TYPE(dbcsr_mp_obj)                                 :: mp_environ_tmp

      INTEGER, DIMENSION(:), POINTER, CONTIGUOUS         :: row_blk_size, col_blk_size

      IF (PRESENT(io_unit)) THEN
      IF (io_unit > 0) THEN
         WRITE (io_unit, "(A)") "starting tall-and-skinny benchmark"
      END IF
      END IF
      CALL timeset("benchmark_tas_mm", handle1)
      CALL dbcsr_tas_multiply(transa, transb, transc, dbcsr_scalar(1.0_real_8), matrix_a, matrix_b, &
                              dbcsr_scalar(0.0_real_8), matrix_c, &
                              filter_eps=filter_eps, unit_nr=io_unit)
      CALL timestop(handle1)
      IF (PRESENT(io_unit)) THEN
      IF (io_unit > 0) THEN
         WRITE (io_unit, "(A)") "tall-and-skinny benchmark completed"
      END IF
      END IF

      IF (compare_dbcsr) THEN
         CALL dbcsr_tas_convert_to_dbcsr(matrix_a, dbcsr_a)
         CALL dbcsr_tas_convert_to_dbcsr(matrix_b, dbcsr_b)
         CALL dbcsr_tas_convert_to_dbcsr(matrix_c, dbcsr_c)

         CALL dbcsr_tas_get_split_info(dbcsr_tas_info(matrix_a), mp_comm=mp_comm)
         npdims(:) = 0
         CALL mp_cart_create(mp_comm, 2, npdims, myploc, comm_dbcsr)

         ALLOCATE (rd_a(dbcsr_nblkrows_total(dbcsr_a))); ALLOCATE (cd_a(dbcsr_nblkcols_total(dbcsr_a)))
         ALLOCATE (rd_b(dbcsr_nblkrows_total(dbcsr_b))); ALLOCATE (cd_b(dbcsr_nblkcols_total(dbcsr_b)))
         ALLOCATE (rd_c(dbcsr_nblkrows_total(dbcsr_c))); ALLOCATE (cd_c(dbcsr_nblkcols_total(dbcsr_c)))
         CALL dbcsr_tas_default_distvec(INT(dbcsr_nblkrows_total(dbcsr_a)), npdims(1), dbcsr_row_block_sizes(dbcsr_a), rd_a)
         CALL dbcsr_tas_default_distvec(INT(dbcsr_nblkcols_total(dbcsr_a)), npdims(2), dbcsr_col_block_sizes(dbcsr_a), cd_a)
         CALL dbcsr_tas_default_distvec(INT(dbcsr_nblkrows_total(dbcsr_b)), npdims(1), dbcsr_row_block_sizes(dbcsr_b), rd_b)
         CALL dbcsr_tas_default_distvec(INT(dbcsr_nblkcols_total(dbcsr_b)), npdims(2), dbcsr_col_block_sizes(dbcsr_b), cd_b)
         CALL dbcsr_tas_default_distvec(INT(dbcsr_nblkrows_total(dbcsr_c)), npdims(1), dbcsr_row_block_sizes(dbcsr_c), rd_c)
         CALL dbcsr_tas_default_distvec(INT(dbcsr_nblkcols_total(dbcsr_c)), npdims(2), dbcsr_col_block_sizes(dbcsr_c), cd_c)

         mp_environ_tmp = dbcsr_mp_environ(comm_dbcsr)
         CALL dbcsr_distribution_new(dist_a, mp_environ_tmp, rd_a, cd_a, reuse_arrays=.TRUE.)
         CALL dbcsr_distribution_new(dist_b, mp_environ_tmp, rd_b, cd_b, reuse_arrays=.TRUE.)
         CALL dbcsr_distribution_new(dist_c, mp_environ_tmp, rd_c, cd_c, reuse_arrays=.TRUE.)
         CALL dbcsr_mp_release(mp_environ_tmp)

         row_blk_size => dbcsr_row_block_sizes(dbcsr_a)
         col_blk_size => dbcsr_col_block_sizes(dbcsr_a)
         CALL dbcsr_create(matrix=dbcsr_a_mm, name=dbcsr_a%name, dist=dist_a, matrix_type=dbcsr_type_no_symmetry, &
                           row_blk_size=row_blk_size, col_blk_size=col_blk_size, &
                           data_type=dbcsr_type_real_8)
         row_blk_size => dbcsr_row_block_sizes(dbcsr_b)
         col_blk_size => dbcsr_col_block_sizes(dbcsr_b)
         CALL dbcsr_create(matrix=dbcsr_b_mm, name=dbcsr_b%name, dist=dist_b, matrix_type=dbcsr_type_no_symmetry, &
                           row_blk_size=row_blk_size, col_blk_size=col_blk_size, &
                           data_type=dbcsr_type_real_8)
         row_blk_size => dbcsr_row_block_sizes(dbcsr_c)
         col_blk_size => dbcsr_col_block_sizes(dbcsr_c)
         CALL dbcsr_create(matrix=dbcsr_c_mm, name=dbcsr_c%name, dist=dist_c, matrix_type=dbcsr_type_no_symmetry, &
                           row_blk_size=row_blk_size, col_blk_size=col_blk_size, &
                           data_type=dbcsr_type_real_8)

         CALL dbcsr_finalize(dbcsr_a_mm)
         CALL dbcsr_finalize(dbcsr_b_mm)
         CALL dbcsr_finalize(dbcsr_c_mm)

         CALL dbcsr_complete_redistribute(dbcsr_a, dbcsr_a_mm)
         CALL dbcsr_complete_redistribute(dbcsr_b, dbcsr_b_mm)
         IF (PRESENT(io_unit)) THEN
         IF (io_unit > 0) THEN
            WRITE (io_unit, "(A)") "starting dbcsr benchmark"
         END IF
         END IF
         CALL timeset("benchmark_dbcsr_mm", handle2)
         CALL dbcsr_multiply(transa, transb, dbcsr_scalar(1.0_real_8), dbcsr_a_mm, dbcsr_b_mm, &
                             dbcsr_scalar(0.0_real_8), dbcsr_c_mm, filter_eps=filter_eps)
         CALL timestop(handle2)
         IF (PRESENT(io_unit)) THEN
         IF (io_unit > 0) THEN
            WRITE (io_unit, "(A)") "dbcsr benchmark completed"
         END IF
         END IF

         CALL dbcsr_release(dbcsr_a)
         CALL dbcsr_release(dbcsr_b)
         CALL dbcsr_release(dbcsr_c)
         CALL dbcsr_release(dbcsr_a_mm)
         CALL dbcsr_release(dbcsr_b_mm)
         CALL dbcsr_release(dbcsr_c_mm)
         CALL dbcsr_distribution_release(dist_a)
         CALL dbcsr_distribution_release(dist_b)
         CALL dbcsr_distribution_release(dist_c)

         CALL mp_comm_free(comm_dbcsr)
      END IF

   END SUBROUTINE

   SUBROUTINE dbcsr_tas_test_mm(transa, transb, transc, matrix_a, matrix_b, matrix_c, filter_eps, unit_nr, log_verbose)
      !! Test tall-and-skinny matrix multiplication for accuracy
      CHARACTER(LEN=1), INTENT(IN)                       :: transa, transb, transc
      TYPE(dbcsr_tas_type), INTENT(INOUT)                  :: matrix_a, matrix_b, matrix_c
      INTEGER, INTENT(IN)                                :: unit_nr
      LOGICAL, INTENT(IN), OPTIONAL                      :: log_verbose
      REAL(KIND=real_8), INTENT(IN), OPTIONAL            :: filter_eps

      CHARACTER(LEN=1)                                   :: transa_prv, transb_prv
      INTEGER                                            :: io_unit, mynode, &
                                                            numnodes
      INTEGER, DIMENSION(2)                              :: myploc, npdims
      INTEGER, DIMENSION(:), POINTER, CONTIGUOUS         :: cd_a, cd_b, cd_c, &
                                                            rd_a, rd_b, rd_c
      REAL(KIND=real_8)                                  :: norm, rc_cs, sq_cs
      TYPE(dbcsr_distribution_obj)                       :: dist_a, dist_b, dist_c
      TYPE(dbcsr_mp_obj)                                 :: mp_environ_tmp
      TYPE(dbcsr_type)                                   :: dbcsr_a, dbcsr_a_mm, dbcsr_b, &
                                                            dbcsr_b_mm, dbcsr_c, dbcsr_c_mm, &
                                                            dbcsr_c_mm_check
      TYPE(mp_comm_type)                                 :: comm_dbcsr, mp_comm
      REAL(KIND=real_8), PARAMETER :: test_tol = 1.0E-10_real_8

      CALL dbcsr_tas_get_split_info(dbcsr_tas_info(matrix_a), mp_comm=mp_comm)
      CALL mp_environ(numnodes, mynode, mp_comm)
      io_unit = -1
      IF (mynode .EQ. 0) io_unit = unit_nr

      CALL dbcsr_tas_multiply(transa, transb, transc, dbcsr_scalar(1.0_real_8), matrix_a, matrix_b, &
                              dbcsr_scalar(0.0_real_8), matrix_c, &
                              filter_eps=filter_eps, unit_nr=io_unit, log_verbose=log_verbose, optimize_dist=.TRUE.)

      CALL dbcsr_tas_convert_to_dbcsr(matrix_a, dbcsr_a)
      CALL dbcsr_tas_convert_to_dbcsr(matrix_b, dbcsr_b)
      CALL dbcsr_tas_convert_to_dbcsr(matrix_c, dbcsr_c)

      npdims(:) = 0
      CALL mp_cart_create(mp_comm, 2, npdims, myploc, comm_dbcsr)

      ALLOCATE (rd_a(dbcsr_nblkrows_total(dbcsr_a))); ALLOCATE (cd_a(dbcsr_nblkcols_total(dbcsr_a)))
      ALLOCATE (rd_b(dbcsr_nblkrows_total(dbcsr_b))); ALLOCATE (cd_b(dbcsr_nblkcols_total(dbcsr_b)))
      ALLOCATE (rd_c(dbcsr_nblkrows_total(dbcsr_c))); ALLOCATE (cd_c(dbcsr_nblkcols_total(dbcsr_c)))
      CALL dbcsr_tas_default_distvec(INT(dbcsr_nblkrows_total(dbcsr_a)), npdims(1), dbcsr_row_block_sizes(dbcsr_a), rd_a)
      CALL dbcsr_tas_default_distvec(INT(dbcsr_nblkcols_total(dbcsr_a)), npdims(2), dbcsr_col_block_sizes(dbcsr_a), cd_a)
      CALL dbcsr_tas_default_distvec(INT(dbcsr_nblkrows_total(dbcsr_b)), npdims(1), dbcsr_row_block_sizes(dbcsr_b), rd_b)
      CALL dbcsr_tas_default_distvec(INT(dbcsr_nblkcols_total(dbcsr_b)), npdims(2), dbcsr_col_block_sizes(dbcsr_b), cd_b)
      CALL dbcsr_tas_default_distvec(INT(dbcsr_nblkrows_total(dbcsr_c)), npdims(1), dbcsr_row_block_sizes(dbcsr_c), rd_c)
      CALL dbcsr_tas_default_distvec(INT(dbcsr_nblkcols_total(dbcsr_c)), npdims(2), dbcsr_col_block_sizes(dbcsr_c), cd_c)

      mp_environ_tmp = dbcsr_mp_environ(comm_dbcsr)
      CALL dbcsr_distribution_new(dist_a, mp_environ_tmp, rd_a, cd_a, reuse_arrays=.TRUE.)
      CALL dbcsr_distribution_new(dist_b, mp_environ_tmp, rd_b, cd_b, reuse_arrays=.TRUE.)
      CALL dbcsr_distribution_new(dist_c, mp_environ_tmp, rd_c, cd_c, reuse_arrays=.TRUE.)
      CALL dbcsr_mp_release(mp_environ_tmp)

      CALL dbcsr_create(matrix=dbcsr_a_mm, name="matrix a", dist=dist_a, matrix_type=dbcsr_type_no_symmetry, &
                        row_blk_size_obj=dbcsr_a%row_blk_size, col_blk_size_obj=dbcsr_a%col_blk_size, &
                        data_type=dbcsr_type_real_8)

      CALL dbcsr_create(matrix=dbcsr_b_mm, name="matrix b", dist=dist_b, matrix_type=dbcsr_type_no_symmetry, &
                        row_blk_size_obj=dbcsr_b%row_blk_size, col_blk_size_obj=dbcsr_b%col_blk_size, &
                        data_type=dbcsr_type_real_8)

      CALL dbcsr_create(matrix=dbcsr_c_mm, name="matrix c", dist=dist_c, matrix_type=dbcsr_type_no_symmetry, &
                        row_blk_size_obj=dbcsr_c%row_blk_size, col_blk_size_obj=dbcsr_c%col_blk_size, &
                        data_type=dbcsr_type_real_8)

      CALL dbcsr_create(matrix=dbcsr_c_mm_check, name="matrix c check", dist=dist_c, matrix_type=dbcsr_type_no_symmetry, &
                        row_blk_size_obj=dbcsr_c%row_blk_size, col_blk_size_obj=dbcsr_c%col_blk_size, &
                        data_type=dbcsr_type_real_8)

      CALL dbcsr_finalize(dbcsr_a_mm)
      CALL dbcsr_finalize(dbcsr_b_mm)
      CALL dbcsr_finalize(dbcsr_c_mm)
      CALL dbcsr_finalize(dbcsr_c_mm_check)

      CALL dbcsr_complete_redistribute(dbcsr_a, dbcsr_a_mm)
      CALL dbcsr_complete_redistribute(dbcsr_b, dbcsr_b_mm)
      CALL dbcsr_complete_redistribute(dbcsr_c, dbcsr_c_mm_check)

      transa_prv = transa; transb_prv = transb

      IF (transc == dbcsr_no_transpose) THEN
         CALL dbcsr_multiply(transa_prv, transb_prv, dbcsr_scalar(1.0_real_8), &
                             dbcsr_a_mm, dbcsr_b_mm, dbcsr_scalar(0.0_real_8), dbcsr_c_mm, filter_eps=filter_eps)
      ELSEIF (transc == dbcsr_transpose) THEN
         CALL invert_transpose_flag(transa_prv)
         CALL invert_transpose_flag(transb_prv)
         CALL dbcsr_multiply(transb_prv, transa_prv, dbcsr_scalar(1.0_real_8), &
                             dbcsr_b_mm, dbcsr_a_mm, dbcsr_scalar(0.0_real_8), dbcsr_c_mm, filter_eps=filter_eps)
      END IF

      sq_cs = dbcsr_checksum(dbcsr_c_mm)
      rc_cs = dbcsr_checksum(dbcsr_c_mm_check)
      CALL dbcsr_add(dbcsr_c_mm_check, dbcsr_c_mm, -1.0_real_8, 1.0_real_8)
      norm = dbcsr_maxabs(dbcsr_c_mm_check)

      IF (io_unit > 0) THEN
      IF (ABS(norm) .GT. test_tol) THEN
         WRITE (io_unit, '(A, A, A, A, A, 1X, A)') TRIM(matrix_a%matrix%name), transa, ' X ', TRIM(matrix_b%matrix%name), &
            transb, 'failed!'
         WRITE (io_unit, "(A,1X,E9.2,1X,E9.2)") "checksums", sq_cs, rc_cs
         WRITE (io_unit, "(A,1X,E9.2)") "difference norm", norm
         DBCSR_ABORT("")
      ELSE
         WRITE (io_unit, '(A, A, A, A, A, 1X, A)') TRIM(matrix_a%matrix%name), transa, ' X ', TRIM(matrix_b%matrix%name), &
            transb, 'passed!'
         WRITE (io_unit, "(A,1X,E9.2,1X,E9.2)") "checksums", sq_cs, rc_cs
         WRITE (io_unit, "(A,1X,E9.2)") "difference norm", norm
      END IF
      END IF

      CALL dbcsr_release(dbcsr_a)
      CALL dbcsr_release(dbcsr_a_mm)
      CALL dbcsr_release(dbcsr_b)
      CALL dbcsr_release(dbcsr_b_mm)
      CALL dbcsr_release(dbcsr_c)
      CALL dbcsr_release(dbcsr_c_mm)
      CALL dbcsr_release(dbcsr_c_mm_check)

      CALL dbcsr_distribution_release(dist_a)
      CALL dbcsr_distribution_release(dist_b)
      CALL dbcsr_distribution_release(dist_c)

      CALL mp_comm_free(comm_dbcsr)

   END SUBROUTINE

   FUNCTION dbcsr_tas_checksum(matrix, local, pos)
      !! Calculate checksum of tall-and-skinny matrix consistent with dbcsr_checksum
      TYPE(dbcsr_tas_type), INTENT(IN) :: matrix
      LOGICAL, INTENT(IN), OPTIONAL  :: local, pos
      TYPE(dbcsr_type)               :: dbcsr_m
      REAL(KIND=real_8)              :: dbcsr_tas_checksum

      CALL dbcsr_tas_convert_to_dbcsr(matrix, dbcsr_m)
      dbcsr_tas_checksum = dbcsr_checksum(dbcsr_m, local, pos)
      CALL dbcsr_release(dbcsr_m)
   END FUNCTION

   SUBROUTINE dbcsr_tas_random_bsizes(sizes, repeat, block_sizes)
      !! Create random block sizes
      INTEGER, DIMENSION(:), INTENT(IN)                  :: sizes
      INTEGER, INTENT(IN)                                :: repeat
      INTEGER, DIMENSION(:), INTENT(OUT)                 :: block_sizes

      INTEGER                                            :: d, size_i

      DO d = 1, SIZE(block_sizes)
         size_i = MOD((d - 1)/repeat, SIZE(sizes)) + 1
         block_sizes(d) = sizes(size_i)
      END DO
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_reset_randmat_seed()
      !! Reset the seed used for generating random matrices to default value
      randmat_counter = rand_seed_init
   END SUBROUTINE

END MODULE
