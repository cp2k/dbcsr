!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tas_mm
   !! Matrix multiplication for tall-and-skinny matrices. This uses the k-split (non-recursive) CARMA
   !! algorithm that is communication-optimal as long as the two smaller dimensions have
   !! the same size.
   !! Submatrices are obtained by splitting a dimension of the process grid. Multiplication of
   !! submatrices uses DBCSR Cannon algorithm. Due to unknown sparsity pattern of result matrix, parameters
   !! (group sizes and process grid dimensions) can not be derived from matrix dimensions and need to be
   !! set manually.

   #:include "dbcsr_tas.fypp"

   USE dbcsr_data_methods, ONLY: &
      dbcsr_scalar_zero, dbcsr_scalar, dbcsr_scalar_multiply
   USE dbcsr_data_types, ONLY: &
      dbcsr_scalar_type, dbcsr_type_real_8, dbcsr_type_real_4, dbcsr_type_complex_8, dbcsr_type_complex_4
   USE dbcsr_multiply_api, ONLY: dbcsr_multiply
   USE dbcsr_tas_base, ONLY: &
      dbcsr_tas_create, dbcsr_tas_destroy, dbcsr_tas_distribution_destroy, dbcsr_tas_distribution_new, &
      dbcsr_tas_get_data_type, dbcsr_tas_info, dbcsr_tas_nblkcols_total, &
      dbcsr_tas_nblkrows_total, dbcsr_tas_filter, dbcsr_tas_get_info, dbcsr_tas_iterator_blocks_left, &
      dbcsr_tas_get_nze_total, dbcsr_tas_reserve_blocks, dbcsr_tas_iterator_start, dbcsr_tas_iterator_next_block, &
      dbcsr_tas_iterator_stop, dbcsr_tas_copy, dbcsr_tas_get_block_p, dbcsr_tas_clear, dbcsr_tas_get_num_blocks, &
      dbcsr_tas_nfullrows_total, dbcsr_tas_nfullcols_total
   USE dbcsr_tas_types, ONLY: &
      dbcsr_tas_distribution_type, dbcsr_tas_split_info, dbcsr_tas_type, dbcsr_tas_iterator
   USE dbcsr_tas_global, ONLY: &
      dbcsr_tas_dist_cyclic, dbcsr_tas_dist_arb, dbcsr_tas_distribution, dbcsr_tas_dist_arb_default, &
      dbcsr_tas_rowcol_data, dbcsr_tas_blk_size_one, dbcsr_tas_default_distvec
   USE dbcsr_tas_reshape_ops, ONLY: &
      dbcsr_tas_merge, dbcsr_tas_replicate, dbcsr_tas_reshape
   USE dbcsr_tas_split, ONLY: &
      rowsplit, colsplit, dbcsr_tas_get_split_info, dbcsr_tas_create_split, dbcsr_tas_mp_comm, &
      dbcsr_tas_release_info, accept_pgrid_dims, dbcsr_tas_info_hold, default_nsplit_accept_ratio
   USE dbcsr_tas_util, ONLY: &
      swap, invert_transpose_flag, array_eq, dbcsr_mp_environ
   USE dbcsr_types, ONLY: &
      dbcsr_no_transpose, dbcsr_transpose, dbcsr_type, dbcsr_distribution_obj, dbcsr_mp_obj, &
      dbcsr_type_no_symmetry
   USE dbcsr_kinds, ONLY: &
      int_8, real_8, real_4, default_string_length
   USE dbcsr_mpiwrap, ONLY: &
      mp_environ, mp_sum, mp_comm_free, mp_cart_create, mp_max, mp_sync, mp_comm_type
   USE dbcsr_operations, ONLY: &
      dbcsr_scale, dbcsr_get_info, dbcsr_copy, dbcsr_clear, dbcsr_add, dbcsr_zero
   USE dbcsr_tas_io, ONLY: &
      dbcsr_tas_write_dist, dbcsr_tas_write_matrix_info, dbcsr_tas_write_split_info, prep_output_unit
   USE dbcsr_work_operations, ONLY: dbcsr_create, dbcsr_finalize
   USE dbcsr_transformations, ONLY: dbcsr_redistribute
   USE dbcsr_dist_methods, ONLY: dbcsr_distribution_new
   USE dbcsr_methods, ONLY: &
      dbcsr_mp_release, dbcsr_release, dbcsr_distribution_release, dbcsr_get_nze, dbcsr_nfullrows_total, dbcsr_nfullcols_total
   USE dbcsr_config, ONLY: dbcsr_cfg
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tas_mm'

   PUBLIC :: &
      dbcsr_tas_multiply, &
      dbcsr_tas_batched_mm_init, &
      dbcsr_tas_batched_mm_finalize, &
      dbcsr_tas_result_index, &
      dbcsr_tas_set_batched_state, &
      dbcsr_tas_batched_mm_complete

CONTAINS

   RECURSIVE SUBROUTINE dbcsr_tas_multiply(transa, transb, transc, alpha, matrix_a, matrix_b, beta, matrix_c, &
                                           optimize_dist, split_opt, filter_eps, flop, move_data_a, &
                                           move_data_b, retain_sparsity, simple_split, result_index, unit_nr, log_verbose)
      !! tall-and-skinny matrix-matrix multiplication. Undocumented dummy arguments are identical to
      !! arguments of dbcsr_multiply (see dbcsr_mm, dbcsr_multiply_generic).

      CHARACTER(LEN=1), INTENT(IN)               :: transa, transb, transc
      TYPE(dbcsr_scalar_type), INTENT(IN)        :: alpha, beta
      TYPE(dbcsr_tas_type), TARGET, &
         INTENT(INOUT)                           :: matrix_a, matrix_b, matrix_c
      LOGICAL, INTENT(IN), OPTIONAL              :: optimize_dist
         !! Whether distribution should be optimized internally. In the current implementation this guarantees optimal parameters
         !! only for dense matrices.
      TYPE(dbcsr_tas_split_info), INTENT(OUT), &
         OPTIONAL                       :: split_opt
         !! optionally return split info containing optimal grid and split parameters. This can be used to choose optimal process
         !! grids for subsequent matrix multiplications with matrices of similar shape and sparsity.
      REAL(KIND=real_8), INTENT(IN), OPTIONAL    :: filter_eps
      INTEGER(KIND=int_8), INTENT(OUT), OPTIONAL :: flop
      LOGICAL, INTENT(IN), OPTIONAL              :: move_data_a, move_data_b, simple_split, retain_sparsity
         !! memory optimization: move data to matrix_c such that matrix_a is empty on return
         !! memory optimization: move data to matrix_c such that matrix_b is empty on return
         !! for internal use only
      INTEGER(int_8), DIMENSION(:, :), ALLOCATABLE, INTENT(OUT), OPTIONAL :: result_index
      INTEGER, OPTIONAL, INTENT(IN)              :: unit_nr
         !! unit number for logging output
      LOGICAL, OPTIONAL, INTENT(IN)              :: log_verbose
         !! only for testing: verbose output

      TYPE(dbcsr_tas_type), POINTER              :: matrix_b_rs, matrix_a_rs, matrix_c_rs, &
                                                    matrix_c_rep, matrix_b_rep, matrix_a_rep

      REAL(KIND=real_8)                          :: filter_eps_prv
      INTEGER(KIND=int_8), DIMENSION(2)          :: dims_a, dims_b, dims_c
      INTEGER, DIMENSION(2)                      :: pdims, pcoord, pcoord_sub, pdims_sub
      INTEGER(KIND=int_8), DIMENSION(3)          :: dims
      INTEGER                                    :: max_mm_dim, data_type, handle, handle2, handle3, handle4, &
                                                    unit_nr_prv, nsplit, nsplit_opt, numproc, numproc_sub, iproc, &
                                                    split_rc, split_a, split_b, split_c, &
                                                    batched_repl, max_mm_dim_batched, nsplit_batched
      CHARACTER(LEN=1)                           :: tr_case, transa_prv, transb_prv, transc_prv
      TYPE(dbcsr_scalar_type)                    :: zero
      LOGICAL                                    :: new_a, new_b, new_c, simple_split_prv, opt_pgrid, &
                                                    move_a, move_b, do_batched, &
                                                    nodata_3
      TYPE(dbcsr_tas_split_info)                 :: info, info_a, info_b, info_c
      CHARACTER(LEN=*), PARAMETER                :: routineN = 'dbcsr_tas_multiply'
      INTEGER(KIND=int_8)                        :: nze_a, nze_b, nze_c, nze_c_sum
      TYPE(dbcsr_type)                           :: matrix_a_mm, matrix_b_mm, matrix_c_mm
      TYPE(mp_comm_type)                         :: mp_comm, comm_tmp, mp_comm_group, mp_comm_mm, mp_comm_opt

      CALL timeset(routineN, handle)
      CALL mp_sync(matrix_a%dist%info%mp_comm)
      CALL timeset("dbcsr_tas_total", handle2)

      NULLIFY (matrix_b_rs, matrix_a_rs, matrix_c_rs)

      unit_nr_prv = prep_output_unit(unit_nr)

      IF (PRESENT(simple_split)) THEN
         simple_split_prv = simple_split
      ELSE
         simple_split_prv = .FALSE.

         info_a = dbcsr_tas_info(matrix_a); info_b = dbcsr_tas_info(matrix_b); info_c = dbcsr_tas_info(matrix_c)
         IF (info_a%strict_split(1) .OR. info_b%strict_split(1) .OR. info_c%strict_split(1)) simple_split_prv = .TRUE.
      END IF

      nodata_3 = .TRUE.
      IF (PRESENT(retain_sparsity)) THEN
         IF (retain_sparsity) nodata_3 = .FALSE.
      END IF

      ! get prestored info for multiplication strategy in case of batched mm
      batched_repl = 0
      do_batched = .FALSE.
      IF (matrix_a%do_batched > 0) THEN
         do_batched = .TRUE.
         IF (matrix_a%do_batched == 3) THEN
            DBCSR_ASSERT(batched_repl == 0)
            batched_repl = 1
            CALL dbcsr_tas_get_split_info( &
               dbcsr_tas_info(matrix_a%mm_storage%store_batched_repl), &
               nsplit=nsplit_batched)
            DBCSR_ASSERT(nsplit_batched > 0)
            max_mm_dim_batched = 3
         END IF
      END IF

      IF (matrix_b%do_batched > 0) THEN
         do_batched = .TRUE.
         IF (matrix_b%do_batched == 3) THEN
            DBCSR_ASSERT(batched_repl == 0)
            batched_repl = 2
            CALL dbcsr_tas_get_split_info( &
               dbcsr_tas_info(matrix_b%mm_storage%store_batched_repl), &
               nsplit=nsplit_batched)
            DBCSR_ASSERT(nsplit_batched > 0)
            max_mm_dim_batched = 1
         END IF
      END IF

      IF (matrix_c%do_batched > 0) THEN
         do_batched = .TRUE.
         IF (matrix_c%do_batched == 3) THEN
            DBCSR_ASSERT(batched_repl == 0)
            batched_repl = 3
            CALL dbcsr_tas_get_split_info( &
               dbcsr_tas_info(matrix_c%mm_storage%store_batched_repl), &
               nsplit=nsplit_batched)
            DBCSR_ASSERT(nsplit_batched > 0)
            max_mm_dim_batched = 2
         END IF
      END IF

      move_a = .FALSE.
      move_b = .FALSE.

      IF (PRESENT(move_data_a)) move_a = move_data_a
      IF (PRESENT(move_data_b)) move_b = move_data_b

      IF (.NOT. dbcsr_tas_get_data_type(matrix_a) .EQ. dbcsr_tas_get_data_type(matrix_b)) THEN
         DBCSR_ABORT("matrices must have same datatype")
      END IF

      data_type = dbcsr_tas_get_data_type(matrix_a)

      transa_prv = transa; transb_prv = transb; transc_prv = transc

      dims_a = [dbcsr_tas_nblkrows_total(matrix_a), dbcsr_tas_nblkcols_total(matrix_a)]
      dims_b = [dbcsr_tas_nblkrows_total(matrix_b), dbcsr_tas_nblkcols_total(matrix_b)]
      dims_c = [dbcsr_tas_nblkrows_total(matrix_c), dbcsr_tas_nblkcols_total(matrix_c)]

      IF (unit_nr_prv .GT. 0) THEN
         WRITE (unit_nr_prv, '(A)') repeat("-", 80)
         WRITE (unit_nr_prv, '(A,1X,A,1X,A,1X,A,1X,A,1X,A)') "DBCSR TAS MATRIX MULTIPLICATION:", &
            TRIM(matrix_a%matrix%name), 'x', TRIM(matrix_b%matrix%name), '=', TRIM(matrix_c%matrix%name)
         WRITE (unit_nr_prv, '(A)') repeat("-", 80)
      END IF
      IF (do_batched) THEN
         IF (unit_nr_prv > 0) THEN
            WRITE (unit_nr_prv, "(T2,A)") &
               "BATCHED PROCESSING OF MATMUL"
            IF (batched_repl > 0) THEN
               WRITE (unit_nr_prv, "(T4,A,T80,I1)") "reusing replicated matrix:", batched_repl
            END IF
         END IF
      END IF

      IF (transa_prv .EQ. dbcsr_transpose) THEN
         CALL swap(dims_a)
      END IF

      IF (transb_prv .EQ. dbcsr_transpose) THEN
         CALL swap(dims_b)
      END IF

      dims_c = [dims_a(1), dims_b(2)]

      IF (.NOT. (dims_a(2) .EQ. dims_b(1))) THEN
         DBCSR_ABORT("inconsistent matrix dimensions")
      END IF

      dims(:) = [dims_a(1), dims_a(2), dims_b(2)]

      tr_case = ''

      IF (unit_nr_prv > 0) THEN
         WRITE (unit_nr_prv, "(T2,A, 1X, I12, 1X, I12, 1X, I12)") "mm dims:", dims(1), dims(2), dims(3)
      END IF

      CALL dbcsr_tas_get_split_info(dbcsr_tas_info(matrix_a), mp_comm=mp_comm)
      CALL mp_environ(numproc, iproc, mp_comm)

      ! derive optimal matrix layout and split factor from occupancies
      nze_a = dbcsr_tas_get_nze_total(matrix_a)
      nze_b = dbcsr_tas_get_nze_total(matrix_b)

      IF (.NOT. simple_split_prv) THEN
         CALL dbcsr_tas_result_index(transa, transb, transc, matrix_a, matrix_b, matrix_c, filter_eps, &
                                     blk_ind=result_index, nze=nze_c, retain_sparsity=retain_sparsity)

         IF (PRESENT(result_index)) THEN
            CALL mp_sync(matrix_a%dist%info%mp_comm)
            CALL timestop(handle2)
            CALL timestop(handle)
            RETURN
         END IF

         max_mm_dim = MAXLOC(dims, 1)
         nsplit = split_factor_estimate(max_mm_dim, nze_a, nze_b, nze_c, numproc)
         nsplit_opt = nsplit

         IF (unit_nr_prv > 0) THEN
            WRITE (unit_nr_prv, "(T2,A)") &
               "MM PARAMETERS"
            WRITE (unit_nr_prv, "(T4,A,T68,I13)") "Est. number of matrix elements per CPU of result matrix:", &
               (nze_c + numproc - 1)/numproc

            WRITE (unit_nr_prv, "(T4,A,T68,I13)") "Est. optimal split factor:", nsplit
         END IF

      ELSEIF (batched_repl > 0) THEN
         nsplit = nsplit_batched
         nsplit_opt = nsplit
         max_mm_dim = max_mm_dim_batched
         IF (unit_nr_prv > 0) THEN
            WRITE (unit_nr_prv, "(T2,A)") &
               "MM PARAMETERS"
            WRITE (unit_nr_prv, "(T4,A,T68,I13)") "Est. optimal split factor:", nsplit
         END IF

      ELSE
         nsplit = 0
         max_mm_dim = MAXLOC(dims, 1)
      END IF

      ! reshape matrices to the optimal layout and split factor
      split_a = rowsplit; split_b = rowsplit; split_c = rowsplit
      SELECT CASE (max_mm_dim)
      CASE (1)

         split_a = rowsplit; split_c = rowsplit
         CALL reshape_mm_compatible(matrix_a, matrix_c, matrix_a_rs, matrix_c_rs, &
                                    new_a, new_c, transa_prv, transc_prv, optimize_dist=optimize_dist, &
                                    nsplit=nsplit, &
                                    opt_nsplit=batched_repl == 0, &
                                    split_rc_1=split_a, split_rc_2=split_c, &
                                    nodata2=nodata_3, comm_new=comm_tmp, &
                                    move_data_1=move_a, unit_nr=unit_nr_prv)

         info = dbcsr_tas_info(matrix_a_rs)
         CALL dbcsr_tas_get_split_info(info, split_rowcol=split_rc, mp_comm=mp_comm)

         new_b = .FALSE.
         IF (matrix_b%do_batched <= 2) THEN
            ALLOCATE (matrix_b_rs)
            CALL reshape_mm_small(mp_comm, matrix_b, matrix_b_rs, transb_prv == dbcsr_transpose, transb_prv, move_data=move_b)
            new_b = .TRUE.
         END IF

         tr_case = transa_prv

         IF (unit_nr_prv > 0) THEN
            IF (tr_case == 'N') THEN
               WRITE (unit_nr_prv, "(T2,A, 1X, A)") "mm case:", "| x + = |"
            ELSE
               WRITE (unit_nr_prv, "(T2,A, 1X, A)") "mm case:", "--T x + = --T"
            END IF
         END IF

      CASE (2)

         split_a = colsplit; split_b = rowsplit
         CALL reshape_mm_compatible(matrix_a, matrix_b, matrix_a_rs, matrix_b_rs, new_a, new_b, transa_prv, transb_prv, &
                                    optimize_dist=optimize_dist, &
                                    nsplit=nsplit, &
                                    opt_nsplit=batched_repl == 0, &
                                    split_rc_1=split_a, split_rc_2=split_b, &
                                    comm_new=comm_tmp, &
                                    move_data_1=move_a, move_data_2=move_b, unit_nr=unit_nr_prv)

         info = dbcsr_tas_info(matrix_a_rs)
         CALL dbcsr_tas_get_split_info(info, split_rowcol=split_rc, mp_comm=mp_comm)

         IF (matrix_c%do_batched == 1) THEN
            matrix_c%mm_storage%batched_beta = beta
         ELSEIF (matrix_c%do_batched > 1) THEN
            matrix_c%mm_storage%batched_beta = &
               dbcsr_scalar_multiply(matrix_c%mm_storage%batched_beta, beta)
         END IF

         IF (matrix_c%do_batched <= 2) THEN
            ALLOCATE (matrix_c_rs)
            CALL reshape_mm_small(mp_comm, matrix_c, matrix_c_rs, transc_prv == dbcsr_transpose, transc_prv, nodata=nodata_3)

            ! just leave sparsity structure for retain sparsity but no values
            IF (.NOT. nodata_3) CALL dbcsr_zero(matrix_c_rs%matrix)

            IF (matrix_c%do_batched >= 1) matrix_c%mm_storage%store_batched => matrix_c_rs
         ELSEIF (matrix_c%do_batched == 3) THEN
            matrix_c_rs => matrix_c%mm_storage%store_batched
         END IF

         new_c = matrix_c%do_batched == 0
         tr_case = transa_prv

         IF (unit_nr_prv > 0) THEN
            IF (tr_case == 'N') THEN
               WRITE (unit_nr_prv, "(T2,A, 1X, A)") "mm case:", "-- x --T = +"
            ELSE
               WRITE (unit_nr_prv, "(T2,A, 1X, A)") "mm case:", "|T x | = +"
            END IF
         END IF

      CASE (3)

         split_b = colsplit; split_c = colsplit
         CALL reshape_mm_compatible(matrix_b, matrix_c, matrix_b_rs, matrix_c_rs, new_b, new_c, transb_prv, &
                                    transc_prv, optimize_dist=optimize_dist, &
                                    nsplit=nsplit, &
                                    opt_nsplit=batched_repl == 0, &
                                    split_rc_1=split_b, split_rc_2=split_c, &
                                    nodata2=nodata_3, comm_new=comm_tmp, &
                                    move_data_1=move_b, unit_nr=unit_nr_prv)
         info = dbcsr_tas_info(matrix_b_rs)
         CALL dbcsr_tas_get_split_info(info, split_rowcol=split_rc, mp_comm=mp_comm)

         new_a = .FALSE.
         IF (matrix_a%do_batched <= 2) THEN
            ALLOCATE (matrix_a_rs)
            CALL reshape_mm_small(mp_comm, matrix_a, matrix_a_rs, transa_prv == dbcsr_transpose, transa_prv, move_data=move_a)
            new_a = .TRUE.
         END IF

         tr_case = transb_prv

         IF (unit_nr_prv > 0) THEN
            IF (tr_case == 'N') THEN
               WRITE (unit_nr_prv, "(T2,A, 1X, A)") "mm case:", "+ x -- = --"
            ELSE
               WRITE (unit_nr_prv, "(T2,A, 1X, A)") "mm case:", "+ x |T = |T"
            END IF
         END IF

      END SELECT

      CALL dbcsr_tas_get_split_info(info, nsplit=nsplit, mp_comm=mp_comm, mp_comm_group=mp_comm_group)

      CALL mp_environ(numproc, pdims, pcoord, mp_comm)
      CALL mp_environ(numproc_sub, pdims_sub, pcoord_sub, mp_comm_group)

      opt_pgrid = .NOT. accept_pgrid_dims(pdims_sub, relative=.TRUE.)

      IF (PRESENT(filter_eps)) THEN
         filter_eps_prv = filter_eps
      ELSE
         filter_eps_prv = 0.0_real_8
      END IF

      IF (unit_nr_prv /= 0) THEN
         IF (unit_nr_prv > 0) THEN
            WRITE (unit_nr_prv, "(T2, A)") "SPLIT / PARALLELIZATION INFO"
         END IF
         CALL dbcsr_tas_write_split_info(info, unit_nr_prv)
         IF (ASSOCIATED(matrix_a_rs)) CALL dbcsr_tas_write_matrix_info(matrix_a_rs, unit_nr_prv, full_info=log_verbose)
         IF (ASSOCIATED(matrix_b_rs)) CALL dbcsr_tas_write_matrix_info(matrix_b_rs, unit_nr_prv, full_info=log_verbose)
         IF (ASSOCIATED(matrix_c_rs)) CALL dbcsr_tas_write_matrix_info(matrix_c_rs, unit_nr_prv, full_info=log_verbose)
         IF (unit_nr_prv > 0) THEN
            IF (opt_pgrid) THEN
               WRITE (unit_nr_prv, "(T4, A, 1X, A)") "Change process grid:", "Yes"
            ELSE
               WRITE (unit_nr_prv, "(T4, A, 1X, A)") "Change process grid:", "No"
            END IF
         END IF
      END IF

      zero = dbcsr_scalar_zero(data_type)

      pdims = 0
      CALL mp_cart_create(mp_comm_group, 2, pdims, pcoord, mp_comm_mm)

      ! Convert DBCSR submatrices to optimized process grids and multiply
      SELECT CASE (max_mm_dim)
      CASE (1)
         IF (matrix_b%do_batched <= 2) THEN
            ALLOCATE (matrix_b_rep)
            CALL dbcsr_tas_replicate(matrix_b_rs%matrix, dbcsr_tas_info(matrix_a_rs), matrix_b_rep, move_data=.TRUE.)
            IF (matrix_b%do_batched == 1 .or. matrix_b%do_batched == 2) THEN
               matrix_b%mm_storage%store_batched_repl => matrix_b_rep
               CALL dbcsr_tas_set_batched_state(matrix_b, state=3)
            END IF
         ELSEIF (matrix_b%do_batched == 3) THEN
            matrix_b_rep => matrix_b%mm_storage%store_batched_repl
         END IF

         IF (new_b) THEN
            CALL dbcsr_tas_destroy(matrix_b_rs)
            DEALLOCATE (matrix_b_rs)
         END IF
         IF (unit_nr_prv /= 0) THEN
            CALL dbcsr_tas_write_dist(matrix_a_rs, unit_nr_prv)
            CALL dbcsr_tas_write_dist(matrix_b_rep, unit_nr_prv, full_info=log_verbose)
         END IF

         CALL convert_to_new_pgrid(mp_comm_mm, matrix_a_rs%matrix, matrix_a_mm, optimize_pgrid=opt_pgrid, move_data=move_a)

         ! keep communicators alive even after releasing TAS matrices (communicator management does not work between DBCSR and TAS)
         info_a = dbcsr_tas_info(matrix_a_rs)
         CALL dbcsr_tas_info_hold(info_a)

         IF (new_a) THEN
            CALL dbcsr_tas_destroy(matrix_a_rs)
            DEALLOCATE (matrix_a_rs)
         END IF
         CALL convert_to_new_pgrid(mp_comm_mm, matrix_b_rep%matrix, matrix_b_mm, optimize_pgrid=opt_pgrid, &
                                   move_data=matrix_b%do_batched == 0)

         info_b = dbcsr_tas_info(matrix_b_rep)
         CALL dbcsr_tas_info_hold(info_b)

         IF (matrix_b%do_batched == 0) THEN
            CALL dbcsr_tas_destroy(matrix_b_rep)
            DEALLOCATE (matrix_b_rep)
         END IF

         CALL convert_to_new_pgrid(mp_comm_mm, matrix_c_rs%matrix, matrix_c_mm, nodata=nodata_3, optimize_pgrid=opt_pgrid)

         info_c = dbcsr_tas_info(matrix_c_rs)
         CALL dbcsr_tas_info_hold(info_c)

         CALL mp_sync(matrix_a%dist%info%mp_comm)
         CALL timeset("dbcsr_tas_dbcsr", handle4)
         SELECT CASE (tr_case)
         CASE (dbcsr_no_transpose)
            CALL timeset("dbcsr_tas_mm_1N", handle3)

            CALL dbcsr_multiply(transa=dbcsr_no_transpose, transb=dbcsr_no_transpose, alpha=alpha, &
                                matrix_a=matrix_a_mm, matrix_b=matrix_b_mm, beta=beta, matrix_c=matrix_c_mm, &
                                filter_eps=filter_eps_prv, retain_sparsity=retain_sparsity, flop=flop)
            CALL timestop(handle3)
         CASE (dbcsr_transpose)
            CALL timeset("dbcsr_tas_mm_1T", handle3)
            CALL dbcsr_multiply(transa=dbcsr_transpose, transb=dbcsr_no_transpose, alpha=alpha, &
                                matrix_a=matrix_b_mm, matrix_b=matrix_a_mm, beta=beta, matrix_c=matrix_c_mm, &
                                filter_eps=filter_eps_prv, retain_sparsity=retain_sparsity, flop=flop)

            CALL timestop(handle3)
         END SELECT
         CALL mp_sync(matrix_a%dist%info%mp_comm)
         CALL timestop(handle4)

         CALL dbcsr_release(matrix_a_mm)
         CALL dbcsr_release(matrix_b_mm)

         nze_c = dbcsr_get_nze(matrix_c_mm)

         IF (.NOT. new_c) THEN
            CALL redistribute_and_sum(matrix_c_mm, matrix_c_rs%matrix, local_copy=.NOT. opt_pgrid, alpha=beta)
         ELSE
            CALL redistribute_and_sum(matrix_c_mm, matrix_c_rs%matrix, local_copy=.NOT. opt_pgrid)
         END IF

         CALL dbcsr_release(matrix_c_mm)

         IF (PRESENT(filter_eps)) CALL dbcsr_tas_filter(matrix_c_rs, filter_eps)

         IF (unit_nr_prv /= 0) THEN
            CALL dbcsr_tas_write_dist(matrix_c_rs, unit_nr_prv)
         END IF

      CASE (2)
         IF (matrix_c%do_batched <= 1) THEN
            ALLOCATE (matrix_c_rep)
            CALL dbcsr_tas_replicate(matrix_c_rs%matrix, dbcsr_tas_info(matrix_a_rs), matrix_c_rep, nodata=nodata_3)
            IF (matrix_c%do_batched == 1) THEN
               matrix_c%mm_storage%store_batched_repl => matrix_c_rep
               CALL dbcsr_tas_set_batched_state(matrix_c, state=3)
            END IF
         ELSEIF (matrix_c%do_batched == 2) THEN
            ALLOCATE (matrix_c_rep)
            CALL dbcsr_tas_replicate(matrix_c_rs%matrix, dbcsr_tas_info(matrix_a_rs), matrix_c_rep, nodata=nodata_3)
            ! just leave sparsity structure for retain sparsity but no values
            IF (.not. nodata_3) CALL dbcsr_zero(matrix_c_rep%matrix)
            matrix_c%mm_storage%store_batched_repl => matrix_c_rep
            CALL dbcsr_tas_set_batched_state(matrix_c, state=3)
         ELSEIF (matrix_c%do_batched == 3) THEN
            matrix_c_rep => matrix_c%mm_storage%store_batched_repl
         END IF

         IF (unit_nr_prv /= 0) THEN
            CALL dbcsr_tas_write_dist(matrix_a_rs, unit_nr_prv)
            CALL dbcsr_tas_write_dist(matrix_b_rs, unit_nr_prv)
         END IF

         CALL convert_to_new_pgrid(mp_comm_mm, matrix_a_rs%matrix, matrix_a_mm, optimize_pgrid=opt_pgrid, move_data=move_a)

         ! keep communicators alive even after releasing TAS matrices (communicator management does not work between DBCSR and TAS)
         info_a = dbcsr_tas_info(matrix_a_rs)
         CALL dbcsr_tas_info_hold(info_a)

         IF (new_a) THEN
            CALL dbcsr_tas_destroy(matrix_a_rs)
            DEALLOCATE (matrix_a_rs)
         END IF

         CALL convert_to_new_pgrid(mp_comm_mm, matrix_b_rs%matrix, matrix_b_mm, optimize_pgrid=opt_pgrid, move_data=move_b)

         info_b = dbcsr_tas_info(matrix_b_rs)
         CALL dbcsr_tas_info_hold(info_b)

         IF (new_b) THEN
            CALL dbcsr_tas_destroy(matrix_b_rs)
            DEALLOCATE (matrix_b_rs)
         END IF

         CALL convert_to_new_pgrid(mp_comm_mm, matrix_c_rep%matrix, matrix_c_mm, nodata=nodata_3, optimize_pgrid=opt_pgrid)

         info_c = dbcsr_tas_info(matrix_c_rep)
         CALL dbcsr_tas_info_hold(info_c)

         CALL mp_sync(matrix_a%dist%info%mp_comm)
         CALL timeset("dbcsr_tas_dbcsr", handle4)
         CALL timeset("dbcsr_tas_mm_2", handle3)
         CALL dbcsr_multiply(transa=transa_prv, transb=transb_prv, alpha=alpha, matrix_a=matrix_a_mm, &
                             matrix_b=matrix_b_mm, beta=beta, matrix_c=matrix_c_mm, &
                             filter_eps=filter_eps_prv/REAL(nsplit, KIND=real_8), retain_sparsity=retain_sparsity, flop=flop)
         CALL mp_sync(matrix_a%dist%info%mp_comm)
         CALL timestop(handle3)
         CALL timestop(handle4)

         CALL dbcsr_release(matrix_a_mm)
         CALL dbcsr_release(matrix_b_mm)

         nze_c = dbcsr_get_nze(matrix_c_mm)

         CALL redistribute_and_sum(matrix_c_mm, matrix_c_rep%matrix, local_copy=.NOT. opt_pgrid, alpha=beta)
         nze_c_sum = dbcsr_tas_get_nze_total(matrix_c_rep)

         CALL dbcsr_release(matrix_c_mm)

         IF (unit_nr_prv /= 0) THEN
            CALL dbcsr_tas_write_dist(matrix_c_rep, unit_nr_prv, full_info=log_verbose)
         END IF

         IF (matrix_c%do_batched == 0) THEN
            CALL dbcsr_tas_merge(matrix_c_rs%matrix, matrix_c_rep, move_data=.TRUE.)
         ELSE
            matrix_c%mm_storage%batched_out = .TRUE. ! postpone merging submatrices to dbcsr_tas_batched_mm_finalize
         END IF

         IF (matrix_c%do_batched == 0) THEN
            CALL dbcsr_tas_destroy(matrix_c_rep)
            DEALLOCATE (matrix_c_rep)
         END IF

         IF (PRESENT(filter_eps)) CALL dbcsr_tas_filter(matrix_c_rs, filter_eps)

         ! set upper limit on memory consumption for replicated matrix and complete batched mm
         ! if limit is exceeded
         IF (nze_c_sum > default_nsplit_accept_ratio*MAX(nze_a, nze_b)) THEN
            CALL dbcsr_tas_batched_mm_complete(matrix_c)
         END IF

      CASE (3)
         IF (matrix_a%do_batched <= 2) THEN
            ALLOCATE (matrix_a_rep)
            CALL dbcsr_tas_replicate(matrix_a_rs%matrix, dbcsr_tas_info(matrix_b_rs), matrix_a_rep, move_data=.TRUE.)
            IF (matrix_a%do_batched == 1 .or. matrix_a%do_batched == 2) THEN
               matrix_a%mm_storage%store_batched_repl => matrix_a_rep
               CALL dbcsr_tas_set_batched_state(matrix_a, state=3)
            END IF
         ELSEIF (matrix_a%do_batched == 3) THEN
            matrix_a_rep => matrix_a%mm_storage%store_batched_repl
         END IF

         IF (new_a) THEN
            CALL dbcsr_tas_destroy(matrix_a_rs)
            DEALLOCATE (matrix_a_rs)
         END IF
         IF (unit_nr_prv /= 0) THEN
            CALL dbcsr_tas_write_dist(matrix_a_rep, unit_nr_prv, full_info=log_verbose)
            CALL dbcsr_tas_write_dist(matrix_b_rs, unit_nr_prv)
         END IF

         CALL convert_to_new_pgrid(mp_comm_mm, matrix_a_rep%matrix, matrix_a_mm, optimize_pgrid=opt_pgrid, &
                                   move_data=matrix_a%do_batched == 0)

         ! keep communicators alive even after releasing TAS matrices (communicator management does not work between DBCSR and TAS)
         info_a = dbcsr_tas_info(matrix_a_rep)
         CALL dbcsr_tas_info_hold(info_a)

         IF (matrix_a%do_batched == 0) THEN
            CALL dbcsr_tas_destroy(matrix_a_rep)
            DEALLOCATE (matrix_a_rep)
         END IF

         CALL convert_to_new_pgrid(mp_comm_mm, matrix_b_rs%matrix, matrix_b_mm, optimize_pgrid=opt_pgrid, move_data=move_b)

         info_b = dbcsr_tas_info(matrix_b_rs)
         CALL dbcsr_tas_info_hold(info_b)

         IF (new_b) THEN
            CALL dbcsr_tas_destroy(matrix_b_rs)
            DEALLOCATE (matrix_b_rs)
         END IF
         CALL convert_to_new_pgrid(mp_comm_mm, matrix_c_rs%matrix, matrix_c_mm, nodata=nodata_3, optimize_pgrid=opt_pgrid)

         info_c = dbcsr_tas_info(matrix_c_rs)
         CALL dbcsr_tas_info_hold(info_c)

         CALL mp_sync(matrix_a%dist%info%mp_comm)
         CALL timeset("dbcsr_tas_dbcsr", handle4)
         SELECT CASE (tr_case)
         CASE (dbcsr_no_transpose)
            CALL timeset("dbcsr_tas_mm_3N", handle3)
            CALL dbcsr_multiply(transa=dbcsr_no_transpose, transb=dbcsr_no_transpose, alpha=alpha, &
                                matrix_a=matrix_a_mm, matrix_b=matrix_b_mm, beta=beta, matrix_c=matrix_c_mm, &
                                filter_eps=filter_eps_prv, retain_sparsity=retain_sparsity, flop=flop)
            CALL timestop(handle3)
         CASE (dbcsr_transpose)
            CALL timeset("dbcsr_tas_mm_3T", handle3)
            CALL dbcsr_multiply(transa=dbcsr_no_transpose, transb=dbcsr_transpose, alpha=alpha, &
                                matrix_a=matrix_b_mm, matrix_b=matrix_a_mm, beta=beta, matrix_c=matrix_c_mm, &
                                filter_eps=filter_eps_prv, retain_sparsity=retain_sparsity, flop=flop)
            CALL timestop(handle3)
         END SELECT
         CALL mp_sync(matrix_a%dist%info%mp_comm)
         CALL timestop(handle4)

         CALL dbcsr_release(matrix_a_mm)
         CALL dbcsr_release(matrix_b_mm)

         nze_c = dbcsr_get_nze(matrix_c_mm)

         IF (.NOT. new_c) THEN
            CALL redistribute_and_sum(matrix_c_mm, matrix_c_rs%matrix, local_copy=.NOT. opt_pgrid, alpha=beta)
         ELSE
            CALL redistribute_and_sum(matrix_c_mm, matrix_c_rs%matrix, local_copy=.NOT. opt_pgrid)
         END IF

         CALL dbcsr_release(matrix_c_mm)

         IF (PRESENT(filter_eps)) CALL dbcsr_tas_filter(matrix_c_rs, filter_eps)

         IF (unit_nr_prv /= 0) THEN
            CALL dbcsr_tas_write_dist(matrix_c_rs, unit_nr_prv)
         END IF
      END SELECT

      CALL mp_comm_free(mp_comm_mm)

      CALL dbcsr_tas_get_split_info(info_c, mp_comm=mp_comm)

      IF (PRESENT(split_opt)) THEN
         SELECT CASE (max_mm_dim)
         CASE (1, 3)
            CALL mp_sum(nze_c, mp_comm)
         CASE (2)
            CALL dbcsr_tas_get_split_info(info_c, mp_comm=mp_comm, mp_comm_group=mp_comm_group)
            CALL mp_sum(nze_c, mp_comm_group)
            CALL mp_max(nze_c, mp_comm)

         END SELECT
         nsplit_opt = split_factor_estimate(max_mm_dim, nze_a, nze_b, nze_c, numproc)
         ! ideally we should rederive the split factor from the actual sparsity of C, but
         ! due to parameter beta, we can not get the sparsity of AxB from DBCSR if not new_c
         mp_comm_opt = dbcsr_tas_mp_comm(mp_comm, split_rc, nsplit_opt)
         CALL dbcsr_tas_create_split(split_opt, mp_comm_opt, split_rc, nsplit_opt, own_comm=.TRUE.)
         IF (unit_nr_prv > 0) THEN
            WRITE (unit_nr_prv, "(T2,A)") &
               "MM PARAMETERS"
            WRITE (unit_nr_prv, "(T4,A,T68,I13)") "Number of matrix elements per CPU of result matrix:", &
               (nze_c + numproc - 1)/numproc

            WRITE (unit_nr_prv, "(T4,A,T68,I13)") "Optimal split factor:", nsplit_opt
         END IF

      END IF

      IF (new_c) THEN
         CALL dbcsr_scale(matrix_c%matrix, beta)
         CALL dbcsr_tas_reshape(matrix_c_rs, matrix_c, summation=.TRUE., transposed=transc_prv /= transc, &
                                move_data=.TRUE.)
         CALL dbcsr_tas_destroy(matrix_c_rs)
         DEALLOCATE (matrix_c_rs)
         IF (PRESENT(filter_eps)) CALL dbcsr_tas_filter(matrix_c, filter_eps)
      ELSEIF (matrix_c%do_batched > 0) THEN
         IF (matrix_c%mm_storage%batched_out) THEN
            matrix_c%mm_storage%batched_trans = transc_prv /= transc
         END IF
      END IF

      IF (PRESENT(move_data_a)) THEN
         IF (move_data_a) CALL dbcsr_tas_clear(matrix_a)
      END IF
      IF (PRESENT(move_data_b)) THEN
         IF (move_data_b) CALL dbcsr_tas_clear(matrix_b)
      END IF

      IF (PRESENT(flop)) THEN
         CALL mp_sum(flop, mp_comm)
         flop = (flop + numproc - 1)/numproc
      END IF

      IF (PRESENT(optimize_dist)) THEN
         IF (optimize_dist) CALL mp_comm_free(comm_tmp)
      END IF
      IF (unit_nr_prv > 0) THEN
         WRITE (unit_nr_prv, '(A)') repeat("-", 80)
         WRITE (unit_nr_prv, '(A,1X,A,1X,A,1X,A,1X,A,1X,A)') "TAS MATRIX MULTIPLICATION DONE"
         WRITE (unit_nr_prv, '(A)') repeat("-", 80)
      END IF

      CALL dbcsr_tas_release_info(info_a)
      CALL dbcsr_tas_release_info(info_b)
      CALL dbcsr_tas_release_info(info_c)

      CALL mp_sync(matrix_a%dist%info%mp_comm)
      CALL timestop(handle2)
      CALL timestop(handle)

   END SUBROUTINE

   SUBROUTINE redistribute_and_sum(matrix_in, matrix_out, local_copy, alpha)
      TYPE(dbcsr_type), INTENT(IN) :: matrix_in
      TYPE(dbcsr_type), INTENT(INOUT) :: matrix_out
      LOGICAL, INTENT(IN), OPTIONAL :: local_copy
      TYPE(dbcsr_scalar_type), INTENT(IN), OPTIONAL :: alpha
      TYPE(dbcsr_type) :: matrix_tmp
      LOGICAL :: local_copy_prv

      IF (PRESENT(local_copy)) THEN
         local_copy_prv = local_copy
      ELSE
         local_copy_prv = .FALSE.
      END IF

      IF (.NOT. local_copy_prv) THEN
         CALL dbcsr_create(matrix_tmp, matrix_out)
         CALL dbcsr_redistribute(matrix_in, matrix_tmp)
         CALL dbcsr_add(matrix_out, matrix_tmp, alpha_scalar=alpha)
         CALL dbcsr_release(matrix_tmp)
      ELSE
         CALL dbcsr_add(matrix_out, matrix_in, alpha_scalar=alpha)
      END IF

   END SUBROUTINE

   SUBROUTINE reshape_mm_small(mp_comm, matrix_in, matrix_out, transposed, trans, nodata, move_data)
      !! Make sure that smallest matrix involved in a multiplication is not split and bring it to
      !! the same process grid as the other 2 matrices.

      TYPE(mp_comm_type), INTENT(IN)               :: mp_comm
         !! communicator that defines Cartesian topology
      TYPE(dbcsr_tas_type), INTENT(INOUT) :: matrix_in
      TYPE(dbcsr_tas_type), INTENT(OUT)   :: matrix_out
      LOGICAL, INTENT(IN)               :: transposed
         !! Whether matrix_out should be transposed
      CHARACTER(LEN=1), INTENT(INOUT)   :: trans
         !! update transpose flag for DBCSR mm according to 'transposed' argument
      LOGICAL, INTENT(IN), OPTIONAL     :: nodata, move_data
         !! Data of matrix_in should not be copied to matrix_out
         !! memory optimization: move data such that matrix_in is empty on return.

      INTEGER                           :: numnodes
      INTEGER(KIND=int_8), DIMENSION(2) :: dims
      INTEGER, DIMENSION(2)             :: pdims, pcoord
      TYPE(dbcsr_tas_dist_arb)            :: new_row_dist, new_col_dist
      TYPE(dbcsr_tas_distribution_type)   :: dist
      LOGICAL                           :: nodata_prv
      CHARACTER(LEN=*), PARAMETER       :: routineN = 'reshape_mm_small'
      INTEGER                           :: handle

      CALL timeset(routineN, handle)

      IF (PRESENT(nodata)) THEN
         nodata_prv = nodata
      ELSE
         nodata_prv = .FALSE.
      END IF

      IF (transposed) THEN
         SELECT CASE (trans)
         CASE (dbcsr_transpose)
            trans = dbcsr_no_transpose
         CASE (dbcsr_no_transpose)
            trans = dbcsr_transpose
         END SELECT
      END IF

      CALL mp_environ(numnodes, pdims, pcoord, mp_comm)

      dims = [dbcsr_tas_nblkrows_total(matrix_in), dbcsr_tas_nblkcols_total(matrix_in)]

      IF (transposed) CALL swap(dims)

      IF (.NOT. transposed) THEN
         new_row_dist = dbcsr_tas_dist_arb_default(pdims(1), dims(1), matrix_in%row_blk_size)
         new_col_dist = dbcsr_tas_dist_arb_default(pdims(2), dims(2), matrix_in%col_blk_size)
         CALL dbcsr_tas_distribution_new(dist, mp_comm, new_row_dist, new_col_dist, nosplit=.TRUE.)
         CALL dbcsr_tas_create(matrix_out, matrix_in%matrix%name, dist, dbcsr_tas_get_data_type(matrix_in), &
                               matrix_in%row_blk_size, matrix_in%col_blk_size, own_dist=.TRUE.)
      ELSE
         new_row_dist = dbcsr_tas_dist_arb_default(pdims(1), dims(1), matrix_in%col_blk_size)
         new_col_dist = dbcsr_tas_dist_arb_default(pdims(2), dims(2), matrix_in%row_blk_size)
         CALL dbcsr_tas_distribution_new(dist, mp_comm, new_row_dist, new_col_dist, nosplit=.TRUE.)
         CALL dbcsr_tas_create(matrix_out, matrix_in%matrix%name, dist, dbcsr_tas_get_data_type(matrix_in), &
                               matrix_in%col_blk_size, matrix_in%row_blk_size, own_dist=.TRUE.)

      END IF
      IF (.NOT. nodata_prv) CALL dbcsr_tas_reshape(matrix_in, matrix_out, transposed=transposed, move_data=move_data)

      CALL timestop(handle)

   END SUBROUTINE

   SUBROUTINE reshape_mm_compatible(matrix1_in, matrix2_in, matrix1_out, matrix2_out, new1, new2, trans1, trans2, &
                                    optimize_dist, nsplit, opt_nsplit, split_rc_1, split_rc_2, nodata1, nodata2, &
                                    move_data_1, move_data_2, comm_new, unit_nr)
      !! Reshape either matrix1 or matrix2 to make sure that their process grids are compatible with
      !! the same split factor.

      TYPE(dbcsr_tas_type), TARGET, &
         INTENT(INOUT)                           :: matrix1_in, matrix2_in
      TYPE(dbcsr_tas_type), POINTER, INTENT(OUT) :: matrix1_out, matrix2_out
      LOGICAL, INTENT(OUT)                       :: new1, new2
         !! Whether matrix1_out is a new matrix or simply pointing to matrix1_in
         !! Whether matrix2_out is a new matrix or simply pointing to matrix2_in
      CHARACTER(LEN=1), INTENT(INOUT)            :: trans1, trans2
         !! transpose flag of matrix1_in for multiplication
         !! transpose flag of matrix2_in for multiplication
      LOGICAL, INTENT(IN), OPTIONAL              :: optimize_dist
         !! experimental: optimize matrix splitting and distribution
      INTEGER, INTENT(IN), OPTIONAL              :: nsplit
         !! Optimal split factor (set to 0 if split factor should not be changed)
      LOGICAL, INTENT(IN), OPTIONAL              :: opt_nsplit
      INTEGER, INTENT(INOUT)                     :: split_rc_1, split_rc_2
         !! Whether to split rows or columns for matrix 1
         !! Whether to split rows or columns for matrix 2
      TYPE(mp_comm_type), INTENT(OUT), OPTIONAL             :: comm_new
         !! returns the new communicator only if optimize_dist
      LOGICAL, OPTIONAL, INTENT(IN)              :: nodata1, nodata2
         !! Don't copy matrix data from matrix1_in to matrix1_out
         !! Don't copy matrix data from matrix2_in to matrix2_out
      LOGICAL, OPTIONAL, INTENT(INOUT)           :: move_data_1, move_data_2
         !! memory optimization: move data such that matrix1_in may be empty on return.
         !! memory optimization: move data such that matrix2_in may be empty on return.
      INTEGER, INTENT(IN), OPTIONAL              :: unit_nr
         !! output unit

      INTEGER(KIND=int_8), DIMENSION(2)          :: dims1, dims2, dims_ref
      INTEGER(KIND=int_8)                        :: d1, d2
      CHARACTER(LEN=*), PARAMETER                :: routineN = 'reshape_mm_compatible'
      INTEGER                                    :: handle, numnodes, unit_nr_prv, &
                                                    nsplit_prv, ref, split_rc_ref
      INTEGER, DIMENSION(2)                      :: pcoord, pdims
      LOGICAL                                    :: optimize_dist_prv, trans1_newdist, trans2_newdist
      TYPE(dbcsr_tas_dist_cyclic)                :: row_dist_1, col_dist_1, row_dist_2, col_dist_2
      TYPE(dbcsr_tas_distribution_type)          :: dist_1, dist_2
      TYPE(dbcsr_tas_split_info)                 :: split_info
      INTEGER(KIND=int_8)                        :: nze1, nze2
      LOGICAL                                    :: nodata1_prv, nodata2_prv
      TYPE(mp_comm_type)                         :: mp_comm

      CALL timeset(routineN, handle)
      new1 = .FALSE.; new2 = .FALSE.

      IF (PRESENT(nodata1)) THEN
         nodata1_prv = nodata1
      ELSE
         nodata1_prv = .FALSE.
      END IF

      IF (PRESENT(nodata2)) THEN
         nodata2_prv = nodata2
      ELSE
         nodata2_prv = .FALSE.
      END IF

      unit_nr_prv = prep_output_unit(unit_nr)

      NULLIFY (matrix1_out, matrix2_out)

      IF (PRESENT(optimize_dist)) THEN
         optimize_dist_prv = optimize_dist
      ELSE
         optimize_dist_prv = .FALSE.
      END IF

      dims1 = [dbcsr_tas_nblkrows_total(matrix1_in), dbcsr_tas_nblkcols_total(matrix1_in)]
      dims2 = [dbcsr_tas_nblkrows_total(matrix2_in), dbcsr_tas_nblkcols_total(matrix2_in)]
      nze1 = dbcsr_tas_get_nze_total(matrix1_in)
      nze2 = dbcsr_tas_get_nze_total(matrix2_in)

      IF (trans1 == dbcsr_transpose) split_rc_1 = MOD(split_rc_1, 2) + 1

      IF (trans2 == dbcsr_transpose) split_rc_2 = MOD(split_rc_2, 2) + 1

      IF (nze1 >= nze2) THEN
         ref = 1
         split_rc_ref = split_rc_1
         dims_ref = dims1
      ELSE
         ref = 2
         split_rc_ref = split_rc_2
         dims_ref = dims2
      END IF

      IF (PRESENT(nsplit)) THEN
         nsplit_prv = nsplit
      ELSE
         nsplit_prv = 0
      END IF

      IF (optimize_dist_prv) THEN
         DBCSR_ASSERT(PRESENT(comm_new))
      END IF

      IF ((.NOT. optimize_dist_prv) .AND. dist_compatible(matrix1_in, matrix2_in, split_rc_1, split_rc_2)) THEN
         CALL change_split(matrix1_in, matrix1_out, nsplit_prv, split_rc_1, new1, &
                           move_data=move_data_1, nodata=nodata1, opt_nsplit=opt_nsplit)
         CALL dbcsr_tas_get_split_info(dbcsr_tas_info(matrix1_out), nsplit=nsplit_prv)
         CALL change_split(matrix2_in, matrix2_out, nsplit_prv, split_rc_2, new2, &
                           move_data=move_data_2, nodata=nodata2, opt_nsplit=.FALSE.)
         IF (unit_nr_prv > 0) THEN
            WRITE (unit_nr_prv, "(T2,A,1X,A,1X,A,1X,A)") "No redistribution of", TRIM(matrix1_in%matrix%name), &
               "and", TRIM(matrix2_in%matrix%name)
            IF (new1) THEN
               WRITE (unit_nr_prv, "(T2,A,1X,A,1X,A)") "Change split factor of", TRIM(matrix1_in%matrix%name), ": Yes"
            ELSE
               WRITE (unit_nr_prv, "(T2,A,1X,A,1X,A)") "Change split factor of", TRIM(matrix1_in%matrix%name), ": No"
            END IF
            IF (new2) THEN
               WRITE (unit_nr_prv, "(T2,A,1X,A,1X,A)") "Change split factor of", TRIM(matrix2_in%matrix%name), ": Yes"
            ELSE
               WRITE (unit_nr_prv, "(T2,A,1X,A,1X,A)") "Change split factor of", TRIM(matrix2_in%matrix%name), ": No"
            END IF
         END IF
      ELSE

         IF (optimize_dist_prv) THEN
            IF (unit_nr_prv > 0) THEN
               WRITE (unit_nr_prv, "(T2,A,1X,A,1X,A,1X,A)") "Optimizing distribution of", TRIM(matrix1_in%matrix%name), &
                  "and", TRIM(matrix2_in%matrix%name)
            END IF

            trans1_newdist = (split_rc_1 == colsplit)
            trans2_newdist = (split_rc_2 == colsplit)

            IF (trans1_newdist) THEN
               CALL swap(dims1)
               CALL invert_transpose_flag(trans1)
            END IF

            IF (trans2_newdist) THEN
               CALL swap(dims2)
               CALL invert_transpose_flag(trans2)
            END IF

            IF (nsplit_prv == 0) THEN
               SELECT CASE (split_rc_ref)
               CASE (rowsplit)
                  d1 = dims_ref(1)
                  d2 = dims_ref(2)
               CASE (colsplit)
                  d1 = dims_ref(2)
                  d2 = dims_ref(1)
               END SELECT
               nsplit_prv = INT((d1 - 1)/d2 + 1)
            END IF

            DBCSR_ASSERT(nsplit_prv > 0)

            CALL dbcsr_tas_get_split_info(dbcsr_tas_info(matrix1_in), mp_comm=mp_comm)
            comm_new = dbcsr_tas_mp_comm(mp_comm, rowsplit, nsplit_prv)
            CALL dbcsr_tas_create_split(split_info, comm_new, rowsplit, nsplit_prv)

            CALL mp_environ(numnodes, pdims, pcoord, comm_new)

            ! use a very simple cyclic distribution that may not be load balanced if block
            ! sizes are not equal. However we can not use arbitrary distributions
            ! for large dimensions since this would require storing distribution vectors as arrays
            ! which can not be stored for large dimensions.
            row_dist_1 = dbcsr_tas_dist_cyclic(1, pdims(1), dims1(1))
            col_dist_1 = dbcsr_tas_dist_cyclic(1, pdims(2), dims1(2))

            row_dist_2 = dbcsr_tas_dist_cyclic(1, pdims(1), dims2(1))
            col_dist_2 = dbcsr_tas_dist_cyclic(1, pdims(2), dims2(2))

            CALL dbcsr_tas_distribution_new(dist_1, comm_new, row_dist_1, col_dist_1, split_info=split_info)
            CALL dbcsr_tas_distribution_new(dist_2, comm_new, row_dist_2, col_dist_2, split_info=split_info)
            CALL dbcsr_tas_release_info(split_info)

            ALLOCATE (matrix1_out)
            IF (.NOT. trans1_newdist) THEN
               CALL dbcsr_tas_create(matrix1_out, matrix1_in%matrix%name, dist_1, dbcsr_tas_get_data_type(matrix1_in), &
                                     matrix1_in%row_blk_size, matrix1_in%col_blk_size, own_dist=.TRUE.)

            ELSE
               CALL dbcsr_tas_create(matrix1_out, matrix1_in%matrix%name, dist_1, dbcsr_tas_get_data_type(matrix1_in), &
                                     matrix1_in%col_blk_size, matrix1_in%row_blk_size, own_dist=.TRUE.)
            END IF

            ALLOCATE (matrix2_out)
            IF (.NOT. trans2_newdist) THEN
               CALL dbcsr_tas_create(matrix2_out, matrix2_in%matrix%name, dist_2, dbcsr_tas_get_data_type(matrix2_in), &
                                     matrix2_in%row_blk_size, matrix2_in%col_blk_size, own_dist=.TRUE.)
            ELSE
               CALL dbcsr_tas_create(matrix2_out, matrix2_in%matrix%name, dist_2, dbcsr_tas_get_data_type(matrix2_in), &
                                     matrix2_in%col_blk_size, matrix2_in%row_blk_size, own_dist=.TRUE.)
            END IF

            IF (.NOT. nodata1_prv) CALL dbcsr_tas_reshape(matrix1_in, matrix1_out, transposed=trans1_newdist, move_data=move_data_1)
            IF (.NOT. nodata2_prv) CALL dbcsr_tas_reshape(matrix2_in, matrix2_out, transposed=trans2_newdist, move_data=move_data_2)
            new1 = .TRUE.
            new2 = .TRUE.

         ELSE
            SELECT CASE (ref)
            CASE (1)
               IF (unit_nr_prv > 0) THEN
                  WRITE (unit_nr_prv, "(T2,A,1X,A)") "Redistribution of", TRIM(matrix2_in%matrix%name)
               END IF

               CALL change_split(matrix1_in, matrix1_out, nsplit_prv, split_rc_1, new1, &
                                 move_data=move_data_1, nodata=nodata1, opt_nsplit=opt_nsplit)

               ALLOCATE (matrix2_out)
               CALL reshape_mm_template(matrix1_out, matrix2_in, matrix2_out, trans2, split_rc_2, &
                                        nodata=nodata2, move_data=move_data_2)
               new2 = .TRUE.
            CASE (2)
               IF (unit_nr_prv > 0) THEN
                  WRITE (unit_nr_prv, "(T2,A,1X,A)") "Redistribution of", TRIM(matrix1_in%matrix%name)
               END IF

               CALL change_split(matrix2_in, matrix2_out, nsplit_prv, split_rc_2, new2, &
                                 move_data=move_data_2, nodata=nodata2, opt_nsplit=opt_nsplit)

               ALLOCATE (matrix1_out)
               CALL reshape_mm_template(matrix2_out, matrix1_in, matrix1_out, trans1, split_rc_1, &
                                        nodata=nodata1, move_data=move_data_1)
               new1 = .TRUE.
            END SELECT
         END IF
      END IF

      IF (PRESENT(move_data_1) .AND. new1) move_data_1 = .TRUE.
      IF (PRESENT(move_data_2) .AND. new2) move_data_2 = .TRUE.

      CALL timestop(handle)

   END SUBROUTINE

   SUBROUTINE change_split(matrix_in, matrix_out, nsplit, split_rowcol, is_new, opt_nsplit, move_data, nodata)
      !! Change split factor without redistribution

      TYPE(dbcsr_tas_type), TARGET, &
         INTENT(INOUT)                           :: matrix_in
      TYPE(dbcsr_tas_type), POINTER, INTENT(OUT) :: matrix_out
      INTEGER, INTENT(IN)                        :: nsplit
         !! new split factor, set to 0 to not change split of matrix_in
      INTEGER, INTENT(IN)                        :: split_rowcol
         !! split rows or columns
      LOGICAL, INTENT(OUT)                       :: is_new
         !! whether matrix_out is new or a pointer to matrix_in
      LOGICAL, INTENT(IN), OPTIONAL              :: opt_nsplit
         !! whether nsplit should be optimized for current process grid
      LOGICAL, INTENT(IN), OPTIONAL              :: nodata
         !! Data of matrix_in should not be copied to matrix_out
      LOGICAL, INTENT(INOUT), OPTIONAL           :: move_data
         !! memory optimization: move data such that matrix_in is empty on return.

      INTEGER                                    :: &
         split_rc, nsplit_old, handle, data_type, nsplit_new, nsplit_prv
      TYPE(dbcsr_tas_split_info)                 :: split_info
      CHARACTER(len=default_string_length)       :: name
      TYPE(dbcsr_tas_distribution_type)          :: dist
      LOGICAL                                    :: nodata_prv
      CLASS(dbcsr_tas_distribution), ALLOCATABLE :: rdist, cdist
      CLASS(dbcsr_tas_rowcol_data), ALLOCATABLE  :: rbsize, cbsize
      TYPE(mp_comm_type)                         :: mp_comm
      CHARACTER(LEN=*), PARAMETER                :: routineN = 'change_split'

      NULLIFY (matrix_out)

      is_new = .TRUE.

      CALL dbcsr_tas_get_split_info(dbcsr_tas_info(matrix_in), mp_comm=mp_comm, &
                                    split_rowcol=split_rc, nsplit=nsplit_old)

      IF (nsplit == 0) THEN
         IF (split_rowcol == split_rc) THEN
            matrix_out => matrix_in
            is_new = .FALSE.
            RETURN
         ELSE
            nsplit_prv = 1
         END IF
      ELSE
         nsplit_prv = nsplit
      END IF

      CALL timeset(routineN, handle)

      nodata_prv = .FALSE.
      IF (PRESENT(nodata)) nodata_prv = nodata

      CALL dbcsr_tas_get_info(matrix_in, data_type=data_type, name=name, &
                              row_blk_size=rbsize, col_blk_size=cbsize, &
                              proc_row_dist=rdist, proc_col_dist=cdist)

      CALL dbcsr_tas_create_split(split_info, mp_comm, split_rowcol, nsplit_prv, opt_nsplit=opt_nsplit)

      CALL dbcsr_tas_get_split_info(split_info, nsplit=nsplit_new)

      IF (nsplit_old == nsplit_new .AND. split_rc == split_rowcol) THEN
         matrix_out => matrix_in
         is_new = .FALSE.
         CALL dbcsr_tas_release_info(split_info)
         CALL timestop(handle)
         RETURN
      END IF

      CALL dbcsr_tas_distribution_new(dist, mp_comm, rdist, cdist, &
                                      split_info=split_info)

      CALL dbcsr_tas_release_info(split_info)

      ALLOCATE (matrix_out)
      CALL dbcsr_tas_create(matrix_out, name, dist, &
                            data_type, &
                            rbsize, cbsize, own_dist=.TRUE.)

      IF (.NOT. nodata_prv) CALL dbcsr_tas_copy(matrix_out, matrix_in)

      IF (PRESENT(move_data)) THEN
         IF (.NOT. nodata_prv) THEN
            IF (move_data) CALL dbcsr_tas_clear(matrix_in)
            move_data = .TRUE.
         END IF
      END IF

      CALL timestop(handle)
   END SUBROUTINE

   FUNCTION dist_compatible(mat_a, mat_b, split_rc_a, split_rc_b, unit_nr)
      !! Check whether matrices have same distribution and same split.
      TYPE(dbcsr_tas_type), INTENT(IN)           :: mat_a, mat_b
      INTEGER, INTENT(IN)                        :: split_rc_a, split_rc_b
      INTEGER, INTENT(IN), OPTIONAL              :: unit_nr
      LOGICAL                                    :: dist_compatible

      INTEGER                                    :: same_local_rowcols, split_check_a, split_check_b
      TYPE(dbcsr_tas_split_info)                 :: info_a, info_b
      INTEGER                                    :: unit_nr_prv, numproc
      INTEGER, DIMENSION(2)                      :: pdims_a, pdims_b, pcoord_a, pcoord_b
      INTEGER(int_8), DIMENSION(:), ALLOCATABLE         :: local_rowcols_a, local_rowcols_b

      unit_nr_prv = prep_output_unit(unit_nr)

      dist_compatible = .FALSE.

      info_a = dbcsr_tas_info(mat_a)
      info_b = dbcsr_tas_info(mat_b)
      CALL dbcsr_tas_get_split_info(info_a, split_rowcol=split_check_a)
      CALL dbcsr_tas_get_split_info(info_b, split_rowcol=split_check_b)
      IF (split_check_b /= split_rc_b .OR. split_check_a /= split_rc_a .OR. split_rc_a /= split_rc_b) THEN
         IF (unit_nr_prv > 0) THEN
            WRITE (unit_nr_prv, *) "matrix layout a not compatible", split_check_a, split_rc_a
            WRITE (unit_nr_prv, *) "matrix layout b not compatible", split_check_b, split_rc_b
         END IF
         RETURN
      END IF

      ! check if communicators are equivalent
      ! Note: mpi_comm_compare is not sufficient since this does not compare associated Cartesian grids.
      ! It's sufficient to check dimensions of global grid, subgrids will be determined later on (change_split)
      CALL mp_environ(numproc, pdims_a, pcoord_a, info_a%mp_comm)
      CALL mp_environ(numproc, pdims_b, pcoord_b, info_b%mp_comm)
      IF (.NOT. array_eq(pdims_a, pdims_b)) THEN
         IF (unit_nr_prv > 0) THEN
            WRITE (unit_nr_prv, *) "mp dims not compatible:", pdims_a, "|", pdims_b
         END IF
         RETURN
      END IF

      ! check that distribution is the same by comparing local rows / columns for each matrix
      SELECT CASE (split_rc_a)
      CASE (rowsplit)
         CALL dbcsr_tas_get_info(mat_a, local_rows=local_rowcols_a)
         CALL dbcsr_tas_get_info(mat_b, local_rows=local_rowcols_b)
      CASE (colsplit)
         CALL dbcsr_tas_get_info(mat_a, local_cols=local_rowcols_a)
         CALL dbcsr_tas_get_info(mat_b, local_cols=local_rowcols_b)
      END SELECT

      same_local_rowcols = MERGE(1, 0, array_eq(local_rowcols_a, local_rowcols_b))

      CALL mp_sum(same_local_rowcols, info_a%mp_comm)

      IF (same_local_rowcols == numproc) THEN
         dist_compatible = .TRUE.
      ELSE
         IF (unit_nr_prv > 0) THEN
            WRITE (unit_nr_prv, *) "local rowcols not compatible"
            WRITE (unit_nr_prv, *) "local rowcols A", local_rowcols_a
            WRITE (unit_nr_prv, *) "local rowcols B", local_rowcols_b
         END IF
      END IF

   END FUNCTION

   SUBROUTINE reshape_mm_template(template, matrix_in, matrix_out, trans, split_rc, nodata, move_data)
      !! Reshape matrix_in s.t. it has same process grid, distribution and split as template
      TYPE(dbcsr_tas_type), INTENT(IN)           :: template
      TYPE(dbcsr_tas_type), INTENT(INOUT)        :: matrix_in
      TYPE(dbcsr_tas_type), INTENT(OUT)          :: matrix_out
      CHARACTER(LEN=1), INTENT(INOUT)            :: trans
      INTEGER, INTENT(IN)                        :: split_rc
      LOGICAL, INTENT(IN), OPTIONAL              :: nodata, move_data
      CLASS(dbcsr_tas_distribution), ALLOCATABLE :: row_dist, col_dist

      TYPE(dbcsr_tas_distribution_type)          :: dist_new
      TYPE(dbcsr_tas_split_info)                 :: info_template, info_matrix
      INTEGER                                    :: dim_split_template, dim_split_matrix, &
                                                    numnodes, handle
      INTEGER, DIMENSION(2)                      :: pcoord, pdims
      LOGICAL                                    :: nodata_prv, transposed
      TYPE(mp_comm_type)                         :: mp_comm
      CHARACTER(LEN=*), PARAMETER :: routineN = 'reshape_mm_template'

      CALL timeset(routineN, handle)

      IF (PRESENT(nodata)) THEN
         nodata_prv = nodata
      ELSE
         nodata_prv = .FALSE.
      END IF

      info_template = dbcsr_tas_info(template)
      info_matrix = dbcsr_tas_info(matrix_in)

      dim_split_template = info_template%split_rowcol
      dim_split_matrix = split_rc

      transposed = dim_split_template .NE. dim_split_matrix
      IF (transposed) THEN
         SELECT CASE (trans)
         CASE (dbcsr_transpose)
            trans = dbcsr_no_transpose
         CASE (dbcsr_no_transpose)
            trans = dbcsr_transpose
         END SELECT
      END IF

      CALL mp_environ(numnodes, pdims, pcoord, info_template%mp_comm)

      SELECT CASE (dim_split_template)
      CASE (1)
         IF (.NOT. transposed) THEN
            ALLOCATE (row_dist, source=template%dist%row_dist)
            ALLOCATE (col_dist, source=dbcsr_tas_dist_arb_default(pdims(2), matrix_in%nblkcols, matrix_in%col_blk_size))
         ELSE
            ALLOCATE (row_dist, source=template%dist%row_dist)
            ALLOCATE (col_dist, source=dbcsr_tas_dist_arb_default(pdims(2), matrix_in%nblkrows, matrix_in%row_blk_size))
         END IF
      CASE (2)
         IF (.NOT. transposed) THEN
            ALLOCATE (row_dist, source=dbcsr_tas_dist_arb_default(pdims(1), matrix_in%nblkrows, matrix_in%row_blk_size))
            ALLOCATE (col_dist, source=template%dist%col_dist)
         ELSE
            ALLOCATE (row_dist, source=dbcsr_tas_dist_arb_default(pdims(1), matrix_in%nblkcols, matrix_in%col_blk_size))
            ALLOCATE (col_dist, source=template%dist%col_dist)
         END IF
      END SELECT

      CALL dbcsr_tas_get_split_info(info_template, mp_comm=mp_comm)
      CALL dbcsr_tas_distribution_new(dist_new, mp_comm, row_dist, col_dist, split_info=info_template)
      IF (.NOT. transposed) THEN
         CALL dbcsr_tas_create(matrix_out, matrix_in%matrix%name, dist_new, dbcsr_tas_get_data_type(matrix_in), &
                               matrix_in%row_blk_size, matrix_in%col_blk_size, own_dist=.TRUE.)
      ELSE
         CALL dbcsr_tas_create(matrix_out, matrix_in%matrix%name, dist_new, dbcsr_tas_get_data_type(matrix_in), &
                               matrix_in%col_blk_size, matrix_in%row_blk_size, own_dist=.TRUE.)
      END IF

      IF (.NOT. nodata_prv) CALL dbcsr_tas_reshape(matrix_in, matrix_out, transposed=transposed, move_data=move_data)

      CALL timestop(handle)

   END SUBROUTINE

   SUBROUTINE dbcsr_tas_result_index(transa, transb, transc, matrix_a, matrix_b, matrix_c, filter_eps, &
                                     unit_nr, blk_ind, nze, retain_sparsity)
      !! Estimate sparsity pattern of C resulting from A x B = C by multiplying the block norms of A and B
      !! Same dummy arguments as dbcsr_tas_multiply
      CHARACTER(LEN=1), INTENT(IN)               :: transa, transb, transc
      TYPE(dbcsr_tas_type), INTENT(INOUT), TARGET        :: matrix_a, matrix_b, matrix_c
      TYPE(dbcsr_tas_type), POINTER                      :: matrix_a_bnorm, matrix_b_bnorm, matrix_c_bnorm
      REAL(KIND=real_8), INTENT(IN), OPTIONAL    :: filter_eps
      INTEGER, INTENT(IN), OPTIONAL              :: unit_nr
      INTEGER(int_8), DIMENSION(:, :), ALLOCATABLE, INTENT(OUT), OPTIONAL :: blk_ind
      LOGICAL, INTENT(IN), OPTIONAL :: retain_sparsity
      INTEGER(int_8), INTENT(OUT), OPTIONAL :: nze

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_result_index'
      LOGICAL :: retain_sparsity_prv
      INTEGER :: bn, row_size, col_size, handle, iblk, nblk
      INTEGER(int_8) :: row, col
      TYPE(dbcsr_tas_iterator) :: iter
      TYPE(mp_comm_type) :: mp_comm

      CALL timeset(routineN, handle)

      IF (PRESENT(retain_sparsity)) THEN
         retain_sparsity_prv = retain_sparsity
      ELSE
         retain_sparsity_prv = .FALSE.
      END IF

      IF (.NOT. retain_sparsity_prv) THEN
         ALLOCATE (matrix_a_bnorm, matrix_b_bnorm, matrix_c_bnorm)
         CALL create_block_norms_matrix(matrix_a, matrix_a_bnorm)
         CALL create_block_norms_matrix(matrix_b, matrix_b_bnorm)
         CALL create_block_norms_matrix(matrix_c, matrix_c_bnorm, nodata=.TRUE.)

         CALL dbcsr_tas_multiply(transa, transb, transc, dbcsr_scalar(1.0_real_8), matrix_a_bnorm, &
                                 matrix_b_bnorm, dbcsr_scalar(0.0_real_8), matrix_c_bnorm, &
                                 filter_eps=filter_eps, move_data_a=.TRUE., move_data_b=.TRUE., &
                                 simple_split=.TRUE., unit_nr=unit_nr)
         CALL dbcsr_tas_destroy(matrix_a_bnorm)
         CALL dbcsr_tas_destroy(matrix_b_bnorm)

         DEALLOCATE (matrix_a_bnorm, matrix_b_bnorm)
      ELSE
         matrix_c_bnorm => matrix_c
      END IF

      nblk = dbcsr_tas_get_num_blocks(matrix_c_bnorm)
      IF (PRESENT(blk_ind)) ALLOCATE (blk_ind(nblk, 2))

      CALL dbcsr_tas_iterator_start(iter, matrix_c_bnorm)
      IF (PRESENT(nze)) nze = 0
      DO iblk = 1, nblk
         CALL dbcsr_tas_iterator_next_block(iter, row, col, bn)
         row_size = matrix_c%row_blk_size%data(row)
         col_size = matrix_c%col_blk_size%data(col)
         IF (PRESENT(nze)) nze = nze + row_size*col_size
         IF (PRESENT(blk_ind)) blk_ind(iblk, :) = [row, col]
      END DO
      CALL dbcsr_tas_iterator_stop(iter)

      IF (PRESENT(nze)) THEN
         CALL dbcsr_tas_get_split_info(dbcsr_tas_info(matrix_a), mp_comm=mp_comm)
         CALL mp_sum(nze, mp_comm)
      END IF

      IF (.NOT. retain_sparsity_prv) THEN
         CALL dbcsr_tas_destroy(matrix_c_bnorm)
         DEALLOCATE (matrix_c_bnorm)
      END IF

      CALL timestop(handle)

   END SUBROUTINE

   FUNCTION split_factor_estimate(max_mm_dim, nze_a, nze_b, nze_c, numnodes) RESULT(nsplit)
      !! Estimate optimal split factor for AxB=C from occupancies (number of non-zero elements)
      !! This estimate is based on the minimization of communication volume whereby
      !! the communication of CARMA n-split step and CANNON-multiplication of submatrices are
      !! considered.
      !! \result estimated split factor

      INTEGER, INTENT(IN)                         :: max_mm_dim
      INTEGER(KIND=int_8), INTENT(IN)             :: nze_a, nze_b, nze_c
         !! number of non-zeroes in A
         !! number of non-zeroes in B
         !! number of non-zeroes in C
      INTEGER, INTENT(IN)                         :: numnodes
         !! number of MPI ranks
      INTEGER                                     :: nsplit
      INTEGER(KIND=int_8)                         :: max_nze, min_nze
      REAL(real_8) :: s_opt_factor

      s_opt_factor = dbcsr_cfg%tas_split_factor%val

      SELECT CASE (max_mm_dim)
      CASE (1)
         min_nze = MAX(nze_b, 1_int_8)
         max_nze = MAX(MAXVAL([nze_a, nze_c]), 1_int_8)
      CASE (2)
         min_nze = MAX(nze_c, 1_int_8)
         max_nze = MAX(MAXVAL([nze_a, nze_b]), 1_int_8)
      CASE (3)
         min_nze = MAX(nze_a, 1_int_8)
         max_nze = MAX(MAXVAL([nze_b, nze_c]), 1_int_8)
      CASE DEFAULT
         DBCSR_ABORT("")
      END SELECT

      nsplit = INT(MIN(INT(numnodes, KIND=int_8), NINT(REAL(max_nze, real_8)/(REAL(min_nze, real_8)*s_opt_factor), KIND=int_8)))
      IF (nsplit == 0) nsplit = 1

   END FUNCTION

   SUBROUTINE create_block_norms_matrix(matrix_in, matrix_out, nodata)
      !! Create a matrix with block sizes one that contains the block norms of matrix_in
      TYPE(dbcsr_tas_type), INTENT(INOUT)        :: matrix_in
      TYPE(dbcsr_tas_type), INTENT(OUT)          :: matrix_out
      LOGICAL, INTENT(IN), OPTIONAL              :: nodata
      TYPE(dbcsr_tas_blk_size_one)               :: row_blk_size, col_blk_size
      TYPE(dbcsr_tas_iterator)                   :: iter
      INTEGER(KIND=int_8)                        :: row, column, nblkrows, nblkcols
      CHARACTER(len=default_string_length)       :: name
      INTEGER                                    :: data_type

      #:for dparam, dtype, dsuffix in dtype_float_list
         ${dtype}$, DIMENSION(:, :), POINTER        :: block_get_${dsuffix}$
         ${dtype}$, DIMENSION(:, :), POINTER        :: block_put_${dsuffix}$
      #:endfor
      LOGICAL                                    :: tr, nodata_prv, found

      DBCSR_ASSERT(matrix_in%valid)

      IF (PRESENT(nodata)) THEN
         nodata_prv = nodata
      ELSE
         nodata_prv = .FALSE.
      END IF

      CALL dbcsr_tas_get_info(matrix_in, data_type=data_type, name=name, &
                              nblkrows_total=nblkrows, nblkcols_total=nblkcols)

      row_blk_size = dbcsr_tas_blk_size_one(nblkrows)
      col_blk_size = dbcsr_tas_blk_size_one(nblkcols)

      ! not sure if assumption that same distribution can be taken still holds
      CALL dbcsr_tas_create(matrix_out, name, matrix_in%dist, &
                            data_type, &
                            row_blk_size, col_blk_size)

      IF (.NOT. nodata_prv) THEN
         CALL dbcsr_tas_reserve_blocks(matrix_in, matrix_out)

         CALL dbcsr_tas_iterator_start(iter, matrix_in)

         DO WHILE (dbcsr_tas_iterator_blocks_left(iter))

            #:for dparam, dtype, dsuffix in dtype_float_list
               IF (data_type == ${dparam}$) THEN
                  CALL dbcsr_tas_iterator_next_block(iter, row, column, block_get_${dsuffix}$, tr)
                  CALL dbcsr_tas_get_block_p(matrix_out, row, column, block_put_${dsuffix}$, tr, found)
                  DBCSR_ASSERT(found)
                  block_put_${dsuffix}$ (1, 1) = SQRT(SUM(block_get_${dsuffix}$**2)) ! norm2 works only for real
               END IF
            #:endfor
         END DO
         CALL dbcsr_tas_iterator_stop(iter)
      END IF

   END SUBROUTINE

   SUBROUTINE convert_to_new_pgrid(mp_comm_cart, matrix_in, matrix_out, move_data, nodata, optimize_pgrid)
      !! Convert a DBCSR matrix to a new process grid

      TYPE(mp_comm_type), INTENT(IN)             :: mp_comm_cart
         !! new process grid
      TYPE(dbcsr_type), INTENT(INOUT)            :: matrix_in
      TYPE(dbcsr_type), INTENT(OUT)              :: matrix_out
      LOGICAL, INTENT(IN), OPTIONAL              :: move_data, nodata
         !! memory optimization: move data such that matrix_in is empty on return.
         !! Data of matrix_in should not be copied to matrix_out
      LOGICAL, INTENT(IN), OPTIONAL              :: optimize_pgrid
         !! Whether to change process grid

      INTEGER                                    :: &
         nbrows, nbcols, data_type, nproc, handle
      INTEGER, DIMENSION(2)                      :: pdims, pcoord
      INTEGER, DIMENSION(:), POINTER, CONTIGUOUS :: row_dist, col_dist, rbsize, rcsize
      TYPE(dbcsr_distribution_obj)               :: dist, dist_old
      TYPE(dbcsr_mp_obj)                         :: mp_obj
      CHARACTER(len=default_string_length)       :: name
      LOGICAL                                    :: nodata_prv, optimize_pgrid_prv
      CHARACTER(LEN=*), PARAMETER                :: routineN = 'convert_to_new_pgrid'

      NULLIFY (row_dist, col_dist, rbsize, rcsize)

      CALL timeset(routineN, handle)

      IF (PRESENT(optimize_pgrid)) THEN
         optimize_pgrid_prv = optimize_pgrid
      ELSE
         optimize_pgrid_prv = .TRUE.
      END IF

      IF (PRESENT(nodata)) THEN
         nodata_prv = nodata
      ELSE
         nodata_prv = .FALSE.
      END IF

      IF (.NOT. optimize_pgrid_prv) THEN
         CALL dbcsr_create(matrix_out, template=matrix_in)
         IF (.NOT. nodata_prv) CALL dbcsr_copy(matrix_out, matrix_in)
         CALL timestop(handle)
         RETURN
      END IF

      CALL dbcsr_get_info(matrix_in, nblkrows_total=nbrows, nblkcols_total=nbcols, &
                          row_blk_size=rbsize, col_blk_size=rcsize, &
                          data_type=data_type, distribution=dist_old, name=name)
      CALL mp_environ(nproc, pdims, pcoord, mp_comm_cart)

      ALLOCATE (row_dist(nbrows), col_dist(nbcols))
      CALL dbcsr_tas_default_distvec(nbrows, pdims(1), rbsize, row_dist)
      CALL dbcsr_tas_default_distvec(nbcols, pdims(2), rcsize, col_dist)

      mp_obj = dbcsr_mp_environ(mp_comm_cart)
      CALL dbcsr_distribution_new(dist, mp_obj, row_dist, col_dist, reuse_arrays=.TRUE.)
      CALL dbcsr_mp_release(mp_obj)

      CALL dbcsr_create(matrix_out, name, dist, dbcsr_type_no_symmetry, rbsize, rcsize, data_type=data_type)
      CALL dbcsr_distribution_release(dist)

      IF (.NOT. nodata_prv) THEN
         CALL dbcsr_redistribute(matrix_in, matrix_out)
         IF (PRESENT(move_data)) THEN
            IF (move_data) CALL dbcsr_clear(matrix_in)
         END IF
      END IF

      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_batched_mm_init(matrix)
      TYPE(dbcsr_tas_type), INTENT(INOUT) :: matrix
      CALL dbcsr_tas_set_batched_state(matrix, state=1)
      ALLOCATE (matrix%mm_storage)
      matrix%mm_storage%batched_out = .FALSE.
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_batched_mm_finalize(matrix)
      TYPE(dbcsr_tas_type), INTENT(INOUT) :: matrix
      INTEGER :: handle

      CALL mp_sync(matrix%dist%info%mp_comm)
      CALL timeset("dbcsr_tas_total", handle)

      IF (matrix%do_batched == 0) RETURN

      IF (matrix%mm_storage%batched_out) THEN
         CALL dbcsr_scale(matrix%matrix, matrix%mm_storage%batched_beta)
      END IF

      CALL dbcsr_tas_batched_mm_complete(matrix)

      matrix%mm_storage%batched_out = .FALSE.

      DEALLOCATE (matrix%mm_storage)
      CALL dbcsr_tas_set_batched_state(matrix, state=0)

      CALL mp_sync(matrix%dist%info%mp_comm)
      CALL timestop(handle)

   END SUBROUTINE

   SUBROUTINE dbcsr_tas_set_batched_state(matrix, state, opt_grid)
      !! set state flags during batched multiplication

      TYPE(dbcsr_tas_type), INTENT(INOUT) :: matrix
      LOGICAL, INTENT(IN), OPTIONAL :: opt_grid
      !! whether process grid was already optimized and should not be changed
      INTEGER, INTENT(IN), OPTIONAL :: state
      !! - 0 no batched MM
      !! - 1 batched MM but mm_storage not yet initialized
      !! - 2 batched MM and mm_storage requires update
      !! - 3 batched MM and mm_storage initialized

      IF (PRESENT(opt_grid)) THEN
         matrix%has_opt_pgrid = opt_grid
         matrix%dist%info%strict_split(1) = .TRUE.
      END IF

      IF (PRESENT(state)) THEN
         matrix%do_batched = state
         SELECT CASE (state)
         CASE (0, 1)
            ! reset to default
            IF (matrix%has_opt_pgrid) THEN
               matrix%dist%info%strict_split(1) = .TRUE.
            ELSE
               matrix%dist%info%strict_split(1) = matrix%dist%info%strict_split(2)
            END IF
         CASE (2, 3)
            matrix%dist%info%strict_split(1) = .TRUE.
         CASE DEFAULT
            DBCSR_ABORT("should not happen")
         END SELECT
      END IF
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_batched_mm_complete(matrix, warn)
      TYPE(dbcsr_tas_type), INTENT(INOUT) :: matrix
      LOGICAL, INTENT(IN), OPTIONAL :: warn

      IF (matrix%do_batched == 0) RETURN
      ASSOCIATE (storage => matrix%mm_storage)
         IF (PRESENT(warn)) THEN
            IF (warn .AND. matrix%do_batched == 3) THEN
               CALL dbcsr_warn(__LOCATION__, &
                               "Optimizations for batched multiplication are disabled because of conflicting data access")
            END IF
         END IF
         IF (storage%batched_out .AND. matrix%do_batched == 3) THEN

            CALL dbcsr_tas_merge(storage%store_batched%matrix, &
                                 storage%store_batched_repl, move_data=.TRUE.)

            CALL dbcsr_tas_reshape(storage%store_batched, matrix, summation=.TRUE., &
                                   transposed=storage%batched_trans, move_data=.TRUE.)
            CALL dbcsr_tas_destroy(storage%store_batched)
            DEALLOCATE (storage%store_batched)
         END IF

         IF (ASSOCIATED(storage%store_batched_repl)) THEN
            CALL dbcsr_tas_destroy(storage%store_batched_repl)
            DEALLOCATE (storage%store_batched_repl)
         END IF
      END ASSOCIATE

      CALL dbcsr_tas_set_batched_state(matrix, state=2)

   END SUBROUTINE

END MODULE
