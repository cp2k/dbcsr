!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Matrix multiplication for tall-and-skinny matrices. This uses the k-split (non-recursive) CARMA
!>        algorithm that is communication-optimal as long as the two smaller dimensions have
!>        the same size.
!>
!> Submatrices are obtained by splitting a dimension of the process grid. Multiplication of
!> submatrices uses DBCSR Cannon algorithm. Due to unknown sparsity pattern of result matrix, parameters
!> (group sizes and process grid dimensions) can not be derived from matrix dimensions and need to be
!> set manually.
!> \author Patrick Seewald
! **************************************************************************************************

MODULE dbcsr_tas_mm

   USE dbcsr_data_methods, ONLY: dbcsr_scalar_zero
   USE dbcsr_data_types, ONLY: dbcsr_scalar_type
   USE dbcsr_multiply_api, ONLY: dbcsr_multiply
   USE dbcsr_tas_base, ONLY: &
      dbcsr_tas_create, dbcsr_tas_destroy, dbcsr_tas_distribution_destroy, dbcsr_tas_distribution_new, &
      dbcsr_tas_get_data_type, dbcsr_tas_info, dbcsr_tas_nblkcols_total, &
      dbcsr_tas_nblkrows_total, dbcsr_tas_filter
   USE dbcsr_tas_types, ONLY: dbcsr_tas_distribution_type, &
                              dbcsr_tas_create_split_info, &
                              dbcsr_tas_type
   USE dbcsr_tas_global, ONLY: dbcsr_tas_dist_cyclic, &
                               dbcsr_tas_dist_arb, &
                               dbcsr_tas_distribution, &
                               dbcsr_tas_dist_arb_default
   USE dbcsr_tas_reshape_ops, ONLY: dbcsr_tas_merge, &
                                    dbcsr_tas_replicate, &
                                    dbcsr_tas_reshape
   USE dbcsr_tas_split, ONLY: colsplit, &
                              dbcsr_tas_get_split_info, &
                              dbcsr_tas_create_split, &
                              dbcsr_tas_mp_comm, &
                              dbcsr_tas_release_info
   USE dbcsr_tas_util, ONLY: swap, &
                             invert_transpose_flag, &
                             array_eq_i8
   USE dbcsr_types, ONLY: dbcsr_no_transpose, &
                          dbcsr_transpose
   USE dbcsr_kinds, ONLY: int_8, &
                          real_8
   USE dbcsr_mpiwrap, ONLY: mp_comm_compare, &
                            mp_environ, &
                            mp_sum, &
                            mp_comm_free
   USE dbcsr_operations, ONLY: dbcsr_scale
   USE dbcsr_tas_io, ONLY: dbcsr_tas_write_dist, &
                           dbcsr_tas_write_matrix_info, &
                           dbcsr_tas_write_split_info
#include "../base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tas_mm'

   PUBLIC :: dbcsr_tas_multiply

CONTAINS

! **************************************************************************************************
!> \brief tall-and-skinny matrix-matrix multiplication. Undocumented dummy arguments are identical to
!> arguments of dbcsr_multiply (see dbcsr_mm, dbcsr_multiply_generic).
!> \param transa ...
!> \param transb ...
!> \param transc ...
!> \param alpha ...
!> \param matrix_a ...
!> \param matrix_b ...
!> \param beta ...
!> \param matrix_c ...
!> \param optimize_dist Whether distribution should be optimized internally. Experimental and should
!> be used for dense matrices only.
!> \param filter_eps ...
!> \param flop ...
!> \param move_data memory optimization: move data to matrix_c such that matrix_a and matrix_b are
!> empty on return
!> \param io_unit unit number for logging output
!> \param log_verbose only for testing: verbose output
! **************************************************************************************************
   SUBROUTINE dbcsr_tas_multiply(transa, transb, transc, alpha, matrix_a, matrix_b, beta, matrix_c, &
                                 optimize_dist, filter_eps, flop, move_data, io_unit, log_verbose)
      CHARACTER(LEN=1), INTENT(IN)               :: transa, transb, transc
      TYPE(dbcsr_scalar_type), INTENT(IN)        :: alpha, beta
      TYPE(dbcsr_tas_type), TARGET, INTENT(INOUT)  :: matrix_a, matrix_b, matrix_c
      LOGICAL, INTENT(IN), OPTIONAL              :: optimize_dist
      REAL(KIND=real_8), INTENT(IN), OPTIONAL    :: filter_eps
      INTEGER(KIND=int_8), INTENT(OUT), OPTIONAL :: flop
      LOGICAL, INTENT(IN), OPTIONAL              :: move_data
      INTEGER, OPTIONAL, INTENT(IN)              :: io_unit
      LOGICAL, OPTIONAL, INTENT(IN)              :: log_verbose

      TYPE(dbcsr_tas_type), POINTER                :: matrix_b_rs => NULL(), matrix_a_rs => NULL(), matrix_c_rs => NULL()
      TYPE(dbcsr_tas_type)                         :: matrix_c_rep, matrix_b_rep, matrix_a_rep

      REAL(KIND=real_8)                          :: filter_eps_prv
      INTEGER(KIND=int_8), DIMENSION(2)          :: dims_a, dims_b, dims_c
      INTEGER, DIMENSION(2) :: pdims, pcoord
      INTEGER(KIND=int_8), DIMENSION(3)          :: dims
      INTEGER                                    :: max_mm_dim, data_type, numnodes, mp_comm, comm_tmp, &
                                                    handle, handle2, io_unit_prv, nsplit
      CHARACTER(LEN=1)                           :: tr_case, transa_prv, transb_prv, transc_prv
      TYPE(dbcsr_scalar_type)                    :: zero
      LOGICAL                                    :: new_a, new_b, new_c, move_prv
      TYPE(dbcsr_tas_create_split_info)                   :: info
      CHARACTER(LEN=*), PARAMETER                :: routineN = 'dbcsr_tas_multiply', &
                                                    routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)

      IF (PRESENT(io_unit)) THEN
         io_unit_prv = io_unit
      ELSE
         io_unit_prv = 0
      ENDIF

      IF (PRESENT(move_data)) THEN
         move_prv = move_data
      ELSE
         move_prv = .FALSE.
      ENDIF

      IF (.NOT. dbcsr_tas_get_data_type(matrix_a) .EQ. dbcsr_tas_get_data_type(matrix_b)) THEN
         DBCSR_ABORT("matrices must have same datatype")
      ENDIF

      data_type = dbcsr_tas_get_data_type(matrix_a)

      transa_prv = transa; transb_prv = transb; transc_prv = transc

      dims_a = [dbcsr_tas_nblkrows_total(matrix_a), dbcsr_tas_nblkcols_total(matrix_a)]
      dims_b = [dbcsr_tas_nblkrows_total(matrix_b), dbcsr_tas_nblkcols_total(matrix_b)]
      dims_c = [dbcsr_tas_nblkrows_total(matrix_c), dbcsr_tas_nblkcols_total(matrix_c)]

      IF (PRESENT(io_unit)) THEN
         IF (io_unit_prv .GT. 0) THEN
            WRITE (io_unit_prv, '(A)') repeat("-", 80)
            WRITE (io_unit_prv, '(A,1X,A,1X,A,1X,A,1X,A,1X,A)') "DBCSR TALL-AND-SKINNY MATRIX MULTIPLICATION:", &
               TRIM(matrix_a%matrix%name), 'x', TRIM(matrix_b%matrix%name), '=', TRIM(matrix_c%matrix%name)
            WRITE (io_unit_prv, '(A)') repeat("-", 80)
         ENDIF

      ENDIF

      IF (transa_prv .EQ. dbcsr_transpose) THEN
         CALL swap(dims_a)
      ENDIF

      IF (transb_prv .EQ. dbcsr_transpose) THEN
         CALL swap(dims_b)
      ENDIF

      dims_c = [dims_a(1), dims_b(2)]

      IF (.NOT. (dims_a(2) .EQ. dims_b(1))) THEN
         DBCSR_ABORT("inconsistent matrix dimensions")
      ENDIF

      dims(:) = [dims_a(1), dims_a(2), dims_b(2)]

      max_mm_dim = MAXLOC(dims, 1)

      tr_case = ''

      IF (io_unit_prv > 0) THEN
         WRITE (io_unit_prv, "(T2,A, 1X, I12, 1X, I12, 1X, I12)") "mm dims:", dims(1), dims(2), dims(3)
      ENDIF

      ! ideally decision which matrix to reshape should be based on occupancy not on size
      ! but since we don't know occupancy of result matrix, heuristics is based on size

      SELECT CASE (max_mm_dim)
      CASE (1)

         CALL reshape_mm_compatible(matrix_a, matrix_c, matrix_a_rs, matrix_c_rs, &
                             new_a, new_c, transa_prv, transc_prv, optimize_dist=optimize_dist, nodata2=.TRUE., comm_new=comm_tmp, &
                                    move_data=move_data, unit_nr=io_unit)

         CALL dbcsr_tas_get_split_info(dbcsr_tas_info(matrix_a_rs), mp_comm=mp_comm)

         ALLOCATE (matrix_b_rs)
         CALL reshape_mm_small(mp_comm, matrix_b, matrix_b_rs, transb_prv == dbcsr_transpose, transb_prv, move_data=move_data)
         new_b = .TRUE.
         tr_case = transa_prv

         IF (io_unit_prv > 0) THEN
            IF (tr_case == 'N') THEN
               WRITE (io_unit_prv, "(T2,A, 1X, A)") "mm case:", "| x + = |"
            ELSE
               WRITE (io_unit_prv, "(T2,A, 1X, A)") "mm case:", "--T x + = --T"
            ENDIF
         ENDIF

      CASE (2)

         CALL reshape_mm_compatible(matrix_a, matrix_b, matrix_a_rs, matrix_b_rs, new_a, new_b, transa_prv, transb_prv, &
                                    optimize_dist=optimize_dist, comm_new=comm_tmp, &
                                    move_data=move_data, unit_nr=io_unit)

         CALL dbcsr_tas_get_split_info(dbcsr_tas_info(matrix_a_rs), mp_comm=mp_comm)

         ALLOCATE (matrix_c_rs)
         CALL reshape_mm_small(mp_comm, matrix_c, matrix_c_rs, transc_prv == dbcsr_transpose, transc_prv, nodata=.TRUE.)
         new_c = .TRUE.
         tr_case = transa_prv

         IF (io_unit_prv > 0) THEN
            IF (tr_case == 'N') THEN
               WRITE (io_unit_prv, "(T2,A, 1X, A)") "mm case:", "-- x --T = +"
            ELSE
               WRITE (io_unit_prv, "(T2,A, 1X, A)") "mm case:", "|T x | = +"
            ENDIF
         ENDIF

      CASE (3)

         CALL reshape_mm_compatible(matrix_b, matrix_c, matrix_b_rs, matrix_c_rs, new_b, new_c, transb_prv, &
                                    transc_prv, optimize_dist=optimize_dist, nodata2=.TRUE., comm_new=comm_tmp, &
                                    move_data=move_data, unit_nr=io_unit)

         CALL dbcsr_tas_get_split_info(dbcsr_tas_info(matrix_b_rs), mp_comm=mp_comm)

         ALLOCATE (matrix_a_rs)
         CALL reshape_mm_small(mp_comm, matrix_a, matrix_a_rs, transa_prv == dbcsr_transpose, transa_prv, move_data=move_data)
         new_a = .TRUE.
         tr_case = transb_prv

         IF (io_unit_prv > 0) THEN
            IF (tr_case == 'N') THEN
               WRITE (io_unit_prv, "(T2,A, 1X, A)") "mm case:", "+ x -- = --"
            ELSE
               WRITE (io_unit_prv, "(T2,A, 1X, A)") "mm case:", "+ x |T = |T"
            ENDIF
         ENDIF

      END SELECT

      info = dbcsr_tas_info(matrix_a_rs)
      CALL dbcsr_tas_get_split_info(info, nsplit=nsplit, mp_comm=mp_comm)
      CALL mp_environ(numnodes, pdims, pcoord, mp_comm)

      IF (PRESENT(filter_eps)) THEN
         filter_eps_prv = filter_eps
      ELSE
         filter_eps_prv = 0.0_real_8
      ENDIF

      IF (PRESENT(io_unit)) THEN
         IF (io_unit_prv > 0) THEN
            WRITE (io_unit_prv, "(T2, A)") "SPLIT INFO"
         ENDIF
         CALL dbcsr_tas_write_split_info(info, "Split info:", io_unit_prv)
         CALL dbcsr_tas_write_matrix_info(matrix_a_rs, io_unit_prv, full_info=log_verbose)
         CALL dbcsr_tas_write_matrix_info(matrix_b_rs, io_unit_prv, full_info=log_verbose)
         CALL dbcsr_tas_write_matrix_info(matrix_c_rs, io_unit_prv, full_info=log_verbose)
      ENDIF

      zero = dbcsr_scalar_zero(data_type)
      SELECT CASE (max_mm_dim)
      CASE (1)
         CALL dbcsr_tas_replicate(matrix_b_rs%matrix, dbcsr_tas_info(matrix_a_rs), matrix_b_rep, move_data=.TRUE.)
         SELECT CASE (tr_case)
         CASE (dbcsr_no_transpose)
            CALL timeset(routineN//"_dbcsr_mm_1N", handle2)

            CALL dbcsr_multiply(transa=dbcsr_no_transpose, transb=dbcsr_no_transpose, alpha=alpha, &
                                matrix_a=matrix_a_rs%matrix, matrix_b=matrix_b_rep%matrix, beta=beta, matrix_c=matrix_c_rs%matrix, &
                                filter_eps=filter_eps_prv, flop=flop)
            CALL timestop(handle2)
         CASE (dbcsr_transpose)
            CALL timeset(routineN//"_dbcsr_mm_1T", handle2)
            CALL dbcsr_multiply(transa=dbcsr_transpose, transb=dbcsr_no_transpose, alpha=alpha, &
                                matrix_a=matrix_b_rep%matrix, matrix_b=matrix_a_rs%matrix, beta=beta, matrix_c=matrix_c_rs%matrix, &
                                filter_eps=filter_eps_prv, flop=flop)

            CALL timestop(handle2)
         END SELECT
         IF (PRESENT(io_unit)) THEN
            CALL dbcsr_tas_write_dist(matrix_a_rs, io_unit)
            CALL dbcsr_tas_write_dist(matrix_b_rep, io_unit, full_info=log_verbose)
            CALL dbcsr_tas_write_dist(matrix_c_rs, io_unit)
         ENDIF
         CALL dbcsr_tas_distribution_destroy(matrix_b_rep%dist)
         CALL dbcsr_tas_destroy(matrix_b_rep)
      CASE (2)
         CALL dbcsr_tas_replicate(matrix_c_rs%matrix, dbcsr_tas_info(matrix_a_rs), matrix_c_rep, nodata=.TRUE.)

         CALL timeset(routineN//"_dbcsr_mm_2", handle2)
         CALL dbcsr_multiply(transa=transa_prv, transb=transb_prv, alpha=alpha, matrix_a=matrix_a_rs%matrix, &
                             matrix_b=matrix_b_rs%matrix, beta=beta, matrix_c=matrix_c_rep%matrix, &
                             filter_eps=filter_eps_prv/REAL(nsplit, KIND=real_8), flop=flop)
         CALL timestop(handle2)
         IF (PRESENT(io_unit)) THEN
            CALL dbcsr_tas_write_dist(matrix_a_rs, io_unit)
            CALL dbcsr_tas_write_dist(matrix_b_rs, io_unit)
            CALL dbcsr_tas_write_dist(matrix_c_rep, io_unit, full_info=log_verbose)
         ENDIF

         CALL dbcsr_tas_merge(matrix_c_rs%matrix, matrix_c_rep, move_data=.TRUE.)
         CALL dbcsr_tas_filter(matrix_c_rs, filter_eps_prv)
         CALL dbcsr_tas_distribution_destroy(matrix_c_rep%dist)
         CALL dbcsr_tas_destroy(matrix_c_rep)
      CASE (3)
         CALL dbcsr_tas_replicate(matrix_a_rs%matrix, dbcsr_tas_info(matrix_b_rs), matrix_a_rep, move_data=.TRUE.)
         SELECT CASE (tr_case)
         CASE (dbcsr_no_transpose)
            CALL timeset(routineN//"_dbcsr_mm_3N", handle2)
            CALL dbcsr_multiply(transa=dbcsr_no_transpose, transb=dbcsr_no_transpose, alpha=alpha, &
                                matrix_a=matrix_a_rep%matrix, matrix_b=matrix_b_rs%matrix, beta=beta, matrix_c=matrix_c_rs%matrix, &
                                filter_eps=filter_eps_prv, flop=flop)
            CALL timestop(handle2)
         CASE (dbcsr_transpose)
            CALL timeset(routineN//"_dbcsr_mm_3T", handle2)
            CALL dbcsr_multiply(transa=dbcsr_no_transpose, transb=dbcsr_transpose, alpha=alpha, &
                                matrix_a=matrix_b_rs%matrix, matrix_b=matrix_a_rep%matrix, beta=beta, matrix_c=matrix_c_rs%matrix, &
                                filter_eps=filter_eps_prv, flop=flop)
            CALL timestop(handle2)
         END SELECT
         IF (PRESENT(io_unit)) THEN
            CALL dbcsr_tas_write_dist(matrix_a_rep, io_unit, full_info=log_verbose)
            CALL dbcsr_tas_write_dist(matrix_b_rs, io_unit)
            CALL dbcsr_tas_write_dist(matrix_c_rs, io_unit)
         ENDIF
         CALL dbcsr_tas_distribution_destroy(matrix_a_rep%dist)
         CALL dbcsr_tas_destroy(matrix_a_rep)
      END SELECT

      IF (new_a) THEN
         CALL dbcsr_tas_destroy(matrix_a_rs)
         DEALLOCATE (matrix_a_rs)
      ENDIF
      IF (new_b) THEN
         CALL dbcsr_tas_destroy(matrix_b_rs)
         DEALLOCATE (matrix_b_rs)
      ENDIF
      IF (new_c) THEN
         CALL dbcsr_scale(matrix_c%matrix, beta)
         CALL dbcsr_tas_reshape(matrix_c_rs, matrix_c, summation=.TRUE., transposed=transc_prv /= transc, &
                                move_data=.TRUE.)
         IF (PRESENT(filter_eps)) CALL dbcsr_tas_filter(matrix_c, filter_eps)
         CALL dbcsr_tas_destroy(matrix_c_rs)
         DEALLOCATE (matrix_c_rs)
      ENDIF

      IF (PRESENT(flop)) THEN
         CALL mp_sum(flop, mp_comm)
         flop = (flop + numnodes - 1)/numnodes
      ENDIF

      IF (PRESENT(optimize_dist)) THEN
         IF (optimize_dist) CALL mp_comm_free(comm_tmp)
      ENDIF
      IF (io_unit_prv > 0) THEN
         WRITE (io_unit_prv, '(A)') repeat("-", 80)
         WRITE (io_unit_prv, '(A,1X,A,1X,A,1X,A,1X,A,1X,A)') "TALL-AND-SKINNY MATRIX MULTIPLICATION DONE"
         WRITE (io_unit_prv, '(A)') repeat("-", 80)
      ENDIF

      CALL timestop(handle)

   END SUBROUTINE

! **************************************************************************************************
!> \brief Make sure that smallest matrix involved in a multiplication is not split and bring it to
!>        the same process grid as the other 2 matrices.
!> \param mp_comm communicator that defines Cartesian topology
!> \param matrix_in ...
!> \param matrix_out ...
!> \param transposed Whether matrix_out should be transposed
!> \param trans update transpose flag for DBCSR mm according to 'transposed' argument
!> \param nodata Data of matrix_in should not be copied to matrix_out
!> \param move_data memory optimization: move data such that matrix_in is empty on return.
! **************************************************************************************************
   SUBROUTINE reshape_mm_small(mp_comm, matrix_in, matrix_out, transposed, trans, nodata, move_data)
      INTEGER, INTENT(IN)               :: mp_comm
      TYPE(dbcsr_tas_type), INTENT(INOUT) :: matrix_in
      TYPE(dbcsr_tas_type), INTENT(OUT)   :: matrix_out
      LOGICAL, INTENT(IN)               :: transposed
      CHARACTER(LEN=1), INTENT(INOUT)   :: trans
      LOGICAL, INTENT(IN), OPTIONAL     :: nodata, move_data

      INTEGER                           :: numnodes
      INTEGER(KIND=int_8), DIMENSION(2) :: dims
      INTEGER, DIMENSION(2)             :: pdims, pcoord
      TYPE(dbcsr_tas_dist_arb)            :: new_row_dist, new_col_dist
      TYPE(dbcsr_tas_distribution_type)   :: dist
      LOGICAL                           :: nodata_prv
      CHARACTER(LEN=*), PARAMETER       :: routineN = 'reshape_mm_small', &
                                           routineP = moduleN//':'//routineN
      INTEGER                           :: handle

      CALL timeset(routineN, handle)

      IF (PRESENT(nodata)) THEN
         nodata_prv = nodata
      ELSE
         nodata_prv = .FALSE.
      ENDIF

      IF (transposed) THEN
         SELECT CASE (trans)
         CASE (dbcsr_transpose)
            trans = dbcsr_no_transpose
         CASE (dbcsr_no_transpose)
            trans = dbcsr_transpose
         END SELECT
      ENDIF

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

      ENDIF
      IF (.NOT. nodata_prv) CALL dbcsr_tas_reshape(matrix_in, matrix_out, transposed=transposed, move_data=move_data)

      CALL timestop(handle)

   END SUBROUTINE

! **************************************************************************************************
!> \brief Reshape either matrix1 or matrix2 to make sure that their process grids are compatible with
!> the same split factor.
!> \param matrix1_in ...
!> \param matrix2_in ...
!> \param matrix1_out ...
!> \param matrix2_out ...
!> \param new1 Whether matrix1_out is a new matrix or simply pointing to matrix1_in
!> \param new2 Whether matrix2_out is a new matrix or simply pointing to matrix2_in
!> \param trans1 transpose flag of matrix1_in for multiplication
!> \param trans2 transpose flag of matrix2_in for multiplication
!> \param optimize_dist experimental: optimize matrix splitting and distribution
!> \param nodata1 Don't copy matrix data from matrix1_in to matrix1_out
!> \param nodata2 Don't copy matrix data from matrix2_in to matrix2_out
!> \param move_data memory optimization: move data such that matrix*_in are empty on return.
!> \param comm_new returns the new communicator only if optimize_dist
!> \param unit_nr output unit
! **************************************************************************************************
   SUBROUTINE reshape_mm_compatible(matrix1_in, matrix2_in, matrix1_out, matrix2_out, new1, new2, trans1, trans2, &
                                    optimize_dist, nodata1, nodata2, move_data, comm_new, unit_nr)
      TYPE(dbcsr_tas_type), TARGET, INTENT(INOUT) :: matrix1_in, matrix2_in
      TYPE(dbcsr_tas_type), POINTER, INTENT(OUT) :: matrix1_out, matrix2_out
      LOGICAL, INTENT(OUT) :: new1, new2
      CHARACTER(LEN=1), INTENT(INOUT) :: trans1, trans2
      LOGICAL, INTENT(IN), OPTIONAL :: optimize_dist
      INTEGER, INTENT(OUT), OPTIONAL :: comm_new
      LOGICAL, OPTIONAL, INTENT(IN) :: nodata1, nodata2, move_data
      INTEGER, INTENT(IN), OPTIONAL :: unit_nr

      INTEGER(KIND=int_8), DIMENSION(2) :: dims1, dims2
      CHARACTER(LEN=*), PARAMETER :: routineN = 'reshape_mm_compatible', &
                                     routineP = moduleN//':'//routineN
      INTEGER :: handle, mp_comm, numnodes, split_rc_1, split_rc_2, io_unit
      INTEGER, DIMENSION(2) :: pcoord, pdims
      LOGICAL :: optimize_dist_prv, trans1_newdist, trans2_newdist
      TYPE(dbcsr_tas_dist_cyclic) :: row_dist_1, col_dist_1, row_dist_2, col_dist_2
      TYPE(dbcsr_tas_distribution_type) :: dist_1, dist_2
      TYPE(dbcsr_tas_create_split_info) :: split_info

      IF (PRESENT(unit_nr)) THEN
         io_unit = unit_nr
      ELSE
         io_unit = 0
      ENDIF

      IF (PRESENT(optimize_dist)) THEN
         optimize_dist_prv = optimize_dist
      ELSE
         optimize_dist_prv = .FALSE.
      ENDIF

      IF (optimize_dist_prv) THEN
         DBCSR_ASSERT(PRESENT(comm_new))
      ENDIF

      CALL timeset(routineN, handle)
      new1 = .FALSE.; new2 = .FALSE.

      IF ((.NOT. optimize_dist_prv) .AND. dist_compatible(matrix1_in, matrix2_in)) THEN
         matrix1_out => matrix1_in
         matrix2_out => matrix2_in
         IF (io_unit > 0) THEN
            WRITE (io_unit, "(T2,A,1X,A,1X,A,1X,A)") "No redistribution of", TRIM(matrix1_in%matrix%name), &
               "and", TRIM(matrix2_in%matrix%name)
         ENDIF
      ELSE

         dims1 = [dbcsr_tas_nblkrows_total(matrix1_in), dbcsr_tas_nblkcols_total(matrix1_in)]
         dims2 = [dbcsr_tas_nblkrows_total(matrix2_in), dbcsr_tas_nblkcols_total(matrix2_in)]

         IF (optimize_dist_prv) THEN
            IF (io_unit > 0) THEN
               WRITE (io_unit, "(T2,A,1X,A,1X,A,1X,A)") "Optimizing distribution of", TRIM(matrix1_in%matrix%name), &
                  "and", TRIM(matrix2_in%matrix%name)
            ENDIF
            CALL dbcsr_tas_get_split_info(dbcsr_tas_info(matrix1_in), mp_comm=mp_comm, split_rowcol=split_rc_1)
            CALL dbcsr_tas_get_split_info(dbcsr_tas_info(matrix2_in), split_rowcol=split_rc_2)

            trans1_newdist = (split_rc_1 == colsplit)
            trans2_newdist = (split_rc_2 == colsplit)

            IF (trans1_newdist) THEN
               CALL swap(dims1)
               CALL invert_transpose_flag(trans1)
            ENDIF

            IF (trans2_newdist) THEN
               CALL swap(dims2)
               CALL invert_transpose_flag(trans2)
            ENDIF

            IF (PRODUCT(dims1) .GE. PRODUCT(dims2)) THEN
               comm_new = dbcsr_tas_mp_comm(mp_comm, dims1(1), dims1(2))
               CALL dbcsr_tas_create_split(split_info, comm_new, dims1(1), dims1(2))
            ELSE
               comm_new = dbcsr_tas_mp_comm(mp_comm, dims2(1), dims2(2))
               CALL dbcsr_tas_create_split(split_info, comm_new, dims2(1), dims2(2))
            ENDIF

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
            ENDIF

            ALLOCATE (matrix2_out)
            IF (.NOT. trans2_newdist) THEN
               CALL dbcsr_tas_create(matrix2_out, matrix2_in%matrix%name, dist_2, dbcsr_tas_get_data_type(matrix2_in), &
                                     matrix2_in%row_blk_size, matrix2_in%col_blk_size, own_dist=.TRUE.)
            ELSE
               CALL dbcsr_tas_create(matrix2_out, matrix2_in%matrix%name, dist_2, dbcsr_tas_get_data_type(matrix2_in), &
                                     matrix2_in%col_blk_size, matrix2_in%row_blk_size, own_dist=.TRUE.)
            ENDIF

            CALL dbcsr_tas_reshape(matrix1_in, matrix1_out, transposed=trans1_newdist, move_data=move_data)
            CALL dbcsr_tas_reshape(matrix2_in, matrix2_out, transposed=trans2_newdist, move_data=move_data)
            new1 = .TRUE.
            new2 = .TRUE.

         ELSE
            IF (PRODUCT(dims1) .GE. PRODUCT(dims2)) THEN
               IF (io_unit > 0) THEN
                  WRITE (io_unit, "(T2,A,1X,A)") "Redistribution of", TRIM(matrix2_in%matrix%name)
               ENDIF

               ALLOCATE (matrix2_out)
               CALL reshape_mm_template(matrix1_in, matrix2_in, matrix2_out, trans2, nodata=nodata2, move_data=move_data)
               new2 = .TRUE.
               matrix1_out => matrix1_in
            ELSE
               IF (io_unit > 0) THEN
                  WRITE (io_unit, "(T2,A,1X,A)") "Redistribution of", TRIM(matrix1_in%matrix%name)
               ENDIF
               ALLOCATE (matrix1_out)
               CALL reshape_mm_template(matrix2_in, matrix1_in, matrix1_out, trans1, nodata=nodata1, move_data=move_data)
               new1 = .TRUE.
               matrix2_out => matrix2_in
            ENDIF
         ENDIF
      ENDIF

      CALL timestop(handle)

   END SUBROUTINE

! **************************************************************************************************
!> \brief Check whether matrices have same distribution and same split.
!> \param mat_a ...
!> \param mat_b ...
!> \param io_unit ...
!> \return ...
! **************************************************************************************************
   FUNCTION dist_compatible(mat_a, mat_b, io_unit)
      TYPE(dbcsr_tas_type), INTENT(IN)                     :: mat_a, mat_b
      INTEGER, INTENT(IN), OPTIONAL :: io_unit
      LOGICAL                                            :: dist_compatible

      INTEGER                                            :: res, same_local_rowcols
      TYPE(dbcsr_tas_create_split_info)                           :: info_a, info_b
      INTEGER :: io_unit_prv, numproc, iproc

      IF (PRESENT(io_unit)) THEN
         io_unit_prv = io_unit
      ELSE
         io_unit_prv = 0
      ENDIF

      dist_compatible = .FALSE.

      info_a = dbcsr_tas_info(mat_a)
      info_b = dbcsr_tas_info(mat_b)

      CALL mp_environ(numproc, iproc, info_a%mp_comm)

      CALL mp_comm_compare(info_a%mp_comm_group, info_b%mp_comm_group, res)
      IF (res .GT. 1) THEN
         RETURN
      ENDIF

      IF (io_unit_prv > 0) THEN
         WRITE (io_unit_prv, *) "mp comm compatible"
      ENDIF

      IF (mat_a%dist%info%split_rowcol == mat_b%dist%info%split_rowcol) THEN

         IF (io_unit_prv > 0) THEN
            WRITE (io_unit_prv, *) "split compatible"
         ENDIF

         same_local_rowcols = MERGE(1, 0, array_eq_i8(mat_a%dist%local_rowcols, mat_b%dist%local_rowcols))
         CALL mp_sum(same_local_rowcols, info_a%mp_comm)

         IF (same_local_rowcols == numproc) THEN
            IF (io_unit_prv > 0) THEN
               WRITE (io_unit_prv, *) "local rowcols compatible"
            ENDIF
            dist_compatible = .TRUE.
         ELSE
            IF (io_unit_prv > 0) THEN
               WRITE (io_unit_prv, *) "local rowcols A", mat_a%dist%local_rowcols
               WRITE (io_unit_prv, *) "local rowcols B", mat_b%dist%local_rowcols
            ENDIF
         ENDIF
      ENDIF

      IF (io_unit_prv > 0) THEN
         WRITE (io_unit_prv, *) "is compatible?", dist_compatible
      ENDIF

   END FUNCTION

! **************************************************************************************************
!> \brief Reshape matrix_in s.t. it has same process grid, distribution and split as matrix_in
!> \param template ...
!> \param matrix_in ...
!> \param matrix_out ...
!> \param trans ...
!> \param nodata ...
!> \param move_data ...
! **************************************************************************************************
   SUBROUTINE reshape_mm_template(template, matrix_in, matrix_out, trans, nodata, move_data)
      TYPE(dbcsr_tas_type), INTENT(IN) :: template
      TYPE(dbcsr_tas_type), INTENT(INOUT) :: matrix_in
      TYPE(dbcsr_tas_type), INTENT(OUT) :: matrix_out
      CHARACTER(LEN=1), INTENT(INOUT) :: trans
      LOGICAL, INTENT(IN), OPTIONAL :: nodata, move_data
      CLASS(dbcsr_tas_distribution), ALLOCATABLE :: row_dist, col_dist

      TYPE(dbcsr_tas_distribution_type) :: dist_new
      TYPE(dbcsr_tas_create_split_info) :: info_template, info_matrix
      INTEGER :: mp_comm, dim_split_template, dim_split_matrix
      LOGICAL :: nodata_prv
      INTEGER, DIMENSION(2)                              :: pcoord, pdims
      INTEGER :: numnodes, handle
      LOGICAL :: transposed

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reshape_mm_template', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)

      IF (PRESENT(nodata)) THEN
         nodata_prv = nodata
      ELSE
         nodata_prv = .FALSE.
      ENDIF

      info_template = dbcsr_tas_info(template)
      info_matrix = dbcsr_tas_info(matrix_in)

      dim_split_template = info_template%split_rowcol
      dim_split_matrix = info_matrix%split_rowcol

      transposed = dim_split_template .NE. dim_split_matrix
      IF (transposed) THEN
         SELECT CASE (trans)
         CASE (dbcsr_transpose)
            trans = dbcsr_no_transpose
         CASE (dbcsr_no_transpose)
            trans = dbcsr_transpose
         END SELECT
      ENDIF

      CALL mp_environ(numnodes, pdims, pcoord, info_template%mp_comm)

      SELECT CASE (dim_split_template)
      CASE (1)
         IF (.NOT. transposed) THEN
            ALLOCATE (row_dist, source=template%dist%row_dist)
            ALLOCATE (col_dist, source=dbcsr_tas_dist_arb_default(pdims(2), matrix_in%nblkcols, matrix_in%col_blk_size))
         ELSE
            ALLOCATE (row_dist, source=template%dist%row_dist)
            ALLOCATE (col_dist, source=dbcsr_tas_dist_arb_default(pdims(2), matrix_in%nblkrows, matrix_in%row_blk_size))
         ENDIF
      CASE (2)
         IF (.NOT. transposed) THEN
            ALLOCATE (row_dist, source=dbcsr_tas_dist_arb_default(pdims(1), matrix_in%nblkrows, matrix_in%row_blk_size))
            ALLOCATE (col_dist, source=template%dist%col_dist)
         ELSE
            ALLOCATE (row_dist, source=dbcsr_tas_dist_arb_default(pdims(1), matrix_in%nblkcols, matrix_in%col_blk_size))
            ALLOCATE (col_dist, source=template%dist%col_dist)
         ENDIF
      END SELECT

      CALL dbcsr_tas_get_split_info(info_template, mp_comm=mp_comm)
      CALL dbcsr_tas_distribution_new(dist_new, mp_comm, row_dist, col_dist, split_info=info_template)
      IF (.NOT. transposed) THEN
         CALL dbcsr_tas_create(matrix_out, matrix_in%matrix%name, dist_new, dbcsr_tas_get_data_type(matrix_in), &
                               matrix_in%row_blk_size, matrix_in%col_blk_size, own_dist=.TRUE.)
      ELSE
         CALL dbcsr_tas_create(matrix_out, matrix_in%matrix%name, dist_new, dbcsr_tas_get_data_type(matrix_in), &
                               matrix_in%col_blk_size, matrix_in%row_blk_size, own_dist=.TRUE.)
      ENDIF

      IF (.NOT. nodata_prv) CALL dbcsr_tas_reshape(matrix_in, matrix_out, transposed=transposed, move_data=move_data)

      CALL timestop(handle)

   END SUBROUTINE

END MODULE
