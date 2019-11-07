!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tas_base
   !! Tall-and-skinny matrices: base routines similar to DBCSR API, mostly wrappers around existing
   !! DBCSR routines.

#:include "dbcsr_tas.fypp"

   USE dbcsr_block_access, ONLY: &
      dbcsr_get_block_p, dbcsr_put_block, dbcsr_reserve_blocks
   USE dbcsr_data_methods, ONLY: &
      dbcsr_data_new, dbcsr_data_release, dbcsr_type_1d_to_2d
   USE dbcsr_data_methods_low, ONLY: &
      dbcsr_data_clear_pointer, dbcsr_data_init
   USE dbcsr_data_types, ONLY: &
      dbcsr_data_obj, dbcsr_scalar_type
   USE dbcsr_dist_methods, ONLY: &
      dbcsr_distribution_col_dist, dbcsr_distribution_new, dbcsr_distribution_row_dist, dbcsr_distribution_hold
   USE dbcsr_iterator_operations, ONLY: &
      dbcsr_iterator_blocks_left, dbcsr_iterator_next_block, dbcsr_iterator_start, dbcsr_iterator_stop
   USE dbcsr_methods, ONLY: &
      dbcsr_distribution_release, dbcsr_get_data_type, dbcsr_mp_release, dbcsr_nblkcols_total, &
      dbcsr_nblkrows_total, dbcsr_nfullrows_total, dbcsr_nfullcols_total, dbcsr_release, &
      dbcsr_get_data_size, dbcsr_get_num_blocks, dbcsr_get_nze
   USE dbcsr_operations, ONLY: &
      dbcsr_get_info, dbcsr_set, dbcsr_filter, dbcsr_clear
   USE dbcsr_tas_types, ONLY: &
      dbcsr_tas_distribution_type, dbcsr_tas_iterator, dbcsr_tas_split_info, dbcsr_tas_type
   USE dbcsr_tas_global, ONLY: &
      dbcsr_tas_blk_size_arb, dbcsr_tas_dist_arb, dbcsr_tas_distribution, dbcsr_tas_rowcol_data
   USE dbcsr_tas_split, ONLY: &
      block_index_global_to_local, block_index_local_to_global, colsplit, &
      dbcsr_tas_info_hold, dbcsr_tas_release_info, dbcsr_tas_create_split, &
      group_to_mrowcol, rowsplit, dbcsr_tas_get_split_info
   USE dbcsr_tas_util, ONLY: &
      dbcsr_mp_environ, index_unique
   USE dbcsr_types, ONLY: &
      dbcsr_distribution_obj, dbcsr_iterator, dbcsr_mp_obj, dbcsr_type, dbcsr_type_no_symmetry
   USE dbcsr_work_operations, ONLY: &
      dbcsr_create, dbcsr_finalize
   USE dbcsr_kinds, ONLY: &
      default_string_length, int_8, real_8, real_4
   USE dbcsr_mpiwrap, ONLY: &
      mp_cart_rank, mp_environ, mp_sum, mp_max
#include "../base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tas_base'

   PUBLIC :: &
      ! DBCSR wrappers / interface routines
      dbcsr_tas_blk_sizes, &
      dbcsr_tas_clear, &
      dbcsr_tas_copy, &
      dbcsr_tas_create, &
      dbcsr_tas_destroy, &
      dbcsr_tas_distribution_destroy, &
      dbcsr_tas_distribution_new, &
      dbcsr_tas_filter, &
      dbcsr_tas_finalize, &
      dbcsr_tas_get_block_p, &
      dbcsr_tas_get_data_size, &
      dbcsr_tas_get_data_type, &
      dbcsr_tas_get_info, &
      dbcsr_tas_get_num_blocks, &
      dbcsr_tas_get_nze, &
      dbcsr_tas_get_nze_total, &
      dbcsr_tas_get_num_blocks_total, &
      dbcsr_tas_get_stored_coordinates, &
      dbcsr_tas_info, &
      dbcsr_tas_iterator_blocks_left, &
      dbcsr_tas_iterator_next_block, &
      dbcsr_tas_iterator_start, &
      dbcsr_tas_iterator_stop, &
      dbcsr_tas_nblkcols_local, &
      dbcsr_tas_nblkcols_total, &
      dbcsr_tas_nblkrows_local, &
      dbcsr_tas_nblkrows_total, &
      dbcsr_tas_put_block, &
      dbcsr_tas_reserve_blocks, &
      dbcsr_tas_set, &
      dbcsr_repl_get_stored_coordinates
   PUBLIC :: &
      ! conversion routines
      dbcsr_tas_convert_to_dbcsr, &
      dbcsr_tas_convert_to_tas

   INTERFACE dbcsr_tas_create
      MODULE PROCEDURE dbcsr_tas_create_new
      MODULE PROCEDURE dbcsr_tas_create_template
   END INTERFACE

   INTERFACE dbcsr_tas_get_block_p
      MODULE PROCEDURE dbcsr_tas_get_block_p_area
   END INTERFACE

   INTERFACE dbcsr_tas_put_block
      MODULE PROCEDURE dbcsr_tas_put_block_area
   END INTERFACE

   INTERFACE dbcsr_tas_reserve_blocks
      MODULE PROCEDURE dbcsr_tas_reserve_blocks_template
      MODULE PROCEDURE dbcsr_tas_reserve_blocks_index
   END INTERFACE

   INTERFACE dbcsr_tas_iterator_next_block
      MODULE PROCEDURE dbcsr_tas_iterator_next_area_block
      MODULE PROCEDURE dbcsr_tas_iterator_next_block_index
   END INTERFACE

   INTERFACE dbcsr_tas_iterator_next_block
#:for dparam, dtype, dsuffix in dtype_float_list
      MODULE PROCEDURE dbcsr_tas_iterator_next_block_${dsuffix}$
#:endfor
   END INTERFACE

   INTERFACE dbcsr_tas_put_block
#:for dparam, dtype, dsuffix in dtype_float_list
      MODULE PROCEDURE dbcsr_tas_put_block_${dsuffix}$
#:endfor
   END INTERFACE

   INTERFACE dbcsr_tas_get_block_p
#:for dparam, dtype, dsuffix in dtype_float_list
      MODULE PROCEDURE dbcsr_tas_get_block_p_${dsuffix}$
#:endfor
   END INTERFACE

   INTERFACE dbcsr_tas_set
#:for dparam, dtype, dsuffix in dtype_float_list
      MODULE PROCEDURE dbcsr_tas_set_${dsuffix}$
#:endfor
   END INTERFACE

   INTERFACE dbcsr_tas_filter
#:for dparam, dtype, dsuffix in dtype_float_list
      MODULE PROCEDURE dbcsr_tas_filter_${dsuffix}$
#:endfor
   END INTERFACE

CONTAINS

   SUBROUTINE dbcsr_tas_create_new(matrix, name, dist, data_type, &
                                   row_blk_size, col_blk_size, own_dist)
      !! Create new tall-and-skinny matrix.
      !! Exactly like dbcsr_create_new but with custom types for row_blk_size and col_blk_size instead of
      !! arrays.

      TYPE(dbcsr_tas_type), INTENT(OUT)              :: matrix
      CHARACTER(len=*), INTENT(IN)                   :: name
      TYPE(dbcsr_tas_distribution_type), INTENT(INOUT) :: dist
      INTEGER, INTENT(IN)                            :: data_type
      CLASS(dbcsr_tas_rowcol_data), INTENT(IN)       :: row_blk_size, col_blk_size
      LOGICAL, INTENT(IN), OPTIONAL                  :: own_dist
         !! whether matrix should own distribution

      TYPE(dbcsr_tas_split_info)                     :: info

      INTEGER, DIMENSION(:), POINTER                 :: row_blk_size_vec => NULL(), col_blk_size_vec => NULL()
      INTEGER                                        :: nrows, ncols, irow, col, icol, row
      CHARACTER(LEN=*), PARAMETER                    :: routineN = 'dbcsr_tas_create_new', &
                                                        routineP = moduleN//':'//routineN
      INTEGER                                        :: handle

      CALL timeset(routineN, handle)

      CALL dbcsr_tas_copy_distribution(dist, matrix%dist, own_dist)
      matrix%nblkrows = row_blk_size%nmrowcol
      matrix%nblkcols = col_blk_size%nmrowcol

      DBCSR_ASSERT(matrix%nblkrows == dist%row_dist%nmrowcol)
      DBCSR_ASSERT(matrix%nblkcols == dist%col_dist%nmrowcol)

      matrix%nfullrows = row_blk_size%nfullrowcol
      matrix%nfullcols = col_blk_size%nfullrowcol

      ALLOCATE (matrix%row_blk_size, source=row_blk_size)
      ALLOCATE (matrix%col_blk_size, source=col_blk_size)

      info = dbcsr_tas_info(matrix)

      SELECT CASE (info%split_rowcol)
      CASE (rowsplit)
         matrix%nblkrowscols_split = matrix%nblkrows

         ASSOCIATE (rows=>dist%local_rowcols)
            nrows = SIZE(rows)
            ncols = INT(dist%col_dist%nmrowcol)
            ALLOCATE (row_blk_size_vec(nrows))
            ALLOCATE (col_blk_size_vec(ncols))
            DO irow = 1, nrows
               row_blk_size_vec(irow) = row_blk_size%data(rows(irow))
            ENDDO
            DO col = 1, ncols
               col_blk_size_vec(col) = col_blk_size%data(INT(col, KIND=int_8))
            ENDDO
         END ASSOCIATE
      CASE (colsplit)
         matrix%nblkrowscols_split = matrix%nblkcols

         ASSOCIATE (cols=>dist%local_rowcols)
            ncols = SIZE(cols)
            nrows = INT(dist%row_dist%nmrowcol)
            ALLOCATE (row_blk_size_vec(nrows))
            ALLOCATE (col_blk_size_vec(ncols))
            DO icol = 1, ncols
               col_blk_size_vec(icol) = col_blk_size%data(cols(icol))
            ENDDO
            DO row = 1, nrows
               row_blk_size_vec(row) = row_blk_size%data(INT(row, KIND=int_8))
            ENDDO
         END ASSOCIATE
      END SELECT

      CALL dbcsr_create(matrix=matrix%matrix, &
                        name=name, &
                        dist=dist%dbcsr_dist, &
                        matrix_type=dbcsr_type_no_symmetry, &
                        row_blk_size=row_blk_size_vec, &
                        col_blk_size=col_blk_size_vec, &
                        data_type=data_type, &
                        reuse_arrays=.TRUE.)

      matrix%valid = .TRUE.
      CALL timestop(handle)

   END SUBROUTINE

   SUBROUTINE dbcsr_tas_create_template(matrix_in, matrix, name)
      !! Create matrix from template
      TYPE(dbcsr_tas_type), INTENT(INOUT)                  :: matrix_in
      TYPE(dbcsr_tas_type), INTENT(OUT)                    :: matrix
      CHARACTER(len=*), INTENT(IN), OPTIONAL             :: name

      CALL dbcsr_create(matrix%matrix, template=matrix_in%matrix, matrix_type=dbcsr_type_no_symmetry)
      CALL dbcsr_finalize(matrix%matrix)

      CALL dbcsr_tas_copy_distribution(matrix_in%dist, matrix%dist)
      ALLOCATE (matrix%row_blk_size, source=matrix_in%row_blk_size)
      ALLOCATE (matrix%col_blk_size, source=matrix_in%col_blk_size)
      matrix%nblkrows = matrix_in%nblkrows
      matrix%nblkcols = matrix_in%nblkcols
      matrix%nblkrowscols_split = matrix_in%nblkrowscols_split
      matrix%nfullrows = matrix_in%nfullrows
      matrix%nfullcols = matrix_in%nfullcols
      matrix%valid = .TRUE.

      IF (PRESENT(name)) THEN
         matrix%matrix%name = name
      ELSE
         matrix%matrix%name = matrix_in%matrix%name
      ENDIF

   END SUBROUTINE

   SUBROUTINE dbcsr_tas_destroy(matrix)
      TYPE(dbcsr_tas_type), INTENT(INOUT)                  :: matrix

      CALL dbcsr_release(matrix%matrix)
      CALL dbcsr_tas_distribution_destroy(matrix%dist)
      DEALLOCATE (matrix%row_blk_size)
      DEALLOCATE (matrix%col_blk_size)
      matrix%valid = .FALSE.
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_copy(matrix_b, matrix_a, summation)
      !! Copy matrix_a to matrix_b

      TYPE(dbcsr_tas_type), INTENT(INOUT)                  :: matrix_b
      TYPE(dbcsr_tas_type), INTENT(IN)                     :: matrix_a
      LOGICAL, INTENT(IN), OPTIONAL                      :: summation
         !! Whether to sum matrices b = a + b

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_copy', routineP = moduleN//':'//routineN

      INTEGER                                            :: handle
      TYPE(dbcsr_tas_iterator)                             :: iter
      TYPE(dbcsr_data_obj)                               :: block
      LOGICAL :: transposed
      INTEGER   :: data_type
      INTEGER(KIND=int_8)                                :: row, column

      CALL timeset(routineN, handle)
      DBCSR_ASSERT(matrix_b%valid)

      CALL dbcsr_tas_get_info(matrix_a, data_type=data_type)

      IF (PRESENT(summation)) THEN
         IF (.NOT. summation) CALL dbcsr_tas_clear(matrix_b)
      ELSE
         CALL dbcsr_tas_clear(matrix_b)
      ENDIF

      CALL dbcsr_tas_reserve_blocks(matrix_a, matrix_b)

      CALL dbcsr_data_init(block)
      CALL dbcsr_data_new(block, dbcsr_type_1d_to_2d(data_type))
      CALL dbcsr_tas_iterator_start(iter, matrix_a)
      DO WHILE (dbcsr_tas_iterator_blocks_left(iter))
         CALL dbcsr_tas_iterator_next_block(iter, row, column, block, transposed)
         CALL dbcsr_tas_put_block(matrix_b, row, column, block, transposed, summation=summation)
      ENDDO
      CALL dbcsr_tas_iterator_stop(iter)
      CALL dbcsr_data_clear_pointer(block)
      CALL dbcsr_data_release(block)
      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_reserve_blocks_template(matrix_in, matrix_out)
      !! Make sure that matrix_out has same blocks reserved as matrix_in. This assumes that both
      !! matrices have same number of block rows and block columns.

      TYPE(dbcsr_tas_type), INTENT(IN) :: matrix_in
      TYPE(dbcsr_tas_type), INTENT(INOUT) :: matrix_out
      TYPE(dbcsr_tas_iterator)                        :: iter
      INTEGER :: iblk, nblk
      INTEGER(KIND=int_8)                  :: row, column
      INTEGER :: blk
      INTEGER(KIND=int_8), DIMENSION(:), ALLOCATABLE :: row_res, col_res

      nblk = dbcsr_tas_get_num_blocks(matrix_in)
      ALLOCATE (row_res(nblk), col_res(nblk))
      CALL dbcsr_tas_iterator_start(iter, matrix_in)
      DO iblk = 1, nblk
         CALL dbcsr_tas_iterator_next_block(iter, row, column, blk)
         row_res(iblk) = row
         col_res(iblk) = column
      ENDDO
      DBCSR_ASSERT(.NOT. dbcsr_tas_iterator_blocks_left(iter))
      CALL dbcsr_tas_iterator_stop(iter)

      CALL dbcsr_tas_reserve_blocks(matrix_out, row_res, col_res)
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_finalize(matrix)
      TYPE(dbcsr_tas_type), INTENT(INOUT)                  :: matrix

      CALL dbcsr_finalize(matrix%matrix)
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_distribution_new(dist, mp_comm, &
                                         row_dist, col_dist, split_info, nosplit)
      !! create new distribution.
      !! Exactly like dbcsr_distribution_new but with custom types for row_dist and col_dist instead of
      !! arrays.

      TYPE(dbcsr_tas_distribution_type), INTENT(OUT)   :: dist
      INTEGER, INTENT(IN)                              :: mp_comm
      CLASS(dbcsr_tas_distribution), INTENT(IN)        :: row_dist, col_dist
      TYPE(dbcsr_tas_split_info), INTENT(IN), OPTIONAL :: split_info
         !! Strategy of how to split process grid (optional). If not present a default split heuristic is applied.
      LOGICAL, INTENT(IN), OPTIONAL                    :: nosplit
         !! if .TRUE. don't split process grid (optional)
      !LOGICAL, INTENT(IN), OPTIONAL                    :: strict_split

      TYPE(dbcsr_tas_split_info)                       :: split_info_prv

      INTEGER, DIMENSION(:), POINTER                   :: row_dist_vec => NULL()
      INTEGER, DIMENSION(:), POINTER                   :: col_dist_vec => NULL()
      TYPE(dbcsr_mp_obj)                               :: mp_environ_tmp
      INTEGER                                          :: nrows, ncols, irow, col, icol, row, &
                                                          split_rowcol, nsplit, handle
      LOGICAL                                          :: opt_nsplit
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_distribution_new', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)
      IF (PRESENT(split_info)) THEN
         CALL dbcsr_tas_info_hold(split_info)
         split_info_prv = split_info
      ELSE
         ! default split heuristic: split into submatrices that have roughly same block dimensions
         IF (row_dist%nmrowcol >= col_dist%nmrowcol) THEN
            split_rowcol = rowsplit
            nsplit = INT((row_dist%nmrowcol - 1)/col_dist%nmrowcol + 1)
         ELSE
            split_rowcol = colsplit
            nsplit = INT((col_dist%nmrowcol - 1)/row_dist%nmrowcol + 1)
         ENDIF
         opt_nsplit = .TRUE.
         IF (PRESENT(nosplit)) THEN
            IF (nosplit) THEN
               nsplit = 1
               opt_nsplit = .FALSE.
            ENDIF
         ENDIF
         CALL dbcsr_tas_create_split(split_info_prv, mp_comm, split_rowcol, nsplit=nsplit, opt_nsplit=opt_nsplit)
      ENDIF

      SELECT CASE (split_info_prv%split_rowcol)
      CASE (rowsplit)
         CALL group_to_mrowcol(split_info_prv, row_dist, split_info_prv%igroup, dist%local_rowcols)
         nrows = SIZE(dist%local_rowcols)
         ncols = INT(col_dist%nmrowcol)
         ALLOCATE (row_dist_vec(nrows))
         ALLOCATE (col_dist_vec(ncols))
         DO irow = 1, nrows
            row_dist_vec(irow) = row_dist%dist(dist%local_rowcols(irow)) - split_info_prv%pgrid_split_size*split_info_prv%igroup
         ENDDO
         DO col = 1, ncols
            col_dist_vec(col) = col_dist%dist(INT(col, KIND=int_8))
         ENDDO
      CASE (colsplit)
         CALL group_to_mrowcol(split_info_prv, col_dist, split_info_prv%igroup, dist%local_rowcols)
         ncols = SIZE(dist%local_rowcols)
         nrows = INT(row_dist%nmrowcol)
         ALLOCATE (col_dist_vec(ncols))
         ALLOCATE (row_dist_vec(nrows))
         DO icol = 1, ncols
            col_dist_vec(icol) = col_dist%dist(dist%local_rowcols(icol)) - split_info_prv%pgrid_split_size*split_info_prv%igroup
         ENDDO
         DO row = 1, nrows
            row_dist_vec(row) = row_dist%dist(INT(row, KIND=int_8))
         ENDDO
      END SELECT

      mp_environ_tmp = dbcsr_mp_environ(split_info_prv%mp_comm_group)

      dist%info = split_info_prv

      CALL dbcsr_distribution_new(dist%dbcsr_dist, mp_environ_tmp, row_dist_vec, col_dist_vec, reuse_arrays=.TRUE.)
      ALLOCATE (dist%row_dist, source=row_dist)
      ALLOCATE (dist%col_dist, source=col_dist)
      CALL dbcsr_mp_release(mp_environ_tmp)

      !IF(PRESENT(strict_split)) dist%strict_split = strict_split

      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_distribution_destroy(dist)
      TYPE(dbcsr_tas_distribution_type), INTENT(INOUT)     :: dist

      IF (ALLOCATED(dist%row_dist)) THEN
         DEALLOCATE (dist%row_dist)
      ENDIF
      IF (ALLOCATED(dist%col_dist)) THEN
         DEALLOCATE (dist%col_dist)
      ENDIF
      IF (ALLOCATED(dist%local_rowcols)) THEN
         DEALLOCATE (dist%local_rowcols)
      ENDIF
      CALL dbcsr_tas_release_info(dist%info)
      CALL dbcsr_distribution_release(dist%dbcsr_dist)
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_get_stored_coordinates(matrix, row, column, processor)
      !! As dbcsr_get_stored_coordinates

      TYPE(dbcsr_tas_type), INTENT(IN)                   :: matrix
      INTEGER(KIND=int_8), INTENT(IN)                    :: row, column
         !! global matrix blocked row
         !! global matrix blocked column
      INTEGER, INTENT(OUT)                               :: processor
         !! process ID

      INTEGER, DIMENSION(2)                              :: pcoord
      TYPE(dbcsr_tas_split_info)                         :: info

      pcoord(1) = matrix%dist%row_dist%dist(row)
      pcoord(2) = matrix%dist%col_dist%dist(column)
      info = dbcsr_tas_info(matrix)
      CALL mp_cart_rank(info%mp_comm, pcoord, processor)

   END SUBROUTINE

   SUBROUTINE dbcsr_repl_get_stored_coordinates(matrix, row, column, processors)
      !! Get all processors for a given row/col combination if matrix is replicated on each process
      !! subgroup.

      TYPE(dbcsr_tas_type), INTENT(IN)                     :: matrix
         !! tall-and-skinny matrix whose DBCSR submatrices are replicated matrices
      INTEGER, INTENT(IN)                                :: row, column
         !! row of a submatrix
         !! column of a submatrix
      INTEGER, DIMENSION(:), INTENT(OUT)                 :: processors

      INTEGER                                            :: igroup
      INTEGER(KIND=int_8)                                :: col_s, row_s
      INTEGER, DIMENSION(2)                              :: pcoord
      TYPE(dbcsr_tas_split_info)                         :: info

      row_s = INT(row, KIND=int_8); col_s = INT(column, KIND=int_8)

      info = dbcsr_tas_info(matrix)
      pcoord(1) = matrix%dist%row_dist%dist(row_s)
      pcoord(2) = matrix%dist%col_dist%dist(col_s)

      DO igroup = 0, info%ngroup - 1
         CALL mp_cart_rank(info%mp_comm, pcoord, processors(igroup + 1))
         SELECT CASE (info%split_rowcol)
         CASE (rowsplit)
            row_s = row_s + dbcsr_tas_nblkrows_local(matrix)
            pcoord(1) = matrix%dist%row_dist%dist(row_s)
         CASE (colsplit)
            col_s = col_s + dbcsr_tas_nblkcols_local(matrix)
            pcoord(2) = matrix%dist%col_dist%dist(col_s)
         END SELECT
      ENDDO
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_convert_to_dbcsr(matrix_rect, matrix_dbcsr)
      !! Convert a tall-and-skinny matrix into a normal DBCSR matrix.
      !! This is not recommended for matrices with a very large dimension.

      TYPE(dbcsr_tas_type), INTENT(IN)                     :: matrix_rect
      TYPE(dbcsr_type), INTENT(OUT)                      :: matrix_dbcsr

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_convert_to_dbcsr', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: handle
      INTEGER(KIND=int_8)                                :: col, row
      INTEGER, DIMENSION(:), POINTER                     :: col_dist_vec => NULL(), col_size_vec => NULL(), &
                                                            row_dist_vec => NULL(), row_size_vec => NULL()
      LOGICAL                                            :: tr
      TYPE(dbcsr_data_obj)                               :: block
      TYPE(dbcsr_distribution_obj)                       :: dist
      TYPE(dbcsr_mp_obj)                                 :: mp_environ_tmp
      TYPE(dbcsr_tas_iterator)                           :: iter
      TYPE(dbcsr_tas_split_info)                         :: info
      INTEGER                                            :: block_number, rb_count, nblks_local
      INTEGER, DIMENSION(:), ALLOCATABLE                 :: nz_rows, nz_cols

      CALL timeset(routineN, handle)

      info = dbcsr_tas_info(matrix_rect)

      ALLOCATE (row_dist_vec(matrix_rect%nblkrows))
      ALLOCATE (row_size_vec(matrix_rect%nblkrows))
      ALLOCATE (col_dist_vec(matrix_rect%nblkcols))
      ALLOCATE (col_size_vec(matrix_rect%nblkcols))

      DO row = 1, matrix_rect%nblkrows
         row_dist_vec(row) = matrix_rect%dist%row_dist%dist(row)
         row_size_vec(row) = matrix_rect%row_blk_size%data(row)
      ENDDO

      DO col = 1, matrix_rect%nblkcols
         col_dist_vec(col) = matrix_rect%dist%col_dist%dist(col)
         col_size_vec(col) = matrix_rect%col_blk_size%data(col)
      ENDDO

      mp_environ_tmp = dbcsr_mp_environ(info%mp_comm)
      CALL dbcsr_distribution_new(dist, mp_environ_tmp, row_dist_vec, col_dist_vec, reuse_arrays=.TRUE.)
      CALL dbcsr_mp_release(mp_environ_tmp)

      CALL dbcsr_create(matrix=matrix_dbcsr, &
                        name=TRIM(matrix_rect%matrix%name), &
                        dist=dist, &
                        matrix_type=dbcsr_type_no_symmetry, &
                        row_blk_size=row_size_vec, &
                        col_blk_size=col_size_vec, &
                        data_type=dbcsr_get_data_type(matrix_rect%matrix), &
                        reuse_arrays=.TRUE.)

      CALL dbcsr_distribution_release(dist)

      CALL dbcsr_data_init(block)
      CALL dbcsr_data_new(block, dbcsr_type_1d_to_2d(dbcsr_tas_get_data_type(matrix_rect)))

      nblks_local = dbcsr_tas_get_num_blocks(matrix_rect)
      CALL dbcsr_tas_iterator_start(iter, matrix_rect)
      ALLOCATE (nz_rows(nblks_local), nz_cols(nblks_local))
      rb_count = 0
      DO WHILE (dbcsr_tas_iterator_blocks_left(iter))
         CALL dbcsr_tas_iterator_next_block(iter, row, col, block_number)
         rb_count = rb_count + 1
         nz_rows(rb_count) = INT(row)
         nz_cols(rb_count) = INT(col)
      ENDDO

      CALL dbcsr_reserve_blocks(matrix_dbcsr, nz_rows, nz_cols)

      CALL dbcsr_tas_iterator_stop(iter)
      CALL dbcsr_tas_iterator_start(iter, matrix_rect)
      DO WHILE (dbcsr_tas_iterator_blocks_left(iter))
         CALL dbcsr_tas_iterator_next_block(iter, row, col, block, tr)
         CALL dbcsr_put_block(matrix_dbcsr, INT(row), INT(col), block)
      ENDDO

      CALL dbcsr_tas_iterator_stop(iter)
      CALL dbcsr_data_clear_pointer(block)
      CALL dbcsr_data_release(block)

      CALL dbcsr_finalize(matrix_dbcsr)

      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_convert_to_tas(info, matrix_rect, matrix_dbcsr)
      !! Converts a DBCSR matrix into the tall-and-skinny matrix type.

      TYPE(dbcsr_tas_split_info), INTENT(IN)               :: info
         !! Strategy of how to split process grid
      TYPE(dbcsr_tas_type), INTENT(OUT)                    :: matrix_rect
      TYPE(dbcsr_type), INTENT(IN)                         :: matrix_dbcsr

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_convert_to_tas', &
                                     routineP = moduleN//':'//routineN

      CHARACTER(len=default_string_length)               :: name
      INTEGER                                            :: col, data_type, handle, numnodes, row
      INTEGER(KIND=int_8)                                :: nbcols, nbrows
      INTEGER, DIMENSION(2)                              :: pcoord, pdims
      INTEGER, DIMENSION(:), POINTER                     :: col_blk_size => NULL(), row_blk_size => NULL()
      LOGICAL                                            :: tr
      TYPE(dbcsr_data_obj)                               :: block
      TYPE(dbcsr_distribution_obj)                       :: dbcsr_dist
      TYPE(dbcsr_iterator)                               :: iter
      TYPE(dbcsr_tas_blk_size_arb)                         :: col_blk_size_obj, row_blk_size_obj
      TYPE(dbcsr_tas_dist_arb)                             :: col_dist_obj, row_dist_obj
      TYPE(dbcsr_tas_distribution_type)                    :: dist

      CALL timeset(routineN, handle)
      CALL mp_environ(numnodes, pdims, pcoord, info%mp_comm)
      CALL dbcsr_get_info(matrix_dbcsr, distribution=dbcsr_dist, name=name, data_type=data_type, &
                          row_blk_size=row_blk_size, col_blk_size=col_blk_size)

      nbrows = dbcsr_nblkrows_total(matrix_dbcsr)
      nbcols = dbcsr_nblkcols_total(matrix_dbcsr)

      row_dist_obj = dbcsr_tas_dist_arb(dbcsr_distribution_row_dist(dbcsr_dist), pdims(1), nbrows)
      col_dist_obj = dbcsr_tas_dist_arb(dbcsr_distribution_col_dist(dbcsr_dist), pdims(2), nbcols)

      row_blk_size_obj = dbcsr_tas_blk_size_arb(row_blk_size)
      col_blk_size_obj = dbcsr_tas_blk_size_arb(col_blk_size)

      CALL dbcsr_tas_distribution_new(dist, info%mp_comm, row_dist_obj, col_dist_obj)

      CALL dbcsr_tas_create(matrix_rect, TRIM(name)//"_compressed", &
                            dist, data_type, row_blk_size_obj, col_blk_size_obj)

      CALL dbcsr_data_init(block)

      CALL dbcsr_data_new(block, dbcsr_type_1d_to_2d(data_type))
      CALL dbcsr_iterator_start(iter, matrix_dbcsr)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, row, col, block, tr)
         CALL dbcsr_tas_put_block(matrix_rect, INT(row, KIND=int_8), INT(col, KIND=int_8), block)
      ENDDO
      CALL dbcsr_iterator_stop(iter)
      CALL dbcsr_data_clear_pointer(block)
      CALL dbcsr_data_release(block)

      CALL dbcsr_tas_finalize(matrix_rect)

      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_iterator_start(iter, matrix_in)
      !! As dbcsr_iterator_start
      TYPE(dbcsr_tas_iterator), INTENT(INOUT)              :: iter
      TYPE(dbcsr_tas_type), INTENT(IN)                     :: matrix_in

      CALL dbcsr_iterator_start(iter%iter, matrix_in%matrix)

      iter%dist = matrix_in%dist
   END SUBROUTINE

   FUNCTION dbcsr_tas_iterator_blocks_left(iter)
      !! As dbcsr_iterator_blocks_left
      TYPE(dbcsr_tas_iterator), INTENT(IN)                 :: iter
      LOGICAL                                            :: dbcsr_tas_iterator_blocks_left

      dbcsr_tas_iterator_blocks_left = dbcsr_iterator_blocks_left(iter%iter)
   END FUNCTION

   SUBROUTINE dbcsr_tas_iterator_stop(iter)
      !! As dbcsr_iterator_stop
      TYPE(dbcsr_tas_iterator), INTENT(INOUT)              :: iter

      CALL dbcsr_iterator_stop(iter%iter)
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_iterator_next_area_block(iterator, row, column, block, transposed, block_number, &
      !! As dbcsr_iterator_next_block
                                                 row_size, col_size)
      TYPE(dbcsr_tas_iterator), INTENT(INOUT)              :: iterator
      INTEGER(KIND=int_8), INTENT(OUT)                   :: row, column
      TYPE(dbcsr_data_obj), INTENT(INOUT)                :: block
      LOGICAL, INTENT(OUT)                               :: transposed
      INTEGER, INTENT(OUT), OPTIONAL                     :: block_number, row_size, col_size

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_iterator_next_area_block', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: column_group, handle, row_group

      CALL timeset(routineN, handle)

      CALL dbcsr_iterator_next_block(iterator%iter, row_group, column_group, block, transposed, block_number, &
                                     row_size, col_size)

      CALL block_index_local_to_global(iterator%dist%info, iterator%dist, row_group=row_group, column_group=column_group, &
                                       row=row, column=column)

      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_iterator_next_block_index(iterator, row, column, block_number, &
                                                  transposed, blk_p, row_size, col_size)
      !! As dbcsr_iterator_next_block

      TYPE(dbcsr_tas_iterator), INTENT(INOUT)              :: iterator
      INTEGER(KIND=int_8), INTENT(OUT)                   :: row, column
         !! global block row
         !! global block column
      INTEGER, INTENT(OUT)                               :: block_number
      LOGICAL, INTENT(OUT), OPTIONAL                     :: transposed
      INTEGER, INTENT(OUT), OPTIONAL                     :: blk_p, row_size, col_size

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_iterator_next_block_index', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: column_group, handle, row_group

      CALL timeset(routineN, handle)
      CALL dbcsr_iterator_next_block(iterator%iter, row_group, column_group, block_number, transposed, blk_p, &
                                     row_size, col_size)

      CALL block_index_local_to_global(iterator%dist%info, iterator%dist, row_group=row_group, column_group=column_group, &
                                       row=row, column=column)

      CALL timestop(handle)

   END SUBROUTINE

   SUBROUTINE dbcsr_tas_reserve_blocks_index(matrix, rows, columns)
      !! As dbcsr_reserve_blocks
      TYPE(dbcsr_tas_type), INTENT(INOUT)                  :: matrix
      INTEGER(KIND=int_8), DIMENSION(:), INTENT(IN)      :: rows, columns

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_reserve_blocks_index', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: handle, icol, irow
      INTEGER, DIMENSION(SIZE(rows), 2)                :: rowcols_group
      INTEGER, DIMENSION(:, :), ALLOCATABLE             :: rowcols_group_u

      CALL timeset(routineN, handle)

      DO irow = 1, SIZE(rows)
         CALL block_index_global_to_local(dbcsr_tas_info(matrix), matrix%dist, row=rows(irow), &
                                          row_group=rowcols_group(irow, 1))
      ENDDO

      DO icol = 1, SIZE(columns)
         CALL block_index_global_to_local(dbcsr_tas_info(matrix), matrix%dist, column=columns(icol), &
                                          column_group=rowcols_group(icol, 2))
      ENDDO

      CALL index_unique(rowcols_group, rowcols_group_u) ! make sure that index is unique, not sure
      ! if this is really needed or whether DBCSR
      ! takes care of duplicate indices

      CALL dbcsr_reserve_blocks(matrix%matrix, rowcols_group_u(:, 1), rowcols_group_u(:, 2))

      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_put_block_area(matrix, row, col, block, transposed, summation, &
      !! As dbcsr_put_block
                                       scale)
      TYPE(dbcsr_tas_type), INTENT(INOUT)                  :: matrix
      INTEGER(KIND=int_8), INTENT(IN)                    :: row, col
      TYPE(dbcsr_data_obj)                               :: block
      LOGICAL, INTENT(IN), OPTIONAL                      :: transposed, summation
      TYPE(dbcsr_scalar_type), INTENT(IN), OPTIONAL      :: scale

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_put_block_area', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: col_group, handle, row_group

      CALL timeset(routineN, handle)

      CALL block_index_global_to_local(dbcsr_tas_info(matrix), matrix%dist, row=row, column=col, &
                                       row_group=row_group, column_group=col_group)
      CALL dbcsr_put_block(matrix%matrix, row_group, col_group, block, transposed=transposed, summation=summation, scale=scale)

      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_get_block_p_area(matrix, row, col, block, transposed, found, row_size, col_size)
      !! As dbcsr_get_block_p
      TYPE(dbcsr_tas_type), INTENT(INOUT)                  :: matrix
      INTEGER(KIND=int_8), INTENT(IN)                    :: row, col
      TYPE(dbcsr_data_obj), INTENT(INOUT)                :: block
      LOGICAL, INTENT(OUT)                               :: transposed, found
      INTEGER, INTENT(OUT), OPTIONAL                     :: row_size, col_size

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_get_block_p_area', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: col_group, handle, row_group

      CALL timeset(routineN, handle)

      CALL block_index_global_to_local(dbcsr_tas_info(matrix), matrix%dist, row=row, column=col, &
                                       row_group=row_group, column_group=col_group)

      CALL dbcsr_get_block_p(matrix%matrix, row_group, col_group, block, transposed, found, row_size=row_size, col_size=col_size)

      CALL timestop(handle)

   END SUBROUTINE

   SUBROUTINE dbcsr_tas_copy_distribution(dist_in, dist_out, own_dist)
      !! Copy a distribution

      TYPE(dbcsr_tas_distribution_type), INTENT(INOUT)     :: dist_in
      TYPE(dbcsr_tas_distribution_type), INTENT(OUT)       :: dist_out
      LOGICAL, INTENT(IN), OPTIONAL                      :: own_dist
         !! Whether distribution should be owned by dist_out

      LOGICAL                                            :: own_dist_prv

      IF (PRESENT(own_dist)) THEN
         own_dist_prv = own_dist
      ELSE
         own_dist_prv = .FALSE.
      ENDIF

      IF (.NOT. own_dist_prv) THEN
         CALL dbcsr_distribution_hold(dist_in%dbcsr_dist)
         CALL dbcsr_tas_info_hold(dist_in%info)
      ENDIF

      dist_out = dist_in
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_blk_sizes(matrix, row, col, row_size, col_size)
      !! Get block size for a given row & column
      TYPE(dbcsr_tas_type), INTENT(IN)                     :: matrix
      INTEGER(KIND=int_8), INTENT(IN)                    :: row, col
      INTEGER, INTENT(OUT)                               :: row_size, col_size
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_blk_sizes', &
                                     routineP = moduleN//':'//routineN
      INTEGER :: handle

      CALL timeset(routineN, handle)

      row_size = matrix%row_blk_size%data(row)
      col_size = matrix%col_blk_size%data(col)

      CALL timestop(handle)
   END SUBROUTINE

   FUNCTION dbcsr_tas_info(matrix)
      !! get info on mpi grid splitting
      TYPE(dbcsr_tas_type), INTENT(IN)                     :: matrix
      TYPE(dbcsr_tas_split_info)                           :: dbcsr_tas_info

      dbcsr_tas_info = matrix%dist%info
   END FUNCTION

   FUNCTION dbcsr_tas_get_data_type(matrix) RESULT(data_type)
      TYPE(dbcsr_tas_type), INTENT(IN)                     :: matrix
      INTEGER                                            :: data_type

      CALL dbcsr_get_info(matrix%matrix, data_type=data_type)
   END FUNCTION

   FUNCTION dbcsr_tas_nblkrows_total(matrix) RESULT(nblkrows_total)
      TYPE(dbcsr_tas_type), INTENT(IN)                     :: matrix
      INTEGER(KIND=int_8)                                :: nblkrows_total

      nblkrows_total = matrix%nblkrows
   END FUNCTION

   FUNCTION dbcsr_tas_nfullrows_total(matrix) RESULT(nfullrows_total)
      TYPE(dbcsr_tas_type), INTENT(IN)                     :: matrix
      INTEGER(KIND=int_8)                                :: nfullrows_total

      nfullrows_total = matrix%nfullrows
   END FUNCTION

   FUNCTION dbcsr_tas_nblkcols_total(matrix) RESULT(nblkcols_total)
      TYPE(dbcsr_tas_type), INTENT(IN)                     :: matrix
      INTEGER(KIND=int_8)                                :: nblkcols_total

      nblkcols_total = matrix%nblkcols
   END FUNCTION

   FUNCTION dbcsr_tas_nfullcols_total(matrix) RESULT(nfullcols_total)
      TYPE(dbcsr_tas_type), INTENT(IN)                     :: matrix
      INTEGER(KIND=int_8)                                :: nfullcols_total

      nfullcols_total = matrix%nfullcols
   END FUNCTION
   FUNCTION dbcsr_tas_nblkcols_local(matrix) RESULT(nblkcols_local)
      TYPE(dbcsr_tas_type), INTENT(IN)                     :: matrix
      INTEGER                                            :: nblkcols_local

      nblkcols_local = dbcsr_nblkcols_total(matrix%matrix)
   END FUNCTION

   FUNCTION dbcsr_tas_nblkrows_local(matrix) RESULT(nblkrows_local)
      TYPE(dbcsr_tas_type), INTENT(IN)                     :: matrix
      INTEGER                                            :: nblkrows_local

      nblkrows_local = dbcsr_nblkrows_total(matrix%matrix)
   END FUNCTION

   FUNCTION dbcsr_tas_nfullrows_local(matrix) RESULT(nfullrows_local)
      TYPE(dbcsr_tas_type), INTENT(IN)                     :: matrix
      INTEGER                                            :: nfullrows_local

      nfullrows_local = dbcsr_nfullrows_total(matrix%matrix)
   END FUNCTION

   FUNCTION dbcsr_tas_nfullcols_local(matrix) RESULT(nfullcols_local)
      TYPE(dbcsr_tas_type), INTENT(IN)                     :: matrix
      INTEGER                                            :: nfullcols_local

      nfullcols_local = dbcsr_nfullcols_total(matrix%matrix)
   END FUNCTION

   PURE FUNCTION dbcsr_tas_get_num_blocks(matrix) RESULT(num_blocks)
      !! As dbcsr_get_num_blocks: get number of local blocks
      TYPE(dbcsr_tas_type), INTENT(IN) :: matrix
      INTEGER                        :: num_blocks

      num_blocks = dbcsr_get_num_blocks(matrix%matrix)
   END FUNCTION

   FUNCTION dbcsr_tas_get_num_blocks_total(matrix) RESULT(num_blocks)
      !! get total number of blocks
      TYPE(dbcsr_tas_type), INTENT(IN) :: matrix
      INTEGER(KIND=int_8)              :: num_blocks
      TYPE(dbcsr_tas_split_info)       :: info

      info = dbcsr_tas_info(matrix)
      num_blocks = dbcsr_tas_get_num_blocks(matrix)
      CALL mp_sum(num_blocks, info%mp_comm)

   END FUNCTION

   PURE FUNCTION dbcsr_tas_get_nze(matrix)
      !! As dbcsr_get_nze: get number of local non-zero elements
      TYPE(dbcsr_tas_type), INTENT(IN) :: matrix
      INTEGER :: dbcsr_tas_get_nze

      dbcsr_tas_get_nze = dbcsr_get_nze(matrix%matrix)

   END FUNCTION

   FUNCTION dbcsr_tas_get_nze_total(matrix)
      !! Get total number of non-zero elements
      TYPE(dbcsr_tas_type), INTENT(IN) :: matrix
      INTEGER(KIND=int_8)              :: dbcsr_tas_get_nze_total
      TYPE(dbcsr_tas_split_info)       :: info

      dbcsr_tas_get_nze_total = dbcsr_tas_get_nze(matrix)
      info = dbcsr_tas_info(matrix)
      CALL mp_sum(dbcsr_tas_get_nze_total, info%mp_comm)
   END FUNCTION

   FUNCTION dbcsr_tas_get_data_size(matrix) RESULT(data_size)
      !! As dbcsr_get_data_size
      TYPE(dbcsr_tas_type), INTENT(IN) :: matrix
      INTEGER                          :: data_size

      data_size = dbcsr_get_data_size(matrix%matrix)
   END FUNCTION

   SUBROUTINE dbcsr_tas_clear(matrix)
      !! Clear matrix (erase all data)
      TYPE(dbcsr_tas_type), INTENT(INOUT) :: matrix

      CALL dbcsr_clear(matrix%matrix)
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_get_info(matrix, nblkrows_total, nblkcols_total, &
                                 nfullrows_total, nfullcols_total, &
                                 nblkrows_local, nblkcols_local, &
                                 nfullrows_local, nfullcols_local, &
                                 nprow, npcol, my_prow, my_pcol, &
                                 local_rows, local_cols, proc_row_dist, proc_col_dist, &
                                 row_blk_size, col_blk_size, distribution, name, data_area, &
                                 matrix_type, data_type)

      TYPE(dbcsr_tas_type), INTENT(IN)                                :: matrix
      INTEGER(KIND=int_8), INTENT(OUT), OPTIONAL                      :: nblkrows_total, nblkcols_total, nfullrows_total, &
                                                                         nfullcols_total
      INTEGER, INTENT(OUT), OPTIONAL                                  :: nblkrows_local, nblkcols_local, nfullrows_local, &
                                                                         nfullcols_local, nprow, npcol, my_prow, my_pcol
      INTEGER(KIND=int_8), DIMENSION(:), OPTIONAL, ALLOCATABLE        :: local_rows, local_cols
      CLASS(dbcsr_tas_distribution), ALLOCATABLE, OPTIONAL, &
         INTENT(OUT)                                                  :: proc_row_dist, proc_col_dist
      CLASS(dbcsr_tas_rowcol_data), ALLOCATABLE, OPTIONAL, &
         INTENT(OUT)                                                  :: row_blk_size, col_blk_size
      TYPE(dbcsr_tas_distribution_type), OPTIONAL                     :: distribution
      CHARACTER(len=*), INTENT(OUT), OPTIONAL                         :: name
      TYPE(dbcsr_data_obj), INTENT(OUT), OPTIONAL                     :: data_area
      CHARACTER, OPTIONAL                                             :: matrix_type
      INTEGER, OPTIONAL                                               :: data_type

      TYPE(dbcsr_tas_split_info)                                      :: info
      INTEGER                                                         :: numnodes, irow, icol
      INTEGER, DIMENSION(2)                                           :: pdims, pcoord
      INTEGER, DIMENSION(:), POINTER                                  :: local_rows_local => NULL(), local_cols_local => NULL()

      CALL dbcsr_get_info(matrix%matrix, nblkrows_local=nblkrows_local, nblkcols_local=nblkcols_local, &
                          nfullrows_local=nfullrows_local, nfullcols_local=nfullcols_local, &
                          local_rows=local_rows_local, local_cols=local_cols_local, &
                          name=name, data_area=data_area, matrix_type=matrix_type, data_type=data_type)

      IF (PRESENT(nblkrows_total)) nblkrows_total = dbcsr_tas_nblkrows_total(matrix)
      IF (PRESENT(nblkcols_total)) nblkcols_total = dbcsr_tas_nblkcols_total(matrix)

      IF (PRESENT(nfullrows_total)) nfullrows_total = dbcsr_tas_nfullrows_total(matrix)
      IF (PRESENT(nfullcols_total)) nfullcols_total = dbcsr_tas_nfullcols_total(matrix)

      info = dbcsr_tas_info(matrix)

      CALL mp_environ(numnodes, pdims, pcoord, info%mp_comm)
      IF (PRESENT(my_prow)) my_prow = pcoord(1)
      IF (PRESENT(my_pcol)) my_pcol = pcoord(2)
      IF (PRESENT(nprow)) nprow = pdims(1)
      IF (PRESENT(npcol)) npcol = pdims(2)

      IF (PRESENT(local_rows)) THEN
         ALLOCATE (local_rows(SIZE(local_rows_local)))
         DO irow = 1, SIZE(local_rows_local)
            CALL block_index_local_to_global(info, matrix%dist, row_group=local_rows_local(irow), row=local_rows(irow))
         ENDDO
      ENDIF

      IF (PRESENT(local_cols)) THEN
         ALLOCATE (local_cols(SIZE(local_cols_local)))
         DO icol = 1, SIZE(local_cols_local)
            CALL block_index_local_to_global(info, matrix%dist, column_group=local_cols_local(irow), column=local_cols(irow))
         ENDDO
      ENDIF

      IF (PRESENT(proc_row_dist)) ALLOCATE (proc_row_dist, SOURCE=matrix%dist%row_dist)
      IF (PRESENT(proc_col_dist)) ALLOCATE (proc_col_dist, SOURCE=matrix%dist%col_dist)

      IF (PRESENT(row_blk_size)) ALLOCATE (row_blk_size, SOURCE=matrix%row_blk_size)
      IF (PRESENT(col_blk_size)) ALLOCATE (col_blk_size, SOURCE=matrix%col_blk_size)
      IF (PRESENT(distribution)) distribution = matrix%dist

   END SUBROUTINE

#:for dparam, dtype, dsuffix in dtype_float_list
   SUBROUTINE dbcsr_tas_iterator_next_block_${dsuffix}$ (iterator, row, column, block, transposed, block_number, &
      !! As dbcsr_iterator_next_block
                                                         row_size, col_size)
      TYPE(dbcsr_tas_iterator), INTENT(INOUT) :: iterator
      INTEGER(KIND=int_8), INTENT(OUT)      :: row, column
      ${dtype}$, DIMENSION(:, :), POINTER   :: block
      LOGICAL, INTENT(OUT)                  :: transposed
      INTEGER, INTENT(OUT), OPTIONAL        :: block_number
      INTEGER, INTENT(OUT), OPTIONAL        :: row_size, col_size

      INTEGER                               :: row_group, column_group
      CHARACTER(LEN=*), PARAMETER           :: routineN = 'dbcsr_tas_iterator_next_block_${dsuffix}$', &
                                               routineP = moduleN//':'//routineN
      INTEGER                               :: handle

      CALL timeset(routineN, handle)
      CALL dbcsr_iterator_next_block(iterator%iter, row_group, column_group, block, transposed, block_number, &
                                     row_size, col_size)

      CALL block_index_local_to_global(iterator%dist%info, iterator%dist, row_group=row_group, column_group=column_group, &
                                       row=row, column=column)

      CALL timestop(handle)
   END SUBROUTINE
#:endfor

#:for dparam, dtype, dsuffix in dtype_float_list
   !! As dbcsr_put_block
   SUBROUTINE dbcsr_tas_put_block_${dsuffix}$ (matrix, row, col, block, transposed, summation, &
                                               scale)
      TYPE(dbcsr_tas_type), INTENT(INOUT)     :: matrix
      ${dtype}$, DIMENSION(:, :), INTENT(IN) :: block
      LOGICAL, INTENT(IN), OPTIONAL         :: transposed, summation
      INTEGER(KIND=int_8), INTENT(IN)       :: row, col
      ${dtype}$, INTENT(IN), OPTIONAL       :: scale

      INTEGER                     :: col_group, row_group
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_put_block_${dsuffix}$', &
                                     routineP = moduleN//':'//routineN
      INTEGER                     :: handle

      CALL timeset(routineN, handle)
      CALL block_index_global_to_local(matrix%dist%info, matrix%dist, row=row, column=col, &
                                       row_group=row_group, column_group=col_group)

      CALL dbcsr_put_block(matrix%matrix, row_group, col_group, block, transposed=transposed, summation=summation, scale=scale)

      CALL timestop(handle)
   END SUBROUTINE
#:endfor

#:for dparam, dtype, dsuffix in dtype_float_list
   SUBROUTINE dbcsr_tas_get_block_p_${dsuffix}$ (matrix, row, col, block, transposed, found, row_size, col_size)
      !! As dbcsr_get_block_p
      TYPE(dbcsr_tas_type), INTENT(INOUT)  :: matrix
      INTEGER(KIND=int_8), INTENT(IN)    :: row, col
      ${dtype}$, DIMENSION(:, :), POINTER :: block
      LOGICAL, INTENT(OUT)               :: transposed
      LOGICAL, INTENT(OUT)               :: found
      INTEGER, INTENT(OUT), OPTIONAL     :: row_size, col_size

      INTEGER                     :: col_group, row_group
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_get_block_p_${dsuffix}$', &
                                     routineP = moduleN//':'//routineN
      INTEGER                     :: handle

      CALL timeset(routineN, handle)

      CALL block_index_global_to_local(matrix%dist%info, matrix%dist, row=row, column=col, &
                                       row_group=row_group, column_group=col_group)

      CALL dbcsr_get_block_p(matrix%matrix, row_group, col_group, block, transposed, found, row_size=row_size, col_size=col_size)

      CALL timestop(handle)
   END SUBROUTINE
#:endfor

#:for dparam, dtype, dsuffix in dtype_float_list
   SUBROUTINE dbcsr_tas_set_${dsuffix}$ (matrix, alpha)
      !! As dbcsr_set
      TYPE(dbcsr_tas_type), INTENT(INOUT)    :: matrix
      ${dtype}$, INTENT(IN)                :: alpha
      CALL dbcsr_set(matrix%matrix, alpha)
   END SUBROUTINE
#:endfor

#:for dparam, dtype, dsuffix in dtype_float_list
   SUBROUTINE dbcsr_tas_filter_${dsuffix}$ (matrix, eps, method, use_absolute)
      !! As dbcsr_filter
      TYPE(dbcsr_tas_type), INTENT(INOUT)    :: matrix
      ${dtype}$, INTENT(IN)                :: eps
      INTEGER, INTENT(IN), OPTIONAL        :: method
      LOGICAL, INTENT(IN), OPTIONAL        :: use_absolute

      CALL dbcsr_filter(matrix%matrix, eps, method, use_absolute)

   END SUBROUTINE
#:endfor

END MODULE
