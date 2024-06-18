!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_block_access
   !! DBCSR block access
   USE dbcsr_array_types, ONLY: array_data
   USE dbcsr_btree, ONLY: btree_add, &
                          btree_data_cp2d, &
                          btree_data_dp2d, &
                          btree_data_sp2d, &
                          btree_data_zp2d, &
                          btree_find
   USE dbcsr_block_operations, ONLY: dbcsr_data_clear, &
                                     dbcsr_data_set
   USE dbcsr_config, ONLY: default_resize_factor
   USE dbcsr_data_methods, ONLY: dbcsr_data_clear_pointer, &
                                 dbcsr_data_ensure_size, &
                                 dbcsr_data_get_size_referenced, &
                                 dbcsr_data_set_pointer, &
                                 dbcsr_get_data, &
                                 dbcsr_get_data_p, &
                                 dbcsr_get_data_p_s, &
                                 dbcsr_get_data_p_d, &
                                 dbcsr_get_data_p_c
   USE dbcsr_dist_methods, ONLY: dbcsr_distribution_local_cols, &
                                 dbcsr_distribution_local_rows, &
                                 dbcsr_distribution_mp
   USE dbcsr_dist_operations, ONLY: dbcsr_get_block_index, &
                                    dbcsr_get_stored_block_info, &
                                    dbcsr_get_stored_coordinates
   USE dbcsr_index_operations, ONLY: dbcsr_addto_index_array, &
                                     dbcsr_clearfrom_index_array, &
                                     dbcsr_expand_row_index, &
                                     dbcsr_make_dbcsr_index, &
                                     dbcsr_sort_indices, &
                                     merge_index_arrays
   USE dbcsr_methods, ONLY: &
      dbcsr_blk_column_size, dbcsr_blk_row_size, dbcsr_distribution, dbcsr_get_data_type, &
      dbcsr_get_num_blocks, dbcsr_mutable_instantiated, dbcsr_mutable_new, dbcsr_nblkrows_total, &
      dbcsr_use_mutable, dbcsr_wm_use_mutable
   USE dbcsr_mp_methods, ONLY: dbcsr_mp_mynode
   USE dbcsr_ptr_util, ONLY: pointer_rank_remap2, &
                             pointer_view
   USE dbcsr_toollib, ONLY: make_coordinate_tuple, &
                            swap
   USE dbcsr_types, ONLY: &
      dbcsr_data_obj, dbcsr_scalar_type, dbcsr_slot_blk_p, dbcsr_slot_col_i, dbcsr_slot_nblks, &
      dbcsr_slot_nze, dbcsr_type, dbcsr_type_complex_4, dbcsr_type_complex_4_2d, &
      dbcsr_type_complex_8, dbcsr_type_complex_8_2d, dbcsr_type_real_4, dbcsr_type_real_4_2d, &
      dbcsr_type_real_8, dbcsr_type_real_8_2d
   USE dbcsr_work_operations, ONLY: add_work_coordinate, &
                                    dbcsr_work_create
   USE dbcsr_kinds, ONLY: dp, &
                          int_8, &
                          real_4, &
                          real_8
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_block_access'

   PUBLIC :: dbcsr_get_block_p, &
             dbcsr_put_block, dbcsr_remove_block

   PUBLIC :: dbcsr_reserve_block2d, &
             dbcsr_reserve_blocks, dbcsr_reserve_all_blocks, dbcsr_reserve_diag_blocks

   INTERFACE dbcsr_get_block_p
      MODULE PROCEDURE dbcsr_get_block_p_d, dbcsr_get_block_p_s, &
         dbcsr_get_block_p_z, dbcsr_get_block_p_c
      MODULE PROCEDURE dbcsr_get_2d_block_p_d, dbcsr_get_2d_block_p_s, &
         dbcsr_get_2d_block_p_z, dbcsr_get_2d_block_p_c
      MODULE PROCEDURE dbcsr_get_block_p_area
   END INTERFACE

   INTERFACE dbcsr_put_block
      MODULE PROCEDURE dbcsr_put_block_area
      MODULE PROCEDURE dbcsr_put_block_d, dbcsr_put_block_s, &
         dbcsr_put_block_z, dbcsr_put_block_c
      MODULE PROCEDURE dbcsr_put_block2d_d, dbcsr_put_block2d_s, &
         dbcsr_put_block2d_z, dbcsr_put_block2d_c
   END INTERFACE

   INTERFACE dbcsr_reserve_block2d
      MODULE PROCEDURE dbcsr_reserve_block2d_s, dbcsr_reserve_block2d_d, &
         dbcsr_reserve_block2d_c, dbcsr_reserve_block2d_z
   END INTERFACE

   INTERFACE dbcsr_set_block_pointer
      MODULE PROCEDURE dbcsr_set_block_pointer_any
      MODULE PROCEDURE dbcsr_set_block_pointer_2d_s, &
         dbcsr_set_block_pointer_2d_d, &
         dbcsr_set_block_pointer_2d_c, &
         dbcsr_set_block_pointer_2d_z
   END INTERFACE

   LOGICAL, PARAMETER :: careful_mod = .FALSE.
   LOGICAL, PARAMETER :: debug_mod = .FALSE.

   INTEGER, PARAMETER, PRIVATE :: rpslot_owner = 1
   INTEGER, PARAMETER, PRIVATE :: rpslot_addblks = 2
   INTEGER, PARAMETER, PRIVATE :: rpslot_addoffset = 3
   INTEGER, PARAMETER, PRIVATE :: rpslot_oldblks = 4
   INTEGER, PARAMETER, PRIVATE :: rpslot_oldoffset = 5
   INTEGER, PARAMETER, PRIVATE :: rpslot_totaloffset = 6
   INTEGER, PARAMETER, PRIVATE :: rpnslots = 6

   LOGICAL, PARAMETER, PRIVATE :: detailed_timing = .FALSE.

   TYPE block_parameters
      LOGICAL :: tr = .FALSE.
      INTEGER :: logical_rows = -1, logical_cols = -1
      INTEGER :: offset = -1, nze = -1
   END TYPE block_parameters

   TYPE dgemm_join
      INTEGER :: p_a = -1, p_b = -1, p_c = -1
      INTEGER :: last_k = -1, last_n = -1
      TYPE(dbcsr_scalar_type) :: alpha = dbcsr_scalar_type(), beta = dbcsr_scalar_type()
   END TYPE dgemm_join

CONTAINS

   SUBROUTINE dbcsr_remove_block(matrix, row, col, block_nze, block_number)
      !! Marks a block for removal from a DBCSR matrix. Handles
      !! symmetric matrices.

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
         !! DBCSR matrix
      INTEGER, INTENT(IN)                                :: row, col, block_nze
         !! row of block to remove
         !! column of block to remove
      INTEGER, INTENT(IN), OPTIONAL                      :: block_number
         !! the block number, if it is known

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_remove_block'

      INTEGER                                            :: b, c, error_handle, r
      LOGICAL                                            :: found, tr

!   ---------------------------------------------------------------------------

      IF (careful_mod) CALL timeset(routineN, error_handle)
      IF (PRESENT(block_number)) THEN
         b = block_number
         IF (block_number .GT. matrix%nblks) &
            DBCSR_ABORT("Block number too big.")
         found = .TRUE.
      ELSE
         CALL dbcsr_get_block_index(matrix, row, col, r, c, tr, found, b)
      END IF
      b = ABS(b)
      IF (found .AND. b .GT. 0) THEN
         ! Mark the block for deletion.
         matrix%blk_p(b) = 0
         matrix%valid = .FALSE.
         ! update nze accordingly
         matrix%nze = matrix%nze - block_nze
         IF (debug_mod) THEN
            IF (matrix%nze < 0) DBCSR_ABORT("nze < 0!")
         END IF
      ELSE
         IF (debug_mod) THEN
            IF (b .EQ. 0) &
               DBCSR_WARN("Block does not exist or is already deleted.")
         END IF
      END IF
      IF (careful_mod) CALL timestop(error_handle)
   END SUBROUTINE dbcsr_remove_block

   SUBROUTINE dbcsr_get_block_p_area(matrix, row, col, block, tr, found, &
                                     row_size, col_size)
      !! Gets a block from a dbcsr matrix as a data area
      !!
      !! Data area
      !! The pointer encapsulated in the data area points to data stored in the
      !! matrix. It must be 2-dimensional.

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! DBCSR matrix
      INTEGER, INTENT(IN)                                :: row, col
         !! the row
         !! the column
      TYPE(dbcsr_data_obj), INTENT(INOUT)                :: block
         !! the block to get
      LOGICAL, INTENT(OUT)                               :: tr, found
         !! whether the data is transposed
         !! whether the block exists in the matrix
      INTEGER, INTENT(OUT), OPTIONAL                     :: row_size, col_size
         !! logical row size of block
         !! logical column size of block

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_block_p_area'

      INTEGER                                            :: blk, csize, error_handle, iw, offset, &
                                                            rsize, stored_col, stored_row
      LOGICAL                                            :: stored_tr
      TYPE(btree_data_cp2d)                              :: data_block_c
      TYPE(btree_data_dp2d)                              :: data_block_d
      TYPE(btree_data_sp2d)                              :: data_block_s
      TYPE(btree_data_zp2d)                              :: data_block_z

!   ---------------------------------------------------------------------------

      IF (careful_mod) CALL timeset(routineN, error_handle)
      CALL dbcsr_get_block_index(matrix, row, col, stored_row, stored_col, &
                                 stored_tr, found, blk, offset)

      tr = stored_tr

      rsize = dbcsr_blk_row_size(matrix, stored_row)
      csize = dbcsr_blk_column_size(matrix, stored_col)
      IF (PRESENT(row_size)) row_size = rsize
      IF (PRESENT(col_size)) col_size = csize

      CALL dbcsr_data_clear_pointer(block)
      IF (found) THEN
         CALL dbcsr_set_block_pointer(matrix, block, rsize, csize, stored_tr, offset)
      ELSEIF (ASSOCIATED(matrix%wms)) THEN
         iw = 1
!$       iw = omp_get_thread_num() + 1
         IF (.NOT. dbcsr_use_mutable(matrix)) &
            DBCSR_ABORT("Can not retrieve blocks from non-mutable work matrices.")
         IF (dbcsr_mutable_instantiated(matrix%wms(iw)%mutable)) THEN
            SELECT CASE (block%d%data_type)
            CASE (dbcsr_type_real_4_2d)
               CALL btree_find( &
                  matrix%wms(iw)%mutable%m%btree_s, &
                  make_coordinate_tuple(stored_row, stored_col), &
                  data_block_s, found)
               IF (found) THEN
                  CALL dbcsr_data_set_pointer(block, data_block_s%p)
               END IF
            CASE (dbcsr_type_real_8_2d)
               CALL btree_find( &
                  matrix%wms(iw)%mutable%m%btree_d, &
                  make_coordinate_tuple(stored_row, stored_col), &
                  data_block_d, found)
               IF (found) THEN
                  CALL dbcsr_data_set_pointer(block, data_block_d%p)
               END IF
            CASE (dbcsr_type_complex_4_2d)
               CALL btree_find( &
                  matrix%wms(iw)%mutable%m%btree_c, &
                  make_coordinate_tuple(stored_row, stored_col), &
                  data_block_c, found)
               IF (found) THEN
                  CALL dbcsr_data_set_pointer(block, data_block_c%p)
               END IF
            CASE (dbcsr_type_complex_8_2d)
               CALL btree_find( &
                  matrix%wms(iw)%mutable%m%btree_z, &
                  make_coordinate_tuple(stored_row, stored_col), &
                  data_block_z, found)
               IF (found) THEN
                  CALL dbcsr_data_set_pointer(block, data_block_z%p)
               END IF
            CASE default
               DBCSR_ABORT("Only 2-D data for block pointers!")
            END SELECT
         END IF
      END IF
      IF (careful_mod) CALL timestop(error_handle)
   END SUBROUTINE dbcsr_get_block_p_area

   SUBROUTINE dbcsr_put_block_area(matrix, row, col, block, lb_row_col, transposed, &
                                   summation, flop, scale)
      !! We allow :
      !! matrix(dp) [+]= [scale(dp)] * block(dp)
      !! matrix(dp) [+]= [scale(dp)] * block(sp)
      !! matrix(sp) [+]= [scale(dp)] * block(sp)

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      INTEGER, INTENT(IN)                                :: row, col
      TYPE(dbcsr_data_obj)                               :: block
      INTEGER, DIMENSION(2), INTENT(INOUT), OPTIONAL     :: lb_row_col
      LOGICAL, INTENT(IN), OPTIONAL                      :: transposed, summation
      INTEGER(KIND=int_8), INTENT(INOUT), OPTIONAL       :: flop
      TYPE(dbcsr_scalar_type), INTENT(IN), OPTIONAL      :: scale

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_put_block_area'

      INTEGER                                            :: data_type_m, error_handle
      LOGICAL                                            :: do_scale

!   ---------------------------------------------------------------------------

      IF (careful_mod) CALL timeset(routineN, error_handle)
      data_type_m = dbcsr_get_data_type(matrix)
      do_scale = PRESENT(scale)
      IF (do_scale) THEN
         !IF(data_type_m /= scale%data_type) &
         !   DBCSR_ABORT("Incompatible data types matrix="//data_type_m//" scale="//scale%data_type)
      END IF
      IF (.NOT. ASSOCIATED(block%d)) &
         DBCSR_ABORT("Can only add valid data block!")
      SELECT CASE (block%d%data_type)
      CASE (dbcsr_type_real_4)
         IF (do_scale) THEN
            IF (data_type_m .EQ. dbcsr_type_real_4) THEN
               CALL dbcsr_put_block(matrix, row, col, dbcsr_get_data_p_s(block), lb_row_col, transposed, &
                                    summation, flop, scale=scale%r_sp)
            ELSEIF (data_type_m .EQ. dbcsr_type_real_8) THEN
               CALL dbcsr_put_block(matrix, row, col, &
                                    REAL(dbcsr_get_data_p_s(block), real_8), lb_row_col, transposed, &
                                    summation, flop, scale=REAL(scale%r_sp, real_8))
            END IF
         ELSE
            IF (data_type_m .EQ. dbcsr_type_real_4) THEN
               CALL dbcsr_put_block(matrix, row, col, dbcsr_get_data_p_s(block), lb_row_col, transposed, &
                                    summation, flop)
            ELSEIF (data_type_m .EQ. dbcsr_type_real_8) THEN
               CALL dbcsr_put_block(matrix, row, col, &
                                    REAL(dbcsr_get_data_p_s(block), real_8), lb_row_col, transposed, &
                                    summation, flop)
            END IF
         END IF
      CASE (dbcsr_type_real_8)
         IF (do_scale) THEN
            CALL dbcsr_put_block(matrix, row, col, dbcsr_get_data_p_d(block), lb_row_col, transposed, &
                                 summation, flop, scale=scale%r_dp)
         ELSE
            CALL dbcsr_put_block(matrix, row, col, dbcsr_get_data_p_d(block), lb_row_col, transposed, &
                                 summation, flop)
         END IF
      CASE (dbcsr_type_complex_4)
         IF (do_scale) THEN
            CALL dbcsr_put_block(matrix, row, col, dbcsr_get_data_p_c(block), lb_row_col, transposed, &
                                 summation, flop, scale=scale%c_sp)
         ELSE
            CALL dbcsr_put_block(matrix, row, col, dbcsr_get_data_p_c(block), lb_row_col, transposed, &
                                 summation, flop)
         END IF
      CASE (dbcsr_type_complex_8)
         IF (do_scale) THEN
            CALL dbcsr_put_block(matrix, row, col, block%d%c_dp, lb_row_col, transposed, &
                                 summation, flop, scale=scale%c_dp)
         ELSE
            CALL dbcsr_put_block(matrix, row, col, block%d%c_dp, lb_row_col, transposed, &
                                 summation, flop)
         END IF
      CASE (dbcsr_type_real_4_2d)
         IF (do_scale) THEN
            CALL dbcsr_put_block(matrix, row, col, block%d%r2_sp, lb_row_col, transposed, &
                                 summation, flop, scale=scale%r_sp)
         ELSE
            CALL dbcsr_put_block(matrix, row, col, block%d%r2_sp, lb_row_col, transposed, &
                                 summation, flop)
         END IF
      CASE (dbcsr_type_real_8_2d)
         IF (do_scale) THEN
            CALL dbcsr_put_block(matrix, row, col, block%d%r2_dp, lb_row_col, transposed, &
                                 summation, flop, scale=scale%r_dp)
         ELSE
            CALL dbcsr_put_block(matrix, row, col, block%d%r2_dp, lb_row_col, transposed, &
                                 summation, flop)
         END IF
      CASE (dbcsr_type_complex_4_2d)
         IF (do_scale) THEN
            CALL dbcsr_put_block(matrix, row, col, block%d%c2_sp, lb_row_col, transposed, &
                                 summation, flop, scale=scale%c_sp)
         ELSE
            CALL dbcsr_put_block(matrix, row, col, block%d%c2_sp, lb_row_col, transposed, &
                                 summation, flop)
         END IF
      CASE (dbcsr_type_complex_8_2d)
         IF (do_scale) THEN
            CALL dbcsr_put_block(matrix, row, col, block%d%c2_dp, lb_row_col, transposed, &
                                 summation, flop, scale=scale%c_dp)
         ELSE
            CALL dbcsr_put_block(matrix, row, col, block%d%c2_dp, lb_row_col, transposed, &
                                 summation, flop)
         END IF
      CASE default
         DBCSR_ABORT("Invalid data type")
      END SELECT
      IF (careful_mod) CALL timestop(error_handle)
   END SUBROUTINE dbcsr_put_block_area

   SUBROUTINE dbcsr_reserve_all_blocks(matrix)
      !! Inserts all blocks of a dbcsr matrix to make it a full matrix.
      !! Thus obviously not linear scaling.

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
         !! Matrix into which blocks should be added.

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_reserve_all_blocks'

      INTEGER                                            :: blk_count, col, col_local, col_s, &
                                                            error_handle, myrank, rank, row, &
                                                            row_local, row_s
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: columns, rows
      INTEGER, DIMENSION(:), POINTER                     :: local_cols, local_rows
      LOGICAL                                            :: tr

      CALL timeset(routineN, error_handle)

      myrank = dbcsr_mp_mynode(dbcsr_distribution_mp(dbcsr_distribution(matrix)))
      local_rows => dbcsr_distribution_local_rows(dbcsr_distribution(matrix))
      local_cols => dbcsr_distribution_local_cols(dbcsr_distribution(matrix))

      blk_count = 0
      ! should be possible to loop only over the local blockrows/blockcols
      DO row_local = 1, SIZE(local_rows)
         DO col_local = 1, SIZE(local_cols)
            tr = .FALSE.
            row = local_rows(row_local)
            col = local_cols(col_local)
            row_s = row; col_s = col
            CALL dbcsr_get_stored_coordinates(matrix, row_s, col_s, rank)
            ! is that the correct condition for symmetric matrices ?
            IF (rank .EQ. myrank .AND. row_s .EQ. row .AND. col_s .EQ. col) blk_count = blk_count + 1
         END DO
      END DO

      ALLOCATE (rows(blk_count), columns(blk_count))

      blk_count = 0
      DO row_local = 1, SIZE(local_rows)
         DO col_local = 1, SIZE(local_cols)
            tr = .FALSE.
            row = local_rows(row_local)
            col = local_cols(col_local)
            row_s = row; col_s = col
            CALL dbcsr_get_stored_coordinates(matrix, row_s, col_s, rank)
            IF (rank .EQ. myrank .AND. row_s .EQ. row .AND. col_s .EQ. col) THEN
               blk_count = blk_count + 1
               rows(blk_count) = row
               columns(blk_count) = col
            END IF
         END DO
      END DO

      CALL dbcsr_reserve_blocks(matrix, rows, columns)

      CALL timestop(error_handle)

   END SUBROUTINE dbcsr_reserve_all_blocks

   SUBROUTINE dbcsr_reserve_diag_blocks(matrix)
      !! Inserts diagonal blocks of a dbcsr matrix to make it a matrix with at least all diagonal blocks present

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
         !! Matrix into which blocks should be added.

      INTEGER                                            :: blk_count, col, col_s, myrank, rank, &
                                                            row, row_s
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: columns, rows
      LOGICAL                                            :: tr

      myrank = dbcsr_mp_mynode(dbcsr_distribution_mp(dbcsr_distribution(matrix)))

      blk_count = 0
      ! should be possible to loop only over the local blockrows/blockcols
      DO row = 1, dbcsr_nblkrows_total(matrix)
         col = row
         tr = .FALSE.
         row_s = row; col_s = col
         CALL dbcsr_get_stored_coordinates(matrix, row_s, col_s, rank)
         IF (rank .EQ. myrank .AND. row_s .EQ. row .AND. col_s .EQ. col) blk_count = blk_count + 1
      END DO

      ALLOCATE (rows(blk_count), columns(blk_count))

      blk_count = 0
      DO row = 1, dbcsr_nblkrows_total(matrix)
         col = row
         tr = .FALSE.
         row_s = row; col_s = col
         CALL dbcsr_get_stored_coordinates(matrix, row_s, col_s, rank)
         IF (rank .EQ. myrank .AND. row_s .EQ. row .AND. col_s .EQ. col) THEN
            blk_count = blk_count + 1
            rows(blk_count) = row
            columns(blk_count) = col
         END IF
      END DO

      CALL dbcsr_reserve_blocks(matrix, rows, columns)

   END SUBROUTINE dbcsr_reserve_diag_blocks

   SUBROUTINE dbcsr_reserve_blocks(matrix, rows, columns, blk_pointers)
      !! Inserts block reservations into a matrix, avoiding the work matrix.
      !!
      !! Data
      !! No data can be specified; instead, space is reserved and zeroed. To
      !! add data, call dbcsr_put_block afterwards.
      !!
      !! Reserving existing blocks
      !! Duplicates are not added, but allocations may be greater than
      !! the minimum necessary.
      !!
      !! blk_pointers
      !! When blk_pointers is passed, the newly added blocks use these pointers.
      !! No data is cleared in this case

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
         !! Matrix into which blocks should be added.
      INTEGER, DIMENSION(:), INTENT(IN)                  :: rows, columns
         !! Rows of the blocks to add
         !! Columns of the blocks to add
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL        :: blk_pointers
         !! block pointers to use for new blocks

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_reserve_blocks'

      INTEGER :: blk, blk_p, data_size_new, data_size_old, handle, nblkrows, nblks_actual_added, &
                 nblks_added, nblks_new, nblks_old, new_data_sizes, nze
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: add_blkp, add_cols, add_rows, &
                                                            added_sizes, new_blk_p, new_col_i, &
                                                            new_row_i, old_row_i
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: added_blk_info

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)
      IF (SIZE(rows) /= SIZE(columns)) &
         DBCSR_ABORT("Size of rows and columns array must match.")
      IF (PRESENT(blk_pointers)) THEN
         IF (SIZE(rows) /= SIZE(blk_pointers)) &
            DBCSR_ABORT("Size of rows and block pointecs arrays must match.")
         data_size_old = 0
      ELSE
         ! Get current data size
         data_size_old = dbcsr_data_get_size_referenced(matrix%data_area)
      END IF
      ! Ensures that the rows and columns are sorted.
      nblks_added = SIZE(rows)
      ALLOCATE (add_rows(nblks_added))
      add_rows(:) = rows(:)
      ALLOCATE (add_cols(nblks_added))
      add_cols(:) = columns(:)
      IF (PRESENT(blk_pointers)) THEN
         ALLOCATE (add_blkp(nblks_added))
         add_blkp(:) = blk_pointers(:)
         CALL dbcsr_sort_indices(nblks_added, add_rows, add_cols, &
                                 blk_p=add_blkp)
      ELSE
         CALL dbcsr_sort_indices(nblks_added, add_rows, add_cols)
      END IF
      nblks_old = dbcsr_get_num_blocks(matrix)
      nblkrows = dbcsr_nblkrows_total(matrix)
      IF (SIZE(rows) .GT. 0 .AND. nblkrows .LE. 0) &
         DBCSR_ABORT("Can not add blocks to matrix with no rows.")
      ! Adjust the index.
      ! Get the old row indices
      ALLOCATE (old_row_i(nblks_old))
      CALL dbcsr_expand_row_index(matrix%row_p, old_row_i, &
                                  nblkrows, nblks_old)
      ! Calculate new block pointers. Possibly high estimates.
      new_data_sizes = 0
      blk_p = data_size_old + 1 ! New blocks start at the end of the old
      ALLOCATE (added_blk_info(3, nblks_added))
      ALLOCATE (added_sizes(nblks_added))
      DO blk = 1, nblks_added
         IF (PRESENT(blk_pointers)) THEN
            blk_p = add_blkp(blk)
         END IF
         added_blk_info(1:3, blk) = (/add_rows(blk), add_cols(blk), blk_p/)
         nze = dbcsr_blk_row_size(matrix, add_rows(blk)) &
               *dbcsr_blk_column_size(matrix, add_cols(blk))
         added_sizes(blk) = nze
         blk_p = blk_p + nze
      END DO
      DEALLOCATE (add_rows)
      DEALLOCATE (add_cols)
      IF (PRESENT(blk_pointers)) DEALLOCATE (add_blkp)
      !
      nblks_new = nblks_old + nblks_added ! Possibly high estimate
      ALLOCATE (new_row_i(nblks_new))
      ALLOCATE (new_col_i(nblks_new))
      ALLOCATE (new_blk_p(nblks_new))
      ! Merge the two indices
      IF (PRESENT(blk_pointers)) THEN
         CALL merge_index_arrays(new_row_i, new_col_i, new_blk_p, nblks_new, &
                                 old_row_i, matrix%col_i, matrix%blk_p, nblks_old, &
                                 added_blk_info, nblks_added, added_nblks=nblks_actual_added)
         data_size_new = 0
      ELSE
         CALL merge_index_arrays(new_row_i, new_col_i, new_blk_p, nblks_new, &
                                 old_row_i, matrix%col_i, matrix%blk_p, nblks_old, &
                                 added_blk_info, nblks_added, added_nblks=nblks_actual_added, &
                                 added_sizes=added_sizes, added_size_offset=data_size_old + 1, &
                                 added_size=data_size_new)
      END IF
      nblks_new = nblks_actual_added + nblks_old
      ! Free some memory
      DEALLOCATE (added_blk_info)
      DEALLOCATE (added_sizes)
      DEALLOCATE (old_row_i)
      ! We can skip this if no block was actually added.
      IF (nblks_actual_added .GT. 0) THEN
         ! Write the new index
         matrix%nblks = nblks_new
         matrix%nze = matrix%nze + data_size_new
         matrix%index(dbcsr_slot_nblks) = matrix%nblks
         matrix%index(dbcsr_slot_nze) = matrix%index(dbcsr_slot_nze)
         CALL dbcsr_clearfrom_index_array(matrix, dbcsr_slot_col_i)
         CALL dbcsr_clearfrom_index_array(matrix, dbcsr_slot_blk_p)
         CALL dbcsr_addto_index_array(matrix, dbcsr_slot_col_i, &
                                      new_col_i(1:nblks_new), &
                                      extra=nblks_new)
         CALL dbcsr_addto_index_array(matrix, dbcsr_slot_blk_p, &
                                      new_blk_p(1:nblks_new))
         CALL dbcsr_make_dbcsr_index(matrix%row_p, new_row_i(1:nblks_new), &
                                     nblkrows, nblks_new)
         IF (.NOT. PRESENT(blk_pointers)) THEN
            ! Resize data area to fit the new blocks.
            CALL dbcsr_data_ensure_size(matrix%data_area, &
                                        data_size=matrix%nze)
            ! Zero the new data blocks.
            CALL dbcsr_data_clear(matrix%data_area, &
                                  lb=data_size_old + 1, ub=matrix%nze)
         END IF
      END IF
      CALL timestop(handle)
   END SUBROUTINE dbcsr_reserve_blocks

   SUBROUTINE dbcsr_set_block_pointer_any(matrix, pointer_any, &
                                          rsize, csize, main_tr, base_offset)
      !! Sets a pointer, possibly using the buffers.

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! Matrix to use
      TYPE(dbcsr_data_obj), INTENT(INOUT)                :: pointer_any
         !! The pointer to set
      INTEGER, INTENT(IN)                                :: rsize, csize
         !! Row sizes of block to point to
         !! Column sizes of block to point to
      LOGICAL, INTENT(IN)                                :: main_tr
         !! Whether block is transposed in the matrix
      INTEGER, INTENT(IN)                                :: base_offset
         !! The block pointer

!   ---------------------------------------------------------------------------

      IF (main_tr) THEN
         CALL dbcsr_data_set_pointer(pointer_any, csize, rsize, &
                                     matrix%data_area, source_lb=base_offset)
      ELSE
         CALL dbcsr_data_set_pointer(pointer_any, rsize, csize, &
                                     matrix%data_area, source_lb=base_offset)
      END IF
   END SUBROUTINE dbcsr_set_block_pointer_any

   #:include '../data/dbcsr.fypp'

   #:for n, nametype1, base1, prec1, kind1, type1, dkind1 in inst_params_float
      SUBROUTINE dbcsr_get_2d_block_p_${nametype1}$ (matrix, row, col, block, tr, found, &
                                                     row_size, col_size)
     !! Gets a 2-d block from a dbcsr matrix

         TYPE(dbcsr_type), INTENT(INOUT)           :: matrix
        !! DBCSR matrix
         INTEGER, INTENT(IN)                      :: row, col
        !! the row
        !! the column
         ${type1}$, DIMENSION(:, :), POINTER         :: block
        !! the block to get (rank-2 array)
         LOGICAL, INTENT(OUT)                     :: tr
        !! whether the data is transposed
         LOGICAL, INTENT(OUT)                     :: found
        !! whether the block exists in the matrix
         INTEGER, INTENT(OUT), OPTIONAL           :: row_size, col_size
        !! logical row size of block
        !! logical column size of block

         CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_2d_block_p_${nametype1}$'

         ${type1}$, DIMENSION(:), POINTER         :: block_1d
         INTEGER                                  :: rsize, csize, &
                                                     blk, nze, offset, &
                                                     stored_row, &
                                                     stored_col, iw, nwms
         INTEGER                                  :: error_handle
         TYPE(btree_data_${nametype1}$p2d)          :: data_block
         LOGICAL                                  :: stored_tr
         ${type1}$, DIMENSION(1, 1), TARGET, SAVE    :: block0
!   ---------------------------------------------------------------------------
         IF (careful_mod) CALL timeset(routineN, error_handle)
         IF (debug_mod) THEN
            IF (matrix%data_type /= ${dkind1}$) &
               DBCSR_ABORT("Data type mismatch for requested block.")
         END IF

         CALL dbcsr_get_block_index(matrix, row, col, stored_row, stored_col, &
                                    stored_tr, found, blk, offset)
         tr = stored_tr

         rsize = dbcsr_blk_row_size(matrix, stored_row)
         csize = dbcsr_blk_column_size(matrix, stored_col)
         IF (PRESENT(row_size)) row_size = rsize
         IF (PRESENT(col_size)) col_size = csize

         NULLIFY (block)
         IF (found) THEN
            nze = rsize*csize
            IF (nze .eq. 0) THEN
               found = .TRUE.
               block => block0(1:0, 1:0)
            ELSE
               block_1d => pointer_view(dbcsr_get_data_p( &
                                        matrix%data_area, ${zero1[n]}$), offset, offset + nze - 1)
               CALL dbcsr_set_block_pointer(matrix, block, rsize, csize, offset)
            END IF
         ELSEIF (ASSOCIATED(matrix%wms)) THEN
            nwms = SIZE(matrix%wms)
            iw = 1
!$          IF (nwms < omp_get_num_threads()) &
!$             DBCSR_ABORT("Number of work matrices not equal to number of threads")
!$          iw = omp_get_thread_num() + 1
            IF (.NOT. dbcsr_use_mutable(matrix)) &
               DBCSR_ABORT("Can not retrieve blocks from non-mutable work matrices.")
            IF (dbcsr_use_mutable(matrix)) THEN
               IF (.NOT. dbcsr_mutable_instantiated(matrix%wms(iw)%mutable)) THEN
                  CALL dbcsr_mutable_new(matrix%wms(iw)%mutable, &
                                         dbcsr_get_data_type(matrix))
               END IF
               CALL btree_find( &
                  matrix%wms(iw)%mutable%m%btree_${nametype1}$, &
                  make_coordinate_tuple(stored_row, stored_col), &
                  data_block, found)
               IF (found) THEN
                  block => data_block%p
               END IF
            END IF
         END IF
         IF (careful_mod) CALL timestop(error_handle)
      END SUBROUTINE dbcsr_get_2d_block_p_${nametype1}$

      SUBROUTINE dbcsr_get_block_p_${nametype1}$ (matrix, row, col, block, tr, found, &
                                                  row_size, col_size)
     !! Gets a 1-d block from a dbcsr matrix

         TYPE(dbcsr_type), INTENT(IN)              :: matrix
        !! DBCSR matrix
         INTEGER, INTENT(IN)                      :: row, col
        !! the row
        !! the column
         ${type1}$, DIMENSION(:), POINTER           :: block
        !! the block to get (rank-1 array)
         LOGICAL, INTENT(OUT)                     :: tr
        !! whether the data is transposed
         LOGICAL, INTENT(OUT)                     :: found
        !! whether the block exists in the matrix
         INTEGER, INTENT(OUT), OPTIONAL           :: row_size, col_size
        !! logical row size of block
        !! logical column size of block

         INTEGER                                  :: blk, csize, &
                                                     nze, offset, &
                                                     rsize, stored_row, &
                                                     stored_col
         LOGICAL                                  :: stored_tr

!   ---------------------------------------------------------------------------

         IF (debug_mod) THEN
            IF (matrix%data_type /= ${dkind1}$) &
               DBCSR_ABORT("Data type mismatch for requested block.")
         END IF

         CALL dbcsr_get_block_index(matrix, row, col, stored_row, stored_col, &
                                    stored_tr, found, blk, offset)
         tr = stored_tr

         rsize = dbcsr_blk_row_size(matrix, stored_row)
         csize = dbcsr_blk_column_size(matrix, stored_col)
         IF (PRESENT(row_size)) row_size = rsize
         IF (PRESENT(col_size)) col_size = csize

         NULLIFY (block)
         IF (found) THEN
            nze = rsize*csize
            !
            block => pointer_view( &
                     dbcsr_get_data_p(matrix%data_area, ${zero1[n]}$), offset, offset + nze - 1 &
                     )
         ELSEIF (ASSOCIATED(matrix%wms)) THEN
            IF (.NOT. dbcsr_use_mutable(matrix)) &
               DBCSR_ABORT("Can not retrieve blocks from non-mutable work matrices.")
            IF (dbcsr_use_mutable(matrix)) &
               DBCSR_ABORT("Can not retrieve rank-1 block pointers from mutable work matrices.")
         END IF
      END SUBROUTINE dbcsr_get_block_p_${nametype1}$

      SUBROUTINE dbcsr_reserve_block2d_${nametype1}$ (matrix, row, col, block, &
                                                      transposed, existed)
     !! Put a 2-D block in a DBCSR matrix using the btree

         TYPE(dbcsr_type), INTENT(INOUT)           :: matrix
        !! DBCSR matrix
         INTEGER, INTENT(IN)                      :: row, col
        !! the row
        !! the column
         ${type1}$, DIMENSION(:, :), POINTER         :: block
        !! the block to reserve; added if not NULL
         LOGICAL, INTENT(IN), OPTIONAL            :: transposed
        !! the block holds transposed data
         LOGICAL, INTENT(OUT), OPTIONAL           :: existed
        !! block already existed

         TYPE(btree_data_${nametype1}$p2d)          :: data_block, data_block2
         INTEGER                                  :: col_size, row_size, &
                                                     stored_row, stored_col, &
                                                     iw, nwms
         INTEGER, DIMENSION(:), POINTER           :: col_blk_size, row_blk_size
         LOGICAL                                  :: found, gift, tr, sym_tr
         ${type1}$, DIMENSION(:, :), POINTER         :: original_block

!   ---------------------------------------------------------------------------

         gift = ASSOCIATED(block)
         IF (gift) THEN
            original_block => block
         ELSE
            NULLIFY (original_block)
         END IF
         row_blk_size => array_data(matrix%row_blk_size)
         col_blk_size => array_data(matrix%col_blk_size)
         row_size = row_blk_size(row)
         col_size = col_blk_size(col)

         stored_row = row; stored_col = col
         IF (PRESENT(transposed)) THEN
            tr = transposed
         ELSE
            tr = .FALSE.
         END IF
         sym_tr = .FALSE.
         CALL dbcsr_get_stored_coordinates(matrix, stored_row, stored_col)
         IF (.NOT. ASSOCIATED(matrix%wms)) THEN
            CALL dbcsr_work_create(matrix, work_mutable=.TRUE.)
!$OMP        MASTER
            matrix%valid = .FALSE.
!$OMP        END MASTER
!$OMP        BARRIER
         END IF

         NULLIFY (data_block%p)
         IF (.NOT. gift) THEN
            ALLOCATE (data_block%p(row_size, col_size))
            block => data_block%p
         ELSE
            data_block%p => block
         END IF
         data_block%tr = tr

         nwms = SIZE(matrix%wms)
         iw = 1
!$       IF (nwms < omp_get_num_threads()) &
!$          DBCSR_ABORT("Number of work matrices not equal to number of threads")
!$       iw = omp_get_thread_num() + 1
         CALL btree_add(matrix%wms(iw)%mutable%m%btree_${nametype1}$, &
                        make_coordinate_tuple(stored_row, stored_col), &
                        data_block, found, data_block2)

         IF (.NOT. found) THEN
#if defined(_OPENMP) && (200711 <= _OPENMP)
!$OMP ATOMIC WRITE
            matrix%valid = .FALSE.
#else
!$OMP CRITICAL (critical_reserve_block2d)
            matrix%valid = .FALSE.
!$OMP END CRITICAL (critical_reserve_block2d)
#endif
            matrix%wms(iw)%lastblk = matrix%wms(iw)%lastblk + 1
            matrix%wms(iw)%datasize = matrix%wms(iw)%datasize + row_size*col_size
         ELSE
            IF (.NOT. gift) THEN
               DEALLOCATE (data_block%p)
            ELSE
               DEALLOCATE (original_block)
            END IF
            block => data_block2%p
         END IF
         IF (PRESENT(existed)) existed = found
      END SUBROUTINE dbcsr_reserve_block2d_${nametype1}$

      SUBROUTINE dbcsr_put_block2d_${nametype1}$ (matrix, row, col, block, lb_row_col, transposed, &
                                                  summation, flop, scale)
     !! Put a 2-D block in a DBCSR matrix

         TYPE(dbcsr_type), INTENT(INOUT)           :: matrix
        !! DBCSR matrix
         INTEGER, INTENT(IN)                      :: row, col
        !! the row
        !! the column
         ${type1}$, DIMENSION(:, :), INTENT(IN), &
            CONTIGUOUS, TARGET                    :: block
        !! the block to put
         INTEGER, DIMENSION(2), OPTIONAL, INTENT(INOUT) :: lb_row_col
         LOGICAL, INTENT(IN), OPTIONAL            :: transposed, summation
        !! the block is transposed
        !! if block exists, then sum the new block to the old one instead of replacing it
         INTEGER(KIND=int_8), INTENT(INOUT), OPTIONAL :: flop
         ${type1}$, INTENT(IN), OPTIONAL            :: scale
        !! scale the block being added

         ${type1}$, DIMENSION(:), POINTER           :: block_1d

         NULLIFY (block_1d)

         block_1d(1:SIZE(block)) => block

         CALL dbcsr_put_block(matrix, row, col, block_1d, lb_row_col, transposed, summation, flop, scale)
      END SUBROUTINE dbcsr_put_block2d_${nametype1}$

      SUBROUTINE dbcsr_put_block_${nametype1}$ (matrix, row, col, block, lb_row_col, transposed, &
                                                summation, flop, scale)
     !! Inserts a block in a dbcsr matrix.
     !! If the block exists, the current data is overwritten.

         TYPE(dbcsr_type), INTENT(INOUT)           :: matrix
        !! DBCSR matrix
         INTEGER, INTENT(IN)                      :: row, col
        !! the logical row
        !! the logical column
         ${type1}$, DIMENSION(:), CONTIGUOUS, INTENT(IN) :: block
        !! the block to put
         INTEGER, DIMENSION(2), OPTIONAL, INTENT(INOUT) :: lb_row_col
         LOGICAL, INTENT(IN), OPTIONAL            :: transposed, summation
        !! the block is transposed
        !! if block exists, then sum the new block to the old one instead of replacing it
         INTEGER(KIND=int_8), INTENT(INOUT), OPTIONAL :: flop
         ${type1}$, INTENT(IN), OPTIONAL            :: scale
        !! scale the OBblock being added

         TYPE(btree_data_${nametype1}$p2d)          :: data_block, data_block2
         INTEGER                                  :: blk, col_size, &
                                                     nze, offset, &
                                                     row_size, blk_p, &
                                                     stored_row, stored_col, &
                                                     iw, nwms
         LOGICAL                                  :: found, tr, do_sum, tr_diff
         ${type1}$, DIMENSION(:), POINTER         :: block_1d
         INTEGER(KIND=int_8)                      :: my_flop

!   ---------------------------------------------------------------------------
         IF (PRESENT(transposed)) THEN
            tr = transposed
         ELSE
            tr = .FALSE.
         END IF
         IF (PRESENT(summation)) THEN
            do_sum = summation
         ELSE
            do_sum = .FALSE.
         END IF
         my_flop = 0
         row_size = dbcsr_blk_row_size(matrix, row)
         col_size = dbcsr_blk_column_size(matrix, col)
         IF (tr) CALL swap(row_size, col_size)

         stored_row = row; stored_col = col
         nze = row_size*col_size
         !
         IF (debug_mod .AND. SIZE(block) < nze) &
            DBCSR_ABORT("Invalid block dimensions")
         CALL dbcsr_get_stored_block_info(matrix, stored_row, stored_col, &
                                          found, blk, lb_row_col, offset)
         IF (found) THEN
            ! let's copy the block
            offset = ABS(offset)
            ! Fix the index if the new block's transpose flag is different
            ! from the old one.
            tr_diff = .FALSE.
            IF (matrix%blk_p(blk) .LT. 0 .NEQV. tr) THEN
               tr_diff = .TRUE.
               matrix%blk_p(blk) = -matrix%blk_p(blk)
            END IF
            block_1d => pointer_view(dbcsr_get_data_p( &
                                     matrix%data_area, ${zero1[n]}$), offset, offset + nze - 1)
            IF (nze .GT. 0) THEN
               IF (do_sum) THEN
                  IF (tr_diff) &
                     block_1d = RESHAPE(TRANSPOSE(RESHAPE(block_1d, (/col_size, row_size/))), (/nze/))
                  IF (PRESENT(scale)) THEN
                     CALL ${nametype1}$axpy(nze, scale, block(1:nze), 1, &
                                            block_1d, 1)
                  ELSE
                     CALL ${nametype1}$axpy(nze, ${one1[n]}$, block(1:nze), 1, &
                                            block_1d, 1)
                  END IF
                  my_flop = my_flop + nze*2
               ELSE
                  IF (PRESENT(scale)) THEN
                     CALL ${nametype1}$copy(nze, scale*block(1:nze), 1, &
                                            block_1d, 1)
                  ELSE
                     CALL ${nametype1}$copy(nze, block(1:nze), 1, &
                                            block_1d, 1)
                  END IF
               END IF
            END IF
         ELSE
        !!@@@
            !call dbcsr_assert (associated (matrix%wms), dbcsr_fatal_level,&
            !     dbcsr_caller_error, routineN, "Work matrices not prepared")
            IF (.NOT. ASSOCIATED(matrix%wms)) THEN
               CALL dbcsr_work_create(matrix, nblks_guess=1, &
                                      sizedata_guess=nze)
            END IF
            nwms = SIZE(matrix%wms)
            iw = 1
!$          IF (debug_mod .AND. nwms < omp_get_num_threads()) &
!$             DBCSR_ABORT("Number of work matrices not equal to number of threads")
!$          iw = omp_get_thread_num() + 1
            blk_p = matrix%wms(iw)%datasize + 1
            IF (.NOT. dbcsr_wm_use_mutable(matrix%wms(iw))) THEN
               IF (tr) blk_p = -blk_p
               CALL add_work_coordinate(matrix%wms(iw), row, col, blk_p)
               CALL dbcsr_data_ensure_size(matrix%wms(iw)%data_area, &
                                           matrix%wms(iw)%datasize + nze, &
                                           factor=default_resize_factor)
               IF (PRESENT(scale)) THEN
                  CALL dbcsr_data_set(matrix%wms(iw)%data_area, ABS(blk_p), &
                                      data_size=nze, src=scale*block, source_lb=1)
               ELSE
                  CALL dbcsr_data_set(matrix%wms(iw)%data_area, ABS(blk_p), &
                                      data_size=nze, src=block, source_lb=1)
               END IF
            ELSE
               ALLOCATE (data_block%p(row_size, col_size))
               IF (PRESENT(scale)) THEN
                  data_block%p(:, :) = scale*RESHAPE(block, (/row_size, col_size/))
               ELSE
                  data_block%p(:, :) = RESHAPE(block, (/row_size, col_size/))
               END IF
               data_block%tr = tr
               IF (.NOT. dbcsr_mutable_instantiated(matrix%wms(iw)%mutable)) THEN
                  CALL dbcsr_mutable_new(matrix%wms(iw)%mutable, &
                                         dbcsr_get_data_type(matrix))
               END IF
               IF (.NOT. do_sum) THEN
                  CALL btree_add( &
                     matrix%wms(iw)%mutable%m%btree_${nametype1}$, &
                     make_coordinate_tuple(stored_row, stored_col), &
                     data_block, found, data_block2, replace=.TRUE.)
                  IF (found) THEN
                     IF (.NOT. ASSOCIATED(data_block2%p)) &
                        DBCSR_WARN("Data was not present in block")
                     IF (ASSOCIATED(data_block2%p)) DEALLOCATE (data_block2%p)
                  END IF
               ELSE
                  CALL btree_add( &
                     matrix%wms(iw)%mutable%m%btree_${nametype1}$, &
                     make_coordinate_tuple(stored_row, stored_col), &
                     data_block, found, data_block2, replace=.FALSE.)
                  IF (found) THEN
                     IF (nze > 0) &
                        CALL ${nametype1}$axpy(nze, ${one1[n]}$, block, 1, &
                                               data_block2%p, 1)
                     IF (.NOT. ASSOCIATED(data_block%p)) &
                        DBCSR_WARN("Data was not present in block")
                     IF (ASSOCIATED(data_block%p)) DEALLOCATE (data_block%p)
                  END IF
               END IF
               IF (.NOT. found) THEN
                  matrix%wms(iw)%lastblk = matrix%wms(iw)%lastblk + 1
               END IF
            END IF
            IF (.NOT. found) THEN
               matrix%wms(iw)%datasize = matrix%wms(iw)%datasize + nze
            END IF
!$OMP ATOMIC WRITE
            matrix%valid = .FALSE.
         END IF
         IF (PRESENT(flop)) flop = flop + my_flop
      END SUBROUTINE dbcsr_put_block_${nametype1}$

      SUBROUTINE dbcsr_set_block_pointer_2d_${nametype1}$ ( &
         matrix, pointer_any, rsize, csize, base_offset)
     !! Sets a pointer, possibly using the buffers.

         TYPE(dbcsr_type), INTENT(IN)              :: matrix
        !! Matrix to use
         ${type1}$, DIMENSION(:, :), POINTER         :: pointer_any
        !! The pointer to set
         INTEGER, INTENT(IN)                      :: rsize, csize
        !! Row size of block to point to
        !! Column size of block to point to
         INTEGER, INTENT(IN)                      :: base_offset
        !! The block pointer

         CHARACTER(len=*), PARAMETER :: &
            routineN = 'dbcsr_set_block_pointer_2d_${nametype1}$'

         INTEGER                                  :: error_handler
         ${type1}$, DIMENSION(:), POINTER           :: lin_blk_p

!   ---------------------------------------------------------------------------

         IF (careful_mod) CALL timeset(routineN, error_handler)
         CALL dbcsr_get_data(matrix%data_area, lin_blk_p, &
                             lb=base_offset, ub=base_offset + rsize*csize - 1)
         CALL pointer_rank_remap2(pointer_any, rsize, csize, &
                                  lin_blk_p)
         IF (careful_mod) CALL timestop(error_handler)
      END SUBROUTINE dbcsr_set_block_pointer_2d_${nametype1}$
   #:endfor

END MODULE dbcsr_block_access
