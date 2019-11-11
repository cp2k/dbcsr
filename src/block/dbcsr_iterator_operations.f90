!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

#:include '../data/dbcsr.fypp'
#:for n, nametype1, base1, prec1, kind1, type1, dkind1 in inst_params_float
  SUBROUTINE iterator_next_1d_block_${nametype1}$ (iterator, row, column, block, &
                                                   transposed, block_number, row_size, col_size, row_offset, col_offset)
     !! Gets the next data block, single/double precision real/complex

     TYPE(dbcsr_iterator), INTENT(INOUT)      :: iterator
        !! the iterator
     INTEGER, INTENT(OUT)                     :: row, column
        !! row of the data block
        !! column of the data block
     ${type1}$, DIMENSION(:), POINTER :: block
        !! pointer to the data block
     LOGICAL, INTENT(OUT)                     :: transposed
        !! whether the block data is transposed
     INTEGER, INTENT(OUT), OPTIONAL           :: block_number
        !! block number
     INTEGER, INTENT(OUT), OPTIONAL           :: row_size, col_size, &
                                                 row_offset, col_offset
        !! logical row size of block
        !! logical column size of block

     CHARACTER(len=*), PARAMETER :: routineN = 'iterator_next_1d_block_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER                                  :: blk_p, bp, csize, nze, rsize

!   ---------------------------------------------------------------------------
! If we're pointing to a valid block, return that block.

     IF (iterator%pos .LE. iterator%nblks &
         .AND. iterator%pos .NE. 0) THEN
        row = iterator%row
        column = iterator%col_i(iterator%pos)
        IF (iterator%transpose) CALL swap(row, column)
        blk_p = iterator%blk_p(iterator%pos)
        transposed = blk_p .LT. 0
        bp = ABS(blk_p)
        rsize = iterator%row_size
        csize = iterator%cbs(column)
        nze = rsize*csize
        IF (PRESENT(row_size)) row_size = rsize
        IF (PRESENT(col_size)) col_size = csize
        IF (PRESENT(row_offset)) row_offset = iterator%row_offset
        IF (PRESENT(col_offset)) col_offset = iterator%coff(column)
        CALL dbcsr_get_data(iterator%data_area, block, &
                            lb=bp, ub=bp + nze - 1)
        IF (PRESENT(block_number)) block_number = iterator%pos
        ! Move to the next non-deleted position.
        CALL iterator_advance(iterator)
        CALL update_row_info(iterator)
     ELSE
        row = 0
        column = 0
        NULLIFY (block)
        IF (PRESENT(block_number)) block_number = 0
     ENDIF
  END SUBROUTINE iterator_next_1d_block_${nametype1}$

  SUBROUTINE iterator_next_2d_block_${nametype1}$ (iterator, row, column, &
                                                   block, transposed, &
                                                   block_number, row_size, col_size, row_offset, col_offset)
     !! Gets the next data block, single/double precision real/complex

     TYPE(dbcsr_iterator), INTENT(INOUT)      :: iterator
        !! the iterator
     INTEGER, INTENT(OUT)                     :: row, column
        !! row of the data block
        !! column of the data block
     ${type1}$, DIMENSION(:, :), &
        POINTER                                :: block
        !! pointer to the data block
     LOGICAL, INTENT(OUT)                     :: transposed
        !! whether the block data is transposed
     INTEGER, INTENT(OUT), OPTIONAL           :: block_number
        !! block number
     INTEGER, INTENT(OUT), OPTIONAL           :: row_size, col_size, row_offset, col_offset
        !! logical row size of block
        !! logical column size of block

     CHARACTER(len=*), PARAMETER :: routineN = 'iterator_next_2d_block_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER                                  :: blk_p, bp, csize, nze, rsize, &
                                                 block_row_size, block_col_size
     ${type1}$, DIMENSION(:), POINTER           :: lin_blk_p
     INTEGER                                  :: error_handle

!   ---------------------------------------------------------------------------
! If we're pointing to a valid block, return that block.

     IF (careful_mod) CALL timeset(routineN, error_handle)
     IF (iterator%pos .LE. iterator%nblks &
         .AND. iterator%pos .NE. 0) THEN
        row = iterator%row
        column = iterator%col_i(iterator%pos)
        IF (iterator%transpose) CALL swap(row, column)
        blk_p = iterator%blk_p(iterator%pos)
        transposed = blk_p .LT. 0
        bp = ABS(blk_p)
        rsize = iterator%row_size
        csize = iterator%cbs(column)
        block_row_size = rsize
        block_col_size = csize
        IF (PRESENT(row_size)) row_size = rsize
        IF (PRESENT(col_size)) col_size = csize
        IF (PRESENT(row_offset)) row_offset = iterator%row_offset
        IF (PRESENT(col_offset)) col_offset = iterator%coff(column)
        nze = rsize*csize
        IF (transposed) CALL swap(rsize, csize)
        CALL dbcsr_get_data(iterator%data_area, lin_blk_p, &
                            lb=bp, ub=bp + nze - 1)
        CALL pointer_rank_remap2(block, rsize, csize, lin_blk_p)
        IF (PRESENT(block_number)) block_number = iterator%pos
        ! Move to the next non-deleted position.
        CALL iterator_advance(iterator)
        CALL update_row_info(iterator)
     ELSE
        row = 0
        column = 0
        NULLIFY (block)
        IF (PRESENT(block_number)) block_number = 0
     ENDIF
     IF (careful_mod) CALL timestop(error_handle)
  END SUBROUTINE iterator_next_2d_block_${nametype1}$
#:endfor
