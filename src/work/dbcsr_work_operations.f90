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
  SUBROUTINE tree_to_linear_${nametype1}$ (wm)
     !! Converts mutable data to linear (array) type.

     USE dbcsr_btree, &
        ONLY: btree_2d_data_${nametype1}$ => btree_data_${nametype1}$p2d, &
              btree_destroy_${nametype1}$ => btree_delete, &
              btree_size_${nametype1}$ => btree_get_entries
     TYPE(dbcsr_work_type), INTENT(INOUT)     :: wm
        !! work matrix to convert

     CHARACTER(len=*), PARAMETER :: routineN = 'tree_to_linear_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER                                  :: blk, blk_p, treesize, &
                                                 error_handler, needed_size
     INTEGER(KIND=int_8), ALLOCATABLE, &
        DIMENSION(:)                           :: keys
     ${type1}$, DIMENSION(:), POINTER           :: target_data
     ${type1}$, DIMENSION(:, :), POINTER        :: block_2d
     TYPE(btree_2d_data_${nametype1}$), ALLOCATABLE, &
        DIMENSION(:)                           :: values

!   ---------------------------------------------------------------------------

     CALL timeset(routineN, error_handler)
     ! srt = .TRUE. ! Not needed because of the copy
     treesize = btree_size_${nametype1}$ (wm%mutable%m%btree_${nametype1}$)
     IF (wm%lastblk .NE. treesize) &
        DBCSR_ABORT("Mismatch in number of blocks")
     ALLOCATE (keys(treesize), values(treesize))
     CALL btree_destroy_${nametype1}$ (wm%mutable%m%btree_${nametype1}$, keys, values)
     CALL ensure_array_size(wm%row_i, ub=treesize)
     CALL ensure_array_size(wm%col_i, ub=treesize)
     CALL dbcsr_unpack_i8_2i4(keys, wm%row_i, &
                              wm%col_i)
     ! For now we also fill the data, sloooowly, but this should
     ! be avoided and the data should be copied directly from the
     ! source in the subroutine's main loop.
     CALL ensure_array_size(wm%blk_p, ub=treesize)
     needed_size = 0
     DO blk = 1, treesize
        block_2d => values(blk)%p
        needed_size = needed_size + SIZE(block_2d)
     ENDDO
     wm%datasize = needed_size
     CALL dbcsr_data_ensure_size(wm%data_area, &
                                 wm%datasize)
     target_data => dbcsr_get_data_p_${nametype1}$ (wm%data_area)
     blk_p = 1
     DO blk = 1, treesize
        block_2d => values(blk)%p
        IF (.NOT. values(blk)%tr) THEN
           wm%blk_p(blk) = blk_p
        ELSE
           wm%blk_p(blk) = -blk_p
        ENDIF
        CALL block_copy_${nametype1}$ (target_data, block_2d, &
                                       SIZE(block_2d), blk_p, 1)
        blk_p = blk_p + SIZE(block_2d)
        DEALLOCATE (block_2d)
     ENDDO
     DEALLOCATE (keys, values)
     CALL dbcsr_mutable_release(wm%mutable)
     CALL timestop(error_handler)
  END SUBROUTINE tree_to_linear_${nametype1}$

#:endfor
