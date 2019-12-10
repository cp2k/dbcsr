!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tas_util

   !! often used utilities for tall-and-skinny matrices

   USE dbcsr_mp_methods, ONLY: dbcsr_mp_new
   USE dbcsr_types, ONLY: dbcsr_mp_obj, &
                          dbcsr_transpose, &
                          dbcsr_no_transpose
   USE dbcsr_kinds, ONLY: int_8
   USE dbcsr_mpiwrap, ONLY: mp_cart_rank, &
                            mp_environ
   USE dbcsr_index_operations, ONLY: dbcsr_sort_indices
#include "../base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tas_util'

   PUBLIC :: &
      array_eq_i8, &
      dbcsr_mp_environ, &
      index_unique, &
      invert_transpose_flag, &
      swap

   INTERFACE swap
      MODULE PROCEDURE swap_i8
      MODULE PROCEDURE swap_i
   END INTERFACE

CONTAINS
   FUNCTION dbcsr_mp_environ(mp_comm)
      !! Create a dbcsr mp environment from communicator
      INTEGER, INTENT(IN)                                :: mp_comm
      TYPE(dbcsr_mp_obj)                                 :: dbcsr_mp_environ

      INTEGER                                            :: mynode, numnodes, pcol, prow
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: pgrid
      INTEGER, DIMENSION(2)                              :: coord, mycoord, npdims

      CALL mp_environ(numnodes, npdims, mycoord, mp_comm)
      CALL mp_environ(numnodes, mynode, mp_comm)
      ALLOCATE (pgrid(0:npdims(1) - 1, 0:npdims(2) - 1))
      DO prow = 0, npdims(1) - 1
         DO pcol = 0, npdims(2) - 1
            coord = (/prow, pcol/)
            CALL mp_cart_rank(mp_comm, coord, pgrid(prow, pcol))
         ENDDO
      ENDDO

      DBCSR_ASSERT(mynode == pgrid(mycoord(1), mycoord(2)))

      CALL dbcsr_mp_new(dbcsr_mp_environ, mp_comm, pgrid, mynode, numnodes, mycoord(1), mycoord(2))
   END FUNCTION

   SUBROUTINE swap_i8(arr)
      INTEGER(KIND=int_8), DIMENSION(2), INTENT(INOUT)   :: arr

      INTEGER(KIND=int_8)                                :: tmp

      tmp = arr(1)
      arr(1) = arr(2)
      arr(2) = tmp
   END SUBROUTINE

   SUBROUTINE swap_i(arr)
      INTEGER, DIMENSION(2), INTENT(INOUT)               :: arr

      INTEGER                                            :: tmp

      tmp = arr(1)
      arr(1) = arr(2)
      arr(2) = tmp
   END SUBROUTINE

   SUBROUTINE index_unique(index_in, index_out)
      !! Get all unique elements in index_in
      INTEGER, DIMENSION(:, :), INTENT(IN)   :: index_in
      INTEGER, ALLOCATABLE, &
         DIMENSION(:, :), INTENT(OUT)                    :: index_out

      INTEGER                                            :: blk, count, orig_size
      INTEGER, ALLOCATABLE, DIMENSION(:, :)  :: index_tmp
      INTEGER, DIMENSION(2)                  :: prev_index
      INTEGER, DIMENSION(1:SIZE(index_in, 1) &
                         , 1:SIZE(index_in, 2))                          :: index_sorted

      orig_size = SIZE(index_in, 1)
      ALLOCATE (index_tmp(orig_size, 2))
      index_sorted(:, :) = index_in(:, :)
      CALL dbcsr_sort_indices(orig_size, index_sorted(:, 1), index_sorted(:, 2))
      count = 0
      prev_index(:) = [0, 0]
      DO blk = 1, orig_size
         IF (ANY(index_sorted(blk, :) .NE. prev_index(:))) THEN
            count = count + 1
            index_tmp(count, :) = index_sorted(blk, :)
            prev_index(:) = index_sorted(blk, :)
         ENDIF
      ENDDO

      ALLOCATE (index_out(count, 2))
      index_out(:, :) = index_tmp(1:count, :)

   END SUBROUTINE

   SUBROUTINE invert_transpose_flag(trans_flag)
      CHARACTER(LEN=1), INTENT(INOUT)                    :: trans_flag

      IF (trans_flag == dbcsr_transpose) THEN
         trans_flag = dbcsr_no_transpose
      ELSEIF (trans_flag == dbcsr_no_transpose) THEN
         trans_flag = dbcsr_transpose
      ENDIF
   END SUBROUTINE

   PURE FUNCTION array_eq_i8(arr1, arr2)
      INTEGER(KIND=int_8), DIMENSION(:), INTENT(IN)      :: arr1, arr2
      LOGICAL                                            :: array_eq_i8

      array_eq_i8 = .FALSE.
      IF (SIZE(arr1) .EQ. SIZE(arr2)) array_eq_i8 = ALL(arr1 == arr2)

   END FUNCTION

END MODULE
