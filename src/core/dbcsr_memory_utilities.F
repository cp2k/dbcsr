!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_memory_utilities
   !! Utility routines for the memory handling.

   USE dbcsr_kinds, ONLY: default_path_length, &
                          default_string_length, &
                          dp, int_8
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_memory_utilities'

   PUBLIC :: reallocate

   INTERFACE reallocate
      MODULE PROCEDURE reallocate_c1, reallocate_c2, reallocate_c3, reallocate_c4, &
         reallocate_i1, reallocate_i2, reallocate_i3, reallocate_i4, &
         reallocate_r1, reallocate_r2, reallocate_r3, reallocate_r4, &
         reallocate_r5, reallocate_s1, reallocate_l1, reallocate_8i1, &
         reallocate_8i2
   END INTERFACE

CONTAINS

#! Fypp macro for common subroutine body
#:def reallocate(rank)

#:for i in range(1, rank+1)
   INTEGER :: lb${i}$, lb${i}$_old, ub${i}$, ub${i}$_old
#:endfor

   IF (ASSOCIATED(p)) THEN

#:for i in range(1, rank+1)
      lb${i}$_old = LBOUND(p, ${i}$)
      ub${i}$_old = UBOUND(p, ${i}$)
      lb${i}$ = MAX(lb${i}$_new, lb${i}$_old)
      ub${i}$ = MIN(ub${i}$_new, ub${i}$_old)
#:endfor

#:set old_bounds = ','.join(['lb{0}:ub{0}'.format(i) for i in range(1, rank+1)])
#:set old_size  = '*'.join(['(ub{0}-lb{0}+1)'.format(i) for i in range(1, rank+1)])

      ALLOCATE (work(${old_bounds}$))
      work(${old_bounds}$) = p(${old_bounds}$)
      DEALLOCATE (p)
   END IF

#:set new_bounds = ','.join(['lb{0}_new:ub{0}_new'.format(i) for i in range(1, rank+1)])
#:set new_size  = '*'.join(['(ub{0}_new-lb{0}_new+1)'.format(i) for i in range(1, rank+1)])

   ALLOCATE (p(${new_bounds}$))
   p = zero

   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
      p(${old_bounds}$) = work(${old_bounds}$)
      DEALLOCATE (work)
   END IF

#:enddef

   SUBROUTINE reallocate_c1(p, lb1_new, ub1_new)
      !! (Re)Allocate a complex vector with a new dimension

      COMPLEX(KIND=dp), DIMENSION(:), POINTER            :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_c1', routineP = moduleN//':'//routineN
      COMPLEX(KIND=dp), PARAMETER                        :: zero = (0.0_dp, 0.0_dp)

      COMPLEX(KIND=dp), ALLOCATABLE, DIMENSION(:)        :: work

$: reallocate(rank=1)

   END SUBROUTINE reallocate_c1

   SUBROUTINE reallocate_c2(p, lb1_new, ub1_new, lb2_new, ub2_new)
      !! (Re)Allocate a complex array 2D with a new dimension

      COMPLEX(KIND=dp), DIMENSION(:, :), POINTER         :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_c2', routineP = moduleN//':'//routineN
      COMPLEX(KIND=dp), PARAMETER                        :: zero = (0.0_dp, 0.0_dp)

      COMPLEX(KIND=dp), ALLOCATABLE, DIMENSION(:, :)     :: work

$: reallocate(rank=2)

   END SUBROUTINE reallocate_c2

   SUBROUTINE reallocate_c3(p, lb1_new, ub1_new, lb2_new, ub2_new, lb3_new, ub3_new)
      !! (Re)Allocate a complex array 3D with a new dimension

      COMPLEX(KIND=dp), DIMENSION(:, :, :), POINTER      :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new, &
                                                            lb3_new, ub3_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_c3', routineP = moduleN//':'//routineN
      COMPLEX(KIND=dp), PARAMETER                        :: zero = (0.0_dp, 0.0_dp)

      COMPLEX(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :)  :: work

$: reallocate(rank=3)

   END SUBROUTINE reallocate_c3

   SUBROUTINE reallocate_c4(p, lb1_new, ub1_new, lb2_new, ub2_new, lb3_new, ub3_new, &
      !! (Re)Allocate a complex array 4D with a new dimension
                            lb4_new, ub4_new)

      COMPLEX(KIND=dp), DIMENSION(:, :, :, :), POINTER   :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new, &
                                                            lb3_new, ub3_new, lb4_new, ub4_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_c4', routineP = moduleN//':'//routineN
      COMPLEX(KIND=dp), PARAMETER                        :: zero = (0.0_dp, 0.0_dp)

      COMPLEX(KIND=dp), ALLOCATABLE, &
         DIMENSION(:, :, :, :)                           :: work

$: reallocate(rank=4)

   END SUBROUTINE reallocate_c4

   SUBROUTINE reallocate_i1(p, lb1_new, ub1_new)
      !! (Re)Allocate an integer vector with a new dimension.

      INTEGER, DIMENSION(:), POINTER                     :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_i1', routineP = moduleN//':'//routineN
      INTEGER, PARAMETER                                 :: zero = 0

      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: work

$: reallocate(rank=1)

   END SUBROUTINE reallocate_i1

   SUBROUTINE reallocate_i2(p, lb1_new, ub1_new, lb2_new, ub2_new)
      !! (Re)Allocate an integer array with a new dimension.

      INTEGER, DIMENSION(:, :), POINTER                  :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_i2', routineP = moduleN//':'//routineN
      INTEGER, PARAMETER                                 :: zero = 0

      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: work

$: reallocate(rank=2)

   END SUBROUTINE reallocate_i2

   SUBROUTINE reallocate_i3(p, lb1_new, ub1_new, lb2_new, ub2_new, lb3_new, ub3_new)
      !! (Re)Allocate an integer array 3D with a new dimension.

      INTEGER, DIMENSION(:, :, :), POINTER               :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new, &
                                                            lb3_new, ub3_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_i3', routineP = moduleN//':'//routineN
      INTEGER, PARAMETER                                 :: zero = 0

      INTEGER, ALLOCATABLE, DIMENSION(:, :, :)           :: work

$: reallocate(rank=3)

   END SUBROUTINE reallocate_i3

   SUBROUTINE reallocate_i4(p, lb1_new, ub1_new, lb2_new, ub2_new, lb3_new, ub3_new, &
      !! (Re)Allocate an integer array 3D with a new dimension.
                            lb4_new, ub4_new)

      INTEGER, DIMENSION(:, :, :, :), POINTER            :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new, &
                                                            lb3_new, ub3_new, lb4_new, ub4_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_i4', routineP = moduleN//':'//routineN
      INTEGER, PARAMETER                                 :: zero = 0

      INTEGER, ALLOCATABLE, DIMENSION(:, :, :, :)        :: work

$: reallocate(rank=4)

   END SUBROUTINE reallocate_i4

   SUBROUTINE reallocate_8i1(p, lb1_new, ub1_new)
      !! (Re)Allocate an integer (int_8) vector with a new dimension.

      INTEGER(KIND=int_8), DIMENSION(:), POINTER         :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_8i1', routineP = moduleN//':'//routineN
      INTEGER(KIND=int_8), PARAMETER                     :: zero = 0

      INTEGER(KIND=int_8), ALLOCATABLE, DIMENSION(:)     :: work

$: reallocate(rank=1)

   END SUBROUTINE reallocate_8i1

   SUBROUTINE reallocate_8i2(p, lb1_new, ub1_new, lb2_new, ub2_new)
      !! (Re)Allocate an integer (int_8) array with a new dimension.

      INTEGER(kind=int_8), DIMENSION(:, :), POINTER      :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_8i2', routineP = moduleN//':'//routineN
      INTEGER(KIND=int_8), PARAMETER                     :: zero = 0

      INTEGER(KIND=int_8), ALLOCATABLE, DIMENSION(:, :)  :: work

$: reallocate(rank=2)

   END SUBROUTINE reallocate_8i2

   SUBROUTINE reallocate_r1(p, lb1_new, ub1_new)
      !! (Re)Allocate an real vector with a new dimension.

      REAL(KIND=dp), DIMENSION(:), POINTER               :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_r1', routineP = moduleN//':'//routineN
      REAL(KIND=dp), PARAMETER                           :: zero = 0.0_dp

      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: work

$: reallocate(rank=1)

   END SUBROUTINE reallocate_r1

   SUBROUTINE reallocate_r2(p, lb1_new, ub1_new, lb2_new, ub2_new)
      !! (Re)Allocate an real array with new dimensions.

      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_r2', routineP = moduleN//':'//routineN
      REAL(KIND=dp), PARAMETER                           :: zero = 0.0_dp

      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: work

$: reallocate(rank=2)

   END SUBROUTINE reallocate_r2

   SUBROUTINE reallocate_r3(p, lb1_new, ub1_new, lb2_new, ub2_new, lb3_new, ub3_new)
      !! (Re)Allocate an real array 3D with new dimensions.

      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new, &
                                                            lb3_new, ub3_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_r3', routineP = moduleN//':'//routineN
      REAL(KIND=dp), PARAMETER                           :: zero = 0.0_dp

      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :)     :: work

$: reallocate(rank=3)

   END SUBROUTINE reallocate_r3

   SUBROUTINE reallocate_r4(p, lb1_new, ub1_new, lb2_new, ub2_new, lb3_new, ub3_new, &
      !! (Re)Allocate an real array 4D with new dimensions.
                            lb4_new, ub4_new)

      REAL(KIND=dp), DIMENSION(:, :, :, :), POINTER      :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new, &
                                                            lb3_new, ub3_new, lb4_new, ub4_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_r4', routineP = moduleN//':'//routineN
      REAL(KIND=dp), PARAMETER                           :: zero = 0.0_dp

      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :, :)  :: work

$: reallocate(rank=4)

   END SUBROUTINE reallocate_r4

   SUBROUTINE reallocate_r5(p, lb1_new, ub1_new, lb2_new, ub2_new, lb3_new, ub3_new, &
      !! (Re)Allocate an real array 5D with new dimensions.
                            lb4_new, ub4_new, lb5_new, ub5_new)

      REAL(KIND=dp), DIMENSION(:, :, :, :, :), POINTER   :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new, &
                                                            lb3_new, ub3_new, lb4_new, ub4_new, &
                                                            lb5_new, ub5_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_r5', routineP = moduleN//':'//routineN
      REAL(KIND=dp), PARAMETER                           :: zero = 0.0_dp

      REAL(KIND=dp), ALLOCATABLE, &
         DIMENSION(:, :, :, :, :)                        :: work

$: reallocate(rank=5)

   END SUBROUTINE reallocate_r5

   SUBROUTINE reallocate_l1(p, lb1_new, ub1_new)
      !! (Re)Allocate a logical vector with a new dimension.

      LOGICAL, DIMENSION(:), POINTER                     :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_l1', routineP = moduleN//':'//routineN
      LOGICAL, PARAMETER                                 :: zero = .FALSE.

      LOGICAL, ALLOCATABLE, DIMENSION(:)                 :: work

$: reallocate(rank=1)

   END SUBROUTINE reallocate_l1

   SUBROUTINE reallocate_s1(p_short, lb_new, ub_new, p_long)
      !! (Re)Allocate a vector of string variables
      !! @note
      !! (Maybe outdated) This routine does not work on SUN/Solaris!!!
      !! It should probably not be used. (Thomas Chassaing)

      CHARACTER(LEN=default_string_length), &
         DIMENSION(:), OPTIONAL, POINTER                 :: p_short
      INTEGER, INTENT(IN)                                :: lb_new, ub_new
      CHARACTER(LEN=default_path_length), DIMENSION(:), &
         OPTIONAL, POINTER                               :: p_long

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_s1', routineP = moduleN//':'//routineN

      IF (PRESENT(p_short)) THEN
         CALL reallocate_ss1(p_short, lb_new, ub_new)
      ELSE IF (PRESENT(p_long)) THEN
         CALL reallocate_ls1(p_long, lb_new, ub_new)
      ELSE
         DBCSR_ABORT("At least one of the two optional arguments is required")
      END IF

   END SUBROUTINE reallocate_s1

   SUBROUTINE reallocate_ss1(p, lb1_new, ub1_new)
      !! (Re)Allocate a vector of string variables of default string length.
      !! @note
      !! (Maybe outdated) This routine does not work on SUN/Solaris!!!
      !! It should probably not be used. (Thomas Chassaing)

      CHARACTER(LEN=default_string_length), &
         DIMENSION(:), POINTER                           :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_ss1', routineP = moduleN//':'//routineN
      CHARACTER(LEN=default_string_length), PARAMETER    :: zero = ""

      CHARACTER(LEN=default_string_length), &
         ALLOCATABLE, DIMENSION(:)                       :: work

$: reallocate(rank=1)

   END SUBROUTINE reallocate_ss1

   SUBROUTINE reallocate_ls1(p, lb1_new, ub1_new)
      !! (Re)Allocate a vector of string variables of default path length.
      !! @note
      !! (Maybe outdated) This routine does not work on SUN/Solaris!!!
      !! It should probably not be used. (Thomas Chassaing)

      CHARACTER(LEN=default_path_length), DIMENSION(:), &
         POINTER                                         :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_ls1', routineP = moduleN//':'//routineN
      CHARACTER(LEN=default_path_length), PARAMETER      :: zero = ""

      CHARACTER(LEN=default_path_length), ALLOCATABLE, &
         DIMENSION(:)                                    :: work

$: reallocate(rank=1)

   END SUBROUTINE reallocate_ls1

END MODULE dbcsr_memory_utilities
