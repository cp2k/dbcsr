# 1 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F" 1
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

# 37 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
# 72 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

   SUBROUTINE reallocate_c1(p, lb1_new, ub1_new)
      !! (Re)Allocate a complex vector with a new dimension

      COMPLEX(KIND=dp), DIMENSION(:), POINTER            :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_c1', routineP = moduleN//':'//routineN
      COMPLEX(KIND=dp), PARAMETER                        :: zero = (0.0_dp, 0.0_dp)

      COMPLEX(KIND=dp), ALLOCATABLE, DIMENSION(:)        :: work


# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1))
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1) = p(lb1:ub1)
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new))
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1) = work(lb1:ub1)
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


   END SUBROUTINE reallocate_c1

   SUBROUTINE reallocate_c2(p, lb1_new, ub1_new, lb2_new, ub2_new)
      !! (Re)Allocate a complex array 2D with a new dimension

      COMPLEX(KIND=dp), DIMENSION(:, :), POINTER         :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_c2', routineP = moduleN//':'//routineN
      COMPLEX(KIND=dp), PARAMETER                        :: zero = (0.0_dp, 0.0_dp)

      COMPLEX(KIND=dp), ALLOCATABLE, DIMENSION(:, :)     :: work


# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb2, lb2_old, ub2, ub2_old
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2_old = LBOUND(p, 2)
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2_old = UBOUND(p, 2)
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2 = MAX(lb2_new, lb2_old)
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2 = MIN(ub2_new, ub2_old)
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1,lb2:ub2))
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1,lb2:ub2) = p(lb1:ub1,lb2:ub2)
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new,lb2_new:ub2_new))
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1,lb2:ub2) = work(lb1:ub1,lb2:ub2)
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


   END SUBROUTINE reallocate_c2

   SUBROUTINE reallocate_c3(p, lb1_new, ub1_new, lb2_new, ub2_new, lb3_new, ub3_new)
      !! (Re)Allocate a complex array 3D with a new dimension

      COMPLEX(KIND=dp), DIMENSION(:, :, :), POINTER      :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new, &
                                                            lb3_new, ub3_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_c3', routineP = moduleN//':'//routineN
      COMPLEX(KIND=dp), PARAMETER                        :: zero = (0.0_dp, 0.0_dp)

      COMPLEX(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :)  :: work


# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb2, lb2_old, ub2, ub2_old
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb3, lb3_old, ub3, ub3_old
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2_old = LBOUND(p, 2)
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2_old = UBOUND(p, 2)
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2 = MAX(lb2_new, lb2_old)
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2 = MIN(ub2_new, ub2_old)
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb3_old = LBOUND(p, 3)
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub3_old = UBOUND(p, 3)
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb3 = MAX(lb3_new, lb3_old)
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub3 = MIN(ub3_new, ub3_old)
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1,lb2:ub2,lb3:ub3))
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1,lb2:ub2,lb3:ub3) = p(lb1:ub1,lb2:ub2,lb3:ub3)
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new,lb2_new:ub2_new,lb3_new:ub3_new))
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1,lb2:ub2,lb3:ub3) = work(lb1:ub1,lb2:ub2,lb3:ub3)
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


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


# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb2, lb2_old, ub2, ub2_old
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb3, lb3_old, ub3, ub3_old
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb4, lb4_old, ub4, ub4_old
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2_old = LBOUND(p, 2)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2_old = UBOUND(p, 2)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2 = MAX(lb2_new, lb2_old)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2 = MIN(ub2_new, ub2_old)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb3_old = LBOUND(p, 3)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub3_old = UBOUND(p, 3)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb3 = MAX(lb3_new, lb3_old)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub3 = MIN(ub3_new, ub3_old)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb4_old = LBOUND(p, 4)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub4_old = UBOUND(p, 4)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb4 = MAX(lb4_new, lb4_old)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub4 = MIN(ub4_new, ub4_old)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4))
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4) = p(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new,lb2_new:ub2_new,lb3_new:ub3_new,lb4_new:ub4_new))
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4) = work(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 133 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


   END SUBROUTINE reallocate_c4

   SUBROUTINE reallocate_i1(p, lb1_new, ub1_new)
      !! (Re)Allocate an integer vector with a new dimension.

      INTEGER, DIMENSION(:), POINTER                     :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_i1', routineP = moduleN//':'//routineN
      INTEGER, PARAMETER                                 :: zero = 0

      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: work


# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1))
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1) = p(lb1:ub1)
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new))
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1) = work(lb1:ub1)
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


   END SUBROUTINE reallocate_i1

   SUBROUTINE reallocate_i2(p, lb1_new, ub1_new, lb2_new, ub2_new)
      !! (Re)Allocate an integer array with a new dimension.

      INTEGER, DIMENSION(:, :), POINTER                  :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_i2', routineP = moduleN//':'//routineN
      INTEGER, PARAMETER                                 :: zero = 0

      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: work


# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb2, lb2_old, ub2, ub2_old
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2_old = LBOUND(p, 2)
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2_old = UBOUND(p, 2)
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2 = MAX(lb2_new, lb2_old)
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2 = MIN(ub2_new, ub2_old)
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1,lb2:ub2))
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1,lb2:ub2) = p(lb1:ub1,lb2:ub2)
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new,lb2_new:ub2_new))
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1,lb2:ub2) = work(lb1:ub1,lb2:ub2)
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 163 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


   END SUBROUTINE reallocate_i2

   SUBROUTINE reallocate_i3(p, lb1_new, ub1_new, lb2_new, ub2_new, lb3_new, ub3_new)
      !! (Re)Allocate an integer array 3D with a new dimension.

      INTEGER, DIMENSION(:, :, :), POINTER               :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new, &
                                                            lb3_new, ub3_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_i3', routineP = moduleN//':'//routineN
      INTEGER, PARAMETER                                 :: zero = 0

      INTEGER, ALLOCATABLE, DIMENSION(:, :, :)           :: work


# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb2, lb2_old, ub2, ub2_old
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb3, lb3_old, ub3, ub3_old
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2_old = LBOUND(p, 2)
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2_old = UBOUND(p, 2)
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2 = MAX(lb2_new, lb2_old)
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2 = MIN(ub2_new, ub2_old)
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb3_old = LBOUND(p, 3)
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub3_old = UBOUND(p, 3)
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb3 = MAX(lb3_new, lb3_old)
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub3 = MIN(ub3_new, ub3_old)
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1,lb2:ub2,lb3:ub3))
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1,lb2:ub2,lb3:ub3) = p(lb1:ub1,lb2:ub2,lb3:ub3)
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new,lb2_new:ub2_new,lb3_new:ub3_new))
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1,lb2:ub2,lb3:ub3) = work(lb1:ub1,lb2:ub2,lb3:ub3)
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


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


# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb2, lb2_old, ub2, ub2_old
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb3, lb3_old, ub3, ub3_old
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb4, lb4_old, ub4, ub4_old
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2_old = LBOUND(p, 2)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2_old = UBOUND(p, 2)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2 = MAX(lb2_new, lb2_old)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2 = MIN(ub2_new, ub2_old)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb3_old = LBOUND(p, 3)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub3_old = UBOUND(p, 3)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb3 = MAX(lb3_new, lb3_old)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub3 = MIN(ub3_new, ub3_old)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb4_old = LBOUND(p, 4)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub4_old = UBOUND(p, 4)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb4 = MAX(lb4_new, lb4_old)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub4 = MIN(ub4_new, ub4_old)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4))
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4) = p(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new,lb2_new:ub2_new,lb3_new:ub3_new,lb4_new:ub4_new))
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4) = work(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 196 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


   END SUBROUTINE reallocate_i4

   SUBROUTINE reallocate_8i1(p, lb1_new, ub1_new)
      !! (Re)Allocate an integer (int_8) vector with a new dimension.

      INTEGER(KIND=int_8), DIMENSION(:), POINTER         :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_8i1', routineP = moduleN//':'//routineN
      INTEGER(KIND=int_8), PARAMETER                     :: zero = 0

      INTEGER(KIND=int_8), ALLOCATABLE, DIMENSION(:)     :: work


# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1))
# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1) = p(lb1:ub1)
# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new))
# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1) = work(lb1:ub1)
# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 211 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


   END SUBROUTINE reallocate_8i1

   SUBROUTINE reallocate_8i2(p, lb1_new, ub1_new, lb2_new, ub2_new)
      !! (Re)Allocate an integer (int_8) array with a new dimension.

      INTEGER(kind=int_8), DIMENSION(:, :), POINTER      :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_8i2', routineP = moduleN//':'//routineN
      INTEGER(KIND=int_8), PARAMETER                     :: zero = 0

      INTEGER(KIND=int_8), ALLOCATABLE, DIMENSION(:, :)  :: work


# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb2, lb2_old, ub2, ub2_old
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2_old = LBOUND(p, 2)
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2_old = UBOUND(p, 2)
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2 = MAX(lb2_new, lb2_old)
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2 = MIN(ub2_new, ub2_old)
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1,lb2:ub2))
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1,lb2:ub2) = p(lb1:ub1,lb2:ub2)
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new,lb2_new:ub2_new))
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1,lb2:ub2) = work(lb1:ub1,lb2:ub2)
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 226 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


   END SUBROUTINE reallocate_8i2

   SUBROUTINE reallocate_r1(p, lb1_new, ub1_new)
      !! (Re)Allocate an real vector with a new dimension.

      REAL(KIND=dp), DIMENSION(:), POINTER               :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_r1', routineP = moduleN//':'//routineN
      REAL(KIND=dp), PARAMETER                           :: zero = 0.0_dp

      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: work


# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1))
# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1) = p(lb1:ub1)
# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new))
# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1) = work(lb1:ub1)
# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 241 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


   END SUBROUTINE reallocate_r1

   SUBROUTINE reallocate_r2(p, lb1_new, ub1_new, lb2_new, ub2_new)
      !! (Re)Allocate an real array with new dimensions.

      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_r2', routineP = moduleN//':'//routineN
      REAL(KIND=dp), PARAMETER                           :: zero = 0.0_dp

      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: work


# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb2, lb2_old, ub2, ub2_old
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2_old = LBOUND(p, 2)
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2_old = UBOUND(p, 2)
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2 = MAX(lb2_new, lb2_old)
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2 = MIN(ub2_new, ub2_old)
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1,lb2:ub2))
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1,lb2:ub2) = p(lb1:ub1,lb2:ub2)
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new,lb2_new:ub2_new))
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1,lb2:ub2) = work(lb1:ub1,lb2:ub2)
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 256 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


   END SUBROUTINE reallocate_r2

   SUBROUTINE reallocate_r3(p, lb1_new, ub1_new, lb2_new, ub2_new, lb3_new, ub3_new)
      !! (Re)Allocate an real array 3D with new dimensions.

      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new, lb2_new, ub2_new, &
                                                            lb3_new, ub3_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_r3', routineP = moduleN//':'//routineN
      REAL(KIND=dp), PARAMETER                           :: zero = 0.0_dp

      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :)     :: work


# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb2, lb2_old, ub2, ub2_old
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb3, lb3_old, ub3, ub3_old
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2_old = LBOUND(p, 2)
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2_old = UBOUND(p, 2)
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2 = MAX(lb2_new, lb2_old)
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2 = MIN(ub2_new, ub2_old)
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb3_old = LBOUND(p, 3)
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub3_old = UBOUND(p, 3)
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb3 = MAX(lb3_new, lb3_old)
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub3 = MIN(ub3_new, ub3_old)
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1,lb2:ub2,lb3:ub3))
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1,lb2:ub2,lb3:ub3) = p(lb1:ub1,lb2:ub2,lb3:ub3)
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new,lb2_new:ub2_new,lb3_new:ub3_new))
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1,lb2:ub2,lb3:ub3) = work(lb1:ub1,lb2:ub2,lb3:ub3)
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 272 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


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


# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb2, lb2_old, ub2, ub2_old
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb3, lb3_old, ub3, ub3_old
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb4, lb4_old, ub4, ub4_old
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2_old = LBOUND(p, 2)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2_old = UBOUND(p, 2)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2 = MAX(lb2_new, lb2_old)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2 = MIN(ub2_new, ub2_old)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb3_old = LBOUND(p, 3)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub3_old = UBOUND(p, 3)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb3 = MAX(lb3_new, lb3_old)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub3 = MIN(ub3_new, ub3_old)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb4_old = LBOUND(p, 4)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub4_old = UBOUND(p, 4)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb4 = MAX(lb4_new, lb4_old)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub4 = MIN(ub4_new, ub4_old)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4))
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4) = p(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new,lb2_new:ub2_new,lb3_new:ub3_new,lb4_new:ub4_new))
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4) = work(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 289 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


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


# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb2, lb2_old, ub2, ub2_old
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb3, lb3_old, ub3, ub3_old
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb4, lb4_old, ub4, ub4_old
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb5, lb5_old, ub5, ub5_old
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2_old = LBOUND(p, 2)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2_old = UBOUND(p, 2)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb2 = MAX(lb2_new, lb2_old)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub2 = MIN(ub2_new, ub2_old)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb3_old = LBOUND(p, 3)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub3_old = UBOUND(p, 3)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb3 = MAX(lb3_new, lb3_old)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub3 = MIN(ub3_new, ub3_old)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb4_old = LBOUND(p, 4)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub4_old = UBOUND(p, 4)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb4 = MAX(lb4_new, lb4_old)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub4 = MIN(ub4_new, ub4_old)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb5_old = LBOUND(p, 5)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub5_old = UBOUND(p, 5)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb5 = MAX(lb5_new, lb5_old)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub5 = MIN(ub5_new, ub5_old)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4,lb5:ub5))
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4,lb5:ub5) = p(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4,lb5:ub5)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new,lb2_new:ub2_new,lb3_new:ub3_new,lb4_new:ub4_new,lb5_new:ub5_new))
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4,lb5:ub5) = work(lb1:ub1,lb2:ub2,lb3:ub3,lb4:ub4,lb5:ub5)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


   END SUBROUTINE reallocate_r5

   SUBROUTINE reallocate_l1(p, lb1_new, ub1_new)
      !! (Re)Allocate a logical vector with a new dimension.

      LOGICAL, DIMENSION(:), POINTER                     :: p
      INTEGER, INTENT(IN)                                :: lb1_new, ub1_new

      CHARACTER(LEN=*), PARAMETER :: routineN = 'reallocate_l1', routineP = moduleN//':'//routineN
      LOGICAL, PARAMETER                                 :: zero = .FALSE.

      LOGICAL, ALLOCATABLE, DIMENSION(:)                 :: work


# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1))
# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1) = p(lb1:ub1)
# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new))
# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1) = work(lb1:ub1)
# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 323 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


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


# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1))
# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1) = p(lb1:ub1)
# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new))
# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1) = work(lb1:ub1)
# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 367 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


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


# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   INTEGER :: lb1, lb1_old, ub1, ub1_old
# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p)) THEN
# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1_old = LBOUND(p, 1)
# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1_old = UBOUND(p, 1)
# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      lb1 = MAX(lb1_new, lb1_old)
# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ub1 = MIN(ub1_new, ub1_old)
# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      ALLOCATE (work(lb1:ub1))
# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      work(lb1:ub1) = p(lb1:ub1)
# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (p)
# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   ALLOCATE (p(lb1_new:ub1_new))
# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   p = zero
# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"

# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   IF (ASSOCIATED(p) .AND. ALLOCATED(work)) THEN
# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      p(lb1:ub1) = work(lb1:ub1)
# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
      DEALLOCATE (work)
# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"
   END IF
# 387 "/__w/dbcsr/dbcsr/src/core/dbcsr_memory_utilities.F"


   END SUBROUTINE reallocate_ls1

END MODULE dbcsr_memory_utilities
