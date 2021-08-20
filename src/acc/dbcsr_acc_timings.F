!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_acc_timings
   !! Accelerator support
   USE ISO_C_BINDING, ONLY: C_INT, C_PTR, C_F_POINTER
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_acc_timings'

CONTAINS

   SUBROUTINE dbcsr_timeset_F(routineN, routineN_len, handle) BIND(C, name="c_dbcsr_timeset")

      TYPE(C_PTR), INTENT(IN)      :: routineN
      INTEGER(KIND=C_INT), INTENT(IN)        :: routineN_len
      INTEGER(KIND=C_INT), INTENT(OUT)       :: handle

      CHARACTER, POINTER                     :: a(:)
      CHARACTER(len=routineN_len)            :: routineName
      INTEGER                                :: i

      CALL C_F_POINTER(routineN, a, [routineN_len])

      ! Convert character array "a" to scalar character string
      ! "routineName"
      DO i = 1, routineN_len
         routineName(i:i) = a(i)
      END DO

#if defined (__DBCSR_ACC)
      CALL timeset(routineName, handle)
#else
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
      handle = 0
#endif

   END SUBROUTINE dbcsr_timeset_F

   SUBROUTINE dbcsr_timestop_F(handle) BIND(C, name="c_dbcsr_timestop")

      INTEGER(KIND=C_INT), INTENT(IN)        :: handle

#if ! defined (__DBCSR_ACC)
      MARK_USED(handle)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else
      CALL timestop(handle)
#endif
   END SUBROUTINE dbcsr_timestop_F

END MODULE dbcsr_acc_timings
