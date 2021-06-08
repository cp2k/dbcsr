!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! Copyright(C) 2021 Advanced Micro Devices, Inc. All rights reserved.                              !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_hip_profiling

   INTERFACE
      SUBROUTINE roctxMarkA(message) BIND(c, name="roctxMarkA")
         USE ISO_C_BINDING, ONLY: C_CHAR
         IMPLICIT NONE
         CHARACTER(C_CHAR) :: message(*)
      END SUBROUTINE roctxMarkA

      FUNCTION roctxRangePushA(message) BIND(c, name="roctxRangePushA")
         USE ISO_C_BINDING, ONLY: C_INT, &
                                  C_CHAR
         IMPLICIT NONE
         INTEGER(C_INT) :: roctxRangePushA
         CHARACTER(C_CHAR) :: message(*)
      END FUNCTION roctxRangePushA

      SUBROUTINE roctxRangePop() BIND(c, name="roctxRangePop")
         IMPLICIT NONE
      END SUBROUTINE roctxRangePop
   END INTERFACE

END MODULE dbcsr_hip_profiling

