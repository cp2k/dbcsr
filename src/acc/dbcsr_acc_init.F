!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright (C) 2000 - 2018  CP2K developers group                                               !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief   Accelerator support
!> \author  Ole Schuett, Shoshana Jakobovits
!> \date    2018
! **************************************************************************************************
MODULE dbcsr_acc_init
#if defined (__ACC)
   USE ISO_C_BINDING, ONLY: C_INT, C_CHAR, C_PTR, C_NULL_PTR, C_NULL_CHAR, C_ASSOCIATED
#endif
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_acc_init'

   PUBLIC :: acc_init

#if defined (__ACC)

   INTERFACE
      FUNCTION cuda_drv_init() RESULT(istat) BIND(C, name="acc_init")
         IMPORT
         INTEGER(KIND=C_INT)                      :: istat
      END FUNCTION cuda_drv_init
   END INTERFACE

#endif
CONTAINS

! **************************************************************************************************
!> \brief Fortran-wrapper for acc_init
!> \author  Shoshana Jakobovits
! **************************************************************************************************
   SUBROUTINE acc_init()

#if ! defined (__ACC)
      DBCSR_ABORT("__ACC not compiled in.")
#else
      INTEGER                                  :: istat
      istat = cuda_drv_init()
      IF (istat /= 0) &
         DBCSR_ABORT("acc_drv_init failed")
#endif
   END SUBROUTINE acc_init

END MODULE dbcsr_acc_init
