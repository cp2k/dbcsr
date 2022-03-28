!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_acc_init
   !! Accelerator support
#if defined (__DBCSR_ACC)
   USE ISO_C_BINDING, ONLY: C_INT, C_CHAR, C_PTR, C_NULL_PTR, C_NULL_CHAR, C_ASSOCIATED
#endif
   USE dbcsr_acc_device, ONLY: dbcsr_acc_set_active_device
   USE dbcsr_config, ONLY: get_accdrv_active_device_id
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_acc_init'

   PUBLIC :: acc_init, acc_finalize

#if defined (__DBCSR_ACC)

   INTERFACE
      FUNCTION acc_interface_drv_init() RESULT(istat) BIND(C, name="c_dbcsr_acc_init")
         IMPORT
         INTEGER(KIND=C_INT)                      :: istat
      END FUNCTION acc_interface_drv_init
   END INTERFACE

   INTERFACE
      FUNCTION acc_interface_drv_finalize() RESULT(istat) BIND(C, name="c_dbcsr_acc_finalize")
         IMPORT
         INTEGER(KIND=C_INT)                      :: istat
      END FUNCTION acc_interface_drv_finalize
   END INTERFACE

#endif
CONTAINS

   SUBROUTINE acc_init()

#if ! defined (__DBCSR_ACC)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else
      INTEGER                                  :: istat
      ! Set active device first
      CALL dbcsr_acc_set_active_device(get_accdrv_active_device_id())
!$OMP    PARALLEL DEFAULT(NONE) PRIVATE(istat)
!$OMP    MASTER
      istat = acc_interface_drv_init()
      IF (istat /= 0) &
         DBCSR_ABORT("acc_init failed")
!$OMP    END MASTER
!$OMP    END PARALLEL
#endif
   END SUBROUTINE acc_init

   SUBROUTINE acc_finalize()

#if ! defined (__DBCSR_ACC)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else
      INTEGER                                :: istat
!$OMP    PARALLEL DEFAULT(NONE) PRIVATE(istat)
!$OMP    MASTER
      istat = acc_interface_drv_finalize()
      IF (istat /= 0) &
         DBCSR_ABORT("acc_finalize failed")
!$OMP    END MASTER
!$OMP    END PARALLEL
#endif
   END SUBROUTINE acc_finalize

END MODULE dbcsr_acc_init
