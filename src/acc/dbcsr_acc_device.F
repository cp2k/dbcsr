!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_acc_device
#if defined (__DBCSR_ACC)
   USE ISO_C_BINDING, ONLY: C_INT
#endif
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_level

   IMPLICIT NONE

   PUBLIC :: dbcsr_acc_get_ndevices, dbcsr_acc_set_active_device, dbcsr_acc_clear_errors
   PUBLIC :: acc_device_synchronize

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_acc_device'

#if defined (__DBCSR_ACC)
   INTERFACE
      FUNCTION acc_get_ndevices_cu(n_devices) RESULT(istat) &
         BIND(C, name="c_dbcsr_acc_get_ndevices")
         IMPORT
         INTEGER(KIND=C_INT), INTENT(OUT)         :: n_devices
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION acc_get_ndevices_cu

      FUNCTION acc_set_active_device_cu(device_id) RESULT(istat) &
         BIND(C, name="c_dbcsr_acc_set_active_device")
         IMPORT
         INTEGER(KIND=C_INT), INTENT(IN), VALUE   :: device_id
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION acc_set_active_device_cu

      FUNCTION acc_device_synchronize_cu() RESULT(istat) &
         BIND(C, name="c_dbcsr_acc_device_synchronize")
         IMPORT
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION acc_device_synchronize_cu

      SUBROUTINE acc_clear_errors_cu() &
         BIND(C, name="c_dbcsr_acc_clear_errors")
      END SUBROUTINE acc_clear_errors_cu

   END INTERFACE

#endif

CONTAINS

   FUNCTION dbcsr_acc_get_ndevices() RESULT(n)
      !! Get number of accelerator devices

      INTEGER                                  :: n
         !! number of accelerator devices

#if defined (__DBCSR_ACC)
      INTEGER                                  :: istat
#endif

      n = 0
#if defined (__DBCSR_ACC)
      istat = acc_get_ndevices_cu(n)
      IF (istat /= 0) &
         DBCSR_ABORT("dbcsr_acc_get_ndevices: failed")
#endif
   END FUNCTION dbcsr_acc_get_ndevices

   SUBROUTINE dbcsr_acc_set_active_device(device_id)
      !! Set active accelerator device

      INTEGER :: device_id

#if defined (__DBCSR_ACC)
      INTEGER :: istat

!$    IF (0 == omp_get_level()) THEN
         istat = 0
!$OMP    PARALLEL DEFAULT(NONE) SHARED(device_id) REDUCTION(MAX:istat)
         istat = acc_set_active_device_cu(device_id)
!$OMP    END PARALLEL
!$    ELSE
         istat = acc_set_active_device_cu(device_id)
!$    END IF
      IF (istat /= 0) &
         DBCSR_ABORT("dbcsr_acc_set_active_device: failed")

#else
      MARK_USED(device_id)
      DBCSR_ABORT("__DBCSR_ACC not compiled in")
#endif
   END SUBROUTINE dbcsr_acc_set_active_device

   SUBROUTINE dbcsr_acc_clear_errors()
      !! Clear GPU errors
#if defined (__DBCSR_ACC)
      CALL acc_clear_errors_cu()
#else
      DBCSR_ABORT("__DBCSR_ACC not compiled in")
#endif
   END SUBROUTINE dbcsr_acc_clear_errors

   SUBROUTINE acc_device_synchronize()
      !! Fortran-wrapper for waiting for work on all streams to complete

#if defined (__DBCSR_ACC)
      INTEGER                                  :: istat
      istat = acc_device_synchronize_cu()
      IF (istat /= 0) &
         DBCSR_ABORT("acc_device_synchronize failed")
#else
      DBCSR_ABORT("__DBCSR_ACC not compiled in")
#endif
   END SUBROUTINE acc_device_synchronize

END MODULE dbcsr_acc_device
