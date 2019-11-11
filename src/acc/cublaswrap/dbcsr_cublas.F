!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_cublas
   USE ISO_C_BINDING, ONLY: C_CHAR, &
                            C_DOUBLE, &
                            C_INT, &
                            C_NULL_PTR, &
                            C_PTR
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_cublas'

   PUBLIC :: cublas_handle_init
   PUBLIC :: cublas_handle_destroy
   PUBLIC :: cublas_handle_type

#if (__DBCSR_ACC == 2)
   PUBLIC :: cublas_dgemm_cu
#endif

   TYPE cublas_handle_type
      TYPE(C_PTR) :: handle_ptr = c_null_ptr
   END TYPE cublas_handle_type

#if (__DBCSR_ACC == 2)
   INTERFACE
      FUNCTION cublas_create_cu(handle) &
         RESULT(istat) &
         BIND(C, name="cublas_create")
         IMPORT
         TYPE(C_PTR), INTENT(IN)                   :: handle
         INTEGER(KIND=C_INT)                       :: istat
      END FUNCTION cublas_create_cu

      FUNCTION cublas_destroy_cu(handle) &
         RESULT(istat) &
         BIND(C, name="cublas_destroy")
         IMPORT
         TYPE(C_PTR), INTENT(IN), VALUE            :: handle
         INTEGER(KIND=C_INT)                       :: istat
      END FUNCTION cublas_destroy_cu

      FUNCTION cublas_dgemm_cu(handle, transa, transb, &
                               m, n, k, a_offset, b_offset, c_offset, &
                               a_data, b_data, c_data, alpha, beta, stream) &
         RESULT(istat) &
         BIND(C, name="cublas_dgemm")
         IMPORT
         TYPE(C_PTR), INTENT(IN), VALUE            :: handle
         CHARACTER(KIND=C_CHAR), INTENT(IN), VALUE :: transa, transb
         INTEGER(KIND=C_INT), INTENT(IN), VALUE    :: m, n, k, a_offset, b_offset, c_offset
         TYPE(C_PTR), INTENT(IN), VALUE            :: a_data, b_data, c_data
         REAL(KIND=C_DOUBLE), INTENT(IN), VALUE    :: alpha, beta
         TYPE(C_PTR), INTENT(IN), VALUE            :: stream
         INTEGER(KIND=C_INT)                       :: istat
      END FUNCTION cublas_dgemm_cu

   END INTERFACE
#endif

CONTAINS

   SUBROUTINE cublas_handle_init(cublas_handle)
      TYPE(cublas_handle_type), INTENT(OUT) :: cublas_handle

      CHARACTER(len=*), PARAMETER :: routineN = 'cublas_handle_init', &
                                     routineP = moduleN//':'//routineN

#if (__DBCSR_ACC == 2)
      INTEGER(KIND=C_INT)      :: err
      INTEGER                  :: handle

      CALL timeset(routineN, handle)
      err = cublas_create_cu(cublas_handle%handle_ptr)
      IF (err /= 0) &
         DBCSR_ABORT("cublas cannot be created")
      CALL timestop(handle)
#else
      MARK_USED(cublas_handle)
#endif
   END SUBROUTINE

   SUBROUTINE cublas_handle_destroy(cublas_handle)
      TYPE(cublas_handle_type), INTENT(INOUT) :: cublas_handle

      CHARACTER(len=*), PARAMETER :: routineN = 'cublas_handle_destroy', &
                                     routineP = moduleN//':'//routineN

#if (__DBCSR_ACC == 2)
      INTEGER(KIND=C_INT)      :: err
      INTEGER                  :: handle

      CALL timeset(routineN, handle)
      err = cublas_destroy_cu(cublas_handle%handle_ptr)
      IF (err /= 0) &
         DBCSR_ABORT("cublas cannot be destroyed")
      cublas_handle%handle_ptr = c_null_ptr
      CALL timestop(handle)
#else
      MARK_USED(cublas_handle)
#endif
   END SUBROUTINE

END MODULE dbcsr_cublas
