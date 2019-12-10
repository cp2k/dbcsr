!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_hipblas
   USE ISO_C_BINDING, ONLY: C_CHAR, &
                            C_DOUBLE, &
                            C_INT, &
                            C_NULL_PTR, &
                            C_PTR
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_hipblas'

   PUBLIC :: hipblas_handle_init
   PUBLIC :: hipblas_handle_destroy
   PUBLIC :: hipblas_handle_type

#if (__DBCSR_ACC == 3)
   PUBLIC :: hipblas_dgemm_hip
#endif

   TYPE hipblas_handle_type
      TYPE(C_PTR) :: handle_ptr = c_null_ptr
   END TYPE hipblas_handle_type

#if (__DBCSR_ACC == 3)
   INTERFACE
      FUNCTION hipblas_create_hip(handle) &
         RESULT(istat) &
         BIND(C, name="hipblas_create")
         IMPORT
         TYPE(C_PTR), INTENT(IN)                   :: handle
         INTEGER(KIND=C_INT)                       :: istat
      END FUNCTION hipblas_create_hip

      FUNCTION hipblas_destroy_hip(handle) &
         RESULT(istat) &
         BIND(C, name="hipblas_destroy")
         IMPORT
         TYPE(C_PTR), INTENT(IN), VALUE            :: handle
         INTEGER(KIND=C_INT)                       :: istat
      END FUNCTION hipblas_destroy_hip

      FUNCTION hipblas_dgemm_hip(handle, transa, transb, &
                                 m, n, k, a_offset, b_offset, c_offset, &
                                 a_data, b_data, c_data, alpha, beta, stream) &
         RESULT(istat) &
         BIND(C, name="hipblas_dgemm")
         IMPORT
         TYPE(C_PTR), INTENT(IN), VALUE            :: handle
         CHARACTER(KIND=C_CHAR), INTENT(IN), VALUE :: transa, transb
         INTEGER(KIND=C_INT), INTENT(IN), VALUE    :: m, n, k, a_offset, b_offset, c_offset
         TYPE(C_PTR), INTENT(IN), VALUE            :: a_data, b_data, c_data
         REAL(KIND=C_DOUBLE), INTENT(IN), VALUE    :: alpha, beta
         TYPE(C_PTR), INTENT(IN), VALUE            :: stream
         INTEGER(KIND=C_INT)                       :: istat
      END FUNCTION hipblas_dgemm_hip

   END INTERFACE
#endif

CONTAINS

   SUBROUTINE hipblas_handle_init(hipblas_handle)
      TYPE(hipblas_handle_type), INTENT(OUT) :: hipblas_handle

      CHARACTER(len=*), PARAMETER :: routineN = 'hipblas_handle_init', &
                                     routineP = moduleN//':'//routineN

#if (__DBCSR_ACC == 3)
      INTEGER(KIND=C_INT)      :: err
      INTEGER                  :: handle

      CALL timeset(routineN, handle)
      err = hipblas_create_hip(hipblas_handle%handle_ptr)
      IF (err /= 0) &
         DBCSR_ABORT("hipblas cannot be created")
      CALL timestop(handle)
#else
      MARK_USED(hipblas_handle)
#endif
   END SUBROUTINE

   SUBROUTINE hipblas_handle_destroy(hipblas_handle)
      TYPE(hipblas_handle_type), INTENT(INOUT) :: hipblas_handle

      CHARACTER(len=*), PARAMETER :: routineN = 'hipblas_handle_destroy', &
                                     routineP = moduleN//':'//routineN

#if (__DBCSR_ACC == 3)
      INTEGER(KIND=C_INT)      :: err
      INTEGER                  :: handle

      CALL timeset(routineN, handle)
      err = hipblas_destroy_hip(hipblas_handle%handle_ptr)
      IF (err /= 0) &
         DBCSR_ABORT("hipblas cannot be destroyed")
      hipblas_handle%handle_ptr = c_null_ptr
      CALL timestop(handle)
#else
      MARK_USED(hipblas_handle)
#endif
   END SUBROUTINE

END MODULE dbcsr_hipblas
