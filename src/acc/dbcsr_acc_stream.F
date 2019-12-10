!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_acc_stream
   !! Accelerator support
#if defined (__DBCSR_ACC)
   USE ISO_C_BINDING, ONLY: C_INT, C_CHAR, C_PTR, C_NULL_PTR, C_NULL_CHAR, C_ASSOCIATED
#endif
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_acc_stream'

   PUBLIC :: acc_stream_type
   PUBLIC :: acc_stream_create, acc_stream_destroy
   PUBLIC :: acc_stream_synchronize
   PUBLIC :: acc_stream_priority_range
   PUBLIC :: acc_stream_equal, acc_stream_associated
   PUBLIC :: acc_stream_cptr

   TYPE acc_stream_type
      PRIVATE
#if defined (__DBCSR_ACC)
      TYPE(C_PTR) :: cptr = C_NULL_PTR
#else
      INTEGER :: dummy = 1
#endif
   END TYPE acc_stream_type

#if defined (__DBCSR_ACC)

   INTERFACE
      FUNCTION acc_interface_stream_create(stream_ptr, name, priority) RESULT(istat) BIND(C, name="acc_stream_create")
         IMPORT
         TYPE(C_PTR)                              :: stream_ptr
         CHARACTER(KIND=C_CHAR), DIMENSION(*)     :: name
         INTEGER(KIND=C_INT), VALUE               :: priority
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION acc_interface_stream_create
   END INTERFACE

   INTERFACE
      FUNCTION acc_interface_stream_priority_range(least, greatest) RESULT(istat) BIND(C, name="acc_stream_priority_range")
         IMPORT
         INTEGER(KIND=C_INT)                      :: least, greatest, istat

      END FUNCTION acc_interface_stream_priority_range
   END INTERFACE

   INTERFACE
      FUNCTION acc_interface_stream_destroy(stream_ptr) RESULT(istat) BIND(C, name="acc_stream_destroy")
         IMPORT
         TYPE(C_PTR), VALUE                       :: stream_ptr
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION acc_interface_stream_destroy
   END INTERFACE

   INTERFACE
      FUNCTION acc_interface_stream_sync(stream_ptr) RESULT(istat) BIND(C, name="acc_stream_sync")
         IMPORT
         TYPE(C_PTR), VALUE                       :: stream_ptr
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION acc_interface_stream_sync
   END INTERFACE

#endif
CONTAINS

#if ! defined (__DBCSR_ACC)
   FUNCTION acc_stream_cptr(this) RESULT(res)
      !! Returns C-pointer of given stream.

      INTEGER, INTENT(in)                                :: this
         !! stream ID
      LOGICAL                                            :: res
         !! false (accelerator support is not enabled)

      MARK_USED(this)
      res = .FALSE.
   END FUNCTION acc_stream_cptr

#else
   FUNCTION acc_stream_cptr(this) RESULT(res)
      !! Returns C-pointer of given stream.

      TYPE(acc_stream_type), INTENT(in)                  :: this
         !! stream ID
      TYPE(C_PTR)                                        :: res
         !! C-pointer of a given stream

      res = this%cptr
   END FUNCTION acc_stream_cptr
#endif

   SUBROUTINE acc_stream_create(this, name, priority)
      !! Fortran-wrapper for creation of a CUDA/HIP stream.

      TYPE(acc_stream_type), INTENT(OUT) :: this
         !! stream
      CHARACTER(LEN=*), INTENT(IN)             :: name
         !! stream name
      INTEGER, INTENT(IN), OPTIONAL            :: priority
         !! stream priority

#if ! defined (__DBCSR_ACC)
      MARK_USED(this)
      MARK_USED(name)
      MARK_USED(priority)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else
      INTEGER                                  :: istat, my_priority

      my_priority = -1
      IF (PRESENT(priority)) &
         my_priority = priority

      IF (C_ASSOCIATED(this%cptr)) &
         DBCSR_ABORT("acc_stream_create: stream already allocated")

      istat = acc_interface_stream_create(this%cptr, name//c_null_char, my_priority)

      IF (istat /= 0 .OR. .NOT. C_ASSOCIATED(this%cptr)) &
         DBCSR_ABORT("acc_stream_create failed")
#endif
   END SUBROUTINE acc_stream_create

   SUBROUTINE acc_stream_destroy(this)
      !! Fortran-wrapper for destruction of a CUDA/HIP stream.

      TYPE(acc_stream_type), &
         INTENT(INOUT)                          :: this
         !! stream

#if ! defined (__DBCSR_ACC)
      MARK_USED(this)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else
      INTEGER                                  :: istat
      IF (.NOT. C_ASSOCIATED(this%cptr)) &
         DBCSR_ABORT("acc_stream_destroy: stream not allocated")
      istat = acc_interface_stream_destroy(this%cptr)
      IF (istat /= 0) &
         DBCSR_ABORT("acc_stream_destroy failed")
      this%cptr = C_NULL_PTR
#endif
   END SUBROUTINE acc_stream_destroy

   SUBROUTINE acc_stream_synchronize(this)
      !! Fortran-wrapper for waiting for CUDA/HIP stream tasks to complete.

      TYPE(acc_stream_type), &
         INTENT(IN)                             :: this
         !! stream

#if ! defined (__DBCSR_ACC)
      MARK_USED(this)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else
      INTEGER                                  :: istat
      IF (.NOT. C_ASSOCIATED(this%cptr)) &
         DBCSR_ABORT("acc_stream_synchronize: stream not allocated")
      istat = acc_interface_stream_sync(this%cptr)
      IF (istat /= 0) &
         DBCSR_ABORT("acc_stream_synchronize failed")
#endif
   END SUBROUTINE acc_stream_synchronize

   SUBROUTINE acc_stream_priority_range(least, greatest)
      !! Fortran-wrapper for getting CUDA/HIP streams' priority range.

      INTEGER, INTENT(OUT)                     :: least, greatest
         !! least priority
         !! greatest priroity

#if ! defined (__DBCSR_ACC)
      least = -1; greatest = -1 ! assign intent-out arguments to silence compiler warnings
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else
      INTEGER                                  :: istat
      istat = acc_interface_stream_priority_range(least, greatest)
      IF (istat /= 0) &
         DBCSR_ABORT("acc_stream_priority_range failed")
#endif
   END SUBROUTINE acc_stream_priority_range

   FUNCTION acc_stream_equal(this, other) RESULT(res)
      !! Checks if two streams are equal

      TYPE(acc_stream_type), INTENT(IN) :: this, other
         !! first stream
         !! second stream
      LOGICAL                                  :: res
         !! true if equal, false otherwise
#if ! defined (__DBCSR_ACC)
      MARK_USED(this)
      MARK_USED(other)
      res = .TRUE.
#else
      res = C_ASSOCIATED(this%cptr, other%cptr)
#endif
   END FUNCTION acc_stream_equal

   FUNCTION acc_stream_associated(this) RESULT(res)
      !! Checks if a streams is associated

      TYPE(acc_stream_type), INTENT(IN) :: this
         !! stream
      LOGICAL                                  :: res
         !! true if associated, false otherwise
#if ! defined (__DBCSR_ACC)
      MARK_USED(this)
      res = .FALSE.
#else
      res = C_ASSOCIATED(this%cptr)
#endif
   END FUNCTION acc_stream_associated

END MODULE dbcsr_acc_stream
