!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_acc_hostmem
   !! Accelerator support
#if defined (__DBCSR_ACC)
   USE ISO_C_BINDING, ONLY: C_INT, C_SIZE_T, C_PTR, C_LOC, C_F_POINTER
#endif
   USE dbcsr_kinds, ONLY: int_4, &
                          int_4_size, &
                          int_8, &
                          int_8_size, &
                          real_4, &
                          real_4_size, &
                          real_8, &
                          real_8_size
   USE dbcsr_acc_stream, ONLY: acc_stream_associated, &
                               acc_stream_cptr, &
                               acc_stream_type
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_acc_hostmem'

   LOGICAL, PARAMETER :: careful_mod = .TRUE.

   PUBLIC :: acc_hostmem_allocate, acc_hostmem_deallocate

   INTERFACE acc_hostmem_allocate
      MODULE PROCEDURE acc_hostmem_alloc_i4, acc_hostmem_alloc_i8
      MODULE PROCEDURE acc_hostmem_alloc_r4, acc_hostmem_alloc_r8
      MODULE PROCEDURE acc_hostmem_alloc_c4, acc_hostmem_alloc_c8
      MODULE PROCEDURE acc_hostmem_alloc_i4_2D, acc_hostmem_alloc_i8_2D
      MODULE PROCEDURE acc_hostmem_alloc_r4_2D, acc_hostmem_alloc_r8_2D
      MODULE PROCEDURE acc_hostmem_alloc_c4_2D, acc_hostmem_alloc_c8_2D
   END INTERFACE

   INTERFACE acc_hostmem_deallocate
      MODULE PROCEDURE acc_hostmem_dealloc_i4, acc_hostmem_dealloc_i8
      MODULE PROCEDURE acc_hostmem_dealloc_r4, acc_hostmem_dealloc_r8
      MODULE PROCEDURE acc_hostmem_dealloc_c4, acc_hostmem_dealloc_c8
      MODULE PROCEDURE acc_hostmem_dealloc_i4_2D, acc_hostmem_dealloc_i8_2D
      MODULE PROCEDURE acc_hostmem_dealloc_r4_2D, acc_hostmem_dealloc_r8_2D
      MODULE PROCEDURE acc_hostmem_dealloc_c4_2D, acc_hostmem_dealloc_c8_2D
   END INTERFACE

#if defined (__DBCSR_ACC)

   INTERFACE
      FUNCTION cuda_host_mem_alloc_cu(mem, n, stream_ptr) RESULT(istat) BIND(C, name="acc_host_mem_allocate")
         IMPORT
         TYPE(C_PTR)                              :: mem
         INTEGER(KIND=C_SIZE_T), INTENT(IN), &
            VALUE                                  :: n
         TYPE(C_PTR), VALUE                       :: stream_ptr
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION cuda_host_mem_alloc_cu
   END INTERFACE

   INTERFACE
      FUNCTION cuda_host_mem_dealloc_cu(mem, stream_ptr) RESULT(istat) bind(C, name="acc_host_mem_deallocate")
         IMPORT
         TYPE(C_PTR), VALUE                       :: mem, stream_ptr
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION cuda_host_mem_dealloc_cu
   END INTERFACE

#endif

CONTAINS

#if defined (__DBCSR_ACC)
   SUBROUTINE acc_hostmem_alloc_raw(host_mem_c_ptr, n_bytes, stream)
      !! Helper-routine performing allocation of host-pinned cuda memory.

      TYPE(C_PTR), INTENT(OUT)                           :: host_mem_c_ptr
         !! pointer to allocated memory
      INTEGER, INTENT(IN)                                :: n_bytes
         !! number of bytes to allocate
      TYPE(acc_stream_type), INTENT(IN)                  :: stream

      CHARACTER(len=*), PARAMETER :: routineN = 'acc_hostmem_alloc_raw', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: istat
      TYPE(C_PTR)                                        :: stream_cptr

      IF (.NOT. acc_stream_associated(stream)) &
         DBCSR_ABORT("acc_hostmem_alloc_raw: stream not associated")

      stream_cptr = acc_stream_cptr(stream)

      istat = cuda_host_mem_alloc_cu(host_mem_c_ptr, INT(n_bytes, KIND=C_SIZE_T), stream_cptr)
      IF (istat /= 0) &
         DBCSR_ABORT("acc_hostmem_alloc_raw: Could not allocate host pinned memory")
   END SUBROUTINE acc_hostmem_alloc_raw
#endif

#if defined (__DBCSR_ACC)
   SUBROUTINE acc_hostmem_dealloc_raw(host_mem_c_ptr, stream)
      TYPE(C_PTR), INTENT(IN)                            :: host_mem_c_ptr
      TYPE(acc_stream_type), INTENT(IN)                  :: stream

      CHARACTER(len=*), PARAMETER :: routineN = 'acc_hostmem_dealloc_raw', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: istat
      TYPE(C_PTR)                                        :: stream_cptr

! Workaround for a segmentation fault on ORNL's Summit
!$OMP CRITICAL

      IF (.NOT. acc_stream_associated(stream)) &
         DBCSR_ABORT("acc_hostmem_dealloc_raw: stream not associated")

      stream_cptr = acc_stream_cptr(stream)

      istat = cuda_host_mem_dealloc_cu(host_mem_c_ptr, stream_cptr)
      IF (istat /= 0) &
         DBCSR_ABORT("acc_hostmem_dealloc_raw: Could not deallocate host pinned memory")
!$OMP END CRITICAL
   END SUBROUTINE acc_hostmem_dealloc_raw
#endif



   SUBROUTINE acc_hostmem_alloc_i4 (host_mem, n, stream)
      !! Allocates 1D fortan-array as cuda host-pinned memory.

      INTEGER(kind=int_4), DIMENSION(:), POINTER          :: host_mem
         !! pointer to array
      INTEGER, INTENT(IN)                      :: n
         !! size given in terms of item-count (not bytes!)
      TYPE(acc_stream_type), INTENT(IN)        :: stream
#if defined (__DBCSR_ACC)
      TYPE(C_PTR)                              :: host_mem_c_ptr

      CALL acc_hostmem_alloc_raw(host_mem_c_ptr, MAX(1, n)*int_4_size, stream)
      CALL C_F_POINTER(host_mem_c_ptr, host_mem, (/MAX(1, n)/))
#else
      MARK_USED(host_mem)
      MARK_USED(n)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_alloc_i4: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_alloc_i4

   SUBROUTINE acc_hostmem_alloc_i4_2D(host_mem, n1, n2, stream)
      !! Allocates 2D fortan-array as cuda host-pinned memory.

      INTEGER(kind=int_4), DIMENSION(:, :), POINTER        :: host_mem
         !! pointer to array
      INTEGER, INTENT(IN)                      :: n1, n2
         !! sizes given in terms of item-count (not bytes!)
         !! sizes given in terms of item-count (not bytes!)
      TYPE(acc_stream_type), INTENT(IN)        :: stream
#if defined (__DBCSR_ACC)
      TYPE(C_PTR)                              :: host_mem_c_ptr
      INTEGER                                  :: n_bytes

      n_bytes = MAX(1, n1)*MAX(1, n2)*int_4_size
      CALL acc_hostmem_alloc_raw(host_mem_c_ptr, n_bytes, stream)
      CALL C_F_POINTER(host_mem_c_ptr, host_mem, (/MAX(1, n1), MAX(1, n2)/))
#else
      MARK_USED(host_mem)
      MARK_USED(n1)
      MARK_USED(n2)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_alloc_i4_2D: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_alloc_i4_2D

   SUBROUTINE acc_hostmem_dealloc_i4 (host_mem, stream)
      !! Deallocates a 1D fortan-array, which is cuda host-pinned memory.

      INTEGER(kind=int_4), DIMENSION(:), POINTER          :: host_mem
         !! pointer to array
      TYPE(acc_stream_type), INTENT(IN)        :: stream
      CHARACTER(len=*), PARAMETER :: routineN = 'acc_hostmem_dealloc_i4', &
                                     routineP = moduleN//':'//routineN

      IF (SIZE(host_mem) == 0) RETURN
#if defined (__DBCSR_ACC)
      CALL acc_hostmem_dealloc_raw(C_LOC(host_mem(1)), stream)
#else
      MARK_USED(host_mem)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_dealloc_i4: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_dealloc_i4

   SUBROUTINE acc_hostmem_dealloc_i4_2D(host_mem, stream)
      !! Deallocates a 2D fortan-array, which is cuda host-pinned memory.

      INTEGER(kind=int_4), DIMENSION(:, :), POINTER        :: host_mem
         !! pointer to array
      TYPE(acc_stream_type), INTENT(IN)        :: stream
      CHARACTER(len=*), PARAMETER :: routineN = 'acc_hostmem_dealloc_i4_2D', &
                                     routineP = moduleN//':'//routineN

      IF (SIZE(host_mem) == 0) RETURN
#if defined (__DBCSR_ACC)
      CALL acc_hostmem_dealloc_raw(C_LOC(host_mem(1, 1)), stream)
#else
      MARK_USED(host_mem)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_dealloc_i4: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_dealloc_i4_2D


   SUBROUTINE acc_hostmem_alloc_i8 (host_mem, n, stream)
      !! Allocates 1D fortan-array as cuda host-pinned memory.

      INTEGER(kind=int_8), DIMENSION(:), POINTER          :: host_mem
         !! pointer to array
      INTEGER, INTENT(IN)                      :: n
         !! size given in terms of item-count (not bytes!)
      TYPE(acc_stream_type), INTENT(IN)        :: stream
#if defined (__DBCSR_ACC)
      TYPE(C_PTR)                              :: host_mem_c_ptr

      CALL acc_hostmem_alloc_raw(host_mem_c_ptr, MAX(1, n)*int_8_size, stream)
      CALL C_F_POINTER(host_mem_c_ptr, host_mem, (/MAX(1, n)/))
#else
      MARK_USED(host_mem)
      MARK_USED(n)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_alloc_i8: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_alloc_i8

   SUBROUTINE acc_hostmem_alloc_i8_2D(host_mem, n1, n2, stream)
      !! Allocates 2D fortan-array as cuda host-pinned memory.

      INTEGER(kind=int_8), DIMENSION(:, :), POINTER        :: host_mem
         !! pointer to array
      INTEGER, INTENT(IN)                      :: n1, n2
         !! sizes given in terms of item-count (not bytes!)
         !! sizes given in terms of item-count (not bytes!)
      TYPE(acc_stream_type), INTENT(IN)        :: stream
#if defined (__DBCSR_ACC)
      TYPE(C_PTR)                              :: host_mem_c_ptr
      INTEGER                                  :: n_bytes

      n_bytes = MAX(1, n1)*MAX(1, n2)*int_8_size
      CALL acc_hostmem_alloc_raw(host_mem_c_ptr, n_bytes, stream)
      CALL C_F_POINTER(host_mem_c_ptr, host_mem, (/MAX(1, n1), MAX(1, n2)/))
#else
      MARK_USED(host_mem)
      MARK_USED(n1)
      MARK_USED(n2)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_alloc_i8_2D: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_alloc_i8_2D

   SUBROUTINE acc_hostmem_dealloc_i8 (host_mem, stream)
      !! Deallocates a 1D fortan-array, which is cuda host-pinned memory.

      INTEGER(kind=int_8), DIMENSION(:), POINTER          :: host_mem
         !! pointer to array
      TYPE(acc_stream_type), INTENT(IN)        :: stream
      CHARACTER(len=*), PARAMETER :: routineN = 'acc_hostmem_dealloc_i8', &
                                     routineP = moduleN//':'//routineN

      IF (SIZE(host_mem) == 0) RETURN
#if defined (__DBCSR_ACC)
      CALL acc_hostmem_dealloc_raw(C_LOC(host_mem(1)), stream)
#else
      MARK_USED(host_mem)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_dealloc_i8: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_dealloc_i8

   SUBROUTINE acc_hostmem_dealloc_i8_2D(host_mem, stream)
      !! Deallocates a 2D fortan-array, which is cuda host-pinned memory.

      INTEGER(kind=int_8), DIMENSION(:, :), POINTER        :: host_mem
         !! pointer to array
      TYPE(acc_stream_type), INTENT(IN)        :: stream
      CHARACTER(len=*), PARAMETER :: routineN = 'acc_hostmem_dealloc_i8_2D', &
                                     routineP = moduleN//':'//routineN

      IF (SIZE(host_mem) == 0) RETURN
#if defined (__DBCSR_ACC)
      CALL acc_hostmem_dealloc_raw(C_LOC(host_mem(1, 1)), stream)
#else
      MARK_USED(host_mem)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_dealloc_i8: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_dealloc_i8_2D


   SUBROUTINE acc_hostmem_alloc_r4 (host_mem, n, stream)
      !! Allocates 1D fortan-array as cuda host-pinned memory.

      REAL(kind=real_4), DIMENSION(:), POINTER          :: host_mem
         !! pointer to array
      INTEGER, INTENT(IN)                      :: n
         !! size given in terms of item-count (not bytes!)
      TYPE(acc_stream_type), INTENT(IN)        :: stream
#if defined (__DBCSR_ACC)
      TYPE(C_PTR)                              :: host_mem_c_ptr

      CALL acc_hostmem_alloc_raw(host_mem_c_ptr, MAX(1, n)*real_4_size, stream)
      CALL C_F_POINTER(host_mem_c_ptr, host_mem, (/MAX(1, n)/))
#else
      MARK_USED(host_mem)
      MARK_USED(n)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_alloc_r4: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_alloc_r4

   SUBROUTINE acc_hostmem_alloc_r4_2D(host_mem, n1, n2, stream)
      !! Allocates 2D fortan-array as cuda host-pinned memory.

      REAL(kind=real_4), DIMENSION(:, :), POINTER        :: host_mem
         !! pointer to array
      INTEGER, INTENT(IN)                      :: n1, n2
         !! sizes given in terms of item-count (not bytes!)
         !! sizes given in terms of item-count (not bytes!)
      TYPE(acc_stream_type), INTENT(IN)        :: stream
#if defined (__DBCSR_ACC)
      TYPE(C_PTR)                              :: host_mem_c_ptr
      INTEGER                                  :: n_bytes

      n_bytes = MAX(1, n1)*MAX(1, n2)*real_4_size
      CALL acc_hostmem_alloc_raw(host_mem_c_ptr, n_bytes, stream)
      CALL C_F_POINTER(host_mem_c_ptr, host_mem, (/MAX(1, n1), MAX(1, n2)/))
#else
      MARK_USED(host_mem)
      MARK_USED(n1)
      MARK_USED(n2)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_alloc_r4_2D: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_alloc_r4_2D

   SUBROUTINE acc_hostmem_dealloc_r4 (host_mem, stream)
      !! Deallocates a 1D fortan-array, which is cuda host-pinned memory.

      REAL(kind=real_4), DIMENSION(:), POINTER          :: host_mem
         !! pointer to array
      TYPE(acc_stream_type), INTENT(IN)        :: stream
      CHARACTER(len=*), PARAMETER :: routineN = 'acc_hostmem_dealloc_r4', &
                                     routineP = moduleN//':'//routineN

      IF (SIZE(host_mem) == 0) RETURN
#if defined (__DBCSR_ACC)
      CALL acc_hostmem_dealloc_raw(C_LOC(host_mem(1)), stream)
#else
      MARK_USED(host_mem)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_dealloc_r4: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_dealloc_r4

   SUBROUTINE acc_hostmem_dealloc_r4_2D(host_mem, stream)
      !! Deallocates a 2D fortan-array, which is cuda host-pinned memory.

      REAL(kind=real_4), DIMENSION(:, :), POINTER        :: host_mem
         !! pointer to array
      TYPE(acc_stream_type), INTENT(IN)        :: stream
      CHARACTER(len=*), PARAMETER :: routineN = 'acc_hostmem_dealloc_r4_2D', &
                                     routineP = moduleN//':'//routineN

      IF (SIZE(host_mem) == 0) RETURN
#if defined (__DBCSR_ACC)
      CALL acc_hostmem_dealloc_raw(C_LOC(host_mem(1, 1)), stream)
#else
      MARK_USED(host_mem)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_dealloc_r4: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_dealloc_r4_2D


   SUBROUTINE acc_hostmem_alloc_r8 (host_mem, n, stream)
      !! Allocates 1D fortan-array as cuda host-pinned memory.

      REAL(kind=real_8), DIMENSION(:), POINTER          :: host_mem
         !! pointer to array
      INTEGER, INTENT(IN)                      :: n
         !! size given in terms of item-count (not bytes!)
      TYPE(acc_stream_type), INTENT(IN)        :: stream
#if defined (__DBCSR_ACC)
      TYPE(C_PTR)                              :: host_mem_c_ptr

      CALL acc_hostmem_alloc_raw(host_mem_c_ptr, MAX(1, n)*real_8_size, stream)
      CALL C_F_POINTER(host_mem_c_ptr, host_mem, (/MAX(1, n)/))
#else
      MARK_USED(host_mem)
      MARK_USED(n)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_alloc_r8: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_alloc_r8

   SUBROUTINE acc_hostmem_alloc_r8_2D(host_mem, n1, n2, stream)
      !! Allocates 2D fortan-array as cuda host-pinned memory.

      REAL(kind=real_8), DIMENSION(:, :), POINTER        :: host_mem
         !! pointer to array
      INTEGER, INTENT(IN)                      :: n1, n2
         !! sizes given in terms of item-count (not bytes!)
         !! sizes given in terms of item-count (not bytes!)
      TYPE(acc_stream_type), INTENT(IN)        :: stream
#if defined (__DBCSR_ACC)
      TYPE(C_PTR)                              :: host_mem_c_ptr
      INTEGER                                  :: n_bytes

      n_bytes = MAX(1, n1)*MAX(1, n2)*real_8_size
      CALL acc_hostmem_alloc_raw(host_mem_c_ptr, n_bytes, stream)
      CALL C_F_POINTER(host_mem_c_ptr, host_mem, (/MAX(1, n1), MAX(1, n2)/))
#else
      MARK_USED(host_mem)
      MARK_USED(n1)
      MARK_USED(n2)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_alloc_r8_2D: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_alloc_r8_2D

   SUBROUTINE acc_hostmem_dealloc_r8 (host_mem, stream)
      !! Deallocates a 1D fortan-array, which is cuda host-pinned memory.

      REAL(kind=real_8), DIMENSION(:), POINTER          :: host_mem
         !! pointer to array
      TYPE(acc_stream_type), INTENT(IN)        :: stream
      CHARACTER(len=*), PARAMETER :: routineN = 'acc_hostmem_dealloc_r8', &
                                     routineP = moduleN//':'//routineN

      IF (SIZE(host_mem) == 0) RETURN
#if defined (__DBCSR_ACC)
      CALL acc_hostmem_dealloc_raw(C_LOC(host_mem(1)), stream)
#else
      MARK_USED(host_mem)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_dealloc_r8: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_dealloc_r8

   SUBROUTINE acc_hostmem_dealloc_r8_2D(host_mem, stream)
      !! Deallocates a 2D fortan-array, which is cuda host-pinned memory.

      REAL(kind=real_8), DIMENSION(:, :), POINTER        :: host_mem
         !! pointer to array
      TYPE(acc_stream_type), INTENT(IN)        :: stream
      CHARACTER(len=*), PARAMETER :: routineN = 'acc_hostmem_dealloc_r8_2D', &
                                     routineP = moduleN//':'//routineN

      IF (SIZE(host_mem) == 0) RETURN
#if defined (__DBCSR_ACC)
      CALL acc_hostmem_dealloc_raw(C_LOC(host_mem(1, 1)), stream)
#else
      MARK_USED(host_mem)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_dealloc_r8: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_dealloc_r8_2D


   SUBROUTINE acc_hostmem_alloc_c4 (host_mem, n, stream)
      !! Allocates 1D fortan-array as cuda host-pinned memory.

      COMPLEX(kind=real_4), DIMENSION(:), POINTER          :: host_mem
         !! pointer to array
      INTEGER, INTENT(IN)                      :: n
         !! size given in terms of item-count (not bytes!)
      TYPE(acc_stream_type), INTENT(IN)        :: stream
#if defined (__DBCSR_ACC)
      TYPE(C_PTR)                              :: host_mem_c_ptr

      CALL acc_hostmem_alloc_raw(host_mem_c_ptr, MAX(1, n)*2*real_4_size, stream)
      CALL C_F_POINTER(host_mem_c_ptr, host_mem, (/MAX(1, n)/))
#else
      MARK_USED(host_mem)
      MARK_USED(n)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_alloc_c4: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_alloc_c4

   SUBROUTINE acc_hostmem_alloc_c4_2D(host_mem, n1, n2, stream)
      !! Allocates 2D fortan-array as cuda host-pinned memory.

      COMPLEX(kind=real_4), DIMENSION(:, :), POINTER        :: host_mem
         !! pointer to array
      INTEGER, INTENT(IN)                      :: n1, n2
         !! sizes given in terms of item-count (not bytes!)
         !! sizes given in terms of item-count (not bytes!)
      TYPE(acc_stream_type), INTENT(IN)        :: stream
#if defined (__DBCSR_ACC)
      TYPE(C_PTR)                              :: host_mem_c_ptr
      INTEGER                                  :: n_bytes

      n_bytes = MAX(1, n1)*MAX(1, n2)*2*real_4_size
      CALL acc_hostmem_alloc_raw(host_mem_c_ptr, n_bytes, stream)
      CALL C_F_POINTER(host_mem_c_ptr, host_mem, (/MAX(1, n1), MAX(1, n2)/))
#else
      MARK_USED(host_mem)
      MARK_USED(n1)
      MARK_USED(n2)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_alloc_c4_2D: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_alloc_c4_2D

   SUBROUTINE acc_hostmem_dealloc_c4 (host_mem, stream)
      !! Deallocates a 1D fortan-array, which is cuda host-pinned memory.

      COMPLEX(kind=real_4), DIMENSION(:), POINTER          :: host_mem
         !! pointer to array
      TYPE(acc_stream_type), INTENT(IN)        :: stream
      CHARACTER(len=*), PARAMETER :: routineN = 'acc_hostmem_dealloc_c4', &
                                     routineP = moduleN//':'//routineN

      IF (SIZE(host_mem) == 0) RETURN
#if defined (__DBCSR_ACC)
      CALL acc_hostmem_dealloc_raw(C_LOC(host_mem(1)), stream)
#else
      MARK_USED(host_mem)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_dealloc_c4: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_dealloc_c4

   SUBROUTINE acc_hostmem_dealloc_c4_2D(host_mem, stream)
      !! Deallocates a 2D fortan-array, which is cuda host-pinned memory.

      COMPLEX(kind=real_4), DIMENSION(:, :), POINTER        :: host_mem
         !! pointer to array
      TYPE(acc_stream_type), INTENT(IN)        :: stream
      CHARACTER(len=*), PARAMETER :: routineN = 'acc_hostmem_dealloc_c4_2D', &
                                     routineP = moduleN//':'//routineN

      IF (SIZE(host_mem) == 0) RETURN
#if defined (__DBCSR_ACC)
      CALL acc_hostmem_dealloc_raw(C_LOC(host_mem(1, 1)), stream)
#else
      MARK_USED(host_mem)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_dealloc_c4: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_dealloc_c4_2D


   SUBROUTINE acc_hostmem_alloc_c8 (host_mem, n, stream)
      !! Allocates 1D fortan-array as cuda host-pinned memory.

      COMPLEX(kind=real_8), DIMENSION(:), POINTER          :: host_mem
         !! pointer to array
      INTEGER, INTENT(IN)                      :: n
         !! size given in terms of item-count (not bytes!)
      TYPE(acc_stream_type), INTENT(IN)        :: stream
#if defined (__DBCSR_ACC)
      TYPE(C_PTR)                              :: host_mem_c_ptr

      CALL acc_hostmem_alloc_raw(host_mem_c_ptr, MAX(1, n)*2*real_8_size, stream)
      CALL C_F_POINTER(host_mem_c_ptr, host_mem, (/MAX(1, n)/))
#else
      MARK_USED(host_mem)
      MARK_USED(n)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_alloc_c8: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_alloc_c8

   SUBROUTINE acc_hostmem_alloc_c8_2D(host_mem, n1, n2, stream)
      !! Allocates 2D fortan-array as cuda host-pinned memory.

      COMPLEX(kind=real_8), DIMENSION(:, :), POINTER        :: host_mem
         !! pointer to array
      INTEGER, INTENT(IN)                      :: n1, n2
         !! sizes given in terms of item-count (not bytes!)
         !! sizes given in terms of item-count (not bytes!)
      TYPE(acc_stream_type), INTENT(IN)        :: stream
#if defined (__DBCSR_ACC)
      TYPE(C_PTR)                              :: host_mem_c_ptr
      INTEGER                                  :: n_bytes

      n_bytes = MAX(1, n1)*MAX(1, n2)*2*real_8_size
      CALL acc_hostmem_alloc_raw(host_mem_c_ptr, n_bytes, stream)
      CALL C_F_POINTER(host_mem_c_ptr, host_mem, (/MAX(1, n1), MAX(1, n2)/))
#else
      MARK_USED(host_mem)
      MARK_USED(n1)
      MARK_USED(n2)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_alloc_c8_2D: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_alloc_c8_2D

   SUBROUTINE acc_hostmem_dealloc_c8 (host_mem, stream)
      !! Deallocates a 1D fortan-array, which is cuda host-pinned memory.

      COMPLEX(kind=real_8), DIMENSION(:), POINTER          :: host_mem
         !! pointer to array
      TYPE(acc_stream_type), INTENT(IN)        :: stream
      CHARACTER(len=*), PARAMETER :: routineN = 'acc_hostmem_dealloc_c8', &
                                     routineP = moduleN//':'//routineN

      IF (SIZE(host_mem) == 0) RETURN
#if defined (__DBCSR_ACC)
      CALL acc_hostmem_dealloc_raw(C_LOC(host_mem(1)), stream)
#else
      MARK_USED(host_mem)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_dealloc_c8: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_dealloc_c8

   SUBROUTINE acc_hostmem_dealloc_c8_2D(host_mem, stream)
      !! Deallocates a 2D fortan-array, which is cuda host-pinned memory.

      COMPLEX(kind=real_8), DIMENSION(:, :), POINTER        :: host_mem
         !! pointer to array
      TYPE(acc_stream_type), INTENT(IN)        :: stream
      CHARACTER(len=*), PARAMETER :: routineN = 'acc_hostmem_dealloc_c8_2D', &
                                     routineP = moduleN//':'//routineN

      IF (SIZE(host_mem) == 0) RETURN
#if defined (__DBCSR_ACC)
      CALL acc_hostmem_dealloc_raw(C_LOC(host_mem(1, 1)), stream)
#else
      MARK_USED(host_mem)
      MARK_USED(stream)
      DBCSR_ABORT("acc_hostmem_dealloc_c8: ACC not compiled in.")
#endif
   END SUBROUTINE acc_hostmem_dealloc_c8_2D


END MODULE dbcsr_acc_hostmem
