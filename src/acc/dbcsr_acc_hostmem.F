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
   USE dbcsr_acc_device, ONLY: dbcsr_acc_set_active_device
   USE dbcsr_config, ONLY: get_accdrv_active_device_id
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
      FUNCTION acc_interface_host_mem_alloc(mem, n, stream_ptr) RESULT(istat) BIND(C, name="c_dbcsr_acc_host_mem_allocate")
         IMPORT
         TYPE(C_PTR)                              :: mem
         INTEGER(KIND=C_SIZE_T), INTENT(IN), &
            VALUE                                  :: n
         TYPE(C_PTR), VALUE                       :: stream_ptr
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION acc_interface_host_mem_alloc
   END INTERFACE

   INTERFACE
      FUNCTION acc_interface_host_mem_dealloc(mem, stream_ptr) RESULT(istat) BIND(C, name="c_dbcsr_acc_host_mem_deallocate")
         IMPORT
         TYPE(C_PTR), VALUE                       :: mem, stream_ptr
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION acc_interface_host_mem_dealloc
   END INTERFACE

#endif

CONTAINS

#if defined (__DBCSR_ACC)
   SUBROUTINE acc_hostmem_alloc_raw(host_mem_c_ptr, n_bytes, stream)
      !! Helper-routine performing allocation of host-pinned GPU memory.

      TYPE(C_PTR), INTENT(OUT)                           :: host_mem_c_ptr
      !! pointer to allocated memory
      INTEGER(KIND=C_SIZE_T), INTENT(IN)                 :: n_bytes
      !! number of bytes to allocate
      TYPE(acc_stream_type), INTENT(IN)                  :: stream

      INTEGER                                            :: istat
      TYPE(C_PTR)                                        :: stream_cptr

      IF (.NOT. acc_stream_associated(stream)) &
         DBCSR_ABORT("acc_hostmem_alloc_raw: stream not associated")

      stream_cptr = acc_stream_cptr(stream)

      CALL dbcsr_acc_set_active_device(get_accdrv_active_device_id())
      istat = acc_interface_host_mem_alloc(host_mem_c_ptr, n_bytes, stream_cptr)
      IF (istat /= 0) &
         DBCSR_ABORT("acc_hostmem_alloc_raw: Could not allocate host pinned memory")
   END SUBROUTINE acc_hostmem_alloc_raw
#endif

#if defined (__DBCSR_ACC)
   SUBROUTINE acc_hostmem_dealloc_raw(host_mem_c_ptr, stream)
      TYPE(C_PTR), INTENT(IN)                            :: host_mem_c_ptr
      TYPE(acc_stream_type), INTENT(IN)                  :: stream

      INTEGER                                            :: istat
      TYPE(C_PTR)                                        :: stream_cptr

! Workaround for a segmentation fault on ORNL's Summit
!$OMP CRITICAL

      IF (.NOT. acc_stream_associated(stream)) &
         DBCSR_ABORT("acc_hostmem_dealloc_raw: stream not associated")

      stream_cptr = acc_stream_cptr(stream)

      CALL dbcsr_acc_set_active_device(get_accdrv_active_device_id())
      istat = acc_interface_host_mem_dealloc(host_mem_c_ptr, stream_cptr)
      IF (istat /= 0) &
         DBCSR_ABORT("acc_hostmem_dealloc_raw: Could not deallocate host pinned memory")
!$OMP END CRITICAL
   END SUBROUTINE acc_hostmem_dealloc_raw
#endif

   #:set instances = [ &
      ('i4', 'int_4_size',    'INTEGER(kind=int_4)'), &
      ('i8', 'int_8_size',    'INTEGER(kind=int_8)'), &
      ('r4', 'real_4_size',   'REAL(kind=real_4)'), &
      ('r8', 'real_8_size',   'REAL(kind=real_8)'), &
      ('c4', '2*real_4_size', 'COMPLEX(kind=real_4)'), &
      ('c8', '2*real_8_size', 'COMPLEX(kind=real_8)') ]

   #:for nametype, size, type in instances

      SUBROUTINE acc_hostmem_alloc_${nametype}$ (host_mem, n, stream)
      !! Allocates 1D fortan-array as GPU host-pinned memory.

         ${type}$, DIMENSION(:), POINTER          :: host_mem
         !! pointer to array
         INTEGER, INTENT(IN)                      :: n
         !! size given in terms of item-count (not bytes!)
         TYPE(acc_stream_type), INTENT(IN)        :: stream
#if defined (__DBCSR_ACC)
         TYPE(C_PTR)                              :: host_mem_c_ptr
         INTEGER(KIND=C_SIZE_T)                   :: n_bytes

         n_bytes = INT(${size}$, KIND=C_SIZE_T)* &
                   INT(MAX(1, n), KIND=C_SIZE_T)
         CALL acc_hostmem_alloc_raw(host_mem_c_ptr, n_bytes, stream)
         CALL C_F_POINTER(host_mem_c_ptr, host_mem, (/MAX(1, n)/))
#else
         MARK_USED(host_mem)
         MARK_USED(n)
         MARK_USED(stream)
         DBCSR_ABORT("acc_hostmem_alloc_${nametype}$: ACC not compiled in.")
#endif
      END SUBROUTINE acc_hostmem_alloc_${nametype}$

      SUBROUTINE acc_hostmem_alloc_${nametype}$_2D(host_mem, n1, n2, stream)
      !! Allocates 2D fortan-array as GPU host-pinned memory.

         ${type}$, DIMENSION(:, :), POINTER        :: host_mem
         !! pointer to array
         INTEGER, INTENT(IN)                      :: n1, n2
         !! sizes given in terms of item-count (not bytes!)
         !! sizes given in terms of item-count (not bytes!)
         TYPE(acc_stream_type), INTENT(IN)        :: stream
#if defined (__DBCSR_ACC)
         TYPE(C_PTR)                              :: host_mem_c_ptr
         INTEGER(KIND=C_SIZE_T)                   :: n_bytes

         n_bytes = INT(${size}$, KIND=C_SIZE_T)* &
                   INT(MAX(1, n1), KIND=C_SIZE_T)*INT(MAX(1, n2), KIND=C_SIZE_T)
         CALL acc_hostmem_alloc_raw(host_mem_c_ptr, n_bytes, stream)
         CALL C_F_POINTER(host_mem_c_ptr, host_mem, (/MAX(1, n1), MAX(1, n2)/))
#else
         MARK_USED(host_mem)
         MARK_USED(n1)
         MARK_USED(n2)
         MARK_USED(stream)
         DBCSR_ABORT("acc_hostmem_alloc_${nametype}$_2D: ACC not compiled in.")
#endif
      END SUBROUTINE acc_hostmem_alloc_${nametype}$_2D

      SUBROUTINE acc_hostmem_dealloc_${nametype}$ (host_mem, stream)
      !! Deallocates a 1D fortan-array, which is GPU host-pinned memory.

         ${type}$, DIMENSION(:), POINTER          :: host_mem
         !! pointer to array
         TYPE(acc_stream_type), INTENT(IN)        :: stream

         IF (SIZE(host_mem) == 0) RETURN
#if defined (__DBCSR_ACC)
         CALL acc_hostmem_dealloc_raw(C_LOC(host_mem(1)), stream)
#else
         MARK_USED(host_mem)
         MARK_USED(stream)
         DBCSR_ABORT("acc_hostmem_dealloc_${nametype}$: ACC not compiled in.")
#endif
      END SUBROUTINE acc_hostmem_dealloc_${nametype}$

      SUBROUTINE acc_hostmem_dealloc_${nametype}$_2D(host_mem, stream)
      !! Deallocates a 2D fortan-array, which is GPU host-pinned memory.

         ${type}$, DIMENSION(:, :), POINTER        :: host_mem
         !! pointer to array
         TYPE(acc_stream_type), INTENT(IN)        :: stream

         IF (SIZE(host_mem) == 0) RETURN
#if defined (__DBCSR_ACC)
         CALL acc_hostmem_dealloc_raw(C_LOC(host_mem(1, 1)), stream)
#else
         MARK_USED(host_mem)
         MARK_USED(stream)
         DBCSR_ABORT("acc_hostmem_dealloc_${nametype}$: ACC not compiled in.")
#endif
      END SUBROUTINE acc_hostmem_dealloc_${nametype}$_2D

   #:endfor

END MODULE dbcsr_acc_hostmem
