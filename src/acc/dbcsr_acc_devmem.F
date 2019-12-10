!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_acc_devmem
   !! Accelerator support
#if  defined (__DBCSR_ACC)
   USE ISO_C_BINDING, ONLY: C_INT, C_SIZE_T, C_PTR, C_LOC, C_NULL_PTR
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
                               acc_stream_synchronize, &
                               acc_stream_type
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_acc_devmem'

   PUBLIC :: acc_devmem_type
   PUBLIC :: acc_devmem_allocate_bytes, acc_devmem_deallocate
   PUBLIC :: acc_devmem_setzero_bytes
   PUBLIC :: acc_devmem_allocated
   PUBLIC :: acc_devmem_dev2host, acc_devmem_host2dev
   PUBLIC :: acc_devmem_size_in_bytes
   PUBLIC :: acc_devmem_ensure_size_bytes
   PUBLIC :: acc_devmem_cptr
   PUBLIC :: acc_devmem_set_cptr

   INTERFACE acc_devmem_dev2host
      MODULE PROCEDURE dev2host_i4_1D
      MODULE PROCEDURE dev2host_i8_1D
      MODULE PROCEDURE dev2host_r4_1D
      MODULE PROCEDURE dev2host_r8_1D
      MODULE PROCEDURE dev2host_c4_1D
      MODULE PROCEDURE dev2host_c8_1D
   END INTERFACE acc_devmem_dev2host

   INTERFACE acc_devmem_host2dev
      MODULE PROCEDURE host2dev_i4_1D
      MODULE PROCEDURE host2dev_i8_1D
      MODULE PROCEDURE host2dev_r4_1D
      MODULE PROCEDURE host2dev_r8_1D
      MODULE PROCEDURE host2dev_c4_1D
      MODULE PROCEDURE host2dev_c8_1D
      MODULE PROCEDURE host2dev_i4_2D
      MODULE PROCEDURE host2dev_i8_2D
      MODULE PROCEDURE host2dev_r4_2D
      MODULE PROCEDURE host2dev_r8_2D
      MODULE PROCEDURE host2dev_c4_2D
      MODULE PROCEDURE host2dev_c8_2D
   END INTERFACE acc_devmem_host2dev

   TYPE acc_devmem_type
      PRIVATE
      INTEGER                      :: size_in_bytes = -1
#if  defined (__DBCSR_ACC)
      TYPE(C_PTR)                  :: cptr = C_NULL_PTR
#endif
   END TYPE acc_devmem_type

#if  defined (__DBCSR_ACC)

   INTERFACE
      FUNCTION acc_interface_dev_mem_info(free, avail) RESULT(istat) BIND(C, name="acc_dev_mem_info")
         IMPORT
         INTEGER(KIND=C_INT), INTENT(OUT)         :: free, avail
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION acc_interface_dev_mem_info
   END INTERFACE

   INTERFACE
      FUNCTION acc_interface_dev_mem_alloc(mem, n) RESULT(istat) BIND(C, name="acc_dev_mem_allocate")
         IMPORT
         TYPE(C_PTR)                              :: mem
         INTEGER(KIND=C_SIZE_T), INTENT(IN), &
            VALUE                                  :: n
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION acc_interface_dev_mem_alloc
   END INTERFACE

   INTERFACE
      FUNCTION acc_interface_dev_mem_dealloc(mem) RESULT(istat) BIND(C, name="acc_dev_mem_deallocate")
         IMPORT
         TYPE(C_PTR), VALUE                       :: mem
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION acc_interface_dev_mem_dealloc
   END INTERFACE

   INTERFACE
      FUNCTION acc_interface_dev_mem_set_ptr(mem, other, lb) RESULT(istat) BIND(C, name="acc_dev_mem_set_ptr")
         IMPORT
         TYPE(C_PTR)                              :: mem
         TYPE(C_PTR), VALUE                       :: other
         INTEGER(KIND=C_SIZE_T), INTENT(IN), &
            VALUE                                :: lb
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION acc_interface_dev_mem_set_ptr
   END INTERFACE

   INTERFACE
      FUNCTION acc_interface_memzero(this, offset, length, stream_ptr) RESULT(istat) BIND(C, name="acc_memset_zero")
         IMPORT
         TYPE(C_PTR), INTENT(IN), VALUE           :: this
         INTEGER(KIND=C_SIZE_T), INTENT(IN), &
            VALUE                                  :: offset, length
         TYPE(C_PTR), VALUE                       :: stream_ptr
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION acc_interface_memzero
   END INTERFACE

   INTERFACE
      FUNCTION acc_interface_memcpy_h2d(host, dev, count, stream_ptr) RESULT(istat) &
         BIND(C, name="acc_memcpy_h2d")
         IMPORT
         TYPE(C_PTR), INTENT(IN), VALUE           :: host
         TYPE(C_PTR), VALUE                       :: dev
         INTEGER(KIND=C_SIZE_T), INTENT(IN), &
            VALUE                                  :: count
         TYPE(C_PTR), VALUE                       :: stream_ptr
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION acc_interface_memcpy_h2d
   END INTERFACE

   INTERFACE
      FUNCTION acc_interface_memcpy_d2h(dev, host, count, stream_ptr) RESULT(istat) &
         BIND(C, name="acc_memcpy_d2h")
         IMPORT
         TYPE(C_PTR), INTENT(IN), VALUE           :: dev
         TYPE(C_PTR), VALUE                       :: host
         INTEGER(KIND=C_SIZE_T), INTENT(IN), &
            VALUE                                  :: count
         TYPE(C_PTR), VALUE                       :: stream_ptr
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION acc_interface_memcpy_d2h
   END INTERFACE

   INTERFACE
      FUNCTION acc_interface_memcpy_d2d(dev_src, dev_dst, count, stream_ptr) RESULT(istat) &
         BIND(C, name="acc_memcpy_d2d")
         IMPORT
         TYPE(C_PTR), INTENT(IN), VALUE           :: dev_src
         TYPE(C_PTR), VALUE                       :: dev_dst
         INTEGER(KIND=C_SIZE_T), INTENT(IN), &
            VALUE                                  :: count
         TYPE(C_PTR), VALUE                       :: stream_ptr
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION acc_interface_memcpy_d2d
   END INTERFACE

#endif

CONTAINS

   SUBROUTINE acc_devmem_ensure_size_bytes(this, stream, requested_size_in_bytes, nocopy, zero_pad)
      !! Ensures that given devmem has at least the requested size.

      TYPE(acc_devmem_type), &
         INTENT(INOUT)                          :: this
         !! device memory
      TYPE(acc_stream_type), INTENT(IN) :: stream
         !! on which zeroing and memcopying is performed
      INTEGER, INTENT(IN)                      :: requested_size_in_bytes
         !! requested size in bytes
      LOGICAL, INTENT(IN), OPTIONAL            :: nocopy, zero_pad
         !! if after growin old content should NOT be copied over. Default: false.
         !! if after growing the new memory should be zeroed. Default: false.

#if ! defined (__DBCSR_ACC)
      MARK_USED(this)
      MARK_USED(stream)
      MARK_USED(requested_size_in_bytes)
      MARK_USED(nocopy)
      MARK_USED(zero_pad)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else

      LOGICAL                                  :: my_nocopy, my_zero_pad
      TYPE(C_PTR)                              :: old_cptr, new_cptr, stream_cptr
      INTEGER                                  :: new_size, old_size, istat

      IF (this%size_in_bytes < 0) &
         DBCSR_ABORT("acc_devmem_ensure_size_bytes: not allocated")
      IF (.NOT. acc_stream_associated(stream)) &
         DBCSR_ABORT("acc_devmem_ensure_size_bytes: stream not associated")

      IF (this%size_in_bytes < requested_size_in_bytes) THEN
         !WRITE (*,*) "acc_devmem_ensure_size_bytes: growing dev_mem to: ", data_size

         new_size = requested_size_in_bytes
         old_size = this%size_in_bytes
         old_cptr = this%cptr

         new_cptr = C_NULL_PTR
         istat = acc_interface_dev_mem_alloc(new_cptr, INT(new_size, KIND=C_SIZE_T))
         IF (istat /= 0) &
            DBCSR_ABORT("acc_devmem_ensure_size_bytes: alloc failed")

         this%cptr = new_cptr
         this%size_in_bytes = requested_size_in_bytes

         my_zero_pad = .FALSE.
         IF (PRESENT(zero_pad)) my_zero_pad = zero_pad
         IF (my_zero_pad) &
            CALL acc_devmem_setzero_bytes(this, first_byte=old_size + 1, stream=stream)

         my_nocopy = .FALSE.
         IF (PRESENT(nocopy)) my_nocopy = nocopy
         IF (.NOT. my_nocopy) THEN
            stream_cptr = acc_stream_cptr(stream)
            istat = acc_interface_memcpy_d2d(old_cptr, new_cptr, INT(old_size, KIND=C_SIZE_T), stream_cptr)
            IF (istat /= 0) &
               DBCSR_ABORT("acc_devmem_ensure_size_bytes: memcpy failed")
         ENDIF

         CALL acc_stream_synchronize(stream)
         istat = acc_interface_dev_mem_dealloc(old_cptr)
         IF (istat /= 0) &
            DBCSR_ABORT("acc_devmem_ensure_size_bytes: dealloc failed")

      ENDIF
#endif
   END SUBROUTINE acc_devmem_ensure_size_bytes

   FUNCTION acc_devmem_allocated(this) RESULT(res)
      !! Returns a logical, which indicates if the given devmem is allocated.

      TYPE(acc_devmem_type), INTENT(IN)                  :: this
         !! device memory
      LOGICAL                                            :: res
         !! true if device memory is allocated, false otherwise

      res = this%size_in_bytes >= 0
   END FUNCTION acc_devmem_allocated

   FUNCTION acc_devmem_size_in_bytes(this) RESULT(res)
      !! Returns size of given devmem in terms of item count (not bytes!)

      TYPE(acc_devmem_type), INTENT(IN)                  :: this
         !! device memory
      INTEGER                                            :: res
         !! size of device memory (item count)

      IF (this%size_in_bytes < 0) &
         DBCSR_ABORT("acc_devmem_len: not allocated")
      res = this%size_in_bytes
   END FUNCTION acc_devmem_size_in_bytes

#if ! defined (__DBCSR_ACC)
   FUNCTION acc_devmem_cptr(this) RESULT(res)
      !! Returns C-pointer to data of given devmem.

      INTEGER, INTENT(IN)                                :: this
         !! device memory
      LOGICAL                                            :: res
         !! false (accelerator support is not enabled)

      MARK_USED(this)
      res = .FALSE.
   END FUNCTION acc_devmem_cptr
#else
   FUNCTION acc_devmem_cptr(this) RESULT(res)
      !! Returns C-pointer to data of given devmem.

      TYPE(acc_devmem_type), INTENT(IN)                  :: this
         !! device memory
      TYPE(C_PTR)                                        :: res
         !! C-pointer to data of given devmem

      IF (this%size_in_bytes < 0) &
         DBCSR_ABORT("acc_devmem_cptr: not allocated")
      res = this%cptr
   END FUNCTION acc_devmem_cptr
#endif

   SUBROUTINE acc_devmem_set_cptr(this, pointee, size_in_bytes, lb_in_bytes)
      !! Allocates a given devmem.

      TYPE(acc_devmem_type), INTENT(INOUT)     :: this
         !! device memory
      TYPE(acc_devmem_type), INTENT(IN)        :: pointee
      INTEGER, INTENT(IN)                      :: size_in_bytes, lb_in_bytes
         !! size in bytes

#if ! defined (__DBCSR_ACC)
      MARK_USED(this)
      MARK_USED(pointee)
      MARK_USED(size_in_bytes)
      MARK_USED(lb_in_bytes)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else

      INTEGER                                  :: istat

      IF (this%size_in_bytes >= 0) &
         DBCSR_ABORT("acc_devmem_set_cptr: already allocated")
      IF (pointee%size_in_bytes < 0 .AND. size_in_bytes > 0) &
         DBCSR_ABORT("acc_devmem_set_cptr: out-of-bounds")
      IF (size_in_bytes > 0) THEN
         IF ((lb_in_bytes + size_in_bytes) .GT. pointee%size_in_bytes) &
            DBCSR_ABORT("acc_devmem_set_cptr: out-of-bounds")
         this%size_in_bytes = size_in_bytes
         istat = acc_interface_dev_mem_set_ptr(this%cptr, pointee%cptr, INT(lb_in_bytes, KIND=C_SIZE_T))
      ELSE
         ! Empty buffers
         this%size_in_bytes = pointee%size_in_bytes
         this%cptr = pointee%cptr
      ENDIF
#endif
   END SUBROUTINE acc_devmem_set_cptr

   SUBROUTINE acc_devmem_info(free, avail)
      !! Fortran-wrapper for getting GPU memory info.

      INTEGER, INTENT(OUT)                     :: free, avail
         !! free device memory
         !! available device memory

#if ! defined (__DBCSR_ACC)
      avail = -1; free = -1 ! assign intent-out arguments to silence compiler warnings
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else
      INTEGER                                  :: istat
      istat = acc_interface_dev_mem_info(free, avail)
      IF (istat /= 0) &
         DBCSR_ABORT("acc_devmem_info: failed.")
#endif
   END SUBROUTINE acc_devmem_info

   SUBROUTINE acc_devmem_allocate_bytes(this, size_in_bytes)
      !! Allocates a given devmem.

      TYPE(acc_devmem_type), INTENT(INOUT)     :: this
         !! device memory
      INTEGER, INTENT(IN)                      :: size_in_bytes
         !! size in bytes

#if ! defined (__DBCSR_ACC)
      MARK_USED(this)
      MARK_USED(size_in_bytes)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else
      CHARACTER(len=*), PARAMETER :: routineN = 'acc_devmem_allocate', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                  :: istat

      IF (this%size_in_bytes >= 0) &
         DBCSR_ABORT("acc_devmem_alloc: already allocated")
      this%size_in_bytes = size_in_bytes
      IF (size_in_bytes > 0) THEN
         istat = acc_interface_dev_mem_alloc(this%cptr, INT(this%size_in_bytes, KIND=C_SIZE_T))
         IF (istat /= 0) &
            DBCSR_ABORT("acc_devmem_allocate: failed.")
      END IF
#endif
   END SUBROUTINE acc_devmem_allocate_bytes

   SUBROUTINE acc_devmem_deallocate(this)
      !! Deallocates a given devmem.

      TYPE(acc_devmem_type), INTENT(INOUT) :: this
         !! device memory

#if ! defined (__DBCSR_ACC)
      MARK_USED(this)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else
      CHARACTER(len=*), PARAMETER :: routineN = 'acc_devmem_deallocate', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                  :: istat

      IF (this%size_in_bytes < 0) &
         DBCSR_ABORT("acc_devmem_deallocate: double free")
      IF (this%size_in_bytes > 0) THEN
         istat = acc_interface_dev_mem_dealloc(this%cptr)
         IF (istat /= 0) &
            DBCSR_ABORT("acc_devmem_deallocate: failed")
      END IF

      this%size_in_bytes = -1

#endif
   END SUBROUTINE acc_devmem_deallocate

   SUBROUTINE acc_devmem_setzero_bytes(this, first_byte, last_byte, stream)
      !! Sets entries in given devmem to zero, asynchronously.

      TYPE(acc_devmem_type), INTENT(INOUT) :: this
         !! device memory
      INTEGER, INTENT(IN), OPTIONAL        :: first_byte, last_byte
         !! begin of region to zero, defaults to 1 if not given.
         !! end of region to zero, defaults to size if not given.
      TYPE(acc_stream_type), INTENT(IN)    :: stream
         !! stream on which zeroing is performed.

#if ! defined (__DBCSR_ACC)
      MARK_USED(this)
      MARK_USED(first_byte)
      MARK_USED(last_byte)
      MARK_USED(stream)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else
      CHARACTER(len=*), PARAMETER :: routineN = 'acc_devmem_setzero', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                  :: istat
      INTEGER(KIND=C_SIZE_T)                   :: length, offset
      TYPE(C_PTR)                              :: stream_cptr

      offset = 0
      length = this%size_in_bytes
      IF (PRESENT(first_byte)) THEN
         offset = first_byte - 1
         length = length - offset
      ENDIF
      IF (PRESENT(last_byte)) THEN
         length = last_byte
         IF (PRESENT(first_byte)) length = length - first_byte
      ENDIF

      stream_cptr = acc_stream_cptr(stream)

      IF (length > 0) THEN
         istat = acc_interface_memzero(this%cptr, offset, length, stream_cptr)
         IF (istat /= 0) &
            DBCSR_ABORT("acc_devmem_setzero: failed")
      ENDIF
#endif
   END SUBROUTINE acc_devmem_setzero_bytes

#if defined (__DBCSR_ACC)
   SUBROUTINE host2dev_raw(this, hostmem_cptr, n_bytes, stream)
      !! Helper-routine performing actuall host2dev transfers.

      TYPE(acc_devmem_type), INTENT(IN)                  :: this
         !! device memory
      TYPE(C_PTR)                                        :: hostmem_cptr
         !! host memory
      INTEGER, INTENT(IN)                                :: n_bytes
         !! number of bytes
      TYPE(acc_stream_type), INTENT(IN)                  :: stream
         !! stream used for memory transfer

      CHARACTER(len=*), PARAMETER :: routineN = 'host2dev_raw', routineP = moduleN//':'//routineN

      INTEGER                                            :: istat
      TYPE(C_PTR)                                        :: stream_cptr

      IF (this%size_in_bytes < n_bytes) &
         DBCSR_ABORT("acc_devmem_host2dev: devmem too small")

      stream_cptr = acc_stream_cptr(stream)
      IF (n_bytes > 0) THEN
         istat = acc_interface_memcpy_h2d(hostmem_cptr, this%cptr, &
                                          INT(n_bytes, KIND=C_SIZE_T), stream_cptr)
         IF (istat /= 0) &
            DBCSR_ABORT("acc_devmem_host2dev: failed")
      ENDIF
   END SUBROUTINE host2dev_raw
#endif

#if defined (__DBCSR_ACC)
   SUBROUTINE dev2host_raw(this, hostmem_cptr, n_bytes, stream)
      !! Helper-routine performing actual dev2host transfers.

      TYPE(acc_devmem_type), INTENT(IN)                  :: this
         !! device memory
      TYPE(C_PTR)                                        :: hostmem_cptr
         !! host memory
      INTEGER, INTENT(IN)                                :: n_bytes
         !! number of bytes
      TYPE(acc_stream_type), INTENT(IN)                  :: stream
         !! stream used for memory transfer

      CHARACTER(len=*), PARAMETER :: routineN = 'dev2host_raw', routineP = moduleN//':'//routineN

      INTEGER                                            :: istat
      TYPE(C_PTR)                                        :: stream_cptr

      IF (.NOT. acc_devmem_allocated(this)) RETURN

      IF (this%size_in_bytes < n_bytes) &
         DBCSR_ABORT("acc_devmem_dev2host: this too small")

      stream_cptr = acc_stream_cptr(stream)

      IF (n_bytes > 0) THEN
         istat = acc_interface_memcpy_d2h(this%cptr, hostmem_cptr, &
                                          INT(n_bytes, KIND=C_SIZE_T), stream_cptr)
         IF (istat /= 0) &
            DBCSR_ABORT("acc_devmem_dev2host: failed")
      ENDIF
   END SUBROUTINE dev2host_raw
#endif

#:set instances = [ ('i4', 'int_4_size',    'INTEGER(kind=int_4)'), &
                    ('i8', 'int_8_size',    'INTEGER(kind=int_8)'), &
                    ('r4', 'real_4_size',   'REAL(kind=real_4)'), &
                    ('r8', 'real_8_size',   'REAL(kind=real_8)'), &
                    ('c4', '2*real_4_size', 'COMPLEX(kind=real_4)'), &
                    ('c8', '2*real_8_size', 'COMPLEX(kind=real_8)') ]

#:for nametype, size, type in instances

   SUBROUTINE host2dev_${nametype}$_1D(this, hostmem, stream)
      !! Transfers 1D fortran-array from host to GPU devmem.

      TYPE(acc_devmem_type), INTENT(IN)        :: this
         !! device memory
      ${type}$, DIMENSION(:), POINTER          :: hostmem
         !! host memory
      TYPE(acc_stream_type), INTENT(IN)        :: stream
         !! stream

#if ! defined (__DBCSR_ACC)
      MARK_USED(this)
      MARK_USED(hostmem)
      MARK_USED(stream)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else
      CALL host2dev_raw(this, C_LOC(hostmem(1)), ${size}$*SIZE(hostmem), stream)
#endif
   END SUBROUTINE host2dev_${nametype}$_1D

   SUBROUTINE host2dev_${nametype}$_2D(this, hostmem, stream)
      !! Transfers 2D fortran-array from host to GPU devmem.

      TYPE(acc_devmem_type), INTENT(IN)        :: this
         !! device memory
      ${type}$, DIMENSION(:, :), POINTER       :: hostmem
         !! host memory
      TYPE(acc_stream_type), INTENT(IN)        :: stream
         !! stream

#if ! defined (__DBCSR_ACC)
      MARK_USED(this)
      MARK_USED(hostmem)
      MARK_USED(stream)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else
      CALL host2dev_raw(this, C_LOC(hostmem(1, 1)), ${size}$*SIZE(hostmem), stream)
#endif
   END SUBROUTINE host2dev_${nametype}$_2D

   SUBROUTINE dev2host_${nametype}$_1D(this, hostmem, stream)
      !! Transfers GPU devmem to 1D fortran-array.

      TYPE(acc_devmem_type), INTENT(IN)        :: this
         !! device memory
      ${type}$, DIMENSION(:), POINTER          :: hostmem
         !! host memory
      TYPE(acc_stream_type), INTENT(IN)        :: stream
         !! stream

#if ! defined (__DBCSR_ACC)
      MARK_USED(this)
      MARK_USED(hostmem)
      MARK_USED(stream)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else
      CALL dev2host_raw(this, C_LOC(hostmem(1)), ${size}$*SIZE(hostmem), stream)
#endif
   END SUBROUTINE dev2host_${nametype}$_1D

#:endfor

END MODULE dbcsr_acc_devmem
