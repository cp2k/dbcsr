!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_acc_operations
   !! Accelerator support for DBCSR
   USE ISO_C_BINDING, ONLY: C_INT, &
                            C_PTR, &
                            C_CHAR, &
                            C_LOC
   USE dbcsr_acc_devmem, ONLY: acc_devmem_cptr, &
                               acc_devmem_type
   USE dbcsr_acc_stream, ONLY: acc_stream_cptr, &
                               acc_stream_type, &
                               acc_stream_synchronize
   USE dbcsr_config, ONLY: max_kernel_dim
   USE dbcsr_mm_types, ONLY: dbcsr_ps_width
   USE dbcsr_kinds, ONLY: real_8, dp
   USE dbcsr_types, ONLY: dbcsr_type_real_8

#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   LOGICAL, PARAMETER :: careful_mod = .FALSE.

   PUBLIC :: dbcsr_acc_do_mm_stack, dbcsr_acc_transpose

#if defined (__DBCSR_ACC)
   INTERFACE
      FUNCTION libsmm_acc_process_cu(param_stack_host, param_stack_dev, stack_size, &
                                     data_type, a_data, b_data, c_data, m_max, &
                                     n_max, k_max, max_kernel_dim, def_mnk, &
                                     stack_stream_ptr, c_stream_ptr) &
         RESULT(istat) &
         BIND(C, name="libsmm_acc_process")
         IMPORT
         TYPE(C_PTR), INTENT(IN), VALUE           :: param_stack_host
         TYPE(C_PTR), INTENT(IN), VALUE           :: param_stack_dev
         INTEGER(KIND=C_INT), INTENT(IN), VALUE   :: stack_size, data_type
         TYPE(C_PTR), INTENT(IN), VALUE           :: a_data, b_data, c_data
         INTEGER(KIND=C_INT), INTENT(IN), VALUE   :: m_max, n_max, k_max
         INTEGER(KIND=C_INT), INTENT(IN), VALUE   :: max_kernel_dim, def_mnk
         TYPE(C_PTR), VALUE                       :: stack_stream_ptr, c_stream_ptr
         INTEGER(KIND=C_INT)                      :: istat
      END FUNCTION libsmm_acc_process_cu

      FUNCTION libsmm_acc_transpose_cu(trs_stack, offset, nblks, buffer, &
                                       data_type, m, n, max_kernel_dim, stream_ptr) &
         RESULT(istat) &
         BIND(C, name="libsmm_acc_transpose")
         IMPORT
         TYPE(C_PTR), INTENT(IN), VALUE           :: trs_stack
         INTEGER(KIND=C_INT), INTENT(IN), VALUE   :: offset, nblks
         TYPE(C_PTR), INTENT(IN), VALUE           :: buffer
         INTEGER(KIND=C_INT), INTENT(IN), VALUE   :: data_type, m, n
         INTEGER(KIND=C_INT), INTENT(IN), VALUE   :: max_kernel_dim
         TYPE(C_PTR), VALUE                       :: stream_ptr
         INTEGER(KIND=C_INT)                      :: istat
      END FUNCTION libsmm_acc_transpose_cu

   END INTERFACE
#endif

CONTAINS

   SUBROUTINE dbcsr_acc_do_mm_stack(param_stack_host, param_stack_dev, stack_size, data_type, &
                                    a_data, b_data, c_data, m_max, n_max, k_max, def_mnk, &
                                    stack_stream, c_stream, success, generated_acc_untuned)
      !! Launch an accelerated kernel for processing a stack.
      INTEGER, DIMENSION(:, :), TARGET, INTENT(IN) :: param_stack_host
      TYPE(acc_devmem_type), INTENT(IN)            :: param_stack_dev
      INTEGER, INTENT(IN)                          :: stack_size
      INTEGER, INTENT(IN)                          :: data_type
      TYPE(acc_devmem_type), INTENT(IN)            :: a_data, b_data
      TYPE(acc_devmem_type), INTENT(INOUT)         :: c_data
      INTEGER, INTENT(IN)                          :: m_max, n_max, k_max
      LOGICAL, INTENT(IN)                          :: def_mnk
      TYPE(acc_stream_type), INTENT(IN)            :: stack_stream, c_stream
      LOGICAL, INTENT(INOUT)                       :: success, generated_acc_untuned
#if ! defined (__DBCSR_ACC)
      MARK_USED(param_stack_host)
      MARK_USED(param_stack_dev)
      MARK_USED(stack_size)
      MARK_USED(data_type)
      MARK_USED(a_data)
      MARK_USED(b_data)
      MARK_USED(c_data)
      MARK_USED(m_max)
      MARK_USED(n_max)
      MARK_USED(k_max)
      MARK_USED(def_mnk)
      MARK_USED(stack_stream)
      MARK_USED(c_stream)
      MARK_USED(success)
      MARK_USED(generated_acc_untuned)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else
      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_acc_do_mm_stack'

      INTEGER                                  :: error_handle, istat
      INTEGER(KIND=C_INT)                      :: mnk
      INTEGER, DIMENSION(:, :), POINTER         :: param_stack_host_ptr

      param_stack_host_ptr => param_stack_host(:, :)

      IF (careful_mod) CALL timeset(routineN, error_handle)

      mnk = 0
      IF (def_mnk) mnk = 1

      ! Call batched matrix-matrix multiplication in libsmm_acc
      istat = libsmm_acc_process_cu(C_LOC(param_stack_host_ptr), &
                                    acc_devmem_cptr(param_stack_dev), &
                                    INT(stack_size, KIND=C_INT), &
                                    INT(data_type, KIND=C_INT), &
                                    acc_devmem_cptr(a_data), &
                                    acc_devmem_cptr(b_data), &
                                    acc_devmem_cptr(c_data), &
                                    INT(m_max, KIND=C_INT), &
                                    INT(n_max, KIND=C_INT), &
                                    INT(k_max, KIND=C_INT), &
                                    INT(max_kernel_dim, KIND=C_INT), &
                                    mnk, acc_stream_cptr(stack_stream), acc_stream_cptr(c_stream))
!      IF (istat == -10) DBCSR_ABORT("Data type not supported with GPU backend.")
!      IF (istat == -20) DBCSR_ABORT("GPU kernel not JIT-ed.")
      success = (istat .GE. 0) ! false if no suitable kernel was found
      generated_acc_untuned = (istat == 10) ! Generated default untuned kernel

      IF (careful_mod) CALL timestop(error_handle)
#endif
   END SUBROUTINE dbcsr_acc_do_mm_stack

   SUBROUTINE dbcsr_acc_transpose(trs_stack, offset, nblks, data_type, buffer, m, n, stream)
      !! Launch an accelerated transpose kernel
      TYPE(acc_devmem_type), INTENT(IN)        :: trs_stack
      INTEGER, INTENT(IN)                      :: offset
      INTEGER, INTENT(IN)                      :: nblks
      INTEGER, INTENT(IN)                      :: data_type
      TYPE(acc_devmem_type), INTENT(IN)        :: buffer
      INTEGER, INTENT(IN)                      :: m, n
      TYPE(acc_stream_type), INTENT(IN)        :: stream
#if ! defined (__DBCSR_ACC)
      MARK_USED(trs_stack)
      MARK_USED(offset)
      MARK_USED(nblks)
      MARK_USED(data_type)
      MARK_USED(buffer)
      MARK_USED(m)
      MARK_USED(n)
      MARK_USED(stream)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else
      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_acc_transpose'

      INTEGER                                  :: error_handle, istat

      IF (careful_mod) CALL timeset(routineN, error_handle)
      istat = 0

      ! Call batched in-place transpose in libsmm_acc
      IF (m .LE. max_kernel_dim .AND. &
          n .LE. max_kernel_dim) THEN
         istat = libsmm_acc_transpose_cu(acc_devmem_cptr(trs_stack), &
                                         INT(offset, KIND=C_INT), &
                                         INT(nblks, KIND=C_INT), &
                                         acc_devmem_cptr(buffer), &
                                         INT(data_type, KIND=C_INT), &
                                         INT(m, KIND=C_INT), &
                                         INT(n, KIND=C_INT), &
                                         INT(max_kernel_dim, KIND=C_INT), &
                                         acc_stream_cptr(stream))
      END IF

      IF (istat /= 0) DBCSR_ABORT("something went wrong.")
      IF (careful_mod) CALL timestop(error_handle)
#endif
   END SUBROUTINE dbcsr_acc_transpose

END MODULE dbcsr_acc_operations
