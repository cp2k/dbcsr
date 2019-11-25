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
                            C_CHAR
   USE dbcsr_acc_devmem, ONLY: acc_devmem_cptr, &
                               acc_devmem_type
   USE dbcsr_acc_stream, ONLY: acc_stream_cptr, &
                               acc_stream_type, &
                               acc_stream_synchronize
   USE dbcsr_config,     ONLY: max_kernel_dim, &
                               cublas_handles, &
                               hipblas_handles
   USE dbcsr_mm_types, ONLY: dbcsr_ps_width
   USE dbcsr_kinds, ONLY: real_8, dp
   USE dbcsr_types, ONLY: dbcsr_type_real_8

#if (__DBCSR_ACC == 2)
   USE dbcsr_cublas, ONLY: cublas_dgemm_cu
#elif (__DBCSR_ACC == 3)
   USE dbcsr_hipblas, ONLY: hipblas_dgemm_hip
#endif

!$ USE OMP_LIB, ONLY: omp_get_thread_num, omp_get_num_threads

#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_acc_operations'

   LOGICAL, PARAMETER :: careful_mod = .FALSE.

   PUBLIC :: dbcsr_acc_do_mm_stack, dbcsr_acc_transpose

#if defined (__DBCSR_ACC)
   INTERFACE
      FUNCTION libsmm_acc_process_cu(param_stack, stack_size, nparams, &
                                     data_type, &
                                     a_data, b_data, c_data, m_max, n_max, k_max, def_mnk, &
                                     stream_ptr) &
         RESULT(istat) &
         BIND(C, name="libsmm_acc_process")
         IMPORT
         TYPE(C_PTR), INTENT(IN), VALUE           :: param_stack
         INTEGER(KIND=C_INT), INTENT(IN), VALUE   :: stack_size, nparams, data_type
         TYPE(C_PTR), INTENT(IN), VALUE           :: a_data, b_data, c_data
         INTEGER(KIND=C_INT), INTENT(IN), VALUE   :: m_max, n_max, k_max, def_mnk
         TYPE(C_PTR), VALUE                       :: stream_ptr
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION libsmm_acc_process_cu
   END INTERFACE

   INTERFACE
      FUNCTION libsmm_acc_transpose_cu(trs_stack, offset, nblks, buffer, data_type, m, n, stream_ptr) &
         RESULT(istat) &
         BIND(C, name="libsmm_acc_transpose")
         IMPORT
         TYPE(C_PTR), INTENT(IN), VALUE           :: trs_stack
         INTEGER(KIND=C_INT), INTENT(IN), VALUE   :: offset, nblks
         TYPE(C_PTR), INTENT(IN), VALUE           :: buffer
         INTEGER(KIND=C_INT), INTENT(IN), VALUE   :: data_type, m, n
         TYPE(C_PTR), VALUE                       :: stream_ptr
         INTEGER(KIND=C_INT)                      :: istat

      END FUNCTION libsmm_acc_transpose_cu
   END INTERFACE
#endif

CONTAINS

   SUBROUTINE dbcsr_acc_do_mm_stack(param_stack_host, param_stack_dev, stack_size, datatype, &
      !! Launch an accelerated kernel for processing a stack.
                                    a_data, b_data, c_data, m_max, n_max, k_max, def_mnk, stream, success)
      INTEGER, DIMENSION(:, :), TARGET, INTENT(IN) :: param_stack_host
      TYPE(acc_devmem_type), INTENT(IN)            :: param_stack_dev
      INTEGER, INTENT(IN)                          :: stack_size
      INTEGER, INTENT(IN)                          :: datatype
      TYPE(acc_devmem_type), INTENT(IN)            :: a_data, b_data
      TYPE(acc_devmem_type), INTENT(INOUT)         :: c_data
      INTEGER, INTENT(IN)                          :: m_max, n_max, k_max
      LOGICAL, INTENT(IN)                          :: def_mnk
      TYPE(acc_stream_type), INTENT(IN)            :: stream
      LOGICAL, INTENT(INOUT)                       :: success

#if ! defined (__DBCSR_ACC)
      MARK_USED(param_stack_host)
      MARK_USED(param_stack_dev)
      MARK_USED(stack_size)
      MARK_USED(datatype)
      MARK_USED(a_data)
      MARK_USED(b_data)
      MARK_USED(c_data)
      MARK_USED(m_max)
      MARK_USED(n_max)
      MARK_USED(k_max)
      MARK_USED(def_mnk)
      MARK_USED(stream)
      MARK_USED(success)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_acc_do_mm_stack', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                  :: error_handle, istat, ithread, istack
      INTEGER(KIND=C_INT)                      :: mnk
      CHARACTER(KIND=C_CHAR)                   :: transb

      IF (careful_mod) CALL timeset(routineN, error_handle)

      mnk = 0
      IF (def_mnk) mnk = 1
      istat = -100 ! big kernels

      ! Limits the call to small kernels
      IF (m_max .LE. max_kernel_dim .AND. &
          n_max .LE. max_kernel_dim .AND. &
          k_max .LE. max_kernel_dim) THEN
         istat = libsmm_acc_process_cu(acc_devmem_cptr(param_stack_dev), &
                                       INT(stack_size, KIND=C_INT), &
                                       INT(dbcsr_ps_width, KIND=C_INT), &
                                       INT(datatype, KIND=C_INT), &
                                       acc_devmem_cptr(a_data), &
                                       acc_devmem_cptr(b_data), &
                                       acc_devmem_cptr(c_data), &
                                       INT(m_max, KIND=C_INT), &
                                       INT(n_max, KIND=C_INT), &
                                       INT(k_max, KIND=C_INT), &
                                       mnk, acc_stream_cptr(stream))
         IF (istat == 100) DBCSR_ABORT("failed due to accGetLastError().")
      ENDIF

#if (__DBCSR_ACC == 2) || (__DBCSR_ACC == 3)
      ! Call Cublas/Hipblas for big kernels
      IF (datatype .EQ. dbcsr_type_real_8 .AND. istat .EQ. -100) THEN
         ithread = 0
!$       ithread = omp_get_thread_num()
         DO istack = 1, stack_size
            IF (param_stack_host(2, istack) .LE. max_kernel_dim .AND. &
                param_stack_host(3, istack) .LE. max_kernel_dim) THEN
               ! Transpose for B-blocks
               transb = 'T'
            ELSE
               transb = 'N'
            ENDIF
#if (__DBCSR_ACC == 2)
            istat = cublas_dgemm_cu(cublas_handles(ithread + 1)%handle_ptr, &
#else
                                    istat = hipblas_dgemm_hip(hipblas_handles(ithread + 1)%handle_ptr, &
#endif
                                                              'N', transb, &
                                                              INT(param_stack_host(1, istack), KIND=C_INT), &
                                                              INT(param_stack_host(2, istack), KIND=C_INT), &
                                                              INT(param_stack_host(3, istack), KIND=C_INT), &
                                                              INT(param_stack_host(4, istack)-1, KIND=C_INT), &
                                                              INT(param_stack_host(5, istack)-1, KIND=C_INT), &
                                                              INT(param_stack_host(6, istack)-1, KIND=C_INT), &
                                                              acc_devmem_cptr(a_data), &
                                                              acc_devmem_cptr(b_data), &
                                                              acc_devmem_cptr(c_data), &
                                                              1.0_dp, 1.0_dp, &
                                                              acc_stream_cptr(stream))
#if (__DBCSR_ACC == 2)
            IF (istat /= 0) DBCSR_ABORT("failed to run CUBLAS.")
#else
            IF (istat /= 0) DBCSR_ABORT("failed to run HIPBLAS.")
#endif
         ENDDO
         CALL acc_stream_synchronize(stream)
      ENDIF
#else
      MARK_USED(param_stack_host)
      MARK_USED(ithread)
      MARK_USED(istack)
      MARK_USED(transb)
#endif

      success = (istat == 0) ! false if no suitable kernel was found

      IF (careful_mod) CALL timestop(error_handle)
#endif
   END SUBROUTINE dbcsr_acc_do_mm_stack

   SUBROUTINE dbcsr_acc_transpose(trs_stack, offset, nblks, datatype, buffer, m, n, stream)
      !! Launch an accelerated transpose kernel
      TYPE(acc_devmem_type), INTENT(IN)        :: trs_stack
      INTEGER, INTENT(IN)                      :: offset
      INTEGER, INTENT(IN)                      :: nblks
      INTEGER, INTENT(IN)                      :: datatype
      TYPE(acc_devmem_type), INTENT(IN)        :: buffer
      INTEGER, INTENT(IN)                      :: m, n
      TYPE(acc_stream_type), INTENT(IN)        :: stream

#if ! defined (__DBCSR_ACC)
      MARK_USED(trs_stack)
      MARK_USED(offset)
      MARK_USED(nblks)
      MARK_USED(datatype)
      MARK_USED(buffer)
      MARK_USED(m)
      MARK_USED(n)
      MARK_USED(stream)
      DBCSR_ABORT("__DBCSR_ACC not compiled in.")
#else

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_acc_transpose', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                  :: error_handle, istat

      IF (careful_mod) CALL timeset(routineN, error_handle)
      istat = 0

      ! Limits the call to small kernels
      IF (m .LE. max_kernel_dim .AND. &
          n .LE. max_kernel_dim) THEN
         istat = libsmm_acc_transpose_cu(acc_devmem_cptr(trs_stack), &
                                         INT(offset, KIND=C_INT), &
                                         INT(nblks, KIND=C_INT), &
                                         acc_devmem_cptr(buffer), &
                                         INT(datatype, KIND=C_INT), &
                                         INT(m, KIND=C_INT), &
                                         INT(n, KIND=C_INT), &
                                         acc_stream_cptr(stream))
      ENDIF

      IF (istat /= 0) DBCSR_ABORT("something went wrong.")
      IF (careful_mod) CALL timestop(error_handle)
#endif
   END SUBROUTINE dbcsr_acc_transpose

END MODULE dbcsr_acc_operations
