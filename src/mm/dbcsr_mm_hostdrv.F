!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_mm_hostdrv
   !! Stacks of small matrix multiplications
   USE dbcsr_config, ONLY: dbcsr_cfg, &
                           has_acc, &
                           mm_driver_blas, &
                           mm_driver_matmul, &
                           mm_driver_smm, &
                           mm_driver_xsmm
   USE dbcsr_data_methods, ONLY: dbcsr_data_get_size
   USE dbcsr_mm_types, ONLY: dbcsr_ps_width, &
                             p_a_first, &
                             p_b_first, &
                             p_c_first, &
                             p_k, &
                             p_m, &
                             p_n, &
                             stack_descriptor_type
   USE dbcsr_types, ONLY: dbcsr_data_obj, &
                          dbcsr_type, &
                          dbcsr_type_complex_4, &
                          dbcsr_type_complex_8, &
                          dbcsr_type_real_4, &
                          dbcsr_type_real_8, &
                          dbcsr_work_type
   USE dbcsr_kinds, ONLY: dp, &
                          int_8, &
                          real_4, &
                          real_8, &
                          sp
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_mm_hostdrv'

   CHARACTER(len=*), PARAMETER, PRIVATE :: int_print = "(10(1X,I7))"

   PUBLIC :: dbcsr_mm_hostdrv_lib_init, dbcsr_mm_hostdrv_lib_finalize
   PUBLIC :: dbcsr_mm_hostdrv_process
   PUBLIC :: dbcsr_mm_hostdrv_type
   PUBLIC :: dbcsr_mm_hostdrv_init

   LOGICAL, PARAMETER :: debug_mod = .FALSE.
   LOGICAL, PARAMETER :: careful_mod = .FALSE.

   TYPE dbcsr_mm_hostdrv_type
      TYPE(dbcsr_data_obj)          :: data_area
   END TYPE dbcsr_mm_hostdrv_type

CONTAINS

   SUBROUTINE dbcsr_mm_hostdrv_lib_init()
      !! Initialize the library
   END SUBROUTINE dbcsr_mm_hostdrv_lib_init

   SUBROUTINE dbcsr_mm_hostdrv_lib_finalize()
      !! Finalize the library
   END SUBROUTINE dbcsr_mm_hostdrv_lib_finalize

   SUBROUTINE dbcsr_mm_hostdrv_init(this, product_wm)
      !! Initialize the library
      TYPE(dbcsr_mm_hostdrv_type), INTENT(INOUT)         :: this
      TYPE(dbcsr_work_type), POINTER                     :: product_wm

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_mm_hostdrv_init', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: handle

      CALL timeset(routineN, handle)

      this%data_area = product_wm%data_area

      CALL timestop(handle)

   END SUBROUTINE dbcsr_mm_hostdrv_init

   SUBROUTINE dbcsr_mm_hostdrv_process(this, left, right, params, stack_size, &
                                       stack_descr, success, used_smm)
      !! Calls the various drivers that process the stack.

      TYPE(dbcsr_mm_hostdrv_type), INTENT(INOUT)         :: this
      TYPE(dbcsr_type), INTENT(IN)                       :: left, right
         !! Left-matrix data
         !! Right-matrix data
      INTEGER, INTENT(IN)                                :: stack_size
      INTEGER, DIMENSION(1:dbcsr_ps_width, stack_size), &
         INTENT(INOUT)                                   :: params
         !! Stack of GEMM parameters
      TYPE(stack_descriptor_type), INTENT(IN)            :: stack_descr
      LOGICAL, INTENT(OUT)                               :: success, used_smm

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_mm_hostdrv_process', &
                                     routineP = moduleN//':'//routineN
      LOGICAL, PARAMETER                                 :: careful = careful_mod, dbg = .FALSE.

      INTEGER                                            :: error_handle, sp
      REAL(KIND=dp)                                      :: rnd

      IF (has_acc) & !for cpu-only runs this is called too often
         CALL timeset(routineN, error_handle)

      success = .TRUE. !host driver never fails...hopefully
      used_smm = .FALSE.

      IF (dbg) THEN
         CALL RANDOM_NUMBER(rnd)
         IF (rnd < 0.01_dp) THEN
            WRITE (*, *) routineN//" Stack size", stack_size, dbcsr_ps_width
            CALL print_gemm_parameters(params(:, 1:stack_size))
         ENDIF
      ENDIF

      ! Verify stack consistency.  Only the upper bound is verified.
      IF (careful) THEN
         DO sp = 1, stack_size
            IF (params(p_a_first, sp) + params(p_m, sp)*params(p_k, sp) - 1 > dbcsr_data_get_size(left%data_area)) &
               DBCSR_ABORT("A data out of bounds.")
            IF (params(p_b_first, sp) + params(p_k, sp)*params(p_n, sp) - 1 > dbcsr_data_get_size(right%data_area)) &
               DBCSR_ABORT("B data out of bounds.")
            IF (params(p_c_first, sp) + params(p_m, sp)*params(p_n, sp) - 1 > dbcsr_data_get_size(this%data_area)) &
               DBCSR_ABORT("C data out of bounds.")
         ENDDO
      ENDIF

      SELECT CASE (dbcsr_cfg%mm_driver)
      CASE (mm_driver_matmul)
         SELECT CASE (this%data_area%d%data_type)
         CASE (dbcsr_type_real_4)
            CALL internal_process_mm_stack_s(params, &
                                             stack_size, &
                                             left%data_area%d%r_sp, right%data_area%d%r_sp, this%data_area%d%r_sp)
         CASE (dbcsr_type_real_8)
            CALL internal_process_mm_stack_d(params, &
                                             stack_size, &
                                             left%data_area%d%r_dp, right%data_area%d%r_dp, this%data_area%d%r_dp)
         CASE (dbcsr_type_complex_4)
            CALL internal_process_mm_stack_c(params, &
                                             stack_size, &
                                             left%data_area%d%c_sp, right%data_area%d%c_sp, this%data_area%d%c_sp)
         CASE (dbcsr_type_complex_8)
            CALL internal_process_mm_stack_z(params, &
                                             stack_size, &
                                             left%data_area%d%c_dp, right%data_area%d%c_dp, this%data_area%d%c_dp)
         CASE default
            DBCSR_ABORT("Invalid data type")
         END SELECT
      CASE (mm_driver_smm)
         SELECT CASE (this%data_area%d%data_type)
         CASE (dbcsr_type_real_4)
            CALL smm_process_mm_stack_s(stack_descr, params, &
                                        stack_size, &
                                        left%data_area%d%r_sp, right%data_area%d%r_sp, this%data_area%d%r_sp, used_smm)
         CASE (dbcsr_type_real_8)
            CALL smm_process_mm_stack_d(stack_descr, params, &
                                        stack_size, &
                                        left%data_area%d%r_dp, right%data_area%d%r_dp, this%data_area%d%r_dp, used_smm)
         CASE (dbcsr_type_complex_4)
            CALL smm_process_mm_stack_c(stack_descr, params, &
                                        stack_size, &
                                        left%data_area%d%c_sp, right%data_area%d%c_sp, this%data_area%d%c_sp, used_smm)
         CASE (dbcsr_type_complex_8)
            CALL smm_process_mm_stack_z(stack_descr, params, &
                                        stack_size, &
                                        left%data_area%d%c_dp, right%data_area%d%c_dp, this%data_area%d%c_dp, used_smm)
         CASE default
            DBCSR_ABORT("Invalid data type")
         END SELECT
#if defined(__LIBXSMM)
      CASE (mm_driver_xsmm)
         SELECT CASE (this%data_area%d%data_type)
#if TO_VERSION(1, 10, 0) < TO_VERSION(LIBXSMM_CONFIG_VERSION_MAJOR, LIBXSMM_CONFIG_VERSION_MINOR, LIBXSMM_CONFIG_VERSION_UPDATE)
         CASE (dbcsr_type_real_4)
            CALL xsmm_process_mm_batch_s(stack_descr, params, stack_size, &
                                         left%data_area%d%r_sp, right%data_area%d%r_sp, this%data_area%d%r_sp, used_smm)
         CASE (dbcsr_type_real_8)
            CALL xsmm_process_mm_batch_d(stack_descr, params, stack_size, &
                                         left%data_area%d%r_dp, right%data_area%d%r_dp, this%data_area%d%r_dp, used_smm)
         CASE (dbcsr_type_complex_4)
            CALL xsmm_process_mm_batch_c(stack_descr, params, stack_size, &
                                         left%data_area%d%c_sp, right%data_area%d%c_sp, this%data_area%d%c_sp, used_smm)
         CASE (dbcsr_type_complex_8)
            CALL xsmm_process_mm_batch_z(stack_descr, params, stack_size, &
                                         left%data_area%d%c_dp, right%data_area%d%c_dp, this%data_area%d%c_dp, used_smm)
#else
         CASE (dbcsr_type_real_4)
            CALL xsmm_process_mm_stack_s(stack_descr, params, stack_size, &
                                         left%data_area%d%r_sp, right%data_area%d%r_sp, this%data_area%d%r_sp, used_smm)
         CASE (dbcsr_type_real_8)
            CALL xsmm_process_mm_stack_d(stack_descr, params, stack_size, &
                                         left%data_area%d%r_dp, right%data_area%d%r_dp, this%data_area%d%r_dp, used_smm)
         CASE (dbcsr_type_complex_4)
            CALL xsmm_process_mm_stack_c(stack_descr, params, stack_size, &
                                         left%data_area%d%c_sp, right%data_area%d%c_sp, this%data_area%d%c_sp, used_smm)
         CASE (dbcsr_type_complex_8)
            CALL xsmm_process_mm_stack_z(stack_descr, params, stack_size, &
                                         left%data_area%d%c_dp, right%data_area%d%c_dp, this%data_area%d%c_dp, used_smm)
#endif
         CASE default
            DBCSR_ABORT("Invalid data type")
         END SELECT
#endif
      CASE (mm_driver_blas)
         SELECT CASE (this%data_area%d%data_type)
         CASE (dbcsr_type_real_4)
            CALL blas_process_mm_stack_s(params, &
                                         stack_size, &
                                         left%data_area%d%r_sp, right%data_area%d%r_sp, this%data_area%d%r_sp)
         CASE (dbcsr_type_real_8)
            CALL blas_process_mm_stack_d(params, &
                                         stack_size, &
                                         left%data_area%d%r_dp, right%data_area%d%r_dp, this%data_area%d%r_dp)
         CASE (dbcsr_type_complex_4)
            CALL blas_process_mm_stack_c(params, &
                                         stack_size, &
                                         left%data_area%d%c_sp, right%data_area%d%c_sp, this%data_area%d%c_sp)
         CASE (dbcsr_type_complex_8)
            CALL blas_process_mm_stack_z(params, &
                                         stack_size, &
                                         left%data_area%d%c_dp, right%data_area%d%c_dp, this%data_area%d%c_dp)
         CASE default
            DBCSR_ABORT("Invalid data type")
         END SELECT
      CASE default
         DBCSR_ABORT("Invalid multiplication driver")
      END SELECT

      IF (has_acc) & !for cpu-only runs this is called too often
         CALL timestop(error_handle)

   END SUBROUTINE dbcsr_mm_hostdrv_process

   SUBROUTINE print_gemm_parameters(params)
      !! Helper-routine used by dbcsr_mm_hostdrv_process to print debug info.
      INTEGER, DIMENSION(:, :), INTENT(in)               :: params

      INTEGER                                            :: sp

      DO sp = 1, SIZE(params, 2)
         WRITE (*, '(1X,A,1X,I7,":",3(1X,I4,","),".",3(1X,I12,","))') &
            "GEMM PARAMETERS", &
            sp, &
            params(p_m, sp), &
            params(p_k, sp), &
            params(p_n, sp), &
            params(p_a_first, sp), &
            params(p_b_first, sp), &
            params(p_c_first, sp)
      ENDDO
   END SUBROUTINE print_gemm_parameters

#:include "dbcsr_mm_hostdrv.f90"

END MODULE dbcsr_mm_hostdrv
