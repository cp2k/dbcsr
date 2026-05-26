# 1 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F" 1
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
                           use_acc, &
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
      TYPE(dbcsr_data_obj)          :: data_area = dbcsr_data_obj()
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

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_mm_hostdrv_init'

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

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_mm_hostdrv_process'
      LOGICAL, PARAMETER                                 :: careful = careful_mod, dbg = debug_mod

      INTEGER                                            :: error_handle, sp
      REAL(KIND=dp)                                      :: rnd

      IF (use_acc()) & !for cpu-only runs this is called too often
         CALL timeset(routineN, error_handle)

      success = .TRUE. !host driver never fails...hopefully
      used_smm = .FALSE.

      IF (dbg) THEN
         CALL RANDOM_NUMBER(rnd)
         IF (rnd < 0.01_dp) THEN
            WRITE (*, *) routineN//" Stack size", stack_size, dbcsr_ps_width
            CALL print_gemm_parameters(params(:, 1:stack_size))
         END IF
      END IF

      ! Verify stack consistency.  Only the upper bound is verified.
      IF (careful) THEN
         DO sp = 1, stack_size
            IF (params(p_a_first, sp) + params(p_m, sp)*params(p_k, sp) - 1 > dbcsr_data_get_size(left%data_area)) &
               DBCSR_ABORT("A data out of bounds.")
            IF (params(p_b_first, sp) + params(p_k, sp)*params(p_n, sp) - 1 > dbcsr_data_get_size(right%data_area)) &
               DBCSR_ABORT("B data out of bounds.")
            IF (params(p_c_first, sp) + params(p_m, sp)*params(p_n, sp) - 1 > dbcsr_data_get_size(this%data_area)) &
               DBCSR_ABORT("C data out of bounds.")
         END DO
      END IF

      SELECT CASE (dbcsr_cfg%mm_driver%val)
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
#if defined(__LIBXS)
      CASE (mm_driver_xsmm)
         SELECT CASE (this%data_area%d%data_type)
         CASE (dbcsr_type_real_4)
            CALL libxs_process_mm_stack_s(stack_descr, params, stack_size, &
                                          left%data_area%d%r_sp, right%data_area%d%r_sp, this%data_area%d%r_sp, used_smm)
         CASE (dbcsr_type_real_8)
            CALL libxs_process_mm_stack_d(stack_descr, params, stack_size, &
                                          left%data_area%d%r_dp, right%data_area%d%r_dp, this%data_area%d%r_dp, used_smm)
         CASE (dbcsr_type_complex_4)
            CALL libxs_process_mm_stack_c(stack_descr, params, stack_size, &
                                          left%data_area%d%c_sp, right%data_area%d%c_sp, this%data_area%d%c_sp, used_smm)
         CASE (dbcsr_type_complex_8)
            CALL libxs_process_mm_stack_z(stack_descr, params, stack_size, &
                                          left%data_area%d%c_dp, right%data_area%d%c_dp, this%data_area%d%c_dp, used_smm)
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

      IF (use_acc()) & !for cpu-only runs this is called too often
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
      END DO
   END SUBROUTINE print_gemm_parameters

# 1 "/__w/dbcsr/dbcsr/src/mm/../data/dbcsr.fypp" 1
# 9 "/__w/dbcsr/dbcsr/src/mm/../data/dbcsr.fypp"

# 11 "/__w/dbcsr/dbcsr/src/mm/../data/dbcsr.fypp"

# 169 "/__w/dbcsr/dbcsr/src/mm/../data/dbcsr.fypp"
# 247 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F" 2
# 248 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
      SUBROUTINE blas_process_mm_stack_d (params, &
                                                      stack_size, &
                                                      a_data, b_data, c_data)
     !! Processes MM stack and issues BLAS xGEMM calls

         INTEGER, INTENT(IN)                       :: stack_size
        !! Number of parameters
         INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
            INTENT(IN)                              :: params
        !! Stack of MM parameters
         REAL(kind=real_8), DIMENSION(*), INTENT(IN)         :: a_data, &
            b_data
        !! Left-matrix data
        !! Right-matrix data
         REAL(kind=real_8), DIMENSION(*), INTENT(INOUT)      :: c_data
        !! Product data

         INTEGER                                   :: sp

!   ---------------------------------------------------------------------------

         DO sp = 1, stack_size
            CALL DGEMM ('N', &
                                   'N', &
                                   params(p_m, sp), params(p_n, sp), & !m, n
                                   params(p_k, sp), & ! k
                                   1.0_real_8, & ! alpha
                                   a_data(params(p_a_first, sp)), & ! A
                                   params(p_m, sp), & !lda
                                   b_data(params(p_b_first, sp)), & ! B
                                   params(p_k, sp), & !ldb
                                   1.0_real_8, & ! beta
                                   c_data(params(p_c_first, sp)), params(p_m, sp))
         END DO
      END SUBROUTINE blas_process_mm_stack_d

      SUBROUTINE internal_process_mm_stack_d (params, stack_size, &
                                                          a_data, b_data, c_data)
     !! Processes MM stack and issues internal MM calls.

         INTEGER, INTENT(IN)                       :: stack_size
        !! Number of parameters
         INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
            INTENT(IN)                              :: params
        !! Stack of MM parameters
         REAL(kind=real_8), DIMENSION(*), INTENT(IN)         :: a_data, &
            b_data
        !! Left-matrix data
        !! Right-matrix data
         REAL(kind=real_8), DIMENSION(*), INTENT(INOUT)      :: c_data
        !! Product data

         INTEGER                                   :: sp

!   ---------------------------------------------------------------------------

         DO sp = 1, stack_size
            CALL internal_mm_d_nn( &
               params(p_m, sp), &
               params(p_n, sp), &
               params(p_k, sp), &
               a_data(params(p_a_first, sp)), &
               b_data(params(p_b_first, sp)), &
               c_data(params(p_c_first, sp)))
         END DO
      END SUBROUTINE internal_process_mm_stack_d

      SUBROUTINE smm_process_mm_stack_d (stack_descr, params, &
                                                     stack_size, &
                                                     a_data, b_data, c_data, used_smm)
     !! Processes MM stack and issues SMM library calls

         INTEGER, INTENT(IN)                       :: stack_size
        !! Number of parameters
         TYPE(stack_descriptor_type), INTENT(IN)   :: stack_descr
         INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
            INTENT(IN)                              :: params
        !! Stack of MM parameters
         REAL(kind=real_8), DIMENSION(*), INTENT(IN)         :: a_data, &
            b_data
        !! Left-matrix data
        !! Right-matrix data
         REAL(kind=real_8), DIMENSION(*), INTENT(INOUT)      :: c_data
        !! Product data
         LOGICAL, INTENT(OUT)                      :: used_smm

#if defined(__HAS_smm_dnn)

         INTEGER                                   :: sp

         ! TODO we have no way of knowing which calls to libsmm actually resolve to BLAS
         ! Fixing this requires an interface change to libsmm.
         used_smm = .TRUE.

#if defined(__HAS_smm_vec)
         IF (stack_descr%defined_mnk) THEN
            CALL smm_vec_dnn(stack_descr%m, stack_descr%n, stack_descr%k, &
                                         a_data, b_data, c_data, stack_size, &
                                         dbcsr_ps_width, params, p_a_first, p_b_first, p_c_first)
            RETURN
         END IF
#endif

         DO sp = 1, stack_size
            CALL smm_dnn( &
               params(p_m, sp), &
               params(p_n, sp), &
               params(p_k, sp), &
               a_data(params(p_a_first, sp)), &
               b_data(params(p_b_first, sp)), &
               c_data(params(p_c_first, sp)))
         END DO

#else
         ! We do not want to abort here, fall back to BLAS.
         used_smm = .FALSE.
         CALL blas_process_mm_stack_d (params, stack_size, a_data, b_data, c_data)
#endif

         MARK_USED(stack_descr)
      END SUBROUTINE smm_process_mm_stack_d

#if defined(__LIBXS)
      SUBROUTINE libxs_process_mm_stack_d (stack_descr, params, &
                                                       stack_size, a_data, b_data, c_data, used_smm)
     !! Processes MM stack using LIBXS indexed GEMM batch.
     !! Real types use LIBXS dispatch; complex types fall through to BLAS.
# 376 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
            USE LIBXS, ONLY: libxs_gemm_config_t, &
                             libxs_gemm_dispatch, libxs_gemm_index, &
                             C_LOC, C_SIZEOF, &
                             LIBXS_DATATYPE_F64
            USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_FUNLOC
# 382 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
#if defined(__MKL)
            INTERFACE
               FUNCTION mkl_cblas_jit_create_dgemm(jitter, &
                                                               layout, transa, transb, m, n, k, &
                                                               alpha, lda, ldb, beta, ldc) &
                  RESULT(status) BIND(C)
                  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR, C_INT, C_DOUBLE
                  INTEGER(C_INT) :: status
                  TYPE(C_PTR) :: jitter
                  INTEGER(C_INT), VALUE :: layout, transa, transb
                  INTEGER(C_INT), VALUE :: m, n, k, lda, ldb, ldc
                  REAL(C_DOUBLE), VALUE :: alpha, beta
               END FUNCTION
               FUNCTION mkl_jit_get_dgemm_ptr(jitter) &
                  RESULT(ptr) BIND(C)
                  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_FUNPTR, C_PTR
                  TYPE(C_FUNPTR) :: ptr
                  TYPE(C_PTR), INTENT(IN), VALUE :: jitter
               END FUNCTION
            END INTERFACE
#endif
#if defined(__LIBXSMM)
            INTERFACE
               FUNCTION libxsmm_dispatch_gemm(shape, flags, prefetch) &
                  RESULT(fn) BIND(C)
                  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_FUNPTR, C_INT
                  INTEGER(C_INT), INTENT(IN) :: shape(10)
                  INTEGER(C_INT), INTENT(IN), VALUE :: flags, prefetch
                  TYPE(C_FUNPTR) :: fn
               END FUNCTION
            END INTERFACE
#endif
#if defined(__BLAS) || defined(__MKL)
            INTERFACE
               SUBROUTINE dgemm_blas(transa, transb, &
                                                 m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                  BIND(C, NAME="dgemm_")
                  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT, C_CHAR, C_DOUBLE
                  CHARACTER(1, C_CHAR), INTENT(IN) :: transa, transb
                  INTEGER(C_INT), INTENT(IN) :: m, n, k, lda, ldb, ldc
                  REAL(C_DOUBLE), INTENT(IN) :: alpha, beta, a(*), b(*)
                  REAL(C_DOUBLE), INTENT(INOUT) :: c(*)
               END SUBROUTINE
            END INTERFACE
#endif
# 428 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
         INTEGER, INTENT(IN)                            :: stack_size
         TYPE(stack_descriptor_type), INTENT(IN)        :: stack_descr
         INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
            INTENT(IN), TARGET                          :: params
         REAL(kind=real_8), DIMENSION(*), TARGET, INTENT(IN)    :: a_data
         REAL(kind=real_8), DIMENSION(*), TARGET, INTENT(IN)    :: b_data
         REAL(kind=real_8), DIMENSION(*), TARGET, INTENT(INOUT) :: c_data
         LOGICAL, INTENT(OUT)                           :: used_smm

# 438 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
            TYPE(libxs_gemm_config_t) :: config
            INTEGER :: m, n, k, rc
# 441 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"

         used_smm = .FALSE.
# 446 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
# 447 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
            IF (stack_descr%defined_mnk) THEN
               m = stack_descr%m; n = stack_descr%n; k = stack_descr%k
               rc = libxs_gemm_dispatch(config, &
                                        datatype=LIBXS_DATATYPE_F64, &
                                        transa='N', transb='N', &
                                        m=m, n=n, k=k, lda=m, ldb=k, ldc=m, &
                                        alpha=1D0, beta=1D0 &
#if defined(__MKL)
                                        , jit_create_dgemm= &
                                        C_FUNLOC(mkl_cblas_jit_create_dgemm) &
                                        , jit_get_dgemm= &
                                        C_FUNLOC(mkl_jit_get_dgemm_ptr) &
#endif
#if defined(__LIBXSMM)
                                        , xgemm_dispatch= &
                                        C_FUNLOC(libxsmm_dispatch_gemm) &
#endif
#if defined(__BLAS) || defined(__MKL)
                                        , dgemm_blas= &
                                        C_FUNLOC(dgemm_blas) &
#endif
                                        )
               IF (0 /= rc) THEN
                  CALL libxs_gemm_index( &
                     C_LOC(a_data), C_LOC(params(p_a_first, 1)), &
                     C_LOC(b_data), C_LOC(params(p_b_first, 1)), &
                     C_LOC(c_data), C_LOC(params(p_c_first, 1)), &
                     INT(C_SIZEOF(params(1, 1)))*dbcsr_ps_width, 1, &
                     stack_size, config)
                  used_smm = .TRUE.
               END IF
            END IF
# 480 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
         IF (.NOT. used_smm) THEN
            CALL blas_process_mm_stack_d (params, stack_size, &
                                                      a_data, b_data, c_data)
         END IF
      END SUBROUTINE libxs_process_mm_stack_d
#endif

      PURE SUBROUTINE internal_mm_d_nn( &
         M, N, K, A, B, C)
         INTEGER, INTENT(IN)                      :: M, N, K
         REAL(kind=real_8), INTENT(INOUT)                   :: C(M, N)
         REAL(kind=real_8), INTENT(IN)                      :: B(K, N)
         REAL(kind=real_8), INTENT(IN)                      :: A(M, K)
         C(:, :) = C(:, :) + MATMUL(A, B)
      END SUBROUTINE internal_mm_d_nn
# 248 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
      SUBROUTINE blas_process_mm_stack_s (params, &
                                                      stack_size, &
                                                      a_data, b_data, c_data)
     !! Processes MM stack and issues BLAS xGEMM calls

         INTEGER, INTENT(IN)                       :: stack_size
        !! Number of parameters
         INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
            INTENT(IN)                              :: params
        !! Stack of MM parameters
         REAL(kind=real_4), DIMENSION(*), INTENT(IN)         :: a_data, &
            b_data
        !! Left-matrix data
        !! Right-matrix data
         REAL(kind=real_4), DIMENSION(*), INTENT(INOUT)      :: c_data
        !! Product data

         INTEGER                                   :: sp

!   ---------------------------------------------------------------------------

         DO sp = 1, stack_size
            CALL SGEMM ('N', &
                                   'N', &
                                   params(p_m, sp), params(p_n, sp), & !m, n
                                   params(p_k, sp), & ! k
                                   1.0_real_4, & ! alpha
                                   a_data(params(p_a_first, sp)), & ! A
                                   params(p_m, sp), & !lda
                                   b_data(params(p_b_first, sp)), & ! B
                                   params(p_k, sp), & !ldb
                                   1.0_real_4, & ! beta
                                   c_data(params(p_c_first, sp)), params(p_m, sp))
         END DO
      END SUBROUTINE blas_process_mm_stack_s

      SUBROUTINE internal_process_mm_stack_s (params, stack_size, &
                                                          a_data, b_data, c_data)
     !! Processes MM stack and issues internal MM calls.

         INTEGER, INTENT(IN)                       :: stack_size
        !! Number of parameters
         INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
            INTENT(IN)                              :: params
        !! Stack of MM parameters
         REAL(kind=real_4), DIMENSION(*), INTENT(IN)         :: a_data, &
            b_data
        !! Left-matrix data
        !! Right-matrix data
         REAL(kind=real_4), DIMENSION(*), INTENT(INOUT)      :: c_data
        !! Product data

         INTEGER                                   :: sp

!   ---------------------------------------------------------------------------

         DO sp = 1, stack_size
            CALL internal_mm_s_nn( &
               params(p_m, sp), &
               params(p_n, sp), &
               params(p_k, sp), &
               a_data(params(p_a_first, sp)), &
               b_data(params(p_b_first, sp)), &
               c_data(params(p_c_first, sp)))
         END DO
      END SUBROUTINE internal_process_mm_stack_s

      SUBROUTINE smm_process_mm_stack_s (stack_descr, params, &
                                                     stack_size, &
                                                     a_data, b_data, c_data, used_smm)
     !! Processes MM stack and issues SMM library calls

         INTEGER, INTENT(IN)                       :: stack_size
        !! Number of parameters
         TYPE(stack_descriptor_type), INTENT(IN)   :: stack_descr
         INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
            INTENT(IN)                              :: params
        !! Stack of MM parameters
         REAL(kind=real_4), DIMENSION(*), INTENT(IN)         :: a_data, &
            b_data
        !! Left-matrix data
        !! Right-matrix data
         REAL(kind=real_4), DIMENSION(*), INTENT(INOUT)      :: c_data
        !! Product data
         LOGICAL, INTENT(OUT)                      :: used_smm

#if defined(__HAS_smm_snn)

         INTEGER                                   :: sp

         ! TODO we have no way of knowing which calls to libsmm actually resolve to BLAS
         ! Fixing this requires an interface change to libsmm.
         used_smm = .TRUE.

#if defined(__HAS_smm_vec)
         IF (stack_descr%defined_mnk) THEN
            CALL smm_vec_snn(stack_descr%m, stack_descr%n, stack_descr%k, &
                                         a_data, b_data, c_data, stack_size, &
                                         dbcsr_ps_width, params, p_a_first, p_b_first, p_c_first)
            RETURN
         END IF
#endif

         DO sp = 1, stack_size
            CALL smm_snn( &
               params(p_m, sp), &
               params(p_n, sp), &
               params(p_k, sp), &
               a_data(params(p_a_first, sp)), &
               b_data(params(p_b_first, sp)), &
               c_data(params(p_c_first, sp)))
         END DO

#else
         ! We do not want to abort here, fall back to BLAS.
         used_smm = .FALSE.
         CALL blas_process_mm_stack_s (params, stack_size, a_data, b_data, c_data)
#endif

         MARK_USED(stack_descr)
      END SUBROUTINE smm_process_mm_stack_s

#if defined(__LIBXS)
      SUBROUTINE libxs_process_mm_stack_s (stack_descr, params, &
                                                       stack_size, a_data, b_data, c_data, used_smm)
     !! Processes MM stack using LIBXS indexed GEMM batch.
     !! Real types use LIBXS dispatch; complex types fall through to BLAS.
# 376 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
            USE LIBXS, ONLY: libxs_gemm_config_t, &
                             libxs_gemm_dispatch, libxs_gemm_index, &
                             C_LOC, C_SIZEOF, &
                             LIBXS_DATATYPE_F32
            USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_FUNLOC
# 382 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
#if defined(__MKL)
            INTERFACE
               FUNCTION mkl_cblas_jit_create_sgemm(jitter, &
                                                               layout, transa, transb, m, n, k, &
                                                               alpha, lda, ldb, beta, ldc) &
                  RESULT(status) BIND(C)
                  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR, C_INT, C_FLOAT
                  INTEGER(C_INT) :: status
                  TYPE(C_PTR) :: jitter
                  INTEGER(C_INT), VALUE :: layout, transa, transb
                  INTEGER(C_INT), VALUE :: m, n, k, lda, ldb, ldc
                  REAL(C_FLOAT), VALUE :: alpha, beta
               END FUNCTION
               FUNCTION mkl_jit_get_sgemm_ptr(jitter) &
                  RESULT(ptr) BIND(C)
                  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_FUNPTR, C_PTR
                  TYPE(C_FUNPTR) :: ptr
                  TYPE(C_PTR), INTENT(IN), VALUE :: jitter
               END FUNCTION
            END INTERFACE
#endif
#if defined(__LIBXSMM)
            INTERFACE
               FUNCTION libxsmm_dispatch_gemm(shape, flags, prefetch) &
                  RESULT(fn) BIND(C)
                  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_FUNPTR, C_INT
                  INTEGER(C_INT), INTENT(IN) :: shape(10)
                  INTEGER(C_INT), INTENT(IN), VALUE :: flags, prefetch
                  TYPE(C_FUNPTR) :: fn
               END FUNCTION
            END INTERFACE
#endif
#if defined(__BLAS) || defined(__MKL)
            INTERFACE
               SUBROUTINE sgemm_blas(transa, transb, &
                                                 m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                  BIND(C, NAME="sgemm_")
                  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT, C_CHAR, C_FLOAT
                  CHARACTER(1, C_CHAR), INTENT(IN) :: transa, transb
                  INTEGER(C_INT), INTENT(IN) :: m, n, k, lda, ldb, ldc
                  REAL(C_FLOAT), INTENT(IN) :: alpha, beta, a(*), b(*)
                  REAL(C_FLOAT), INTENT(INOUT) :: c(*)
               END SUBROUTINE
            END INTERFACE
#endif
# 428 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
         INTEGER, INTENT(IN)                            :: stack_size
         TYPE(stack_descriptor_type), INTENT(IN)        :: stack_descr
         INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
            INTENT(IN), TARGET                          :: params
         REAL(kind=real_4), DIMENSION(*), TARGET, INTENT(IN)    :: a_data
         REAL(kind=real_4), DIMENSION(*), TARGET, INTENT(IN)    :: b_data
         REAL(kind=real_4), DIMENSION(*), TARGET, INTENT(INOUT) :: c_data
         LOGICAL, INTENT(OUT)                           :: used_smm

# 438 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
            TYPE(libxs_gemm_config_t) :: config
            INTEGER :: m, n, k, rc
# 441 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"

         used_smm = .FALSE.
# 446 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
# 447 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
            IF (stack_descr%defined_mnk) THEN
               m = stack_descr%m; n = stack_descr%n; k = stack_descr%k
               rc = libxs_gemm_dispatch(config, &
                                        datatype=LIBXS_DATATYPE_F32, &
                                        transa='N', transb='N', &
                                        m=m, n=n, k=k, lda=m, ldb=k, ldc=m, &
                                        alpha=1D0, beta=1D0 &
#if defined(__MKL)
                                        , jit_create_sgemm= &
                                        C_FUNLOC(mkl_cblas_jit_create_sgemm) &
                                        , jit_get_sgemm= &
                                        C_FUNLOC(mkl_jit_get_sgemm_ptr) &
#endif
#if defined(__LIBXSMM)
                                        , xgemm_dispatch= &
                                        C_FUNLOC(libxsmm_dispatch_gemm) &
#endif
#if defined(__BLAS) || defined(__MKL)
                                        , sgemm_blas= &
                                        C_FUNLOC(sgemm_blas) &
#endif
                                        )
               IF (0 /= rc) THEN
                  CALL libxs_gemm_index( &
                     C_LOC(a_data), C_LOC(params(p_a_first, 1)), &
                     C_LOC(b_data), C_LOC(params(p_b_first, 1)), &
                     C_LOC(c_data), C_LOC(params(p_c_first, 1)), &
                     INT(C_SIZEOF(params(1, 1)))*dbcsr_ps_width, 1, &
                     stack_size, config)
                  used_smm = .TRUE.
               END IF
            END IF
# 480 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
         IF (.NOT. used_smm) THEN
            CALL blas_process_mm_stack_s (params, stack_size, &
                                                      a_data, b_data, c_data)
         END IF
      END SUBROUTINE libxs_process_mm_stack_s
#endif

      PURE SUBROUTINE internal_mm_s_nn( &
         M, N, K, A, B, C)
         INTEGER, INTENT(IN)                      :: M, N, K
         REAL(kind=real_4), INTENT(INOUT)                   :: C(M, N)
         REAL(kind=real_4), INTENT(IN)                      :: B(K, N)
         REAL(kind=real_4), INTENT(IN)                      :: A(M, K)
         C(:, :) = C(:, :) + MATMUL(A, B)
      END SUBROUTINE internal_mm_s_nn
# 248 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
      SUBROUTINE blas_process_mm_stack_z (params, &
                                                      stack_size, &
                                                      a_data, b_data, c_data)
     !! Processes MM stack and issues BLAS xGEMM calls

         INTEGER, INTENT(IN)                       :: stack_size
        !! Number of parameters
         INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
            INTENT(IN)                              :: params
        !! Stack of MM parameters
         COMPLEX(kind=real_8), DIMENSION(*), INTENT(IN)         :: a_data, &
            b_data
        !! Left-matrix data
        !! Right-matrix data
         COMPLEX(kind=real_8), DIMENSION(*), INTENT(INOUT)      :: c_data
        !! Product data

         INTEGER                                   :: sp

!   ---------------------------------------------------------------------------

         DO sp = 1, stack_size
            CALL ZGEMM ('N', &
                                   'N', &
                                   params(p_m, sp), params(p_n, sp), & !m, n
                                   params(p_k, sp), & ! k
                                   CMPLX(1.0, 0.0, real_8), & ! alpha
                                   a_data(params(p_a_first, sp)), & ! A
                                   params(p_m, sp), & !lda
                                   b_data(params(p_b_first, sp)), & ! B
                                   params(p_k, sp), & !ldb
                                   CMPLX(1.0, 0.0, real_8), & ! beta
                                   c_data(params(p_c_first, sp)), params(p_m, sp))
         END DO
      END SUBROUTINE blas_process_mm_stack_z

      SUBROUTINE internal_process_mm_stack_z (params, stack_size, &
                                                          a_data, b_data, c_data)
     !! Processes MM stack and issues internal MM calls.

         INTEGER, INTENT(IN)                       :: stack_size
        !! Number of parameters
         INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
            INTENT(IN)                              :: params
        !! Stack of MM parameters
         COMPLEX(kind=real_8), DIMENSION(*), INTENT(IN)         :: a_data, &
            b_data
        !! Left-matrix data
        !! Right-matrix data
         COMPLEX(kind=real_8), DIMENSION(*), INTENT(INOUT)      :: c_data
        !! Product data

         INTEGER                                   :: sp

!   ---------------------------------------------------------------------------

         DO sp = 1, stack_size
            CALL internal_mm_z_nn( &
               params(p_m, sp), &
               params(p_n, sp), &
               params(p_k, sp), &
               a_data(params(p_a_first, sp)), &
               b_data(params(p_b_first, sp)), &
               c_data(params(p_c_first, sp)))
         END DO
      END SUBROUTINE internal_process_mm_stack_z

      SUBROUTINE smm_process_mm_stack_z (stack_descr, params, &
                                                     stack_size, &
                                                     a_data, b_data, c_data, used_smm)
     !! Processes MM stack and issues SMM library calls

         INTEGER, INTENT(IN)                       :: stack_size
        !! Number of parameters
         TYPE(stack_descriptor_type), INTENT(IN)   :: stack_descr
         INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
            INTENT(IN)                              :: params
        !! Stack of MM parameters
         COMPLEX(kind=real_8), DIMENSION(*), INTENT(IN)         :: a_data, &
            b_data
        !! Left-matrix data
        !! Right-matrix data
         COMPLEX(kind=real_8), DIMENSION(*), INTENT(INOUT)      :: c_data
        !! Product data
         LOGICAL, INTENT(OUT)                      :: used_smm

#if defined(__HAS_smm_znn)

         INTEGER                                   :: sp

         ! TODO we have no way of knowing which calls to libsmm actually resolve to BLAS
         ! Fixing this requires an interface change to libsmm.
         used_smm = .TRUE.

#if defined(__HAS_smm_vec)
         IF (stack_descr%defined_mnk) THEN
            CALL smm_vec_znn(stack_descr%m, stack_descr%n, stack_descr%k, &
                                         a_data, b_data, c_data, stack_size, &
                                         dbcsr_ps_width, params, p_a_first, p_b_first, p_c_first)
            RETURN
         END IF
#endif

         DO sp = 1, stack_size
            CALL smm_znn( &
               params(p_m, sp), &
               params(p_n, sp), &
               params(p_k, sp), &
               a_data(params(p_a_first, sp)), &
               b_data(params(p_b_first, sp)), &
               c_data(params(p_c_first, sp)))
         END DO

#else
         ! We do not want to abort here, fall back to BLAS.
         used_smm = .FALSE.
         CALL blas_process_mm_stack_z (params, stack_size, a_data, b_data, c_data)
#endif

         MARK_USED(stack_descr)
      END SUBROUTINE smm_process_mm_stack_z

#if defined(__LIBXS)
      SUBROUTINE libxs_process_mm_stack_z (stack_descr, params, &
                                                       stack_size, a_data, b_data, c_data, used_smm)
     !! Processes MM stack using LIBXS indexed GEMM batch.
     !! Real types use LIBXS dispatch; complex types fall through to BLAS.
# 428 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
         INTEGER, INTENT(IN)                            :: stack_size
         TYPE(stack_descriptor_type), INTENT(IN)        :: stack_descr
         INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
            INTENT(IN), TARGET                          :: params
         COMPLEX(kind=real_8), DIMENSION(*), TARGET, INTENT(IN)    :: a_data
         COMPLEX(kind=real_8), DIMENSION(*), TARGET, INTENT(IN)    :: b_data
         COMPLEX(kind=real_8), DIMENSION(*), TARGET, INTENT(INOUT) :: c_data
         LOGICAL, INTENT(OUT)                           :: used_smm

# 441 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"

         used_smm = .FALSE.
# 444 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
            MARK_USED(stack_descr)
# 446 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
# 480 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
         IF (.NOT. used_smm) THEN
            CALL blas_process_mm_stack_z (params, stack_size, &
                                                      a_data, b_data, c_data)
         END IF
      END SUBROUTINE libxs_process_mm_stack_z
#endif

      PURE SUBROUTINE internal_mm_z_nn( &
         M, N, K, A, B, C)
         INTEGER, INTENT(IN)                      :: M, N, K
         COMPLEX(kind=real_8), INTENT(INOUT)                   :: C(M, N)
         COMPLEX(kind=real_8), INTENT(IN)                      :: B(K, N)
         COMPLEX(kind=real_8), INTENT(IN)                      :: A(M, K)
         C(:, :) = C(:, :) + MATMUL(A, B)
      END SUBROUTINE internal_mm_z_nn
# 248 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
      SUBROUTINE blas_process_mm_stack_c (params, &
                                                      stack_size, &
                                                      a_data, b_data, c_data)
     !! Processes MM stack and issues BLAS xGEMM calls

         INTEGER, INTENT(IN)                       :: stack_size
        !! Number of parameters
         INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
            INTENT(IN)                              :: params
        !! Stack of MM parameters
         COMPLEX(kind=real_4), DIMENSION(*), INTENT(IN)         :: a_data, &
            b_data
        !! Left-matrix data
        !! Right-matrix data
         COMPLEX(kind=real_4), DIMENSION(*), INTENT(INOUT)      :: c_data
        !! Product data

         INTEGER                                   :: sp

!   ---------------------------------------------------------------------------

         DO sp = 1, stack_size
            CALL CGEMM ('N', &
                                   'N', &
                                   params(p_m, sp), params(p_n, sp), & !m, n
                                   params(p_k, sp), & ! k
                                   CMPLX(1.0, 0.0, real_4), & ! alpha
                                   a_data(params(p_a_first, sp)), & ! A
                                   params(p_m, sp), & !lda
                                   b_data(params(p_b_first, sp)), & ! B
                                   params(p_k, sp), & !ldb
                                   CMPLX(1.0, 0.0, real_4), & ! beta
                                   c_data(params(p_c_first, sp)), params(p_m, sp))
         END DO
      END SUBROUTINE blas_process_mm_stack_c

      SUBROUTINE internal_process_mm_stack_c (params, stack_size, &
                                                          a_data, b_data, c_data)
     !! Processes MM stack and issues internal MM calls.

         INTEGER, INTENT(IN)                       :: stack_size
        !! Number of parameters
         INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
            INTENT(IN)                              :: params
        !! Stack of MM parameters
         COMPLEX(kind=real_4), DIMENSION(*), INTENT(IN)         :: a_data, &
            b_data
        !! Left-matrix data
        !! Right-matrix data
         COMPLEX(kind=real_4), DIMENSION(*), INTENT(INOUT)      :: c_data
        !! Product data

         INTEGER                                   :: sp

!   ---------------------------------------------------------------------------

         DO sp = 1, stack_size
            CALL internal_mm_c_nn( &
               params(p_m, sp), &
               params(p_n, sp), &
               params(p_k, sp), &
               a_data(params(p_a_first, sp)), &
               b_data(params(p_b_first, sp)), &
               c_data(params(p_c_first, sp)))
         END DO
      END SUBROUTINE internal_process_mm_stack_c

      SUBROUTINE smm_process_mm_stack_c (stack_descr, params, &
                                                     stack_size, &
                                                     a_data, b_data, c_data, used_smm)
     !! Processes MM stack and issues SMM library calls

         INTEGER, INTENT(IN)                       :: stack_size
        !! Number of parameters
         TYPE(stack_descriptor_type), INTENT(IN)   :: stack_descr
         INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
            INTENT(IN)                              :: params
        !! Stack of MM parameters
         COMPLEX(kind=real_4), DIMENSION(*), INTENT(IN)         :: a_data, &
            b_data
        !! Left-matrix data
        !! Right-matrix data
         COMPLEX(kind=real_4), DIMENSION(*), INTENT(INOUT)      :: c_data
        !! Product data
         LOGICAL, INTENT(OUT)                      :: used_smm

#if defined(__HAS_smm_cnn)

         INTEGER                                   :: sp

         ! TODO we have no way of knowing which calls to libsmm actually resolve to BLAS
         ! Fixing this requires an interface change to libsmm.
         used_smm = .TRUE.

#if defined(__HAS_smm_vec)
         IF (stack_descr%defined_mnk) THEN
            CALL smm_vec_cnn(stack_descr%m, stack_descr%n, stack_descr%k, &
                                         a_data, b_data, c_data, stack_size, &
                                         dbcsr_ps_width, params, p_a_first, p_b_first, p_c_first)
            RETURN
         END IF
#endif

         DO sp = 1, stack_size
            CALL smm_cnn( &
               params(p_m, sp), &
               params(p_n, sp), &
               params(p_k, sp), &
               a_data(params(p_a_first, sp)), &
               b_data(params(p_b_first, sp)), &
               c_data(params(p_c_first, sp)))
         END DO

#else
         ! We do not want to abort here, fall back to BLAS.
         used_smm = .FALSE.
         CALL blas_process_mm_stack_c (params, stack_size, a_data, b_data, c_data)
#endif

         MARK_USED(stack_descr)
      END SUBROUTINE smm_process_mm_stack_c

#if defined(__LIBXS)
      SUBROUTINE libxs_process_mm_stack_c (stack_descr, params, &
                                                       stack_size, a_data, b_data, c_data, used_smm)
     !! Processes MM stack using LIBXS indexed GEMM batch.
     !! Real types use LIBXS dispatch; complex types fall through to BLAS.
# 428 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
         INTEGER, INTENT(IN)                            :: stack_size
         TYPE(stack_descriptor_type), INTENT(IN)        :: stack_descr
         INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
            INTENT(IN), TARGET                          :: params
         COMPLEX(kind=real_4), DIMENSION(*), TARGET, INTENT(IN)    :: a_data
         COMPLEX(kind=real_4), DIMENSION(*), TARGET, INTENT(IN)    :: b_data
         COMPLEX(kind=real_4), DIMENSION(*), TARGET, INTENT(INOUT) :: c_data
         LOGICAL, INTENT(OUT)                           :: used_smm

# 441 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"

         used_smm = .FALSE.
# 444 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
            MARK_USED(stack_descr)
# 446 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
# 480 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"
         IF (.NOT. used_smm) THEN
            CALL blas_process_mm_stack_c (params, stack_size, &
                                                      a_data, b_data, c_data)
         END IF
      END SUBROUTINE libxs_process_mm_stack_c
#endif

      PURE SUBROUTINE internal_mm_c_nn( &
         M, N, K, A, B, C)
         INTEGER, INTENT(IN)                      :: M, N, K
         COMPLEX(kind=real_4), INTENT(INOUT)                   :: C(M, N)
         COMPLEX(kind=real_4), INTENT(IN)                      :: B(K, N)
         COMPLEX(kind=real_4), INTENT(IN)                      :: A(M, K)
         C(:, :) = C(:, :) + MATMUL(A, B)
      END SUBROUTINE internal_mm_c_nn
# 496 "/__w/dbcsr/dbcsr/src/mm/dbcsr_mm_hostdrv.F"

END MODULE dbcsr_mm_hostdrv
