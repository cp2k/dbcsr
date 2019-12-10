!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

#:include '../data/dbcsr.fypp'
#:for n, nametype1, base1, prec1, kind1, type1, dkind1 in inst_params_float
  SUBROUTINE blas_process_mm_stack_${nametype1}$ (params, &
                                                  stack_size, &
                                                  a_data, b_data, c_data)
     !! Processes MM stack and issues BLAS xGEMM calls

     INTEGER, INTENT(IN)                       :: stack_size
        !! Number of parameters
     INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
        INTENT(IN)                              :: params
        !! Stack of MM parameters
     ${type1}$, DIMENSION(*), INTENT(IN)         :: a_data, &
        b_data
        !! Left-matrix data
        !! Right-matrix data
     ${type1}$, DIMENSION(*), INTENT(INOUT)      :: c_data
        !! Product data

     CHARACTER(len=*), PARAMETER :: routineN = 'blas_process_mm_stack_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER                                   :: sp

!   ---------------------------------------------------------------------------

     DO sp = 1, stack_size
        CALL ${gemmname1[n]}$ ('N', &
                               'N', &
                               params(p_m, sp), params(p_n, sp), & !m, n
                               params(p_k, sp), & ! k
                               ${one1[n]}$, & ! alpha
                               a_data(params(p_a_first, sp)), & ! A
                               params(p_m, sp), & !lda
                               b_data(params(p_b_first, sp)), & ! B
                               params(p_k, sp), & !ldb
                               ${one1[n]}$, & ! beta
                               c_data(params(p_c_first, sp)), params(p_m, sp))
     ENDDO
  END SUBROUTINE blas_process_mm_stack_${nametype1}$

  SUBROUTINE internal_process_mm_stack_${nametype1}$ (params, stack_size, &
                                                      a_data, b_data, c_data)
     !! Processes MM stack and issues internal MM calls.

     INTEGER, INTENT(IN)                       :: stack_size
        !! Number of parameters
     INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
        INTENT(IN)                              :: params
        !! Stack of MM parameters
     ${type1}$, DIMENSION(*), INTENT(IN)         :: a_data, &
        b_data
        !! Left-matrix data
        !! Right-matrix data
     ${type1}$, DIMENSION(*), INTENT(INOUT)      :: c_data
        !! Product data

     CHARACTER(len=*), PARAMETER :: routineN = 'internal_process_mm_stack_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER                                   :: sp

!   ---------------------------------------------------------------------------

     DO sp = 1, stack_size
        CALL internal_mm_${nametype1}$_nn( &
           params(p_m, sp), &
           params(p_n, sp), &
           params(p_k, sp), &
           a_data(params(p_a_first, sp)), &
           b_data(params(p_b_first, sp)), &
           c_data(params(p_c_first, sp)))
     ENDDO
  END SUBROUTINE internal_process_mm_stack_${nametype1}$

  SUBROUTINE smm_process_mm_stack_${nametype1}$ (stack_descr, params, &
                                                 stack_size, &
                                                 a_data, b_data, c_data, used_smm)
     !! Processes MM stack and issues SMM library calls

     INTEGER, INTENT(IN)                       :: stack_size
        !! Number of parameters
     TYPE(stack_descriptor_type), INTENT(IN)   :: stack_descr
     INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
        INTENT(IN)                              :: params
        !! Stack of MM parameters
     ${type1}$, DIMENSION(*), INTENT(IN)         :: a_data, &
        b_data
        !! Left-matrix data
        !! Right-matrix data
     ${type1}$, DIMENSION(*), INTENT(INOUT)      :: c_data
        !! Product data
     LOGICAL, INTENT(OUT)                      :: used_smm

     CHARACTER(len=*), PARAMETER :: routineN = 'smm_process_mm_stack_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

#if defined(__HAS_smm_${nametype1}$nn)

     INTEGER                                   :: sp

     ! TODO we have no way of knowing which calls to libsmm actually resolve to BLAS
     ! Fixing this requires an interface change to libsmm.
     used_smm = .TRUE.

#if defined(__HAS_smm_vec)
     IF (stack_descr%defined_mnk) THEN
        CALL smm_vec_${nametype1}$nn(stack_descr%m, stack_descr%n, stack_descr%k, &
                                     a_data, b_data, c_data, stack_size, &
                                     dbcsr_ps_width, params, p_a_first, p_b_first, p_c_first)
        RETURN
     ENDIF
#endif

     DO sp = 1, stack_size
        CALL smm_${nametype1}$nn( &
           params(p_m, sp), &
           params(p_n, sp), &
           params(p_k, sp), &
           a_data(params(p_a_first, sp)), &
           b_data(params(p_b_first, sp)), &
           c_data(params(p_c_first, sp)))
     ENDDO

#else
     ! We do not want to abort here, fall back to BLAS.
     used_smm = .FALSE.
     CALL blas_process_mm_stack_${nametype1}$ (params, stack_size, a_data, b_data, c_data)
#endif

     MARK_USED(stack_descr)
  END SUBROUTINE smm_process_mm_stack_${nametype1}$

#if defined(__LIBXSMM) && TO_VERSION(1, 10, 0) < TO_VERSION(LIBXSMM_CONFIG_VERSION_MAJOR, LIBXSMM_CONFIG_VERSION_MINOR, LIBXSMM_CONFIG_VERSION_UPDATE)
  SUBROUTINE xsmm_process_mm_batch_${nametype1}$ (stack_descr, params, &
                                                  stack_size, a_data, b_data, c_data, used_smm)
     !! Processes MM stack and issues libxsmm calls
#if ${xsmm_supported[n]}$
#if !defined(DBCSR_LIBXSMM_GEMM_BATCH)
#define DBCSR_LIBXSMM_GEMM_BATCH libxsmm_gemm_batch
#endif
     ! Caution: This dependency is ignored by makedep.py, because libxsmm.F is kinda empty.
     USE libxsmm, ONLY: LIBXSMM_GEMM_PRECISION => ${'LIBXSMM_GEMM_PRECISION_F'+bits1[n]}$, &
                        libxsmm_gemm => libxsmm_${nametype1}$gemm, &
                        libxsmm_gemm_batch => DBCSR_LIBXSMM_GEMM_BATCH, &
                        libxsmm_ptr0
     REAL(${kind1}$), PARAMETER :: one = 1.0_${kind1}$
     INTEGER :: sp
#endif
     INTEGER, INTENT(IN)                            :: stack_size
        !! Number of parameters
     TYPE(stack_descriptor_type), INTENT(IN)        :: stack_descr
     INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
        INTENT(IN)                                  :: params
        !! Stack of MM parameters
     ${type1}$, DIMENSION(*), TARGET, INTENT(IN)    :: a_data
        !! Left-matrix data
     ${type1}$, DIMENSION(*), TARGET, INTENT(IN)    :: b_data
        !! Right-matrix data
     ${type1}$, DIMENSION(*), TARGET, INTENT(INOUT) :: c_data
        !! Product data
     LOGICAL, INTENT(OUT)                           :: used_smm
        !! Flag to signal if an efficient kernel was used

     CHARACTER(len=*), PARAMETER :: routineN = 'xsmm_process_mm_batch_${nametype1}$', &
                                    routineP = moduleN//':'//routineN
#if ${xsmm_supported[n]}$
     IF (stack_descr%defined_mnk) THEN ! homogeneous stack
        CALL libxsmm_gemm_batch(LIBXSMM_GEMM_PRECISION, LIBXSMM_GEMM_PRECISION, 'N', 'N', &
                                m=stack_descr%m, n=stack_descr%n, k=stack_descr%k, &
                                alpha=libxsmm_ptr0(one), a=libxsmm_ptr0(a_data(LBOUND(a_data,1))), &
                                lda=stack_descr%m, &
                                b=libxsmm_ptr0(b_data(LBOUND(b_data,1))), &
                                ldb=stack_descr%k, &
                                beta=libxsmm_ptr0(one),  c=libxsmm_ptr0(c_data(LBOUND(c_data,1))), &
                                ldc=stack_descr%m, index_base=1, &
                                index_stride=KIND(params)*dbcsr_ps_width, &
                                stride_a=libxsmm_ptr0(params(p_a_first,1)), &
                                stride_b=libxsmm_ptr0(params(p_b_first,1)), &
                                stride_c=libxsmm_ptr0(params(p_c_first,1)), &
                                batchsize=stack_size)
        used_smm = .TRUE.
     ELSE ! Dispatch for every (different) matrix
        DO sp = 1, stack_size
           CALL libxsmm_gemm(m=params(p_m, sp), n=params(p_n, sp), k=params(p_k, sp), &
                             a=a_data(params(p_a_first,sp)), &
                             b=b_data(params(p_b_first,sp)), &
                             c=c_data(params(p_c_first,sp)), &
                             alpha=one, beta=one)
        ENDDO
        used_smm = .FALSE.
     ENDIF
#else
     MARK_USED(stack_descr)
     ! We do not want to abort here, fall back to BLAS.
     CALL blas_process_mm_stack_${nametype1}$ (params, stack_size, a_data, b_data, c_data)
     used_smm = .FALSE.
#endif
  END SUBROUTINE xsmm_process_mm_batch_${nametype1}$
#endif

  SUBROUTINE xsmm_process_mm_stack_${nametype1}$ (stack_descr, params, &
                                                  stack_size, a_data, b_data, c_data, used_smm)
     !! Processes MM stack and issues libxsmm calls
#if defined(__LIBXSMM) && ${xsmm_supported[n]}$
     ! Caution: This dependency is ignored by makedep.py, because libxsmm.F is kinda empty.
     USE libxsmm, ONLY: libxsmm_function => libxsmm_${nametype1}$mmfunction, &
                        libxsmm_dispatch => libxsmm_${nametype1}$mmdispatch, &
                        libxsmm_available => libxsmm_${nametype1}$mmavailable, &
                        libxsmm_call => libxsmm_${nametype1}$mmcall, &
                        libxsmm_gemm => libxsmm_${nametype1}$gemm, &
                        LIBXSMM_PREFETCH_NONE, &
                        LIBXSMM_PREFETCH, &
                        LIBXSMM_ROW_MAJOR, &
                        LIBXSMM_COL_MAJOR, &
                        LIBXSMM_MAX_MNK, &
                        LIBXSMM_FLAGS
     INTEGER, PARAMETER :: LIBXSMM_DEFAULT_PREFETCH = LIBXSMM_PREFETCH
     INTEGER, PARAMETER :: LIBXSMM_DEFAULT_FLAGS = LIBXSMM_FLAGS
     REAL(${kind1}$), PARAMETER :: one = 1.0_${kind1}$
     REAL(${kind1}$), DIMENSION(:, :), POINTER :: a_ptr, b_ptr, c_ptr
     INTEGER :: m, n, k, sp, fa, fb, fc
     LOGICAL :: processed
     TYPE(libxsmm_function) :: func
     INTEGER(int_8) :: threshold
     INTEGER :: pa, pb, pc
#endif
     INTEGER, INTENT(IN)                            :: stack_size
        !! Number of parameters
     TYPE(stack_descriptor_type), INTENT(IN)        :: stack_descr
     INTEGER, DIMENSION(dbcsr_ps_width, 1:stack_size), &
        INTENT(IN)                                  :: params
        !! Stack of MM parameters
     ${type1}$, DIMENSION(*), TARGET, INTENT(IN)    :: a_data
        !! Left-matrix data
     ${type1}$, DIMENSION(*), TARGET, INTENT(IN)    :: b_data
        !! Right-matrix data
     ${type1}$, DIMENSION(*), TARGET, INTENT(INOUT) :: c_data
        !! Product data
     LOGICAL, INTENT(OUT)                           :: used_smm
        !! Flag to signal if an efficient kernel was used

     CHARACTER(len=*), PARAMETER :: routineN = 'libxsmm_process_mm_stack_${nametype1}$', &
                                    routineP = moduleN//':'//routineN
#if defined(__LIBXSMM) && ${xsmm_supported[n]}$
     DBCSR_ASSERT(LIBXSMM_COL_MAJOR /= 0 .AND. LIBXSMM_ROW_MAJOR == 0)
     processed = .FALSE.
     used_smm = .FALSE.

     ! check whether the matrix stack is homogeneous or not
     IF (stack_descr%defined_mnk) THEN
        threshold = INT(stack_descr%m, int_8)* &
                    INT(stack_descr%n, int_8)* &
                    INT(stack_descr%k, int_8)

        ! check if matrices are too large for LIBXSMM (BLAS is likely more efficient)
        IF (threshold <= LIBXSMM_MAX_MNK) THEN
           ! try to get a function pointer from libxsmm
           CALL libxsmm_dispatch(func, &
                                 m=stack_descr%m, n=stack_descr%n, k=stack_descr%k, alpha=one, beta=one, &
                                 flags=LIBXSMM_DEFAULT_FLAGS, prefetch=LIBXSMM_DEFAULT_PREFETCH)

           IF (libxsmm_available(func)) THEN
              ! load first stack entry
              DBCSR_ASSERT(stack_size > 0)
              pa = params(p_a_first, 1)
              pb = params(p_b_first, 1)
              pc = params(p_c_first, 1)

              DO sp = 1, stack_size - 1
                 fa = pa; fb = pb; fc = pc
                 ! prefetch next blocks
                 pa = params(p_a_first, sp + 1)
                 pb = params(p_b_first, sp + 1)
                 pc = params(p_c_first, sp + 1)

                 ! condition evaluates at compile-time (PARAMETER)
                 IF (LIBXSMM_DEFAULT_PREFETCH /= LIBXSMM_PREFETCH_NONE) THEN
                    CALL libxsmm_call(func, &
                                      a=a_data(fa), b=b_data(fb), c=c_data(fc), &
                                      ! provide locations of the next operand set
                                      pa=a_data(pa), pb=b_data(pb), pc=c_data(pc))
                 ELSE
                    CALL libxsmm_call(func, &
                                      a=a_data(fa), b=b_data(fb), c=c_data(fc))
                 ENDIF
              ENDDO

              ! handle last stack entry without out-of-bounds access
              fa = pa; fb = pb; fc = pc

              ! condition evaluates at compile-time (PARAMETER)
              IF (LIBXSMM_DEFAULT_PREFETCH /= LIBXSMM_PREFETCH_NONE) THEN
                 CALL libxsmm_call(func, &
                                   a=a_data(fa), b=b_data(fb), c=c_data(fc), &
                                   ! prefetch same blocks
                                   pa=a_data(pa), pb=b_data(pb), pc=c_data(pc))
              ELSE
                 CALL libxsmm_call(func, &
                                   a=a_data(fa), b=b_data(fb), c=c_data(fc))
              ENDIF

              processed = .TRUE.
              used_smm = .TRUE.
           ENDIF
        ELSE
           CALL blas_process_mm_stack_${nametype1}$ (params, stack_size, a_data, b_data, c_data)
           processed = .TRUE.
        ENDIF
     ENDIF

     IF (.NOT. processed) THEN
        ! Dispatch interface was not used, call regular interface.
        ! Should only happen for inhomogeneous stacks.
        ! Counted as used_smm = .FALSE.
        DO sp = 1, stack_size
           m = params(p_m, sp)
           n = params(p_n, sp)
           k = params(p_k, sp)
           fa = params(p_a_first, sp)
           fb = params(p_b_first, sp)
           fc = params(p_c_first, sp)
           ! somewhat expensive pointer remapping required
           a_ptr(1:m, 1:k) => a_data(fa:fa + (m*k))
           b_ptr(1:k, 1:n) => b_data(fb:fb + (k*n))
           c_ptr(1:m, 1:n) => c_data(fc:fc + (m*n))
           CALL libxsmm_gemm(m=m, n=n, k=k, a=a_ptr, b=b_ptr, c=c_ptr, &
                             alpha=one, beta=one)
        ENDDO
     ENDIF
#else
     MARK_USED(stack_descr)
     ! We do not want to abort here, fall back to BLAS.
     CALL blas_process_mm_stack_${nametype1}$ (params, stack_size, a_data, b_data, c_data)
     used_smm = .FALSE.
#endif
  END SUBROUTINE xsmm_process_mm_stack_${nametype1}$

  PURE SUBROUTINE internal_mm_${nametype1}$_nn( &
     M, N, K, A, B, C)
     INTEGER, INTENT(IN)                      :: M, N, K
     ${type1}$, INTENT(INOUT)                   :: C(M, N)
     ${type1}$, INTENT(IN)                      :: B(K, N)
     ${type1}$, INTENT(IN)                      :: A(M, K)
     C(:, :) = C(:, :) + MATMUL(A, B)
  END SUBROUTINE internal_mm_${nametype1}$_nn
#:endfor
