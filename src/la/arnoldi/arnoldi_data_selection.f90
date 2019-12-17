!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

!!! Here come the methods handling the selection of eigenvalues and eigenvectors !!!
!!! If you want a personal method, simply created a Subroutine returning the index
!!! array selected ind which contains as the first nval_out entries the index of the evals

#:include 'arnoldi.fypp'
#:for nametype1, type_prec, real_zero, nametype_zero, type_nametype1, vartype in inst_params_1
! **************************************************************************************************
!> \brief ...
!> \param arnoldi_data ...
! **************************************************************************************************
  SUBROUTINE select_evals_${nametype1}$(arnoldi_data)
     TYPE(arnoldi_data_type)                :: arnoldi_data

     CHARACTER(LEN=*), PARAMETER :: routineN = 'select_evals_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER                                  :: my_crit, last_el, my_ind, i
     REAL(${type_prec}$)                        :: convergence
     TYPE(arnoldi_data_${nametype1}$_type),POINTER   :: ar_data
     TYPE(arnoldi_control_type), POINTER           :: control

     control => get_control(arnoldi_data)
     ar_data => get_data_${nametype1}$(arnoldi_data)

     last_el=control%current_step
     convergence=REAL(0.0, ${type_prec}$)
     my_crit=control%selection_crit
     control%nval_out=MIN(control%nval_req, control%current_step)
     SELECT CASE(my_crit)
        ! minimum and maximum real eval
     CASE(1)
        CALL index_min_max_real_eval_${nametype1}$(ar_data%evals, control%current_step, control%selected_ind, control%nval_out)
        ! n maximum real eval
     CASE(2)
        CALL index_nmax_real_eval_${nametype1}$(ar_data%evals, control%current_step, control%selected_ind, control%nval_out)
        ! n minimum real eval
     CASE(3)
        CALL index_nmin_real_eval_${nametype1}$(ar_data%evals, control%current_step, control%selected_ind, control%nval_out)
     CASE DEFAULT
        DBCSR_ABORT("unknown selection index")
     END SELECT
     ! test whether we are converged
     DO i=1, control%nval_out
        my_ind=control%selected_ind(i)
        convergence=MAX(convergence, &
                        ABS(ar_data%revec(last_el, my_ind)*ar_data%Hessenberg(last_el+1, last_el)))
     END DO
     control%converged=convergence.LT.control%threshold

  END SUBROUTINE select_evals_${nametype1}$

! **************************************************************************************************
!> \brief ...
!> \param evals ...
!> \param current_step ...
!> \param selected_ind ...
!> \param neval ...
! **************************************************************************************************
  SUBROUTINE index_min_max_real_eval_${nametype1}$(evals, current_step, selected_ind, neval)
     COMPLEX(${type_prec}$), DIMENSION(:)       :: evals
     INTEGER, INTENT(IN)                      :: current_step
     INTEGER, DIMENSION(:)                    :: selected_ind
     INTEGER                                  :: neval

     CHARACTER(LEN=*), PARAMETER :: routineN = 'index_min_max_real_eval_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER, DIMENSION(current_step)         :: indexing
     REAL(${type_prec}$), DIMENSION(current_step)        :: tmp_array
     INTEGER                                   :: i

     neval=0
     selected_ind=0
     tmp_array(1:current_step)=REAL(evals(1:current_step), ${type_prec}$)
     CALL sort(tmp_array, current_step, indexing)
     DO i=1,current_step
        IF(ABS(AIMAG(evals(indexing(i))))<EPSILON(${real_zero}$))THEN
           selected_ind(1)=indexing(i)
           neval=neval+1
           EXIT
        END IF
     END DO
     DO i=current_step,1,-1
        IF(ABS(AIMAG(evals(indexing(i))))<EPSILON(${real_zero}$))THEN
           selected_ind(2)=indexing(i)
           neval=neval+1
           EXIT
        END IF
     END DO

  END SUBROUTINE index_min_max_real_eval_${nametype1}$

! **************************************************************************************************
!> \brief ...
!> \param evals ...
!> \param current_step ...
!> \param selected_ind ...
!> \param neval ...
! **************************************************************************************************
  SUBROUTINE index_nmax_real_eval_${nametype1}$(evals, current_step, selected_ind, neval)
     COMPLEX(${type_prec}$), DIMENSION(:)       :: evals
     INTEGER, INTENT(IN)                      :: current_step
     INTEGER, DIMENSION(:)                    :: selected_ind
     INTEGER                                  :: neval

     CHARACTER(LEN=*), PARAMETER :: routineN = 'index_nmax_real_eval_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER                                  :: i, nlimit
     INTEGER, DIMENSION(current_step)         :: indexing
     REAL(${type_prec}$), DIMENSION(current_step)        :: tmp_array

     nlimit=neval; neval=0
     selected_ind=0
     tmp_array(1:current_step)=REAL(evals(1:current_step), ${type_prec}$)
     CALL sort(tmp_array, current_step, indexing)
     DO i=1, current_step
        IF(ABS(AIMAG(evals(indexing(current_step+1-i))))<EPSILON(${real_zero}$))THEN
           selected_ind(i)=indexing(current_step+1-i)
           neval=neval+1
           IF(neval==nlimit)EXIT
        END IF
     END DO

  END SUBROUTINE index_nmax_real_eval_${nametype1}$

! **************************************************************************************************
!> \brief ...
!> \param evals ...
!> \param current_step ...
!> \param selected_ind ...
!> \param neval ...
! **************************************************************************************************
  SUBROUTINE index_nmin_real_eval_${nametype1}$(evals, current_step, selected_ind, neval)
     COMPLEX(${type_prec}$), DIMENSION(:)       :: evals
     INTEGER, INTENT(IN)                      :: current_step
     INTEGER, DIMENSION(:)                    :: selected_ind
     INTEGER                                  :: neval,nlimit

     CHARACTER(LEN=*), PARAMETER :: routineN = 'index_nmin_real_eval_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER                                  :: i
     INTEGER, DIMENSION(current_step)         :: indexing
     REAL(${type_prec}$), DIMENSION(current_step)        :: tmp_array

     nlimit=neval; neval=0
     selected_ind=0
     tmp_array(1:current_step)=REAL(evals(1:current_step), ${type_prec}$)
     CALL sort(tmp_array, current_step, indexing)
     DO i=1, current_step
        IF(ABS(AIMAG(evals(indexing(i))))<EPSILON(${real_zero}$))THEN
           selected_ind(i)=indexing(i)
           neval=neval+1
           IF(neval==nlimit)EXIT
        END IF
     END DO

  END SUBROUTINE index_nmin_real_eval_${nametype1}$

#:endfor
