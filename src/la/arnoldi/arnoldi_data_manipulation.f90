#:include 'arnoldi.fypp'
#:for nametype1, type_prec, real_zero, nametype_zero, type_nametype1, vartype in inst_params_1
! **************************************************************************************************
!> \brief ...
!> \param arnoldi_data ...
!> \param matrix ...
!> \param max_iter ...
! **************************************************************************************************
  SUBROUTINE setup_arnoldi_data_${nametype1}$ (arnoldi_data, matrix, max_iter)
    TYPE(arnoldi_data_type)                 :: arnoldi_data
    TYPE(dbcsr_p_type), DIMENSION(:)     :: matrix
    INTEGER                                 :: max_iter

    CHARACTER(LEN=*), PARAMETER :: routineN = 'allocate_arnoldi_data_${nametype1}$', &
      routineP = moduleN//':'//routineN

    INTEGER                                           :: nrow_local
    TYPE(arnoldi_data_${nametype1}$_type), POINTER      :: ar_data

    ALLOCATE(ar_data)
    CALL dbcsr_get_info(matrix=matrix(1)%matrix, nfullrows_local=nrow_local)
    ALLOCATE(ar_data%f_vec(nrow_local))
    ALLOCATE(ar_data%x_vec(nrow_local))
    ALLOCATE(ar_data%Hessenberg(max_iter+1, max_iter))
    ALLOCATE(ar_data%local_history(nrow_local, max_iter))

    ALLOCATE(ar_data%evals(max_iter))
    ALLOCATE(ar_data%revec(max_iter, max_iter))

    CALL set_data_${nametype1}$(arnoldi_data,ar_data)

  END SUBROUTINE setup_arnoldi_data_${nametype1}$

! **************************************************************************************************
!> \brief ...
!> \param arnoldi_data ...
! **************************************************************************************************
  SUBROUTINE deallocate_arnoldi_data_${nametype1}$ (arnoldi_data)
    TYPE(arnoldi_data_type)                     :: arnoldi_data

    CHARACTER(LEN=*), PARAMETER :: routineN = 'deallocate_arnoldi_data_${nametype1}$', &
      routineP = moduleN//':'//routineN

    TYPE(arnoldi_data_${nametype1}$_type), POINTER            :: ar_data

    ar_data=>get_data_${nametype1}$(arnoldi_data)
    IF(ASSOCIATED(ar_data%f_vec))DEALLOCATE(ar_data%f_vec)
    IF(ASSOCIATED(ar_data%x_vec))DEALLOCATE(ar_data%x_vec)
    IF(ASSOCIATED(ar_data%Hessenberg))DEALLOCATE(ar_data%Hessenberg)
    IF(ASSOCIATED(ar_data%local_history))DEALLOCATE(ar_data%local_history)
    IF(ASSOCIATED(ar_data%evals))DEALLOCATE(ar_data%evals)
    IF(ASSOCIATED(ar_data%revec))DEALLOCATE(ar_data%revec)
    DEALLOCATE(ar_data)

  END SUBROUTINE deallocate_arnoldi_data_${nametype1}$

! **************************************************************************************************
!> \brief ...
!> \param arnoldi_data ...
!> \param ind ...
!> \param matrix ...
!> \param vector ...
! **************************************************************************************************
  SUBROUTINE get_selected_ritz_vector_${nametype1}$(arnoldi_data,ind,matrix,vector)
    TYPE(arnoldi_data_type)                 :: arnoldi_data
    INTEGER                                  :: ind
    TYPE(dbcsr_type)                          :: matrix
    TYPE(dbcsr_type)                          :: vector

    CHARACTER(LEN=*), PARAMETER :: routineN = 'get_selected_ritz_vector_${nametype1}$', &
      routineP = moduleN//':'//routineN

    TYPE(arnoldi_data_${nametype1}$_type), POINTER      :: ar_data
    INTEGER                                           :: vsize, myind, sspace_size, i
    INTEGER, DIMENSION(:), POINTER           :: selected_ind
    COMPLEX(${type_prec}$),DIMENSION(:),ALLOCATABLE       :: ritz_v
    ${type_nametype1}$, DIMENSION(:), POINTER          :: data_vec
    TYPE(arnoldi_control_type), POINTER           :: control

    control=>get_control(arnoldi_data)
    selected_ind=>get_sel_ind(arnoldi_data)
    ar_data=>get_data_${nametype1}$(arnoldi_data)
    sspace_size=get_subsp_size(arnoldi_data)
    vsize=SIZE(ar_data%f_vec)
    myind=selected_ind(ind)
    ALLOCATE(ritz_v(vsize))
    ritz_v=CMPLX(0.0,0.0,${type_prec}$)

    CALL dbcsr_release(vector)
    CALL create_col_vec_from_matrix(vector,matrix,1)
    IF(control%local_comp)THEN
       DO i=1,sspace_size
          ritz_v(:)=ritz_v(:)+ar_data%local_history(:,i)*ar_data%revec(i,myind)
       END DO
       data_vec => dbcsr_get_data_p (vector, select_data_type=${nametype_zero}$)
       ! is a bit odd but ritz_v is always complex and matrix type determines where it goes
       ! again I hope the user knows what is required
       data_vec(1:vsize) =${vartype}$(ritz_v(1:vsize),KIND=${type_prec}$)
    END IF

    DEALLOCATE(ritz_v)

  END SUBROUTINE get_selected_ritz_vector_${nametype1}$


! **************************************************************************************************
!> \brief ...
!> \param arnoldi_data ...
!> \param vector ...
! **************************************************************************************************
  SUBROUTINE set_initial_vector_${nametype1}$(arnoldi_data,vector)
    TYPE(arnoldi_data_type)                 :: arnoldi_data
    TYPE(dbcsr_type)                          :: vector

    CHARACTER(LEN=*), PARAMETER :: routineN = 'set_initial_vector_${nametype1}$', &
      routineP = moduleN//':'//routineN

    TYPE(arnoldi_data_${nametype1}$_type), POINTER     :: ar_data
    ${type_nametype1}$, DIMENSION(:), POINTER          :: data_vec
    INTEGER                                           :: nrow_local, ncol_local
    TYPE(arnoldi_control_type), POINTER           :: control

    control=>get_control(arnoldi_data)

    CALL dbcsr_get_info(matrix=vector, nfullrows_local=nrow_local, nfullcols_local=ncol_local)
    ar_data=>get_data_${nametype1}$(arnoldi_data)
    data_vec => dbcsr_get_data_p (vector, select_data_type=${nametype_zero}$)
    IF(nrow_local*ncol_local>0)ar_data%f_vec(1:nrow_local)=data_vec(1:nrow_local)

  END SUBROUTINE set_initial_vector_${nametype1}$
#:endfor
