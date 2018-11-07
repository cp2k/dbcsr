!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

#:include 'dbcsr.fypp'
#:for nametype1, type1, zero1 in inst_params_all
! **************************************************************************************************
!> \brief Returns a pointer with different bounds.
!> \param[in] original   original data pointer
!> \param[in] lb lower and upper bound for the new pointer view
!> \param[in] ub lower and upper bound for the new pointer view
!> \return new pointer
! **************************************************************************************************
  FUNCTION pointer_view_${nametype1}$ (original, lb, ub) RESULT (view)
    ${type1}$, DIMENSION(:), POINTER :: original, view
    INTEGER, INTENT(IN)                  :: lb, ub
    view => original(lb:ub)
  END FUNCTION pointer_view_${nametype1}$


! **************************************************************************************************
!> \brief Ensures that an array is appropriately large.
!> \param[in,out] array       array to verify and possibly resize
!> \param[in] lb    (optional) desired array lower bound
!> \param[in] ub    desired array upper bound
!> \param[in] factor          (optional) factor by which to exagerrate
!>                            enlargements
!> \param[in] nocopy          (optional) copy array on enlargement; default
!>                            is to copy
!> \param[in] memory_type     (optional) use special memory
!> \param[in] zero_pad        (optional) zero new allocations; default is to
!>                            write nothing
! **************************************************************************************************
  SUBROUTINE ensure_array_size_${nametype1}$(array, array_resize, lb, ub, factor,&
       nocopy, memory_type, zero_pad)
    ${type1}$, DIMENSION(:), POINTER                 :: array
    ${type1}$, DIMENSION(:), POINTER, OPTIONAL       :: array_resize
    INTEGER, INTENT(IN), OPTIONAL                  :: lb
    INTEGER, INTENT(IN)                            :: ub
    REAL(KIND=dp), INTENT(IN), OPTIONAL            :: factor
    LOGICAL, INTENT(IN), OPTIONAL                  :: nocopy, zero_pad
    TYPE(dbcsr_memtype_type), INTENT(IN), OPTIONAL :: memory_type

    CHARACTER(len=*), PARAMETER :: routineN = 'ensure_array_size_${nametype1}$', &
      routineP = moduleN//':'//routineN

    INTEGER                                  :: lb_new, lb_orig, &
                                                ub_new, ub_orig, old_size,&
                                                size_increase
    TYPE(dbcsr_memtype_type)                 :: mem_type
    LOGICAL                                  :: dbg, docopy, &
                                                pad
    ${type1}$, DIMENSION(:), POINTER           :: newarray

!   ---------------------------------------------------------------------------
    !CALL timeset(routineN, error_handler)
    dbg = .FALSE.

    IF (PRESENT(array_resize)) NULLIFY(array_resize)

    IF (PRESENT (nocopy)) THEN
       docopy = .NOT. nocopy
    ELSE
       docopy = .TRUE.
    ENDIF
    IF (PRESENT (memory_type)) THEN
       mem_type = memory_type
    ELSE
       mem_type = dbcsr_memtype_default
    ENDIF
    lb_new = 1
    IF (PRESENT (lb)) lb_new = lb
    pad = .FALSE.
    IF (PRESENT (zero_pad)) pad = zero_pad
    !> Creates a new array if it doesn't yet exist.
    IF (.NOT.ASSOCIATED(array)) THEN
       IF(lb_new /= 1) &
          DBCSR_ABORT("Arrays must start at 1")
       CALL mem_alloc_${nametype1}$ (array, ub, mem_type=mem_type)
       IF (pad .AND. ub .GT. 0) CALL mem_zero_${nametype1}$ (array, ub)
       !CALL timestop(error_handler)
       RETURN
    ENDIF
    lb_orig = LBOUND(array,1)
    ub_orig = UBOUND(array,1)
    old_size = ub_orig - lb_orig + 1
    ! The existing array is big enough.
    IF (lb_orig.LE.lb_new .AND. ub_orig.GE.ub) THEN
       !CALL timestop(error_handler)
       RETURN
    ENDIF
    ! A reallocation must be performed
    IF(dbg) WRITE(*,*)routineP//' Current bounds are',lb_orig,':',ub_orig,&
         '; special?' !,mem_type
    !CALL timeset(routineN,timing_handle)
    IF (lb_orig.GT.lb_new) THEN
       IF (PRESENT(factor)) THEN
          size_increase = lb_orig - lb_new
          size_increase = MAX (NINT(size_increase*factor),&
                               NINT(old_size*(factor-1)),0)
          lb_new = MIN (lb_orig, lb_new - size_increase)
       ELSE
          lb_new = lb_orig
       ENDIF
    ENDIF
    IF (ub_orig.LT.ub) THEN
       IF (PRESENT(factor)) THEN
          size_increase = ub - ub_orig
          size_increase = MAX (NINT(size_increase*factor),&
                               NINT(old_size*(factor-1)),0)
          ub_new = MAX (ub_orig, ub + size_increase)
       ELSE
          ub_new = ub
       ENDIF
    ELSE
       ub_new = ub
    ENDIF
    IF(dbg) WRITE(*,*)routineP//' Resizing to bounds',lb_new,':',ub_new
    !
    ! Deallocates the old array if it's not needed to copy the old data.
    IF(.NOT.docopy) THEN
       IF (PRESENT(array_resize)) THEN
          array_resize => array
          NULLIFY(array)
       ELSE
          CALL mem_dealloc_${nametype1}$ (array, mem_type=mem_type)
       ENDIF
    ENDIF
    !
    ! Allocates the new array
    IF(lb_new /= 1) &
       DBCSR_ABORT("Arrays must start at 1")
    CALL mem_alloc_${nametype1}$ (newarray, ub_new-lb_new+1, mem_type)
    !
    ! Now copy and/or zero pad.
    IF(docopy) THEN
       IF(dbg .AND. (lb_new.GT.lb_orig .OR. ub_new.LT.ub_orig))&
            DBCSR_ABORT("Old extent exceeds the new one.")
       IF (ub_orig-lb_orig+1 .gt. 0) THEN
          !newarray(lb_orig:ub_orig) = array(lb_orig:ub_orig)
          CALL mem_copy_${nametype1}$ (newarray(lb_orig:ub_orig),&
               array(lb_orig:ub_orig), ub_orig-lb_orig+1)
       ENDIF
       IF (pad) THEN
          !newarray(lb_new:lb_orig-1) = 0
          CALL mem_zero_${nametype1}$ (newarray(lb_new:lb_orig-1), (lb_orig-1)-lb_new+1)
          !newarray(ub_orig+1:ub_new) = 0
          CALL mem_zero_${nametype1}$ (newarray(ub_orig+1:ub_new), ub_new-(ub_orig+1)+1)
       ENDIF
       IF (PRESENT(array_resize)) THEN
          array_resize => array
          NULLIFY(array)
       ELSE
          CALL mem_dealloc_${nametype1}$ (array, mem_type=mem_type)
       ENDIF
    ELSEIF (pad) THEN
       !newarray(:) = ${zero1}$
       CALL mem_zero_${nametype1}$ (newarray, SIZE(newarray))
    ENDIF
    array => newarray
    IF (dbg) WRITE(*,*)routineP//' New array size', SIZE(array)
    !CALL timestop(error_handler)
  END SUBROUTINE ensure_array_size_${nametype1}$

! **************************************************************************************************
!> \brief Copies memory area
!> \param[out] dst   destination memory
!> \param[in] src    source memory
!> \param[in] n      length of copy
! **************************************************************************************************
  SUBROUTINE mem_copy_${nametype1}$ (dst, src, n)
    INTEGER, INTENT(IN) :: n
    ${type1}$, DIMENSION(1:n), INTENT(OUT) :: dst
    ${type1}$, DIMENSION(1:n), INTENT(IN) :: src
    !$OMP PARALLEL WORKSHARE DEFAULT(none) SHARED(dst,src)
    dst(:) = src(:)
    !$OMP END PARALLEL WORKSHARE
  END SUBROUTINE mem_copy_${nametype1}$

! **************************************************************************************************
!> \brief Zeros memory area
!> \param[out] dst   destination memory
!> \param[in] n      length of elements to zero
! **************************************************************************************************
  SUBROUTINE mem_zero_${nametype1}$ (dst, n)
    INTEGER, INTENT(IN) :: n
    ${type1}$, DIMENSION(1:n), INTENT(OUT) :: dst
    !$OMP PARALLEL WORKSHARE DEFAULT(none) SHARED(dst)
    dst(:) = ${zero1}$
    !$OMP END PARALLEL WORKSHARE
  END SUBROUTINE mem_zero_${nametype1}$


! **************************************************************************************************
!> \brief Allocates memory
!> \param[out] mem        memory to allocate
!> \param[in] n           length of elements to allocate
!> \param[in] mem_type    memory type
! **************************************************************************************************
  SUBROUTINE mem_alloc_${nametype1}$ (mem, n, mem_type)
    ${type1}$, DIMENSION(:), POINTER        :: mem
    INTEGER, INTENT(IN)                   :: n
    TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
    CHARACTER(len=*), PARAMETER :: routineN = 'mem_alloc_${nametype1}$', &
      routineP = moduleN//':'//routineN
    INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

    IF (careful_mod) &
       CALL timeset (routineN, error_handle)

    IF(mem_type%acc_hostalloc .AND. n>1) THEN
       CALL acc_hostmem_allocate(mem, n, mem_type%acc_stream)
    ELSE IF(mem_type%mpi .AND. dbcsr_data_allocation%use_mpi_allocator) THEN
!$omp critical(allocate)
       CALL mp_allocate(mem, n)
!$omp end critical(allocate)
    ELSE
       ALLOCATE(mem(n))
    ENDIF

    IF (careful_mod) &
       CALL timestop (error_handle)
  END SUBROUTINE mem_alloc_${nametype1}$


! **************************************************************************************************
!> \brief Allocates memory
!> \param[out] mem        memory to allocate
!> \param[in] sizes length of elements to allocate
!> \param[in] mem_type    memory type
! **************************************************************************************************
  SUBROUTINE mem_alloc_${nametype1}$_2d (mem, sizes, mem_type)
    ${type1}$, DIMENSION(:,:), POINTER      :: mem
    INTEGER, DIMENSION(2), INTENT(IN)     :: sizes
    TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
    CHARACTER(len=*), PARAMETER :: routineN = 'mem_alloc_${nametype1}$_2d', &
      routineP = moduleN//':'//routineN
    INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

    IF (careful_mod) &
       CALL timeset (routineN, error_handle)

    IF(mem_type%acc_hostalloc) THEN
       DBCSR_ABORT("Accelerator hostalloc not supported for 2D arrays.")
       !CALL acc_hostmem_allocate(mem, n, mem_type%acc_stream)
    ELSE IF(mem_type%mpi) THEN
       DBCSR_ABORT("MPI allocate not supported for 2D arrays.")
       !CALL mp_allocate(mem, n)
    ELSE
       ALLOCATE(mem(sizes(1), sizes(2)))
    ENDIF

    IF (careful_mod) &
       CALL timestop (error_handle)
  END SUBROUTINE mem_alloc_${nametype1}$_2d


! **************************************************************************************************
!> \brief Deallocates memory
!> \param[out] mem        memory to allocate
!> \param[in] mem_type    memory type
! **************************************************************************************************
  SUBROUTINE mem_dealloc_${nametype1}$ (mem, mem_type)
    ${type1}$, DIMENSION(:), POINTER        :: mem
    TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
    CHARACTER(len=*), PARAMETER :: routineN = 'mem_dealloc_${nametype1}$', &
      routineP = moduleN//':'//routineN
    INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

    IF (careful_mod) &
       CALL timeset (routineN, error_handle)

    IF(mem_type%acc_hostalloc .AND. SIZE(mem)>1) THEN
       CALL acc_hostmem_deallocate(mem, mem_type%acc_stream)
    ELSE IF(mem_type%mpi .AND. dbcsr_data_allocation%use_mpi_allocator) THEN
       CALL mp_deallocate(mem)
    ELSE
       DEALLOCATE(mem)
    ENDIF

    IF (careful_mod) &
       CALL timestop (error_handle)
  END SUBROUTINE mem_dealloc_${nametype1}$


! **************************************************************************************************
!> \brief Deallocates memory
!> \param[out] mem        memory to allocate
!> \param[in] mem_type    memory type
! **************************************************************************************************
  SUBROUTINE mem_dealloc_${nametype1}$_2d (mem, mem_type)
    ${type1}$, DIMENSION(:,:), POINTER      :: mem
    TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
    CHARACTER(len=*), PARAMETER :: routineN = 'mem_dealloc_${nametype1}$', &
      routineP = moduleN//':'//routineN
    INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

    IF (careful_mod) &
       CALL timeset (routineN, error_handle)

    IF(mem_type%acc_hostalloc) THEN
       DBCSR_ABORT("Accelerator host deallocate not supported for 2D arrays.")
       !CALL acc_hostmem_deallocate(mem, mem_type%acc_stream)
    ELSE IF(mem_type%mpi) THEN
       DBCSR_ABORT("MPI deallocate not supported for 2D arrays.")
       !CALL mp_deallocate(mem)
    ELSE
       DEALLOCATE(mem)
    ENDIF

    IF (careful_mod) &
       CALL timestop (error_handle)
  END SUBROUTINE mem_dealloc_${nametype1}$_2d


! **************************************************************************************************
!> \brief Sets a rank-2 pointer to rank-1 data using Fortran 2003 pointer
!>        rank remapping.
!> \param r2p ...
!> \param d1 ...
!> \param d2 ...
!> \param r1p ...
! **************************************************************************************************
  SUBROUTINE pointer_${nametype1}$_rank_remap2 (r2p, d1, d2, r1p)
    INTEGER, INTENT(IN)                      :: d1, d2
    ${type1}$, DIMENSION(:, :), &
      POINTER                                :: r2p
    ${type1}$, DIMENSION(:), &
      POINTER                                :: r1p

    r2p(1:d1,1:d2) => r1p(1:d1*d2)
  END SUBROUTINE pointer_${nametype1}$_rank_remap2
#:endfor
