!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

#:include 'dbcsr.fypp'
#:for n, nametype1, base1, prec1, kind1, type1, dkind1 in inst_params_float
  SUBROUTINE set_data_p_${nametype1}$ (area, p)
     !! Sets a data pointer.
     !!
     !! Assumptions
     !! Assumes that no memory will be lost when repointing the
     !! pointer in the data area and that the area is initialized.

     TYPE(dbcsr_data_obj), INTENT(INOUT)      :: area
        !! target data area
     ${type1}$, DIMENSION(:), POINTER :: p
        !! source data pointer

     CHARACTER(len=*), PARAMETER :: routineN = 'set_data_p_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

!   ---------------------------------------------------------------------------
     IF (area%d%data_type /= ${dkind1}$) &
        DBCSR_ABORT("set_data_p_${nametype1}$: data-area has wrong type")

     area%d%${base1}$_${prec1}$ => p
  END SUBROUTINE set_data_p_${nametype1}$

  SUBROUTINE set_data_p_2d_${nametype1}$ (area, p)
     !! Sets a data pointer.
     !!
     !! Assumptions
     !! Assumes that no memory will be lost when repointing the
     !! pointer in the data area and that the area is initialized.

     TYPE(dbcsr_data_obj), INTENT(INOUT)      :: area
        !! target data area
     ${type1}$, DIMENSION(:, :), POINTER         :: p
        !! source data pointer

     CHARACTER(len=*), PARAMETER :: routineN = 'set_data_p_2d_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

!   ---------------------------------------------------------------------------

     IF (area%d%data_type /= ${dkind1}$_2d) &
        DBCSR_ABORT("set_data_p_2d_${nametype1}$: data-area has wrong type")

     area%d%${base1}$2_${prec1}$ => p
  END SUBROUTINE set_data_p_2d_${nametype1}$

  FUNCTION dbcsr_get_data_c_${nametype1}$ (area, select_data_type, lb, ub) RESULT(DATA)
     !! Returns the single/double precision real/complex data
     !!
     !! Calling
     !! This routine is hidden behind the dbcsr_get_data interface, hence the
     !! need for the select_data_type argument.
     !! see dbcsr_get_data_p_${nametype1}$

     TYPE(dbcsr_data_obj), INTENT(IN)         :: area
        !! data area
     ${type1}$, INTENT(IN)            :: select_data_type
        !! force datatype
     INTEGER, INTENT(IN), OPTIONAL  :: lb, ub
        !! lower bound for pointer
        !! upper bound for pointer
     ${type1}$, DIMENSION(:), POINTER :: DATA
        !! pointer to data

     CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_data_c_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER                        :: l, u

!   ---------------------------------------------------------------------------

     ! The select_data_type argument is needed to make this function unique
     ! enough to use in the interface.
     IF (KIND(select_data_type) .NE. KIND(DATA)) &
        DBCSR_ABORT("compiler borken")

     IF (ASSOCIATED(area%d)) THEN
        IF (area%d%data_type /= ${dkind1}$) &
           DBCSR_ABORT("dbcsr_get_data_c_${nametype1}$: data-area has wrong type")
        IF (PRESENT(lb) .OR. PRESENT(ub)) THEN
           l = LBOUND(area%d%${base1}$_${prec1}$, 1)
           IF (PRESENT(lb)) l = lb
           u = UBOUND(area%d%${base1}$_${prec1}$, 1)
           IF (PRESENT(ub)) u = ub
           IF (debug_mod) THEN
              IF (l .LT. LBOUND(area%d%${base1}$_${prec1}$, 1)) &
                 DBCSR_ABORT("Out of bounds")
              IF (u .GT. UBOUND(area%d%${base1}$_${prec1}$, 1)) &
                 DBCSR_ABORT("Out of bounds")
           ENDIF
           DATA => area%d%${base1}$_${prec1}$ (l:u)
        ELSE
           DATA => area%d%${base1}$_${prec1}$
        ENDIF
     ELSE
        NULLIFY (DATA)
     ENDIF
  END FUNCTION dbcsr_get_data_c_${nametype1}$

  FUNCTION dbcsr_get_data_p_${nametype1}$ (area, lb, ub) RESULT(DATA)
     !! Returns the single/double precision real/complex data
     !! \brief dbcsr_get_data_c_${nametype1}$
     !!
     !! Calling
     !! This routine can be called explicitly.

     TYPE(dbcsr_data_obj), INTENT(IN)         :: area
        !! data area
     ${type1}$, DIMENSION(:), POINTER :: DATA
        !! pointer to data
     INTEGER, INTENT(IN), OPTIONAL  :: lb, ub
        !! lower bound for pointer
        !! upper bound for pointer

     CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_data_p_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER                        :: l, u
!   ---------------------------------------------------------------------------

     IF (ASSOCIATED(area%d)) THEN
        IF (area%d%data_type /= ${dkind1}$) &
           DBCSR_ABORT("dbcsr_get_data_p_${nametype1}$: data-area has wrong type")
        IF (PRESENT(lb) .OR. PRESENT(ub)) THEN
           l = LBOUND(area%d%${base1}$_${prec1}$, 1)
           IF (PRESENT(lb)) l = lb
           u = UBOUND(area%d%${base1}$_${prec1}$, 1)
           IF (PRESENT(ub)) u = ub
           IF (debug_mod) THEN
              IF (l .LT. LBOUND(area%d%${base1}$_${prec1}$, 1)) &
                 DBCSR_ABORT("Out of bounds")
              IF (u .GT. UBOUND(area%d%${base1}$_${prec1}$, 1)) &
                 DBCSR_ABORT("Out of bounds")
           ENDIF
           DATA => area%d%${base1}$_${prec1}$ (l:u)
        ELSE
           DATA => area%d%${base1}$_${prec1}$
        ENDIF
     ELSE
        NULLIFY (DATA)
     ENDIF
  END FUNCTION dbcsr_get_data_p_${nametype1}$

  FUNCTION dbcsr_get_data_p_2d_${nametype1}$ (area, lb, ub) RESULT(DATA)
     !! Returns the single/double precision real/complex data
     !! \brief dbcsr_get_data_c_${nametype1}$
     !!
     !! Calling
     !! This routine can be called explicitly.

     TYPE(dbcsr_data_obj), INTENT(IN)            :: area
        !! data area
     ${type1}$, DIMENSION(:, :), POINTER            :: DATA
        !! pointer to data
     INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL :: lb, ub
        !! lower bound for pointer
        !! upper bound for pointer

     CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_data_p_2d_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER, DIMENSION(2)          :: l, u
!   ---------------------------------------------------------------------------

     IF (ASSOCIATED(area%d)) THEN
        IF (area%d%data_type /= ${dkind1}$_2d) &
           DBCSR_ABORT("dbcsr_get_data_p_2d_${nametype1}$: data-area has wrong type")
        IF (PRESENT(lb) .OR. PRESENT(ub)) THEN
           l = LBOUND(area%d%${base1}$2_${prec1}$)
           IF (PRESENT(lb)) l = lb
           u = UBOUND(area%d%${base1}$2_${prec1}$)
           IF (PRESENT(ub)) u = ub
           IF (debug_mod) THEN
              IF (l(1) .LT. LBOUND(area%d%${base1}$2_${prec1}$, 1)) &
                 DBCSR_ABORT("Out of bounds")
              IF (l(2) .LT. LBOUND(area%d%${base1}$2_${prec1}$, 2)) &
                 DBCSR_ABORT("Out of bounds")
              IF (u(1) .GT. UBOUND(area%d%${base1}$2_${prec1}$, 1)) &
                 DBCSR_ABORT("Out of bounds")
              IF (u(2) .GT. UBOUND(area%d%${base1}$2_${prec1}$, 2)) &
                 DBCSR_ABORT("Out of bounds")
           ENDIF
           DATA => area%d%${base1}$2_${prec1}$ (l(1):u(1), l(2):u(2))
        ELSE
           DATA => area%d%${base1}$2_${prec1}$
        ENDIF
     ELSE
        NULLIFY (DATA)
     ENDIF
  END FUNCTION dbcsr_get_data_p_2d_${nametype1}$

  SUBROUTINE get_data_${nametype1}$ (area, DATA, lb, ub)
     !! Returns the single/double precision real/complex data

     TYPE(dbcsr_data_obj), INTENT(IN)  :: area
        !! data area
     ${type1}$, DIMENSION(:), POINTER    :: DATA
        !! pointer to data
     INTEGER, INTENT(IN), OPTIONAL     :: lb, ub
        !! lower bound for pointer
        !! upper bound for pointer

     CHARACTER(len=*), PARAMETER :: routineN = 'get_data_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER                        :: l, u
!   ---------------------------------------------------------------------------

     IF (ASSOCIATED(area%d)) THEN
        IF (area%d%data_type /= ${dkind1}$) &
           DBCSR_ABORT("get_data_${nametype1}$: data-area has wrong type")
        IF (PRESENT(lb) .OR. PRESENT(ub)) THEN
           l = LBOUND(area%d%${base1}$_${prec1}$, 1)
           IF (PRESENT(lb)) l = lb
           u = UBOUND(area%d%${base1}$_${prec1}$, 1)
           IF (PRESENT(ub)) u = ub
           IF (debug_mod) THEN
              IF (l < LBOUND(area%d%${base1}$_${prec1}$, 1)) &
                 DBCSR_ABORT("Out of bounds")
              IF (u > UBOUND(area%d%${base1}$_${prec1}$, 1)) &
                 DBCSR_ABORT("Out of bounds")
           ENDIF
           DATA => area%d%${base1}$_${prec1}$ (l:u)
        ELSE
           DATA => area%d%${base1}$_${prec1}$
        ENDIF
     ELSE
        NULLIFY (DATA)
     ENDIF
  END SUBROUTINE get_data_${nametype1}$

  SUBROUTINE get_data_2d_${nametype1}$ (area, DATA, lb, ub)
     !! Returns the single/double precision real/complex data

     TYPE(dbcsr_data_obj), INTENT(IN)            :: area
        !! data area
     ${type1}$, DIMENSION(:, :), POINTER            :: DATA
        !! pointer to data
     INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL :: lb, ub
        !! lower bound for pointer
        !! upper bound for pointer

     CHARACTER(len=*), PARAMETER :: routineN = 'get_data_2d_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER, DIMENSION(2)          :: l, u
!   ---------------------------------------------------------------------------

     IF (ASSOCIATED(area%d)) THEN
        IF (area%d%data_type /= ${dkind1}$_2d) &
           DBCSR_ABORT("get_data_2d_${nametype1}$: data-area has wrong type")
        IF (PRESENT(lb) .OR. PRESENT(ub)) THEN
           l = LBOUND(area%d%${base1}$2_${prec1}$)
           IF (PRESENT(lb)) l = lb
           u = UBOUND(area%d%${base1}$2_${prec1}$)
           IF (PRESENT(ub)) u = ub
           IF (debug_mod) THEN
              IF (l(1) < LBOUND(area%d%${base1}$2_${prec1}$, 1)) &
                 DBCSR_ABORT("Out of bounds")
              IF (l(2) < LBOUND(area%d%${base1}$2_${prec1}$, 2)) &
                 DBCSR_ABORT("Out of bounds")
              IF (u(1) > UBOUND(area%d%${base1}$2_${prec1}$, 1)) &
                 DBCSR_ABORT("Out of bounds")
              IF (u(2) > UBOUND(area%d%${base1}$2_${prec1}$, 2)) &
                 DBCSR_ABORT("Out of bounds")
           ENDIF
           DATA => area%d%${base1}$2_${prec1}$ (l(1):u(1), l(2):u(2))
        ELSE
           DATA => area%d%${base1}$2_${prec1}$
        ENDIF
     ELSE
        NULLIFY (DATA)
     ENDIF
  END SUBROUTINE get_data_2d_${nametype1}$

  ELEMENTAL FUNCTION dbcsr_scalar_${nametype1}$ (scalar) RESULT(encapsulated_scalar)
     !! Sets a scalar in an encapsulated data structure

     ${type1}$, INTENT(IN)       :: scalar
        !! scalar to encapsulate
     TYPE(dbcsr_scalar_type)   :: encapsulated_scalar
        !! encapsulated scalar

     CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_scalar_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

!   ---------------------------------------------------------------------------

     encapsulated_scalar = dbcsr_scalar_zero(${dkind1}$)
     encapsulated_scalar%${base1}$_${prec1}$ = scalar
  END FUNCTION dbcsr_scalar_${nametype1}$

  ELEMENTAL SUBROUTINE dbcsr_scalar_get_value_${nametype1}$ (encapsulated_scalar, value)
     !! Sets a scalar in an encapsulated data structure

     TYPE(dbcsr_scalar_type), INTENT(IN) :: encapsulated_scalar
        !! encapsulated scalar
     ${type1}$, INTENT(OUT)                :: value
        !! value of the scalar

     CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_scalar_get_value_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

!   ---------------------------------------------------------------------------

     value = encapsulated_scalar%${base1}$_${prec1}$
  END SUBROUTINE dbcsr_scalar_get_value_${nametype1}$
#:endfor
