!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright (C) 2000 - 2018  CP2K developers group                                               !
!--------------------------------------------------------------------------------------------------!

#:include 'dbcsr_array_utils.fypp'

! **************************************************************************************************
!> \brief various utilities that regard array of different kinds:
!>      output, allocation,...
!>      maybe it is not a good idea mixing output and memeory utils...
!> \par History
!>      12.2001 first version [fawzi]
!>      3.2002 templatized [fawzi]
!> \author Fawzi Mohamed
! **************************************************************************************************
MODULE dbcsr_array_utils
  USE dbcsr_machine,                   ONLY: m_flush
  USE dbcsr_log_handling,              ONLY: dbcsr_to_string

  USE dbcsr_kinds,                     ONLY: ${uselist(usekinds)}$

#include "base/dbcsr_base_uses.f90"
  IMPLICIT NONE
  PRIVATE

  LOGICAL, PRIVATE, PARAMETER :: debug_this_module=.TRUE.
  CHARACTER(len=*), PRIVATE, PARAMETER :: moduleN='dbcsr_array_utils'

#:for nametype in nametype1
  PUBLIC :: dbcsr_1d_${nametype}$_p_type, &
            dbcsr_2d_${nametype}$_p_type, &
            dbcsr_3d_${nametype}$_p_type, &
            dbcsr_1d_${nametype}$_guarantee_size, &
            dbcsr_1d_${nametype}$_write, &
            dbcsr_2d_${nametype}$_write,&
            dbcsr_2d_${nametype}$_guarantee_size,&
            dbcsr_1d_${nametype}$_bsearch
#:endfor

  ! generic interfaces
  PUBLIC :: dbcsr_guarantee_size

  INTERFACE dbcsr_guarantee_size
#:for nametype in nametype1
     MODULE PROCEDURE dbcsr_1d_${nametype}$_guarantee_size,&
                      dbcsr_2d_${nametype}$_guarantee_size
#:endfor
  END INTERFACE

!***

#:for nametype1, type1, defaultFormatType1, lessQ in inst_params

! **************************************************************************************************
!> \brief represent a pointer to a 1d array
!> \par History
!>      02.2003 created [fawzi]
!> \author fawzi
! **************************************************************************************************
  type dbcsr_1d_${nametype1}$_p_type
     ${type1}$, dimension(:), pointer :: array
  end type dbcsr_1d_${nametype1}$_p_type

! **************************************************************************************************
!> \brief represent a pointer to a 2d array
!> \par History
!>      02.2003 created [fawzi]
!> \author fawzi
! **************************************************************************************************
  type dbcsr_2d_${nametype1}$_p_type
     ${type1}$, dimension(:,:), pointer :: array
  end type dbcsr_2d_${nametype1}$_p_type

! **************************************************************************************************
!> \brief represent a pointer to a 3d array
!> \par History
!>      02.2003 created [fawzi]
!> \author fawzi
! **************************************************************************************************
  type dbcsr_3d_${nametype1}$_p_type
     ${type1}$, dimension(:,:,:), pointer :: array
  end type dbcsr_3d_${nametype1}$_p_type

#:endfor

CONTAINS

#:for nametype1, type1, defaultFormatType1, lessQ in inst_params
! **************************************************************************************************
!> \brief writes an array to the given unit
!> \param array the array to write
!> \param unit_nr the unit to write to (defaults to the standard out)
!> \param el_format the format of a single element
!> \par History
!>      4.2002 created [fawzi]
!> \author Fawzi Mohamed
!> \note
!>      maybe I will move to a comma separated paretized list
! **************************************************************************************************
SUBROUTINE dbcsr_1d_${nametype1}$_write(array, unit_nr, el_format)
  ${type1}$, INTENT(in) :: array(:)
  INTEGER, INTENT(in) :: unit_nr
  CHARACTER(len=*), INTENT(in), OPTIONAL :: el_format

  INTEGER :: iostat, i
  CHARACTER(len=*), PARAMETER :: routineN='dbcsr_1d_${nametype1}$_write',&
        routineP=moduleN//':'//routineN
  CHARACTER(len=*), PARAMETER :: defaultFormat=${defaultFormatType1}$

  WRITE(unit=unit_nr,fmt="('( ')",advance="no", iostat=iostat)
  DBCSR_ASSERT(iostat==0)
  IF (PRESENT(el_format)) THEN
     DO i=1,SIZE(array)-1
        WRITE(unit=unit_nr,fmt=el_format,advance="no") array(i)
        IF (MOD(i,5).EQ.0) THEN  ! only a few elements per line
           WRITE(unit=unit_nr,fmt="(',')")
        ELSE
           WRITE(unit=unit_nr,fmt="(',')",advance="no")
        ENDIF
     END DO
     IF (SIZE(array)>0) &
          WRITE(unit=unit_nr,fmt=el_format,advance="no") array(SIZE(array))
  ELSE
     DO i=1,SIZE(array)-1
        WRITE(unit=unit_nr,fmt=defaultFormat,advance="no") array(i)
        IF (MOD(i,5).EQ.0) THEN  ! only a few elements per line
           WRITE(unit=unit_nr,fmt="(',')")
        ELSE
           WRITE(unit=unit_nr,fmt="(',')",advance="no")
        ENDIF
     END DO
     IF (SIZE(array)>0) &
          WRITE(unit=unit_nr,fmt=defaultFormat,advance="no") array(SIZE(array))
  END IF
  WRITE(unit=unit_nr,fmt="(' )')")
  call m_flush(unit_nr)

END SUBROUTINE dbcsr_1d_${nametype1}$_write

! **************************************************************************************************
!> \brief writes an array to the given unit
!> \param array the array to write
!> \param unit_nr the unit to write to (defaults to the standard out)
!> \param el_format the format of a single element
!> \par History
!>      4.2002 created [fawzi]
!> \author Fawzi Mohamed
!> \note
!>      maybe I will move to a comma separated parentized list
! **************************************************************************************************
SUBROUTINE dbcsr_2d_${nametype1}$_write(array, unit_nr, el_format)
  ${type1}$, INTENT(in) :: array(:,:)
  INTEGER, INTENT(in) :: unit_nr
  CHARACTER(len=*), INTENT(in), OPTIONAL :: el_format

  INTEGER :: iostat, i
  CHARACTER(len=*), PARAMETER :: routineN='dbcsr_2d_${nametype1}$_write',&
        routineP=moduleN//':'//routineN
  CHARACTER(len=*), PARAMETER :: defaultFormat=${defaultFormatType1}$
  CHARACTER(len=200) :: fmtstr
  CHARACTER(len=10) :: nRiga

  nRiga=dbcsr_to_string(SIZE(array,2))
  DO i=1,SIZE(array,1)
     IF (PRESENT(el_format)) THEN
        fmtstr='(" ",'//nRiga//el_format//')'
        WRITE(unit=unit_nr,fmt=fmtstr,iostat=iostat) array(i,:)
     ELSE
        fmtstr = '(" ",'//nRiga//defaultFormat//')'
        WRITE(unit=unit_nr,fmt=fmtstr,iostat=iostat) array(i,:)
     END IF
     DBCSR_ASSERT(iostat==0)
  END DO
  call m_flush(unit_nr)
END SUBROUTINE dbcsr_2d_${nametype1}$_write

! **************************************************************************************************
!> \brief If the size of the array is changes reallocate it.
!>      Issues a warning when the size changes (but not on allocation
!>      and deallocation).
!>
!>      The data is NOT preserved (if you want to preserve the data see
!>      the realloc in the module memory_utilities)
!> \param array the array to reallocate if necessary
!> \param n the wanted size
!> \par History
!>      12.2001 first version [fawzi]
!>      3.2002 templatized [fawzi]
!> \author Fawzi Mohamed
!> \note
!>      this is a different behaviour than the realloc in the module
!>      memory_utilities. It is quite low level
! **************************************************************************************************
  SUBROUTINE dbcsr_1d_${nametype1}$_guarantee_size(array, n)
    ${type1}$, POINTER :: array(:)
    INTEGER, INTENT(in) :: n

    CHARACTER(len=*), PARAMETER :: routineN='dbcsr_1d_${nametype1}$_guarantee_size',&
         routineP=moduleN//':'//routineN

    DBCSR_ASSERT(n>=0)
    IF (ASSOCIATED(array)) THEN
       IF (SIZE(array) /= n) THEN
          DBCSR_WARN('size has changed')
          DEALLOCATE(array)
       END IF
    END IF
    IF (.NOT.ASSOCIATED(array)) THEN
       ALLOCATE(array(n))
    END IF
  END SUBROUTINE dbcsr_1d_${nametype1}$_guarantee_size

! **************************************************************************************************
!> \brief If the size of the array is changes reallocate it.
!>      Issues a warning when the size changes (but not on allocation
!>      and deallocation).
!>
!>      The data is NOT preserved (if you want to preserve the data see
!>      the realloc in the module memory_utilities)
!> \param array the array to reallocate if necessary
!> \param n_rows the wanted number of rows
!> \param n_cols the wanted number of cols
!> \par History
!>      5.2001 first version [fawzi]
!> \author Fawzi Mohamed
!> \note
!>      this is a different behaviour than the realloc in the module
!>      memory_utilities. It is quite low level
! **************************************************************************************************
  SUBROUTINE dbcsr_2d_${nametype1}$_guarantee_size(array, n_rows, n_cols)
    ${type1}$, POINTER :: array(:,:)
    INTEGER, INTENT(in) :: n_rows, n_cols

    CHARACTER(len=*), PARAMETER :: routineN='dbcsr_2d_${nametype1}$_guarantee_size',&
         routineP=moduleN//':'//routineN

    DBCSR_ASSERT(n_cols>=0)
    DBCSR_ASSERT(n_rows>=0)
    IF (ASSOCIATED(array)) THEN
       IF (SIZE(array,1) /= n_rows .OR. SIZE(array,2) /= n_cols) THEN
          DBCSR_WARN('size has changed')
          DEALLOCATE(array)
       END IF
    END IF
    IF (.NOT.ASSOCIATED(array)) THEN
       ALLOCATE(array(n_rows,n_cols))
    END IF
  END SUBROUTINE dbcsr_2d_${nametype1}$_guarantee_size

! **************************************************************************************************
!> \brief returns the index at which the element el should be inserted in the
!>      array to keep it ordered (array(i)>=el).
!>      If the element is bigger than all the elements in the array returns
!>      the last index+1.
!> \param array the array to search
!> \param el the element to look for
!> \param l_index the lower index for binary search (defaults to 1)
!> \param u_index the upper index for binary search (defaults to size(array))
!> \return ...
!> \par History
!>      06.2003 created [fawzi]
!> \author Fawzi Mohamed
!> \note
!>      the array should be ordered in growing order
! **************************************************************************************************
FUNCTION dbcsr_1d_${nametype1}$_bsearch(array, el, l_index, u_index)&
     result(res)
  ${type1}$, intent(in) :: array(:)
  ${type1}$, intent(in) :: el
  INTEGER, INTENT(in), OPTIONAL :: l_index, u_index
  integer :: res

  character(len=*), parameter :: routineN='dbcsr_1d_${nametype1}$_bsearch',&
        routineP=moduleN//':'//routineN
  INTEGER :: lindex,uindex,aindex

  lindex=1
  uindex=size(array)
  if (present(l_index)) lindex=l_index
  if (present(u_index)) uindex=u_index
  DO WHILE (lindex <= uindex)
     aindex=(lindex+uindex)/2
     IF (@{lessQ(array(aindex),el)}@) THEN
        lindex=aindex+1
     ELSE
        uindex=aindex-1
     END IF
  END DO
  res=lindex
END FUNCTION dbcsr_1d_${nametype1}$_bsearch
#:endfor

END MODULE dbcsr_array_utils
