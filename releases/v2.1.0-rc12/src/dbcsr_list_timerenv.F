# 1 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F" 1
!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

# 1 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.fypp" 1
# 9 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.fypp"

# 327 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.fypp"
# 11 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F" 2
# 12 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! An array-based list which grows on demand.
   !! When the internal array is full, a new array of twice the size will be
   !! allocated and the items are copied over.
   !! This list can also be used as a stack.
   !! Have look at list_push(), list_pop() and list_peek().

MODULE dbcsr_list_timerenv
   USE dbcsr_timings_types, ONLY: timer_env_type


# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
#include "base/dbcsr_base_uses.f90"
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
IMPLICIT NONE
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
PRIVATE
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
PUBLIC :: list_timerenv_type, list_timerenv_init, list_timerenv_push, list_timerenv_pop, list_timerenv_peek, list_timerenv_insert,&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    & list_timerenv_set, list_timerenv_get, list_timerenv_del, list_timerenv_clear, list_timerenv_size, list_timerenv_destroy,&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    & list_timerenv_isready
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
!this is an internal type
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
TYPE private_item_type_timerenv
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   PRIVATE
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   TYPE(timer_env_type), POINTER                   :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
END TYPE private_item_type_timerenv
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
!this is an internal type
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
TYPE private_item_p_type_timerenv
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   PRIVATE
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   TYPE(private_item_type_timerenv), POINTER :: p => Null()
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
END TYPE private_item_p_type_timerenv
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
! this is the public type, which holds a list-instance
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
TYPE list_timerenv_type
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   PRIVATE
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   TYPE(private_item_p_type_timerenv), DIMENSION(:), POINTER   :: arr => Null()
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   INTEGER                                       :: size = -1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
END TYPE list_timerenv_type
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
CONTAINS
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
FUNCTION list_timerenv_isready(list) RESULT(res)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Test if the given list has been initialized.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(list_timerenv_type), intent(in)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    LOGICAL                                     :: res
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    res = ASSOCIATED(list%arr)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
END FUNCTION list_timerenv_isready
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
SUBROUTINE list_timerenv_init(list, initial_capacity)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Allocates the internal data-structures of the given list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! This has to be called before any of the other routines.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! For deallocation call list_[valuetype]_destroy.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(list_timerenv_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    INTEGER, INTENT(in), OPTIONAL               :: initial_capacity
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
      !! The initial size of the internal array (default=11).
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    INTEGER                                     :: stat
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    INTEGER                                     :: initial_capacity_
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    initial_capacity_ = 11
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    If(PRESENT(initial_capacity)) initial_capacity_ = initial_capacity
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(initial_capacity_ < 0) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
      DBCSR_ABORT("list_timerenv_create: initial_capacity < 0")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_create: list is already initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    ALLOCATE(list%arr(initial_capacity_), stat=stat)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF (stat/=0)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_init: allocation failed")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    list%size = 0
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
END SUBROUTINE list_timerenv_init
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
SUBROUTINE list_timerenv_destroy(list)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Deallocated the internal data-structures of the given list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Caution: If the stored values are pointers, their targets will
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! not get deallocated by this routine.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(list_timerenv_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    INTEGER :: i
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_destroy: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    do i=1, list%size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       deallocate(list%arr(i)%p)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    end do
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    deallocate(list%arr)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    list%size = -1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
END SUBROUTINE list_timerenv_destroy
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
SUBROUTINE list_timerenv_set(list, value, pos)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Assings the given value to the given position in the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Thereby, the former value at that position gets overwritten.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! If the position is out of bounds, the program stops.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(list_timerenv_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(timer_env_type), POINTER, intent(in) :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    INTEGER, intent(in) :: pos
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
      !! Position in the list - must fulfill 0 < pos < list_size+1.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_set: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(pos < 1)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_set: pos < 1")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(pos > list%size)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_set: pos > size")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    list%arr(pos)%p%value => value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
END SUBROUTINE list_timerenv_set
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
SUBROUTINE list_timerenv_push(list, value)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Appends the given value at the end of the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(list_timerenv_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(timer_env_type), POINTER, intent(in)                  :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    INTEGER                                     :: stat
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_push: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    if(list%size == size(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       call change_capacity_timerenv(list, 2*size(list%arr)+1)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    list%size = list%size + 1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    ALLOCATE(list%arr(list%size)%p, stat=stat)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF (stat/=0)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_push: allocation failed")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    list%arr(list%size)%p%value => value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
END SUBROUTINE list_timerenv_push
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
SUBROUTINE list_timerenv_insert(list, value, pos)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Inserts the given value at the given position within the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Values which lay behind the insertion-position move one position up.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(list_timerenv_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(timer_env_type), POINTER, intent(in) :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    INTEGER, intent(in) :: pos
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
      !! Position in the list - must fulfill 0 < pos < list_size+2 .
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    INTEGER :: i, stat
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_insert: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(pos < 1)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_insert: pos < 1")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(pos > list%size+1)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_insert: pos > size+1")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    if(list%size == size(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       call change_capacity_timerenv(list, 2*size(list%arr)+1)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    list%size = list%size + 1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    do i=list%size, pos+1, -1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       list%arr(i)%p => list%arr(i-1)%p
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    end do
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    ALLOCATE(list%arr(pos)%p, stat=stat)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
     IF (stat/=0)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
        DBCSR_ABORT("list_timerenv_insert: allocation failed.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    list%arr(pos)%p%value => value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
END SUBROUTINE list_timerenv_insert
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
FUNCTION list_timerenv_peek(list) RESULT(value)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Returns the last element in the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Is equivalent to: list_timerenv_get(list, list_timerenv_size(list))
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(list_timerenv_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(timer_env_type), POINTER  :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_peek: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(list%size < 1) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_peek: list is empty.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    value => list%arr(list%size)%p%value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
END FUNCTION list_timerenv_peek
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
FUNCTION list_timerenv_pop(list) RESULT(value)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Returns the last element in the list and removes it.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Is equivialent to:
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! value = list_timerenv_get(list, list_timerenv_size(list))
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! call list_timerenv_del(list, list_timerenv_size(list))
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(list_timerenv_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(timer_env_type), POINTER  :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_pop: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(list%size < 1) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_pop: list is empty.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    value => list%arr(list%size)%p%value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    deallocate(list%arr(list%size)%p)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    list%size = list%size - 1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
END FUNCTION list_timerenv_pop
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
SUBROUTINE list_timerenv_clear(list)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Removes all values from the list. The list itself is not deallocated.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(list_timerenv_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    INTEGER :: i
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_clear: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    do i=1, list%size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       deallocate(list%arr(i)%p)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    end do
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    list%size = 0
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
END SUBROUTINE list_timerenv_clear
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
!
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
FUNCTION list_timerenv_get(list, pos) RESULT(value)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Returns the value at the given position from the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(list_timerenv_type), intent(in)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    INTEGER, intent(in) :: pos
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
      !! Position in the list - must fulfill 0 < pos < list_size+1 .
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(timer_env_type), POINTER :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_get: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(pos < 1)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_get: pos < 1")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(pos > list%size)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_get: pos > size")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    value => list%arr(pos)%p%value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
END FUNCTION list_timerenv_get
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
SUBROUTINE list_timerenv_del(list, pos)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Removes the value at the given position from the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(list_timerenv_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    INTEGER, intent(in) :: pos
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
      !! Position in the list - must fulfill 0 < pos < list_size+1 .
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    INTEGER :: i
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_del: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(pos < 1)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_det: pos < 1")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(pos > list%size)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_det: pos > size")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    deallocate(list%arr(pos)%p)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    do i=pos, list%size-1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       list%arr(i)%p => list%arr(i+1)%p
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    end do
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    list%size = list%size - 1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
END SUBROUTINE list_timerenv_del
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
FUNCTION list_timerenv_size(list) RESULT(size)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Returns the current size of the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    TYPE(list_timerenv_type), intent(in)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    INTEGER :: size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
       DBCSR_ABORT("list_timerenv_size: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
    size = list%size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
END FUNCTION list_timerenv_size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
SUBROUTINE change_capacity_timerenv(list, new_capacity)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   !! Internal routine for changing the size of the internal array.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   TYPE(list_timerenv_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   INTEGER, intent(in) :: new_capacity
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   INTEGER :: i, new_cap, stat
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   TYPE(private_item_p_type_timerenv), DIMENSION(:), POINTER :: old_arr
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   new_cap = new_capacity
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   IF(new_cap < 0) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
      DBCSR_ABORT("list_timerenv_change_capacity: new_capacity < 0")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   IF(new_cap < list%size) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
      DBCSR_ABORT("list_timerenv_change_capacity: new_capacity < size")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   IF(new_cap > HUGE(i)) THEN
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
      IF(size(list%arr) == HUGE(i)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
      DBCSR_ABORT("list_timerenv_change_capacity: list has reached integer limit.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
      new_cap = HUGE(i) ! grow as far as possible
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   END IF
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   old_arr => list%arr
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   allocate(list%arr(new_cap), stat=stat)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   IF (stat/=0)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
      DBCSR_ABORT("list_timerenv_change_capacity: allocation failed")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   do i=1, list%size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
      allocate(list%arr(i)%p, stat=stat)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
      IF (stat/=0)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
         DBCSR_ABORT("list_timerenv_change_capacity: allocation failed")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
      list%arr(i)%p%value => old_arr(i)%p%value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
      deallocate(old_arr(i)%p)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   end do
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
   deallocate(old_arr)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_timerenv.F"
END SUBROUTINE change_capacity_timerenv

END MODULE dbcsr_list_timerenv
