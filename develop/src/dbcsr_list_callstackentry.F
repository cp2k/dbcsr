# 1 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F" 1
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
# 11 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F" 2
# 12 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! An array-based list which grows on demand.
   !! When the internal array is full, a new array of twice the size will be
   !! allocated and the items are copied over.
   !! This list can also be used as a stack.
   !! Have look at list_push(), list_pop() and list_peek().

MODULE dbcsr_list_callstackentry
   USE dbcsr_timings_base_type, ONLY: routine_report_type, routine_stat_type, callstack_entry_type


# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
#include "base/dbcsr_base_uses.f90"
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
IMPLICIT NONE
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
PRIVATE
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
PUBLIC :: list_callstackentry_type, list_callstackentry_init, list_callstackentry_push, list_callstackentry_pop,&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    & list_callstackentry_peek, list_callstackentry_insert, list_callstackentry_set, list_callstackentry_get,&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    & list_callstackentry_del, list_callstackentry_clear, list_callstackentry_size, list_callstackentry_destroy,&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    & list_callstackentry_isready
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
!this is an internal type
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
TYPE private_item_type_callstackentry
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   PRIVATE
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   TYPE(callstack_entry_type)                   :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
END TYPE private_item_type_callstackentry
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
!this is an internal type
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
TYPE private_item_p_type_callstackentry
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   PRIVATE
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   TYPE(private_item_type_callstackentry), POINTER :: p => Null()
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
END TYPE private_item_p_type_callstackentry
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
! this is the public type, which holds a list-instance
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
TYPE list_callstackentry_type
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   PRIVATE
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   TYPE(private_item_p_type_callstackentry), DIMENSION(:), POINTER   :: arr => Null()
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   INTEGER                                       :: size = -1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
END TYPE list_callstackentry_type
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
CONTAINS
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
FUNCTION list_callstackentry_isready(list) RESULT(res)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Test if the given list has been initialized.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(list_callstackentry_type), intent(in)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    LOGICAL                                     :: res
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    res = ASSOCIATED(list%arr)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
END FUNCTION list_callstackentry_isready
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
SUBROUTINE list_callstackentry_init(list, initial_capacity)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Allocates the internal data-structures of the given list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! This has to be called before any of the other routines.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! For deallocation call list_[valuetype]_destroy.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(list_callstackentry_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    INTEGER, INTENT(in), OPTIONAL               :: initial_capacity
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
      !! The initial size of the internal array (default=11).
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    INTEGER                                     :: stat
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    INTEGER                                     :: initial_capacity_
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    initial_capacity_ = 11
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    If(PRESENT(initial_capacity)) initial_capacity_ = initial_capacity
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(initial_capacity_ < 0) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
      DBCSR_ABORT("list_callstackentry_create: initial_capacity < 0")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_create: list is already initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    ALLOCATE(list%arr(initial_capacity_), stat=stat)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF (stat/=0)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_init: allocation failed")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    list%size = 0
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
END SUBROUTINE list_callstackentry_init
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
SUBROUTINE list_callstackentry_destroy(list)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Deallocated the internal data-structures of the given list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Caution: If the stored values are pointers, their targets will
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! not get deallocated by this routine.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(list_callstackentry_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    INTEGER :: i
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_destroy: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    do i=1, list%size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       deallocate(list%arr(i)%p)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    end do
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    deallocate(list%arr)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    list%size = -1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
END SUBROUTINE list_callstackentry_destroy
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
SUBROUTINE list_callstackentry_set(list, value, pos)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Assings the given value to the given position in the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Thereby, the former value at that position gets overwritten.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! If the position is out of bounds, the program stops.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(list_callstackentry_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(callstack_entry_type), intent(in) :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    INTEGER, intent(in) :: pos
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
      !! Position in the list - must fulfill 0 < pos < list_size+1.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_set: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(pos < 1)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_set: pos < 1")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(pos > list%size)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_set: pos > size")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    list%arr(pos)%p%value = value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
END SUBROUTINE list_callstackentry_set
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
SUBROUTINE list_callstackentry_push(list, value)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Appends the given value at the end of the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(list_callstackentry_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(callstack_entry_type), intent(in)                  :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    INTEGER                                     :: stat
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_push: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    if(list%size == size(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       call change_capacity_callstackentry(list, 2*size(list%arr)+1)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    list%size = list%size + 1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    ALLOCATE(list%arr(list%size)%p, stat=stat)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF (stat/=0)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_push: allocation failed")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    list%arr(list%size)%p%value = value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
END SUBROUTINE list_callstackentry_push
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
SUBROUTINE list_callstackentry_insert(list, value, pos)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Inserts the given value at the given position within the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Values which lay behind the insertion-position move one position up.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(list_callstackentry_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(callstack_entry_type), intent(in) :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    INTEGER, intent(in) :: pos
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
      !! Position in the list - must fulfill 0 < pos < list_size+2 .
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    INTEGER :: i, stat
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_insert: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(pos < 1)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_insert: pos < 1")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(pos > list%size+1)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_insert: pos > size+1")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    if(list%size == size(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       call change_capacity_callstackentry(list, 2*size(list%arr)+1)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    list%size = list%size + 1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    do i=list%size, pos+1, -1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       list%arr(i)%p => list%arr(i-1)%p
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    end do
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    ALLOCATE(list%arr(pos)%p, stat=stat)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
     IF (stat/=0)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
        DBCSR_ABORT("list_callstackentry_insert: allocation failed.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    list%arr(pos)%p%value = value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
END SUBROUTINE list_callstackentry_insert
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
FUNCTION list_callstackentry_peek(list) RESULT(value)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Returns the last element in the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Is equivalent to: list_callstackentry_get(list, list_callstackentry_size(list))
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(list_callstackentry_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(callstack_entry_type)  :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_peek: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(list%size < 1) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_peek: list is empty.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    value = list%arr(list%size)%p%value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
END FUNCTION list_callstackentry_peek
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
FUNCTION list_callstackentry_pop(list) RESULT(value)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Returns the last element in the list and removes it.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Is equivialent to:
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! value = list_callstackentry_get(list, list_callstackentry_size(list))
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! call list_callstackentry_del(list, list_callstackentry_size(list))
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(list_callstackentry_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(callstack_entry_type)  :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_pop: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(list%size < 1) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_pop: list is empty.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    value = list%arr(list%size)%p%value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    deallocate(list%arr(list%size)%p)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    list%size = list%size - 1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
END FUNCTION list_callstackentry_pop
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
SUBROUTINE list_callstackentry_clear(list)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Removes all values from the list. The list itself is not deallocated.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(list_callstackentry_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    INTEGER :: i
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_clear: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    do i=1, list%size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       deallocate(list%arr(i)%p)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    end do
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    list%size = 0
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
END SUBROUTINE list_callstackentry_clear
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
!
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
FUNCTION list_callstackentry_get(list, pos) RESULT(value)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Returns the value at the given position from the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(list_callstackentry_type), intent(in)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    INTEGER, intent(in) :: pos
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
      !! Position in the list - must fulfill 0 < pos < list_size+1 .
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(callstack_entry_type) :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_get: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(pos < 1)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_get: pos < 1")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(pos > list%size)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_get: pos > size")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    value = list%arr(pos)%p%value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
END FUNCTION list_callstackentry_get
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
SUBROUTINE list_callstackentry_del(list, pos)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Removes the value at the given position from the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(list_callstackentry_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    INTEGER, intent(in) :: pos
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
      !! Position in the list - must fulfill 0 < pos < list_size+1 .
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    INTEGER :: i
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_del: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(pos < 1)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_det: pos < 1")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(pos > list%size)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_det: pos > size")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    deallocate(list%arr(pos)%p)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    do i=pos, list%size-1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       list%arr(i)%p => list%arr(i+1)%p
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    end do
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    list%size = list%size - 1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
END SUBROUTINE list_callstackentry_del
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
FUNCTION list_callstackentry_size(list) RESULT(size)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Returns the current size of the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    TYPE(list_callstackentry_type), intent(in)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    INTEGER :: size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
       DBCSR_ABORT("list_callstackentry_size: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
    size = list%size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
END FUNCTION list_callstackentry_size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
SUBROUTINE change_capacity_callstackentry(list, new_capacity)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   !! Internal routine for changing the size of the internal array.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   TYPE(list_callstackentry_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   INTEGER, intent(in) :: new_capacity
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   INTEGER :: i, new_cap, stat
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   TYPE(private_item_p_type_callstackentry), DIMENSION(:), POINTER :: old_arr
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   new_cap = new_capacity
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   IF(new_cap < 0) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
      DBCSR_ABORT("list_callstackentry_change_capacity: new_capacity < 0")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   IF(new_cap < list%size) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
      DBCSR_ABORT("list_callstackentry_change_capacity: new_capacity < size")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   IF(new_cap > HUGE(i)) THEN
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
      IF(size(list%arr) == HUGE(i)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
      DBCSR_ABORT("list_callstackentry_change_capacity: list has reached integer limit.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
      new_cap = HUGE(i) ! grow as far as possible
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   END IF
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   old_arr => list%arr
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   allocate(list%arr(new_cap), stat=stat)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   IF (stat/=0)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
      DBCSR_ABORT("list_callstackentry_change_capacity: allocation failed")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   do i=1, list%size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
      allocate(list%arr(i)%p, stat=stat)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
      IF (stat/=0)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
         DBCSR_ABORT("list_callstackentry_change_capacity: allocation failed")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
      list%arr(i)%p%value = old_arr(i)%p%value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
      deallocate(old_arr(i)%p)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   end do
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
   deallocate(old_arr)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_callstackentry.F"
END SUBROUTINE change_capacity_callstackentry

END MODULE dbcsr_list_callstackentry
