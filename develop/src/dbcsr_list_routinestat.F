# 1 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F" 1
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
# 11 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F" 2
# 12 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! An array-based list which grows on demand.
   !! When the internal array is full, a new array of twice the size will be
   !! allocated and the items are copied over.
   !! This list can also be used as a stack.
   !! Have look at list_push(), list_pop() and list_peek().

MODULE dbcsr_list_routinestat
   USE dbcsr_timings_base_type, ONLY: callstack_entry_type, routine_stat_type, routine_report_type


# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
#include "base/dbcsr_base_uses.f90"
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
IMPLICIT NONE
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
PRIVATE
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
PUBLIC :: list_routinestat_type, list_routinestat_init, list_routinestat_push, list_routinestat_pop, list_routinestat_peek,&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    & list_routinestat_insert, list_routinestat_set, list_routinestat_get, list_routinestat_del, list_routinestat_clear,&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    & list_routinestat_size, list_routinestat_destroy, list_routinestat_isready
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
!this is an internal type
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
TYPE private_item_type_routinestat
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   PRIVATE
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   TYPE(routine_stat_type), POINTER                   :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
END TYPE private_item_type_routinestat
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
!this is an internal type
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
TYPE private_item_p_type_routinestat
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   PRIVATE
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   TYPE(private_item_type_routinestat), POINTER :: p => Null()
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
END TYPE private_item_p_type_routinestat
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
! this is the public type, which holds a list-instance
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
TYPE list_routinestat_type
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   PRIVATE
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   TYPE(private_item_p_type_routinestat), DIMENSION(:), POINTER   :: arr => Null()
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   INTEGER                                       :: size = -1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
END TYPE list_routinestat_type
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
CONTAINS
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
FUNCTION list_routinestat_isready(list) RESULT(res)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Test if the given list has been initialized.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(list_routinestat_type), intent(in)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    LOGICAL                                     :: res
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    res = ASSOCIATED(list%arr)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
END FUNCTION list_routinestat_isready
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
SUBROUTINE list_routinestat_init(list, initial_capacity)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Allocates the internal data-structures of the given list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! This has to be called before any of the other routines.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! For deallocation call list_[valuetype]_destroy.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(list_routinestat_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    INTEGER, INTENT(in), OPTIONAL               :: initial_capacity
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
      !! The initial size of the internal array (default=11).
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    INTEGER                                     :: stat
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    INTEGER                                     :: initial_capacity_
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    initial_capacity_ = 11
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    If(PRESENT(initial_capacity)) initial_capacity_ = initial_capacity
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(initial_capacity_ < 0) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
      DBCSR_ABORT("list_routinestat_create: initial_capacity < 0")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_create: list is already initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    ALLOCATE(list%arr(initial_capacity_), stat=stat)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF (stat/=0)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_init: allocation failed")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    list%size = 0
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
END SUBROUTINE list_routinestat_init
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
SUBROUTINE list_routinestat_destroy(list)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Deallocated the internal data-structures of the given list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Caution: If the stored values are pointers, their targets will
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! not get deallocated by this routine.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(list_routinestat_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    INTEGER :: i
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_destroy: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    do i=1, list%size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       deallocate(list%arr(i)%p)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    end do
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    deallocate(list%arr)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    list%size = -1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
END SUBROUTINE list_routinestat_destroy
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
SUBROUTINE list_routinestat_set(list, value, pos)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Assings the given value to the given position in the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Thereby, the former value at that position gets overwritten.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! If the position is out of bounds, the program stops.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(list_routinestat_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(routine_stat_type), POINTER, intent(in) :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    INTEGER, intent(in) :: pos
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
      !! Position in the list - must fulfill 0 < pos < list_size+1.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_set: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(pos < 1)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_set: pos < 1")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(pos > list%size)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_set: pos > size")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    list%arr(pos)%p%value => value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
END SUBROUTINE list_routinestat_set
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
SUBROUTINE list_routinestat_push(list, value)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Appends the given value at the end of the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(list_routinestat_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(routine_stat_type), POINTER, intent(in)                  :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    INTEGER                                     :: stat
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_push: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    if(list%size == size(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       call change_capacity_routinestat(list, 2*size(list%arr)+1)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    list%size = list%size + 1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    ALLOCATE(list%arr(list%size)%p, stat=stat)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF (stat/=0)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_push: allocation failed")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    list%arr(list%size)%p%value => value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
END SUBROUTINE list_routinestat_push
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
SUBROUTINE list_routinestat_insert(list, value, pos)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Inserts the given value at the given position within the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Values which lay behind the insertion-position move one position up.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(list_routinestat_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(routine_stat_type), POINTER, intent(in) :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    INTEGER, intent(in) :: pos
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
      !! Position in the list - must fulfill 0 < pos < list_size+2 .
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    INTEGER :: i, stat
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_insert: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(pos < 1)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_insert: pos < 1")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(pos > list%size+1)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_insert: pos > size+1")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    if(list%size == size(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       call change_capacity_routinestat(list, 2*size(list%arr)+1)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    list%size = list%size + 1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    do i=list%size, pos+1, -1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       list%arr(i)%p => list%arr(i-1)%p
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    end do
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    ALLOCATE(list%arr(pos)%p, stat=stat)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
     IF (stat/=0)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
        DBCSR_ABORT("list_routinestat_insert: allocation failed.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    list%arr(pos)%p%value => value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
END SUBROUTINE list_routinestat_insert
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
FUNCTION list_routinestat_peek(list) RESULT(value)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Returns the last element in the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Is equivalent to: list_routinestat_get(list, list_routinestat_size(list))
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(list_routinestat_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(routine_stat_type), POINTER  :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_peek: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(list%size < 1) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_peek: list is empty.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    value => list%arr(list%size)%p%value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
END FUNCTION list_routinestat_peek
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
FUNCTION list_routinestat_pop(list) RESULT(value)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Returns the last element in the list and removes it.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Is equivialent to:
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! value = list_routinestat_get(list, list_routinestat_size(list))
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! call list_routinestat_del(list, list_routinestat_size(list))
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(list_routinestat_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(routine_stat_type), POINTER  :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_pop: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(list%size < 1) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_pop: list is empty.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    value => list%arr(list%size)%p%value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    deallocate(list%arr(list%size)%p)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    list%size = list%size - 1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
END FUNCTION list_routinestat_pop
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
SUBROUTINE list_routinestat_clear(list)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Removes all values from the list. The list itself is not deallocated.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(list_routinestat_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    INTEGER :: i
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_clear: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    do i=1, list%size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       deallocate(list%arr(i)%p)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    end do
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    list%size = 0
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
END SUBROUTINE list_routinestat_clear
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
!
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
FUNCTION list_routinestat_get(list, pos) RESULT(value)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Returns the value at the given position from the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(list_routinestat_type), intent(in)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    INTEGER, intent(in) :: pos
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
      !! Position in the list - must fulfill 0 < pos < list_size+1 .
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(routine_stat_type), POINTER :: value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_get: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(pos < 1)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_get: pos < 1")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(pos > list%size)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_get: pos > size")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    value => list%arr(pos)%p%value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
END FUNCTION list_routinestat_get
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
SUBROUTINE list_routinestat_del(list, pos)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Removes the value at the given position from the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(list_routinestat_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    INTEGER, intent(in) :: pos
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
      !! Position in the list - must fulfill 0 < pos < list_size+1 .
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    INTEGER :: i
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_del: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(pos < 1)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_det: pos < 1")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(pos > list%size)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_det: pos > size")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    deallocate(list%arr(pos)%p)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    do i=pos, list%size-1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       list%arr(i)%p => list%arr(i+1)%p
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    end do
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    list%size = list%size - 1
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
END SUBROUTINE list_routinestat_del
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
FUNCTION list_routinestat_size(list) RESULT(size)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Returns the current size of the list.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    TYPE(list_routinestat_type), intent(in)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    INTEGER :: size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    IF(.not. ASSOCIATED(list%arr)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
       DBCSR_ABORT("list_routinestat_size: list is not initialized.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
    size = list%size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
END FUNCTION list_routinestat_size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
SUBROUTINE change_capacity_routinestat(list, new_capacity)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   !! Internal routine for changing the size of the internal array.
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   TYPE(list_routinestat_type), intent(inout)  :: list
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   INTEGER, intent(in) :: new_capacity
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   INTEGER :: i, new_cap, stat
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   TYPE(private_item_p_type_routinestat), DIMENSION(:), POINTER :: old_arr
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   new_cap = new_capacity
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   IF(new_cap < 0) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
      DBCSR_ABORT("list_routinestat_change_capacity: new_capacity < 0")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   IF(new_cap < list%size) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
      DBCSR_ABORT("list_routinestat_change_capacity: new_capacity < size")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   IF(new_cap > HUGE(i)) THEN
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
      IF(size(list%arr) == HUGE(i)) &
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
      DBCSR_ABORT("list_routinestat_change_capacity: list has reached integer limit.")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
      new_cap = HUGE(i) ! grow as far as possible
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   END IF
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   old_arr => list%arr
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   allocate(list%arr(new_cap), stat=stat)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   IF (stat/=0)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
      DBCSR_ABORT("list_routinestat_change_capacity: allocation failed")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   do i=1, list%size
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
      allocate(list%arr(i)%p, stat=stat)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
      IF (stat/=0)&
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
         DBCSR_ABORT("list_routinestat_change_capacity: allocation failed")
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
      list%arr(i)%p%value => old_arr(i)%p%value
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
      deallocate(old_arr(i)%p)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   end do
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
   deallocate(old_arr)
# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"

# 21 "/__w/dbcsr/dbcsr/src/core/dbcsr_list_routinestat.F"
END SUBROUTINE change_capacity_routinestat

END MODULE dbcsr_list_routinestat
