!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!


   !! An array-based list which grows on demand.
   !! When the internal array is full, a new array of twice the size will be
   !! allocated and the items are copied over.
   !! This list can also be used as a stack.
   !! Have look at list_push(), list_pop() and list_peek().

MODULE dbcsr_list_timerenv
   USE dbcsr_timings_types, ONLY: timer_env_type


#include "base/dbcsr_base_uses.f90"
IMPLICIT NONE
PRIVATE

PUBLIC :: list_timerenv_type, list_timerenv_init, list_timerenv_push, list_timerenv_pop, list_timerenv_peek, list_timerenv_insert,&
    & list_timerenv_set, list_timerenv_get, list_timerenv_del, list_timerenv_clear, list_timerenv_size, list_timerenv_destroy,&
    & list_timerenv_isready

!this is an internal type
TYPE private_item_type_timerenv
   PRIVATE
   TYPE(timer_env_type), POINTER                   :: value
END TYPE private_item_type_timerenv

!this is an internal type
TYPE private_item_p_type_timerenv
   PRIVATE
   TYPE(private_item_type_timerenv), POINTER :: p => Null()
END TYPE private_item_p_type_timerenv

! this is the public type, which holds a list-instance
TYPE list_timerenv_type
   PRIVATE
   TYPE(private_item_p_type_timerenv), DIMENSION(:), POINTER   :: arr => Null()
   INTEGER                                       :: size = -1
END TYPE list_timerenv_type

CONTAINS

FUNCTION list_timerenv_isready(list) RESULT(res)
   !! Test if the given list has been initialized.
    TYPE(list_timerenv_type), intent(in)  :: list
    LOGICAL                                     :: res
    res = ASSOCIATED(list%arr)
END FUNCTION list_timerenv_isready

SUBROUTINE list_timerenv_init(list, initial_capacity)
   !! Allocates the internal data-structures of the given list.
   !! This has to be called before any of the other routines.
   !! For deallocation call list_[valuetype]_destroy.

    TYPE(list_timerenv_type), intent(inout)  :: list
    INTEGER, INTENT(in), OPTIONAL               :: initial_capacity
      !! The initial size of the internal array (default=11).
    INTEGER                                     :: stat
    INTEGER                                     :: initial_capacity_

    initial_capacity_ = 11
    If(PRESENT(initial_capacity)) initial_capacity_ = initial_capacity

    IF(initial_capacity_ < 0) &
      DBCSR_ABORT("list_timerenv_create: initial_capacity < 0")

    IF(ASSOCIATED(list%arr)) &
       DBCSR_ABORT("list_timerenv_create: list is already initialized.")

    ALLOCATE(list%arr(initial_capacity_), stat=stat)
    IF (stat/=0)&
       DBCSR_ABORT("list_timerenv_init: allocation failed")

    list%size = 0
END SUBROUTINE list_timerenv_init

SUBROUTINE list_timerenv_destroy(list)
   !! Deallocated the internal data-structures of the given list.
   !! Caution: If the stored values are pointers, their targets will
   !! not get deallocated by this routine.

    TYPE(list_timerenv_type), intent(inout)  :: list
    INTEGER :: i
    IF(.not. ASSOCIATED(list%arr)) &
       DBCSR_ABORT("list_timerenv_destroy: list is not initialized.")

    do i=1, list%size
       deallocate(list%arr(i)%p)
    end do
    deallocate(list%arr)
    list%size = -1
END SUBROUTINE list_timerenv_destroy


SUBROUTINE list_timerenv_set(list, value, pos)
   !! Assings the given value to the given position in the list.
   !! Thereby, the former value at that position gets overwritten.
   !! If the position is out of bounds, the program stops.

    TYPE(list_timerenv_type), intent(inout)  :: list
    TYPE(timer_env_type), POINTER, intent(in) :: value
    INTEGER, intent(in) :: pos
      !! Position in the list - must fulfill 0 < pos < list_size+1.
    IF(.not. ASSOCIATED(list%arr)) &
       DBCSR_ABORT("list_timerenv_set: list is not initialized.")
    IF(pos < 1)&
       DBCSR_ABORT("list_timerenv_set: pos < 1")
    IF(pos > list%size)&
       DBCSR_ABORT("list_timerenv_set: pos > size")
    list%arr(pos)%p%value => value
END SUBROUTINE list_timerenv_set


SUBROUTINE list_timerenv_push(list, value)
   !! Appends the given value at the end of the list.
    TYPE(list_timerenv_type), intent(inout)  :: list
    TYPE(timer_env_type), POINTER, intent(in)                  :: value
    INTEGER                                     :: stat

    IF(.not. ASSOCIATED(list%arr)) &
       DBCSR_ABORT("list_timerenv_push: list is not initialized.")
    if(list%size == size(list%arr)) &
       call change_capacity_timerenv(list, 2*size(list%arr)+1)

    list%size = list%size + 1
    ALLOCATE(list%arr(list%size)%p, stat=stat)
    IF (stat/=0)&
       DBCSR_ABORT("list_timerenv_push: allocation failed")
    list%arr(list%size)%p%value => value
END SUBROUTINE list_timerenv_push


SUBROUTINE list_timerenv_insert(list, value, pos)
   !! Inserts the given value at the given position within the list.
   !! Values which lay behind the insertion-position move one position up.

    TYPE(list_timerenv_type), intent(inout)  :: list
    TYPE(timer_env_type), POINTER, intent(in) :: value
    INTEGER, intent(in) :: pos
      !! Position in the list - must fulfill 0 < pos < list_size+2 .
    INTEGER :: i, stat

    IF(.not. ASSOCIATED(list%arr)) &
       DBCSR_ABORT("list_timerenv_insert: list is not initialized.")
    IF(pos < 1)&
       DBCSR_ABORT("list_timerenv_insert: pos < 1")
    IF(pos > list%size+1)&
       DBCSR_ABORT("list_timerenv_insert: pos > size+1")

    if(list%size == size(list%arr)) &
       call change_capacity_timerenv(list, 2*size(list%arr)+1)

    list%size = list%size + 1
    do i=list%size, pos+1, -1
       list%arr(i)%p => list%arr(i-1)%p
    end do

    ALLOCATE(list%arr(pos)%p, stat=stat)
     IF (stat/=0)&
        DBCSR_ABORT("list_timerenv_insert: allocation failed.")
    list%arr(pos)%p%value => value
END SUBROUTINE list_timerenv_insert


FUNCTION list_timerenv_peek(list) RESULT(value)
   !! Returns the last element in the list.
   !! Is equivalent to: list_timerenv_get(list, list_timerenv_size(list))

    TYPE(list_timerenv_type), intent(inout)  :: list
    TYPE(timer_env_type), POINTER  :: value

    IF(.not. ASSOCIATED(list%arr)) &
       DBCSR_ABORT("list_timerenv_peek: list is not initialized.")
    IF(list%size < 1) &
       DBCSR_ABORT("list_timerenv_peek: list is empty.")

    value => list%arr(list%size)%p%value
END FUNCTION list_timerenv_peek


FUNCTION list_timerenv_pop(list) RESULT(value)
   !! Returns the last element in the list and removes it.
   !! Is equivialent to:
   !! value = list_timerenv_get(list, list_timerenv_size(list))
   !! call list_timerenv_del(list, list_timerenv_size(list))

    TYPE(list_timerenv_type), intent(inout)  :: list
    TYPE(timer_env_type), POINTER  :: value

    IF(.not. ASSOCIATED(list%arr)) &
       DBCSR_ABORT("list_timerenv_pop: list is not initialized.")
    IF(list%size < 1) &
       DBCSR_ABORT("list_timerenv_pop: list is empty.")

    value => list%arr(list%size)%p%value
    deallocate(list%arr(list%size)%p)
    list%size = list%size - 1
END FUNCTION list_timerenv_pop



SUBROUTINE list_timerenv_clear(list)
   !! Removes all values from the list. The list itself is not deallocated.
    TYPE(list_timerenv_type), intent(inout)  :: list
    INTEGER :: i

    IF(.not. ASSOCIATED(list%arr)) &
       DBCSR_ABORT("list_timerenv_clear: list is not initialized.")

    do i=1, list%size
       deallocate(list%arr(i)%p)
    end do
    list%size = 0
END SUBROUTINE list_timerenv_clear


!
FUNCTION list_timerenv_get(list, pos) RESULT(value)
   !! Returns the value at the given position from the list.

    TYPE(list_timerenv_type), intent(in)  :: list
    INTEGER, intent(in) :: pos
      !! Position in the list - must fulfill 0 < pos < list_size+1 .
    TYPE(timer_env_type), POINTER :: value

    IF(.not. ASSOCIATED(list%arr)) &
       DBCSR_ABORT("list_timerenv_get: list is not initialized.")
    IF(pos < 1)&
       DBCSR_ABORT("list_timerenv_get: pos < 1")
    IF(pos > list%size)&
       DBCSR_ABORT("list_timerenv_get: pos > size")

    value => list%arr(pos)%p%value

END FUNCTION list_timerenv_get


SUBROUTINE list_timerenv_del(list, pos)
   !! Removes the value at the given position from the list.

    TYPE(list_timerenv_type), intent(inout)  :: list
    INTEGER, intent(in) :: pos
      !! Position in the list - must fulfill 0 < pos < list_size+1 .
    INTEGER :: i

    IF(.not. ASSOCIATED(list%arr)) &
       DBCSR_ABORT("list_timerenv_del: list is not initialized.")
    IF(pos < 1)&
       DBCSR_ABORT("list_timerenv_det: pos < 1")
    IF(pos > list%size)&
       DBCSR_ABORT("list_timerenv_det: pos > size")

    deallocate(list%arr(pos)%p)
    do i=pos, list%size-1
       list%arr(i)%p => list%arr(i+1)%p
    end do

    list%size = list%size - 1

END SUBROUTINE list_timerenv_del

FUNCTION list_timerenv_size(list) RESULT(size)
   !! Returns the current size of the list.
    TYPE(list_timerenv_type), intent(in)  :: list
    INTEGER :: size

    IF(.not. ASSOCIATED(list%arr)) &
       DBCSR_ABORT("list_timerenv_size: list is not initialized.")

    size = list%size
END FUNCTION list_timerenv_size


SUBROUTINE change_capacity_timerenv(list, new_capacity)
   !! Internal routine for changing the size of the internal array.
   TYPE(list_timerenv_type), intent(inout)  :: list
   INTEGER, intent(in) :: new_capacity
   INTEGER :: i, new_cap, stat
   TYPE(private_item_p_type_timerenv), DIMENSION(:), POINTER :: old_arr

   new_cap = new_capacity
   IF(new_cap < 0) &
      DBCSR_ABORT("list_timerenv_change_capacity: new_capacity < 0")
   IF(new_cap < list%size) &
      DBCSR_ABORT("list_timerenv_change_capacity: new_capacity < size")
   IF(new_cap > HUGE(i)) THEN
      IF(size(list%arr) == HUGE(i)) &
      DBCSR_ABORT("list_timerenv_change_capacity: list has reached integer limit.")
      new_cap = HUGE(i) ! grow as far as possible
   END IF

   old_arr => list%arr
   allocate(list%arr(new_cap), stat=stat)
   IF (stat/=0)&
      DBCSR_ABORT("list_timerenv_change_capacity: allocation failed")

   do i=1, list%size
      allocate(list%arr(i)%p, stat=stat)
      IF (stat/=0)&
         DBCSR_ABORT("list_timerenv_change_capacity: allocation failed")
      list%arr(i)%p%value => old_arr(i)%p%value
      deallocate(old_arr(i)%p)
   end do
   deallocate(old_arr)

END SUBROUTINE change_capacity_timerenv

END MODULE dbcsr_list_timerenv
