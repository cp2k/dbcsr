!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

#:include 'dbcsr_dict.fypp'
MODULE dbcsr_dict
   !! A dictionary (also known as hashtable or hashmap).
   !! Internally the dictionary uses an array to holds its data.
   !! If this array reaches a load-factor of 75%, a new array with twice the
   !! size will be allocated and the items are then copied over.
   !! This ensures that the dictionary will perform operations in O(1).

   USE dbcsr_kinds, ONLY: ${uselist(usekinds)}$
   USE dbcsr_timings_base_type, ONLY: ${uselist(usetimings)}$
#include "base/dbcsr_base_uses.f90"
   IMPLICIT NONE

   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_dict'

   PUBLIC :: dict_init, dict_items, dict_haskey, &
             dict_set, dict_get, dict_size, dict_destroy

#:for keytype, valuetype in keyvalue_list
   PUBLIC :: dict_${keytype}$_${valuetype}$_type
   PUBLIC :: dict_${keytype}$_${valuetype}$_item_type
#:endfor

#:for keytype, valuetype, keytype_fort, valuetype_fort, key_assign, value_assign, isequal in inst_params
   !this is an internal type
   !Calculating hashes might be expensive, therefore they are stored
   !for use during change_capacity().
   TYPE private_item_type_${keytype}$_${valuetype}$
      PRIVATE
      ${keytype_fort}$                            :: key
      ${valuetype_fort}$                          :: value
      INTEGER(KIND=int_8)                         :: hash
      TYPE(private_item_type_${keytype}$_${valuetype}$), POINTER          :: next => Null()
   END TYPE private_item_type_${keytype}$_${valuetype}$

   !this is an internal type
   TYPE private_item_p_type_${keytype}$_${valuetype}$
      PRIVATE
      TYPE(private_item_type_${keytype}$_${valuetype}$), POINTER :: p => Null()
   END TYPE private_item_p_type_${keytype}$_${valuetype}$

   ! this is the public type, which holds a dictionary-instance
   TYPE dict_${keytype}$_${valuetype}$_type
      PRIVATE
      TYPE(private_item_p_type_${keytype}$_${valuetype}$), DIMENSION(:), POINTER      :: buckets => Null()
      INTEGER                                               :: size = -1
   END TYPE dict_${keytype}$_${valuetype}$_type

   ! this is a public type, its returned by dict_items()
   TYPE dict_${keytype}$_${valuetype}$_item_type
      ${keytype_fort}$        :: key
      ${valuetype_fort}$      :: value
   END TYPE dict_${keytype}$_${valuetype}$_item_type
#:endfor

   INTERFACE dict_init
#:for keytype, valuetype in keyvalue_list
      MODULE PROCEDURE dict_${keytype}$_${valuetype}$_init
#:endfor
   END INTERFACE

   INTERFACE dict_isready
#:for keytype, valuetype in keyvalue_list
      MODULE PROCEDURE dict_${keytype}$_${valuetype}$_isready
#:endfor
   END INTERFACE

   INTERFACE dict_haskey
#:for keytype, valuetype in keyvalue_list
      MODULE PROCEDURE dict_${keytype}$_${valuetype}$_haskey
#:endfor
   END INTERFACE

   INTERFACE dict_set
#:for keytype, valuetype in keyvalue_list
      MODULE PROCEDURE dict_${keytype}$_${valuetype}$_set
#:endfor
   END INTERFACE

   INTERFACE dict_get
#:for keytype, valuetype in keyvalue_list
      MODULE PROCEDURE dict_${keytype}$_${valuetype}$_get
#:endfor
   END INTERFACE

   INTERFACE dict_del
#:for keytype, valuetype in keyvalue_list
      MODULE PROCEDURE dict_${keytype}$_${valuetype}$_del
#:endfor
   END INTERFACE

   INTERFACE dict_items
#:for keytype, valuetype in keyvalue_list
      MODULE PROCEDURE dict_${keytype}$_${valuetype}$_items
#:endfor
   END INTERFACE

   INTERFACE dict_size
#:for keytype, valuetype in keyvalue_list
      MODULE PROCEDURE dict_${keytype}$_${valuetype}$_size
#:endfor
   END INTERFACE

   INTERFACE dict_update
#:for keytype, valuetype in keyvalue_list
      MODULE PROCEDURE dict_${keytype}$_${valuetype}$_update
#:endfor
   END INTERFACE

   INTERFACE dict_destroy
#:for keytype, valuetype in keyvalue_list
      MODULE PROCEDURE dict_${keytype}$_${valuetype}$_destroy
#:endfor
   END INTERFACE

CONTAINS

#:for fct in hash_fct
$:fct
#:endfor

#:for keytype, valuetype, keytype_fort, valuetype_fort, key_assign, value_assign, isequal in inst_params
   SUBROUTINE dict_${keytype}$_${valuetype}$_init(dict, initial_capacity)
      !! Allocates the internal data-structures of the given dictionary.

      TYPE(dict_${keytype}$_${valuetype}$_type), intent(inout)  :: dict
      INTEGER, INTENT(in), OPTIONAL                         :: initial_capacity
         !! The initial size of the internal array (default=11).

      INTEGER :: initial_capacity_
      initial_capacity_ = 11
      IF (PRESENT(initial_capacity)) initial_capacity_ = initial_capacity

      IF (initial_capacity_ < 1) &
         DBCSR_ABORT("dict_${keytype}$_${valuetype}$_init: initial_capacity < 1")

      IF (ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_${keytype}$_${valuetype}$_init: dictionary is already initialized.")

      ALLOCATE (dict%buckets(initial_capacity_))
      dict%size = 0

   END SUBROUTINE dict_${keytype}$_${valuetype}$_init

   FUNCTION dict_${keytype}$_${valuetype}$_isready(dict) RESULT(res)
      !! Test if the given dictionary has been initialized.
      TYPE(dict_${keytype}$_${valuetype}$_type), intent(inout)  :: dict
      LOGICAL                                               :: res
      res = ASSOCIATED(dict%buckets)
   END FUNCTION dict_${keytype}$_${valuetype}$_isready

   SUBROUTINE dict_${keytype}$_${valuetype}$_destroy(dict)
      !! Deallocated the internal data-structures if the given dictionary.
      !! Caution: If the stored keys or values are pointers, their targets will
      !! not get deallocated by this routine.

      TYPE(dict_${keytype}$_${valuetype}$_type), intent(inout)  :: dict
      TYPE(private_item_type_${keytype}$_${valuetype}$), POINTER  :: item, prev_item
      INTEGER :: i

      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_${keytype}$_${valuetype}$_destroy: dictionary is not initialized.")

      do i = 1, size(dict%buckets)
         item => dict%buckets(i)%p
         do while (ASSOCIATED(item))
            prev_item => item
            item => item%next
            deallocate (prev_item)
         end do
      end do

      deallocate (dict%buckets)
      dict%size = -1

   END SUBROUTINE dict_${keytype}$_${valuetype}$_destroy

   SUBROUTINE dict_${keytype}$_${valuetype}$_set(dict, key, value)
      !! Stores, and possibly overwrites, a given value under a given key.
      TYPE(dict_${keytype}$_${valuetype}$_type), intent(inout)  :: dict
      ${keytype_fort}$, intent(in)                          :: key
      ${valuetype_fort}$, intent(in)                          :: value
      INTEGER(KIND=int_8)                                   :: hash
      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_${keytype}$_${valuetype}$_set: dictionary is not initialized.")

      hash = hash_${keytype}$ (key)
      call set_hashed_${keytype}$_${valuetype}$ (dict, key, value, hash)
   END SUBROUTINE dict_${keytype}$_${valuetype}$_set

   RECURSIVE SUBROUTINE set_hashed_${keytype}$_${valuetype}$ (dict, key, value, hash)
      !! Common code used internally by dict_set() and change_capacity().
      TYPE(dict_${keytype}$_${valuetype}$_type), intent(inout)  :: dict
      ${keytype_fort}$, intent(in)                          :: key
      ${valuetype_fort}$, intent(in)                          :: value
      INTEGER(KIND=int_8), intent(in)                       :: hash
      TYPE(private_item_type_${keytype}$_${valuetype}$), POINTER  :: item, new_item
      INTEGER(KIND=int_8)                                   :: idx

      idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1

      ! if already in dict just update its value
      item => dict%buckets(idx)%p
      do while (ASSOCIATED(item))
         IF (item%hash == hash) THEN
            IF (@{isequal(item%key, key)}@) THEN
               item%value ${value_assign}$value
               return
            END IF
         END IF
         item => item%next
      end do

      ! check load-factor
      IF (4*dict%size > 3*size(dict%buckets)) THEN ! load-factor > 75%
         call change_capacity_${keytype}$_${valuetype}$ (dict, 2*size(dict%buckets)) !double capacity
         idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1
      END IF

      ! create a new item
      allocate (new_item)
      new_item%hash = hash
      new_item%key ${key_assign}$key
      new_item%value ${value_assign}$value
      new_item%next => dict%buckets(idx)%p
      dict%buckets(idx)%p => new_item
      dict%size = dict%size + 1

   END SUBROUTINE set_hashed_${keytype}$_${valuetype}$

   RECURSIVE SUBROUTINE change_capacity_${keytype}$_${valuetype}$ (dict, new_capacity)
      !! Internal routine for changing the dictionary's capacity.
      TYPE(dict_${keytype}$_${valuetype}$_type), intent(inout)  :: dict
      INTEGER, intent(in) :: new_capacity
      INTEGER :: i, old_size, new_cap
      TYPE(private_item_type_${keytype}$_${valuetype}$), POINTER  :: item, prev_item
      TYPE(private_item_p_type_${keytype}$_${valuetype}$), DIMENSION(:), POINTER  :: old_buckets
      new_cap = new_capacity
      ! pre checks
      IF (new_cap > HUGE(i)) THEN
         IF (size(dict%buckets) == HUGE(i)) RETURN ! reached maximum - stay there.
         new_cap = HUGE(i) ! grow as far as possible
      END IF
      IF (new_cap < 1) &
         DBCSR_ABORT("dict_${keytype}$_${valuetype}$_change_capacity: new_capacity < 1.")
      IF (4*dict%size > 3*new_cap) &
         DBCSR_ABORT("dict_${keytype}$_${valuetype}$_change_capacity: new_capacity too small.")

      old_size = dict%size
      old_buckets => dict%buckets
      ALLOCATE (dict%buckets(new_capacity))
      dict%size = 0
      do i = 1, size(old_buckets)
         item => old_buckets(i)%p
         do while (ASSOCIATED(item))
            call set_hashed_${keytype}$_${valuetype}$ (dict, item%key, item%value, item%hash)
            prev_item => item
            item => item%next
            deallocate (prev_item)
         end do
      end do

      deallocate (old_buckets)

      IF (old_size /= dict%size) &
         DBCSR_ABORT("dict_${keytype}$_${valuetype}$_change_capacity: assertion failed")
   END SUBROUTINE change_capacity_${keytype}$_${valuetype}$

   FUNCTION dict_${keytype}$_${valuetype}$_get(dict, key, default_value) RESULT(value)
      !! Gets a value for a given key from the dictionary.
      !! If the key is not found the default_value will be returned.
      !! If the key is not found and default_value was not provided the program stops.
      TYPE(dict_${keytype}$_${valuetype}$_type), intent(inout)  :: dict
      ${keytype_fort}$                                        :: key
      ${valuetype_fort}$, intent(in), optional                :: default_value
      ${valuetype_fort}$                                      :: value
      TYPE(private_item_type_${keytype}$_${valuetype}$), POINTER                      :: item
      INTEGER(KIND=int_8)                                   :: hash, idx

      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_${keytype}$_${valuetype}$_get: dictionary is not initialized.")

      hash = hash_${keytype}$ (key)
      idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1

      item => dict%buckets(idx)%p
      do while (ASSOCIATED(item))
         IF (item%hash == hash) THEN
            IF (@{isequal(item%key, key)}@) THEN
               value ${value_assign}$item%value
               return
            END IF
         END IF
         item => item%next
      end do

      IF (PRESENT(default_value)) THEN
         value ${value_assign}$default_value
         return
      END IF

      DBCSR_ABORT("dict_${keytype}$_${valuetype}$_get: Key not found in dictionary.")
   END FUNCTION dict_${keytype}$_${valuetype}$_get

   SUBROUTINE dict_${keytype}$_${valuetype}$_del(dict, key)
      !! Remove the value for a given key from the dictionary.
      !! If the key is not found the program stops.

      TYPE(dict_${keytype}$_${valuetype}$_type), intent(inout)  :: dict
      ${keytype_fort}$                                        :: key
      TYPE(private_item_type_${keytype}$_${valuetype}$), POINTER                      :: item, prev_item
      INTEGER(KIND=int_8)                                   :: hash, idx

      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_${keytype}$_${valuetype}$_del: dictionary is not initialized.")

      hash = hash_${keytype}$ (key)
      idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1

      item => dict%buckets(idx)%p
      prev_item => Null()
      do while (ASSOCIATED(item))
         IF (item%hash == hash) THEN
            IF (@{isequal(item%key, key)}@) THEN
               IF (ASSOCIATED(prev_item)) THEN
                  prev_item%next => item%next
               ELSE
                  dict%buckets(idx)%p => item%next
               END IF
               deallocate (item)
               dict%size = dict%size - 1
               return
            END IF
         END IF
         prev_item => item
         item => item%next
      end do

      DBCSR_ABORT("dict_${keytype}$_${valuetype}$_del: Key not found in dictionary.")
   END SUBROUTINE dict_${keytype}$_${valuetype}$_del

   FUNCTION dict_${keytype}$_${valuetype}$_size(dict) RESULT(size)
      !! Returns the number of key/value-items currently stored in the dictionary.
      TYPE(dict_${keytype}$_${valuetype}$_type), intent(inout)  :: dict
      INTEGER :: size

      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_${keytype}$_${valuetype}$_size: dictionary is not initialized.")

      size = dict%size
   END FUNCTION dict_${keytype}$_${valuetype}$_size

   FUNCTION dict_${keytype}$_${valuetype}$_haskey(dict, key) RESULT(res)
      !! Checks whether a given key is currently stored in the dictionary.
      TYPE(dict_${keytype}$_${valuetype}$_type), intent(inout)  :: dict
      ${keytype_fort}$                                        :: key
      LOGICAL                                               :: res
      TYPE(private_item_type_${keytype}$_${valuetype}$), POINTER                      :: item
      INTEGER(KIND=int_8)                                   :: hash, idx

      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_${keytype}$_${valuetype}$_haskey: dictionary is not initialized.")

      res = .FALSE.
      IF (dict%size == 0) RETURN

      hash = hash_${keytype}$ (key)
      idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1

      item => dict%buckets(idx)%p
      do while (ASSOCIATED(item))
         IF (item%hash == hash) THEN
            IF (@{isequal(item%key, key)}@) THEN
               res = .TRUE.
               return
            END IF
         END IF
         item => item%next
      end do

   END FUNCTION dict_${keytype}$_${valuetype}$_haskey

   FUNCTION dict_${keytype}$_${valuetype}$_items(dict) RESULT(items)
      !! Returns a pointer to an array of all key/value-items stored in the dictionary.
      !! Caution: The caller is responsible for deallocating targeted array after usage.
      TYPE(dict_${keytype}$_${valuetype}$_type), intent(inout)  :: dict
      TYPE(dict_${keytype}$_${valuetype}$_item_type), dimension(:), POINTER :: items

      TYPE(private_item_type_${keytype}$_${valuetype}$), POINTER  :: item
      INTEGER :: i, j

      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_${keytype}$_${valuetype}$_items: dictionary is not initialized.")

      allocate (items(dict%size))
      j = 1
      do i = 1, size(dict%buckets)
         item => dict%buckets(i)%p
         do while (ASSOCIATED(item))
            items(j)%key ${key_assign}$item%key
            items(j)%value ${value_assign}$item%value
            j = j + 1
            item => item%next
         end do
      end do

      IF (j /= dict%size + 1) &
         DBCSR_ABORT("dict_${keytype}$_${valuetype}$_items: assertion failed!")
   END FUNCTION dict_${keytype}$_${valuetype}$_items

   SUBROUTINE dict_${keytype}$_${valuetype}$_update(dict, from_dict)
      !! Copies all key/values-items from one dictionary to another.
      !! Afterwards dict will contain all items from the from_dict and
      !! additionally all its previous items, which were not overwritten.
      !! The two dictionaries have to be of the same type.

      TYPE(dict_${keytype}$_${valuetype}$_type), intent(inout)  :: dict
         !! destination of items
      TYPE(dict_${keytype}$_${valuetype}$_type), intent(inout)  :: from_dict
         !! source of items - will not be change
      TYPE(dict_${keytype}$_${valuetype}$_item_type), dimension(:), &
         POINTER :: from_items
      INTEGER :: i

      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_${keytype}$_${valuetype}$_update: dictionary is not initialized.")
      IF (.not. ASSOCIATED(from_dict%buckets)) &
         DBCSR_ABORT("dict_${keytype}$_${valuetype}$_update: from_dict is not initialized.")

      from_items => dict_${keytype}$_${valuetype}$_items(from_dict)
      do i = 1, size(from_items)
         call dict_${keytype}$_${valuetype}$_set(dict, &
                                                 from_items(i)%key, from_items(i)%value)
      end do
      deallocate (from_items)
   END SUBROUTINE dict_${keytype}$_${valuetype}$_update

#:endfor

END MODULE dbcsr_dict
