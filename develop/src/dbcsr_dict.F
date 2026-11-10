# 1 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F" 1
!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

# 1 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.fypp" 1
# 9 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.fypp"

# 81 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.fypp"
# 11 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F" 2
MODULE dbcsr_dict
   !! A dictionary (also known as hashtable or hashmap).
   !! Internally the dictionary uses an array to holds its data.
   !! If this array reaches a load-factor of 75%, a new array with twice the
   !! size will be allocated and the items are then copied over.
   !! This ensures that the dictionary will perform operations in O(1).

   USE dbcsr_kinds, ONLY: int_8, int_4, default_string_length
# 19 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   USE dbcsr_timings_base_type, ONLY: call_stat_type
# 20 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
#include "base/dbcsr_base_uses.f90"
   IMPLICIT NONE

   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_dict'

   PUBLIC :: dict_init, dict_items, dict_haskey, &
             dict_set, dict_get, dict_size, dict_destroy

# 30 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      PUBLIC :: dict_str_i4_type
# 31 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      PUBLIC :: dict_str_i4_item_type
# 32 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 30 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      PUBLIC :: dict_i4tuple_callstat_type
# 31 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      PUBLIC :: dict_i4tuple_callstat_item_type
# 32 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 33 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

# 35 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !this is an internal type
      !Calculating hashes might be expensive, therefore they are stored
      !for use during change_capacity().
      TYPE private_item_type_str_i4
# 39 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         PRIVATE
         CHARACTER(LEN=default_string_length)                            :: key =""
# 41 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(kind=int_4)                          :: value =-1_int_4
# 42 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(KIND=int_8)                         :: hash = -1_int_8
         TYPE(private_item_type_str_i4), POINTER          :: next => Null()
# 44 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      END TYPE private_item_type_str_i4
# 45 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      !this is an internal type
      TYPE private_item_p_type_str_i4
# 48 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         PRIVATE
         TYPE(private_item_type_str_i4), POINTER :: p => Null()
# 50 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      END TYPE private_item_p_type_str_i4
# 51 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      ! this is the public type, which holds a dictionary-instance
      TYPE dict_str_i4_type
# 54 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         PRIVATE
         TYPE(private_item_p_type_str_i4), DIMENSION(:), POINTER      :: buckets => Null()
# 56 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER                                               :: size = -1
      END TYPE dict_str_i4_type
# 58 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      ! this is a public type, its returned by dict_items()
      TYPE dict_str_i4_item_type
# 61 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         CHARACTER(LEN=default_string_length)        :: key =""
# 62 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(kind=int_4)      :: value =-1_int_4
# 63 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      END TYPE dict_str_i4_item_type
# 64 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 35 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !this is an internal type
      !Calculating hashes might be expensive, therefore they are stored
      !for use during change_capacity().
      TYPE private_item_type_i4tuple_callstat
# 39 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         PRIVATE
         INTEGER(kind=int_4), dimension(2)                            :: key =-1_int_4
# 41 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         TYPE(call_stat_type), POINTER                          :: value =>NULL()
# 42 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(KIND=int_8)                         :: hash = -1_int_8
         TYPE(private_item_type_i4tuple_callstat), POINTER          :: next => Null()
# 44 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      END TYPE private_item_type_i4tuple_callstat
# 45 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      !this is an internal type
      TYPE private_item_p_type_i4tuple_callstat
# 48 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         PRIVATE
         TYPE(private_item_type_i4tuple_callstat), POINTER :: p => Null()
# 50 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      END TYPE private_item_p_type_i4tuple_callstat
# 51 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      ! this is the public type, which holds a dictionary-instance
      TYPE dict_i4tuple_callstat_type
# 54 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         PRIVATE
         TYPE(private_item_p_type_i4tuple_callstat), DIMENSION(:), POINTER      :: buckets => Null()
# 56 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER                                               :: size = -1
      END TYPE dict_i4tuple_callstat_type
# 58 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      ! this is a public type, its returned by dict_items()
      TYPE dict_i4tuple_callstat_item_type
# 61 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(kind=int_4), dimension(2)        :: key =-1_int_4
# 62 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         TYPE(call_stat_type), POINTER      :: value =>NULL()
# 63 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      END TYPE dict_i4tuple_callstat_item_type
# 64 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 65 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

   INTERFACE dict_init
# 68 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         MODULE PROCEDURE dict_str_i4_init
# 69 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 68 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         MODULE PROCEDURE dict_i4tuple_callstat_init
# 69 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 70 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   END INTERFACE

   INTERFACE dict_haskey
# 74 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         MODULE PROCEDURE dict_str_i4_haskey
# 75 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 74 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         MODULE PROCEDURE dict_i4tuple_callstat_haskey
# 75 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 76 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   END INTERFACE

   INTERFACE dict_set
# 80 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         MODULE PROCEDURE dict_str_i4_set
# 81 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 80 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         MODULE PROCEDURE dict_i4tuple_callstat_set
# 81 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 82 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   END INTERFACE

   INTERFACE dict_get
# 86 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         MODULE PROCEDURE dict_str_i4_get
# 87 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 86 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         MODULE PROCEDURE dict_i4tuple_callstat_get
# 87 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 88 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   END INTERFACE

   INTERFACE dict_items
# 92 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         MODULE PROCEDURE dict_str_i4_items
# 93 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 92 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         MODULE PROCEDURE dict_i4tuple_callstat_items
# 93 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 94 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   END INTERFACE

   INTERFACE dict_size
# 98 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         MODULE PROCEDURE dict_str_i4_size
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 98 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         MODULE PROCEDURE dict_i4tuple_callstat_size
# 99 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 100 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   END INTERFACE

   INTERFACE dict_destroy
# 104 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         MODULE PROCEDURE dict_str_i4_destroy
# 105 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 104 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         MODULE PROCEDURE dict_i4tuple_callstat_destroy
# 105 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 106 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   END INTERFACE

CONTAINS

# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
FUNCTION hash_str(key) RESULT(hash)
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   !! This is joaat_hash from string_table.F, generates the hash of a given string
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   !! @note
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   !!       http://en.wikipedia.org/wiki/Hash_table
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   !!       http://www.burtleburtle.net/bob/hash/doobs.html
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
    CHARACTER(LEN=*), INTENT(IN)             :: key
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
       !! key a string of any length
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
    INTEGER(KIND=int_8)                      :: hash
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
    INTEGER(KIND=int_8), PARAMETER           :: b32 = 2_int_8**32-1_int_8
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
    INTEGER                                  :: i
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
    hash=0_int_8
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
    DO i=1,LEN(key)
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
       hash=IAND(hash+ICHAR(key(i:i))                ,b32)
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
       hash=IAND(     hash+IAND(ISHFT(hash,10),b32)  ,b32)
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
       hash=IAND(IEOR(hash,IAND(ISHFT(hash,-6),b32)) ,b32)
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
    ENDDO
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
    hash=IAND(     hash+IAND(ISHFT(hash,  3),b32)  ,b32)
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
    hash=IAND(IEOR(hash,IAND(ISHFT(hash,-11),b32)) ,b32)
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
    hash=IAND(     hash+IAND(ISHFT(hash, 15),b32)  ,b32)
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
END FUNCTION hash_str
# 112 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
FUNCTION hash_i4tuple(key) RESULT(hash)
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   INTEGER(kind=int_4), dimension(2), INTENT(IN) :: key
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   INTEGER(kind=int_4)                           :: hash
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   hash = SUM(key)
# 111 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
END FUNCTION hash_i4tuple
# 112 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
# 113 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      SUBROUTINE dict_str_i4_init(dict, initial_capacity)
# 116 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Allocates the internal data-structures of the given dictionary.

         TYPE(dict_str_i4_type), intent(inout)  :: dict
# 119 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER, INTENT(in), OPTIONAL                         :: initial_capacity
         !! The initial size of the internal array (default=11).

         INTEGER :: initial_capacity_
         initial_capacity_ = 11
         IF (PRESENT(initial_capacity)) initial_capacity_ = initial_capacity

         IF (initial_capacity_ < 1) &
            DBCSR_ABORT("dict_str_i4_init: initial_capacity < 1")
# 128 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         IF (ASSOCIATED(dict%buckets)) &
            DBCSR_ABORT("dict_str_i4_init: dictionary is already initialized.")
# 131 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         ALLOCATE (dict%buckets(initial_capacity_))
         dict%size = 0

      END SUBROUTINE dict_str_i4_init
# 136 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      SUBROUTINE dict_str_i4_destroy(dict)
# 138 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Deallocated the internal data-structures if the given dictionary.
      !! Caution: If the stored keys or values are pointers, their targets will
      !! not get deallocated by this routine.

         TYPE(dict_str_i4_type), intent(inout)  :: dict
# 143 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         TYPE(private_item_type_str_i4), POINTER  :: item, prev_item
# 144 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER :: i

         IF (.not. ASSOCIATED(dict%buckets)) &
            DBCSR_ABORT("dict_str_i4_destroy: dictionary is not initialized.")
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

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

      END SUBROUTINE dict_str_i4_destroy
# 162 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      SUBROUTINE dict_str_i4_set(dict, key, value)
# 164 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Stores, and possibly overwrites, a given value under a given key.
         TYPE(dict_str_i4_type), intent(inout)  :: dict
# 166 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         CHARACTER(LEN=default_string_length), intent(in)                          :: key
# 167 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(kind=int_4), intent(in)                          :: value
# 168 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(KIND=int_8)                                   :: hash
         IF (.not. ASSOCIATED(dict%buckets)) &
            DBCSR_ABORT("dict_str_i4_set: dictionary is not initialized.")
# 171 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         hash = hash_str (key)
# 173 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         call set_hashed_str_i4 (dict, key, value, hash)
# 174 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      END SUBROUTINE dict_str_i4_set
# 175 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      RECURSIVE SUBROUTINE set_hashed_str_i4 (dict, key, value, hash)
# 177 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Common code used internally by dict_set() and change_capacity().
         TYPE(dict_str_i4_type), intent(inout)  :: dict
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         CHARACTER(LEN=default_string_length), intent(in)                          :: key
# 180 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(kind=int_4), intent(in)                          :: value
# 181 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(KIND=int_8), intent(in)                       :: hash
         TYPE(private_item_type_str_i4), POINTER  :: item, new_item
# 183 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(KIND=int_8)                                   :: idx

         idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1

         ! if already in dict just update its value
         item => dict%buckets(idx)%p
         do while (ASSOCIATED(item))
            IF (item%hash == hash) THEN
               IF (item%key == key) THEN
# 192 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
                  item%value =value
# 193 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
                  return
               END IF
            END IF
            item => item%next
         end do

         ! check load-factor
         IF (4*dict%size > 3*size(dict%buckets)) THEN ! load-factor > 75%
            call change_capacity_str_i4 (dict, 2*size(dict%buckets)) !double capacity
# 202 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
            idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1
         END IF

         ! create a new item
         allocate (new_item)
         new_item%hash = hash
         new_item%key =key
# 209 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         new_item%value =value
# 210 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         new_item%next => dict%buckets(idx)%p
         dict%buckets(idx)%p => new_item
         dict%size = dict%size + 1

      END SUBROUTINE set_hashed_str_i4
# 215 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      RECURSIVE SUBROUTINE change_capacity_str_i4 (dict, new_capacity)
# 217 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Internal routine for changing the dictionary's capacity.
         TYPE(dict_str_i4_type), intent(inout)  :: dict
# 219 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER, intent(in) :: new_capacity
         INTEGER :: i, old_size, new_cap
         TYPE(private_item_type_str_i4), POINTER  :: item, prev_item
# 222 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         TYPE(private_item_p_type_str_i4), DIMENSION(:), POINTER  :: old_buckets
# 223 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         new_cap = new_capacity
         ! pre checks
         IF (new_cap > HUGE(i)) THEN
            IF (size(dict%buckets) == HUGE(i)) RETURN ! reached maximum - stay there.
            new_cap = HUGE(i) ! grow as far as possible
         END IF
         IF (new_cap < 1) &
            DBCSR_ABORT("dict_str_i4_change_capacity: new_capacity < 1.")
# 231 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         IF (4*dict%size > 3*new_cap) &
            DBCSR_ABORT("dict_str_i4_change_capacity: new_capacity too small.")
# 233 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         old_size = dict%size
         old_buckets => dict%buckets
         ALLOCATE (dict%buckets(new_capacity))
         dict%size = 0
         do i = 1, size(old_buckets)
            item => old_buckets(i)%p
            do while (ASSOCIATED(item))
               call set_hashed_str_i4 (dict, item%key, item%value, item%hash)
# 242 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
               prev_item => item
               item => item%next
               deallocate (prev_item)
            end do
         end do

         deallocate (old_buckets)

         IF (old_size /= dict%size) &
            DBCSR_ABORT("dict_str_i4_change_capacity: assertion failed")
# 252 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      END SUBROUTINE change_capacity_str_i4
# 253 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      FUNCTION dict_str_i4_get(dict, key, default_value) RESULT(value)
# 255 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Gets a value for a given key from the dictionary.
      !! If the key is not found the default_value will be returned.
      !! If the key is not found and default_value was not provided the program stops.
         TYPE(dict_str_i4_type), intent(inout)  :: dict
# 259 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         CHARACTER(LEN=default_string_length)                                        :: key
# 260 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(kind=int_4), intent(in), optional                :: default_value
# 261 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(kind=int_4)                                      :: value
# 262 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         TYPE(private_item_type_str_i4), POINTER                      :: item
# 263 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(KIND=int_8)                                   :: hash, idx

# 266 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
            value = 0
# 270 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         IF (.not. ASSOCIATED(dict%buckets)) &
            DBCSR_ABORT("dict_str_i4_get: dictionary is not initialized.")
# 273 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         hash = hash_str (key)
# 275 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1

         item => dict%buckets(idx)%p
         do while (ASSOCIATED(item))
            IF (item%hash == hash) THEN
               IF (item%key == key) THEN
# 281 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
                  value =item%value
# 282 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
                  return
               END IF
            END IF
            item => item%next
         end do

         IF (PRESENT(default_value)) THEN
            value =default_value
# 290 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
            return
         END IF

         DBCSR_ABORT("dict_str_i4_get: Key not found in dictionary.")
# 294 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      END FUNCTION dict_str_i4_get
# 295 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      FUNCTION dict_str_i4_size(dict) RESULT(size)
# 297 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Returns the number of key/value-items currently stored in the dictionary.
         TYPE(dict_str_i4_type), intent(inout)  :: dict
# 299 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER :: size

         IF (.not. ASSOCIATED(dict%buckets)) &
            DBCSR_ABORT("dict_str_i4_size: dictionary is not initialized.")
# 303 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         size = dict%size
      END FUNCTION dict_str_i4_size
# 306 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      FUNCTION dict_str_i4_haskey(dict, key) RESULT(res)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Checks whether a given key is currently stored in the dictionary.
         TYPE(dict_str_i4_type), intent(inout)  :: dict
# 310 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         CHARACTER(LEN=default_string_length)                                        :: key
# 311 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         LOGICAL                                               :: res
         TYPE(private_item_type_str_i4), POINTER                      :: item
# 313 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(KIND=int_8)                                   :: hash, idx

         IF (.not. ASSOCIATED(dict%buckets)) &
            DBCSR_ABORT("dict_str_i4_haskey: dictionary is not initialized.")
# 317 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         res = .FALSE.
         IF (dict%size == 0) RETURN

         hash = hash_str (key)
# 322 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1

         item => dict%buckets(idx)%p
         do while (ASSOCIATED(item))
            IF (item%hash == hash) THEN
               IF (item%key == key) THEN
# 328 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
                  res = .TRUE.
                  return
               END IF
            END IF
            item => item%next
         end do

      END FUNCTION dict_str_i4_haskey
# 336 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      FUNCTION dict_str_i4_items(dict) RESULT(items)
# 338 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Returns a pointer to an array of all key/value-items stored in the dictionary.
      !! Caution: The caller is responsible for deallocating targeted array after usage.
         TYPE(dict_str_i4_type), intent(inout)  :: dict
# 341 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         TYPE(dict_str_i4_item_type), dimension(:), POINTER :: items
# 342 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         TYPE(private_item_type_str_i4), POINTER  :: item
# 344 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER :: i, j

         IF (.not. ASSOCIATED(dict%buckets)) &
            DBCSR_ABORT("dict_str_i4_items: dictionary is not initialized.")
# 348 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         allocate (items(dict%size))
         j = 1
         do i = 1, size(dict%buckets)
            item => dict%buckets(i)%p
            do while (ASSOCIATED(item))
               items(j)%key =item%key
# 355 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
               items(j)%value =item%value
# 356 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
               j = j + 1
               item => item%next
            end do
         end do

         IF (j /= dict%size + 1) &
            DBCSR_ABORT("dict_str_i4_items: assertion failed!")
# 363 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      END FUNCTION dict_str_i4_items
# 364 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      SUBROUTINE dict_i4tuple_callstat_init(dict, initial_capacity)
# 116 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Allocates the internal data-structures of the given dictionary.

         TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
# 119 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER, INTENT(in), OPTIONAL                         :: initial_capacity
         !! The initial size of the internal array (default=11).

         INTEGER :: initial_capacity_
         initial_capacity_ = 11
         IF (PRESENT(initial_capacity)) initial_capacity_ = initial_capacity

         IF (initial_capacity_ < 1) &
            DBCSR_ABORT("dict_i4tuple_callstat_init: initial_capacity < 1")
# 128 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         IF (ASSOCIATED(dict%buckets)) &
            DBCSR_ABORT("dict_i4tuple_callstat_init: dictionary is already initialized.")
# 131 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         ALLOCATE (dict%buckets(initial_capacity_))
         dict%size = 0

      END SUBROUTINE dict_i4tuple_callstat_init
# 136 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      SUBROUTINE dict_i4tuple_callstat_destroy(dict)
# 138 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Deallocated the internal data-structures if the given dictionary.
      !! Caution: If the stored keys or values are pointers, their targets will
      !! not get deallocated by this routine.

         TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
# 143 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         TYPE(private_item_type_i4tuple_callstat), POINTER  :: item, prev_item
# 144 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER :: i

         IF (.not. ASSOCIATED(dict%buckets)) &
            DBCSR_ABORT("dict_i4tuple_callstat_destroy: dictionary is not initialized.")
# 148 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

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

      END SUBROUTINE dict_i4tuple_callstat_destroy
# 162 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      SUBROUTINE dict_i4tuple_callstat_set(dict, key, value)
# 164 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Stores, and possibly overwrites, a given value under a given key.
         TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
# 166 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(kind=int_4), dimension(2), intent(in)                          :: key
# 167 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         TYPE(call_stat_type), POINTER, intent(in)                          :: value
# 168 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(KIND=int_8)                                   :: hash
         IF (.not. ASSOCIATED(dict%buckets)) &
            DBCSR_ABORT("dict_i4tuple_callstat_set: dictionary is not initialized.")
# 171 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         hash = hash_i4tuple (key)
# 173 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         call set_hashed_i4tuple_callstat (dict, key, value, hash)
# 174 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      END SUBROUTINE dict_i4tuple_callstat_set
# 175 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      RECURSIVE SUBROUTINE set_hashed_i4tuple_callstat (dict, key, value, hash)
# 177 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Common code used internally by dict_set() and change_capacity().
         TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
# 179 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(kind=int_4), dimension(2), intent(in)                          :: key
# 180 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         TYPE(call_stat_type), POINTER, intent(in)                          :: value
# 181 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(KIND=int_8), intent(in)                       :: hash
         TYPE(private_item_type_i4tuple_callstat), POINTER  :: item, new_item
# 183 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(KIND=int_8)                                   :: idx

         idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1

         ! if already in dict just update its value
         item => dict%buckets(idx)%p
         do while (ASSOCIATED(item))
            IF (item%hash == hash) THEN
               IF (ALL(item%key==key)) THEN
# 192 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
                  item%value =>value
# 193 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
                  return
               END IF
            END IF
            item => item%next
         end do

         ! check load-factor
         IF (4*dict%size > 3*size(dict%buckets)) THEN ! load-factor > 75%
            call change_capacity_i4tuple_callstat (dict, 2*size(dict%buckets)) !double capacity
# 202 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
            idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1
         END IF

         ! create a new item
         allocate (new_item)
         new_item%hash = hash
         new_item%key =key
# 209 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         new_item%value =>value
# 210 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         new_item%next => dict%buckets(idx)%p
         dict%buckets(idx)%p => new_item
         dict%size = dict%size + 1

      END SUBROUTINE set_hashed_i4tuple_callstat
# 215 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      RECURSIVE SUBROUTINE change_capacity_i4tuple_callstat (dict, new_capacity)
# 217 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Internal routine for changing the dictionary's capacity.
         TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
# 219 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER, intent(in) :: new_capacity
         INTEGER :: i, old_size, new_cap
         TYPE(private_item_type_i4tuple_callstat), POINTER  :: item, prev_item
# 222 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         TYPE(private_item_p_type_i4tuple_callstat), DIMENSION(:), POINTER  :: old_buckets
# 223 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         new_cap = new_capacity
         ! pre checks
         IF (new_cap > HUGE(i)) THEN
            IF (size(dict%buckets) == HUGE(i)) RETURN ! reached maximum - stay there.
            new_cap = HUGE(i) ! grow as far as possible
         END IF
         IF (new_cap < 1) &
            DBCSR_ABORT("dict_i4tuple_callstat_change_capacity: new_capacity < 1.")
# 231 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         IF (4*dict%size > 3*new_cap) &
            DBCSR_ABORT("dict_i4tuple_callstat_change_capacity: new_capacity too small.")
# 233 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         old_size = dict%size
         old_buckets => dict%buckets
         ALLOCATE (dict%buckets(new_capacity))
         dict%size = 0
         do i = 1, size(old_buckets)
            item => old_buckets(i)%p
            do while (ASSOCIATED(item))
               call set_hashed_i4tuple_callstat (dict, item%key, item%value, item%hash)
# 242 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
               prev_item => item
               item => item%next
               deallocate (prev_item)
            end do
         end do

         deallocate (old_buckets)

         IF (old_size /= dict%size) &
            DBCSR_ABORT("dict_i4tuple_callstat_change_capacity: assertion failed")
# 252 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      END SUBROUTINE change_capacity_i4tuple_callstat
# 253 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      FUNCTION dict_i4tuple_callstat_get(dict, key, default_value) RESULT(value)
# 255 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Gets a value for a given key from the dictionary.
      !! If the key is not found the default_value will be returned.
      !! If the key is not found and default_value was not provided the program stops.
         TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
# 259 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(kind=int_4), dimension(2)                                        :: key
# 260 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         TYPE(call_stat_type), POINTER, intent(in), optional                :: default_value
# 261 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         TYPE(call_stat_type), POINTER                                      :: value
# 262 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         TYPE(private_item_type_i4tuple_callstat), POINTER                      :: item
# 263 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(KIND=int_8)                                   :: hash, idx

# 268 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
            NULLIFY (value)
# 270 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         IF (.not. ASSOCIATED(dict%buckets)) &
            DBCSR_ABORT("dict_i4tuple_callstat_get: dictionary is not initialized.")
# 273 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         hash = hash_i4tuple (key)
# 275 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1

         item => dict%buckets(idx)%p
         do while (ASSOCIATED(item))
            IF (item%hash == hash) THEN
               IF (ALL(item%key==key)) THEN
# 281 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
                  value =>item%value
# 282 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
                  return
               END IF
            END IF
            item => item%next
         end do

         IF (PRESENT(default_value)) THEN
            value =>default_value
# 290 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
            return
         END IF

         DBCSR_ABORT("dict_i4tuple_callstat_get: Key not found in dictionary.")
# 294 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      END FUNCTION dict_i4tuple_callstat_get
# 295 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      FUNCTION dict_i4tuple_callstat_size(dict) RESULT(size)
# 297 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Returns the number of key/value-items currently stored in the dictionary.
         TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
# 299 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER :: size

         IF (.not. ASSOCIATED(dict%buckets)) &
            DBCSR_ABORT("dict_i4tuple_callstat_size: dictionary is not initialized.")
# 303 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         size = dict%size
      END FUNCTION dict_i4tuple_callstat_size
# 306 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      FUNCTION dict_i4tuple_callstat_haskey(dict, key) RESULT(res)
# 308 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Checks whether a given key is currently stored in the dictionary.
         TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
# 310 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(kind=int_4), dimension(2)                                        :: key
# 311 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         LOGICAL                                               :: res
         TYPE(private_item_type_i4tuple_callstat), POINTER                      :: item
# 313 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER(KIND=int_8)                                   :: hash, idx

         IF (.not. ASSOCIATED(dict%buckets)) &
            DBCSR_ABORT("dict_i4tuple_callstat_haskey: dictionary is not initialized.")
# 317 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         res = .FALSE.
         IF (dict%size == 0) RETURN

         hash = hash_i4tuple (key)
# 322 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1

         item => dict%buckets(idx)%p
         do while (ASSOCIATED(item))
            IF (item%hash == hash) THEN
               IF (ALL(item%key==key)) THEN
# 328 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
                  res = .TRUE.
                  return
               END IF
            END IF
            item => item%next
         end do

      END FUNCTION dict_i4tuple_callstat_haskey
# 336 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      FUNCTION dict_i4tuple_callstat_items(dict) RESULT(items)
# 338 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      !! Returns a pointer to an array of all key/value-items stored in the dictionary.
      !! Caution: The caller is responsible for deallocating targeted array after usage.
         TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
# 341 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         TYPE(dict_i4tuple_callstat_item_type), dimension(:), POINTER :: items
# 342 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         TYPE(private_item_type_i4tuple_callstat), POINTER  :: item
# 344 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
         INTEGER :: i, j

         IF (.not. ASSOCIATED(dict%buckets)) &
            DBCSR_ABORT("dict_i4tuple_callstat_items: dictionary is not initialized.")
# 348 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

         allocate (items(dict%size))
         j = 1
         do i = 1, size(dict%buckets)
            item => dict%buckets(i)%p
            do while (ASSOCIATED(item))
               items(j)%key =item%key
# 355 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
               items(j)%value =>item%value
# 356 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
               j = j + 1
               item => item%next
            end do
         end do

         IF (j /= dict%size + 1) &
            DBCSR_ABORT("dict_i4tuple_callstat_items: assertion failed!")
# 363 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      END FUNCTION dict_i4tuple_callstat_items
# 364 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

# 366 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

END MODULE dbcsr_dict
