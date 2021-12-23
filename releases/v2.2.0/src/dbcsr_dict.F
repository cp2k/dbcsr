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

# 79 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.fypp"
# 11 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F" 2
MODULE dbcsr_dict
   !! A dictionary (also known as hashtable or hashmap).
   !! Internally the dictionary uses an array to holds its data.
   !! If this array reaches a load-factor of 75%, a new array with twice the
   !! size will be allocated and the items are then copied over.
   !! This ensures that the dictionary will perform operations in O(1).

   USE dbcsr_kinds, ONLY: int_8, int_4, default_string_length
   USE dbcsr_timings_base_type, ONLY: call_stat_type
#include "base/dbcsr_base_uses.f90"
   IMPLICIT NONE

   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_dict'

   PUBLIC :: dict_init, dict_items, dict_haskey, &
             dict_set, dict_get, dict_size, dict_destroy

# 30 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   PUBLIC :: dict_str_i4_type
   PUBLIC :: dict_str_i4_item_type
# 30 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   PUBLIC :: dict_i4tuple_callstat_type
   PUBLIC :: dict_i4tuple_callstat_item_type
# 33 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

# 35 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   !this is an internal type
   !Calculating hashes might be expensive, therefore they are stored
   !for use during change_capacity().
   TYPE private_item_type_str_i4
      PRIVATE
      CHARACTER(LEN=default_string_length)                            :: key
      INTEGER(kind=int_4)                          :: value
      INTEGER(KIND=int_8)                         :: hash
      TYPE(private_item_type_str_i4), POINTER          :: next => Null()
   END TYPE private_item_type_str_i4

   !this is an internal type
   TYPE private_item_p_type_str_i4
      PRIVATE
      TYPE(private_item_type_str_i4), POINTER :: p => Null()
   END TYPE private_item_p_type_str_i4

   ! this is the public type, which holds a dictionary-instance
   TYPE dict_str_i4_type
      PRIVATE
      TYPE(private_item_p_type_str_i4), DIMENSION(:), POINTER      :: buckets => Null()
      INTEGER                                               :: size = -1
   END TYPE dict_str_i4_type

   ! this is a public type, its returned by dict_items()
   TYPE dict_str_i4_item_type
      CHARACTER(LEN=default_string_length)        :: key
      INTEGER(kind=int_4)      :: value
   END TYPE dict_str_i4_item_type
# 35 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   !this is an internal type
   !Calculating hashes might be expensive, therefore they are stored
   !for use during change_capacity().
   TYPE private_item_type_i4tuple_callstat
      PRIVATE
      INTEGER(kind=int_4), dimension(2)                            :: key
      TYPE(call_stat_type), POINTER                          :: value
      INTEGER(KIND=int_8)                         :: hash
      TYPE(private_item_type_i4tuple_callstat), POINTER          :: next => Null()
   END TYPE private_item_type_i4tuple_callstat

   !this is an internal type
   TYPE private_item_p_type_i4tuple_callstat
      PRIVATE
      TYPE(private_item_type_i4tuple_callstat), POINTER :: p => Null()
   END TYPE private_item_p_type_i4tuple_callstat

   ! this is the public type, which holds a dictionary-instance
   TYPE dict_i4tuple_callstat_type
      PRIVATE
      TYPE(private_item_p_type_i4tuple_callstat), DIMENSION(:), POINTER      :: buckets => Null()
      INTEGER                                               :: size = -1
   END TYPE dict_i4tuple_callstat_type

   ! this is a public type, its returned by dict_items()
   TYPE dict_i4tuple_callstat_item_type
      INTEGER(kind=int_4), dimension(2)        :: key
      TYPE(call_stat_type), POINTER      :: value
   END TYPE dict_i4tuple_callstat_item_type
# 65 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

   INTERFACE dict_init
# 68 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      MODULE PROCEDURE dict_str_i4_init
# 68 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      MODULE PROCEDURE dict_i4tuple_callstat_init
# 70 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   END INTERFACE

   INTERFACE dict_haskey
# 74 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      MODULE PROCEDURE dict_str_i4_haskey
# 74 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      MODULE PROCEDURE dict_i4tuple_callstat_haskey
# 76 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   END INTERFACE

   INTERFACE dict_set
# 80 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      MODULE PROCEDURE dict_str_i4_set
# 80 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      MODULE PROCEDURE dict_i4tuple_callstat_set
# 82 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   END INTERFACE

   INTERFACE dict_get
# 86 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      MODULE PROCEDURE dict_str_i4_get
# 86 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      MODULE PROCEDURE dict_i4tuple_callstat_get
# 88 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   END INTERFACE

   INTERFACE dict_items
# 92 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      MODULE PROCEDURE dict_str_i4_items
# 92 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      MODULE PROCEDURE dict_i4tuple_callstat_items
# 94 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   END INTERFACE

   INTERFACE dict_size
# 98 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      MODULE PROCEDURE dict_str_i4_size
# 98 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      MODULE PROCEDURE dict_i4tuple_callstat_size
# 100 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   END INTERFACE

   INTERFACE dict_destroy
# 104 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      MODULE PROCEDURE dict_str_i4_destroy
# 104 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      MODULE PROCEDURE dict_i4tuple_callstat_destroy
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
# 113 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   SUBROUTINE dict_str_i4_init(dict, initial_capacity)
      !! Allocates the internal data-structures of the given dictionary.

      TYPE(dict_str_i4_type), intent(inout)  :: dict
      INTEGER, INTENT(in), OPTIONAL                         :: initial_capacity
         !! The initial size of the internal array (default=11).

      INTEGER :: initial_capacity_
      initial_capacity_ = 11
      IF (PRESENT(initial_capacity)) initial_capacity_ = initial_capacity

      IF (initial_capacity_ < 1) &
         DBCSR_ABORT("dict_str_i4_init: initial_capacity < 1")

      IF (ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_str_i4_init: dictionary is already initialized.")

      ALLOCATE (dict%buckets(initial_capacity_))
      dict%size = 0

   END SUBROUTINE dict_str_i4_init

   SUBROUTINE dict_str_i4_destroy(dict)
      !! Deallocated the internal data-structures if the given dictionary.
      !! Caution: If the stored keys or values are pointers, their targets will
      !! not get deallocated by this routine.

      TYPE(dict_str_i4_type), intent(inout)  :: dict
      TYPE(private_item_type_str_i4), POINTER  :: item, prev_item
      INTEGER :: i

      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_str_i4_destroy: dictionary is not initialized.")

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

   SUBROUTINE dict_str_i4_set(dict, key, value)
      !! Stores, and possibly overwrites, a given value under a given key.
      TYPE(dict_str_i4_type), intent(inout)  :: dict
      CHARACTER(LEN=default_string_length), intent(in)                          :: key
      INTEGER(kind=int_4), intent(in)                          :: value
      INTEGER(KIND=int_8)                                   :: hash
      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_str_i4_set: dictionary is not initialized.")

      hash = hash_str (key)
      call set_hashed_str_i4 (dict, key, value, hash)
   END SUBROUTINE dict_str_i4_set

   RECURSIVE SUBROUTINE set_hashed_str_i4 (dict, key, value, hash)
      !! Common code used internally by dict_set() and change_capacity().
      TYPE(dict_str_i4_type), intent(inout)  :: dict
      CHARACTER(LEN=default_string_length), intent(in)                          :: key
      INTEGER(kind=int_4), intent(in)                          :: value
      INTEGER(KIND=int_8), intent(in)                       :: hash
      TYPE(private_item_type_str_i4), POINTER  :: item, new_item
      INTEGER(KIND=int_8)                                   :: idx

      idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1

      ! if already in dict just update its value
      item => dict%buckets(idx)%p
      do while (ASSOCIATED(item))
         IF (item%hash == hash) THEN
            IF (item%key == key) THEN
               item%value =value
               return
            END IF
         END IF
         item => item%next
      end do

      ! check load-factor
      IF (4*dict%size > 3*size(dict%buckets)) THEN ! load-factor > 75%
         call change_capacity_str_i4 (dict, 2*size(dict%buckets)) !double capacity
         idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1
      END IF

      ! create a new item
      allocate (new_item)
      new_item%hash = hash
      new_item%key =key
      new_item%value =value
      new_item%next => dict%buckets(idx)%p
      dict%buckets(idx)%p => new_item
      dict%size = dict%size + 1

   END SUBROUTINE set_hashed_str_i4

   RECURSIVE SUBROUTINE change_capacity_str_i4 (dict, new_capacity)
      !! Internal routine for changing the dictionary's capacity.
      TYPE(dict_str_i4_type), intent(inout)  :: dict
      INTEGER, intent(in) :: new_capacity
      INTEGER :: i, old_size, new_cap
      TYPE(private_item_type_str_i4), POINTER  :: item, prev_item
      TYPE(private_item_p_type_str_i4), DIMENSION(:), POINTER  :: old_buckets
      new_cap = new_capacity
      ! pre checks
      IF (new_cap > HUGE(i)) THEN
         IF (size(dict%buckets) == HUGE(i)) RETURN ! reached maximum - stay there.
         new_cap = HUGE(i) ! grow as far as possible
      END IF
      IF (new_cap < 1) &
         DBCSR_ABORT("dict_str_i4_change_capacity: new_capacity < 1.")
      IF (4*dict%size > 3*new_cap) &
         DBCSR_ABORT("dict_str_i4_change_capacity: new_capacity too small.")

      old_size = dict%size
      old_buckets => dict%buckets
      ALLOCATE (dict%buckets(new_capacity))
      dict%size = 0
      do i = 1, size(old_buckets)
         item => old_buckets(i)%p
         do while (ASSOCIATED(item))
            call set_hashed_str_i4 (dict, item%key, item%value, item%hash)
            prev_item => item
            item => item%next
            deallocate (prev_item)
         end do
      end do

      deallocate (old_buckets)

      IF (old_size /= dict%size) &
         DBCSR_ABORT("dict_str_i4_change_capacity: assertion failed")
   END SUBROUTINE change_capacity_str_i4

   FUNCTION dict_str_i4_get(dict, key, default_value) RESULT(value)
      !! Gets a value for a given key from the dictionary.
      !! If the key is not found the default_value will be returned.
      !! If the key is not found and default_value was not provided the program stops.
      TYPE(dict_str_i4_type), intent(inout)  :: dict
      CHARACTER(LEN=default_string_length)                                        :: key
      INTEGER(kind=int_4), intent(in), optional                :: default_value
      INTEGER(kind=int_4)                                      :: value
      TYPE(private_item_type_str_i4), POINTER                      :: item
      INTEGER(KIND=int_8)                                   :: hash, idx

# 266 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      value = 0
# 270 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_str_i4_get: dictionary is not initialized.")

      hash = hash_str (key)
      idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1

      item => dict%buckets(idx)%p
      do while (ASSOCIATED(item))
         IF (item%hash == hash) THEN
            IF (item%key == key) THEN
               value =item%value
               return
            END IF
         END IF
         item => item%next
      end do

      IF (PRESENT(default_value)) THEN
         value =default_value
         return
      END IF

      DBCSR_ABORT("dict_str_i4_get: Key not found in dictionary.")
   END FUNCTION dict_str_i4_get

   FUNCTION dict_str_i4_size(dict) RESULT(size)
      !! Returns the number of key/value-items currently stored in the dictionary.
      TYPE(dict_str_i4_type), intent(inout)  :: dict
      INTEGER :: size

      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_str_i4_size: dictionary is not initialized.")

      size = dict%size
   END FUNCTION dict_str_i4_size

   FUNCTION dict_str_i4_haskey(dict, key) RESULT(res)
      !! Checks whether a given key is currently stored in the dictionary.
      TYPE(dict_str_i4_type), intent(inout)  :: dict
      CHARACTER(LEN=default_string_length)                                        :: key
      LOGICAL                                               :: res
      TYPE(private_item_type_str_i4), POINTER                      :: item
      INTEGER(KIND=int_8)                                   :: hash, idx

      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_str_i4_haskey: dictionary is not initialized.")

      res = .FALSE.
      IF (dict%size == 0) RETURN

      hash = hash_str (key)
      idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1

      item => dict%buckets(idx)%p
      do while (ASSOCIATED(item))
         IF (item%hash == hash) THEN
            IF (item%key == key) THEN
               res = .TRUE.
               return
            END IF
         END IF
         item => item%next
      end do

   END FUNCTION dict_str_i4_haskey

   FUNCTION dict_str_i4_items(dict) RESULT(items)
      !! Returns a pointer to an array of all key/value-items stored in the dictionary.
      !! Caution: The caller is responsible for deallocating targeted array after usage.
      TYPE(dict_str_i4_type), intent(inout)  :: dict
      TYPE(dict_str_i4_item_type), dimension(:), POINTER :: items

      TYPE(private_item_type_str_i4), POINTER  :: item
      INTEGER :: i, j

      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_str_i4_items: dictionary is not initialized.")

      allocate (items(dict%size))
      j = 1
      do i = 1, size(dict%buckets)
         item => dict%buckets(i)%p
         do while (ASSOCIATED(item))
            items(j)%key =item%key
            items(j)%value =item%value
            j = j + 1
            item => item%next
         end do
      end do

      IF (j /= dict%size + 1) &
         DBCSR_ABORT("dict_str_i4_items: assertion failed!")
   END FUNCTION dict_str_i4_items

# 115 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
   SUBROUTINE dict_i4tuple_callstat_init(dict, initial_capacity)
      !! Allocates the internal data-structures of the given dictionary.

      TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
      INTEGER, INTENT(in), OPTIONAL                         :: initial_capacity
         !! The initial size of the internal array (default=11).

      INTEGER :: initial_capacity_
      initial_capacity_ = 11
      IF (PRESENT(initial_capacity)) initial_capacity_ = initial_capacity

      IF (initial_capacity_ < 1) &
         DBCSR_ABORT("dict_i4tuple_callstat_init: initial_capacity < 1")

      IF (ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_i4tuple_callstat_init: dictionary is already initialized.")

      ALLOCATE (dict%buckets(initial_capacity_))
      dict%size = 0

   END SUBROUTINE dict_i4tuple_callstat_init

   SUBROUTINE dict_i4tuple_callstat_destroy(dict)
      !! Deallocated the internal data-structures if the given dictionary.
      !! Caution: If the stored keys or values are pointers, their targets will
      !! not get deallocated by this routine.

      TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
      TYPE(private_item_type_i4tuple_callstat), POINTER  :: item, prev_item
      INTEGER :: i

      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_i4tuple_callstat_destroy: dictionary is not initialized.")

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

   SUBROUTINE dict_i4tuple_callstat_set(dict, key, value)
      !! Stores, and possibly overwrites, a given value under a given key.
      TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
      INTEGER(kind=int_4), dimension(2), intent(in)                          :: key
      TYPE(call_stat_type), POINTER, intent(in)                          :: value
      INTEGER(KIND=int_8)                                   :: hash
      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_i4tuple_callstat_set: dictionary is not initialized.")

      hash = hash_i4tuple (key)
      call set_hashed_i4tuple_callstat (dict, key, value, hash)
   END SUBROUTINE dict_i4tuple_callstat_set

   RECURSIVE SUBROUTINE set_hashed_i4tuple_callstat (dict, key, value, hash)
      !! Common code used internally by dict_set() and change_capacity().
      TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
      INTEGER(kind=int_4), dimension(2), intent(in)                          :: key
      TYPE(call_stat_type), POINTER, intent(in)                          :: value
      INTEGER(KIND=int_8), intent(in)                       :: hash
      TYPE(private_item_type_i4tuple_callstat), POINTER  :: item, new_item
      INTEGER(KIND=int_8)                                   :: idx

      idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1

      ! if already in dict just update its value
      item => dict%buckets(idx)%p
      do while (ASSOCIATED(item))
         IF (item%hash == hash) THEN
            IF (ALL(item%key==key)) THEN
               item%value =>value
               return
            END IF
         END IF
         item => item%next
      end do

      ! check load-factor
      IF (4*dict%size > 3*size(dict%buckets)) THEN ! load-factor > 75%
         call change_capacity_i4tuple_callstat (dict, 2*size(dict%buckets)) !double capacity
         idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1
      END IF

      ! create a new item
      allocate (new_item)
      new_item%hash = hash
      new_item%key =key
      new_item%value =>value
      new_item%next => dict%buckets(idx)%p
      dict%buckets(idx)%p => new_item
      dict%size = dict%size + 1

   END SUBROUTINE set_hashed_i4tuple_callstat

   RECURSIVE SUBROUTINE change_capacity_i4tuple_callstat (dict, new_capacity)
      !! Internal routine for changing the dictionary's capacity.
      TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
      INTEGER, intent(in) :: new_capacity
      INTEGER :: i, old_size, new_cap
      TYPE(private_item_type_i4tuple_callstat), POINTER  :: item, prev_item
      TYPE(private_item_p_type_i4tuple_callstat), DIMENSION(:), POINTER  :: old_buckets
      new_cap = new_capacity
      ! pre checks
      IF (new_cap > HUGE(i)) THEN
         IF (size(dict%buckets) == HUGE(i)) RETURN ! reached maximum - stay there.
         new_cap = HUGE(i) ! grow as far as possible
      END IF
      IF (new_cap < 1) &
         DBCSR_ABORT("dict_i4tuple_callstat_change_capacity: new_capacity < 1.")
      IF (4*dict%size > 3*new_cap) &
         DBCSR_ABORT("dict_i4tuple_callstat_change_capacity: new_capacity too small.")

      old_size = dict%size
      old_buckets => dict%buckets
      ALLOCATE (dict%buckets(new_capacity))
      dict%size = 0
      do i = 1, size(old_buckets)
         item => old_buckets(i)%p
         do while (ASSOCIATED(item))
            call set_hashed_i4tuple_callstat (dict, item%key, item%value, item%hash)
            prev_item => item
            item => item%next
            deallocate (prev_item)
         end do
      end do

      deallocate (old_buckets)

      IF (old_size /= dict%size) &
         DBCSR_ABORT("dict_i4tuple_callstat_change_capacity: assertion failed")
   END SUBROUTINE change_capacity_i4tuple_callstat

   FUNCTION dict_i4tuple_callstat_get(dict, key, default_value) RESULT(value)
      !! Gets a value for a given key from the dictionary.
      !! If the key is not found the default_value will be returned.
      !! If the key is not found and default_value was not provided the program stops.
      TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
      INTEGER(kind=int_4), dimension(2)                                        :: key
      TYPE(call_stat_type), POINTER, intent(in), optional                :: default_value
      TYPE(call_stat_type), POINTER                                      :: value
      TYPE(private_item_type_i4tuple_callstat), POINTER                      :: item
      INTEGER(KIND=int_8)                                   :: hash, idx

# 268 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"
      NULLIFY (value)
# 270 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_i4tuple_callstat_get: dictionary is not initialized.")

      hash = hash_i4tuple (key)
      idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1

      item => dict%buckets(idx)%p
      do while (ASSOCIATED(item))
         IF (item%hash == hash) THEN
            IF (ALL(item%key==key)) THEN
               value =>item%value
               return
            END IF
         END IF
         item => item%next
      end do

      IF (PRESENT(default_value)) THEN
         value =>default_value
         return
      END IF

      DBCSR_ABORT("dict_i4tuple_callstat_get: Key not found in dictionary.")
   END FUNCTION dict_i4tuple_callstat_get

   FUNCTION dict_i4tuple_callstat_size(dict) RESULT(size)
      !! Returns the number of key/value-items currently stored in the dictionary.
      TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
      INTEGER :: size

      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_i4tuple_callstat_size: dictionary is not initialized.")

      size = dict%size
   END FUNCTION dict_i4tuple_callstat_size

   FUNCTION dict_i4tuple_callstat_haskey(dict, key) RESULT(res)
      !! Checks whether a given key is currently stored in the dictionary.
      TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
      INTEGER(kind=int_4), dimension(2)                                        :: key
      LOGICAL                                               :: res
      TYPE(private_item_type_i4tuple_callstat), POINTER                      :: item
      INTEGER(KIND=int_8)                                   :: hash, idx

      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_i4tuple_callstat_haskey: dictionary is not initialized.")

      res = .FALSE.
      IF (dict%size == 0) RETURN

      hash = hash_i4tuple (key)
      idx = MOD(hash, INT(size(dict%buckets), KIND=int_8)) + 1

      item => dict%buckets(idx)%p
      do while (ASSOCIATED(item))
         IF (item%hash == hash) THEN
            IF (ALL(item%key==key)) THEN
               res = .TRUE.
               return
            END IF
         END IF
         item => item%next
      end do

   END FUNCTION dict_i4tuple_callstat_haskey

   FUNCTION dict_i4tuple_callstat_items(dict) RESULT(items)
      !! Returns a pointer to an array of all key/value-items stored in the dictionary.
      !! Caution: The caller is responsible for deallocating targeted array after usage.
      TYPE(dict_i4tuple_callstat_type), intent(inout)  :: dict
      TYPE(dict_i4tuple_callstat_item_type), dimension(:), POINTER :: items

      TYPE(private_item_type_i4tuple_callstat), POINTER  :: item
      INTEGER :: i, j

      IF (.not. ASSOCIATED(dict%buckets)) &
         DBCSR_ABORT("dict_i4tuple_callstat_items: dictionary is not initialized.")

      allocate (items(dict%size))
      j = 1
      do i = 1, size(dict%buckets)
         item => dict%buckets(i)%p
         do while (ASSOCIATED(item))
            items(j)%key =item%key
            items(j)%value =>item%value
            j = j + 1
            item => item%next
         end do
      end do

      IF (j /= dict%size + 1) &
         DBCSR_ABORT("dict_i4tuple_callstat_items: assertion failed!")
   END FUNCTION dict_i4tuple_callstat_items

# 366 "/__w/dbcsr/dbcsr/src/core/dbcsr_dict.F"

END MODULE dbcsr_dict
