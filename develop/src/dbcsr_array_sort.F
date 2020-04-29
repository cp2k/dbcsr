# 1 "/__w/dbcsr/dbcsr/src/utils/dbcsr_array_sort.F" 1
!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

# 1 "/__w/dbcsr/dbcsr/src/utils/dbcsr_array_sort.fypp" 1
# 9 "/__w/dbcsr/dbcsr/src/utils/dbcsr_array_sort.fypp"

# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_array_sort.fypp"
# 11 "/__w/dbcsr/dbcsr/src/utils/dbcsr_array_sort.F" 2

MODULE dbcsr_array_sort
   !! Routine for sorting an array
   !! @note
   !! CP2K:
   !! Please use the interface defined in util.F for calling sort().
   !! DBCSR:
   !! Please use the interface defined in dbcsr_toollib.F for calling sort().

   USE dbcsr_kinds, ONLY: int_4, real_4, real_8, int_8

   IMPLICIT NONE
   PRIVATE

   LOGICAL, PRIVATE, PARAMETER :: debug_this_module = .FALSE.
   CHARACTER(len=*), PRIVATE, PARAMETER :: moduleN = 'dbcsr_array_sort'

# 29 "/__w/dbcsr/dbcsr/src/utils/dbcsr_array_sort.F"
   PUBLIC :: dbcsr_1d_d_sort
# 29 "/__w/dbcsr/dbcsr/src/utils/dbcsr_array_sort.F"
   PUBLIC :: dbcsr_1d_s_sort
# 29 "/__w/dbcsr/dbcsr/src/utils/dbcsr_array_sort.F"
   PUBLIC :: dbcsr_1d_i4_sort
# 29 "/__w/dbcsr/dbcsr/src/utils/dbcsr_array_sort.F"
   PUBLIC :: dbcsr_1d_i8_sort
# 31 "/__w/dbcsr/dbcsr/src/utils/dbcsr_array_sort.F"

CONTAINS

# 35 "/__w/dbcsr/dbcsr/src/utils/dbcsr_array_sort.F"
   subroutine dbcsr_1d_d_sort(arr, n, indices)
      !! Sorts an array inplace using a combination of merge- and bubble-sort.
      !! It also returns the indices, which the elements had before the sort.

      integer, intent(in)                  :: n
         !! length of array
      REAL(kind=real_8), dimension(1:n), intent(inout) :: arr
         !! the array to sort
      integer, dimension(1:n), intent(out)   :: indices
         !! returns elements-indices before the sort

      integer :: i
      REAL(kind=real_8), pointer, CONTIGUOUS     :: tmp_arr(:)
      integer, pointer, CONTIGUOUS       :: tmp_idx(:)

      if (n == 0) return ! for some reason this is a frequent case in cp2k

      ! scratch space used during the merge step
      allocate (tmp_arr((size(arr) + 1)/2), tmp_idx((size(arr) + 1)/2))

      indices = (/(i, i=1, size(arr))/)

      call dbcsr_1d_d_sort_low(arr(1:n), indices, tmp_arr, tmp_idx)

      deallocate (tmp_arr, tmp_idx)

   end subroutine dbcsr_1d_d_sort

   recursive subroutine dbcsr_1d_d_sort_low(arr, indices, tmp_arr, tmp_idx)
      !! The actual sort routine.
      !! Only dbcsr_1d_d_sort and itself should call this.

      REAL(kind=real_8), dimension(:), intent(inout) :: arr
         !! the array to sort
      integer, dimension(size(arr)), intent(inout) :: indices
         !! elements-indices before the sort
      REAL(kind=real_8), dimension((size(arr) + 1)/2), intent(inout) :: tmp_arr
         !! scratch space
      integer, dimension((size(arr) + 1)/2), intent(inout) :: tmp_idx
         !! scratch space
      REAL(kind=real_8) :: a
      integer :: t, m, i, j, k
      LOGICAL :: swapped
      ! a,t:  used during swapping of elements in arr and indices

      swapped = .TRUE.

      ! If only a few elements are left we switch to bubble-sort for efficiency.
      if (size(arr) <= 7) then ! 7 seems to be a good choice for the moment
         DO j = size(arr) - 1, 1, -1
            swapped = .FALSE.
            DO i = 1, j
               IF (arr(i+1) < arr(i)) THEN
                  ! swap arr(i) with arr(i+1)
                  a = arr(i)
                  arr(i) = arr(i + 1)
                  arr(i + 1) = a
                  ! swap indices(i) with indices(i+1)
                  t = indices(i)
                  indices(i) = indices(i + 1)
                  indices(i + 1) = t
                  swapped = .true.
               END IF
            END DO
            IF (.NOT. swapped) EXIT
         END DO
         return
      endif

      ! split list in half and recursively sort both sublists
      m = (size(arr) + 1)/2 ! index where we going to divide the list in two
      call dbcsr_1d_d_sort_low(arr(1:m), indices(1:m), tmp_arr, tmp_idx)
      call dbcsr_1d_d_sort_low(arr(m + 1:), indices(m + 1:), tmp_arr, tmp_idx)

      ! Check for a special case: Can we just concatenate the two sorted sublists?
      ! This leads to O(n) scaling if the input is already sorted.
      if (arr(m+1) < arr(m)) then
         ! ...no - let's merge the two sorted sublists arr(:m) and arr(m+1:)
         ! Merge will be performed directly in arr. Need backup of first sublist.
         tmp_arr(1:m) = arr(1:m)
         tmp_idx(1:m) = indices(1:m)
         i = 1; ! number of elements consumed from 1st sublist
         j = 1; ! number of elements consumed from 2nd sublist
         k = 1; ! number of elements already merged

         do while (i <= m .and. j <= size(arr) - m)
         if (arr(m+j) < tmp_arr(i)) then
            arr(k) = arr(m + j)
            indices(k) = indices(m + j)
            j = j + 1
         else
            arr(k) = tmp_arr(i)
            indices(k) = tmp_idx(i)
            i = i + 1
         endif
         k = k + 1
         enddo

         ! One of the two sublist is now empty.
         ! Copy possibly remaining tail of 1st sublist
         do while (i <= m)
            arr(k) = tmp_arr(i)
            indices(k) = tmp_idx(i)
            i = i + 1
            k = k + 1
         enddo

         ! The possibly remaining tail of 2nd sublist is already at the right spot.

      endif

   end subroutine dbcsr_1d_d_sort_low
# 35 "/__w/dbcsr/dbcsr/src/utils/dbcsr_array_sort.F"
   subroutine dbcsr_1d_s_sort(arr, n, indices)
      !! Sorts an array inplace using a combination of merge- and bubble-sort.
      !! It also returns the indices, which the elements had before the sort.

      integer, intent(in)                  :: n
         !! length of array
      REAL(kind=real_4), dimension(1:n), intent(inout) :: arr
         !! the array to sort
      integer, dimension(1:n), intent(out)   :: indices
         !! returns elements-indices before the sort

      integer :: i
      REAL(kind=real_4), pointer, CONTIGUOUS     :: tmp_arr(:)
      integer, pointer, CONTIGUOUS       :: tmp_idx(:)

      if (n == 0) return ! for some reason this is a frequent case in cp2k

      ! scratch space used during the merge step
      allocate (tmp_arr((size(arr) + 1)/2), tmp_idx((size(arr) + 1)/2))

      indices = (/(i, i=1, size(arr))/)

      call dbcsr_1d_s_sort_low(arr(1:n), indices, tmp_arr, tmp_idx)

      deallocate (tmp_arr, tmp_idx)

   end subroutine dbcsr_1d_s_sort

   recursive subroutine dbcsr_1d_s_sort_low(arr, indices, tmp_arr, tmp_idx)
      !! The actual sort routine.
      !! Only dbcsr_1d_s_sort and itself should call this.

      REAL(kind=real_4), dimension(:), intent(inout) :: arr
         !! the array to sort
      integer, dimension(size(arr)), intent(inout) :: indices
         !! elements-indices before the sort
      REAL(kind=real_4), dimension((size(arr) + 1)/2), intent(inout) :: tmp_arr
         !! scratch space
      integer, dimension((size(arr) + 1)/2), intent(inout) :: tmp_idx
         !! scratch space
      REAL(kind=real_4) :: a
      integer :: t, m, i, j, k
      LOGICAL :: swapped
      ! a,t:  used during swapping of elements in arr and indices

      swapped = .TRUE.

      ! If only a few elements are left we switch to bubble-sort for efficiency.
      if (size(arr) <= 7) then ! 7 seems to be a good choice for the moment
         DO j = size(arr) - 1, 1, -1
            swapped = .FALSE.
            DO i = 1, j
               IF (arr(i+1) < arr(i)) THEN
                  ! swap arr(i) with arr(i+1)
                  a = arr(i)
                  arr(i) = arr(i + 1)
                  arr(i + 1) = a
                  ! swap indices(i) with indices(i+1)
                  t = indices(i)
                  indices(i) = indices(i + 1)
                  indices(i + 1) = t
                  swapped = .true.
               END IF
            END DO
            IF (.NOT. swapped) EXIT
         END DO
         return
      endif

      ! split list in half and recursively sort both sublists
      m = (size(arr) + 1)/2 ! index where we going to divide the list in two
      call dbcsr_1d_s_sort_low(arr(1:m), indices(1:m), tmp_arr, tmp_idx)
      call dbcsr_1d_s_sort_low(arr(m + 1:), indices(m + 1:), tmp_arr, tmp_idx)

      ! Check for a special case: Can we just concatenate the two sorted sublists?
      ! This leads to O(n) scaling if the input is already sorted.
      if (arr(m+1) < arr(m)) then
         ! ...no - let's merge the two sorted sublists arr(:m) and arr(m+1:)
         ! Merge will be performed directly in arr. Need backup of first sublist.
         tmp_arr(1:m) = arr(1:m)
         tmp_idx(1:m) = indices(1:m)
         i = 1; ! number of elements consumed from 1st sublist
         j = 1; ! number of elements consumed from 2nd sublist
         k = 1; ! number of elements already merged

         do while (i <= m .and. j <= size(arr) - m)
         if (arr(m+j) < tmp_arr(i)) then
            arr(k) = arr(m + j)
            indices(k) = indices(m + j)
            j = j + 1
         else
            arr(k) = tmp_arr(i)
            indices(k) = tmp_idx(i)
            i = i + 1
         endif
         k = k + 1
         enddo

         ! One of the two sublist is now empty.
         ! Copy possibly remaining tail of 1st sublist
         do while (i <= m)
            arr(k) = tmp_arr(i)
            indices(k) = tmp_idx(i)
            i = i + 1
            k = k + 1
         enddo

         ! The possibly remaining tail of 2nd sublist is already at the right spot.

      endif

   end subroutine dbcsr_1d_s_sort_low
# 35 "/__w/dbcsr/dbcsr/src/utils/dbcsr_array_sort.F"
   subroutine dbcsr_1d_i4_sort(arr, n, indices)
      !! Sorts an array inplace using a combination of merge- and bubble-sort.
      !! It also returns the indices, which the elements had before the sort.

      integer, intent(in)                  :: n
         !! length of array
      INTEGER(kind=int_4), dimension(1:n), intent(inout) :: arr
         !! the array to sort
      integer, dimension(1:n), intent(out)   :: indices
         !! returns elements-indices before the sort

      integer :: i
      INTEGER(kind=int_4), pointer, CONTIGUOUS     :: tmp_arr(:)
      integer, pointer, CONTIGUOUS       :: tmp_idx(:)

      if (n == 0) return ! for some reason this is a frequent case in cp2k

      ! scratch space used during the merge step
      allocate (tmp_arr((size(arr) + 1)/2), tmp_idx((size(arr) + 1)/2))

      indices = (/(i, i=1, size(arr))/)

      call dbcsr_1d_i4_sort_low(arr(1:n), indices, tmp_arr, tmp_idx)

      deallocate (tmp_arr, tmp_idx)

   end subroutine dbcsr_1d_i4_sort

   recursive subroutine dbcsr_1d_i4_sort_low(arr, indices, tmp_arr, tmp_idx)
      !! The actual sort routine.
      !! Only dbcsr_1d_i4_sort and itself should call this.

      INTEGER(kind=int_4), dimension(:), intent(inout) :: arr
         !! the array to sort
      integer, dimension(size(arr)), intent(inout) :: indices
         !! elements-indices before the sort
      INTEGER(kind=int_4), dimension((size(arr) + 1)/2), intent(inout) :: tmp_arr
         !! scratch space
      integer, dimension((size(arr) + 1)/2), intent(inout) :: tmp_idx
         !! scratch space
      INTEGER(kind=int_4) :: a
      integer :: t, m, i, j, k
      LOGICAL :: swapped
      ! a,t:  used during swapping of elements in arr and indices

      swapped = .TRUE.

      ! If only a few elements are left we switch to bubble-sort for efficiency.
      if (size(arr) <= 7) then ! 7 seems to be a good choice for the moment
         DO j = size(arr) - 1, 1, -1
            swapped = .FALSE.
            DO i = 1, j
               IF (arr(i+1) < arr(i)) THEN
                  ! swap arr(i) with arr(i+1)
                  a = arr(i)
                  arr(i) = arr(i + 1)
                  arr(i + 1) = a
                  ! swap indices(i) with indices(i+1)
                  t = indices(i)
                  indices(i) = indices(i + 1)
                  indices(i + 1) = t
                  swapped = .true.
               END IF
            END DO
            IF (.NOT. swapped) EXIT
         END DO
         return
      endif

      ! split list in half and recursively sort both sublists
      m = (size(arr) + 1)/2 ! index where we going to divide the list in two
      call dbcsr_1d_i4_sort_low(arr(1:m), indices(1:m), tmp_arr, tmp_idx)
      call dbcsr_1d_i4_sort_low(arr(m + 1:), indices(m + 1:), tmp_arr, tmp_idx)

      ! Check for a special case: Can we just concatenate the two sorted sublists?
      ! This leads to O(n) scaling if the input is already sorted.
      if (arr(m+1) < arr(m)) then
         ! ...no - let's merge the two sorted sublists arr(:m) and arr(m+1:)
         ! Merge will be performed directly in arr. Need backup of first sublist.
         tmp_arr(1:m) = arr(1:m)
         tmp_idx(1:m) = indices(1:m)
         i = 1; ! number of elements consumed from 1st sublist
         j = 1; ! number of elements consumed from 2nd sublist
         k = 1; ! number of elements already merged

         do while (i <= m .and. j <= size(arr) - m)
         if (arr(m+j) < tmp_arr(i)) then
            arr(k) = arr(m + j)
            indices(k) = indices(m + j)
            j = j + 1
         else
            arr(k) = tmp_arr(i)
            indices(k) = tmp_idx(i)
            i = i + 1
         endif
         k = k + 1
         enddo

         ! One of the two sublist is now empty.
         ! Copy possibly remaining tail of 1st sublist
         do while (i <= m)
            arr(k) = tmp_arr(i)
            indices(k) = tmp_idx(i)
            i = i + 1
            k = k + 1
         enddo

         ! The possibly remaining tail of 2nd sublist is already at the right spot.

      endif

   end subroutine dbcsr_1d_i4_sort_low
# 35 "/__w/dbcsr/dbcsr/src/utils/dbcsr_array_sort.F"
   subroutine dbcsr_1d_i8_sort(arr, n, indices)
      !! Sorts an array inplace using a combination of merge- and bubble-sort.
      !! It also returns the indices, which the elements had before the sort.

      integer, intent(in)                  :: n
         !! length of array
      INTEGER(kind=int_8), dimension(1:n), intent(inout) :: arr
         !! the array to sort
      integer, dimension(1:n), intent(out)   :: indices
         !! returns elements-indices before the sort

      integer :: i
      INTEGER(kind=int_8), pointer, CONTIGUOUS     :: tmp_arr(:)
      integer, pointer, CONTIGUOUS       :: tmp_idx(:)

      if (n == 0) return ! for some reason this is a frequent case in cp2k

      ! scratch space used during the merge step
      allocate (tmp_arr((size(arr) + 1)/2), tmp_idx((size(arr) + 1)/2))

      indices = (/(i, i=1, size(arr))/)

      call dbcsr_1d_i8_sort_low(arr(1:n), indices, tmp_arr, tmp_idx)

      deallocate (tmp_arr, tmp_idx)

   end subroutine dbcsr_1d_i8_sort

   recursive subroutine dbcsr_1d_i8_sort_low(arr, indices, tmp_arr, tmp_idx)
      !! The actual sort routine.
      !! Only dbcsr_1d_i8_sort and itself should call this.

      INTEGER(kind=int_8), dimension(:), intent(inout) :: arr
         !! the array to sort
      integer, dimension(size(arr)), intent(inout) :: indices
         !! elements-indices before the sort
      INTEGER(kind=int_8), dimension((size(arr) + 1)/2), intent(inout) :: tmp_arr
         !! scratch space
      integer, dimension((size(arr) + 1)/2), intent(inout) :: tmp_idx
         !! scratch space
      INTEGER(kind=int_8) :: a
      integer :: t, m, i, j, k
      LOGICAL :: swapped
      ! a,t:  used during swapping of elements in arr and indices

      swapped = .TRUE.

      ! If only a few elements are left we switch to bubble-sort for efficiency.
      if (size(arr) <= 7) then ! 7 seems to be a good choice for the moment
         DO j = size(arr) - 1, 1, -1
            swapped = .FALSE.
            DO i = 1, j
               IF (arr(i+1) < arr(i)) THEN
                  ! swap arr(i) with arr(i+1)
                  a = arr(i)
                  arr(i) = arr(i + 1)
                  arr(i + 1) = a
                  ! swap indices(i) with indices(i+1)
                  t = indices(i)
                  indices(i) = indices(i + 1)
                  indices(i + 1) = t
                  swapped = .true.
               END IF
            END DO
            IF (.NOT. swapped) EXIT
         END DO
         return
      endif

      ! split list in half and recursively sort both sublists
      m = (size(arr) + 1)/2 ! index where we going to divide the list in two
      call dbcsr_1d_i8_sort_low(arr(1:m), indices(1:m), tmp_arr, tmp_idx)
      call dbcsr_1d_i8_sort_low(arr(m + 1:), indices(m + 1:), tmp_arr, tmp_idx)

      ! Check for a special case: Can we just concatenate the two sorted sublists?
      ! This leads to O(n) scaling if the input is already sorted.
      if (arr(m+1) < arr(m)) then
         ! ...no - let's merge the two sorted sublists arr(:m) and arr(m+1:)
         ! Merge will be performed directly in arr. Need backup of first sublist.
         tmp_arr(1:m) = arr(1:m)
         tmp_idx(1:m) = indices(1:m)
         i = 1; ! number of elements consumed from 1st sublist
         j = 1; ! number of elements consumed from 2nd sublist
         k = 1; ! number of elements already merged

         do while (i <= m .and. j <= size(arr) - m)
         if (arr(m+j) < tmp_arr(i)) then
            arr(k) = arr(m + j)
            indices(k) = indices(m + j)
            j = j + 1
         else
            arr(k) = tmp_arr(i)
            indices(k) = tmp_idx(i)
            i = i + 1
         endif
         k = k + 1
         enddo

         ! One of the two sublist is now empty.
         ! Copy possibly remaining tail of 1st sublist
         do while (i <= m)
            arr(k) = tmp_arr(i)
            indices(k) = tmp_idx(i)
            i = i + 1
            k = k + 1
         enddo

         ! The possibly remaining tail of 2nd sublist is already at the right spot.

      endif

   end subroutine dbcsr_1d_i8_sort_low
# 148 "/__w/dbcsr/dbcsr/src/utils/dbcsr_array_sort.F"

END MODULE dbcsr_array_sort
