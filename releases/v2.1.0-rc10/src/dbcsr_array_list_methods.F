# 1 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F" 1
!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_array_list_methods
   !! Representation of arbitrary number of 1d integer arrays with arbitrary sizes.
   !! This is needed for generic handling of dimension-specific tensor quantities (such as block index).

# 1 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp" 1
# 9 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp"

# 66 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp"
# 15 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F" 2
# 16 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
# 17 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"

   USE dbcsr_tensor_index, ONLY: dbcsr_t_inverse_order
   USE dbcsr_allocate_wrap, ONLY: allocate_any

#include "base/dbcsr_base_uses.f90"
   IMPLICIT NONE
   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_array_list_methods'

   PUBLIC  :: &
      array_eq_i, &
      array_list, &
      array_offsets, &
      array_sublist, &
      create_array_list, &
      destroy_array_list, &
      get_array_elements, &
      get_arrays, &
      get_ith_array, &
      number_of_arrays, &
      reorder_arrays, &
      sizes_of_arrays, &
      sum_of_arrays, &
      check_equal

   TYPE array_list
      INTEGER, DIMENSION(:), ALLOCATABLE :: col_data
      INTEGER, DIMENSION(:), ALLOCATABLE :: ptr
   END TYPE

CONTAINS

   PURE FUNCTION number_of_arrays(list)
      !! number of arrays stored in list
      TYPE(array_list), INTENT(IN) :: list
      INTEGER                      :: number_of_arrays

      number_of_arrays = SIZE(list%ptr) - 1

   END FUNCTION number_of_arrays

   FUNCTION get_array_elements(list, indices)
      !! Get an element for each array.

      TYPE(array_list), INTENT(IN)                           :: list
      INTEGER, DIMENSION(number_of_arrays(list)), INTENT(IN) :: indices
         !! element index for each array
      INTEGER, DIMENSION(number_of_arrays(list))             :: get_array_elements

      INTEGER                                                :: i, ind

      DO i = 1, SIZE(indices)
         ind = indices(i) + list%ptr(i) - 1
         get_array_elements(i) = list%col_data(ind)
      ENDDO

   END FUNCTION get_array_elements

   SUBROUTINE create_array_list(list, ndata, data_1, data_2, data_3, data_4)
      !! collects any number of arrays of different sizes into a single array (list%col_data),
      !! storing the indices that start a new array (list%ptr).
      TYPE(array_list), INTENT(OUT)               :: list
         !! list of arrays
      INTEGER, INTENT(IN)                         :: ndata
         !! number of arrays
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL :: data_1, data_2, data_3, data_4
         !! arrays 1 and 2
      INTEGER                                     :: ptr, size_all

      size_all = 0

# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
      IF (ndata .GE. 1) THEN
         DBCSR_ASSERT(PRESENT(data_1))
         size_all = size_all + SIZE(data_1)
      ENDIF
# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
      IF (ndata .GE. 2) THEN
         DBCSR_ASSERT(PRESENT(data_2))
         size_all = size_all + SIZE(data_2)
      ENDIF
# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
      IF (ndata .GE. 3) THEN
         DBCSR_ASSERT(PRESENT(data_3))
         size_all = size_all + SIZE(data_3)
      ENDIF
# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
      IF (ndata .GE. 4) THEN
         DBCSR_ASSERT(PRESENT(data_4))
         size_all = size_all + SIZE(data_4)
      ENDIF
# 94 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"

      ALLOCATE (list%ptr(ndata + 1))
      ALLOCATE (list%col_data(size_all))

      ptr = 1
      list%ptr(1) = ptr

# 102 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
      IF (ndata .GE. 1) THEN
         list%col_data(ptr:ptr + SIZE(data_1) - 1) = data_1 (:)
         ptr = ptr + SIZE(data_1)
         list%ptr(2) = ptr
      ENDIF
# 102 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
      IF (ndata .GE. 2) THEN
         list%col_data(ptr:ptr + SIZE(data_2) - 1) = data_2 (:)
         ptr = ptr + SIZE(data_2)
         list%ptr(3) = ptr
      ENDIF
# 102 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
      IF (ndata .GE. 3) THEN
         list%col_data(ptr:ptr + SIZE(data_3) - 1) = data_3 (:)
         ptr = ptr + SIZE(data_3)
         list%ptr(4) = ptr
      ENDIF
# 102 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
      IF (ndata .GE. 4) THEN
         list%col_data(ptr:ptr + SIZE(data_4) - 1) = data_4 (:)
         ptr = ptr + SIZE(data_4)
         list%ptr(5) = ptr
      ENDIF
# 108 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"

   END SUBROUTINE

   FUNCTION array_sublist(list, i_selected)
      !! extract a subset of arrays

      TYPE(array_list), INTENT(IN)                           :: list
         !! list of arrays
      INTEGER, DIMENSION(:), INTENT(IN)                      :: i_selected
         !! array numbers to retrieve
      TYPE(array_list)                                       :: array_sublist
      INTEGER :: ndata
      INTEGER, ALLOCATABLE, DIMENSION(:) :: data_1, data_2, data_3, data_4

      ndata = SIZE(i_selected)

# 125 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
      IF (ndata == 1) THEN
         CALL get_arrays(list, data_1, i_selected=i_selected)
         CALL create_array_list(array_sublist, ndata, data_1)
      ENDIF
# 125 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
      IF (ndata == 2) THEN
         CALL get_arrays(list, data_1, data_2, i_selected=i_selected)
         CALL create_array_list(array_sublist, ndata, data_1, data_2)
      ENDIF
# 125 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
      IF (ndata == 3) THEN
         CALL get_arrays(list, data_1, data_2, data_3, i_selected=i_selected)
         CALL create_array_list(array_sublist, ndata, data_1, data_2, data_3)
      ENDIF
# 125 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
      IF (ndata == 4) THEN
         CALL get_arrays(list, data_1, data_2, data_3, data_4, i_selected=i_selected)
         CALL create_array_list(array_sublist, ndata, data_1, data_2, data_3, data_4)
      ENDIF
# 130 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
   END FUNCTION

   SUBROUTINE destroy_array_list(list)
      !! destroy array list.
      TYPE(array_list), INTENT(INOUT) :: list

      DEALLOCATE (list%ptr, list%col_data)
   END SUBROUTINE

   SUBROUTINE get_arrays(list, data_1, data_2, data_3, data_4, i_selected)
      !! Get all arrays contained in list
      TYPE(array_list), INTENT(IN)                       :: list
      INTEGER, ALLOCATABLE, DIMENSION(:), INTENT(OUT), &
         OPTIONAL                                        :: data_1, data_2, data_3, data_4
         !! arrays 1 and 2
      INTEGER, DIMENSION(:), INTENT(IN), &
         OPTIONAL                                        :: i_selected
         !! array numbers to retrieve (if not present, all arrays are returned)
      INTEGER                                            :: i, ndata
      INTEGER, DIMENSION(number_of_arrays(list))         :: o

      o(:) = 0
      IF (PRESENT(i_selected)) THEN
         ndata = SIZE(i_selected)
         o(1:ndata) = i_selected(:)
      ELSE
         ndata = number_of_arrays(list)
         o(1:ndata) = (/(i, i=1, ndata)/)
      ENDIF

      ASSOCIATE (ptr=>list%ptr, col_data=>list%col_data)
# 162 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
         IF (ndata > 0) THEN
            CALL allocate_any(data_1, source=col_data(ptr(o(1)):ptr(o(1) + 1) - 1))
         ENDIF
# 162 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
         IF (ndata > 1) THEN
            CALL allocate_any(data_2, source=col_data(ptr(o(2)):ptr(o(2) + 1) - 1))
         ENDIF
# 162 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
         IF (ndata > 2) THEN
            CALL allocate_any(data_3, source=col_data(ptr(o(3)):ptr(o(3) + 1) - 1))
         ENDIF
# 162 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
         IF (ndata > 3) THEN
            CALL allocate_any(data_4, source=col_data(ptr(o(4)):ptr(o(4) + 1) - 1))
         ENDIF
# 166 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
      END ASSOCIATE

   END SUBROUTINE get_arrays

   SUBROUTINE get_ith_array(list, i, array)
      !! get ith array
      TYPE(array_list), INTENT(IN)                    :: list
      INTEGER, INTENT(IN)                             :: i
      INTEGER, DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: array

      ASSOCIATE (ptr=>list%ptr, col_data=>list%col_data)
         DBCSR_ASSERT(i <= number_of_arrays(list))

         CALL allocate_any(array, source=col_data(ptr(i):ptr(i + 1) - 1))
      END ASSOCIATE
   END SUBROUTINE

   FUNCTION sizes_of_arrays(list)
      !! sizes of arrays stored in list
      TYPE(array_list), INTENT(IN)       :: list
      INTEGER, ALLOCATABLE, DIMENSION(:) :: sizes_of_arrays

      INTEGER                            :: i_data, num_data

      num_data = number_of_arrays(list)
      ALLOCATE (sizes_of_arrays(num_data))
      DO i_data = 1, num_data
         sizes_of_arrays(i_data) = list%ptr(i_data + 1) - list%ptr(i_data)
      ENDDO
   END FUNCTION sizes_of_arrays

   FUNCTION sum_of_arrays(list)
      !! sum of all elements for each array stored in list
      TYPE(array_list), INTENT(IN)       :: list
      INTEGER, ALLOCATABLE, DIMENSION(:) :: sum_of_arrays

      INTEGER                            :: i_data, num_data

      num_data = number_of_arrays(list)
      ALLOCATE (sum_of_arrays(num_data))
      DO i_data = 1, num_data
         sum_of_arrays(i_data) = SUM(list%col_data(list%ptr(i_data):list%ptr(i_data + 1) - 1))
      ENDDO

   END FUNCTION sum_of_arrays

   SUBROUTINE array_offsets(list_in, list_out)
      !! partial sums of array elements.
      TYPE(array_list), INTENT(IN)  :: list_in
      TYPE(array_list), INTENT(OUT) :: list_out

      INTEGER                       :: i_data, i_ptr, num_data, partial_sum

      num_data = number_of_arrays(list_in)
      CALL allocate_any(list_out%ptr, source=list_in%ptr)
      ALLOCATE (list_out%col_data(SIZE(list_in%col_data)))
      DO i_data = 1, num_data
         partial_sum = 1
         DO i_ptr = list_out%ptr(i_data), list_out%ptr(i_data + 1) - 1
            list_out%col_data(i_ptr) = partial_sum
            partial_sum = partial_sum + list_in%col_data(i_ptr)
         ENDDO
      ENDDO
   END SUBROUTINE

   SUBROUTINE reorder_arrays(list_in, list_out, order)
      !! reorder array list.
      TYPE(array_list), INTENT(IN)                     :: list_in
      TYPE(array_list), INTENT(OUT)                    :: list_out
      INTEGER, ALLOCATABLE, DIMENSION(:)               :: data_1, data_2, data_3, data_4
      INTEGER, DIMENSION(number_of_arrays(list_in)), &
         INTENT(IN)                                    :: order

# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
      IF (number_of_arrays(list_in) == 2) THEN
         CALL get_arrays(list_in, data_1, data_2, i_selected=dbcsr_t_inverse_order(order))
         CALL create_array_list(list_out, number_of_arrays(list_in), &
                                data_1, data_2)
      ENDIF
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
      IF (number_of_arrays(list_in) == 3) THEN
         CALL get_arrays(list_in, data_1, data_2, data_3, i_selected=dbcsr_t_inverse_order(order))
         CALL create_array_list(list_out, number_of_arrays(list_in), &
                                data_1, data_2, data_3)
      ENDIF
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"
      IF (number_of_arrays(list_in) == 4) THEN
         CALL get_arrays(list_in, data_1, data_2, data_3, data_4, i_selected=dbcsr_t_inverse_order(order))
         CALL create_array_list(list_out, number_of_arrays(list_in), &
                                data_1, data_2, data_3, data_4)
      ENDIF
# 246 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_array_list_methods.F"

   END SUBROUTINE

   FUNCTION check_equal(list1, list2)
      !! check whether two array lists are equal
      TYPE(array_list), INTENT(IN)  :: list1, list2
      LOGICAL :: check_equal

      check_equal = array_eq_i(list1%col_data, list2%col_data) .AND. array_eq_i(list1%ptr, list2%ptr)
   END FUNCTION

   PURE FUNCTION array_eq_i(arr1, arr2)
      !! check whether two arrays are equal
      INTEGER, INTENT(IN), DIMENSION(:) :: arr1
      INTEGER, INTENT(IN), DIMENSION(:) :: arr2
      LOGICAL                           :: array_eq_i

      array_eq_i = .FALSE.
      IF (SIZE(arr1) .EQ. SIZE(arr2)) array_eq_i = ALL(arr1 == arr2)

   END FUNCTION

END MODULE dbcsr_array_list_methods
