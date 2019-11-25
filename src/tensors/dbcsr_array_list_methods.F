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

#:include "dbcsr_tensor.fypp"
#:set maxdim = maxrank
#:set ndims = range(2,maxdim+1)

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

   SUBROUTINE create_array_list(list, ndata, ${varlist("data")}$)
      !! collects any number of arrays of different sizes into a single array (list%col_data),
      !! storing the indices that start a new array (list%ptr).
      TYPE(array_list), INTENT(OUT)               :: list
         !! list of arrays
      INTEGER, INTENT(IN)                         :: ndata
         !! number of arrays
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL :: ${varlist("data")}$
         !! arrays 1 and 2
      INTEGER                                     :: ptr, size_all

      size_all = 0

#:for dim in range(1, maxdim+1)
      IF (ndata .GE. ${dim}$) THEN
         DBCSR_ASSERT(PRESENT(data_${dim}$))
         size_all = size_all + SIZE(data_${dim}$)
      ENDIF
#:endfor

      ALLOCATE (list%ptr(ndata + 1))
      ALLOCATE (list%col_data(size_all))

      ptr = 1
      list%ptr(1) = ptr

#:for dim in range(1, maxdim+1)
      IF (ndata .GE. ${dim}$) THEN
         list%col_data(ptr:ptr + SIZE(data_${dim}$) - 1) = data_${dim}$ (:)
         ptr = ptr + SIZE(data_${dim}$)
         list%ptr(${dim+1}$) = ptr
      ENDIF
#:endfor

   END SUBROUTINE

   FUNCTION array_sublist(list, i_selected)
      !! extract a subset of arrays

      TYPE(array_list), INTENT(IN)                           :: list
         !! list of arrays
      INTEGER, DIMENSION(:), INTENT(IN)                      :: i_selected
         !! array numbers to retrieve
      TYPE(array_list)                                       :: array_sublist
      INTEGER :: ndata
      INTEGER, ALLOCATABLE, DIMENSION(:) :: ${varlist("data")}$

      ndata = SIZE(i_selected)

#:for dim in range(1, maxdim+1)
      IF (ndata == ${dim}$) THEN
         CALL get_arrays(list, ${varlist("data", nmax=dim)}$, i_selected=i_selected)
         CALL create_array_list(array_sublist, ndata, ${varlist("data", nmax=dim)}$)
      ENDIF
#:endfor
   END FUNCTION

   SUBROUTINE destroy_array_list(list)
      !! destroy array list.
      TYPE(array_list), INTENT(INOUT) :: list

      DEALLOCATE (list%ptr, list%col_data)
   END SUBROUTINE

   SUBROUTINE get_arrays(list, ${varlist("data")}$, i_selected)
      !! Get all arrays contained in list
      TYPE(array_list), INTENT(IN)                       :: list
      INTEGER, ALLOCATABLE, DIMENSION(:), INTENT(OUT), &
         OPTIONAL                                        :: ${varlist("data")}$
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
#:for dim in range(1, maxdim+1)
         IF (ndata > ${dim-1}$) THEN
            CALL allocate_any(data_${dim}$, source=col_data(ptr(o(${dim}$)):ptr(o(${dim}$) + 1) - 1))
         ENDIF
#:endfor
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
      INTEGER, ALLOCATABLE, DIMENSION(:)               :: ${varlist("data")}$
      INTEGER, DIMENSION(number_of_arrays(list_in)), &
         INTENT(IN)                                    :: order

#:for ndim in ndims
      IF (number_of_arrays(list_in) == ${ndim}$) THEN
         CALL get_arrays(list_in, ${varlist("data", nmax=ndim)}$, i_selected=dbcsr_t_inverse_order(order))
         CALL create_array_list(list_out, number_of_arrays(list_in), &
                                ${varlist("data", nmax=ndim)}$)
      ENDIF
#:endfor

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
