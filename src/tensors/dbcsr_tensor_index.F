!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tensor_index
   !! tensor index and mapping to DBCSR index
   USE dbcsr_allocate_wrap, ONLY: allocate_any
   USE dbcsr_kinds, ONLY: int_8
#include "base/dbcsr_base_uses.f90"
   #:include "dbcsr_tensor.fypp"

   IMPLICIT NONE
   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tensor_index'

   PUBLIC :: &
      combine_tensor_index, &
      combine_pgrid_index, &
      create_nd_to_2d_mapping, &
      destroy_nd_to_2d_mapping, &
      get_2d_indices_tensor, &
      get_2d_indices_pgrid, &
      dbcsr_t_get_mapping_info, &
      get_nd_indices_tensor, &
      get_nd_indices_pgrid, &
      nd_to_2d_mapping, &
      ndims_mapping, &
      split_tensor_index, &
      split_pgrid_index, &
      ndims_mapping_row, &
      ndims_mapping_column, &
      dbcsr_t_inverse_order, &
      permute_index

   TYPE nd_to_2d_mapping
      INTEGER                                      :: ndim_nd = -1
      INTEGER                                      :: ndim1_2d = -1
      INTEGER                                      :: ndim2_2d = -1

      INTEGER, DIMENSION(:), ALLOCATABLE           :: dims_nd
      INTEGER(KIND=int_8), DIMENSION(2)            :: dims_2d = -1_int_8
      INTEGER, DIMENSION(:), ALLOCATABLE           :: dims1_2d
      INTEGER, DIMENSION(:), ALLOCATABLE           :: dims2_2d

      INTEGER, DIMENSION(:), ALLOCATABLE           :: map1_2d
      INTEGER, DIMENSION(:), ALLOCATABLE           :: map2_2d
      INTEGER, DIMENSION(:), ALLOCATABLE           :: map_nd

      INTEGER                                      :: base = -1
      LOGICAL                                      :: col_major = .FALSE.
   END TYPE nd_to_2d_mapping

CONTAINS

   SUBROUTINE create_nd_to_2d_mapping(map, dims, map1_2d, map2_2d, base, col_major)
      !! Create all data needed to quickly map between nd index and 2d index.

      TYPE(nd_to_2d_mapping), INTENT(OUT)                :: map
         !! index mapping data
      INTEGER, DIMENSION(:), INTENT(IN)                  :: dims, map1_2d, map2_2d
         !! nd sizes
         !! which nd-indices map to first matrix index and in which order
         !! which nd-indices map to second matrix index and in which order
      INTEGER, INTENT(IN), OPTIONAL                      :: base
         !! base index (1 for Fortran-style, 0 for C-style, default is 1)
      LOGICAL, INTENT(IN), OPTIONAL                      :: col_major
         !! whether index should be column major order (.TRUE. for Fortran-style, .FALSE. for C-style, default is .TRUE.).

      INTEGER                                            :: i

      IF (PRESENT(col_major)) THEN
         map%col_major = col_major
      ELSE
         map%col_major = .TRUE.
      END IF

      IF (PRESENT(base)) THEN
         map%base = base
      ELSE
         map%base = 1
      END IF

      map%ndim1_2d = SIZE(map1_2d)
      map%ndim2_2d = SIZE(map2_2d)
      map%ndim_nd = SIZE(dims)

      CALL allocate_any(map%map1_2d, source=map1_2d)
      CALL allocate_any(map%map2_2d, source=map2_2d)
      CALL allocate_any(map%dims_nd, source=dims)
      CALL allocate_any(map%dims1_2d, source=dims(map1_2d))
      CALL allocate_any(map%dims2_2d, source=dims(map2_2d))

      ALLOCATE (map%map_nd(map%ndim_nd))
      map%map_nd(map1_2d) = (/(i, i=1, SIZE(map1_2d))/)
      map%map_nd(map2_2d) = (/(i + SIZE(map1_2d), i=1, SIZE(map2_2d))/)

      map%dims_2d = [PRODUCT(INT(map%dims1_2d, KIND=int_8)), PRODUCT(INT(map%dims2_2d, KIND=int_8))]

   END SUBROUTINE create_nd_to_2d_mapping

   SUBROUTINE destroy_nd_to_2d_mapping(map)
      TYPE(nd_to_2d_mapping), INTENT(INOUT)              :: map

      DEALLOCATE (map%dims1_2d)
      DEALLOCATE (map%dims2_2d)
      DEALLOCATE (map%map1_2d)
      DEALLOCATE (map%map2_2d)
      DEALLOCATE (map%map_nd)
      DEALLOCATE (map%dims_nd)
   END SUBROUTINE destroy_nd_to_2d_mapping

   PURE FUNCTION ndims_mapping(map)
      TYPE(nd_to_2d_mapping), INTENT(IN)                 :: map
      INTEGER                                            :: ndims_mapping

      ndims_mapping = map%ndim_nd
   END FUNCTION

   PURE FUNCTION ndims_mapping_row(map)
      !! how many tensor dimensions are mapped to matrix row
      TYPE(nd_to_2d_mapping), INTENT(IN) :: map
      INTEGER :: ndims_mapping_row
      ndims_mapping_row = map%ndim1_2d
   END FUNCTION

   PURE FUNCTION ndims_mapping_column(map)
      !! how many tensor dimensions are mapped to matrix column
      TYPE(nd_to_2d_mapping), INTENT(IN) :: map
      INTEGER :: ndims_mapping_column
      ndims_mapping_column = map%ndim2_2d
   END FUNCTION

   PURE SUBROUTINE dbcsr_t_get_mapping_info(map, ndim_nd, ndim1_2d, ndim2_2d, dims_2d_i8, dims_2d, dims_nd, dims1_2d, dims2_2d, &
                                            map1_2d, map2_2d, map_nd, base, col_major)
      !! get mapping info

      TYPE(nd_to_2d_mapping), INTENT(IN)                 :: map
         !! index mapping data.
      INTEGER, INTENT(OUT), OPTIONAL                     :: ndim_nd, ndim1_2d, ndim2_2d
         !! number of dimensions
         !! number of dimensions that map to first 2d index
         !! number of dimensions that map to first 2d index
      INTEGER(KIND=int_8), DIMENSION(2), INTENT(OUT), OPTIONAL       :: dims_2d_i8
      INTEGER, DIMENSION(2), INTENT(OUT), OPTIONAL :: dims_2d
         !! 2d dimensions
      INTEGER, DIMENSION(ndims_mapping(map)), &
         INTENT(OUT), OPTIONAL                           :: dims_nd
         !! nd dimensions
      INTEGER, DIMENSION(ndims_mapping_row(map)), INTENT(OUT), &
         OPTIONAL                                        :: dims1_2d
         !! dimensions that map to first 2d index
      INTEGER, DIMENSION(ndims_mapping_column(map)), INTENT(OUT), &
         OPTIONAL                                        :: dims2_2d
         !! dimensions that map to second 2d index
      INTEGER, DIMENSION(ndims_mapping_row(map)), INTENT(OUT), &
         OPTIONAL                                        :: map1_2d
         !! indices that map to first 2d index
      INTEGER, DIMENSION(ndims_mapping_column(map)), INTENT(OUT), &
         OPTIONAL                                        :: map2_2d
         !! indices that map to second 2d index
      INTEGER, DIMENSION(ndims_mapping(map)), &
         INTENT(OUT), OPTIONAL                           :: map_nd
         !! inverse of [map1_2d, map2_2d]
      INTEGER, INTENT(OUT), OPTIONAL                     :: base
         !! base index
      LOGICAL, INTENT(OUT), OPTIONAL                     :: col_major
         !! is index in column major order

      IF (PRESENT(ndim_nd)) ndim_nd = map%ndim_nd
      IF (PRESENT(ndim1_2d)) ndim1_2d = map%ndim1_2d
      IF (PRESENT(ndim2_2d)) ndim2_2d = map%ndim2_2d
      IF (PRESENT(dims_2d_i8)) dims_2d_i8(:) = map%dims_2d(:)
      IF (PRESENT(dims_2d)) dims_2d(:) = INT(map%dims_2d(:))
      IF (PRESENT(dims_nd)) THEN
         dims_nd(:) = map%dims_nd(:)
      END IF
      IF (PRESENT(dims1_2d)) THEN
         dims1_2d(:) = map%dims1_2d
      END IF
      IF (PRESENT(dims2_2d)) THEN
         dims2_2d(:) = map%dims2_2d
      END IF
      IF (PRESENT(map1_2d)) THEN
         map1_2d(:) = map%map1_2d
      END IF
      IF (PRESENT(map2_2d)) THEN
         map2_2d(:) = map%map2_2d
      END IF
      IF (PRESENT(map_nd)) THEN
         map_nd(:) = map%map_nd(:)
      END IF
      IF (PRESENT(base)) THEN
         base = map%base
      END IF
      IF (PRESENT(col_major)) THEN
         col_major = map%col_major
      END IF

   END SUBROUTINE dbcsr_t_get_mapping_info

   PURE FUNCTION combine_tensor_index(ind_in, dims) RESULT(ind_out)
      !! transform nd index to flat index
      INTEGER, DIMENSION(:), INTENT(IN)                  :: ind_in, dims
         !! nd index
         !! nd dimensions
      INTEGER(KIND=int_8)                                :: ind_out
         !! flat index
      INTEGER                                            :: i_dim

      ind_out = ind_in(SIZE(dims))
      DO i_dim = SIZE(dims) - 1, 1, -1
         ind_out = (ind_out - 1)*dims(i_dim) + ind_in(i_dim)
      END DO

   END FUNCTION

   PURE FUNCTION combine_pgrid_index(ind_in, dims) RESULT(ind_out)
      !! transform nd index to flat index

      INTEGER, DIMENSION(:), INTENT(IN)                  :: ind_in, dims
         !! nd index
         !! nd dimensions
      INTEGER                                            :: ind_out
         !! flat index

      INTEGER                                            :: i_dim

      ind_out = ind_in(1)
      DO i_dim = 2, SIZE(dims)
         ind_out = ind_out*dims(i_dim) + ind_in(i_dim)
      END DO
   END FUNCTION

   PURE FUNCTION split_tensor_index(ind_in, dims) RESULT(ind_out)
      !! transform flat index to nd index

      INTEGER(KIND=int_8), INTENT(IN)                    :: ind_in
         !! flat index
      INTEGER, DIMENSION(:), INTENT(IN)                  :: dims
         !! nd dimensions
      INTEGER, DIMENSION(SIZE(dims))                     :: ind_out
         !! nd index

      INTEGER(KIND=int_8)                                :: tmp
      INTEGER                                            :: i_dim

      tmp = ind_in
      DO i_dim = 1, SIZE(dims)
         ind_out(i_dim) = INT(MOD(tmp - 1, INT(dims(i_dim), int_8)) + 1)
         tmp = (tmp - 1)/dims(i_dim) + 1
      END DO

   END FUNCTION

   PURE FUNCTION split_pgrid_index(ind_in, dims) RESULT(ind_out)
      !! transform flat index to nd index

      INTEGER, INTENT(IN)                                :: ind_in
         !! flat index
      INTEGER, DIMENSION(:), INTENT(IN)                  :: dims
         !! nd dimensions
      INTEGER, DIMENSION(SIZE(dims))                     :: ind_out
         !! nd index

      INTEGER                                            :: tmp
      INTEGER                                            :: i_dim

      tmp = ind_in
      DO i_dim = SIZE(dims), 1, -1
         ind_out(i_dim) = MOD(tmp, dims(i_dim))
         tmp = tmp/dims(i_dim)
      END DO
   END FUNCTION

   PURE FUNCTION get_2d_indices_tensor(map, ind_in) RESULT(ind_out)
      !! transform nd index to 2d index, using info from index mapping.

      TYPE(nd_to_2d_mapping), INTENT(IN)                 :: map
         !! index mapping
      INTEGER, DIMENSION(map%ndim_nd), INTENT(IN) :: ind_in
         !! nd index
      INTEGER(KIND=int_8), DIMENSION(2)                  :: ind_out
         !! 2d index
      INTEGER :: i
      INTEGER, DIMENSION(${maxrank}$)                    :: ind_tmp

      DO i = 1, map%ndim1_2d
         ind_tmp(i) = ind_in(map%map1_2d(i))
      END DO
      ind_out(1) = combine_tensor_index(ind_tmp(:map%ndim1_2d), map%dims1_2d)

      DO i = 1, map%ndim2_2d
         ind_tmp(i) = ind_in(map%map2_2d(i))
      END DO
      ind_out(2) = combine_tensor_index(ind_tmp(:map%ndim2_2d), map%dims2_2d)
   END FUNCTION

   PURE FUNCTION get_2d_indices_pgrid(map, ind_in) RESULT(ind_out)
      !! transform nd index to 2d index, using info from index mapping.

      TYPE(nd_to_2d_mapping), INTENT(IN)                 :: map
         !! index mapping
      INTEGER, DIMENSION(map%ndim_nd), INTENT(IN) :: ind_in
         !! nd index
      INTEGER, DIMENSION(2)                              :: ind_out
         !! 2d index
      INTEGER :: i
      INTEGER, DIMENSION(${maxrank}$)                    :: ind_tmp

      DO i = 1, map%ndim1_2d
         ind_tmp(i) = ind_in(map%map1_2d(i))
      END DO
      ind_out(1) = combine_pgrid_index(ind_tmp(:map%ndim1_2d), map%dims1_2d)

      DO i = 1, map%ndim2_2d
         ind_tmp(i) = ind_in(map%map2_2d(i))
      END DO
      ind_out(2) = combine_pgrid_index(ind_tmp(:map%ndim2_2d), map%dims2_2d)
   END FUNCTION

   PURE FUNCTION get_nd_indices_tensor(map, ind_in) RESULT(ind_out)
      !! transform 2d index to nd index, using info from index mapping.

      TYPE(nd_to_2d_mapping), INTENT(IN)                 :: map
         !! index mapping
      INTEGER(KIND=int_8), DIMENSION(2), INTENT(IN)      :: ind_in
         !! 2d index
      INTEGER, DIMENSION(map%ndim_nd)                    :: ind_out
         !! nd index
      INTEGER, DIMENSION(${maxrank}$)                    :: ind_tmp
      INTEGER                                            :: i

      ind_tmp(:map%ndim1_2d) = split_tensor_index(ind_in(1), map%dims1_2d)

      DO i = 1, map%ndim1_2d
         ind_out(map%map1_2d(i)) = ind_tmp(i)
      END DO

      ind_tmp(:map%ndim2_2d) = split_tensor_index(ind_in(2), map%dims2_2d)

      DO i = 1, map%ndim2_2d
         ind_out(map%map2_2d(i)) = ind_tmp(i)
      END DO

   END FUNCTION

   PURE FUNCTION get_nd_indices_pgrid(map, ind_in) RESULT(ind_out)
      !! transform 2d index to nd index, using info from index mapping.

      TYPE(nd_to_2d_mapping), INTENT(IN)                 :: map
         !! index mapping
      INTEGER, DIMENSION(2), INTENT(IN)                  :: ind_in
         !! 2d index
      INTEGER, DIMENSION(map%ndim_nd)                    :: ind_out
         !! nd index

      ind_out(map%map1_2d) = split_pgrid_index(ind_in(1), map%dims1_2d)
      ind_out(map%map2_2d) = split_pgrid_index(ind_in(2), map%dims2_2d)

   END FUNCTION

   PURE FUNCTION dbcsr_t_inverse_order(order)
      !! Invert order
      INTEGER, DIMENSION(:), INTENT(IN)                  :: order
      INTEGER, DIMENSION(SIZE(order))                    :: dbcsr_t_inverse_order

      INTEGER                                            :: i

      dbcsr_t_inverse_order(order) = (/(i, i=1, SIZE(order))/)
   END FUNCTION

   SUBROUTINE permute_index(map_in, map_out, order)
      !! reorder tensor index (no data)
      TYPE(nd_to_2d_mapping), INTENT(IN)                 :: map_in
      TYPE(nd_to_2d_mapping), INTENT(OUT)                :: map_out
      INTEGER, DIMENSION(ndims_mapping(map_in)), &
         INTENT(IN)                                      :: order

      INTEGER                                            :: ndim_nd
      INTEGER, DIMENSION(ndims_mapping_row(map_in))       :: map1_2d, map1_2d_reorder
      INTEGER, DIMENSION(ndims_mapping_column(map_in))    :: map2_2d, map2_2d_reorder
      INTEGER, DIMENSION(ndims_mapping(map_in))          :: dims_nd, dims_reorder

      CALL dbcsr_t_get_mapping_info(map_in, ndim_nd, dims_nd=dims_nd, map1_2d=map1_2d, map2_2d=map2_2d)

      dims_reorder(order) = dims_nd

      map1_2d_reorder(:) = order(map1_2d)
      map2_2d_reorder(:) = order(map2_2d)

      CALL create_nd_to_2d_mapping(map_out, dims_reorder, map1_2d_reorder, map2_2d_reorder)
   END SUBROUTINE
END MODULE dbcsr_tensor_index
