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

   IMPLICIT NONE
   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tensor_index'

   PUBLIC :: &
      combine_index, &
      create_nd_to_2d_mapping, &
      destroy_nd_to_2d_mapping, &
      get_2d_indices, &
      dbcsr_t_get_mapping_info, &
      get_nd_indices, &
      nd_to_2d_mapping, &
      ndims_mapping, &
      split_index, &
      dbcsr_t_inverse_order, &
      permute_index

   TYPE nd_to_2d_mapping
      INTEGER                                      :: ndim_nd
      INTEGER                                      :: ndim1_2d
      INTEGER                                      :: ndim2_2d

      INTEGER, DIMENSION(:), ALLOCATABLE           :: dims_nd
      INTEGER(KIND=int_8), DIMENSION(2)            :: dims_2d
      INTEGER, DIMENSION(:), ALLOCATABLE           :: dims1_2d
      INTEGER, DIMENSION(:), ALLOCATABLE           :: dims2_2d

      INTEGER, DIMENSION(:), ALLOCATABLE           :: map1_2d
      INTEGER, DIMENSION(:), ALLOCATABLE           :: map2_2d
      INTEGER, DIMENSION(:), ALLOCATABLE           :: map_nd

      INTEGER                                      :: base
      LOGICAL                                      :: col_major
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
      ENDIF

      IF (PRESENT(base)) THEN
         map%base = base
      ELSE
         map%base = 1
      ENDIF

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

   SUBROUTINE dbcsr_t_get_mapping_info(map, ndim_nd, ndim1_2d, ndim2_2d, dims_2d_i8, dims_2d, dims_nd, dims1_2d, dims2_2d, &
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
      INTEGER, ALLOCATABLE, DIMENSION(:), INTENT(OUT), &
         OPTIONAL                                        :: dims1_2d, dims2_2d, map1_2d, map2_2d
         !! dimensions that map to first 2d index
         !! dimensions that map to second 2d index
         !! indices that map to first 2d index
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
      ENDIF
      IF (PRESENT(dims1_2d)) THEN
         CALL allocate_any(dims1_2d, source=map%dims1_2d)
      ENDIF
      IF (PRESENT(dims2_2d)) THEN
         CALL allocate_any(dims2_2d, source=map%dims2_2d)
      ENDIF
      IF (PRESENT(map1_2d)) THEN
         CALL allocate_any(map1_2d, source=map%map1_2d)
      ENDIF
      IF (PRESENT(map2_2d)) THEN
         CALL allocate_any(map2_2d, source=map%map2_2d)
      ENDIF
      IF (PRESENT(map_nd)) THEN
         map_nd(:) = map%map_nd(:)
      ENDIF
      IF (PRESENT(base)) THEN
         base = map%base
      ENDIF
      IF (PRESENT(col_major)) THEN
         col_major = map%col_major
      ENDIF

   END SUBROUTINE dbcsr_t_get_mapping_info

   FUNCTION combine_index(ind_in, dims, base, col_major) RESULT(ind_out)
      !! transform nd index to flat index

      INTEGER, DIMENSION(:), INTENT(IN)                  :: ind_in, dims
         !! nd index
         !! nd dimensions
      INTEGER, INTENT(IN), OPTIONAL                      :: base
         !! base index, default 1 (Fortran style)
      LOGICAL, INTENT(IN), OPTIONAL                      :: col_major
         !! column major ordering, default .TRUE. (Fortran style)
      INTEGER(KIND=int_8)                                :: ind_out
         !! flat index

      INTEGER                                            :: i_dim, my_base
      LOGICAL                                            :: my_col_major

      DBCSR_ASSERT(SIZE(ind_in) .EQ. SIZE(dims))

      IF (PRESENT(base)) THEN
         my_base = base
      ELSE
         my_base = 1
      ENDIF

      IF (PRESENT(col_major)) THEN
         my_col_major = col_major
      ELSE
         my_col_major = .TRUE.
      ENDIF

      IF (my_col_major) THEN
         i_dim = SIZE(dims) - 1
         ind_out = ind_in(i_dim + 1)
         DO WHILE (i_dim .GE. 1)
            ind_out = (ind_out - my_base)*dims(i_dim) + (ind_in(i_dim) - my_base) + my_base
            i_dim = i_dim - 1
         ENDDO
      ELSE
         i_dim = 2
         ind_out = ind_in(i_dim - 1)
         DO WHILE (i_dim .LE. SIZE(dims))
            ind_out = (ind_out - my_base)*dims(i_dim) + (ind_in(i_dim) - my_base) + my_base
            i_dim = i_dim + 1
         ENDDO
      ENDIF
   END FUNCTION combine_index

   FUNCTION split_index(ind_in, dims, base, col_major) RESULT(ind_out)
      !! transform flat index to nd index

      INTEGER(KIND=int_8), INTENT(IN)                    :: ind_in
         !! flat index
      INTEGER, DIMENSION(:), INTENT(IN)                  :: dims
         !! nd dimensions
      INTEGER, INTENT(IN), OPTIONAL                      :: base
         !! base index, default 1 (Fortran style)
      LOGICAL, INTENT(IN), OPTIONAL                      :: col_major
         !! column major ordering, default .TRUE. (Fortran style)
      INTEGER, DIMENSION(SIZE(dims))                     :: ind_out
         !! nd index

      INTEGER(KIND=int_8)                                :: tmp
      INTEGER                                            :: i_dim, my_base
      LOGICAL                                            :: my_col_major

      IF (PRESENT(base)) THEN
         my_base = base
      ELSE
         my_base = 1
      ENDIF

      IF (PRESENT(col_major)) THEN
         my_col_major = col_major
      ELSE
         my_col_major = .TRUE.
      ENDIF

      IF (my_col_major) THEN
         i_dim = 1
         tmp = ind_in
         DO WHILE (i_dim .LE. SIZE(dims))
            ind_out(i_dim) = INT(MOD(tmp - my_base, INT(dims(i_dim), int_8)) + my_base)
            tmp = (tmp - my_base)/dims(i_dim) + my_base
            i_dim = i_dim + 1
         END DO
      ELSE
         i_dim = SIZE(dims)
         tmp = ind_in
         DO WHILE (i_dim .GE. 1)
            ind_out(i_dim) = INT(MOD(tmp - my_base, INT(dims(i_dim), int_8)) + my_base)
            tmp = (tmp - my_base)/dims(i_dim) + my_base
            i_dim = i_dim - 1
         END DO
      ENDIF
   END FUNCTION split_index

   FUNCTION get_2d_indices(map, ind_in) RESULT(ind_out)
      !! transform nd index to 2d index, using info from index mapping.

      TYPE(nd_to_2d_mapping), INTENT(IN)                 :: map
         !! index mapping
      INTEGER, DIMENSION(ndims_mapping(map)), INTENT(IN) :: ind_in
         !! nd index
      INTEGER(KIND=int_8), DIMENSION(2)                  :: ind_out
         !! 2d index

      INTEGER, DIMENSION(map%ndim2_2d)                   :: ind_2
      INTEGER, DIMENSION(map%ndim1_2d)                   :: ind_1

      ind_1(:) = ind_in(map%map1_2d)
      ind_2(:) = ind_in(map%map2_2d)
      ind_out(1) = combine_index(ind_1, map%dims1_2d, base=map%base, col_major=map%col_major)
      ind_out(2) = combine_index(ind_2, map%dims2_2d, base=map%base, col_major=map%col_major)
   END FUNCTION get_2d_indices

   FUNCTION get_nd_indices(map, ind_in) RESULT(ind_out)
      !! transform 2d index to nd index, using info from index mapping.

      TYPE(nd_to_2d_mapping), INTENT(IN)                 :: map
         !! index mapping
      INTEGER(KIND=int_8), DIMENSION(2), INTENT(IN)      :: ind_in
         !! 2d index
      INTEGER, DIMENSION(ndims_mapping(map))             :: ind_out
         !! nd index

      INTEGER, DIMENSION(map%ndim2_2d)                   :: ind_2
      INTEGER, DIMENSION(map%ndim1_2d)                   :: ind_1

      ind_1(:) = split_index(ind_in(1), map%dims1_2d, base=map%base, col_major=map%col_major)
      ind_2(:) = split_index(ind_in(2), map%dims2_2d, base=map%base, col_major=map%col_major)

      ind_out(map%map1_2d) = ind_1
      ind_out(map%map2_2d) = ind_2
   END FUNCTION get_nd_indices

   FUNCTION dbcsr_t_inverse_order(order)
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
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: map1_2d, map1_2d_reorder, map2_2d, &
                                                            map2_2d_reorder
      INTEGER, DIMENSION(ndims_mapping(map_in))          :: dims_nd, dims_reorder

      CALL dbcsr_t_get_mapping_info(map_in, ndim_nd, dims_nd=dims_nd, map1_2d=map1_2d, map2_2d=map2_2d)

      dims_reorder(order) = dims_nd

      CALL allocate_any(map1_2d_reorder, shape_spec=SHAPE(map1_2d))
      CALL allocate_any(map2_2d_reorder, shape_spec=SHAPE(map2_2d))

      map1_2d_reorder(:) = order(map1_2d)
      map2_2d_reorder(:) = order(map2_2d)

      CALL create_nd_to_2d_mapping(map_out, dims_reorder, map1_2d_reorder, map2_2d_reorder)
   END SUBROUTINE
END MODULE dbcsr_tensor_index
