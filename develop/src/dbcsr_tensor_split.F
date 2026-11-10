# 1 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F" 1
!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tensor_split

   !! Routines to split blocks and to convert between tensors with different block sizes.
# 1 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp" 1
# 9 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp"

# 241 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp"
# 14 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F" 2
# 15 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 16 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

   USE dbcsr_allocate_wrap, ONLY: allocate_any
   USE dbcsr_array_list_methods, ONLY: get_ith_array
   USE dbcsr_tensor_block, ONLY: dbcsr_t_iterator_type, &
                                 dbcsr_t_get_block, &
                                 dbcsr_t_put_block, &
                                 dbcsr_t_iterator_start, &
                                 dbcsr_t_iterator_blocks_left, &
                                 dbcsr_t_iterator_stop, &
                                 dbcsr_t_iterator_next_block, &
                                 dbcsr_t_reserve_blocks, &
                                 dbcsr_t_reserved_block_indices
   USE dbcsr_tensor_index, ONLY: dbcsr_t_get_mapping_info, &
                                 dbcsr_t_inverse_order
   USE dbcsr_tensor_types, ONLY: dbcsr_t_create, &
                                 dbcsr_t_get_data_type, &
                                 dbcsr_t_type, &
                                 ndims_tensor, &
                                 dbcsr_t_distribution_type, &
                                 dbcsr_t_distribution, &
                                 dbcsr_t_distribution_destroy, &
                                 dbcsr_t_distribution_new_expert, &
                                 dbcsr_t_clear, &
                                 dbcsr_t_finalize, &
                                 dbcsr_t_get_num_blocks, &
                                 dbcsr_t_blk_offsets, &
                                 dbcsr_t_blk_sizes, &
                                 ndims_matrix_row, &
                                 ndims_matrix_column, &
                                 dbcsr_t_filter, &
                                 dbcsr_t_copy_contraction_storage
   USE dbcsr_api, ONLY: dbcsr_type_real_4, dbcsr_type_complex_8, dbcsr_type_complex_4, dbcsr_type_real_8
# 48 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
   USE dbcsr_kinds, ONLY: real_4, real_8, dp
# 49 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

#include "base/dbcsr_base_uses.f90"
   IMPLICIT NONE
   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tensor_split'
   PUBLIC :: &
      dbcsr_t_make_compatible_blocks, &
      dbcsr_t_split_blocks, &
      dbcsr_t_split_blocks_generic, &
      dbcsr_t_split_copyback, &
      dbcsr_t_crop

CONTAINS

   SUBROUTINE dbcsr_t_split_blocks_generic(tensor_in, tensor_out, blk_size_1, blk_size_2, blk_size_3, blk_size_4, nodata)
# 64 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      !! Split tensor blocks into smaller blocks
      TYPE(dbcsr_t_type), INTENT(INOUT)               :: tensor_in
         !! Input tensor
      TYPE(dbcsr_t_type), INTENT(OUT)                 :: tensor_out
         !! Output tensor (splitted blocks)
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL     :: blk_size_1, blk_size_2, blk_size_3, blk_size_4
# 70 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         !! block sizes for each of the tensor dimensions
      LOGICAL, INTENT(IN), OPTIONAL                   :: nodata
         !! don't copy data from tensor_in to tensor_out

      TYPE(dbcsr_t_distribution_type)                 :: dist_old, dist_split
      TYPE(dbcsr_t_iterator_type)                     :: iter
      INTEGER, DIMENSION(:), ALLOCATABLE              :: nd_dist_split_1, nd_dist_split_2, nd_dist_split_3, nd_dist_split_4
# 77 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      INTEGER, DIMENSION(:), ALLOCATABLE              :: nd_blk_size_split_1, nd_blk_size_split_2, nd_blk_size_split_3,&
# 77 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          & nd_blk_size_split_4
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      INTEGER, DIMENSION(:), ALLOCATABLE              :: index_split_offset_1, index_split_offset_2, index_split_offset_3,&
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          & index_split_offset_4
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      INTEGER, DIMENSION(:), ALLOCATABLE              :: inblock_offset_1, inblock_offset_2, inblock_offset_3, inblock_offset_4
# 80 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      INTEGER, DIMENSION(:), ALLOCATABLE              :: blk_nsplit_1, blk_nsplit_2, blk_nsplit_3, blk_nsplit_4
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      INTEGER                                         :: split_blk_1, split_blk_2, split_blk_3, split_blk_4
# 82 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      INTEGER :: idim, i, isplit_sum, blk, nsplit, handle, splitsum, bcount
      INTEGER, DIMENSION(:, :), ALLOCATABLE           :: blks_to_allocate
      INTEGER, DIMENSION(:), ALLOCATABLE :: dist_d, blk_size_d, blk_size_split_d, dist_split_d
      INTEGER, DIMENSION(ndims_matrix_row(tensor_in)) :: map1_2d
      INTEGER, DIMENSION(ndims_matrix_column(tensor_in)) :: map2_2d
      INTEGER, DIMENSION(ndims_tensor(tensor_in)) :: blk_index, blk_size, blk_offset, &
                                                     blk_shape
      INTEGER, DIMENSION(4) :: bi_split, inblock_offset
# 90 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      LOGICAL :: found

# 93 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 94 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            REAL(kind=real_8), DIMENSION(:,:), ALLOCATABLE :: block_r_dp_2d
# 95 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 94 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            REAL(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE :: block_r_dp_3d
# 95 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 94 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            REAL(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE :: block_r_dp_4d
# 95 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 96 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 93 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 94 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            REAL(kind=real_4), DIMENSION(:,:), ALLOCATABLE :: block_r_sp_2d
# 95 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 94 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            REAL(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE :: block_r_sp_3d
# 95 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 94 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            REAL(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE :: block_r_sp_4d
# 95 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 96 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 93 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 94 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            COMPLEX(kind=real_8), DIMENSION(:,:), ALLOCATABLE :: block_c_dp_2d
# 95 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 94 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            COMPLEX(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE :: block_c_dp_3d
# 95 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 94 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            COMPLEX(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE :: block_c_dp_4d
# 95 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 96 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 93 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 94 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            COMPLEX(kind=real_4), DIMENSION(:,:), ALLOCATABLE :: block_c_sp_2d
# 95 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 94 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            COMPLEX(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE :: block_c_sp_3d
# 95 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 94 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            COMPLEX(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE :: block_c_sp_4d
# 95 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 96 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 97 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_split_blocks_generic'

      CALL timeset(routineN, handle)

      dist_old = dbcsr_t_distribution(tensor_in)

      DO idim = 1, ndims_tensor(tensor_in)
         CALL get_ith_array(dist_old%nd_dist, idim, dist_d)
         CALL get_ith_array(tensor_in%blk_sizes, idim, blk_size_d)

# 108 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (idim == 1) THEN
# 109 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ! split block index offset for each normal block index
               ALLOCATE (index_split_offset_1 (SIZE(dist_d)))
# 111 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ! how many split blocks for each normal block index
               ALLOCATE (blk_nsplit_1 (SIZE(dist_d)))
# 113 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ! data offset of split blocks w.r.t. corresponding normal block
               ALLOCATE (inblock_offset_1 (SIZE(blk_size_1)))
# 115 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL allocate_any(blk_size_split_d, source=blk_size_1)
# 116 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            END IF
# 108 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (idim == 2) THEN
# 109 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ! split block index offset for each normal block index
               ALLOCATE (index_split_offset_2 (SIZE(dist_d)))
# 111 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ! how many split blocks for each normal block index
               ALLOCATE (blk_nsplit_2 (SIZE(dist_d)))
# 113 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ! data offset of split blocks w.r.t. corresponding normal block
               ALLOCATE (inblock_offset_2 (SIZE(blk_size_2)))
# 115 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL allocate_any(blk_size_split_d, source=blk_size_2)
# 116 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            END IF
# 108 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (idim == 3) THEN
# 109 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ! split block index offset for each normal block index
               ALLOCATE (index_split_offset_3 (SIZE(dist_d)))
# 111 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ! how many split blocks for each normal block index
               ALLOCATE (blk_nsplit_3 (SIZE(dist_d)))
# 113 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ! data offset of split blocks w.r.t. corresponding normal block
               ALLOCATE (inblock_offset_3 (SIZE(blk_size_3)))
# 115 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL allocate_any(blk_size_split_d, source=blk_size_3)
# 116 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            END IF
# 108 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (idim == 4) THEN
# 109 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ! split block index offset for each normal block index
               ALLOCATE (index_split_offset_4 (SIZE(dist_d)))
# 111 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ! how many split blocks for each normal block index
               ALLOCATE (blk_nsplit_4 (SIZE(dist_d)))
# 113 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ! data offset of split blocks w.r.t. corresponding normal block
               ALLOCATE (inblock_offset_4 (SIZE(blk_size_4)))
# 115 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL allocate_any(blk_size_split_d, source=blk_size_4)
# 116 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            END IF
# 118 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

         ! distribution vector for split blocks
         ALLOCATE (dist_split_d(SIZE(blk_size_split_d)))

         isplit_sum = 0 ! counting splits
         DO i = 1, SIZE(blk_size_d)
            nsplit = 0 ! number of splits for current normal block
            splitsum = 0 ! summing split block sizes for current normal block
            DO WHILE (splitsum < blk_size_d(i))
               nsplit = nsplit + 1
               isplit_sum = isplit_sum + 1
# 130 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  IF (idim == 1) inblock_offset_1 (isplit_sum) = splitsum
# 131 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 130 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  IF (idim == 2) inblock_offset_2 (isplit_sum) = splitsum
# 131 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 130 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  IF (idim == 3) inblock_offset_3 (isplit_sum) = splitsum
# 131 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 130 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  IF (idim == 4) inblock_offset_4 (isplit_sum) = splitsum
# 131 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 132 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               dist_split_d(isplit_sum) = dist_d(i)
               splitsum = splitsum + blk_size_split_d(isplit_sum)
            END DO
            DBCSR_ASSERT(splitsum == blk_size_d(i))
# 137 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               IF (idim == 1) THEN
# 138 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  blk_nsplit_1 (i) = nsplit
# 139 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  index_split_offset_1 (i) = isplit_sum - nsplit
# 140 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               END IF
# 137 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               IF (idim == 2) THEN
# 138 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  blk_nsplit_2 (i) = nsplit
# 139 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  index_split_offset_2 (i) = isplit_sum - nsplit
# 140 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               END IF
# 137 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               IF (idim == 3) THEN
# 138 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  blk_nsplit_3 (i) = nsplit
# 139 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  index_split_offset_3 (i) = isplit_sum - nsplit
# 140 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               END IF
# 137 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               IF (idim == 4) THEN
# 138 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  blk_nsplit_4 (i) = nsplit
# 139 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  index_split_offset_4 (i) = isplit_sum - nsplit
# 140 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               END IF
# 142 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         END DO

# 145 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (idim == 1) THEN
# 146 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL allocate_any(nd_dist_split_1, source=dist_split_d)
# 147 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL allocate_any(nd_blk_size_split_1, source=blk_size_split_d)
# 148 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            END IF
# 145 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (idim == 2) THEN
# 146 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL allocate_any(nd_dist_split_2, source=dist_split_d)
# 147 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL allocate_any(nd_blk_size_split_2, source=blk_size_split_d)
# 148 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            END IF
# 145 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (idim == 3) THEN
# 146 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL allocate_any(nd_dist_split_3, source=dist_split_d)
# 147 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL allocate_any(nd_blk_size_split_3, source=blk_size_split_d)
# 148 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            END IF
# 145 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (idim == 4) THEN
# 146 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL allocate_any(nd_dist_split_4, source=dist_split_d)
# 147 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL allocate_any(nd_blk_size_split_4, source=blk_size_split_d)
# 148 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            END IF
# 150 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         DEALLOCATE (dist_split_d)
         DEALLOCATE (blk_size_split_d)

      END DO

      CALL dbcsr_t_get_mapping_info(tensor_in%nd_index_blk, map1_2d=map1_2d, map2_2d=map2_2d)

# 158 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (ndims_tensor(tensor_in) == 2) THEN
# 159 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            CALL dbcsr_t_distribution_new_expert(dist_split, tensor_in%pgrid, map1_2d, map2_2d, &
                                                 nd_dist_split_1, nd_dist_split_2)
# 161 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            CALL dbcsr_t_create(tensor_out, tensor_in%name, dist_split, map1_2d, map2_2d, &
                                dbcsr_t_get_data_type(tensor_in), nd_blk_size_split_1, nd_blk_size_split_2)
# 163 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         END IF
# 158 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (ndims_tensor(tensor_in) == 3) THEN
# 159 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            CALL dbcsr_t_distribution_new_expert(dist_split, tensor_in%pgrid, map1_2d, map2_2d, &
                                                 nd_dist_split_1, nd_dist_split_2, nd_dist_split_3)
# 161 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            CALL dbcsr_t_create(tensor_out, tensor_in%name, dist_split, map1_2d, map2_2d, &
                                dbcsr_t_get_data_type(tensor_in), nd_blk_size_split_1, nd_blk_size_split_2, nd_blk_size_split_3)
# 163 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         END IF
# 158 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (ndims_tensor(tensor_in) == 4) THEN
# 159 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            CALL dbcsr_t_distribution_new_expert(dist_split, tensor_in%pgrid, map1_2d, map2_2d, &
                                                 nd_dist_split_1, nd_dist_split_2, nd_dist_split_3, nd_dist_split_4)
# 161 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            CALL dbcsr_t_create(tensor_out, tensor_in%name, dist_split, map1_2d, map2_2d, &
                                dbcsr_t_get_data_type(tensor_in), nd_blk_size_split_1, nd_blk_size_split_2, nd_blk_size_split_3,&
# 162 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    & nd_blk_size_split_4)
# 163 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         END IF
# 165 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

      CALL dbcsr_t_distribution_destroy(dist_split)

      IF (PRESENT(nodata)) THEN
         IF (nodata) THEN
            CALL timestop(handle)
            RETURN
         END IF
      END IF

      CALL dbcsr_t_iterator_start(iter, tensor_in)

      bcount = 0
      DO WHILE (dbcsr_t_iterator_blocks_left(iter))
         CALL dbcsr_t_iterator_next_block(iter, blk_index, blk, blk_size=blk_size)
# 181 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 2) THEN
# 182 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 183 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
# 184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 183 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
# 184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 185 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                     bcount = bcount + 1
# 187 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        END DO
# 187 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        END DO
# 189 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  END IF
# 181 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 3) THEN
# 182 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 183 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
# 184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 183 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
# 184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 183 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  DO split_blk_3 = 1, blk_nsplit_3 (blk_index(3))
# 184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 185 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                     bcount = bcount + 1
# 187 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        END DO
# 187 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        END DO
# 187 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        END DO
# 189 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  END IF
# 181 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 4) THEN
# 182 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 183 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
# 184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 183 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
# 184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 183 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  DO split_blk_3 = 1, blk_nsplit_3 (blk_index(3))
# 184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 183 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  DO split_blk_4 = 1, blk_nsplit_4 (blk_index(4))
# 184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 185 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                     bcount = bcount + 1
# 187 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        END DO
# 187 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        END DO
# 187 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        END DO
# 187 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        END DO
# 189 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  END IF
# 191 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            END DO
            CALL dbcsr_t_iterator_stop(iter)

            ALLOCATE (blks_to_allocate(bcount, ndims_tensor(tensor_in)))

            CALL dbcsr_t_iterator_start(iter, tensor_in)

            bcount = 0
            DO WHILE (dbcsr_t_iterator_blocks_left(iter))
               CALL dbcsr_t_iterator_next_block(iter, blk_index, blk, blk_size=blk_size, blk_offset=blk_offset)

# 203 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  IF (ndims_tensor(tensor_in) == 2) THEN
# 204 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 205 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           bi_split(1) = index_split_offset_1 (blk_index(1)) + split_blk_1
# 207 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 205 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           bi_split(2) = index_split_offset_2 (blk_index(2)) + split_blk_2
# 207 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 208 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           bcount = bcount + 1
                           blks_to_allocate(bcount, :) = bi_split(1:ndims_tensor(tensor_in))
# 211 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END DO
# 211 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END DO
# 213 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        END IF
# 203 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  IF (ndims_tensor(tensor_in) == 3) THEN
# 204 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 205 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           bi_split(1) = index_split_offset_1 (blk_index(1)) + split_blk_1
# 207 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 205 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           bi_split(2) = index_split_offset_2 (blk_index(2)) + split_blk_2
# 207 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 205 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        DO split_blk_3 = 1, blk_nsplit_3 (blk_index(3))
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           bi_split(3) = index_split_offset_3 (blk_index(3)) + split_blk_3
# 207 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 208 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           bcount = bcount + 1
                           blks_to_allocate(bcount, :) = bi_split(1:ndims_tensor(tensor_in))
# 211 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END DO
# 211 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END DO
# 211 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END DO
# 213 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        END IF
# 203 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  IF (ndims_tensor(tensor_in) == 4) THEN
# 204 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 205 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           bi_split(1) = index_split_offset_1 (blk_index(1)) + split_blk_1
# 207 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 205 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           bi_split(2) = index_split_offset_2 (blk_index(2)) + split_blk_2
# 207 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 205 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        DO split_blk_3 = 1, blk_nsplit_3 (blk_index(3))
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           bi_split(3) = index_split_offset_3 (blk_index(3)) + split_blk_3
# 207 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 205 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        DO split_blk_4 = 1, blk_nsplit_4 (blk_index(4))
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           bi_split(4) = index_split_offset_4 (blk_index(4)) + split_blk_4
# 207 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 208 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           bcount = bcount + 1
                           blks_to_allocate(bcount, :) = bi_split(1:ndims_tensor(tensor_in))
# 211 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END DO
# 211 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END DO
# 211 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END DO
# 211 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END DO
# 213 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        END IF
# 215 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  END DO

                  CALL dbcsr_t_iterator_stop(iter)

                  CALL dbcsr_t_reserve_blocks(tensor_out, blks_to_allocate)

                  CALL dbcsr_t_iterator_start(iter, tensor_in)

                  DO WHILE (dbcsr_t_iterator_blocks_left(iter))
                     CALL dbcsr_t_iterator_next_block(iter, blk_index, blk, blk_size=blk_size, blk_offset=blk_offset)
# 226 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_8) THEN
# 227 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 228 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (ndims_tensor(tensor_in) == 2) THEN
# 229 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_dp_2d, found)
# 230 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 DBCSR_ASSERT(found)
                              END IF
# 228 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (ndims_tensor(tensor_in) == 3) THEN
# 229 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_dp_3d, found)
# 230 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 DBCSR_ASSERT(found)
                              END IF
# 228 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (ndims_tensor(tensor_in) == 4) THEN
# 229 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_dp_4d, found)
# 230 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 DBCSR_ASSERT(found)
                              END IF
# 233 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        END IF
# 226 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_4) THEN
# 227 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 228 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (ndims_tensor(tensor_in) == 2) THEN
# 229 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_sp_2d, found)
# 230 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 DBCSR_ASSERT(found)
                              END IF
# 228 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (ndims_tensor(tensor_in) == 3) THEN
# 229 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_sp_3d, found)
# 230 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 DBCSR_ASSERT(found)
                              END IF
# 228 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (ndims_tensor(tensor_in) == 4) THEN
# 229 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_sp_4d, found)
# 230 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 DBCSR_ASSERT(found)
                              END IF
# 233 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        END IF
# 226 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_8) THEN
# 227 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 228 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (ndims_tensor(tensor_in) == 2) THEN
# 229 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_dp_2d, found)
# 230 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 DBCSR_ASSERT(found)
                              END IF
# 228 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (ndims_tensor(tensor_in) == 3) THEN
# 229 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_dp_3d, found)
# 230 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 DBCSR_ASSERT(found)
                              END IF
# 228 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (ndims_tensor(tensor_in) == 4) THEN
# 229 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_dp_4d, found)
# 230 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 DBCSR_ASSERT(found)
                              END IF
# 233 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        END IF
# 226 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_4) THEN
# 227 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 228 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (ndims_tensor(tensor_in) == 2) THEN
# 229 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_sp_2d, found)
# 230 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 DBCSR_ASSERT(found)
                              END IF
# 228 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (ndims_tensor(tensor_in) == 3) THEN
# 229 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_sp_3d, found)
# 230 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 DBCSR_ASSERT(found)
                              END IF
# 228 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (ndims_tensor(tensor_in) == 4) THEN
# 229 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_sp_4d, found)
# 230 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 DBCSR_ASSERT(found)
                              END IF
# 233 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        END IF
# 235 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 236 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        IF (ndims_tensor(tensor_in) == 2) THEN
# 237 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 238 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
# 239 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 ! split block index
                                 bi_split(1) = index_split_offset_1 (blk_index(1)) + split_blk_1
# 241 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 blk_shape(1) = blk_size_1 (bi_split(1))
# 242 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 238 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
# 239 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 ! split block index
                                 bi_split(2) = index_split_offset_2 (blk_index(2)) + split_blk_2
# 241 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 blk_shape(2) = blk_size_2 (bi_split(2))
# 242 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 243 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 245 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_8) THEN
# 247 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 251 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL dbcsr_t_put_block(tensor_out, bi_split(1:2), &
# 252 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                              blk_shape, &
                                                              block_r_dp_2d( &
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2)))
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    END IF
# 245 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_4) THEN
# 247 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 251 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL dbcsr_t_put_block(tensor_out, bi_split(1:2), &
# 252 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                              blk_shape, &
                                                              block_r_sp_2d( &
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2)))
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    END IF
# 245 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_8) THEN
# 247 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 251 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL dbcsr_t_put_block(tensor_out, bi_split(1:2), &
# 252 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                              blk_shape, &
                                                              block_c_dp_2d( &
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2)))
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    END IF
# 245 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_4) THEN
# 247 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 251 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL dbcsr_t_put_block(tensor_out, bi_split(1:2), &
# 252 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                              blk_shape, &
                                                              block_c_sp_2d( &
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2)))
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    END IF
# 258 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 260 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END DO
# 260 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END DO
# 262 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 264 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_8) THEN
# 265 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       DEALLOCATE (block_r_dp_2d)
# 266 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 264 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_4) THEN
# 265 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       DEALLOCATE (block_r_sp_2d)
# 266 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 264 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_8) THEN
# 265 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       DEALLOCATE (block_c_dp_2d)
# 266 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 264 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_4) THEN
# 265 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       DEALLOCATE (block_c_sp_2d)
# 266 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 268 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 236 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        IF (ndims_tensor(tensor_in) == 3) THEN
# 237 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 238 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
# 239 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 ! split block index
                                 bi_split(1) = index_split_offset_1 (blk_index(1)) + split_blk_1
# 241 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 blk_shape(1) = blk_size_1 (bi_split(1))
# 242 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 238 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
# 239 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 ! split block index
                                 bi_split(2) = index_split_offset_2 (blk_index(2)) + split_blk_2
# 241 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 blk_shape(2) = blk_size_2 (bi_split(2))
# 242 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 238 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              DO split_blk_3 = 1, blk_nsplit_3 (blk_index(3))
# 239 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 ! split block index
                                 bi_split(3) = index_split_offset_3 (blk_index(3)) + split_blk_3
# 241 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 blk_shape(3) = blk_size_3 (bi_split(3))
# 242 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 243 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 245 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_8) THEN
# 247 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(3) = inblock_offset_3 (bi_split(3))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 251 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL dbcsr_t_put_block(tensor_out, bi_split(1:3), &
# 252 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                              blk_shape, &
                                                              block_r_dp_3d( &
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & blk_shape(3)))
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    END IF
# 245 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_4) THEN
# 247 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(3) = inblock_offset_3 (bi_split(3))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 251 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL dbcsr_t_put_block(tensor_out, bi_split(1:3), &
# 252 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                              blk_shape, &
                                                              block_r_sp_3d( &
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & blk_shape(3)))
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    END IF
# 245 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_8) THEN
# 247 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(3) = inblock_offset_3 (bi_split(3))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 251 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL dbcsr_t_put_block(tensor_out, bi_split(1:3), &
# 252 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                              blk_shape, &
                                                              block_c_dp_3d( &
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & blk_shape(3)))
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    END IF
# 245 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_4) THEN
# 247 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(3) = inblock_offset_3 (bi_split(3))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 251 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL dbcsr_t_put_block(tensor_out, bi_split(1:3), &
# 252 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                              blk_shape, &
                                                              block_c_sp_3d( &
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & blk_shape(3)))
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    END IF
# 258 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 260 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END DO
# 260 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END DO
# 260 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END DO
# 262 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 264 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_8) THEN
# 265 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       DEALLOCATE (block_r_dp_3d)
# 266 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 264 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_4) THEN
# 265 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       DEALLOCATE (block_r_sp_3d)
# 266 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 264 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_8) THEN
# 265 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       DEALLOCATE (block_c_dp_3d)
# 266 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 264 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_4) THEN
# 265 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       DEALLOCATE (block_c_sp_3d)
# 266 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 268 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 236 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        IF (ndims_tensor(tensor_in) == 4) THEN
# 237 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 238 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
# 239 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 ! split block index
                                 bi_split(1) = index_split_offset_1 (blk_index(1)) + split_blk_1
# 241 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 blk_shape(1) = blk_size_1 (bi_split(1))
# 242 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 238 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
# 239 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 ! split block index
                                 bi_split(2) = index_split_offset_2 (blk_index(2)) + split_blk_2
# 241 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 blk_shape(2) = blk_size_2 (bi_split(2))
# 242 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 238 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              DO split_blk_3 = 1, blk_nsplit_3 (blk_index(3))
# 239 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 ! split block index
                                 bi_split(3) = index_split_offset_3 (blk_index(3)) + split_blk_3
# 241 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 blk_shape(3) = blk_size_3 (bi_split(3))
# 242 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 238 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              DO split_blk_4 = 1, blk_nsplit_4 (blk_index(4))
# 239 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 ! split block index
                                 bi_split(4) = index_split_offset_4 (blk_index(4)) + split_blk_4
# 241 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 blk_shape(4) = blk_size_4 (bi_split(4))
# 242 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 243 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 245 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_8) THEN
# 247 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(3) = inblock_offset_3 (bi_split(3))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(4) = inblock_offset_4 (bi_split(4))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 251 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL dbcsr_t_put_block(tensor_out, bi_split(1:4), &
# 252 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                              blk_shape, &
                                                              block_r_dp_4d( &
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & blk_shape(3), inblock_offset(4) + 1:inblock_offset(4) + blk_shape(4)))
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    END IF
# 245 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_4) THEN
# 247 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(3) = inblock_offset_3 (bi_split(3))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(4) = inblock_offset_4 (bi_split(4))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 251 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL dbcsr_t_put_block(tensor_out, bi_split(1:4), &
# 252 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                              blk_shape, &
                                                              block_r_sp_4d( &
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & blk_shape(3), inblock_offset(4) + 1:inblock_offset(4) + blk_shape(4)))
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    END IF
# 245 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_8) THEN
# 247 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(3) = inblock_offset_3 (bi_split(3))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(4) = inblock_offset_4 (bi_split(4))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 251 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL dbcsr_t_put_block(tensor_out, bi_split(1:4), &
# 252 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                              blk_shape, &
                                                              block_c_dp_4d( &
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & blk_shape(3), inblock_offset(4) + 1:inblock_offset(4) + blk_shape(4)))
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    END IF
# 245 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_4) THEN
# 247 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(3) = inblock_offset_3 (bi_split(3))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          inblock_offset(4) = inblock_offset_4 (bi_split(4))
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 251 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL dbcsr_t_put_block(tensor_out, bi_split(1:4), &
# 252 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                              blk_shape, &
                                                              block_c_sp_4d( &
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) +&
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & blk_shape(3), inblock_offset(4) + 1:inblock_offset(4) + blk_shape(4)))
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                    END IF
# 258 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 260 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END DO
# 260 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END DO
# 260 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END DO
# 260 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END DO
# 262 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 264 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_8) THEN
# 265 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       DEALLOCATE (block_r_dp_4d)
# 266 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 264 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_4) THEN
# 265 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       DEALLOCATE (block_r_sp_4d)
# 266 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 264 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_8) THEN
# 265 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       DEALLOCATE (block_c_dp_4d)
# 266 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 264 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_4) THEN
# 265 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       DEALLOCATE (block_c_sp_4d)
# 266 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 268 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 270 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        END DO
                        CALL dbcsr_t_iterator_stop(iter)

                        CALL dbcsr_t_finalize(tensor_out)

                        ! remove blocks that are exactly 0, these can occur if a cropping operation was performed before splitting
                        CALL dbcsr_t_filter(tensor_out, TINY(0.0_dp))

                        CALL timestop(handle)

                     END SUBROUTINE

                     SUBROUTINE dbcsr_t_split_blocks(tensor_in, tensor_out, block_sizes, nodata)
      !! Split tensor blocks into smaller blocks of maximum size PRODUCT(block_sizes).

                        TYPE(dbcsr_t_type), INTENT(INOUT)               :: tensor_in
         !! Input tensor
                        TYPE(dbcsr_t_type), INTENT(OUT)                 :: tensor_out
         !! Output tensor (split blocks)
                        INTEGER, DIMENSION(ndims_tensor(tensor_in)), &
                           INTENT(IN)                                   :: block_sizes
         !! block sizes for each of the tensor dimensions
                        LOGICAL, INTENT(IN), OPTIONAL                   :: nodata
         !! don't copy data from tensor_in to tensor_out

                        INTEGER, DIMENSION(:), ALLOCATABLE              :: nd_blk_size_split_1, nd_blk_size_split_2,&
# 295 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & nd_blk_size_split_3, nd_blk_size_split_4
# 296 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        INTEGER :: idim, i, isplit_sum, blk_remainder, nsplit, isplit
                        INTEGER, DIMENSION(:), ALLOCATABLE :: blk_size_d, blk_size_split_d

                        DO idim = 1, ndims_tensor(tensor_in)
                           CALL get_ith_array(tensor_in%blk_sizes, idim, blk_size_d)

                           isplit_sum = 0
                           DO i = 1, SIZE(blk_size_d)
                              nsplit = (blk_size_d(i) + block_sizes(idim) - 1)/block_sizes(idim)
                              isplit_sum = isplit_sum + nsplit
                           END DO

                           ALLOCATE (blk_size_split_d(isplit_sum))

                           isplit_sum = 0
                           DO i = 1, SIZE(blk_size_d)
                              nsplit = (blk_size_d(i) + block_sizes(idim) - 1)/block_sizes(idim)
                              blk_remainder = blk_size_d(i)
                              DO isplit = 1, nsplit
                                 isplit_sum = isplit_sum + 1
                                 blk_size_split_d(isplit_sum) = MIN(block_sizes(idim), blk_remainder)
                                 blk_remainder = blk_remainder - block_sizes(idim)
                              END DO

                           END DO

# 323 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (idim == 1) THEN
# 324 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL allocate_any(nd_blk_size_split_1, source=blk_size_split_d)
# 325 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 323 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (idim == 2) THEN
# 324 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL allocate_any(nd_blk_size_split_2, source=blk_size_split_d)
# 325 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 323 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (idim == 3) THEN
# 324 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL allocate_any(nd_blk_size_split_3, source=blk_size_split_d)
# 325 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 323 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (idim == 4) THEN
# 324 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL allocate_any(nd_blk_size_split_4, source=blk_size_split_d)
# 325 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 327 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           DEALLOCATE (blk_size_split_d)
                        END DO

# 331 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           IF (ndims_tensor(tensor_in) == 2) CALL dbcsr_t_split_blocks_generic(tensor_in, tensor_out, &
# 332 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                                      nd_blk_size_split_1, nd_blk_size_split_2, &
# 333 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                                                      nodata=nodata)
# 331 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           IF (ndims_tensor(tensor_in) == 3) CALL dbcsr_t_split_blocks_generic(tensor_in, tensor_out, &
# 332 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                                      nd_blk_size_split_1, nd_blk_size_split_2,&
# 332 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                                          & nd_blk_size_split_3, &
# 333 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                                                      nodata=nodata)
# 331 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           IF (ndims_tensor(tensor_in) == 4) CALL dbcsr_t_split_blocks_generic(tensor_in, tensor_out, &
# 332 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                                      nd_blk_size_split_1, nd_blk_size_split_2,&
# 332 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                                          & nd_blk_size_split_3, nd_blk_size_split_&
# 332 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                                          &4, &
# 333 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                                                      nodata=nodata)
# 335 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                     END SUBROUTINE

                     SUBROUTINE dbcsr_t_split_copyback(tensor_split_in, tensor_out, summation)
      !! Copy tensor with split blocks to tensor with original block sizes.

                        TYPE(dbcsr_t_type), INTENT(INOUT)               :: tensor_split_in
         !! tensor with smaller blocks
                        TYPE(dbcsr_t_type), INTENT(INOUT)               :: tensor_out
         !! original tensor
                        LOGICAL, INTENT(IN), OPTIONAL                   :: summation
                        INTEGER, DIMENSION(:), ALLOCATABLE              :: first_split_d, last_split_d
                        INTEGER, DIMENSION(:), ALLOCATABLE              :: blk_size_split_d, blk_size_d
                        INTEGER, DIMENSION(:), ALLOCATABLE              :: last_split_1, last_split_2, last_split_3, last_split_4, &
# 349 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                           first_split_1, first_split_2, first_split_3,&
# 349 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                               & first_split_4, &
# 350 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                           split_1, split_2, split_3, split_4
# 351 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                     INTEGER, DIMENSION(:), ALLOCATABLE              :: inblock_offset_1, inblock_offset_2, inblock_offset_3,&
# 351 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                         & inblock_offset_4, blk_size_split_1, blk_size_split_2, blk_size_split_3, blk_size_split_4
# 352 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        INTEGER, DIMENSION(:, :), ALLOCATABLE            :: blks_to_allocate
                        INTEGER                                         :: idim, iblk, blk, bcount
                        INTEGER                                         :: iblk_1, iblk_2, iblk_3, iblk_4, isplit_sum, splitsum,&
# 354 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & nblk
# 355 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        TYPE(dbcsr_t_iterator_type)                     :: iter
                        INTEGER, DIMENSION(ndims_tensor(tensor_out)) :: blk_index, blk_size, blk_offset, blk_shape, blk_index_n
                        LOGICAL                                         :: found

                        INTEGER, DIMENSION(4)                  :: inblock_offset
# 360 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        INTEGER                                            :: handle
                        CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_split_copyback'
# 363 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 364 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              REAL(kind=real_8), DIMENSION(:,:), ALLOCATABLE :: block_r_dp_2d
# 365 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              REAL(kind=real_8), DIMENSION(:,:), ALLOCATABLE :: block_split_r_dp_2d
# 366 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 364 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              REAL(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE :: block_r_dp_3d
# 365 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              REAL(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE :: block_split_r_dp_3d
# 366 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 364 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              REAL(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE :: block_r_dp_4d
# 365 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              REAL(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE :: block_split_r_dp_4d
# 366 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 367 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 363 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 364 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              REAL(kind=real_4), DIMENSION(:,:), ALLOCATABLE :: block_r_sp_2d
# 365 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              REAL(kind=real_4), DIMENSION(:,:), ALLOCATABLE :: block_split_r_sp_2d
# 366 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 364 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              REAL(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE :: block_r_sp_3d
# 365 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              REAL(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE :: block_split_r_sp_3d
# 366 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 364 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              REAL(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE :: block_r_sp_4d
# 365 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              REAL(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE :: block_split_r_sp_4d
# 366 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 367 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 363 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 364 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              COMPLEX(kind=real_8), DIMENSION(:,:), ALLOCATABLE :: block_c_dp_2d
# 365 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              COMPLEX(kind=real_8), DIMENSION(:,:), ALLOCATABLE :: block_split_c_dp_2d
# 366 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 364 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              COMPLEX(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE :: block_c_dp_3d
# 365 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              COMPLEX(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE :: block_split_c_dp_3d
# 366 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 364 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              COMPLEX(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE :: block_c_dp_4d
# 365 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              COMPLEX(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE :: block_split_c_dp_4d
# 366 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 367 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 363 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 364 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              COMPLEX(kind=real_4), DIMENSION(:,:), ALLOCATABLE :: block_c_sp_2d
# 365 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              COMPLEX(kind=real_4), DIMENSION(:,:), ALLOCATABLE :: block_split_c_sp_2d
# 366 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 364 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              COMPLEX(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE :: block_c_sp_3d
# 365 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              COMPLEX(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE :: block_split_c_sp_3d
# 366 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 364 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              COMPLEX(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE :: block_c_sp_4d
# 365 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              COMPLEX(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE :: block_split_c_sp_4d
# 366 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 367 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 368 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                        CALL timeset(routineN, handle)
                        DBCSR_ASSERT(tensor_out%valid)
                        IF (PRESENT(summation)) THEN
                           IF (.NOT. summation) CALL dbcsr_t_clear(tensor_out)
                        ELSE
                           CALL dbcsr_t_clear(tensor_out)
                        END IF

                        DO idim = 1, ndims_tensor(tensor_split_in)
                           CALL get_ith_array(tensor_split_in%blk_sizes, idim, blk_size_split_d)
                           CALL get_ith_array(tensor_out%blk_sizes, idim, blk_size_d)

# 382 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (idim == 1) THEN
# 383 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 ! data offset of split blocks w.r.t. corresponding normal block
                                 ALLOCATE (inblock_offset_1 (SIZE(blk_size_split_d)))
# 385 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 ! normal block index for each split block
                                 ALLOCATE (split_1 (SIZE(blk_size_split_d)))
# 387 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 382 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (idim == 2) THEN
# 383 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 ! data offset of split blocks w.r.t. corresponding normal block
                                 ALLOCATE (inblock_offset_2 (SIZE(blk_size_split_d)))
# 385 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 ! normal block index for each split block
                                 ALLOCATE (split_2 (SIZE(blk_size_split_d)))
# 387 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 382 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (idim == 3) THEN
# 383 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 ! data offset of split blocks w.r.t. corresponding normal block
                                 ALLOCATE (inblock_offset_3 (SIZE(blk_size_split_d)))
# 385 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 ! normal block index for each split block
                                 ALLOCATE (split_3 (SIZE(blk_size_split_d)))
# 387 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 382 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (idim == 4) THEN
# 383 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 ! data offset of split blocks w.r.t. corresponding normal block
                                 ALLOCATE (inblock_offset_4 (SIZE(blk_size_split_d)))
# 385 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 ! normal block index for each split block
                                 ALLOCATE (split_4 (SIZE(blk_size_split_d)))
# 387 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 389 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                           ALLOCATE (last_split_d(SIZE(blk_size_d)))
                           ALLOCATE (first_split_d(SIZE(blk_size_d)))
                           first_split_d(1) = 1
                           isplit_sum = 0
                           DO iblk = 1, SIZE(blk_size_d)
                              splitsum = 0
                              IF (iblk .GT. 1) first_split_d(iblk) = last_split_d(iblk - 1) + 1
                              DO WHILE (splitsum < blk_size_d(iblk))
                                 isplit_sum = isplit_sum + 1
# 400 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (idim == 1) THEN
# 401 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       inblock_offset_1 (isplit_sum) = splitsum
# 402 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       split_1 (isplit_sum) = iblk
# 403 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 400 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (idim == 2) THEN
# 401 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       inblock_offset_2 (isplit_sum) = splitsum
# 402 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       split_2 (isplit_sum) = iblk
# 403 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 400 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (idim == 3) THEN
# 401 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       inblock_offset_3 (isplit_sum) = splitsum
# 402 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       split_3 (isplit_sum) = iblk
# 403 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 400 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (idim == 4) THEN
# 401 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       inblock_offset_4 (isplit_sum) = splitsum
# 402 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       split_4 (isplit_sum) = iblk
# 403 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 405 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 splitsum = splitsum + blk_size_split_d(isplit_sum)
                              END DO
                              DBCSR_ASSERT(splitsum == blk_size_d(iblk))
                              last_split_d(iblk) = isplit_sum
                           END DO
# 411 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (idim == 1) THEN
# 412 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL allocate_any(first_split_1, source=first_split_d)
# 413 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL allocate_any(last_split_1, source=last_split_d)
# 414 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL allocate_any(blk_size_split_1, source=blk_size_split_d)
# 415 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 411 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (idim == 2) THEN
# 412 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL allocate_any(first_split_2, source=first_split_d)
# 413 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL allocate_any(last_split_2, source=last_split_d)
# 414 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL allocate_any(blk_size_split_2, source=blk_size_split_d)
# 415 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 411 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (idim == 3) THEN
# 412 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL allocate_any(first_split_3, source=first_split_d)
# 413 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL allocate_any(last_split_3, source=last_split_d)
# 414 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL allocate_any(blk_size_split_3, source=blk_size_split_d)
# 415 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 411 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (idim == 4) THEN
# 412 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL allocate_any(first_split_4, source=first_split_d)
# 413 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL allocate_any(last_split_4, source=last_split_d)
# 414 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 CALL allocate_any(blk_size_split_4, source=blk_size_split_d)
# 415 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 417 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           DEALLOCATE (first_split_d, last_split_d)
                           DEALLOCATE (blk_size_split_d, blk_size_d)
                        END DO

                        nblk = dbcsr_t_get_num_blocks(tensor_split_in)
                        ALLOCATE (blks_to_allocate(nblk, ndims_tensor(tensor_split_in)))
                        CALL dbcsr_t_iterator_start(iter, tensor_split_in)
                        bcount = 0
                        DO WHILE (dbcsr_t_iterator_blocks_left(iter))
                           CALL dbcsr_t_iterator_next_block(iter, blk_index, blk, blk_size=blk_size)
# 428 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (ndims_tensor(tensor_out) == 2) THEN
# 429 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 430 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    blk_index_n(1) = split_1 (blk_index(1))
# 431 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 430 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    blk_index_n(2) = split_2 (blk_index(2))
# 431 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 432 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 428 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (ndims_tensor(tensor_out) == 3) THEN
# 429 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 430 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    blk_index_n(1) = split_1 (blk_index(1))
# 431 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 430 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    blk_index_n(2) = split_2 (blk_index(2))
# 431 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 430 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    blk_index_n(3) = split_3 (blk_index(3))
# 431 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 432 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 428 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (ndims_tensor(tensor_out) == 4) THEN
# 429 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 430 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    blk_index_n(1) = split_1 (blk_index(1))
# 431 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 430 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    blk_index_n(2) = split_2 (blk_index(2))
# 431 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 430 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    blk_index_n(3) = split_3 (blk_index(3))
# 431 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 430 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    blk_index_n(4) = split_4 (blk_index(4))
# 431 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 432 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END IF
# 434 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           blks_to_allocate(bcount + 1, :) = blk_index_n
                           bcount = bcount + 1
                        END DO
                        CALL dbcsr_t_iterator_stop(iter)
                        CALL dbcsr_t_reserve_blocks(tensor_out, blks_to_allocate)

                        CALL dbcsr_t_iterator_start(iter, tensor_out)
                        DO WHILE (dbcsr_t_iterator_blocks_left(iter))
                           CALL dbcsr_t_iterator_next_block(iter, blk_index, blk, blk_size=blk_size, blk_offset=blk_offset)
# 444 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (dbcsr_t_get_data_type(tensor_out) == dbcsr_type_real_8) THEN
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (ndims_tensor(tensor_out) == 2) THEN
# 448 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(block_r_dp_2d, blk_size)
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       block_r_dp_2d = 0.0_real_8
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(1) = inblock_offset_1 (iblk_1)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(2) = inblock_offset_2 (iblk_2)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                        CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2], &
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                    block_split_r_dp_2d, found)
# 459 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             IF (found) THEN
                                                blk_shape(1:2) = SHAPE(block_split_r_dp_2d)
# 461 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                block_r_dp_2d( &
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2)) = &
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                   block_split_r_dp_2d
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             END IF

# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_r_dp_2d, summation=summation)
# 470 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_r_dp_2d)
# 471 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (ndims_tensor(tensor_out) == 3) THEN
# 448 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(block_r_dp_3d, blk_size)
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       block_r_dp_3d = 0.0_real_8
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_3 = first_split_3 (blk_index(3)), last_split_3 (blk_index(3))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(1) = inblock_offset_1 (iblk_1)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(2) = inblock_offset_2 (iblk_2)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(3) = inblock_offset_3 (iblk_3)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                        CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2, iblk_3], &
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                    block_split_r_dp_3d, found)
# 459 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             IF (found) THEN
                                                blk_shape(1:3) = SHAPE(block_split_r_dp_3d)
# 461 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                block_r_dp_3d( &
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) + blk_shape(3)) = &
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                   block_split_r_dp_3d
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             END IF

# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_r_dp_3d, summation=summation)
# 470 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_r_dp_3d)
# 471 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (ndims_tensor(tensor_out) == 4) THEN
# 448 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(block_r_dp_4d, blk_size)
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       block_r_dp_4d = 0.0_real_8
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_3 = first_split_3 (blk_index(3)), last_split_3 (blk_index(3))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_4 = first_split_4 (blk_index(4)), last_split_4 (blk_index(4))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(1) = inblock_offset_1 (iblk_1)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(2) = inblock_offset_2 (iblk_2)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(3) = inblock_offset_3 (iblk_3)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(4) = inblock_offset_4 (iblk_4)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                        CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2, iblk_3, iblk_4], &
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                    block_split_r_dp_4d, found)
# 459 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             IF (found) THEN
                                                blk_shape(1:4) = SHAPE(block_split_r_dp_4d)
# 461 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                block_r_dp_4d( &
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) + blk_shape(3), inblock_offset(4) +&
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & 1:inblock_offset(4) + blk_shape(4)) = &
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                   block_split_r_dp_4d
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             END IF

# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_r_dp_4d, summation=summation)
# 470 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_r_dp_4d)
# 471 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 473 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 444 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (dbcsr_t_get_data_type(tensor_out) == dbcsr_type_real_4) THEN
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (ndims_tensor(tensor_out) == 2) THEN
# 448 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(block_r_sp_2d, blk_size)
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       block_r_sp_2d = 0.0_real_4
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(1) = inblock_offset_1 (iblk_1)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(2) = inblock_offset_2 (iblk_2)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                        CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2], &
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                    block_split_r_sp_2d, found)
# 459 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             IF (found) THEN
                                                blk_shape(1:2) = SHAPE(block_split_r_sp_2d)
# 461 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                block_r_sp_2d( &
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2)) = &
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                   block_split_r_sp_2d
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             END IF

# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_r_sp_2d, summation=summation)
# 470 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_r_sp_2d)
# 471 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (ndims_tensor(tensor_out) == 3) THEN
# 448 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(block_r_sp_3d, blk_size)
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       block_r_sp_3d = 0.0_real_4
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_3 = first_split_3 (blk_index(3)), last_split_3 (blk_index(3))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(1) = inblock_offset_1 (iblk_1)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(2) = inblock_offset_2 (iblk_2)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(3) = inblock_offset_3 (iblk_3)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                        CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2, iblk_3], &
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                    block_split_r_sp_3d, found)
# 459 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             IF (found) THEN
                                                blk_shape(1:3) = SHAPE(block_split_r_sp_3d)
# 461 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                block_r_sp_3d( &
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) + blk_shape(3)) = &
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                   block_split_r_sp_3d
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             END IF

# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_r_sp_3d, summation=summation)
# 470 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_r_sp_3d)
# 471 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (ndims_tensor(tensor_out) == 4) THEN
# 448 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(block_r_sp_4d, blk_size)
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       block_r_sp_4d = 0.0_real_4
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_3 = first_split_3 (blk_index(3)), last_split_3 (blk_index(3))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_4 = first_split_4 (blk_index(4)), last_split_4 (blk_index(4))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(1) = inblock_offset_1 (iblk_1)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(2) = inblock_offset_2 (iblk_2)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(3) = inblock_offset_3 (iblk_3)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(4) = inblock_offset_4 (iblk_4)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                        CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2, iblk_3, iblk_4], &
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                    block_split_r_sp_4d, found)
# 459 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             IF (found) THEN
                                                blk_shape(1:4) = SHAPE(block_split_r_sp_4d)
# 461 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                block_r_sp_4d( &
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) + blk_shape(3), inblock_offset(4) +&
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & 1:inblock_offset(4) + blk_shape(4)) = &
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                   block_split_r_sp_4d
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             END IF

# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_r_sp_4d, summation=summation)
# 470 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_r_sp_4d)
# 471 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 473 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 444 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (dbcsr_t_get_data_type(tensor_out) == dbcsr_type_complex_8) THEN
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (ndims_tensor(tensor_out) == 2) THEN
# 448 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(block_c_dp_2d, blk_size)
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       block_c_dp_2d = 0.0_real_8
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(1) = inblock_offset_1 (iblk_1)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(2) = inblock_offset_2 (iblk_2)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                        CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2], &
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                    block_split_c_dp_2d, found)
# 459 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             IF (found) THEN
                                                blk_shape(1:2) = SHAPE(block_split_c_dp_2d)
# 461 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                block_c_dp_2d( &
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2)) = &
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                   block_split_c_dp_2d
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             END IF

# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_c_dp_2d, summation=summation)
# 470 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_c_dp_2d)
# 471 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (ndims_tensor(tensor_out) == 3) THEN
# 448 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(block_c_dp_3d, blk_size)
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       block_c_dp_3d = 0.0_real_8
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_3 = first_split_3 (blk_index(3)), last_split_3 (blk_index(3))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(1) = inblock_offset_1 (iblk_1)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(2) = inblock_offset_2 (iblk_2)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(3) = inblock_offset_3 (iblk_3)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                        CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2, iblk_3], &
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                    block_split_c_dp_3d, found)
# 459 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             IF (found) THEN
                                                blk_shape(1:3) = SHAPE(block_split_c_dp_3d)
# 461 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                block_c_dp_3d( &
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) + blk_shape(3)) = &
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                   block_split_c_dp_3d
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             END IF

# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_c_dp_3d, summation=summation)
# 470 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_c_dp_3d)
# 471 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (ndims_tensor(tensor_out) == 4) THEN
# 448 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(block_c_dp_4d, blk_size)
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       block_c_dp_4d = 0.0_real_8
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_3 = first_split_3 (blk_index(3)), last_split_3 (blk_index(3))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_4 = first_split_4 (blk_index(4)), last_split_4 (blk_index(4))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(1) = inblock_offset_1 (iblk_1)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(2) = inblock_offset_2 (iblk_2)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(3) = inblock_offset_3 (iblk_3)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(4) = inblock_offset_4 (iblk_4)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                        CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2, iblk_3, iblk_4], &
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                    block_split_c_dp_4d, found)
# 459 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             IF (found) THEN
                                                blk_shape(1:4) = SHAPE(block_split_c_dp_4d)
# 461 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                block_c_dp_4d( &
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) + blk_shape(3), inblock_offset(4) +&
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & 1:inblock_offset(4) + blk_shape(4)) = &
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                   block_split_c_dp_4d
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             END IF

# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_c_dp_4d, summation=summation)
# 470 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_c_dp_4d)
# 471 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 473 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 444 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              IF (dbcsr_t_get_data_type(tensor_out) == dbcsr_type_complex_4) THEN
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (ndims_tensor(tensor_out) == 2) THEN
# 448 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(block_c_sp_2d, blk_size)
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       block_c_sp_2d = 0.0_real_4
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(1) = inblock_offset_1 (iblk_1)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(2) = inblock_offset_2 (iblk_2)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                        CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2], &
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                    block_split_c_sp_2d, found)
# 459 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             IF (found) THEN
                                                blk_shape(1:2) = SHAPE(block_split_c_sp_2d)
# 461 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                block_c_sp_2d( &
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2)) = &
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                   block_split_c_sp_2d
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             END IF

# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_c_sp_2d, summation=summation)
# 470 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_c_sp_2d)
# 471 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (ndims_tensor(tensor_out) == 3) THEN
# 448 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(block_c_sp_3d, blk_size)
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       block_c_sp_3d = 0.0_real_4
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_3 = first_split_3 (blk_index(3)), last_split_3 (blk_index(3))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(1) = inblock_offset_1 (iblk_1)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(2) = inblock_offset_2 (iblk_2)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(3) = inblock_offset_3 (iblk_3)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                        CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2, iblk_3], &
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                    block_split_c_sp_3d, found)
# 459 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             IF (found) THEN
                                                blk_shape(1:3) = SHAPE(block_split_c_sp_3d)
# 461 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                block_c_sp_3d( &
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) + blk_shape(3)) = &
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                   block_split_c_sp_3d
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             END IF

# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_c_sp_3d, summation=summation)
# 470 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_c_sp_3d)
# 471 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (ndims_tensor(tensor_out) == 4) THEN
# 448 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(block_c_sp_4d, blk_size)
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       block_c_sp_4d = 0.0_real_4
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_3 = first_split_3 (blk_index(3)), last_split_3 (blk_index(3))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            DO iblk_4 = first_split_4 (blk_index(4)), last_split_4 (blk_index(4))
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(1) = inblock_offset_1 (iblk_1)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(2) = inblock_offset_2 (iblk_2)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(3) = inblock_offset_3 (iblk_3)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                inblock_offset(4) = inblock_offset_4 (iblk_4)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                        CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2, iblk_3, iblk_4], &
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                    block_split_c_sp_4d, found)
# 459 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             IF (found) THEN
                                                blk_shape(1:4) = SHAPE(block_split_c_sp_4d)
# 461 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                block_c_sp_4d( &
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) + blk_shape(3), inblock_offset(4) +&
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & 1:inblock_offset(4) + blk_shape(4)) = &
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                   block_split_c_sp_4d
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             END IF

# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 467 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                END DO
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                           CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_c_sp_4d, summation=summation)
# 470 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_c_sp_4d)
# 471 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 473 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 475 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END DO
                              CALL dbcsr_t_iterator_stop(iter)

                              CALL timestop(handle)

                           END SUBROUTINE

       SUBROUTINE dbcsr_t_make_compatible_blocks(tensor1, tensor2, tensor1_split, tensor2_split, order, nodata1, nodata2, move_data)
      !! split two tensors with same total sizes but different block sizes such that they have equal
      !! block sizes
      !! \move_data memory optimization: transfer data s.t. tensor1 and tensor2 may be empty on return

                              TYPE(dbcsr_t_type), INTENT(INOUT) :: tensor1, tensor2
         !! tensor 1 in
         !! tensor 2 in
                              TYPE(dbcsr_t_type), INTENT(OUT)   :: tensor1_split, tensor2_split
         !! tensor 1 with split blocks
         !! tensor 2 with split blocks
                              INTEGER, DIMENSION(ndims_tensor(tensor1)), &
                                 INTENT(IN), OPTIONAL                        :: order

                              LOGICAL, INTENT(IN), OPTIONAL     :: nodata1, nodata2, move_data
         !! don't copy data of tensor 1
         !! don't copy data of tensor 2
                          INTEGER, DIMENSION(:), ALLOCATABLE  :: blk_size_split_1_1, blk_size_split_1_2, blk_size_split_1_3,&
# 499 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              & blk_size_split_1_4, blk_size_split_2_1, blk_size_split_2_2, blk_size_split_2_3,&
# 499 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              & blk_size_split_2_4, &
# 500 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                     blk_size_d_1, blk_size_d_2, blk_size_d_split
                              INTEGER :: size_sum_1, size_sum_2, size_sum, bind_1, bind_2, isplit, bs, idim, i
                              LOGICAL :: move_prv, nodata1_prv, nodata2_prv
                              INTEGER, DIMENSION(ndims_tensor(tensor1)) :: order_prv

                              IF (PRESENT(move_data)) THEN
                                 move_prv = move_data
                              ELSE
                                 move_prv = .FALSE.
                              END IF

                              IF (PRESENT(nodata1)) THEN
                                 nodata1_prv = nodata1
                              ELSE
                                 nodata1_prv = .FALSE.
                              END IF
                              IF (PRESENT(nodata2)) THEN
                                 nodata2_prv = nodata2
                              ELSE
                                 nodata2_prv = .FALSE.
                              END IF

                              IF (PRESENT(order)) THEN
                                 order_prv(:) = dbcsr_t_inverse_order(order)
                              ELSE
                                 order_prv(:) = (/(i, i=1, ndims_tensor(tensor1))/)
                              END IF

                              DO idim = 1, ndims_tensor(tensor2)
                                 CALL get_ith_array(tensor1%blk_sizes, order_prv(idim), blk_size_d_1)
                                 CALL get_ith_array(tensor2%blk_sizes, idim, blk_size_d_2)
                                 ALLOCATE (blk_size_d_split(SIZE(blk_size_d_1) + SIZE(blk_size_d_2)))
                                 size_sum_1 = 0
                                 size_sum_2 = 0
                                 size_sum = 0
                                 bind_1 = 0
                                 bind_2 = 0
                                 isplit = 0

                                 DO WHILE (bind_1 < SIZE(blk_size_d_1) .AND. bind_2 < SIZE(blk_size_d_2))
                                    IF (blk_size_d_1(bind_1 + 1) < blk_size_d_2(bind_2 + 1)) THEN
                                       bind_1 = bind_1 + 1
                                       bs = blk_size_d_1(bind_1)
                                       blk_size_d_2(bind_2 + 1) = blk_size_d_2(bind_2 + 1) - bs
                                       size_sum = size_sum + bs
                                       isplit = isplit + 1
                                       blk_size_d_split(isplit) = bs
                                    ELSEIF (blk_size_d_1(bind_1 + 1) > blk_size_d_2(bind_2 + 1)) THEN
                                       bind_2 = bind_2 + 1
                                       bs = blk_size_d_2(bind_2)
                                       blk_size_d_1(bind_1 + 1) = blk_size_d_1(bind_1 + 1) - bs
                                       size_sum = size_sum + bs
                                       isplit = isplit + 1
                                       blk_size_d_split(isplit) = bs
                                    ELSE
                                       bind_1 = bind_1 + 1
                                       bind_2 = bind_2 + 1
                                       bs = blk_size_d_1(bind_1)
                                       size_sum = size_sum + bs
                                       isplit = isplit + 1
                                       blk_size_d_split(isplit) = bs
                                    END IF
                                 END DO

                                 IF (bind_1 < SIZE(blk_size_d_1)) THEN
                                    bind_1 = bind_1 + 1
                                    bs = blk_size_d_1(bind_1)
                                    size_sum = size_sum + bs
                                    isplit = isplit + 1
                                    blk_size_d_split(isplit) = bs
                                 END IF

                                 IF (bind_2 < SIZE(blk_size_d_2)) THEN
                                    bind_2 = bind_2 + 1
                                    bs = blk_size_d_2(bind_2)
                                    size_sum = size_sum + bs
                                    isplit = isplit + 1
                                    blk_size_d_split(isplit) = bs
                                 END IF

# 581 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (order_prv(idim) == 1) THEN
# 582 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(blk_size_split_1_1, source=blk_size_d_split(:isplit))
# 583 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 581 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (order_prv(idim) == 2) THEN
# 582 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(blk_size_split_1_2, source=blk_size_d_split(:isplit))
# 583 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 581 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (order_prv(idim) == 3) THEN
# 582 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(blk_size_split_1_3, source=blk_size_d_split(:isplit))
# 583 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 581 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (order_prv(idim) == 4) THEN
# 582 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(blk_size_split_1_4, source=blk_size_d_split(:isplit))
# 583 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 585 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 587 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (idim == 1) THEN
# 588 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(blk_size_split_2_1, source=blk_size_d_split(:isplit))
# 589 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 587 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (idim == 2) THEN
# 588 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(blk_size_split_2_2, source=blk_size_d_split(:isplit))
# 589 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 587 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (idim == 3) THEN
# 588 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(blk_size_split_2_3, source=blk_size_d_split(:isplit))
# 589 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 587 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (idim == 4) THEN
# 588 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                       CALL allocate_any(blk_size_split_2_4, source=blk_size_d_split(:isplit))
# 589 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 591 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                 DEALLOCATE (blk_size_d_split, blk_size_d_1, blk_size_d_2)
                              END DO

# 596 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 IF (ndims_tensor(tensor1) == 2) THEN
# 597 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL dbcsr_t_split_blocks_generic(tensor1, tensor1_split, blk_size_split_1_1, blk_size_split_1_2, nodata=nodata1)
# 598 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (move_prv .AND. .NOT. nodata1_prv) CALL dbcsr_t_clear(tensor1)
               CALL dbcsr_t_split_blocks_generic(tensor2, tensor2_split, blk_size_split_2_1, blk_size_split_2_2, nodata=nodata2)
# 600 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (move_prv .AND. .NOT. nodata2_prv) CALL dbcsr_t_clear(tensor2)
                                 END IF
# 596 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 IF (ndims_tensor(tensor1) == 3) THEN
# 597 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL dbcsr_t_split_blocks_generic(tensor1, tensor1_split, blk_size_split_1_1, blk_size_split_1_2,&
# 597 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & blk_size_split_1_3, nodata=nodata1)
# 598 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (move_prv .AND. .NOT. nodata1_prv) CALL dbcsr_t_clear(tensor1)
               CALL dbcsr_t_split_blocks_generic(tensor2, tensor2_split, blk_size_split_2_1, blk_size_split_2_2,&
# 599 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & blk_size_split_2_3, nodata=nodata2)
# 600 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (move_prv .AND. .NOT. nodata2_prv) CALL dbcsr_t_clear(tensor2)
                                 END IF
# 596 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 IF (ndims_tensor(tensor1) == 4) THEN
# 597 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL dbcsr_t_split_blocks_generic(tensor1, tensor1_split, blk_size_split_1_1, blk_size_split_1_2,&
# 597 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & blk_size_split_1_3, blk_size_split_1_4, nodata=nodata1)
# 598 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (move_prv .AND. .NOT. nodata1_prv) CALL dbcsr_t_clear(tensor1)
               CALL dbcsr_t_split_blocks_generic(tensor2, tensor2_split, blk_size_split_2_1, blk_size_split_2_2,&
# 599 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & blk_size_split_2_3, blk_size_split_2_4, nodata=nodata2)
# 600 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (move_prv .AND. .NOT. nodata2_prv) CALL dbcsr_t_clear(tensor2)
                                 END IF
# 603 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                           END SUBROUTINE

                           SUBROUTINE dbcsr_t_crop(tensor_in, tensor_out, bounds, move_data)
                              TYPE(dbcsr_t_type), INTENT(INOUT) :: tensor_in
                              TYPE(dbcsr_t_type), INTENT(OUT) :: tensor_out
                              INTEGER, DIMENSION(2, ndims_tensor(tensor_in)), INTENT(IN) :: bounds
                              LOGICAL, INTENT(IN), OPTIONAL :: move_data
                              INTEGER, DIMENSION(2, ndims_tensor(tensor_in)) :: blk_bounds
                              TYPE(dbcsr_t_iterator_type)                     :: iter
                              INTEGER, DIMENSION(ndims_tensor(tensor_in)) :: blk_index, blk_size, blk_offset
                              LOGICAL :: found, move_data_prv
                              INTEGER :: idim, blk, iblk, iblk_all, nblk
                              INTEGER, DIMENSION(:, :), ALLOCATABLE :: blk_ind, blk_ind_tmp
# 618 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 619 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          REAL(kind=real_8), DIMENSION(:,:), ALLOCATABLE :: block_r_dp_2d, block_put_r_dp_2d
# 620 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 619 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          REAL(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE :: block_r_dp_3d, block_put_r_dp_3d
# 620 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 619 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          REAL(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE :: block_r_dp_4d, block_put_r_dp_4d
# 620 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 621 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 618 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 619 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          REAL(kind=real_4), DIMENSION(:,:), ALLOCATABLE :: block_r_sp_2d, block_put_r_sp_2d
# 620 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 619 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          REAL(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE :: block_r_sp_3d, block_put_r_sp_3d
# 620 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 619 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          REAL(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE :: block_r_sp_4d, block_put_r_sp_4d
# 620 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 621 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 618 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 619 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          COMPLEX(kind=real_8), DIMENSION(:,:), ALLOCATABLE :: block_c_dp_2d, block_put_c_dp_2d
# 620 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 619 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          COMPLEX(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE :: block_c_dp_3d, block_put_c_dp_3d
# 620 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 619 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          COMPLEX(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE :: block_c_dp_4d, block_put_c_dp_4d
# 620 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 621 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 618 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 619 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          COMPLEX(kind=real_4), DIMENSION(:,:), ALLOCATABLE :: block_c_sp_2d, block_put_c_sp_2d
# 620 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 619 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          COMPLEX(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE :: block_c_sp_3d, block_put_c_sp_3d
# 620 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 619 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          COMPLEX(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE :: block_c_sp_4d, block_put_c_sp_4d
# 620 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 621 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 622 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                              IF (PRESENT(move_data)) THEN
                                 move_data_prv = move_data
                              ELSE
                                 move_data_prv = .FALSE.
                              END IF

                              CALL dbcsr_t_create(tensor_in, tensor_out)

                              ! reserve blocks inside bounds
                              ALLOCATE (blk_ind(dbcsr_t_get_num_blocks(tensor_in), ndims_tensor(tensor_in)))
                              CALL dbcsr_t_reserved_block_indices(tensor_in, blk_ind)
                              nblk = dbcsr_t_get_num_blocks(tensor_in)
                              ALLOCATE (blk_ind_tmp(dbcsr_t_get_num_blocks(tensor_in), ndims_tensor(tensor_in)))
                              blk_ind_tmp(:, :) = 0
                              iblk = 0
                              blk_loop: DO iblk_all = 1, nblk
                                 CALL dbcsr_t_blk_offsets(tensor_in, blk_ind(iblk_all, :), blk_offset)
                                 CALL dbcsr_t_blk_sizes(tensor_in, blk_ind(iblk_all, :), blk_size)
                                 DO idim = 1, ndims_tensor(tensor_in)
                                    IF (bounds(1, idim) > blk_offset(idim) - 1 + blk_size(idim)) CYCLE blk_loop
                                    IF (bounds(2, idim) < blk_offset(idim)) CYCLE blk_loop
                                 END DO
                                 iblk = iblk + 1
                                 blk_ind_tmp(iblk, :) = blk_ind(iblk_all, :)
                              END DO blk_loop

                              DEALLOCATE (blk_ind)
                              ALLOCATE (blk_ind(iblk, ndims_tensor(tensor_in)))
                              blk_ind(:, :) = blk_ind_tmp(:iblk, :)

                              CALL dbcsr_t_reserve_blocks(tensor_out, blk_ind)

                              ! copy blocks
                              CALL dbcsr_t_iterator_start(iter, tensor_out)
                              iter_loop: DO WHILE (dbcsr_t_iterator_blocks_left(iter))
                                 CALL dbcsr_t_iterator_next_block(iter, blk_index, blk, blk_size=blk_size, blk_offset=blk_offset)

                                 DO idim = 1, ndims_tensor(tensor_in)
                                    blk_bounds(1, idim) = MAX(bounds(1, idim) - blk_offset(idim) + 1, 1)
                                    blk_bounds(2, idim) = MIN(bounds(2, idim) - blk_offset(idim) + 1, blk_size(idim))
                                 END DO

# 666 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_8) THEN
# 667 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 668 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          IF (ndims_tensor(tensor_in) == 2) THEN
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_dp_2d, found)
# 670 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                             CALL allocate_any(block_put_r_dp_2d, blk_size)
# 672 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             block_put_r_dp_2d = 0.0_real_8
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               block_put_r_dp_2d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2)) = &
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  block_r_dp_2d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2))
# 675 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                            CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_r_dp_2d)
# 676 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_r_dp_2d)
# 677 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_put_r_dp_2d)
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 668 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          IF (ndims_tensor(tensor_in) == 3) THEN
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_dp_3d, found)
# 670 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                             CALL allocate_any(block_put_r_dp_3d, blk_size)
# 672 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             block_put_r_dp_3d = 0.0_real_8
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               block_put_r_dp_3d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & 3):blk_bounds(2,3)) = &
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  block_r_dp_3d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                      & 3):blk_bounds(2,3))
# 675 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                            CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_r_dp_3d)
# 676 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_r_dp_3d)
# 677 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_put_r_dp_3d)
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 668 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          IF (ndims_tensor(tensor_in) == 4) THEN
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_dp_4d, found)
# 670 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                             CALL allocate_any(block_put_r_dp_4d, blk_size)
# 672 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             block_put_r_dp_4d = 0.0_real_8
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               block_put_r_dp_4d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & 3):blk_bounds(2,3), blk_bounds(1, 4):blk_bounds(2,4)) = &
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  block_r_dp_4d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                      & 3):blk_bounds(2,3), blk_bounds(1, 4):blk_bounds(2,4))
# 675 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                            CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_r_dp_4d)
# 676 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_r_dp_4d)
# 677 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_put_r_dp_4d)
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 680 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 666 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_4) THEN
# 667 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 668 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          IF (ndims_tensor(tensor_in) == 2) THEN
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_sp_2d, found)
# 670 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                             CALL allocate_any(block_put_r_sp_2d, blk_size)
# 672 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             block_put_r_sp_2d = 0.0_real_4
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               block_put_r_sp_2d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2)) = &
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  block_r_sp_2d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2))
# 675 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                            CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_r_sp_2d)
# 676 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_r_sp_2d)
# 677 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_put_r_sp_2d)
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 668 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          IF (ndims_tensor(tensor_in) == 3) THEN
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_sp_3d, found)
# 670 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                             CALL allocate_any(block_put_r_sp_3d, blk_size)
# 672 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             block_put_r_sp_3d = 0.0_real_4
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               block_put_r_sp_3d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & 3):blk_bounds(2,3)) = &
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  block_r_sp_3d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                      & 3):blk_bounds(2,3))
# 675 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                            CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_r_sp_3d)
# 676 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_r_sp_3d)
# 677 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_put_r_sp_3d)
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 668 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          IF (ndims_tensor(tensor_in) == 4) THEN
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_sp_4d, found)
# 670 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                             CALL allocate_any(block_put_r_sp_4d, blk_size)
# 672 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             block_put_r_sp_4d = 0.0_real_4
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               block_put_r_sp_4d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & 3):blk_bounds(2,3), blk_bounds(1, 4):blk_bounds(2,4)) = &
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  block_r_sp_4d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                      & 3):blk_bounds(2,3), blk_bounds(1, 4):blk_bounds(2,4))
# 675 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                            CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_r_sp_4d)
# 676 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_r_sp_4d)
# 677 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_put_r_sp_4d)
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 680 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 666 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_8) THEN
# 667 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 668 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          IF (ndims_tensor(tensor_in) == 2) THEN
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_dp_2d, found)
# 670 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                             CALL allocate_any(block_put_c_dp_2d, blk_size)
# 672 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             block_put_c_dp_2d = 0.0_real_8
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               block_put_c_dp_2d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2)) = &
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  block_c_dp_2d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2))
# 675 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                            CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_c_dp_2d)
# 676 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_c_dp_2d)
# 677 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_put_c_dp_2d)
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 668 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          IF (ndims_tensor(tensor_in) == 3) THEN
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_dp_3d, found)
# 670 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                             CALL allocate_any(block_put_c_dp_3d, blk_size)
# 672 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             block_put_c_dp_3d = 0.0_real_8
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               block_put_c_dp_3d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & 3):blk_bounds(2,3)) = &
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  block_c_dp_3d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                      & 3):blk_bounds(2,3))
# 675 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                            CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_c_dp_3d)
# 676 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_c_dp_3d)
# 677 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_put_c_dp_3d)
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 668 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          IF (ndims_tensor(tensor_in) == 4) THEN
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_dp_4d, found)
# 670 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                             CALL allocate_any(block_put_c_dp_4d, blk_size)
# 672 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             block_put_c_dp_4d = 0.0_real_8
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               block_put_c_dp_4d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & 3):blk_bounds(2,3), blk_bounds(1, 4):blk_bounds(2,4)) = &
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  block_c_dp_4d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                      & 3):blk_bounds(2,3), blk_bounds(1, 4):blk_bounds(2,4))
# 675 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                            CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_c_dp_4d)
# 676 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_c_dp_4d)
# 677 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_put_c_dp_4d)
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 680 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 666 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_4) THEN
# 667 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 668 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          IF (ndims_tensor(tensor_in) == 2) THEN
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_sp_2d, found)
# 670 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                             CALL allocate_any(block_put_c_sp_2d, blk_size)
# 672 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             block_put_c_sp_2d = 0.0_real_4
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               block_put_c_sp_2d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2)) = &
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  block_c_sp_2d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2))
# 675 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                            CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_c_sp_2d)
# 676 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_c_sp_2d)
# 677 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_put_c_sp_2d)
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 668 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          IF (ndims_tensor(tensor_in) == 3) THEN
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_sp_3d, found)
# 670 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                             CALL allocate_any(block_put_c_sp_3d, blk_size)
# 672 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             block_put_c_sp_3d = 0.0_real_4
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               block_put_c_sp_3d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & 3):blk_bounds(2,3)) = &
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  block_c_sp_3d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                      & 3):blk_bounds(2,3))
# 675 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                            CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_c_sp_3d)
# 676 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_c_sp_3d)
# 677 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_put_c_sp_3d)
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 668 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          IF (ndims_tensor(tensor_in) == 4) THEN
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_sp_4d, found)
# 670 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                                             CALL allocate_any(block_put_c_sp_4d, blk_size)
# 672 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             block_put_c_sp_4d = 0.0_real_4
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               block_put_c_sp_4d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 673 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & 3):blk_bounds(2,3), blk_bounds(1, 4):blk_bounds(2,4)) = &
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  block_c_sp_4d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 674 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                      & 3):blk_bounds(2,3), blk_bounds(1, 4):blk_bounds(2,4))
# 675 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                            CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_c_sp_4d)
# 676 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_c_sp_4d)
# 677 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             DEALLOCATE (block_put_c_sp_4d)
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                          END IF
# 680 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                    END IF
# 682 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                              END DO iter_loop
                              CALL dbcsr_t_iterator_stop(iter)
                              CALL dbcsr_t_finalize(tensor_out)

                              IF (move_data_prv) CALL dbcsr_t_clear(tensor_in)

                              ! transfer data for batched contraction
                              CALL dbcsr_t_copy_contraction_storage(tensor_in, tensor_out)

                           END SUBROUTINE

                        END MODULE
