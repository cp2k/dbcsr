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

# 66 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp"
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
   USE dbcsr_tensor_index, ONLY: dbcsr_t_get_mapping_info
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
                                 dbcsr_t_blk_sizes
   USE dbcsr_api, ONLY: dbcsr_type_complex_8, dbcsr_type_real_4, dbcsr_type_real_8, dbcsr_type_complex_4
   USE dbcsr_kinds, ONLY: real_4, real_8

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
      !! Split tensor blocks into smaller blocks
      TYPE(dbcsr_t_type), INTENT(INOUT)               :: tensor_in
         !! Input tensor
      TYPE(dbcsr_t_type), INTENT(OUT)                 :: tensor_out
         !! Output tensor (splitted blocks)
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL     :: blk_size_1, blk_size_2, blk_size_3, blk_size_4
         !! block sizes for each of the tensor dimensions
      LOGICAL, INTENT(IN), OPTIONAL                   :: nodata
         !! don't copy data from tensor_in to tensor_out

      TYPE(dbcsr_t_distribution_type)                 :: dist_old, dist_split
      TYPE(dbcsr_t_iterator_type)                     :: iter
      INTEGER, DIMENSION(:), ALLOCATABLE              :: nd_dist_split_1, nd_dist_split_2, nd_dist_split_3, nd_dist_split_4
      INTEGER, DIMENSION(:), ALLOCATABLE              :: nd_blk_size_split_1, nd_blk_size_split_2, nd_blk_size_split_3,&
# 72 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          & nd_blk_size_split_4
      INTEGER, DIMENSION(:), ALLOCATABLE              :: index_split_offset_1, index_split_offset_2, index_split_offset_3,&
# 73 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          & index_split_offset_4
      INTEGER, DIMENSION(:), ALLOCATABLE              :: inblock_offset_1, inblock_offset_2, inblock_offset_3, inblock_offset_4
      INTEGER, DIMENSION(:), ALLOCATABLE              :: blk_nsplit_1, blk_nsplit_2, blk_nsplit_3, blk_nsplit_4
      INTEGER                                         :: split_blk_1, split_blk_2, split_blk_3, split_blk_4
      INTEGER :: idim, i, isplit_sum, blk, nsplit, handle, splitsum, bcount
      INTEGER, DIMENSION(:, :), ALLOCATABLE           :: blks_to_allocate
      INTEGER, DIMENSION(:), ALLOCATABLE :: dist_d, blk_size_d, blk_size_split_d, dist_split_d, &
                                            map1_2d, map2_2d
      INTEGER, DIMENSION(ndims_tensor(tensor_in)) :: blk_index, blk_size, blk_offset, &
                                                     blk_shape
      INTEGER, DIMENSION(4) :: bi_split, inblock_offset
      LOGICAL :: found

# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_8), DIMENSION(:,:), ALLOCATABLE :: block_r_dp_2d
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE :: block_r_dp_3d
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE :: block_r_dp_4d
# 90 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_4), DIMENSION(:,:), ALLOCATABLE :: block_r_sp_2d
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE :: block_r_sp_3d
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE :: block_r_sp_4d
# 90 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_8), DIMENSION(:,:), ALLOCATABLE :: block_c_dp_2d
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE :: block_c_dp_3d
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE :: block_c_dp_4d
# 90 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_4), DIMENSION(:,:), ALLOCATABLE :: block_c_sp_2d
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE :: block_c_sp_3d
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE :: block_c_sp_4d
# 90 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 91 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_split_blocks_generic', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)

      dist_old = dbcsr_t_distribution(tensor_in)

      DO idim = 1, ndims_tensor(tensor_in)
         CALL get_ith_array(dist_old%nd_dist, idim, dist_d)
         CALL get_ith_array(tensor_in%blk_sizes, idim, blk_size_d)

# 103 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 1) THEN
            ! split block index offset for each normal block index
            ALLOCATE (index_split_offset_1 (SIZE(dist_d)))
            ! how many split blocks for each normal block index
            ALLOCATE (blk_nsplit_1 (SIZE(dist_d)))
            ! data offset of split blocks w.r.t. corresponding normal block
            ALLOCATE (inblock_offset_1 (SIZE(blk_size_1)))
            CALL allocate_any(blk_size_split_d, source=blk_size_1)
         ENDIF
# 103 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 2) THEN
            ! split block index offset for each normal block index
            ALLOCATE (index_split_offset_2 (SIZE(dist_d)))
            ! how many split blocks for each normal block index
            ALLOCATE (blk_nsplit_2 (SIZE(dist_d)))
            ! data offset of split blocks w.r.t. corresponding normal block
            ALLOCATE (inblock_offset_2 (SIZE(blk_size_2)))
            CALL allocate_any(blk_size_split_d, source=blk_size_2)
         ENDIF
# 103 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 3) THEN
            ! split block index offset for each normal block index
            ALLOCATE (index_split_offset_3 (SIZE(dist_d)))
            ! how many split blocks for each normal block index
            ALLOCATE (blk_nsplit_3 (SIZE(dist_d)))
            ! data offset of split blocks w.r.t. corresponding normal block
            ALLOCATE (inblock_offset_3 (SIZE(blk_size_3)))
            CALL allocate_any(blk_size_split_d, source=blk_size_3)
         ENDIF
# 103 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 4) THEN
            ! split block index offset for each normal block index
            ALLOCATE (index_split_offset_4 (SIZE(dist_d)))
            ! how many split blocks for each normal block index
            ALLOCATE (blk_nsplit_4 (SIZE(dist_d)))
            ! data offset of split blocks w.r.t. corresponding normal block
            ALLOCATE (inblock_offset_4 (SIZE(blk_size_4)))
            CALL allocate_any(blk_size_split_d, source=blk_size_4)
         ENDIF
# 113 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

         ! distribution vector for split blocks
         ALLOCATE (dist_split_d(SIZE(blk_size_split_d)))

         isplit_sum = 0 ! counting splits
         DO i = 1, SIZE(blk_size_d)
            nsplit = 0 ! number of splits for current normal block
            splitsum = 0 ! summing split block sizes for current normal block
            DO WHILE (splitsum < blk_size_d(i))
               nsplit = nsplit + 1
               isplit_sum = isplit_sum + 1
# 125 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               IF (idim == 1) inblock_offset_1 (isplit_sum) = splitsum
# 125 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               IF (idim == 2) inblock_offset_2 (isplit_sum) = splitsum
# 125 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               IF (idim == 3) inblock_offset_3 (isplit_sum) = splitsum
# 125 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               IF (idim == 4) inblock_offset_4 (isplit_sum) = splitsum
# 127 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               dist_split_d(isplit_sum) = dist_d(i)
               splitsum = splitsum + blk_size_split_d(isplit_sum)
            ENDDO
            DBCSR_ASSERT(splitsum == blk_size_d(i))
# 132 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (idim == 1) THEN
               blk_nsplit_1 (i) = nsplit
               index_split_offset_1 (i) = isplit_sum - nsplit
            ENDIF
# 132 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (idim == 2) THEN
               blk_nsplit_2 (i) = nsplit
               index_split_offset_2 (i) = isplit_sum - nsplit
            ENDIF
# 132 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (idim == 3) THEN
               blk_nsplit_3 (i) = nsplit
               index_split_offset_3 (i) = isplit_sum - nsplit
            ENDIF
# 132 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (idim == 4) THEN
               blk_nsplit_4 (i) = nsplit
               index_split_offset_4 (i) = isplit_sum - nsplit
            ENDIF
# 137 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDDO

# 140 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 1) THEN
            CALL allocate_any(nd_dist_split_1, source=dist_split_d)
            CALL allocate_any(nd_blk_size_split_1, source=blk_size_split_d)
         ENDIF
# 140 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 2) THEN
            CALL allocate_any(nd_dist_split_2, source=dist_split_d)
            CALL allocate_any(nd_blk_size_split_2, source=blk_size_split_d)
         ENDIF
# 140 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 3) THEN
            CALL allocate_any(nd_dist_split_3, source=dist_split_d)
            CALL allocate_any(nd_blk_size_split_3, source=blk_size_split_d)
         ENDIF
# 140 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 4) THEN
            CALL allocate_any(nd_dist_split_4, source=dist_split_d)
            CALL allocate_any(nd_blk_size_split_4, source=blk_size_split_d)
         ENDIF
# 145 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         DEALLOCATE (dist_split_d)
         DEALLOCATE (blk_size_split_d)

      ENDDO

      CALL dbcsr_t_get_mapping_info(tensor_in%nd_index_blk, map1_2d=map1_2d, map2_2d=map2_2d)

# 153 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      IF (ndims_tensor(tensor_in) == 2) THEN
         CALL dbcsr_t_distribution_new_expert(dist_split, tensor_in%pgrid, map1_2d, map2_2d, &
                                              nd_dist_split_1, nd_dist_split_2)
         CALL dbcsr_t_create(tensor_out, tensor_in%name, dist_split, map1_2d, map2_2d, &
                             dbcsr_t_get_data_type(tensor_in), nd_blk_size_split_1, nd_blk_size_split_2)
      ENDIF
# 153 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      IF (ndims_tensor(tensor_in) == 3) THEN
         CALL dbcsr_t_distribution_new_expert(dist_split, tensor_in%pgrid, map1_2d, map2_2d, &
                                              nd_dist_split_1, nd_dist_split_2, nd_dist_split_3)
         CALL dbcsr_t_create(tensor_out, tensor_in%name, dist_split, map1_2d, map2_2d, &
                             dbcsr_t_get_data_type(tensor_in), nd_blk_size_split_1, nd_blk_size_split_2, nd_blk_size_split_3)
      ENDIF
# 153 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      IF (ndims_tensor(tensor_in) == 4) THEN
         CALL dbcsr_t_distribution_new_expert(dist_split, tensor_in%pgrid, map1_2d, map2_2d, &
                                              nd_dist_split_1, nd_dist_split_2, nd_dist_split_3, nd_dist_split_4)
         CALL dbcsr_t_create(tensor_out, tensor_in%name, dist_split, map1_2d, map2_2d, &
                             dbcsr_t_get_data_type(tensor_in), nd_blk_size_split_1, nd_blk_size_split_2, nd_blk_size_split_3,&
# 157 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                 & nd_blk_size_split_4)
      ENDIF
# 160 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

      CALL dbcsr_t_distribution_destroy(dist_split)

      IF (PRESENT(nodata)) THEN
         IF (nodata) THEN
            CALL timestop(handle)
            RETURN
         ENDIF
      ENDIF

      CALL dbcsr_t_iterator_start(iter, tensor_in)

      bcount = 0
      DO WHILE (dbcsr_t_iterator_blocks_left(iter))
         CALL dbcsr_t_iterator_next_block(iter, blk_index, blk, blk_size=blk_size)
# 176 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (ndims_tensor(tensor_in) == 2) THEN
# 178 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
# 178 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
# 180 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               bcount = bcount + 1
# 182 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 182 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 176 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (ndims_tensor(tensor_in) == 3) THEN
# 178 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
# 178 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
# 178 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_3 = 1, blk_nsplit_3 (blk_index(3))
# 180 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               bcount = bcount + 1
# 182 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 182 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 182 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 176 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (ndims_tensor(tensor_in) == 4) THEN
# 178 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
# 178 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
# 178 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_3 = 1, blk_nsplit_3 (blk_index(3))
# 178 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_4 = 1, blk_nsplit_4 (blk_index(4))
# 180 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               bcount = bcount + 1
# 182 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 182 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 182 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 182 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 186 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      ENDDO
      CALL dbcsr_t_iterator_stop(iter)

      ALLOCATE (blks_to_allocate(bcount, ndims_tensor(tensor_in)))

      CALL dbcsr_t_iterator_start(iter, tensor_in)

      bcount = 0
      DO WHILE (dbcsr_t_iterator_blocks_left(iter))
         CALL dbcsr_t_iterator_next_block(iter, blk_index, blk, blk_size=blk_size, blk_offset=blk_offset)

# 198 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (ndims_tensor(tensor_in) == 2) THEN
# 200 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
               bi_split(1) = index_split_offset_1 (blk_index(1)) + split_blk_1
# 200 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
               bi_split(2) = index_split_offset_2 (blk_index(2)) + split_blk_2
# 203 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               bcount = bcount + 1
               blks_to_allocate(bcount, :) = bi_split(1:ndims_tensor(tensor_in))
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 208 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 198 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (ndims_tensor(tensor_in) == 3) THEN
# 200 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
               bi_split(1) = index_split_offset_1 (blk_index(1)) + split_blk_1
# 200 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
               bi_split(2) = index_split_offset_2 (blk_index(2)) + split_blk_2
# 200 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_3 = 1, blk_nsplit_3 (blk_index(3))
               bi_split(3) = index_split_offset_3 (blk_index(3)) + split_blk_3
# 203 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               bcount = bcount + 1
               blks_to_allocate(bcount, :) = bi_split(1:ndims_tensor(tensor_in))
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 208 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 198 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (ndims_tensor(tensor_in) == 4) THEN
# 200 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
               bi_split(1) = index_split_offset_1 (blk_index(1)) + split_blk_1
# 200 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
               bi_split(2) = index_split_offset_2 (blk_index(2)) + split_blk_2
# 200 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_3 = 1, blk_nsplit_3 (blk_index(3))
               bi_split(3) = index_split_offset_3 (blk_index(3)) + split_blk_3
# 200 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_4 = 1, blk_nsplit_4 (blk_index(4))
               bi_split(4) = index_split_offset_4 (blk_index(4)) + split_blk_4
# 203 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               bcount = bcount + 1
               blks_to_allocate(bcount, :) = bi_split(1:ndims_tensor(tensor_in))
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 206 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 208 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 210 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      ENDDO

      CALL dbcsr_t_iterator_stop(iter)

      CALL dbcsr_t_reserve_blocks(tensor_out, blks_to_allocate)

      CALL dbcsr_t_iterator_start(iter, tensor_in)

      DO WHILE (dbcsr_t_iterator_blocks_left(iter))
         CALL dbcsr_t_iterator_next_block(iter, blk_index, blk, blk_size=blk_size, blk_offset=blk_offset)
# 221 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_8) THEN
# 223 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 2) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_dp_2d, found)
               DBCSR_ASSERT(found)
            ENDIF
# 223 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 3) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_dp_3d, found)
               DBCSR_ASSERT(found)
            ENDIF
# 223 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 4) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_dp_4d, found)
               DBCSR_ASSERT(found)
            ENDIF
# 228 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 221 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_4) THEN
# 223 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 2) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_sp_2d, found)
               DBCSR_ASSERT(found)
            ENDIF
# 223 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 3) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_sp_3d, found)
               DBCSR_ASSERT(found)
            ENDIF
# 223 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 4) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_sp_4d, found)
               DBCSR_ASSERT(found)
            ENDIF
# 228 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 221 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_8) THEN
# 223 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 2) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_dp_2d, found)
               DBCSR_ASSERT(found)
            ENDIF
# 223 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 3) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_dp_3d, found)
               DBCSR_ASSERT(found)
            ENDIF
# 223 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 4) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_dp_4d, found)
               DBCSR_ASSERT(found)
            ENDIF
# 228 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 221 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_4) THEN
# 223 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 2) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_sp_2d, found)
               DBCSR_ASSERT(found)
            ENDIF
# 223 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 3) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_sp_3d, found)
               DBCSR_ASSERT(found)
            ENDIF
# 223 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 4) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_sp_4d, found)
               DBCSR_ASSERT(found)
            ENDIF
# 228 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 230 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 231 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (ndims_tensor(tensor_in) == 2) THEN
# 233 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
               ! split block index
               bi_split(1) = index_split_offset_1 (blk_index(1)) + split_blk_1
               blk_shape(1) = blk_size_1 (bi_split(1))
# 233 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
               ! split block index
               bi_split(2) = index_split_offset_2 (blk_index(2)) + split_blk_2
               blk_shape(2) = blk_size_2 (bi_split(2))
# 238 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

               IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_8) THEN

# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 246 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  CALL dbcsr_t_put_block(tensor_out, bi_split(1:2), &
                                         blk_shape, &
                                         block_r_dp_2d( &
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2)))

               ENDIF
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

               IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_4) THEN

# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 246 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  CALL dbcsr_t_put_block(tensor_out, bi_split(1:2), &
                                         blk_shape, &
                                         block_r_sp_2d( &
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2)))

               ENDIF
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

               IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_8) THEN

# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 246 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  CALL dbcsr_t_put_block(tensor_out, bi_split(1:2), &
                                         blk_shape, &
                                         block_c_dp_2d( &
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2)))

               ENDIF
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

               IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_4) THEN

# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 246 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  CALL dbcsr_t_put_block(tensor_out, bi_split(1:2), &
                                         blk_shape, &
                                         block_c_sp_2d( &
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2)))

               ENDIF
# 253 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 257 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 259 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_8) THEN
               DEALLOCATE (block_r_dp_2d)
            ENDIF
# 259 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_4) THEN
               DEALLOCATE (block_r_sp_2d)
            ENDIF
# 259 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_8) THEN
               DEALLOCATE (block_c_dp_2d)
            ENDIF
# 259 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_4) THEN
               DEALLOCATE (block_c_sp_2d)
            ENDIF
# 263 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 231 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (ndims_tensor(tensor_in) == 3) THEN
# 233 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
               ! split block index
               bi_split(1) = index_split_offset_1 (blk_index(1)) + split_blk_1
               blk_shape(1) = blk_size_1 (bi_split(1))
# 233 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
               ! split block index
               bi_split(2) = index_split_offset_2 (blk_index(2)) + split_blk_2
               blk_shape(2) = blk_size_2 (bi_split(2))
# 233 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_3 = 1, blk_nsplit_3 (blk_index(3))
               ! split block index
               bi_split(3) = index_split_offset_3 (blk_index(3)) + split_blk_3
               blk_shape(3) = blk_size_3 (bi_split(3))
# 238 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

               IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_8) THEN

# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(3) = inblock_offset_3 (bi_split(3))
# 246 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  CALL dbcsr_t_put_block(tensor_out, bi_split(1:3), &
                                         blk_shape, &
                                         block_r_dp_3d( &
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & blk_shape(3)))

               ENDIF
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

               IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_4) THEN

# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(3) = inblock_offset_3 (bi_split(3))
# 246 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  CALL dbcsr_t_put_block(tensor_out, bi_split(1:3), &
                                         blk_shape, &
                                         block_r_sp_3d( &
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & blk_shape(3)))

               ENDIF
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

               IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_8) THEN

# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(3) = inblock_offset_3 (bi_split(3))
# 246 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  CALL dbcsr_t_put_block(tensor_out, bi_split(1:3), &
                                         blk_shape, &
                                         block_c_dp_3d( &
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & blk_shape(3)))

               ENDIF
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

               IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_4) THEN

# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(3) = inblock_offset_3 (bi_split(3))
# 246 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  CALL dbcsr_t_put_block(tensor_out, bi_split(1:3), &
                                         blk_shape, &
                                         block_c_sp_3d( &
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & blk_shape(3)))

               ENDIF
# 253 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 257 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 259 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_8) THEN
               DEALLOCATE (block_r_dp_3d)
            ENDIF
# 259 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_4) THEN
               DEALLOCATE (block_r_sp_3d)
            ENDIF
# 259 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_8) THEN
               DEALLOCATE (block_c_dp_3d)
            ENDIF
# 259 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_4) THEN
               DEALLOCATE (block_c_sp_3d)
            ENDIF
# 263 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 231 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (ndims_tensor(tensor_in) == 4) THEN
# 233 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_1 = 1, blk_nsplit_1 (blk_index(1))
               ! split block index
               bi_split(1) = index_split_offset_1 (blk_index(1)) + split_blk_1
               blk_shape(1) = blk_size_1 (bi_split(1))
# 233 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_2 = 1, blk_nsplit_2 (blk_index(2))
               ! split block index
               bi_split(2) = index_split_offset_2 (blk_index(2)) + split_blk_2
               blk_shape(2) = blk_size_2 (bi_split(2))
# 233 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_3 = 1, blk_nsplit_3 (blk_index(3))
               ! split block index
               bi_split(3) = index_split_offset_3 (blk_index(3)) + split_blk_3
               blk_shape(3) = blk_size_3 (bi_split(3))
# 233 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            DO split_blk_4 = 1, blk_nsplit_4 (blk_index(4))
               ! split block index
               bi_split(4) = index_split_offset_4 (blk_index(4)) + split_blk_4
               blk_shape(4) = blk_size_4 (bi_split(4))
# 238 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

               IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_8) THEN

# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(3) = inblock_offset_3 (bi_split(3))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(4) = inblock_offset_4 (bi_split(4))
# 246 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  CALL dbcsr_t_put_block(tensor_out, bi_split(1:4), &
                                         blk_shape, &
                                         block_r_dp_4d( &
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & blk_shape(3), inblock_offset(4) + 1:inblock_offset(4) + blk_shape(4)))

               ENDIF
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

               IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_4) THEN

# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(3) = inblock_offset_3 (bi_split(3))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(4) = inblock_offset_4 (bi_split(4))
# 246 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  CALL dbcsr_t_put_block(tensor_out, bi_split(1:4), &
                                         blk_shape, &
                                         block_r_sp_4d( &
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & blk_shape(3), inblock_offset(4) + 1:inblock_offset(4) + blk_shape(4)))

               ENDIF
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

               IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_8) THEN

# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(3) = inblock_offset_3 (bi_split(3))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(4) = inblock_offset_4 (bi_split(4))
# 246 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  CALL dbcsr_t_put_block(tensor_out, bi_split(1:4), &
                                         blk_shape, &
                                         block_c_dp_4d( &
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & blk_shape(3), inblock_offset(4) + 1:inblock_offset(4) + blk_shape(4)))

               ENDIF
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

               IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_4) THEN

# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (bi_split(1))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (bi_split(2))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(3) = inblock_offset_3 (bi_split(3))
# 244 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(4) = inblock_offset_4 (bi_split(4))
# 246 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  CALL dbcsr_t_put_block(tensor_out, bi_split(1:4), &
                                         blk_shape, &
                                         block_c_sp_4d( &
                                         inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & 1:inblock_offset(2) + blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) +&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                             & blk_shape(3), inblock_offset(4) + 1:inblock_offset(4) + blk_shape(4)))

               ENDIF
# 253 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 255 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            ENDDO
# 257 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

# 259 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_8) THEN
               DEALLOCATE (block_r_dp_4d)
            ENDIF
# 259 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_4) THEN
               DEALLOCATE (block_r_sp_4d)
            ENDIF
# 259 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_8) THEN
               DEALLOCATE (block_c_dp_4d)
            ENDIF
# 259 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_4) THEN
               DEALLOCATE (block_c_sp_4d)
            ENDIF
# 263 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 265 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      ENDDO
      CALL dbcsr_t_iterator_stop(iter)

      CALL dbcsr_t_finalize(tensor_out)

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

      INTEGER, DIMENSION(:), ALLOCATABLE              :: nd_blk_size_split_1, nd_blk_size_split_2, nd_blk_size_split_3,&
# 287 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          & nd_blk_size_split_4
      INTEGER :: idim, i, isplit_sum, blk_remainder, nsplit, isplit
      INTEGER, DIMENSION(:), ALLOCATABLE :: blk_size_d, blk_size_split_d

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_split_blocks', &
                                     routineP = moduleN//':'//routineN

      DO idim = 1, ndims_tensor(tensor_in)
         CALL get_ith_array(tensor_in%blk_sizes, idim, blk_size_d)

         isplit_sum = 0
         DO i = 1, SIZE(blk_size_d)
            nsplit = (blk_size_d(i) + block_sizes(idim) - 1)/block_sizes(idim)
            isplit_sum = isplit_sum + nsplit
         ENDDO

         ALLOCATE (blk_size_split_d(isplit_sum))

         isplit_sum = 0
         DO i = 1, SIZE(blk_size_d)
            nsplit = (blk_size_d(i) + block_sizes(idim) - 1)/block_sizes(idim)
            blk_remainder = blk_size_d(i)
            DO isplit = 1, nsplit
               isplit_sum = isplit_sum + 1
               blk_size_split_d(isplit_sum) = MIN(block_sizes(idim), blk_remainder)
               blk_remainder = blk_remainder - block_sizes(idim)
            ENDDO

         ENDDO

# 318 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 1) THEN
            CALL allocate_any(nd_blk_size_split_1, source=blk_size_split_d)
         ENDIF
# 318 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 2) THEN
            CALL allocate_any(nd_blk_size_split_2, source=blk_size_split_d)
         ENDIF
# 318 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 3) THEN
            CALL allocate_any(nd_blk_size_split_3, source=blk_size_split_d)
         ENDIF
# 318 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 4) THEN
            CALL allocate_any(nd_blk_size_split_4, source=blk_size_split_d)
         ENDIF
# 322 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         DEALLOCATE (blk_size_split_d)
      ENDDO

# 326 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      IF (ndims_tensor(tensor_in) == 2) CALL dbcsr_t_split_blocks_generic(tensor_in, tensor_out, &
                                                                                 nd_blk_size_split_1, nd_blk_size_split_2, &
                                                                                 nodata=nodata)
# 326 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      IF (ndims_tensor(tensor_in) == 3) CALL dbcsr_t_split_blocks_generic(tensor_in, tensor_out, &
                                                                                 nd_blk_size_split_1, nd_blk_size_split_2,&
# 327 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                                     & nd_blk_size_split_3, &
                                                                                 nodata=nodata)
# 326 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      IF (ndims_tensor(tensor_in) == 4) CALL dbcsr_t_split_blocks_generic(tensor_in, tensor_out, &
                                                                                 nd_blk_size_split_1, nd_blk_size_split_2,&
# 327 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                                                                                     & nd_blk_size_split_3, nd_blk_size_split_4, &
                                                                                 nodata=nodata)
# 330 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

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
                                                         first_split_1, first_split_2, first_split_3, first_split_4, &
                                                         split_1, split_2, split_3, split_4
      INTEGER, DIMENSION(:), ALLOCATABLE              :: inblock_offset_1, inblock_offset_2, inblock_offset_3, inblock_offset_4,&
# 346 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
          & blk_size_split_1, blk_size_split_2, blk_size_split_3, blk_size_split_4
      INTEGER, DIMENSION(:, :), ALLOCATABLE            :: blks_to_allocate
      INTEGER                                         :: idim, iblk, blk, bcount
      INTEGER                                         :: iblk_1, iblk_2, iblk_3, iblk_4, isplit_sum, splitsum, nblk
      TYPE(dbcsr_t_iterator_type)                     :: iter
      INTEGER, DIMENSION(ndims_tensor(tensor_out)) :: blk_index, blk_size, blk_offset, blk_shape, blk_index_n
      LOGICAL                                         :: found

      INTEGER, DIMENSION(4)                  :: inblock_offset
      INTEGER                                            :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_split_copyback', &
                                     routineP = moduleN//':'//routineN
# 359 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 360 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_8), DIMENSION(:,:), ALLOCATABLE :: block_r_dp_2d
      REAL(kind=real_8), DIMENSION(:,:), ALLOCATABLE :: block_split_r_dp_2d
# 360 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE :: block_r_dp_3d
      REAL(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE :: block_split_r_dp_3d
# 360 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE :: block_r_dp_4d
      REAL(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE :: block_split_r_dp_4d
# 363 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 359 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 360 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_4), DIMENSION(:,:), ALLOCATABLE :: block_r_sp_2d
      REAL(kind=real_4), DIMENSION(:,:), ALLOCATABLE :: block_split_r_sp_2d
# 360 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE :: block_r_sp_3d
      REAL(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE :: block_split_r_sp_3d
# 360 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE :: block_r_sp_4d
      REAL(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE :: block_split_r_sp_4d
# 363 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 359 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 360 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_8), DIMENSION(:,:), ALLOCATABLE :: block_c_dp_2d
      COMPLEX(kind=real_8), DIMENSION(:,:), ALLOCATABLE :: block_split_c_dp_2d
# 360 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE :: block_c_dp_3d
      COMPLEX(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE :: block_split_c_dp_3d
# 360 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE :: block_c_dp_4d
      COMPLEX(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE :: block_split_c_dp_4d
# 363 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 359 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 360 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_4), DIMENSION(:,:), ALLOCATABLE :: block_c_sp_2d
      COMPLEX(kind=real_4), DIMENSION(:,:), ALLOCATABLE :: block_split_c_sp_2d
# 360 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE :: block_c_sp_3d
      COMPLEX(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE :: block_split_c_sp_3d
# 360 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE :: block_c_sp_4d
      COMPLEX(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE :: block_split_c_sp_4d
# 363 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 364 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

      CALL timeset(routineN, handle)
      DBCSR_ASSERT(tensor_out%valid)
      IF (PRESENT(summation)) THEN
         IF (.NOT. summation) CALL dbcsr_t_clear(tensor_out)
      ELSE
         CALL dbcsr_t_clear(tensor_out)
      ENDIF

      DO idim = 1, ndims_tensor(tensor_split_in)
         CALL get_ith_array(tensor_split_in%blk_sizes, idim, blk_size_split_d)
         CALL get_ith_array(tensor_out%blk_sizes, idim, blk_size_d)

# 378 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 1) THEN
            ! data offset of split blocks w.r.t. corresponding normal block
            ALLOCATE (inblock_offset_1 (SIZE(blk_size_split_d)))
            ! normal block index for each split block
            ALLOCATE (split_1 (SIZE(blk_size_split_d)))
         ENDIF
# 378 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 2) THEN
            ! data offset of split blocks w.r.t. corresponding normal block
            ALLOCATE (inblock_offset_2 (SIZE(blk_size_split_d)))
            ! normal block index for each split block
            ALLOCATE (split_2 (SIZE(blk_size_split_d)))
         ENDIF
# 378 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 3) THEN
            ! data offset of split blocks w.r.t. corresponding normal block
            ALLOCATE (inblock_offset_3 (SIZE(blk_size_split_d)))
            ! normal block index for each split block
            ALLOCATE (split_3 (SIZE(blk_size_split_d)))
         ENDIF
# 378 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 4) THEN
            ! data offset of split blocks w.r.t. corresponding normal block
            ALLOCATE (inblock_offset_4 (SIZE(blk_size_split_d)))
            ! normal block index for each split block
            ALLOCATE (split_4 (SIZE(blk_size_split_d)))
         ENDIF
# 385 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

         ALLOCATE (last_split_d(SIZE(blk_size_d)))
         ALLOCATE (first_split_d(SIZE(blk_size_d)))
         first_split_d(1) = 1
         isplit_sum = 0
         DO iblk = 1, SIZE(blk_size_d)
            splitsum = 0
            IF (iblk .GT. 1) first_split_d(iblk) = last_split_d(iblk - 1) + 1
            DO WHILE (splitsum < blk_size_d(iblk))
               isplit_sum = isplit_sum + 1
# 396 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               IF (idim == 1) THEN
                  inblock_offset_1 (isplit_sum) = splitsum
                  split_1 (isplit_sum) = iblk
               ENDIF
# 396 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               IF (idim == 2) THEN
                  inblock_offset_2 (isplit_sum) = splitsum
                  split_2 (isplit_sum) = iblk
               ENDIF
# 396 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               IF (idim == 3) THEN
                  inblock_offset_3 (isplit_sum) = splitsum
                  split_3 (isplit_sum) = iblk
               ENDIF
# 396 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               IF (idim == 4) THEN
                  inblock_offset_4 (isplit_sum) = splitsum
                  split_4 (isplit_sum) = iblk
               ENDIF
# 401 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               splitsum = splitsum + blk_size_split_d(isplit_sum)
            ENDDO
            DBCSR_ASSERT(splitsum == blk_size_d(iblk))
            last_split_d(iblk) = isplit_sum
         ENDDO
# 407 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 1) THEN
            CALL allocate_any(first_split_1, source=first_split_d)
            CALL allocate_any(last_split_1, source=last_split_d)
            CALL allocate_any(blk_size_split_1, source=blk_size_split_d)
         ENDIF
# 407 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 2) THEN
            CALL allocate_any(first_split_2, source=first_split_d)
            CALL allocate_any(last_split_2, source=last_split_d)
            CALL allocate_any(blk_size_split_2, source=blk_size_split_d)
         ENDIF
# 407 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 3) THEN
            CALL allocate_any(first_split_3, source=first_split_d)
            CALL allocate_any(last_split_3, source=last_split_d)
            CALL allocate_any(blk_size_split_3, source=blk_size_split_d)
         ENDIF
# 407 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 4) THEN
            CALL allocate_any(first_split_4, source=first_split_d)
            CALL allocate_any(last_split_4, source=last_split_d)
            CALL allocate_any(blk_size_split_4, source=blk_size_split_d)
         ENDIF
# 413 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         DEALLOCATE (first_split_d, last_split_d)
         DEALLOCATE (blk_size_split_d, blk_size_d)
      ENDDO

      nblk = dbcsr_t_get_num_blocks(tensor_split_in)
      ALLOCATE (blks_to_allocate(nblk, ndims_tensor(tensor_split_in)))
      CALL dbcsr_t_iterator_start(iter, tensor_split_in)
      bcount = 0
      DO WHILE (dbcsr_t_iterator_blocks_left(iter))
         CALL dbcsr_t_iterator_next_block(iter, blk_index, blk, blk_size=blk_size)
# 424 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (ndims_tensor(tensor_out) == 2) THEN
# 426 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            blk_index_n(1) = split_1 (blk_index(1))
# 426 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            blk_index_n(2) = split_2 (blk_index(2))
# 428 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 424 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (ndims_tensor(tensor_out) == 3) THEN
# 426 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            blk_index_n(1) = split_1 (blk_index(1))
# 426 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            blk_index_n(2) = split_2 (blk_index(2))
# 426 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            blk_index_n(3) = split_3 (blk_index(3))
# 428 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 424 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (ndims_tensor(tensor_out) == 4) THEN
# 426 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            blk_index_n(1) = split_1 (blk_index(1))
# 426 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            blk_index_n(2) = split_2 (blk_index(2))
# 426 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            blk_index_n(3) = split_3 (blk_index(3))
# 426 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            blk_index_n(4) = split_4 (blk_index(4))
# 428 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 430 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         blks_to_allocate(bcount + 1, :) = blk_index_n
         bcount = bcount + 1
      ENDDO
      CALL dbcsr_t_iterator_stop(iter)
      CALL dbcsr_t_reserve_blocks(tensor_out, blks_to_allocate)

      CALL dbcsr_t_iterator_start(iter, tensor_out)
      DO WHILE (dbcsr_t_iterator_blocks_left(iter))
         CALL dbcsr_t_iterator_next_block(iter, blk_index, blk, blk_size=blk_size, blk_offset=blk_offset)
# 440 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (dbcsr_t_get_data_type(tensor_out) == dbcsr_type_real_8) THEN

# 443 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_out) == 2) THEN
               CALL allocate_any(block_r_dp_2d, blk_size)
               block_r_dp_2d = 0.0_real_8
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (iblk_1)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (iblk_2)
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                  CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2], &
                                         block_split_r_dp_2d, found)
                  IF (found) THEN
                     blk_shape(1:2) = SHAPE(block_split_r_dp_2d)
                     block_r_dp_2d( &
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2)) = &
                        block_split_r_dp_2d
                  ENDIF

# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_r_dp_2d, summation=summation)
               DEALLOCATE (block_r_dp_2d)
            ENDIF
# 443 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_out) == 3) THEN
               CALL allocate_any(block_r_dp_3d, blk_size)
               block_r_dp_3d = 0.0_real_8
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_3 = first_split_3 (blk_index(3)), last_split_3 (blk_index(3))
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (iblk_1)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (iblk_2)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(3) = inblock_offset_3 (iblk_3)
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                  CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2, iblk_3], &
                                         block_split_r_dp_3d, found)
                  IF (found) THEN
                     blk_shape(1:3) = SHAPE(block_split_r_dp_3d)
                     block_r_dp_3d( &
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) + blk_shape(3)) = &
                        block_split_r_dp_3d
                  ENDIF

# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_r_dp_3d, summation=summation)
               DEALLOCATE (block_r_dp_3d)
            ENDIF
# 443 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_out) == 4) THEN
               CALL allocate_any(block_r_dp_4d, blk_size)
               block_r_dp_4d = 0.0_real_8
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_3 = first_split_3 (blk_index(3)), last_split_3 (blk_index(3))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_4 = first_split_4 (blk_index(4)), last_split_4 (blk_index(4))
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (iblk_1)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (iblk_2)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(3) = inblock_offset_3 (iblk_3)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(4) = inblock_offset_4 (iblk_4)
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                  CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2, iblk_3, iblk_4], &
                                         block_split_r_dp_4d, found)
                  IF (found) THEN
                     blk_shape(1:4) = SHAPE(block_split_r_dp_4d)
                     block_r_dp_4d( &
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) + blk_shape(3), inblock_offset(4) +&
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & 1:inblock_offset(4) + blk_shape(4)) = &
                        block_split_r_dp_4d
                  ENDIF

# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_r_dp_4d, summation=summation)
               DEALLOCATE (block_r_dp_4d)
            ENDIF
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 440 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (dbcsr_t_get_data_type(tensor_out) == dbcsr_type_real_4) THEN

# 443 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_out) == 2) THEN
               CALL allocate_any(block_r_sp_2d, blk_size)
               block_r_sp_2d = 0.0_real_4
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (iblk_1)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (iblk_2)
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                  CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2], &
                                         block_split_r_sp_2d, found)
                  IF (found) THEN
                     blk_shape(1:2) = SHAPE(block_split_r_sp_2d)
                     block_r_sp_2d( &
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2)) = &
                        block_split_r_sp_2d
                  ENDIF

# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_r_sp_2d, summation=summation)
               DEALLOCATE (block_r_sp_2d)
            ENDIF
# 443 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_out) == 3) THEN
               CALL allocate_any(block_r_sp_3d, blk_size)
               block_r_sp_3d = 0.0_real_4
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_3 = first_split_3 (blk_index(3)), last_split_3 (blk_index(3))
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (iblk_1)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (iblk_2)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(3) = inblock_offset_3 (iblk_3)
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                  CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2, iblk_3], &
                                         block_split_r_sp_3d, found)
                  IF (found) THEN
                     blk_shape(1:3) = SHAPE(block_split_r_sp_3d)
                     block_r_sp_3d( &
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) + blk_shape(3)) = &
                        block_split_r_sp_3d
                  ENDIF

# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_r_sp_3d, summation=summation)
               DEALLOCATE (block_r_sp_3d)
            ENDIF
# 443 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_out) == 4) THEN
               CALL allocate_any(block_r_sp_4d, blk_size)
               block_r_sp_4d = 0.0_real_4
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_3 = first_split_3 (blk_index(3)), last_split_3 (blk_index(3))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_4 = first_split_4 (blk_index(4)), last_split_4 (blk_index(4))
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (iblk_1)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (iblk_2)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(3) = inblock_offset_3 (iblk_3)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(4) = inblock_offset_4 (iblk_4)
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                  CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2, iblk_3, iblk_4], &
                                         block_split_r_sp_4d, found)
                  IF (found) THEN
                     blk_shape(1:4) = SHAPE(block_split_r_sp_4d)
                     block_r_sp_4d( &
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) + blk_shape(3), inblock_offset(4) +&
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & 1:inblock_offset(4) + blk_shape(4)) = &
                        block_split_r_sp_4d
                  ENDIF

# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_r_sp_4d, summation=summation)
               DEALLOCATE (block_r_sp_4d)
            ENDIF
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 440 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (dbcsr_t_get_data_type(tensor_out) == dbcsr_type_complex_8) THEN

# 443 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_out) == 2) THEN
               CALL allocate_any(block_c_dp_2d, blk_size)
               block_c_dp_2d = 0.0_real_8
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (iblk_1)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (iblk_2)
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                  CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2], &
                                         block_split_c_dp_2d, found)
                  IF (found) THEN
                     blk_shape(1:2) = SHAPE(block_split_c_dp_2d)
                     block_c_dp_2d( &
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2)) = &
                        block_split_c_dp_2d
                  ENDIF

# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_c_dp_2d, summation=summation)
               DEALLOCATE (block_c_dp_2d)
            ENDIF
# 443 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_out) == 3) THEN
               CALL allocate_any(block_c_dp_3d, blk_size)
               block_c_dp_3d = 0.0_real_8
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_3 = first_split_3 (blk_index(3)), last_split_3 (blk_index(3))
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (iblk_1)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (iblk_2)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(3) = inblock_offset_3 (iblk_3)
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                  CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2, iblk_3], &
                                         block_split_c_dp_3d, found)
                  IF (found) THEN
                     blk_shape(1:3) = SHAPE(block_split_c_dp_3d)
                     block_c_dp_3d( &
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) + blk_shape(3)) = &
                        block_split_c_dp_3d
                  ENDIF

# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_c_dp_3d, summation=summation)
               DEALLOCATE (block_c_dp_3d)
            ENDIF
# 443 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_out) == 4) THEN
               CALL allocate_any(block_c_dp_4d, blk_size)
               block_c_dp_4d = 0.0_real_8
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_3 = first_split_3 (blk_index(3)), last_split_3 (blk_index(3))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_4 = first_split_4 (blk_index(4)), last_split_4 (blk_index(4))
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (iblk_1)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (iblk_2)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(3) = inblock_offset_3 (iblk_3)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(4) = inblock_offset_4 (iblk_4)
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                  CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2, iblk_3, iblk_4], &
                                         block_split_c_dp_4d, found)
                  IF (found) THEN
                     blk_shape(1:4) = SHAPE(block_split_c_dp_4d)
                     block_c_dp_4d( &
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) + blk_shape(3), inblock_offset(4) +&
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & 1:inblock_offset(4) + blk_shape(4)) = &
                        block_split_c_dp_4d
                  ENDIF

# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_c_dp_4d, summation=summation)
               DEALLOCATE (block_c_dp_4d)
            ENDIF
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 440 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (dbcsr_t_get_data_type(tensor_out) == dbcsr_type_complex_4) THEN

# 443 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_out) == 2) THEN
               CALL allocate_any(block_c_sp_2d, blk_size)
               block_c_sp_2d = 0.0_real_4
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (iblk_1)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (iblk_2)
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                  CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2], &
                                         block_split_c_sp_2d, found)
                  IF (found) THEN
                     blk_shape(1:2) = SHAPE(block_split_c_sp_2d)
                     block_c_sp_2d( &
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2)) = &
                        block_split_c_sp_2d
                  ENDIF

# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_c_sp_2d, summation=summation)
               DEALLOCATE (block_c_sp_2d)
            ENDIF
# 443 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_out) == 3) THEN
               CALL allocate_any(block_c_sp_3d, blk_size)
               block_c_sp_3d = 0.0_real_4
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_3 = first_split_3 (blk_index(3)), last_split_3 (blk_index(3))
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (iblk_1)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (iblk_2)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(3) = inblock_offset_3 (iblk_3)
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                  CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2, iblk_3], &
                                         block_split_c_sp_3d, found)
                  IF (found) THEN
                     blk_shape(1:3) = SHAPE(block_split_c_sp_3d)
                     block_c_sp_3d( &
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) + blk_shape(3)) = &
                        block_split_c_sp_3d
                  ENDIF

# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_c_sp_3d, summation=summation)
               DEALLOCATE (block_c_sp_3d)
            ENDIF
# 443 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_out) == 4) THEN
               CALL allocate_any(block_c_sp_4d, blk_size)
               block_c_sp_4d = 0.0_real_4
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_1 = first_split_1 (blk_index(1)), last_split_1 (blk_index(1))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_2 = first_split_2 (blk_index(2)), last_split_2 (blk_index(2))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_3 = first_split_3 (blk_index(3)), last_split_3 (blk_index(3))
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               DO iblk_4 = first_split_4 (blk_index(4)), last_split_4 (blk_index(4))
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(1) = inblock_offset_1 (iblk_1)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(2) = inblock_offset_2 (iblk_2)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(3) = inblock_offset_3 (iblk_3)
# 450 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                  inblock_offset(4) = inblock_offset_4 (iblk_4)
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

                  CALL dbcsr_t_get_block(tensor_split_in, [iblk_1, iblk_2, iblk_3, iblk_4], &
                                         block_split_c_sp_4d, found)
                  IF (found) THEN
                     blk_shape(1:4) = SHAPE(block_split_c_sp_4d)
                     block_c_sp_4d( &
                        inblock_offset(1) + 1:inblock_offset(1) + blk_shape(1), inblock_offset(2) + 1:inblock_offset(2) +&
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & blk_shape(2), inblock_offset(3) + 1:inblock_offset(3) + blk_shape(3), inblock_offset(4) +&
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                            & 1:inblock_offset(4) + blk_shape(4)) = &
                        block_split_c_sp_4d
                  ENDIF

# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               ENDDO
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_c_sp_4d, summation=summation)
               DEALLOCATE (block_c_sp_4d)
            ENDIF
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 471 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      ENDDO
      CALL dbcsr_t_iterator_stop(iter)

      CALL timestop(handle)

   END SUBROUTINE

   SUBROUTINE dbcsr_t_make_compatible_blocks(tensor1, tensor2, tensor1_split, tensor2_split, nodata1, nodata2, move_data)
      !! split two tensors with same total sizes but different block sizes such that they have equal
      !! block sizes
      !! \move_data memory optimization: transfer data s.t. tensor1 and tensor2 may be empty on return

      TYPE(dbcsr_t_type), INTENT(INOUT) :: tensor1, tensor2
         !! tensor 1 in
         !! tensor 2 in
      TYPE(dbcsr_t_type), INTENT(OUT)   :: tensor1_split, tensor2_split
         !! tensor 1 with split blocks
         !! tensor 2 with split blocks
      LOGICAL, INTENT(IN), OPTIONAL     :: nodata1, nodata2, move_data
         !! don't copy data of tensor 1
         !! don't copy data of tensor 2
      INTEGER, DIMENSION(:), ALLOCATABLE  :: blk_size_split_1, blk_size_split_2, blk_size_split_3, blk_size_split_4, &
                                             blk_size_d_1, blk_size_d_2, blk_size_d_split
      INTEGER :: size_sum_1, size_sum_2, size_sum, bind_1, bind_2, isplit, bs, idim
      LOGICAL :: move_prv, nodata1_prv, nodata2_prv

      IF (PRESENT(move_data)) THEN
         move_prv = move_data
      ELSE
         move_prv = .FALSE.
      ENDIF

      IF (PRESENT(nodata1)) THEN
         nodata1_prv = nodata1
      ELSE
         nodata1_prv = .FALSE.
      ENDIF
      IF (PRESENT(nodata2)) THEN
         nodata2_prv = nodata2
      ELSE
         nodata2_prv = .FALSE.
      ENDIF

      DO idim = 1, ndims_tensor(tensor1)
         CALL get_ith_array(tensor1%blk_sizes, idim, blk_size_d_1)
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
            ENDIF
         ENDDO

         IF (bind_1 < SIZE(blk_size_d_1)) THEN
            bind_1 = bind_1 + 1
            bs = blk_size_d_1(bind_1)
            size_sum = size_sum + bs
            isplit = isplit + 1
            blk_size_d_split(isplit) = bs
         ENDIF

         IF (bind_2 < SIZE(blk_size_d_2)) THEN
            bind_2 = bind_2 + 1
            bs = blk_size_d_2(bind_2)
            size_sum = size_sum + bs
            isplit = isplit + 1
            blk_size_d_split(isplit) = bs
         ENDIF

# 567 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 1) THEN
            CALL allocate_any(blk_size_split_1, source=blk_size_d_split(:isplit))
         ENDIF
# 567 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 2) THEN
            CALL allocate_any(blk_size_split_2, source=blk_size_d_split(:isplit))
         ENDIF
# 567 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 3) THEN
            CALL allocate_any(blk_size_split_3, source=blk_size_d_split(:isplit))
         ENDIF
# 567 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (idim == 4) THEN
            CALL allocate_any(blk_size_split_4, source=blk_size_d_split(:isplit))
         ENDIF
# 571 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

         DEALLOCATE (blk_size_d_split)
      ENDDO

# 576 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      IF (ndims_tensor(tensor1) == 2) THEN
         CALL dbcsr_t_split_blocks_generic(tensor1, tensor1_split, blk_size_split_1, blk_size_split_2, nodata=nodata1)
         IF (move_prv .AND. .NOT. nodata1_prv) CALL dbcsr_t_clear(tensor1)
         CALL dbcsr_t_split_blocks_generic(tensor2, tensor2_split, blk_size_split_1, blk_size_split_2, nodata=nodata2)
         IF (move_prv .AND. .NOT. nodata2_prv) CALL dbcsr_t_clear(tensor2)
      ENDIF
# 576 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      IF (ndims_tensor(tensor1) == 3) THEN
         CALL dbcsr_t_split_blocks_generic(tensor1, tensor1_split, blk_size_split_1, blk_size_split_2, blk_size_split_3,&
# 577 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
             & nodata=nodata1)
         IF (move_prv .AND. .NOT. nodata1_prv) CALL dbcsr_t_clear(tensor1)
         CALL dbcsr_t_split_blocks_generic(tensor2, tensor2_split, blk_size_split_1, blk_size_split_2, blk_size_split_3,&
# 579 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
             & nodata=nodata2)
         IF (move_prv .AND. .NOT. nodata2_prv) CALL dbcsr_t_clear(tensor2)
      ENDIF
# 576 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      IF (ndims_tensor(tensor1) == 4) THEN
         CALL dbcsr_t_split_blocks_generic(tensor1, tensor1_split, blk_size_split_1, blk_size_split_2, blk_size_split_3,&
# 577 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
             & blk_size_split_4, nodata=nodata1)
         IF (move_prv .AND. .NOT. nodata1_prv) CALL dbcsr_t_clear(tensor1)
         CALL dbcsr_t_split_blocks_generic(tensor2, tensor2_split, blk_size_split_1, blk_size_split_2, blk_size_split_3,&
# 579 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
             & blk_size_split_4, nodata=nodata2)
         IF (move_prv .AND. .NOT. nodata2_prv) CALL dbcsr_t_clear(tensor2)
      ENDIF
# 583 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

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
# 598 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 599 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_8), DIMENSION(:,:), ALLOCATABLE :: block_r_dp_2d, block_put_r_dp_2d
# 599 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE :: block_r_dp_3d, block_put_r_dp_3d
# 599 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE :: block_r_dp_4d, block_put_r_dp_4d
# 601 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 598 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 599 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_4), DIMENSION(:,:), ALLOCATABLE :: block_r_sp_2d, block_put_r_sp_2d
# 599 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE :: block_r_sp_3d, block_put_r_sp_3d
# 599 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      REAL(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE :: block_r_sp_4d, block_put_r_sp_4d
# 601 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 598 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 599 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_8), DIMENSION(:,:), ALLOCATABLE :: block_c_dp_2d, block_put_c_dp_2d
# 599 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE :: block_c_dp_3d, block_put_c_dp_3d
# 599 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE :: block_c_dp_4d, block_put_c_dp_4d
# 601 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 598 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 599 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_4), DIMENSION(:,:), ALLOCATABLE :: block_c_sp_2d, block_put_c_sp_2d
# 599 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE :: block_c_sp_3d, block_put_c_sp_3d
# 599 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      COMPLEX(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE :: block_c_sp_4d, block_put_c_sp_4d
# 601 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
# 602 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"

      IF (PRESENT(move_data)) THEN
         move_data_prv = move_data
      ELSE
         move_data_prv = .FALSE.
      ENDIF

      CALL dbcsr_t_create(tensor_in, tensor_out)

      ! reserve blocks inside bounds
      CALL dbcsr_t_reserved_block_indices(tensor_in, blk_ind)
      nblk = dbcsr_t_get_num_blocks(tensor_in)
      ALLOCATE (blk_ind_tmp(nblk, ndims_tensor(tensor_in)))
      blk_ind_tmp(:, :) = 0
      iblk = 0
      blk_loop: DO iblk_all = 1, nblk
         CALL dbcsr_t_blk_offsets(tensor_in, blk_ind(iblk_all, :), blk_offset)
         CALL dbcsr_t_blk_sizes(tensor_in, blk_ind(iblk_all, :), blk_size)
         DO idim = 1, ndims_tensor(tensor_in)
            IF (bounds(1, idim) > blk_offset(idim) - 1 + blk_size(idim)) CYCLE blk_loop
            IF (bounds(2, idim) < blk_offset(idim)) CYCLE blk_loop
         ENDDO
         iblk = iblk + 1
         blk_ind_tmp(iblk, :) = blk_ind(iblk_all, :)
      ENDDO blk_loop

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
         ENDDO

# 645 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_8) THEN
# 647 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 2) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_dp_2d, found)

               CALL allocate_any(block_put_r_dp_2d, blk_size)
               block_put_r_dp_2d = 0.0_real_8
               block_put_r_dp_2d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2)) = &
                  block_r_dp_2d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2))
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_r_dp_2d)
               DEALLOCATE (block_r_dp_2d)
               DEALLOCATE (block_put_r_dp_2d)
            ENDIF
# 647 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 3) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_dp_3d, found)

               CALL allocate_any(block_put_r_dp_3d, blk_size)
               block_put_r_dp_3d = 0.0_real_8
               block_put_r_dp_3d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 652 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & 3):blk_bounds(2,3)) = &
                  block_r_dp_3d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 653 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                      & 3):blk_bounds(2,3))
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_r_dp_3d)
               DEALLOCATE (block_r_dp_3d)
               DEALLOCATE (block_put_r_dp_3d)
            ENDIF
# 647 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 4) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_dp_4d, found)

               CALL allocate_any(block_put_r_dp_4d, blk_size)
               block_put_r_dp_4d = 0.0_real_8
               block_put_r_dp_4d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 652 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & 3):blk_bounds(2,3), blk_bounds(1, 4):blk_bounds(2,4)) = &
                  block_r_dp_4d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 653 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                      & 3):blk_bounds(2,3), blk_bounds(1, 4):blk_bounds(2,4))
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_r_dp_4d)
               DEALLOCATE (block_r_dp_4d)
               DEALLOCATE (block_put_r_dp_4d)
            ENDIF
# 659 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 645 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_real_4) THEN
# 647 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 2) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_sp_2d, found)

               CALL allocate_any(block_put_r_sp_2d, blk_size)
               block_put_r_sp_2d = 0.0_real_4
               block_put_r_sp_2d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2)) = &
                  block_r_sp_2d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2))
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_r_sp_2d)
               DEALLOCATE (block_r_sp_2d)
               DEALLOCATE (block_put_r_sp_2d)
            ENDIF
# 647 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 3) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_sp_3d, found)

               CALL allocate_any(block_put_r_sp_3d, blk_size)
               block_put_r_sp_3d = 0.0_real_4
               block_put_r_sp_3d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 652 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & 3):blk_bounds(2,3)) = &
                  block_r_sp_3d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 653 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                      & 3):blk_bounds(2,3))
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_r_sp_3d)
               DEALLOCATE (block_r_sp_3d)
               DEALLOCATE (block_put_r_sp_3d)
            ENDIF
# 647 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 4) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_r_sp_4d, found)

               CALL allocate_any(block_put_r_sp_4d, blk_size)
               block_put_r_sp_4d = 0.0_real_4
               block_put_r_sp_4d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 652 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & 3):blk_bounds(2,3), blk_bounds(1, 4):blk_bounds(2,4)) = &
                  block_r_sp_4d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 653 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                      & 3):blk_bounds(2,3), blk_bounds(1, 4):blk_bounds(2,4))
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_r_sp_4d)
               DEALLOCATE (block_r_sp_4d)
               DEALLOCATE (block_put_r_sp_4d)
            ENDIF
# 659 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 645 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_8) THEN
# 647 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 2) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_dp_2d, found)

               CALL allocate_any(block_put_c_dp_2d, blk_size)
               block_put_c_dp_2d = 0.0_real_8
               block_put_c_dp_2d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2)) = &
                  block_c_dp_2d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2))
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_c_dp_2d)
               DEALLOCATE (block_c_dp_2d)
               DEALLOCATE (block_put_c_dp_2d)
            ENDIF
# 647 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 3) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_dp_3d, found)

               CALL allocate_any(block_put_c_dp_3d, blk_size)
               block_put_c_dp_3d = 0.0_real_8
               block_put_c_dp_3d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 652 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & 3):blk_bounds(2,3)) = &
                  block_c_dp_3d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 653 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                      & 3):blk_bounds(2,3))
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_c_dp_3d)
               DEALLOCATE (block_c_dp_3d)
               DEALLOCATE (block_put_c_dp_3d)
            ENDIF
# 647 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 4) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_dp_4d, found)

               CALL allocate_any(block_put_c_dp_4d, blk_size)
               block_put_c_dp_4d = 0.0_real_8
               block_put_c_dp_4d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 652 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & 3):blk_bounds(2,3), blk_bounds(1, 4):blk_bounds(2,4)) = &
                  block_c_dp_4d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 653 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                      & 3):blk_bounds(2,3), blk_bounds(1, 4):blk_bounds(2,4))
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_c_dp_4d)
               DEALLOCATE (block_c_dp_4d)
               DEALLOCATE (block_put_c_dp_4d)
            ENDIF
# 659 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 645 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         IF (dbcsr_t_get_data_type(tensor_in) == dbcsr_type_complex_4) THEN
# 647 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 2) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_sp_2d, found)

               CALL allocate_any(block_put_c_sp_2d, blk_size)
               block_put_c_sp_2d = 0.0_real_4
               block_put_c_sp_2d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2)) = &
                  block_c_sp_2d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2))
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_c_sp_2d)
               DEALLOCATE (block_c_sp_2d)
               DEALLOCATE (block_put_c_sp_2d)
            ENDIF
# 647 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 3) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_sp_3d, found)

               CALL allocate_any(block_put_c_sp_3d, blk_size)
               block_put_c_sp_3d = 0.0_real_4
               block_put_c_sp_3d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 652 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & 3):blk_bounds(2,3)) = &
                  block_c_sp_3d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 653 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                      & 3):blk_bounds(2,3))
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_c_sp_3d)
               DEALLOCATE (block_c_sp_3d)
               DEALLOCATE (block_put_c_sp_3d)
            ENDIF
# 647 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
            IF (ndims_tensor(tensor_in) == 4) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_c_sp_4d, found)

               CALL allocate_any(block_put_c_sp_4d, blk_size)
               block_put_c_sp_4d = 0.0_real_4
               block_put_c_sp_4d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 652 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                   & 3):blk_bounds(2,3), blk_bounds(1, 4):blk_bounds(2,4)) = &
                  block_c_sp_4d(blk_bounds(1, 1):blk_bounds(2,1), blk_bounds(1, 2):blk_bounds(2,2), blk_bounds(1,&
# 653 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
                      & 3):blk_bounds(2,3), blk_bounds(1, 4):blk_bounds(2,4))
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_c_sp_4d)
               DEALLOCATE (block_c_sp_4d)
               DEALLOCATE (block_put_c_sp_4d)
            ENDIF
# 659 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
         ENDIF
# 661 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_split.F"
      ENDDO iter_loop
      CALL dbcsr_t_iterator_stop(iter)
      CALL dbcsr_t_finalize(tensor_out)

      IF (move_data_prv) CALL dbcsr_t_clear(tensor_in)

      ! transfer data for batched contraction
      IF (tensor_in%matrix_rep%do_batched > 0) THEN
         ALLOCATE (tensor_out%matrix_rep%mm_storage, SOURCE=tensor_in%matrix_rep%mm_storage)
         tensor_out%matrix_rep%do_batched = tensor_in%matrix_rep%do_batched
      ENDIF

   END SUBROUTINE

END MODULE

