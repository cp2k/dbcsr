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
#:include "dbcsr_tensor.fypp"
#:set maxdim = maxrank
#:set ndims = range(2,maxdim+1)

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
                                 dbcsr_t_distribution_new, &
                                 dbcsr_t_clear, &
                                 dbcsr_t_finalize, &
                                 dbcsr_t_get_num_blocks, &
                                 dbcsr_t_blk_offsets, &
                                 dbcsr_t_blk_sizes
   USE dbcsr_api, ONLY: ${uselist(dtype_float_param)}$
   USE dbcsr_kinds, ONLY: ${uselist(dtype_float_prec)}$

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

   SUBROUTINE dbcsr_t_split_blocks_generic(tensor_in, tensor_out, ${varlist("blk_size")}$, nodata)
      !! Split tensor blocks into smaller blocks
      TYPE(dbcsr_t_type), INTENT(INOUT)               :: tensor_in
         !! Input tensor
      TYPE(dbcsr_t_type), INTENT(OUT)                 :: tensor_out
         !! Output tensor (splitted blocks)
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL     :: ${varlist("blk_size")}$
         !! block sizes for each of the tensor dimensions
      LOGICAL, INTENT(IN), OPTIONAL                   :: nodata
         !! don't copy data from tensor_in to tensor_out

      TYPE(dbcsr_t_distribution_type)                 :: dist_old, dist_split
      TYPE(dbcsr_t_iterator_type)                     :: iter
      INTEGER, DIMENSION(:), ALLOCATABLE              :: ${varlist("nd_dist_split")}$
      INTEGER, DIMENSION(:), ALLOCATABLE              :: ${varlist("nd_blk_size_split")}$
      INTEGER, DIMENSION(:), ALLOCATABLE              :: ${varlist("index_split_offset")}$
      INTEGER, DIMENSION(:), ALLOCATABLE              :: ${varlist("inblock_offset")}$
      INTEGER, DIMENSION(:), ALLOCATABLE              :: ${varlist("blk_nsplit")}$
      INTEGER                                         :: ${varlist("split_blk")}$
      INTEGER :: idim, i, isplit_sum, blk, nsplit, handle, splitsum, bcount
      INTEGER, DIMENSION(:, :), ALLOCATABLE           :: blks_to_allocate
      INTEGER, DIMENSION(:), ALLOCATABLE :: dist_d, blk_size_d, blk_size_split_d, dist_split_d, &
                                            map1_2d, map2_2d
      INTEGER, DIMENSION(ndims_tensor(tensor_in)) :: blk_index, blk_size, blk_offset, &
                                                     blk_shape
      INTEGER, DIMENSION(${maxdim}$) :: bi_split, inblock_offset
      LOGICAL :: found

#:for dparam, dtype, dsuffix in dtype_float_list
#:for ndim in ndims
      ${dtype}$, DIMENSION(${shape_colon(n=ndim)}$), ALLOCATABLE :: block_${dsuffix}$_${ndim}$d
#:endfor
#:endfor
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_split_blocks_generic', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)

      dist_old = dbcsr_t_distribution(tensor_in)

      DO idim = 1, ndims_tensor(tensor_in)
         CALL get_ith_array(dist_old%nd_dist, idim, dist_d)
         CALL get_ith_array(tensor_in%blk_sizes, idim, blk_size_d)

#:for idim in range(1, maxdim+1)
         IF (idim == ${idim}$) THEN
            ! split block index offset for each normal block index
            ALLOCATE (index_split_offset_${idim}$ (SIZE(dist_d)))
            ! how many split blocks for each normal block index
            ALLOCATE (blk_nsplit_${idim}$ (SIZE(dist_d)))
            ! data offset of split blocks w.r.t. corresponding normal block
            ALLOCATE (inblock_offset_${idim}$ (SIZE(blk_size_${idim}$)))
            CALL allocate_any(blk_size_split_d, source=blk_size_${idim}$)
         ENDIF
#:endfor

         ! distribution vector for split blocks
         ALLOCATE (dist_split_d(SIZE(blk_size_split_d)))

         isplit_sum = 0 ! counting splits
         DO i = 1, SIZE(blk_size_d)
            nsplit = 0 ! number of splits for current normal block
            splitsum = 0 ! summing split block sizes for current normal block
            DO WHILE (splitsum < blk_size_d(i))
               nsplit = nsplit + 1
               isplit_sum = isplit_sum + 1
#:for idim in range(1, maxdim+1)
               IF (idim == ${idim}$) inblock_offset_${idim}$ (isplit_sum) = splitsum
#:endfor
               dist_split_d(isplit_sum) = dist_d(i)
               splitsum = splitsum + blk_size_split_d(isplit_sum)
            ENDDO
            DBCSR_ASSERT(splitsum == blk_size_d(i))
#:for idim in range(1, maxdim+1)
            IF (idim == ${idim}$) THEN
               blk_nsplit_${idim}$ (i) = nsplit
               index_split_offset_${idim}$ (i) = isplit_sum - nsplit
            ENDIF
#:endfor
         ENDDO

#:for idim in range(1, maxdim+1)
         IF (idim == ${idim}$) THEN
            CALL allocate_any(nd_dist_split_${idim}$, source=dist_split_d)
            CALL allocate_any(nd_blk_size_split_${idim}$, source=blk_size_split_d)
         ENDIF
#:endfor
         DEALLOCATE (dist_split_d)
         DEALLOCATE (blk_size_split_d)

      ENDDO

      CALL dbcsr_t_get_mapping_info(tensor_in%nd_index_blk, map1_2d=map1_2d, map2_2d=map2_2d)

#:for ndim in ndims
      IF (ndims_tensor(tensor_in) == ${ndim}$) THEN
         CALL dbcsr_t_distribution_new(dist_split, tensor_in%pgrid, map1_2d, map2_2d, &
                                       ${varlist("nd_dist_split", nmax=ndim)}$)
         CALL dbcsr_t_create(tensor_out, tensor_in%name, dist_split, map1_2d, map2_2d, &
                             dbcsr_t_get_data_type(tensor_in), ${varlist("nd_blk_size_split", nmax=ndim)}$)
      ENDIF
#:endfor

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
#:for ndim in ndims
         IF (ndims_tensor(tensor_in) == ${ndim}$) THEN
#:for idim in range(1,ndim+1)
            DO split_blk_${idim}$ = 1, blk_nsplit_${idim}$ (blk_index(${idim}$))
#:endfor
               bcount = bcount + 1
#:for idim in range(1,ndim+1)
            ENDDO
#:endfor
         ENDIF
#:endfor
      ENDDO
      CALL dbcsr_t_iterator_stop(iter)

      ALLOCATE (blks_to_allocate(bcount, ndims_tensor(tensor_in)))

      CALL dbcsr_t_iterator_start(iter, tensor_in)

      bcount = 0
      DO WHILE (dbcsr_t_iterator_blocks_left(iter))
         CALL dbcsr_t_iterator_next_block(iter, blk_index, blk, blk_size=blk_size, blk_offset=blk_offset)

#:for ndim in ndims
         IF (ndims_tensor(tensor_in) == ${ndim}$) THEN
#:for idim in range(1,ndim+1)
            DO split_blk_${idim}$ = 1, blk_nsplit_${idim}$ (blk_index(${idim}$))
               bi_split(${idim}$) = index_split_offset_${idim}$ (blk_index(${idim}$)) + split_blk_${idim}$
#:endfor
               bcount = bcount + 1
               blks_to_allocate(bcount, :) = bi_split(1:ndims_tensor(tensor_in))
#:for idim in range(1,ndim+1)
            ENDDO
#:endfor
         ENDIF
#:endfor
      ENDDO

      CALL dbcsr_t_iterator_stop(iter)

      CALL dbcsr_t_reserve_blocks(tensor_out, blks_to_allocate)

      CALL dbcsr_t_iterator_start(iter, tensor_in)

      DO WHILE (dbcsr_t_iterator_blocks_left(iter))
         CALL dbcsr_t_iterator_next_block(iter, blk_index, blk, blk_size=blk_size, blk_offset=blk_offset)
#:for dparam, dtype, dsuffix in dtype_float_list
         IF (dbcsr_t_get_data_type(tensor_in) == ${dparam}$) THEN
#:for ndim in ndims
            IF (ndims_tensor(tensor_in) == ${ndim}$) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_${dsuffix}$_${ndim}$d, found)
               DBCSR_ASSERT(found)
            ENDIF
#:endfor
         ENDIF
#:endfor
#:for ndim in ndims
         IF (ndims_tensor(tensor_in) == ${ndim}$) THEN
#:for idim in range(1,ndim+1)
            DO split_blk_${idim}$ = 1, blk_nsplit_${idim}$ (blk_index(${idim}$))
               ! split block index
               bi_split(${idim}$) = index_split_offset_${idim}$ (blk_index(${idim}$)) + split_blk_${idim}$
               blk_shape(${idim}$) = blk_size_${idim}$ (bi_split(${idim}$))
#:endfor

#:for dparam, dtype, dsuffix in dtype_float_list

               IF (dbcsr_t_get_data_type(tensor_in) == ${dparam}$) THEN

#:for idim in range(1,ndim+1)
                  inblock_offset(${idim}$) = inblock_offset_${idim}$ (bi_split(${idim}$))
#:endfor
                  CALL dbcsr_t_put_block(tensor_out, bi_split(1:${ndim}$), &
                                         blk_shape, &
                                         block_${dsuffix}$_${ndim}$d( &
                                         ${", ".join(["inblock_offset("+str(idim)+") + 1:inblock_offset("+str(idim)+") + blk_shape("+str(idim)+")" for idim in range(1, ndim+1)])}$))

               ENDIF
#:endfor

#:for idim in range(1,ndim+1)
            ENDDO
#:endfor

#:for dparam, dtype, dsuffix in dtype_float_list
            IF (dbcsr_t_get_data_type(tensor_in) == ${dparam}$) THEN
               DEALLOCATE (block_${dsuffix}$_${ndim}$d)
            ENDIF
#:endfor
         ENDIF
#:endfor
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

      INTEGER, DIMENSION(:), ALLOCATABLE              :: ${varlist("nd_blk_size_split")}$
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

#:for idim in range(1, maxdim+1)
         IF (idim == ${idim}$) THEN
            CALL allocate_any(nd_blk_size_split_${idim}$, source=blk_size_split_d)
         ENDIF
#:endfor
         DEALLOCATE (blk_size_split_d)
      ENDDO

#:for ndim in ndims
      IF (ndims_tensor(tensor_in) == ${ndim}$) CALL dbcsr_t_split_blocks_generic(tensor_in, tensor_out, &
                                                                                 ${varlist("nd_blk_size_split", nmax=ndim)}$, &
                                                                                 nodata=nodata)
#:endfor

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
      INTEGER, DIMENSION(:), ALLOCATABLE              :: ${varlist("last_split")}$, &
                                                         ${varlist("first_split")}$, &
                                                         ${varlist("split")}$
      INTEGER, DIMENSION(:), ALLOCATABLE              :: ${varlist("inblock_offset")}$, ${varlist("blk_size_split")}$
      INTEGER, DIMENSION(:, :), ALLOCATABLE            :: blks_to_allocate
      INTEGER                                         :: idim, iblk, blk, bcount
      INTEGER                                         :: ${varlist("iblk")}$, isplit_sum, splitsum, nblk
      TYPE(dbcsr_t_iterator_type)                     :: iter
      INTEGER, DIMENSION(ndims_tensor(tensor_out)) :: blk_index, blk_size, blk_offset, blk_shape, blk_index_n
      LOGICAL                                         :: found

      INTEGER, DIMENSION(${maxdim}$)                  :: inblock_offset
      INTEGER                                            :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_split_copyback', &
                                     routineP = moduleN//':'//routineN
#:for dparam, dtype, dsuffix in dtype_float_list
#:for ndim in ndims
      ${dtype}$, DIMENSION(${shape_colon(n=ndim)}$), ALLOCATABLE :: block_${dsuffix}$_${ndim}$d
      ${dtype}$, DIMENSION(${shape_colon(n=ndim)}$), ALLOCATABLE :: block_split_${dsuffix}$_${ndim}$d
#:endfor
#:endfor

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

#:for idim in range(1, maxdim+1)
         IF (idim == ${idim}$) THEN
            ! data offset of split blocks w.r.t. corresponding normal block
            ALLOCATE (inblock_offset_${idim}$ (SIZE(blk_size_split_d)))
            ! normal block index for each split block
            ALLOCATE (split_${idim}$ (SIZE(blk_size_split_d)))
         ENDIF
#:endfor

         ALLOCATE (last_split_d(SIZE(blk_size_d)))
         ALLOCATE (first_split_d(SIZE(blk_size_d)))
         first_split_d(1) = 1
         isplit_sum = 0
         DO iblk = 1, SIZE(blk_size_d)
            splitsum = 0
            IF (iblk .GT. 1) first_split_d(iblk) = last_split_d(iblk - 1) + 1
            DO WHILE (splitsum < blk_size_d(iblk))
               isplit_sum = isplit_sum + 1
#:for idim in range(1, maxdim+1)
               IF (idim == ${idim}$) THEN
                  inblock_offset_${idim}$ (isplit_sum) = splitsum
                  split_${idim}$ (isplit_sum) = iblk
               ENDIF
#:endfor
               splitsum = splitsum + blk_size_split_d(isplit_sum)
            ENDDO
            DBCSR_ASSERT(splitsum == blk_size_d(iblk))
            last_split_d(iblk) = isplit_sum
         ENDDO
#:for idim in range(1, maxdim+1)
         IF (idim == ${idim}$) THEN
            CALL allocate_any(first_split_${idim}$, source=first_split_d)
            CALL allocate_any(last_split_${idim}$, source=last_split_d)
            CALL allocate_any(blk_size_split_${idim}$, source=blk_size_split_d)
         ENDIF
#:endfor
         DEALLOCATE (first_split_d, last_split_d)
         DEALLOCATE (blk_size_split_d, blk_size_d)
      ENDDO

      nblk = dbcsr_t_get_num_blocks(tensor_split_in)
      ALLOCATE (blks_to_allocate(nblk, ndims_tensor(tensor_split_in)))
      CALL dbcsr_t_iterator_start(iter, tensor_split_in)
      bcount = 0
      DO WHILE (dbcsr_t_iterator_blocks_left(iter))
         CALL dbcsr_t_iterator_next_block(iter, blk_index, blk, blk_size=blk_size)
#:for ndim in ndims
         IF (ndims_tensor(tensor_out) == ${ndim}$) THEN
#:for idim in range(1,ndim+1)
            blk_index_n(${idim}$) = split_${idim}$ (blk_index(${idim}$))
#:endfor
         ENDIF
#:endfor
         blks_to_allocate(bcount + 1, :) = blk_index_n
         bcount = bcount + 1
      ENDDO
      CALL dbcsr_t_iterator_stop(iter)
      CALL dbcsr_t_reserve_blocks(tensor_out, blks_to_allocate)

      CALL dbcsr_t_iterator_start(iter, tensor_out)
      DO WHILE (dbcsr_t_iterator_blocks_left(iter))
         CALL dbcsr_t_iterator_next_block(iter, blk_index, blk, blk_size=blk_size, blk_offset=blk_offset)
#:for dprec, dparam, dtype, dsuffix in dtype_float_list_prec
         IF (dbcsr_t_get_data_type(tensor_out) == ${dparam}$) THEN

#:for ndim in ndims
            IF (ndims_tensor(tensor_out) == ${ndim}$) THEN
               CALL allocate_any(block_${dsuffix}$_${ndim}$d, blk_size)
               block_${dsuffix}$_${ndim}$d = 0.0_${dprec}$
#:for idim in range(1,ndim+1)
               DO iblk_${idim}$ = first_split_${idim}$ (blk_index(${idim}$)), last_split_${idim}$ (blk_index(${idim}$))
#:endfor
#:for idim in range(1,ndim+1)
                  inblock_offset(${idim}$) = inblock_offset_${idim}$ (iblk_${idim}$)
#:endfor

                  CALL dbcsr_t_get_block(tensor_split_in, [${", ".join(["iblk_"+str(idim) for idim in range(1, ndim+1)])}$], &
                                         block_split_${dsuffix}$_${ndim}$d, found)
                  IF (found) THEN
                     blk_shape(1:${ndim}$) = SHAPE(block_split_${dsuffix}$_${ndim}$d)
                     block_${dsuffix}$_${ndim}$d( &
                        ${", ".join(["inblock_offset("+str(idim)+") + 1:inblock_offset("+str(idim)+") + blk_shape("+str(idim)+")" for idim in range(1, ndim+1)])}$) = &
                        block_split_${dsuffix}$_${ndim}$d
                  ENDIF

#:for idim in range(1,ndim+1)
               ENDDO
#:endfor
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_${dsuffix}$_${ndim}$d, summation=summation)
               DEALLOCATE (block_${dsuffix}$_${ndim}$d)
            ENDIF
#:endfor
         ENDIF
#:endfor
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
      INTEGER, DIMENSION(:), ALLOCATABLE  :: ${varlist("blk_size_split")}$, &
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

#:for idim in range(1, maxdim+1)
         IF (idim == ${idim}$) THEN
            CALL allocate_any(blk_size_split_${idim}$, source=blk_size_d_split(:isplit))
         ENDIF
#:endfor

         DEALLOCATE (blk_size_d_split)
      ENDDO

#:for ndim in ndims
      IF (ndims_tensor(tensor1) == ${ndim}$) THEN
         CALL dbcsr_t_split_blocks_generic(tensor1, tensor1_split, ${varlist("blk_size_split", nmax=ndim)}$, nodata=nodata1)
         IF (move_prv .AND. .NOT. nodata1_prv) CALL dbcsr_t_clear(tensor1)
         CALL dbcsr_t_split_blocks_generic(tensor2, tensor2_split, ${varlist("blk_size_split", nmax=ndim)}$, nodata=nodata2)
         IF (move_prv .AND. .NOT. nodata2_prv) CALL dbcsr_t_clear(tensor2)
      ENDIF
#:endfor

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
#:for dparam, dtype, dsuffix in dtype_float_list
#:for ndim in ndims
      ${dtype}$, DIMENSION(${shape_colon(n=ndim)}$), ALLOCATABLE :: block_${dsuffix}$_${ndim}$d, block_put_${dsuffix}$_${ndim}$d
#:endfor
#:endfor

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

#:for dprec, dparam, dtype, dsuffix in dtype_float_list_prec
         IF (dbcsr_t_get_data_type(tensor_in) == ${dparam}$) THEN
#:for ndim in ndims
            IF (ndims_tensor(tensor_in) == ${ndim}$) THEN
               CALL dbcsr_t_get_block(tensor_in, blk_index, block_${dsuffix}$_${ndim}$d, found)

               CALL allocate_any(block_put_${dsuffix}$_${ndim}$d, blk_size)
               block_put_${dsuffix}$_${ndim}$d = 0.0_${dprec}$
               block_put_${dsuffix}$_${ndim}$d(${", ".join(["blk_bounds(1, "+str(idim)+"):blk_bounds(2,"+str(idim)+")" for idim in range(1, ndim+1)])}$) = &
                  block_${dsuffix}$_${ndim}$d(${", ".join(["blk_bounds(1, "+str(idim)+"):blk_bounds(2,"+str(idim)+")" for idim in range(1, ndim+1)])}$)
               CALL dbcsr_t_put_block(tensor_out, blk_index, blk_size, block_put_${dsuffix}$_${ndim}$d)
               DEALLOCATE (block_${dsuffix}$_${ndim}$d)
               DEALLOCATE (block_put_${dsuffix}$_${ndim}$d)
            ENDIF
#:endfor
         ENDIF
#:endfor
      ENDDO iter_loop
      CALL dbcsr_t_iterator_stop(iter)
      CALL dbcsr_t_finalize(tensor_out)

      IF (move_data_prv) CALL dbcsr_t_clear(tensor_in)
   END SUBROUTINE

END MODULE

