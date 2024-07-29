!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tensor_block
   !! Methods to operate on n-dimensional tensor blocks.

   #:include "dbcsr_tensor.fypp"
   #:set maxdim = maxrank
   #:set ndims = range(2,maxdim+1)

   USE dbcsr_allocate_wrap, ONLY: &
      allocate_any
   USE dbcsr_api, ONLY: &
      ${uselist(dtype_float_param)}$, dbcsr_iterator_type, &
      dbcsr_iterator_next_block, dbcsr_iterator_start, dbcsr_iterator_stop, dbcsr_type, &
      dbcsr_reserve_blocks, dbcsr_scalar_type, dbcsr_finalize, dbcsr_get_num_blocks, &
      dbcsr_type_no_symmetry, dbcsr_desymmetrize, dbcsr_release, dbcsr_has_symmetry
   USE dbcsr_tas_types, ONLY: &
      dbcsr_tas_iterator
   USE dbcsr_tas_base, ONLY: &
      dbcsr_tas_iterator_next_block, dbcsr_tas_iterator_blocks_left, dbcsr_tas_iterator_start, &
      dbcsr_tas_iterator_stop, dbcsr_tas_get_block_p, dbcsr_tas_put_block, dbcsr_tas_reserve_blocks
   USE dbcsr_kinds, ONLY: &
      ${uselist(dtype_float_prec)}$, int_8
   USE dbcsr_tensor_index, ONLY: &
      nd_to_2d_mapping, ndims_mapping, get_nd_indices_tensor, destroy_nd_to_2d_mapping, get_2d_indices_tensor
   USE dbcsr_array_list_methods, ONLY: &
      array_list, get_array_elements, destroy_array_list, sizes_of_arrays, create_array_list, &
      get_arrays
   USE dbcsr_tensor_types, ONLY: &
      dbcsr_t_type, ndims_tensor, dbcsr_t_get_data_type, dbcsr_t_blk_sizes, dbcsr_t_get_num_blocks, &
      dbcsr_t_finalize, ndims_matrix_row, ndims_matrix_column
   USE dbcsr_dist_operations, ONLY: &
      checker_tr
   USE dbcsr_toollib, ONLY: &
      swap
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tensor_block'

   PUBLIC :: &
      block_nd, &
      create_block, &
      dbcsr_t_get_block, &
      dbcsr_t_iterator_blocks_left, &
      dbcsr_t_iterator_next_block, &
      dbcsr_t_iterator_start, &
      dbcsr_t_iterator_stop, &
      dbcsr_t_iterator_type, &
      dbcsr_t_put_block, &
      dbcsr_t_reserve_blocks, &
      dbcsr_t_reserved_block_indices, &
      destroy_block, &
      ndims_iterator

   TYPE dbcsr_t_iterator_type
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && (TO_VERSION(9, 5) > TO_VERSION(__GNUC__, __GNUC_MINOR__))
      TYPE(dbcsr_tas_iterator)    :: iter
      TYPE(nd_to_2d_mapping)      :: nd_index_blk
      TYPE(nd_to_2d_mapping)      :: nd_index
      TYPE(array_list)            :: blk_sizes, blk_offsets
#else
      TYPE(dbcsr_tas_iterator)    :: iter = dbcsr_tas_iterator()
      TYPE(nd_to_2d_mapping)      :: nd_index_blk = nd_to_2d_mapping()
      TYPE(nd_to_2d_mapping)      :: nd_index = nd_to_2d_mapping()
      TYPE(array_list)            :: blk_sizes = array_list(), blk_offsets = array_list()
#endif
   END TYPE dbcsr_t_iterator_type

   #:for dparam, dtype, dsuffix in dtype_float_list
      PUBLIC :: block_nd_${dsuffix}$
   #:endfor

   #:for dparam, dtype, dsuffix in dtype_float_list
      TYPE block_nd_${dsuffix}$
         INTEGER, DIMENSION(:), ALLOCATABLE   :: sizes
         ${dtype}$, DIMENSION(:), ALLOCATABLE :: blk
      END TYPE

   #:endfor

   TYPE block_nd
      #:for dparam, dtype, dsuffix in dtype_float_list
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && (TO_VERSION(9, 5) > TO_VERSION(__GNUC__, __GNUC_MINOR__))
         TYPE(block_nd_${dsuffix}$) :: ${dsuffix}$
#else
         TYPE(block_nd_${dsuffix}$) :: ${dsuffix}$ = block_nd_${dsuffix}$ ()
#endif
      #:endfor
      INTEGER          :: data_type = -1
   END TYPE

   INTERFACE create_block
      #:for dparam, dtype, dsuffix in dtype_float_list
         MODULE PROCEDURE create_block_data_${dsuffix}$
      #:endfor
      MODULE PROCEDURE create_block_nodata
   END INTERFACE

   INTERFACE dbcsr_t_put_block
      #:for dparam, dtype, dsuffix in dtype_float_list
         #:for ndim in ndims
            MODULE PROCEDURE dbcsr_t_put_${ndim}$d_block_${dsuffix}$
         #:endfor
      #:endfor
      MODULE PROCEDURE dbcsr_t_put_anyd_block
   END INTERFACE

   INTERFACE dbcsr_t_get_block
      #:for dparam, dtype, dsuffix in dtype_float_list
         #:for ndim in ndims
            MODULE PROCEDURE dbcsr_t_get_${ndim}$d_block_${dsuffix}$
            MODULE PROCEDURE dbcsr_t_allocate_and_get_${ndim}$d_block_${dsuffix}$
         #:endfor
      #:endfor
      MODULE PROCEDURE dbcsr_t_get_anyd_block
   END INTERFACE

   INTERFACE dbcsr_t_reserve_blocks
      MODULE PROCEDURE dbcsr_t_reserve_blocks_index
      MODULE PROCEDURE dbcsr_t_reserve_blocks_index_array
      MODULE PROCEDURE dbcsr_t_reserve_blocks_template
      MODULE PROCEDURE dbcsr_t_reserve_blocks_tensor_to_matrix
      MODULE PROCEDURE dbcsr_t_reserve_blocks_matrix_to_tensor
   END INTERFACE

CONTAINS

   SUBROUTINE create_block_nodata(block, sizes, data_type)
      !! Create block without data
      TYPE(block_nd), INTENT(OUT)       :: block
      INTEGER, DIMENSION(:), INTENT(IN) :: sizes
      INTEGER, INTENT(IN)               :: data_type

      block%data_type = data_type
      SELECT CASE (data_type)
         #:for dparam, dtype, dsuffix in dtype_float_list
            CASE (${dparam}$)
            CALL create_block_nodata_${dsuffix}$ (block%${dsuffix}$, sizes)
         #:endfor
      END SELECT
   END SUBROUTINE

   SUBROUTINE destroy_block(block)
      !! Destroy block
      TYPE(block_nd), INTENT(INOUT) :: block

      SELECT CASE (block%data_type)
         #:for dparam, dtype, dsuffix in dtype_float_list
            CASE (${dparam}$)
            CALL destroy_block_${dsuffix}$ (block%${dsuffix}$)
         #:endfor
      END SELECT

   END SUBROUTINE

   FUNCTION block_size(block)
      !! block size
      TYPE(block_nd), INTENT(IN)         :: block
      INTEGER, ALLOCATABLE, DIMENSION(:) :: block_size

      block_size = 0 ! invalid
      SELECT CASE (block%data_type)
         #:for dparam, dtype, dsuffix in dtype_float_list
            CASE (${dparam}$)
            CALL allocate_any(block_size, source=block%${dsuffix}$%sizes)
         #:endfor
      END SELECT
   END FUNCTION

   SUBROUTINE dbcsr_t_iterator_start(iterator, tensor)
      !! Generalization of dbcsr_iterator_start for tensors.
      TYPE(dbcsr_t_iterator_type), INTENT(OUT)           :: iterator
      TYPE(dbcsr_t_type), INTENT(IN)                     :: tensor

      DBCSR_ASSERT(tensor%valid)

      CALL dbcsr_tas_iterator_start(iterator%iter, tensor%matrix_rep)
      iterator%nd_index_blk = tensor%nd_index_blk
      iterator%nd_index = tensor%nd_index
      iterator%blk_sizes = tensor%blk_sizes
      iterator%blk_offsets = tensor%blk_offsets

   END SUBROUTINE

   SUBROUTINE dbcsr_t_iterator_stop(iterator)
      !! Generalization of dbcsr_iterator_stop for tensors.
      TYPE(dbcsr_t_iterator_type), INTENT(INOUT) :: iterator

      CALL dbcsr_tas_iterator_stop(iterator%iter)
      CALL destroy_nd_to_2d_mapping(iterator%nd_index)
      CALL destroy_nd_to_2d_mapping(iterator%nd_index_blk)
      CALL destroy_array_list(iterator%blk_sizes)
      CALL destroy_array_list(iterator%blk_offsets)

   END SUBROUTINE

   PURE FUNCTION ndims_iterator(iterator)
      !! Number of dimensions.
      !!
      !! Note: specification function below must be defined before it is used in
      !! the source due to a bug in the IBM XL Fortran compiler (compilation fails)

      TYPE(dbcsr_t_iterator_type), INTENT(IN) :: iterator
      INTEGER                                 :: ndims_iterator

      ndims_iterator = iterator%nd_index%ndim_nd
   END FUNCTION

   SUBROUTINE dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_p, blk_size, blk_offset)
      !! iterate over nd blocks of an nd rank tensor, index only (blocks must be retrieved by calling
      !! dbcsr_t_get_block on tensor).

      TYPE(dbcsr_t_iterator_type), INTENT(INOUT)     :: iterator
      INTEGER, DIMENSION(ndims_iterator(iterator)), &
         INTENT(OUT)                                 :: ind_nd
         !! nd index of block
      INTEGER, INTENT(OUT)                           :: blk
         !! is this needed?
      INTEGER, INTENT(OUT), OPTIONAL                 :: blk_p
         !! is this needed?
      INTEGER, DIMENSION(ndims_iterator(iterator)), &
         INTENT(OUT), OPTIONAL                       :: blk_size, blk_offset
         !! blk size in each dimension
         !! blk offset in each dimension

      INTEGER(KIND=int_8), DIMENSION(2)              :: ind_2d

      CALL dbcsr_tas_iterator_next_block(iterator%iter, ind_2d(1), ind_2d(2), blk, blk_p=blk_p)

      ind_nd(:) = get_nd_indices_tensor(iterator%nd_index_blk, ind_2d)
      IF (PRESENT(blk_size)) blk_size(:) = get_array_elements(iterator%blk_sizes, ind_nd)
      ! note: blk_offset needs to be determined by tensor metadata, can not be derived from 2d row/col
      ! offset since block index mapping is not consistent with element index mapping
      IF (PRESENT(blk_offset)) blk_offset(:) = get_array_elements(iterator%blk_offsets, ind_nd)

   END SUBROUTINE

   FUNCTION dbcsr_t_iterator_blocks_left(iterator)
      !! Generalization of dbcsr_iterator_blocks_left for tensors.
      TYPE(dbcsr_t_iterator_type), INTENT(IN) :: iterator
      LOGICAL                                 :: dbcsr_t_iterator_blocks_left

      dbcsr_t_iterator_blocks_left = dbcsr_tas_iterator_blocks_left(iterator%iter)

   END FUNCTION

   SUBROUTINE dbcsr_t_reserve_blocks_index_array(tensor, blk_ind)
      !! reserve blocks from indices as array object
      TYPE(dbcsr_t_type), INTENT(INOUT)   :: tensor
      INTEGER, DIMENSION(:, :), INTENT(IN) :: blk_ind
      INTEGER                             :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_reserve_blocks_index_array'

      CALL timeset(routineN, handle)
      #:for ndim in ndims
         IF (ndims_tensor(tensor) == ${ndim}$) THEN
            CALL dbcsr_t_reserve_blocks(tensor, ${arrlist("blk_ind", nmax=ndim, ndim_pre=1)}$)
         END IF
      #:endfor
      CALL timestop(handle)

   END SUBROUTINE

   SUBROUTINE dbcsr_t_reserve_blocks_index(tensor, ${varlist("blk_ind")}$)
      !! reserve tensor blocks using block indices
      TYPE(dbcsr_t_type), INTENT(INOUT)           :: tensor
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL :: ${varlist("blk_ind")}$
         !! index of blocks to reserve in each dimension
      INTEGER                                     :: iblk, nblk, handle
      INTEGER(KIND=int_8), ALLOCATABLE, DIMENSION(:)          :: cols, rows
      INTEGER(KIND=int_8), DIMENSION(2)                       :: ind_2d
      TYPE(array_list)                            :: blks
      INTEGER, DIMENSION(ndims_tensor(tensor))   :: iblk_nd, ind_nd, nblk_tmp
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_reserve_blocks_index'

      CALL timeset(routineN, handle)
      DBCSR_ASSERT(tensor%valid)

      CALL create_array_list(blks, ndims_tensor(tensor), &
                             ${varlist("blk_ind")}$)

      nblk_tmp(:) = sizes_of_arrays(blks)
      nblk = nblk_tmp(1)
      ALLOCATE (cols(nblk), rows(nblk))
      DO iblk = 1, nblk
         iblk_nd(:) = iblk
         ind_nd(:) = get_array_elements(blks, iblk_nd)
         ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind_nd)
         rows(iblk) = ind_2d(1); cols(iblk) = ind_2d(2)
      END DO

      CALL dbcsr_tas_reserve_blocks(tensor%matrix_rep, rows=rows, columns=cols)
      CALL dbcsr_t_finalize(tensor)
      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE dbcsr_t_reserve_blocks_template(tensor_in, tensor_out)
      !! reserve tensor blocks using template

      TYPE(dbcsr_t_type), INTENT(IN)    :: tensor_in
         !! template tensor
      TYPE(dbcsr_t_type), INTENT(INOUT) :: tensor_out
      INTEGER                           :: handle

      INTEGER, DIMENSION(dbcsr_t_get_num_blocks(tensor_in), ndims_tensor(tensor_in)) :: blk_ind
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_reserve_blocks_template'

      CALL timeset(routineN, handle)

      CALL dbcsr_t_reserved_block_indices(tensor_in, blk_ind)
      CALL dbcsr_t_reserve_blocks(tensor_out, blk_ind)

      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE dbcsr_t_reserve_blocks_matrix_to_tensor(matrix_in, tensor_out)
      !! reserve tensor blocks using matrix template
      TYPE(dbcsr_type), TARGET, INTENT(IN) :: matrix_in
      TYPE(dbcsr_t_type), INTENT(INOUT)  :: tensor_out
      TYPE(dbcsr_type), POINTER          :: matrix_in_desym

      INTEGER                            :: blk, iblk, nblk
      INTEGER, ALLOCATABLE, DIMENSION(:) :: blk_ind_1, blk_ind_2
      INTEGER, DIMENSION(2)              :: ind_2d
      TYPE(dbcsr_iterator_type)          :: iter
      INTEGER                            :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_reserve_blocks_matrix_to_tensor'

      CALL timeset(routineN, handle)

      IF (dbcsr_has_symmetry(matrix_in)) THEN
         ALLOCATE (matrix_in_desym)
         CALL dbcsr_desymmetrize(matrix_in, matrix_in_desym)
      ELSE
         matrix_in_desym => matrix_in
      END IF

      nblk = dbcsr_get_num_blocks(matrix_in_desym)
      ALLOCATE (blk_ind_1(nblk), blk_ind_2(nblk))
      CALL dbcsr_iterator_start(iter, matrix_in_desym)
      DO iblk = 1, nblk
         CALL dbcsr_iterator_next_block(iter, ind_2d(1), ind_2d(2), blk)
         blk_ind_1(iblk) = ind_2d(1); blk_ind_2(iblk) = ind_2d(2)
      END DO
      CALL dbcsr_iterator_stop(iter)

      CALL dbcsr_t_reserve_blocks(tensor_out, blk_ind_1, blk_ind_2)

      IF (dbcsr_has_symmetry(matrix_in)) THEN
         CALL dbcsr_release(matrix_in_desym)
         DEALLOCATE (matrix_in_desym)
      END IF

      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE dbcsr_t_reserve_blocks_tensor_to_matrix(tensor_in, matrix_out)
      !! reserve matrix blocks using tensor template

      TYPE(dbcsr_t_type), INTENT(IN)        :: tensor_in
      TYPE(dbcsr_type), INTENT(INOUT)       :: matrix_out
      TYPE(dbcsr_t_iterator_type)           :: iter
      INTEGER, ALLOCATABLE, DIMENSION(:) :: blk_ind_1, blk_ind_2

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_reserve_blocks_tensor_to_matrix'
      INTEGER :: handle, blk, iblk, nblk
      INTEGER, DIMENSION(2)              :: ind_2d

      CALL timeset(routineN, handle)

      nblk = dbcsr_t_get_num_blocks(tensor_in)
      ALLOCATE (blk_ind_1(nblk), blk_ind_2(nblk))
      CALL dbcsr_t_iterator_start(iter, tensor_in)

      iblk = 0
      DO WHILE (dbcsr_t_iterator_blocks_left(iter))
         CALL dbcsr_t_iterator_next_block(iter, ind_2d, blk)
         IF (dbcsr_has_symmetry(matrix_out)) THEN
            IF (checker_tr(ind_2d(1), ind_2d(2))) CYCLE
            IF (ind_2d(1) > ind_2d(2)) CALL swap(ind_2d(1), ind_2d(2))
         END IF

         iblk = iblk + 1
         blk_ind_1(iblk) = ind_2d(1); blk_ind_2(iblk) = ind_2d(2)
      END DO
      CALL dbcsr_t_iterator_stop(iter)

      CALL dbcsr_reserve_blocks(matrix_out, blk_ind_1(:iblk), blk_ind_2(:iblk))
      CALL dbcsr_finalize(matrix_out)

      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE dbcsr_t_reserved_block_indices(tensor, blk_ind)
      !! indices of non-zero blocks
      TYPE(dbcsr_t_type), INTENT(IN)            :: tensor
      INTEGER                                   :: blk, iblk, nblk
      TYPE(dbcsr_t_iterator_type)               :: iterator
      INTEGER, DIMENSION(ndims_tensor(tensor))  :: ind_nd
      INTEGER, DIMENSION(dbcsr_t_get_num_blocks(tensor), ndims_tensor(tensor)), INTENT(OUT) :: blk_ind

      DBCSR_ASSERT(tensor%valid)

      nblk = dbcsr_t_get_num_blocks(tensor)

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO iblk = 1, nblk
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk)
         blk_ind(iblk, :) = ind_nd(:)
      END DO
      CALL dbcsr_t_iterator_stop(iterator)

   END SUBROUTINE

   #:for dparam, dtype, dsuffix in dtype_float_list
      SUBROUTINE create_block_data_${dsuffix}$ (block, sizes, array)
      !! Create block from array, array can be n-dimensional.
         TYPE(block_nd), INTENT(OUT)                       :: block
         INTEGER, DIMENSION(:), INTENT(IN)                 :: sizes
         ${dtype}$, DIMENSION(PRODUCT(sizes)), INTENT(IN) :: array

         ASSOCIATE (blk => block%${dsuffix}$)
            block%data_type = ${dparam}$
            CALL allocate_any(blk%sizes, source=sizes)
            CALL allocate_any(blk%blk, source=array)
         END ASSOCIATE
      END SUBROUTINE
   #:endfor

   #:for dparam, dtype, dsuffix in dtype_float_list
      SUBROUTINE create_block_nodata_${dsuffix}$ (block, sizes)
      !! Create and allocate block, but no data.
         INTEGER, INTENT(IN), DIMENSION(:)       :: sizes
         TYPE(block_nd_${dsuffix}$), INTENT(OUT) :: block
         CALL allocate_any(block%sizes, source=sizes)
         ALLOCATE (block%blk(PRODUCT(sizes)))
      END SUBROUTINE
   #:endfor

   #:for dparam, dtype, dsuffix in dtype_float_list
      SUBROUTINE destroy_block_${dsuffix}$ (block)
         TYPE(block_nd_${dsuffix}$), INTENT(INOUT) :: block
         DEALLOCATE (block%blk)
         DEALLOCATE (block%sizes)
      END SUBROUTINE
   #:endfor

   SUBROUTINE dbcsr_t_get_anyd_block(tensor, ind, block, found)
      !! Generic implementation of dbcsr_t_get_block (arbitrary tensor rank and arbitrary datatype)

      TYPE(dbcsr_t_type), INTENT(INOUT)            :: tensor
      INTEGER, DIMENSION(ndims_tensor(tensor)), &
         INTENT(IN)                                :: ind
         !! block index
      TYPE(block_nd), INTENT(OUT)                  :: block
         !! block to get
      LOGICAL, INTENT(OUT)                         :: found
         !! whether block was found

      SELECT CASE (dbcsr_t_get_data_type(tensor))
         #:for dparam, dtype, dsuffix in dtype_float_list
            CASE (${dparam}$)
            CALL dbcsr_t_get_anyd_block_${dsuffix}$ (tensor, ind, block, found)
         #:endfor
      END SELECT
   END SUBROUTINE

   SUBROUTINE dbcsr_t_put_anyd_block(tensor, ind, block, summation, scale)
      !! Generic implementation of dbcsr_t_put_block (arbitrary tensor rank and arbitrary datatype)

      TYPE(dbcsr_t_type), INTENT(INOUT)            :: tensor
      INTEGER, DIMENSION(ndims_tensor(tensor)), &
         INTENT(IN)                                :: ind
         !! block index
      TYPE(block_nd), INTENT(IN)                   :: block
         !! block to put
      LOGICAL, INTENT(IN), OPTIONAL                :: summation
         !! whether block should be summed to existing block
      TYPE(dbcsr_scalar_type), INTENT(IN), OPTIONAL :: scale
         !! scaling factor

      SELECT CASE (block%data_type)
         #:for dparam, dtype, dsuffix in dtype_float_list
            CASE (${dparam}$)
            IF (.NOT. PRESENT(scale)) THEN
               CALL dbcsr_t_put_anyd_block_${dsuffix}$ (tensor, ind, block%${dsuffix}$, summation)
            ELSE
               CALL dbcsr_t_put_anyd_block_${dsuffix}$ (tensor, ind, block%${dsuffix}$, summation, scale=scale%${dsuffix}$)
            END IF
         #:endfor
      END SELECT

   END SUBROUTINE

   #:for dparam, dtype, dsuffix in dtype_float_list
      SUBROUTINE dbcsr_t_put_anyd_block_${dsuffix}$ (tensor, ind, block, summation, scale)
      !! Generic implementation of dbcsr_t_put_block, template for datatype

         TYPE(block_nd_${dsuffix}$), INTENT(IN)       :: block
         !! block to put
         TYPE(dbcsr_t_type), INTENT(INOUT)            :: tensor
         LOGICAL, INTENT(IN), OPTIONAL                :: summation
         !! whether block should be summed to existing block
         ${dtype}$, INTENT(IN), OPTIONAL :: scale
         !! scaling factor
         INTEGER, DIMENSION(ndims_tensor(tensor)), &
            INTENT(IN)                                :: ind
         !! block index

         SELECT CASE (ndims_tensor(tensor))
            #:for ndim in ndims
               CASE (${ndim}$)
               CALL dbcsr_t_put_${ndim}$d_block_${dsuffix}$ (tensor, ind, block%sizes, block%blk, summation=summation, scale=scale)
            #:endfor
         END SELECT
      END SUBROUTINE
   #:endfor

   #:for dparam, dtype, dsuffix in dtype_float_list
      SUBROUTINE dbcsr_t_get_anyd_block_${dsuffix}$ (tensor, ind, block, found)
      !! Generic implementation of dbcsr_t_get_block (arbitrary tensor rank)

         TYPE(block_nd), INTENT(OUT)                  :: block
         !! block to get
         LOGICAL, INTENT(OUT)                         :: found
         !! whether block was found
         TYPE(dbcsr_t_type), INTENT(INOUT)            :: tensor
         INTEGER, DIMENSION(ndims_tensor(tensor)), &
            INTENT(IN)                                :: ind
         !! block index
         INTEGER, DIMENSION(ndims_tensor(tensor))    :: blk_size
         ${dtype}$, DIMENSION(:), ALLOCATABLE         :: block_arr

         CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
         ALLOCATE (block_arr(PRODUCT(blk_size)))

         SELECT CASE (ndims_tensor(tensor))
            #:for ndim in ndims
               CASE (${ndim}$)
               CALL dbcsr_t_get_${ndim}$d_block_${dsuffix}$ (tensor, ind, blk_size, block_arr, found)
            #:endfor
         END SELECT
         CALL create_block(block, blk_size, block_arr)
      END SUBROUTINE
   #:endfor

   #:for dparam, dtype, dsuffix in dtype_float_list
      #:for ndim in ndims
         SUBROUTINE dbcsr_t_put_${ndim}$d_block_${dsuffix}$ (tensor, ind, sizes, block, summation, scale)
      !! Template for dbcsr_t_put_block.

            TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
            INTEGER, DIMENSION(${ndim}$), INTENT(IN) :: ind
         !! block index
            INTEGER, DIMENSION(${ndim}$), INTENT(IN) :: sizes
         !! block size
            ${dtype}$, DIMENSION(${arrlist("sizes", nmax=ndim)}$), &
               INTENT(IN), TARGET                                 :: block
         !! block to put
            LOGICAL, INTENT(IN), OPTIONAL                         :: summation
         !! whether block should be summed to existing block
            ${dtype}$, INTENT(IN), OPTIONAL                       :: scale
         !! scaling factor
            INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
            INTEGER, DIMENSION(2)                                 :: shape_2d
            ${dtype}$, POINTER, DIMENSION(:, :)                   :: block_2d
            INTEGER, DIMENSION(${ndim}$)                          :: shape_nd
            LOGICAL :: found, new_block
            ${dtype}$, DIMENSION(${arrlist("sizes", nmax=ndim)}$) :: block_check

            LOGICAL, PARAMETER :: debug = .FALSE.
            INTEGER :: i

            new_block = .FALSE.

            IF (debug) THEN
               CALL dbcsr_t_get_block(tensor, ind, sizes, block_check, found=found)
               DBCSR_ASSERT(found)
            END IF

            ASSOCIATE (map_nd => tensor%nd_index_blk%map_nd, &
                       map1_2d => tensor%nd_index_blk%map1_2d, &
                       map2_2d => tensor%nd_index_blk%map2_2d)

               shape_2d = [PRODUCT(sizes(map1_2d)), PRODUCT(sizes(map2_2d))]

               IF (ALL([map1_2d, map2_2d] == (/(i, i=1, ${ndim}$)/))) THEN
                  ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
                  block_2d(1:shape_2d(1), 1:shape_2d(2)) => block(${shape_colon(ndim)}$)
               ELSE
                  ! need reshape due to rank reordering
                  ALLOCATE (block_2d(shape_2d(1), shape_2d(2)))
                  new_block = .TRUE.
                  shape_nd(map_nd) = sizes
                  block_2d(:, :) = RESHAPE(RESHAPE(block, SHAPE=shape_nd, order=map_nd), SHAPE=shape_2d)
               END IF

               ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

            END ASSOCIATE

            CALL dbcsr_tas_put_block(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d, summation=summation, &
                                     scale=scale)

            IF (new_block) DEALLOCATE (block_2d)

         END SUBROUTINE
      #:endfor
   #:endfor

   #:for dparam, dtype, dsuffix in dtype_float_list
      #:for ndim in ndims
         SUBROUTINE dbcsr_t_allocate_and_get_${ndim}$d_block_${dsuffix}$ (tensor, ind, block, found)
      !! allocate and get block

            TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
            INTEGER, DIMENSION(${ndim}$), INTENT(IN)  :: ind
         !! block index
            ${dtype}$, DIMENSION(${shape_colon(ndim)}$), &
               ALLOCATABLE, INTENT(OUT)                           :: block
         !! block to get
            LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found
            INTEGER, DIMENSION(${ndim}$)                          :: blk_size

            CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
            CALL allocate_any(block, shape_spec=blk_size)
            CALL dbcsr_t_get_${ndim}$d_block_${dsuffix}$ (tensor, ind, blk_size, block, found)

         END SUBROUTINE
      #:endfor
   #:endfor

   #:for dparam, dtype, dsuffix in dtype_float_list
      #:for ndim in ndims
         SUBROUTINE dbcsr_t_get_${ndim}$d_block_${dsuffix}$ (tensor, ind, sizes, block, found)
      !! Template for dbcsr_t_get_block.

            TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
            INTEGER, DIMENSION(${ndim}$), INTENT(IN) :: ind
         !! block index
            INTEGER, DIMENSION(${ndim}$), INTENT(IN) :: sizes
         !! block size
            ${dtype}$, DIMENSION(${arrlist("sizes", nmax=ndim)}$), &
               INTENT(OUT)                                        :: block
         !! block to get
            LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found

            INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
            ${dtype}$, DIMENSION(:, :), POINTER, CONTIGUOUS       :: block_2d_ptr
            LOGICAL                                               :: tr
            INTEGER                                               :: i
            ${dtype}$, DIMENSION(${shape_colon(ndim)}$), POINTER  :: block_ptr

            NULLIFY (block_2d_ptr)

            ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

            ASSOCIATE (map1_2d => tensor%nd_index_blk%map1_2d, &
                       map2_2d => tensor%nd_index_blk%map2_2d)

               CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)
               DBCSR_ASSERT(.NOT. tr)

               IF (found) THEN
                  IF (ALL([map1_2d, map2_2d] == (/(i, i=1, ${ndim}$)/))) THEN
                     ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
                     block_ptr(${shape_explicit('block', ndim)}$) => block_2d_ptr(:, :)
                     block(${shape_colon(ndim)}$) = block_ptr(${shape_colon(ndim)}$)
                  ELSE
                     ! need reshape due to rank reordering
                     block(${shape_colon(ndim)}$) = RESHAPE(block_2d_ptr, SHAPE=SHAPE(block), ORDER=[map1_2d, map2_2d])
                  END IF
               END IF

            END ASSOCIATE

         END SUBROUTINE
      #:endfor
   #:endfor

END MODULE
