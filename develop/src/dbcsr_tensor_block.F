# 1 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F" 1
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

# 1 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp" 1
# 9 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp"

# 241 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp"
# 14 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F" 2
# 15 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 16 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"

   USE dbcsr_allocate_wrap, ONLY: &
      allocate_any
   USE dbcsr_api, ONLY: &
      dbcsr_type_real_4, dbcsr_type_complex_8, dbcsr_type_complex_4, dbcsr_type_real_8, dbcsr_iterator_type,&
# 20 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
          & dbcsr_iterator_blocks_left, &
      dbcsr_iterator_next_block, dbcsr_iterator_start, dbcsr_iterator_stop, dbcsr_type, &
      dbcsr_reserve_blocks, dbcsr_scalar_type, dbcsr_finalize, dbcsr_get_num_blocks, &
      dbcsr_type_no_symmetry, dbcsr_get_matrix_type, dbcsr_desymmetrize, dbcsr_release, &
      dbcsr_has_symmetry
   USE dbcsr_tas_types, ONLY: &
      dbcsr_tas_iterator
   USE dbcsr_tas_base, ONLY: &
      dbcsr_tas_iterator_next_block, dbcsr_tas_iterator_blocks_left, dbcsr_tas_iterator_start, &
      dbcsr_tas_iterator_stop, dbcsr_tas_get_block_p, dbcsr_tas_put_block, dbcsr_tas_reserve_blocks
   USE dbcsr_kinds, ONLY: &
      real_4, real_8, int_8
   USE dbcsr_tensor_index, ONLY: &
      nd_to_2d_mapping, ndims_mapping, get_nd_indices_tensor, destroy_nd_to_2d_mapping, get_2d_indices_tensor, &
      create_nd_to_2d_mapping
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
      TYPE(dbcsr_tas_iterator)      :: iter
      TYPE(nd_to_2d_mapping)      :: nd_index_blk
      TYPE(nd_to_2d_mapping)      :: nd_index
      TYPE(array_list)            :: blk_sizes, blk_offsets
   END TYPE dbcsr_t_iterator_type

# 74 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   PUBLIC :: block_nd_r_dp
# 74 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   PUBLIC :: block_nd_r_sp
# 74 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   PUBLIC :: block_nd_c_dp
# 74 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   PUBLIC :: block_nd_c_sp
# 76 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"

# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   TYPE block_nd_r_dp
      INTEGER, DIMENSION(:), ALLOCATABLE   :: sizes
      REAL(kind=real_8), DIMENSION(:), ALLOCATABLE :: blk
   END TYPE

# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   TYPE block_nd_r_sp
      INTEGER, DIMENSION(:), ALLOCATABLE   :: sizes
      REAL(kind=real_4), DIMENSION(:), ALLOCATABLE :: blk
   END TYPE

# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   TYPE block_nd_c_dp
      INTEGER, DIMENSION(:), ALLOCATABLE   :: sizes
      COMPLEX(kind=real_8), DIMENSION(:), ALLOCATABLE :: blk
   END TYPE

# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   TYPE block_nd_c_sp
      INTEGER, DIMENSION(:), ALLOCATABLE   :: sizes
      COMPLEX(kind=real_4), DIMENSION(:), ALLOCATABLE :: blk
   END TYPE

# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"

   TYPE block_nd
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      TYPE(block_nd_r_dp) :: r_dp
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      TYPE(block_nd_r_sp) :: r_sp
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      TYPE(block_nd_c_dp) :: c_dp
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      TYPE(block_nd_c_sp) :: c_sp
# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      INTEGER          :: data_type = -1
   END TYPE

   INTERFACE create_block
# 94 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE create_block_data_r_dp
# 94 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE create_block_data_r_sp
# 94 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE create_block_data_c_dp
# 94 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE create_block_data_c_sp
# 96 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE create_block_nodata
   END INTERFACE

   INTERFACE dbcsr_t_put_block
# 101 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 102 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_put_2d_block_r_dp
# 102 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_put_3d_block_r_dp
# 102 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_put_4d_block_r_dp
# 104 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 101 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 102 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_put_2d_block_r_sp
# 102 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_put_3d_block_r_sp
# 102 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_put_4d_block_r_sp
# 104 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 101 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 102 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_put_2d_block_c_dp
# 102 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_put_3d_block_c_dp
# 102 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_put_4d_block_c_dp
# 104 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 101 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 102 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_put_2d_block_c_sp
# 102 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_put_3d_block_c_sp
# 102 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_put_4d_block_c_sp
# 104 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 105 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_put_anyd_block
   END INTERFACE

   INTERFACE dbcsr_t_get_block
# 110 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 111 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_get_2d_block_r_dp
      MODULE PROCEDURE dbcsr_t_allocate_and_get_2d_block_r_dp
# 111 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_get_3d_block_r_dp
      MODULE PROCEDURE dbcsr_t_allocate_and_get_3d_block_r_dp
# 111 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_get_4d_block_r_dp
      MODULE PROCEDURE dbcsr_t_allocate_and_get_4d_block_r_dp
# 114 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 110 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 111 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_get_2d_block_r_sp
      MODULE PROCEDURE dbcsr_t_allocate_and_get_2d_block_r_sp
# 111 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_get_3d_block_r_sp
      MODULE PROCEDURE dbcsr_t_allocate_and_get_3d_block_r_sp
# 111 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_get_4d_block_r_sp
      MODULE PROCEDURE dbcsr_t_allocate_and_get_4d_block_r_sp
# 114 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 110 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 111 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_get_2d_block_c_dp
      MODULE PROCEDURE dbcsr_t_allocate_and_get_2d_block_c_dp
# 111 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_get_3d_block_c_dp
      MODULE PROCEDURE dbcsr_t_allocate_and_get_3d_block_c_dp
# 111 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_get_4d_block_c_dp
      MODULE PROCEDURE dbcsr_t_allocate_and_get_4d_block_c_dp
# 114 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 110 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 111 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_get_2d_block_c_sp
      MODULE PROCEDURE dbcsr_t_allocate_and_get_2d_block_c_sp
# 111 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_get_3d_block_c_sp
      MODULE PROCEDURE dbcsr_t_allocate_and_get_3d_block_c_sp
# 111 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      MODULE PROCEDURE dbcsr_t_get_4d_block_c_sp
      MODULE PROCEDURE dbcsr_t_allocate_and_get_4d_block_c_sp
# 114 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 115 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
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
# 137 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_real_8)
         CALL create_block_nodata_r_dp (block%r_dp, sizes)
# 137 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_real_4)
         CALL create_block_nodata_r_sp (block%r_sp, sizes)
# 137 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_complex_8)
         CALL create_block_nodata_c_dp (block%c_dp, sizes)
# 137 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_complex_4)
         CALL create_block_nodata_c_sp (block%c_sp, sizes)
# 140 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      END SELECT
   END SUBROUTINE

   SUBROUTINE destroy_block(block)
      !! Destroy block
      TYPE(block_nd), INTENT(INOUT) :: block

      SELECT CASE (block%data_type)
# 149 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_real_8)
         CALL destroy_block_r_dp (block%r_dp)
# 149 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_real_4)
         CALL destroy_block_r_sp (block%r_sp)
# 149 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_complex_8)
         CALL destroy_block_c_dp (block%c_dp)
# 149 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_complex_4)
         CALL destroy_block_c_sp (block%c_sp)
# 152 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      END SELECT

   END SUBROUTINE

   FUNCTION block_size(block)
      !! block size
      TYPE(block_nd), INTENT(IN)         :: block
      INTEGER, ALLOCATABLE, DIMENSION(:) :: block_size

      block_size = 0 ! invalid
      SELECT CASE (block%data_type)
# 164 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_real_8)
         CALL allocate_any(block_size, source=block%r_dp%sizes)
# 164 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_real_4)
         CALL allocate_any(block_size, source=block%r_sp%sizes)
# 164 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_complex_8)
         CALL allocate_any(block_size, source=block%c_dp%sizes)
# 164 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_complex_4)
         CALL allocate_any(block_size, source=block%c_sp%sizes)
# 167 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
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
# 256 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      IF (ndims_tensor(tensor) == 2) THEN
         CALL dbcsr_t_reserve_blocks(tensor, blk_ind(:,1), blk_ind(:,2))
      ENDIF
# 256 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      IF (ndims_tensor(tensor) == 3) THEN
         CALL dbcsr_t_reserve_blocks(tensor, blk_ind(:,1), blk_ind(:,2), blk_ind(:,3))
      ENDIF
# 256 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      IF (ndims_tensor(tensor) == 4) THEN
         CALL dbcsr_t_reserve_blocks(tensor, blk_ind(:,1), blk_ind(:,2), blk_ind(:,3), blk_ind(:,4))
      ENDIF
# 260 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CALL timestop(handle)

   END SUBROUTINE

   SUBROUTINE dbcsr_t_reserve_blocks_index(tensor, blk_ind_1, blk_ind_2, blk_ind_3, blk_ind_4)
      !! reserve tensor blocks using block indices
      TYPE(dbcsr_t_type), INTENT(INOUT)           :: tensor
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL :: blk_ind_1, blk_ind_2, blk_ind_3, blk_ind_4
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
                             blk_ind_1, blk_ind_2, blk_ind_3, blk_ind_4)

      nblk_tmp(:) = sizes_of_arrays(blks)
      nblk = nblk_tmp(1)
      ALLOCATE (cols(nblk), rows(nblk))
      DO iblk = 1, nblk
         iblk_nd(:) = iblk
         ind_nd(:) = get_array_elements(blks, iblk_nd)
         ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind_nd)
         rows(iblk) = ind_2d(1); cols(iblk) = ind_2d(2)
      ENDDO

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
      ENDIF

      nblk = dbcsr_get_num_blocks(matrix_in_desym)
      ALLOCATE (blk_ind_1(nblk), blk_ind_2(nblk))
      CALL dbcsr_iterator_start(iter, matrix_in_desym)
      DO iblk = 1, nblk
         CALL dbcsr_iterator_next_block(iter, ind_2d(1), ind_2d(2), blk)
         blk_ind_1(iblk) = ind_2d(1); blk_ind_2(iblk) = ind_2d(2)
      ENDDO
      CALL dbcsr_iterator_stop(iter)

      CALL dbcsr_t_reserve_blocks(tensor_out, blk_ind_1, blk_ind_2)

      IF (dbcsr_has_symmetry(matrix_in)) THEN
         CALL dbcsr_release(matrix_in_desym)
         DEALLOCATE (matrix_in_desym)
      ENDIF

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
         ENDIF

         iblk = iblk + 1
         blk_ind_1(iblk) = ind_2d(1); blk_ind_2(iblk) = ind_2d(2)
      ENDDO
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
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)

   END SUBROUTINE

# 416 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE create_block_data_r_dp (block, sizes, array)
      !! Create block from array, array can be n-dimensional.
      TYPE(block_nd), INTENT(OUT)                       :: block
      INTEGER, DIMENSION(:), INTENT(IN)                 :: sizes
      REAL(kind=real_8), DIMENSION(PRODUCT(sizes)), INTENT(IN) :: array

      ASSOCIATE (blk => block%r_dp)
         block%data_type = dbcsr_type_real_8
         CALL allocate_any(blk%sizes, source=sizes)
         CALL allocate_any(blk%blk, source=array)
      END ASSOCIATE
   END SUBROUTINE
# 416 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE create_block_data_r_sp (block, sizes, array)
      !! Create block from array, array can be n-dimensional.
      TYPE(block_nd), INTENT(OUT)                       :: block
      INTEGER, DIMENSION(:), INTENT(IN)                 :: sizes
      REAL(kind=real_4), DIMENSION(PRODUCT(sizes)), INTENT(IN) :: array

      ASSOCIATE (blk => block%r_sp)
         block%data_type = dbcsr_type_real_4
         CALL allocate_any(blk%sizes, source=sizes)
         CALL allocate_any(blk%blk, source=array)
      END ASSOCIATE
   END SUBROUTINE
# 416 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE create_block_data_c_dp (block, sizes, array)
      !! Create block from array, array can be n-dimensional.
      TYPE(block_nd), INTENT(OUT)                       :: block
      INTEGER, DIMENSION(:), INTENT(IN)                 :: sizes
      COMPLEX(kind=real_8), DIMENSION(PRODUCT(sizes)), INTENT(IN) :: array

      ASSOCIATE (blk => block%c_dp)
         block%data_type = dbcsr_type_complex_8
         CALL allocate_any(blk%sizes, source=sizes)
         CALL allocate_any(blk%blk, source=array)
      END ASSOCIATE
   END SUBROUTINE
# 416 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE create_block_data_c_sp (block, sizes, array)
      !! Create block from array, array can be n-dimensional.
      TYPE(block_nd), INTENT(OUT)                       :: block
      INTEGER, DIMENSION(:), INTENT(IN)                 :: sizes
      COMPLEX(kind=real_4), DIMENSION(PRODUCT(sizes)), INTENT(IN) :: array

      ASSOCIATE (blk => block%c_sp)
         block%data_type = dbcsr_type_complex_4
         CALL allocate_any(blk%sizes, source=sizes)
         CALL allocate_any(blk%blk, source=array)
      END ASSOCIATE
   END SUBROUTINE
# 429 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"

# 431 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE create_block_nodata_r_dp (block, sizes)
      !! Create and allocate block, but no data.
      INTEGER, INTENT(IN), DIMENSION(:)       :: sizes
      TYPE(block_nd_r_dp), INTENT(OUT) :: block
      CALL allocate_any(block%sizes, source=sizes)
      ALLOCATE (block%blk(PRODUCT(sizes)))
   END SUBROUTINE
# 431 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE create_block_nodata_r_sp (block, sizes)
      !! Create and allocate block, but no data.
      INTEGER, INTENT(IN), DIMENSION(:)       :: sizes
      TYPE(block_nd_r_sp), INTENT(OUT) :: block
      CALL allocate_any(block%sizes, source=sizes)
      ALLOCATE (block%blk(PRODUCT(sizes)))
   END SUBROUTINE
# 431 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE create_block_nodata_c_dp (block, sizes)
      !! Create and allocate block, but no data.
      INTEGER, INTENT(IN), DIMENSION(:)       :: sizes
      TYPE(block_nd_c_dp), INTENT(OUT) :: block
      CALL allocate_any(block%sizes, source=sizes)
      ALLOCATE (block%blk(PRODUCT(sizes)))
   END SUBROUTINE
# 431 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE create_block_nodata_c_sp (block, sizes)
      !! Create and allocate block, but no data.
      INTEGER, INTENT(IN), DIMENSION(:)       :: sizes
      TYPE(block_nd_c_sp), INTENT(OUT) :: block
      CALL allocate_any(block%sizes, source=sizes)
      ALLOCATE (block%blk(PRODUCT(sizes)))
   END SUBROUTINE
# 439 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"

# 441 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE destroy_block_r_dp (block)
      TYPE(block_nd_r_dp), INTENT(INOUT) :: block
      DEALLOCATE (block%blk)
      DEALLOCATE (block%sizes)
   END SUBROUTINE
# 441 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE destroy_block_r_sp (block)
      TYPE(block_nd_r_sp), INTENT(INOUT) :: block
      DEALLOCATE (block%blk)
      DEALLOCATE (block%sizes)
   END SUBROUTINE
# 441 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE destroy_block_c_dp (block)
      TYPE(block_nd_c_dp), INTENT(INOUT) :: block
      DEALLOCATE (block%blk)
      DEALLOCATE (block%sizes)
   END SUBROUTINE
# 441 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE destroy_block_c_sp (block)
      TYPE(block_nd_c_sp), INTENT(INOUT) :: block
      DEALLOCATE (block%blk)
      DEALLOCATE (block%sizes)
   END SUBROUTINE
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"

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
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_real_8)
         CALL dbcsr_t_get_anyd_block_r_dp (tensor, ind, block, found)
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_real_4)
         CALL dbcsr_t_get_anyd_block_r_sp (tensor, ind, block, found)
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_complex_8)
         CALL dbcsr_t_get_anyd_block_c_dp (tensor, ind, block, found)
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_complex_4)
         CALL dbcsr_t_get_anyd_block_c_sp (tensor, ind, block, found)
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
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
# 484 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_real_8)
         IF (.NOT. PRESENT(scale)) THEN
            CALL dbcsr_t_put_anyd_block_r_dp (tensor, ind, block%r_dp, summation)
         ELSE
            CALL dbcsr_t_put_anyd_block_r_dp (tensor, ind, block%r_dp, summation, scale=scale%r_dp)
         ENDIF
# 484 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_real_4)
         IF (.NOT. PRESENT(scale)) THEN
            CALL dbcsr_t_put_anyd_block_r_sp (tensor, ind, block%r_sp, summation)
         ELSE
            CALL dbcsr_t_put_anyd_block_r_sp (tensor, ind, block%r_sp, summation, scale=scale%r_sp)
         ENDIF
# 484 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_complex_8)
         IF (.NOT. PRESENT(scale)) THEN
            CALL dbcsr_t_put_anyd_block_c_dp (tensor, ind, block%c_dp, summation)
         ELSE
            CALL dbcsr_t_put_anyd_block_c_dp (tensor, ind, block%c_dp, summation, scale=scale%c_dp)
         ENDIF
# 484 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (dbcsr_type_complex_4)
         IF (.NOT. PRESENT(scale)) THEN
            CALL dbcsr_t_put_anyd_block_c_sp (tensor, ind, block%c_sp, summation)
         ELSE
            CALL dbcsr_t_put_anyd_block_c_sp (tensor, ind, block%c_sp, summation, scale=scale%c_sp)
         ENDIF
# 491 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      END SELECT

   END SUBROUTINE

# 496 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_put_anyd_block_r_dp (tensor, ind, block, summation, scale)
      !! Generic implementation of dbcsr_t_put_block, template for datatype

      TYPE(block_nd_r_dp), INTENT(IN)       :: block
         !! block to put
      TYPE(dbcsr_t_type), INTENT(INOUT)            :: tensor
      LOGICAL, INTENT(IN), OPTIONAL                :: summation
         !! whether block should be summed to existing block
      REAL(kind=real_8), INTENT(IN), OPTIONAL :: scale
         !! scaling factor
      INTEGER, DIMENSION(ndims_tensor(tensor)), &
         INTENT(IN)                                :: ind
         !! block index

      SELECT CASE (ndims_tensor(tensor))
# 512 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (2)
         CALL dbcsr_t_put_2d_block_r_dp (tensor, ind, block%sizes, block%blk, summation=summation, scale=scale)
# 512 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (3)
         CALL dbcsr_t_put_3d_block_r_dp (tensor, ind, block%sizes, block%blk, summation=summation, scale=scale)
# 512 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (4)
         CALL dbcsr_t_put_4d_block_r_dp (tensor, ind, block%sizes, block%blk, summation=summation, scale=scale)
# 515 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      END SELECT
   END SUBROUTINE
# 496 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_put_anyd_block_r_sp (tensor, ind, block, summation, scale)
      !! Generic implementation of dbcsr_t_put_block, template for datatype

      TYPE(block_nd_r_sp), INTENT(IN)       :: block
         !! block to put
      TYPE(dbcsr_t_type), INTENT(INOUT)            :: tensor
      LOGICAL, INTENT(IN), OPTIONAL                :: summation
         !! whether block should be summed to existing block
      REAL(kind=real_4), INTENT(IN), OPTIONAL :: scale
         !! scaling factor
      INTEGER, DIMENSION(ndims_tensor(tensor)), &
         INTENT(IN)                                :: ind
         !! block index

      SELECT CASE (ndims_tensor(tensor))
# 512 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (2)
         CALL dbcsr_t_put_2d_block_r_sp (tensor, ind, block%sizes, block%blk, summation=summation, scale=scale)
# 512 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (3)
         CALL dbcsr_t_put_3d_block_r_sp (tensor, ind, block%sizes, block%blk, summation=summation, scale=scale)
# 512 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (4)
         CALL dbcsr_t_put_4d_block_r_sp (tensor, ind, block%sizes, block%blk, summation=summation, scale=scale)
# 515 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      END SELECT
   END SUBROUTINE
# 496 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_put_anyd_block_c_dp (tensor, ind, block, summation, scale)
      !! Generic implementation of dbcsr_t_put_block, template for datatype

      TYPE(block_nd_c_dp), INTENT(IN)       :: block
         !! block to put
      TYPE(dbcsr_t_type), INTENT(INOUT)            :: tensor
      LOGICAL, INTENT(IN), OPTIONAL                :: summation
         !! whether block should be summed to existing block
      COMPLEX(kind=real_8), INTENT(IN), OPTIONAL :: scale
         !! scaling factor
      INTEGER, DIMENSION(ndims_tensor(tensor)), &
         INTENT(IN)                                :: ind
         !! block index

      SELECT CASE (ndims_tensor(tensor))
# 512 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (2)
         CALL dbcsr_t_put_2d_block_c_dp (tensor, ind, block%sizes, block%blk, summation=summation, scale=scale)
# 512 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (3)
         CALL dbcsr_t_put_3d_block_c_dp (tensor, ind, block%sizes, block%blk, summation=summation, scale=scale)
# 512 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (4)
         CALL dbcsr_t_put_4d_block_c_dp (tensor, ind, block%sizes, block%blk, summation=summation, scale=scale)
# 515 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      END SELECT
   END SUBROUTINE
# 496 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_put_anyd_block_c_sp (tensor, ind, block, summation, scale)
      !! Generic implementation of dbcsr_t_put_block, template for datatype

      TYPE(block_nd_c_sp), INTENT(IN)       :: block
         !! block to put
      TYPE(dbcsr_t_type), INTENT(INOUT)            :: tensor
      LOGICAL, INTENT(IN), OPTIONAL                :: summation
         !! whether block should be summed to existing block
      COMPLEX(kind=real_4), INTENT(IN), OPTIONAL :: scale
         !! scaling factor
      INTEGER, DIMENSION(ndims_tensor(tensor)), &
         INTENT(IN)                                :: ind
         !! block index

      SELECT CASE (ndims_tensor(tensor))
# 512 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (2)
         CALL dbcsr_t_put_2d_block_c_sp (tensor, ind, block%sizes, block%blk, summation=summation, scale=scale)
# 512 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (3)
         CALL dbcsr_t_put_3d_block_c_sp (tensor, ind, block%sizes, block%blk, summation=summation, scale=scale)
# 512 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (4)
         CALL dbcsr_t_put_4d_block_c_sp (tensor, ind, block%sizes, block%blk, summation=summation, scale=scale)
# 515 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      END SELECT
   END SUBROUTINE
# 518 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"

# 520 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_get_anyd_block_r_dp (tensor, ind, block, found)
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
      REAL(kind=real_8), DIMENSION(:), ALLOCATABLE         :: block_arr

      CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
      ALLOCATE (block_arr(PRODUCT(blk_size)))

      SELECT CASE (ndims_tensor(tensor))
# 539 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (2)
         CALL dbcsr_t_get_2d_block_r_dp (tensor, ind, blk_size, block_arr, found)
# 539 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (3)
         CALL dbcsr_t_get_3d_block_r_dp (tensor, ind, blk_size, block_arr, found)
# 539 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (4)
         CALL dbcsr_t_get_4d_block_r_dp (tensor, ind, blk_size, block_arr, found)
# 542 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      END SELECT
      CALL create_block(block, blk_size, block_arr)
   END SUBROUTINE
# 520 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_get_anyd_block_r_sp (tensor, ind, block, found)
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
      REAL(kind=real_4), DIMENSION(:), ALLOCATABLE         :: block_arr

      CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
      ALLOCATE (block_arr(PRODUCT(blk_size)))

      SELECT CASE (ndims_tensor(tensor))
# 539 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (2)
         CALL dbcsr_t_get_2d_block_r_sp (tensor, ind, blk_size, block_arr, found)
# 539 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (3)
         CALL dbcsr_t_get_3d_block_r_sp (tensor, ind, blk_size, block_arr, found)
# 539 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (4)
         CALL dbcsr_t_get_4d_block_r_sp (tensor, ind, blk_size, block_arr, found)
# 542 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      END SELECT
      CALL create_block(block, blk_size, block_arr)
   END SUBROUTINE
# 520 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_get_anyd_block_c_dp (tensor, ind, block, found)
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
      COMPLEX(kind=real_8), DIMENSION(:), ALLOCATABLE         :: block_arr

      CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
      ALLOCATE (block_arr(PRODUCT(blk_size)))

      SELECT CASE (ndims_tensor(tensor))
# 539 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (2)
         CALL dbcsr_t_get_2d_block_c_dp (tensor, ind, blk_size, block_arr, found)
# 539 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (3)
         CALL dbcsr_t_get_3d_block_c_dp (tensor, ind, blk_size, block_arr, found)
# 539 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (4)
         CALL dbcsr_t_get_4d_block_c_dp (tensor, ind, blk_size, block_arr, found)
# 542 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      END SELECT
      CALL create_block(block, blk_size, block_arr)
   END SUBROUTINE
# 520 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_get_anyd_block_c_sp (tensor, ind, block, found)
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
      COMPLEX(kind=real_4), DIMENSION(:), ALLOCATABLE         :: block_arr

      CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
      ALLOCATE (block_arr(PRODUCT(blk_size)))

      SELECT CASE (ndims_tensor(tensor))
# 539 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (2)
         CALL dbcsr_t_get_2d_block_c_sp (tensor, ind, blk_size, block_arr, found)
# 539 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (3)
         CALL dbcsr_t_get_3d_block_c_sp (tensor, ind, blk_size, block_arr, found)
# 539 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      CASE (4)
         CALL dbcsr_t_get_4d_block_c_sp (tensor, ind, blk_size, block_arr, found)
# 542 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
      END SELECT
      CALL create_block(block, blk_size, block_arr)
   END SUBROUTINE
# 546 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"

# 548 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 549 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_put_2d_block_r_dp (tensor, ind, sizes, block, summation, scale)
      !! Template for dbcsr_t_put_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(2), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(2), INTENT(IN) :: sizes
         !! block size
      REAL(kind=real_8), DIMENSION(sizes(1), sizes(2)), &
         INTENT(IN), TARGET                                 :: block
         !! block to put
      LOGICAL, INTENT(IN), OPTIONAL                         :: summation
         !! whether block should be summed to existing block
      REAL(kind=real_8), INTENT(IN), OPTIONAL                       :: scale
         !! scaling factor
      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      INTEGER, DIMENSION(2)                                 :: shape_2d
      REAL(kind=real_8), POINTER, DIMENSION(:, :)                   :: block_2d
      INTEGER, DIMENSION(2)                          :: shape_nd
      LOGICAL :: found, new_block
      REAL(kind=real_8), DIMENSION(sizes(1), sizes(2)) :: block_check

      LOGICAL, PARAMETER :: debug = .FALSE.
      INTEGER :: i

      new_block = .FALSE.

      IF (debug) THEN
         CALL dbcsr_t_get_block(tensor, ind, sizes, block_check, found=found)
         DBCSR_ASSERT(found)
      ENDIF

      ASSOCIATE (map_nd => tensor%nd_index_blk%map_nd, &
                 map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         shape_2d = [PRODUCT(sizes(map1_2d)), PRODUCT(sizes(map2_2d))]

         IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 2)/))) THEN
            ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
            block_2d(1:shape_2d(1), 1:shape_2d(2)) => block(:,:)
         ELSE
            ! need reshape due to rank reordering
            ALLOCATE (block_2d(shape_2d(1), shape_2d(2)))
            new_block = .TRUE.
            shape_nd(map_nd) = sizes
            block_2d(:, :) = RESHAPE(RESHAPE(block, SHAPE=shape_nd, order=map_nd), SHAPE=shape_2d)
         ENDIF

         ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      END ASSOCIATE

      CALL dbcsr_tas_put_block(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d, summation=summation, &
                               scale=scale)

      IF (new_block) DEALLOCATE (block_2d)

   END SUBROUTINE
# 549 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_put_3d_block_r_dp (tensor, ind, sizes, block, summation, scale)
      !! Template for dbcsr_t_put_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(3), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(3), INTENT(IN) :: sizes
         !! block size
      REAL(kind=real_8), DIMENSION(sizes(1), sizes(2), sizes(3)), &
         INTENT(IN), TARGET                                 :: block
         !! block to put
      LOGICAL, INTENT(IN), OPTIONAL                         :: summation
         !! whether block should be summed to existing block
      REAL(kind=real_8), INTENT(IN), OPTIONAL                       :: scale
         !! scaling factor
      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      INTEGER, DIMENSION(2)                                 :: shape_2d
      REAL(kind=real_8), POINTER, DIMENSION(:, :)                   :: block_2d
      INTEGER, DIMENSION(3)                          :: shape_nd
      LOGICAL :: found, new_block
      REAL(kind=real_8), DIMENSION(sizes(1), sizes(2), sizes(3)) :: block_check

      LOGICAL, PARAMETER :: debug = .FALSE.
      INTEGER :: i

      new_block = .FALSE.

      IF (debug) THEN
         CALL dbcsr_t_get_block(tensor, ind, sizes, block_check, found=found)
         DBCSR_ASSERT(found)
      ENDIF

      ASSOCIATE (map_nd => tensor%nd_index_blk%map_nd, &
                 map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         shape_2d = [PRODUCT(sizes(map1_2d)), PRODUCT(sizes(map2_2d))]

         IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 3)/))) THEN
            ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
            block_2d(1:shape_2d(1), 1:shape_2d(2)) => block(:,:,:)
         ELSE
            ! need reshape due to rank reordering
            ALLOCATE (block_2d(shape_2d(1), shape_2d(2)))
            new_block = .TRUE.
            shape_nd(map_nd) = sizes
            block_2d(:, :) = RESHAPE(RESHAPE(block, SHAPE=shape_nd, order=map_nd), SHAPE=shape_2d)
         ENDIF

         ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      END ASSOCIATE

      CALL dbcsr_tas_put_block(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d, summation=summation, &
                               scale=scale)

      IF (new_block) DEALLOCATE (block_2d)

   END SUBROUTINE
# 549 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_put_4d_block_r_dp (tensor, ind, sizes, block, summation, scale)
      !! Template for dbcsr_t_put_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(4), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(4), INTENT(IN) :: sizes
         !! block size
      REAL(kind=real_8), DIMENSION(sizes(1), sizes(2), sizes(3), sizes(4)), &
         INTENT(IN), TARGET                                 :: block
         !! block to put
      LOGICAL, INTENT(IN), OPTIONAL                         :: summation
         !! whether block should be summed to existing block
      REAL(kind=real_8), INTENT(IN), OPTIONAL                       :: scale
         !! scaling factor
      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      INTEGER, DIMENSION(2)                                 :: shape_2d
      REAL(kind=real_8), POINTER, DIMENSION(:, :)                   :: block_2d
      INTEGER, DIMENSION(4)                          :: shape_nd
      LOGICAL :: found, new_block
      REAL(kind=real_8), DIMENSION(sizes(1), sizes(2), sizes(3), sizes(4)) :: block_check

      LOGICAL, PARAMETER :: debug = .FALSE.
      INTEGER :: i

      new_block = .FALSE.

      IF (debug) THEN
         CALL dbcsr_t_get_block(tensor, ind, sizes, block_check, found=found)
         DBCSR_ASSERT(found)
      ENDIF

      ASSOCIATE (map_nd => tensor%nd_index_blk%map_nd, &
                 map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         shape_2d = [PRODUCT(sizes(map1_2d)), PRODUCT(sizes(map2_2d))]

         IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 4)/))) THEN
            ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
            block_2d(1:shape_2d(1), 1:shape_2d(2)) => block(:,:,:,:)
         ELSE
            ! need reshape due to rank reordering
            ALLOCATE (block_2d(shape_2d(1), shape_2d(2)))
            new_block = .TRUE.
            shape_nd(map_nd) = sizes
            block_2d(:, :) = RESHAPE(RESHAPE(block, SHAPE=shape_nd, order=map_nd), SHAPE=shape_2d)
         ENDIF

         ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      END ASSOCIATE

      CALL dbcsr_tas_put_block(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d, summation=summation, &
                               scale=scale)

      IF (new_block) DEALLOCATE (block_2d)

   END SUBROUTINE
# 609 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 548 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 549 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_put_2d_block_r_sp (tensor, ind, sizes, block, summation, scale)
      !! Template for dbcsr_t_put_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(2), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(2), INTENT(IN) :: sizes
         !! block size
      REAL(kind=real_4), DIMENSION(sizes(1), sizes(2)), &
         INTENT(IN), TARGET                                 :: block
         !! block to put
      LOGICAL, INTENT(IN), OPTIONAL                         :: summation
         !! whether block should be summed to existing block
      REAL(kind=real_4), INTENT(IN), OPTIONAL                       :: scale
         !! scaling factor
      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      INTEGER, DIMENSION(2)                                 :: shape_2d
      REAL(kind=real_4), POINTER, DIMENSION(:, :)                   :: block_2d
      INTEGER, DIMENSION(2)                          :: shape_nd
      LOGICAL :: found, new_block
      REAL(kind=real_4), DIMENSION(sizes(1), sizes(2)) :: block_check

      LOGICAL, PARAMETER :: debug = .FALSE.
      INTEGER :: i

      new_block = .FALSE.

      IF (debug) THEN
         CALL dbcsr_t_get_block(tensor, ind, sizes, block_check, found=found)
         DBCSR_ASSERT(found)
      ENDIF

      ASSOCIATE (map_nd => tensor%nd_index_blk%map_nd, &
                 map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         shape_2d = [PRODUCT(sizes(map1_2d)), PRODUCT(sizes(map2_2d))]

         IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 2)/))) THEN
            ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
            block_2d(1:shape_2d(1), 1:shape_2d(2)) => block(:,:)
         ELSE
            ! need reshape due to rank reordering
            ALLOCATE (block_2d(shape_2d(1), shape_2d(2)))
            new_block = .TRUE.
            shape_nd(map_nd) = sizes
            block_2d(:, :) = RESHAPE(RESHAPE(block, SHAPE=shape_nd, order=map_nd), SHAPE=shape_2d)
         ENDIF

         ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      END ASSOCIATE

      CALL dbcsr_tas_put_block(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d, summation=summation, &
                               scale=scale)

      IF (new_block) DEALLOCATE (block_2d)

   END SUBROUTINE
# 549 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_put_3d_block_r_sp (tensor, ind, sizes, block, summation, scale)
      !! Template for dbcsr_t_put_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(3), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(3), INTENT(IN) :: sizes
         !! block size
      REAL(kind=real_4), DIMENSION(sizes(1), sizes(2), sizes(3)), &
         INTENT(IN), TARGET                                 :: block
         !! block to put
      LOGICAL, INTENT(IN), OPTIONAL                         :: summation
         !! whether block should be summed to existing block
      REAL(kind=real_4), INTENT(IN), OPTIONAL                       :: scale
         !! scaling factor
      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      INTEGER, DIMENSION(2)                                 :: shape_2d
      REAL(kind=real_4), POINTER, DIMENSION(:, :)                   :: block_2d
      INTEGER, DIMENSION(3)                          :: shape_nd
      LOGICAL :: found, new_block
      REAL(kind=real_4), DIMENSION(sizes(1), sizes(2), sizes(3)) :: block_check

      LOGICAL, PARAMETER :: debug = .FALSE.
      INTEGER :: i

      new_block = .FALSE.

      IF (debug) THEN
         CALL dbcsr_t_get_block(tensor, ind, sizes, block_check, found=found)
         DBCSR_ASSERT(found)
      ENDIF

      ASSOCIATE (map_nd => tensor%nd_index_blk%map_nd, &
                 map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         shape_2d = [PRODUCT(sizes(map1_2d)), PRODUCT(sizes(map2_2d))]

         IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 3)/))) THEN
            ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
            block_2d(1:shape_2d(1), 1:shape_2d(2)) => block(:,:,:)
         ELSE
            ! need reshape due to rank reordering
            ALLOCATE (block_2d(shape_2d(1), shape_2d(2)))
            new_block = .TRUE.
            shape_nd(map_nd) = sizes
            block_2d(:, :) = RESHAPE(RESHAPE(block, SHAPE=shape_nd, order=map_nd), SHAPE=shape_2d)
         ENDIF

         ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      END ASSOCIATE

      CALL dbcsr_tas_put_block(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d, summation=summation, &
                               scale=scale)

      IF (new_block) DEALLOCATE (block_2d)

   END SUBROUTINE
# 549 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_put_4d_block_r_sp (tensor, ind, sizes, block, summation, scale)
      !! Template for dbcsr_t_put_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(4), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(4), INTENT(IN) :: sizes
         !! block size
      REAL(kind=real_4), DIMENSION(sizes(1), sizes(2), sizes(3), sizes(4)), &
         INTENT(IN), TARGET                                 :: block
         !! block to put
      LOGICAL, INTENT(IN), OPTIONAL                         :: summation
         !! whether block should be summed to existing block
      REAL(kind=real_4), INTENT(IN), OPTIONAL                       :: scale
         !! scaling factor
      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      INTEGER, DIMENSION(2)                                 :: shape_2d
      REAL(kind=real_4), POINTER, DIMENSION(:, :)                   :: block_2d
      INTEGER, DIMENSION(4)                          :: shape_nd
      LOGICAL :: found, new_block
      REAL(kind=real_4), DIMENSION(sizes(1), sizes(2), sizes(3), sizes(4)) :: block_check

      LOGICAL, PARAMETER :: debug = .FALSE.
      INTEGER :: i

      new_block = .FALSE.

      IF (debug) THEN
         CALL dbcsr_t_get_block(tensor, ind, sizes, block_check, found=found)
         DBCSR_ASSERT(found)
      ENDIF

      ASSOCIATE (map_nd => tensor%nd_index_blk%map_nd, &
                 map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         shape_2d = [PRODUCT(sizes(map1_2d)), PRODUCT(sizes(map2_2d))]

         IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 4)/))) THEN
            ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
            block_2d(1:shape_2d(1), 1:shape_2d(2)) => block(:,:,:,:)
         ELSE
            ! need reshape due to rank reordering
            ALLOCATE (block_2d(shape_2d(1), shape_2d(2)))
            new_block = .TRUE.
            shape_nd(map_nd) = sizes
            block_2d(:, :) = RESHAPE(RESHAPE(block, SHAPE=shape_nd, order=map_nd), SHAPE=shape_2d)
         ENDIF

         ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      END ASSOCIATE

      CALL dbcsr_tas_put_block(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d, summation=summation, &
                               scale=scale)

      IF (new_block) DEALLOCATE (block_2d)

   END SUBROUTINE
# 609 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 548 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 549 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_put_2d_block_c_dp (tensor, ind, sizes, block, summation, scale)
      !! Template for dbcsr_t_put_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(2), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(2), INTENT(IN) :: sizes
         !! block size
      COMPLEX(kind=real_8), DIMENSION(sizes(1), sizes(2)), &
         INTENT(IN), TARGET                                 :: block
         !! block to put
      LOGICAL, INTENT(IN), OPTIONAL                         :: summation
         !! whether block should be summed to existing block
      COMPLEX(kind=real_8), INTENT(IN), OPTIONAL                       :: scale
         !! scaling factor
      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      INTEGER, DIMENSION(2)                                 :: shape_2d
      COMPLEX(kind=real_8), POINTER, DIMENSION(:, :)                   :: block_2d
      INTEGER, DIMENSION(2)                          :: shape_nd
      LOGICAL :: found, new_block
      COMPLEX(kind=real_8), DIMENSION(sizes(1), sizes(2)) :: block_check

      LOGICAL, PARAMETER :: debug = .FALSE.
      INTEGER :: i

      new_block = .FALSE.

      IF (debug) THEN
         CALL dbcsr_t_get_block(tensor, ind, sizes, block_check, found=found)
         DBCSR_ASSERT(found)
      ENDIF

      ASSOCIATE (map_nd => tensor%nd_index_blk%map_nd, &
                 map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         shape_2d = [PRODUCT(sizes(map1_2d)), PRODUCT(sizes(map2_2d))]

         IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 2)/))) THEN
            ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
            block_2d(1:shape_2d(1), 1:shape_2d(2)) => block(:,:)
         ELSE
            ! need reshape due to rank reordering
            ALLOCATE (block_2d(shape_2d(1), shape_2d(2)))
            new_block = .TRUE.
            shape_nd(map_nd) = sizes
            block_2d(:, :) = RESHAPE(RESHAPE(block, SHAPE=shape_nd, order=map_nd), SHAPE=shape_2d)
         ENDIF

         ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      END ASSOCIATE

      CALL dbcsr_tas_put_block(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d, summation=summation, &
                               scale=scale)

      IF (new_block) DEALLOCATE (block_2d)

   END SUBROUTINE
# 549 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_put_3d_block_c_dp (tensor, ind, sizes, block, summation, scale)
      !! Template for dbcsr_t_put_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(3), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(3), INTENT(IN) :: sizes
         !! block size
      COMPLEX(kind=real_8), DIMENSION(sizes(1), sizes(2), sizes(3)), &
         INTENT(IN), TARGET                                 :: block
         !! block to put
      LOGICAL, INTENT(IN), OPTIONAL                         :: summation
         !! whether block should be summed to existing block
      COMPLEX(kind=real_8), INTENT(IN), OPTIONAL                       :: scale
         !! scaling factor
      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      INTEGER, DIMENSION(2)                                 :: shape_2d
      COMPLEX(kind=real_8), POINTER, DIMENSION(:, :)                   :: block_2d
      INTEGER, DIMENSION(3)                          :: shape_nd
      LOGICAL :: found, new_block
      COMPLEX(kind=real_8), DIMENSION(sizes(1), sizes(2), sizes(3)) :: block_check

      LOGICAL, PARAMETER :: debug = .FALSE.
      INTEGER :: i

      new_block = .FALSE.

      IF (debug) THEN
         CALL dbcsr_t_get_block(tensor, ind, sizes, block_check, found=found)
         DBCSR_ASSERT(found)
      ENDIF

      ASSOCIATE (map_nd => tensor%nd_index_blk%map_nd, &
                 map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         shape_2d = [PRODUCT(sizes(map1_2d)), PRODUCT(sizes(map2_2d))]

         IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 3)/))) THEN
            ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
            block_2d(1:shape_2d(1), 1:shape_2d(2)) => block(:,:,:)
         ELSE
            ! need reshape due to rank reordering
            ALLOCATE (block_2d(shape_2d(1), shape_2d(2)))
            new_block = .TRUE.
            shape_nd(map_nd) = sizes
            block_2d(:, :) = RESHAPE(RESHAPE(block, SHAPE=shape_nd, order=map_nd), SHAPE=shape_2d)
         ENDIF

         ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      END ASSOCIATE

      CALL dbcsr_tas_put_block(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d, summation=summation, &
                               scale=scale)

      IF (new_block) DEALLOCATE (block_2d)

   END SUBROUTINE
# 549 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_put_4d_block_c_dp (tensor, ind, sizes, block, summation, scale)
      !! Template for dbcsr_t_put_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(4), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(4), INTENT(IN) :: sizes
         !! block size
      COMPLEX(kind=real_8), DIMENSION(sizes(1), sizes(2), sizes(3), sizes(4)), &
         INTENT(IN), TARGET                                 :: block
         !! block to put
      LOGICAL, INTENT(IN), OPTIONAL                         :: summation
         !! whether block should be summed to existing block
      COMPLEX(kind=real_8), INTENT(IN), OPTIONAL                       :: scale
         !! scaling factor
      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      INTEGER, DIMENSION(2)                                 :: shape_2d
      COMPLEX(kind=real_8), POINTER, DIMENSION(:, :)                   :: block_2d
      INTEGER, DIMENSION(4)                          :: shape_nd
      LOGICAL :: found, new_block
      COMPLEX(kind=real_8), DIMENSION(sizes(1), sizes(2), sizes(3), sizes(4)) :: block_check

      LOGICAL, PARAMETER :: debug = .FALSE.
      INTEGER :: i

      new_block = .FALSE.

      IF (debug) THEN
         CALL dbcsr_t_get_block(tensor, ind, sizes, block_check, found=found)
         DBCSR_ASSERT(found)
      ENDIF

      ASSOCIATE (map_nd => tensor%nd_index_blk%map_nd, &
                 map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         shape_2d = [PRODUCT(sizes(map1_2d)), PRODUCT(sizes(map2_2d))]

         IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 4)/))) THEN
            ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
            block_2d(1:shape_2d(1), 1:shape_2d(2)) => block(:,:,:,:)
         ELSE
            ! need reshape due to rank reordering
            ALLOCATE (block_2d(shape_2d(1), shape_2d(2)))
            new_block = .TRUE.
            shape_nd(map_nd) = sizes
            block_2d(:, :) = RESHAPE(RESHAPE(block, SHAPE=shape_nd, order=map_nd), SHAPE=shape_2d)
         ENDIF

         ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      END ASSOCIATE

      CALL dbcsr_tas_put_block(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d, summation=summation, &
                               scale=scale)

      IF (new_block) DEALLOCATE (block_2d)

   END SUBROUTINE
# 609 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 548 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 549 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_put_2d_block_c_sp (tensor, ind, sizes, block, summation, scale)
      !! Template for dbcsr_t_put_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(2), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(2), INTENT(IN) :: sizes
         !! block size
      COMPLEX(kind=real_4), DIMENSION(sizes(1), sizes(2)), &
         INTENT(IN), TARGET                                 :: block
         !! block to put
      LOGICAL, INTENT(IN), OPTIONAL                         :: summation
         !! whether block should be summed to existing block
      COMPLEX(kind=real_4), INTENT(IN), OPTIONAL                       :: scale
         !! scaling factor
      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      INTEGER, DIMENSION(2)                                 :: shape_2d
      COMPLEX(kind=real_4), POINTER, DIMENSION(:, :)                   :: block_2d
      INTEGER, DIMENSION(2)                          :: shape_nd
      LOGICAL :: found, new_block
      COMPLEX(kind=real_4), DIMENSION(sizes(1), sizes(2)) :: block_check

      LOGICAL, PARAMETER :: debug = .FALSE.
      INTEGER :: i

      new_block = .FALSE.

      IF (debug) THEN
         CALL dbcsr_t_get_block(tensor, ind, sizes, block_check, found=found)
         DBCSR_ASSERT(found)
      ENDIF

      ASSOCIATE (map_nd => tensor%nd_index_blk%map_nd, &
                 map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         shape_2d = [PRODUCT(sizes(map1_2d)), PRODUCT(sizes(map2_2d))]

         IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 2)/))) THEN
            ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
            block_2d(1:shape_2d(1), 1:shape_2d(2)) => block(:,:)
         ELSE
            ! need reshape due to rank reordering
            ALLOCATE (block_2d(shape_2d(1), shape_2d(2)))
            new_block = .TRUE.
            shape_nd(map_nd) = sizes
            block_2d(:, :) = RESHAPE(RESHAPE(block, SHAPE=shape_nd, order=map_nd), SHAPE=shape_2d)
         ENDIF

         ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      END ASSOCIATE

      CALL dbcsr_tas_put_block(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d, summation=summation, &
                               scale=scale)

      IF (new_block) DEALLOCATE (block_2d)

   END SUBROUTINE
# 549 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_put_3d_block_c_sp (tensor, ind, sizes, block, summation, scale)
      !! Template for dbcsr_t_put_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(3), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(3), INTENT(IN) :: sizes
         !! block size
      COMPLEX(kind=real_4), DIMENSION(sizes(1), sizes(2), sizes(3)), &
         INTENT(IN), TARGET                                 :: block
         !! block to put
      LOGICAL, INTENT(IN), OPTIONAL                         :: summation
         !! whether block should be summed to existing block
      COMPLEX(kind=real_4), INTENT(IN), OPTIONAL                       :: scale
         !! scaling factor
      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      INTEGER, DIMENSION(2)                                 :: shape_2d
      COMPLEX(kind=real_4), POINTER, DIMENSION(:, :)                   :: block_2d
      INTEGER, DIMENSION(3)                          :: shape_nd
      LOGICAL :: found, new_block
      COMPLEX(kind=real_4), DIMENSION(sizes(1), sizes(2), sizes(3)) :: block_check

      LOGICAL, PARAMETER :: debug = .FALSE.
      INTEGER :: i

      new_block = .FALSE.

      IF (debug) THEN
         CALL dbcsr_t_get_block(tensor, ind, sizes, block_check, found=found)
         DBCSR_ASSERT(found)
      ENDIF

      ASSOCIATE (map_nd => tensor%nd_index_blk%map_nd, &
                 map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         shape_2d = [PRODUCT(sizes(map1_2d)), PRODUCT(sizes(map2_2d))]

         IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 3)/))) THEN
            ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
            block_2d(1:shape_2d(1), 1:shape_2d(2)) => block(:,:,:)
         ELSE
            ! need reshape due to rank reordering
            ALLOCATE (block_2d(shape_2d(1), shape_2d(2)))
            new_block = .TRUE.
            shape_nd(map_nd) = sizes
            block_2d(:, :) = RESHAPE(RESHAPE(block, SHAPE=shape_nd, order=map_nd), SHAPE=shape_2d)
         ENDIF

         ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      END ASSOCIATE

      CALL dbcsr_tas_put_block(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d, summation=summation, &
                               scale=scale)

      IF (new_block) DEALLOCATE (block_2d)

   END SUBROUTINE
# 549 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_put_4d_block_c_sp (tensor, ind, sizes, block, summation, scale)
      !! Template for dbcsr_t_put_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(4), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(4), INTENT(IN) :: sizes
         !! block size
      COMPLEX(kind=real_4), DIMENSION(sizes(1), sizes(2), sizes(3), sizes(4)), &
         INTENT(IN), TARGET                                 :: block
         !! block to put
      LOGICAL, INTENT(IN), OPTIONAL                         :: summation
         !! whether block should be summed to existing block
      COMPLEX(kind=real_4), INTENT(IN), OPTIONAL                       :: scale
         !! scaling factor
      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      INTEGER, DIMENSION(2)                                 :: shape_2d
      COMPLEX(kind=real_4), POINTER, DIMENSION(:, :)                   :: block_2d
      INTEGER, DIMENSION(4)                          :: shape_nd
      LOGICAL :: found, new_block
      COMPLEX(kind=real_4), DIMENSION(sizes(1), sizes(2), sizes(3), sizes(4)) :: block_check

      LOGICAL, PARAMETER :: debug = .FALSE.
      INTEGER :: i

      new_block = .FALSE.

      IF (debug) THEN
         CALL dbcsr_t_get_block(tensor, ind, sizes, block_check, found=found)
         DBCSR_ASSERT(found)
      ENDIF

      ASSOCIATE (map_nd => tensor%nd_index_blk%map_nd, &
                 map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         shape_2d = [PRODUCT(sizes(map1_2d)), PRODUCT(sizes(map2_2d))]

         IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 4)/))) THEN
            ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
            block_2d(1:shape_2d(1), 1:shape_2d(2)) => block(:,:,:,:)
         ELSE
            ! need reshape due to rank reordering
            ALLOCATE (block_2d(shape_2d(1), shape_2d(2)))
            new_block = .TRUE.
            shape_nd(map_nd) = sizes
            block_2d(:, :) = RESHAPE(RESHAPE(block, SHAPE=shape_nd, order=map_nd), SHAPE=shape_2d)
         ENDIF

         ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      END ASSOCIATE

      CALL dbcsr_tas_put_block(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d, summation=summation, &
                               scale=scale)

      IF (new_block) DEALLOCATE (block_2d)

   END SUBROUTINE
# 609 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 610 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"

# 612 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 613 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_allocate_and_get_2d_block_r_dp (tensor, ind, block, found)
      !! allocate and get block

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(2), INTENT(IN)  :: ind
         !! block index
      REAL(kind=real_8), DIMENSION(:,:), &
         ALLOCATABLE, INTENT(OUT)                           :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found
      INTEGER, DIMENSION(2)                          :: blk_size

      CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
      CALL allocate_any(block, shape_spec=blk_size)
      CALL dbcsr_t_get_2d_block_r_dp (tensor, ind, blk_size, block, found)

   END SUBROUTINE
# 613 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_allocate_and_get_3d_block_r_dp (tensor, ind, block, found)
      !! allocate and get block

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(3), INTENT(IN)  :: ind
         !! block index
      REAL(kind=real_8), DIMENSION(:,:,:), &
         ALLOCATABLE, INTENT(OUT)                           :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found
      INTEGER, DIMENSION(3)                          :: blk_size

      CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
      CALL allocate_any(block, shape_spec=blk_size)
      CALL dbcsr_t_get_3d_block_r_dp (tensor, ind, blk_size, block, found)

   END SUBROUTINE
# 613 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_allocate_and_get_4d_block_r_dp (tensor, ind, block, found)
      !! allocate and get block

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(4), INTENT(IN)  :: ind
         !! block index
      REAL(kind=real_8), DIMENSION(:,:,:,:), &
         ALLOCATABLE, INTENT(OUT)                           :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found
      INTEGER, DIMENSION(4)                          :: blk_size

      CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
      CALL allocate_any(block, shape_spec=blk_size)
      CALL dbcsr_t_get_4d_block_r_dp (tensor, ind, blk_size, block, found)

   END SUBROUTINE
# 632 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 612 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 613 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_allocate_and_get_2d_block_r_sp (tensor, ind, block, found)
      !! allocate and get block

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(2), INTENT(IN)  :: ind
         !! block index
      REAL(kind=real_4), DIMENSION(:,:), &
         ALLOCATABLE, INTENT(OUT)                           :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found
      INTEGER, DIMENSION(2)                          :: blk_size

      CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
      CALL allocate_any(block, shape_spec=blk_size)
      CALL dbcsr_t_get_2d_block_r_sp (tensor, ind, blk_size, block, found)

   END SUBROUTINE
# 613 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_allocate_and_get_3d_block_r_sp (tensor, ind, block, found)
      !! allocate and get block

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(3), INTENT(IN)  :: ind
         !! block index
      REAL(kind=real_4), DIMENSION(:,:,:), &
         ALLOCATABLE, INTENT(OUT)                           :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found
      INTEGER, DIMENSION(3)                          :: blk_size

      CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
      CALL allocate_any(block, shape_spec=blk_size)
      CALL dbcsr_t_get_3d_block_r_sp (tensor, ind, blk_size, block, found)

   END SUBROUTINE
# 613 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_allocate_and_get_4d_block_r_sp (tensor, ind, block, found)
      !! allocate and get block

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(4), INTENT(IN)  :: ind
         !! block index
      REAL(kind=real_4), DIMENSION(:,:,:,:), &
         ALLOCATABLE, INTENT(OUT)                           :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found
      INTEGER, DIMENSION(4)                          :: blk_size

      CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
      CALL allocate_any(block, shape_spec=blk_size)
      CALL dbcsr_t_get_4d_block_r_sp (tensor, ind, blk_size, block, found)

   END SUBROUTINE
# 632 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 612 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 613 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_allocate_and_get_2d_block_c_dp (tensor, ind, block, found)
      !! allocate and get block

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(2), INTENT(IN)  :: ind
         !! block index
      COMPLEX(kind=real_8), DIMENSION(:,:), &
         ALLOCATABLE, INTENT(OUT)                           :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found
      INTEGER, DIMENSION(2)                          :: blk_size

      CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
      CALL allocate_any(block, shape_spec=blk_size)
      CALL dbcsr_t_get_2d_block_c_dp (tensor, ind, blk_size, block, found)

   END SUBROUTINE
# 613 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_allocate_and_get_3d_block_c_dp (tensor, ind, block, found)
      !! allocate and get block

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(3), INTENT(IN)  :: ind
         !! block index
      COMPLEX(kind=real_8), DIMENSION(:,:,:), &
         ALLOCATABLE, INTENT(OUT)                           :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found
      INTEGER, DIMENSION(3)                          :: blk_size

      CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
      CALL allocate_any(block, shape_spec=blk_size)
      CALL dbcsr_t_get_3d_block_c_dp (tensor, ind, blk_size, block, found)

   END SUBROUTINE
# 613 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_allocate_and_get_4d_block_c_dp (tensor, ind, block, found)
      !! allocate and get block

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(4), INTENT(IN)  :: ind
         !! block index
      COMPLEX(kind=real_8), DIMENSION(:,:,:,:), &
         ALLOCATABLE, INTENT(OUT)                           :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found
      INTEGER, DIMENSION(4)                          :: blk_size

      CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
      CALL allocate_any(block, shape_spec=blk_size)
      CALL dbcsr_t_get_4d_block_c_dp (tensor, ind, blk_size, block, found)

   END SUBROUTINE
# 632 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 612 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 613 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_allocate_and_get_2d_block_c_sp (tensor, ind, block, found)
      !! allocate and get block

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(2), INTENT(IN)  :: ind
         !! block index
      COMPLEX(kind=real_4), DIMENSION(:,:), &
         ALLOCATABLE, INTENT(OUT)                           :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found
      INTEGER, DIMENSION(2)                          :: blk_size

      CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
      CALL allocate_any(block, shape_spec=blk_size)
      CALL dbcsr_t_get_2d_block_c_sp (tensor, ind, blk_size, block, found)

   END SUBROUTINE
# 613 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_allocate_and_get_3d_block_c_sp (tensor, ind, block, found)
      !! allocate and get block

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(3), INTENT(IN)  :: ind
         !! block index
      COMPLEX(kind=real_4), DIMENSION(:,:,:), &
         ALLOCATABLE, INTENT(OUT)                           :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found
      INTEGER, DIMENSION(3)                          :: blk_size

      CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
      CALL allocate_any(block, shape_spec=blk_size)
      CALL dbcsr_t_get_3d_block_c_sp (tensor, ind, blk_size, block, found)

   END SUBROUTINE
# 613 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_allocate_and_get_4d_block_c_sp (tensor, ind, block, found)
      !! allocate and get block

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(4), INTENT(IN)  :: ind
         !! block index
      COMPLEX(kind=real_4), DIMENSION(:,:,:,:), &
         ALLOCATABLE, INTENT(OUT)                           :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found
      INTEGER, DIMENSION(4)                          :: blk_size

      CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
      CALL allocate_any(block, shape_spec=blk_size)
      CALL dbcsr_t_get_4d_block_c_sp (tensor, ind, blk_size, block, found)

   END SUBROUTINE
# 632 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 633 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"

# 635 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 636 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_get_2d_block_r_dp (tensor, ind, sizes, block, found)
      !! Template for dbcsr_t_get_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(2), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(2), INTENT(IN) :: sizes
         !! block size
      REAL(kind=real_8), DIMENSION(sizes(1), sizes(2)), &
         INTENT(OUT)                                        :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found

      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      REAL(kind=real_8), DIMENSION(:, :), POINTER, CONTIGUOUS       :: block_2d_ptr
      LOGICAL                                               :: tr
      INTEGER                                               :: i
      REAL(kind=real_8), DIMENSION(:,:), POINTER  :: block_ptr

      NULLIFY (block_2d_ptr)

      ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      ASSOCIATE (map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)
         DBCSR_ASSERT(.NOT. tr)

         IF (found) THEN
            IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 2)/))) THEN
               ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
               block_ptr(LBOUND(block, 1):UBOUND(block, 1), LBOUND(block, 2):UBOUND(block, 2)) => block_2d_ptr(:, :)
               block(:,:) = block_ptr(:,:)
            ELSE
               ! need reshape due to rank reordering
               block(:,:) = RESHAPE(block_2d_ptr, SHAPE=SHAPE(block), ORDER=[map1_2d, map2_2d])
            ENDIF
         ENDIF

      END ASSOCIATE

   END SUBROUTINE
# 636 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_get_3d_block_r_dp (tensor, ind, sizes, block, found)
      !! Template for dbcsr_t_get_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(3), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(3), INTENT(IN) :: sizes
         !! block size
      REAL(kind=real_8), DIMENSION(sizes(1), sizes(2), sizes(3)), &
         INTENT(OUT)                                        :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found

      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      REAL(kind=real_8), DIMENSION(:, :), POINTER, CONTIGUOUS       :: block_2d_ptr
      LOGICAL                                               :: tr
      INTEGER                                               :: i
      REAL(kind=real_8), DIMENSION(:,:,:), POINTER  :: block_ptr

      NULLIFY (block_2d_ptr)

      ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      ASSOCIATE (map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)
         DBCSR_ASSERT(.NOT. tr)

         IF (found) THEN
            IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 3)/))) THEN
               ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
               block_ptr(LBOUND(block, 1):UBOUND(block, 1), LBOUND(block, 2):UBOUND(block, 2), LBOUND(block, 3):UBOUND(block, 3))&
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
                   & => block_2d_ptr(:, :)
               block(:,:,:) = block_ptr(:,:,:)
            ELSE
               ! need reshape due to rank reordering
               block(:,:,:) = RESHAPE(block_2d_ptr, SHAPE=SHAPE(block), ORDER=[map1_2d, map2_2d])
            ENDIF
         ENDIF

      END ASSOCIATE

   END SUBROUTINE
# 636 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_get_4d_block_r_dp (tensor, ind, sizes, block, found)
      !! Template for dbcsr_t_get_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(4), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(4), INTENT(IN) :: sizes
         !! block size
      REAL(kind=real_8), DIMENSION(sizes(1), sizes(2), sizes(3), sizes(4)), &
         INTENT(OUT)                                        :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found

      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      REAL(kind=real_8), DIMENSION(:, :), POINTER, CONTIGUOUS       :: block_2d_ptr
      LOGICAL                                               :: tr
      INTEGER                                               :: i
      REAL(kind=real_8), DIMENSION(:,:,:,:), POINTER  :: block_ptr

      NULLIFY (block_2d_ptr)

      ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      ASSOCIATE (map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)
         DBCSR_ASSERT(.NOT. tr)

         IF (found) THEN
            IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 4)/))) THEN
               ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
               block_ptr(LBOUND(block, 1):UBOUND(block, 1), LBOUND(block, 2):UBOUND(block, 2), LBOUND(block, 3):UBOUND(block, 3),&
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
                   & LBOUND(block, 4):UBOUND(block, 4)) => block_2d_ptr(:, :)
               block(:,:,:,:) = block_ptr(:,:,:,:)
            ELSE
               ! need reshape due to rank reordering
               block(:,:,:,:) = RESHAPE(block_2d_ptr, SHAPE=SHAPE(block), ORDER=[map1_2d, map2_2d])
            ENDIF
         ENDIF

      END ASSOCIATE

   END SUBROUTINE
# 681 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 635 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 636 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_get_2d_block_r_sp (tensor, ind, sizes, block, found)
      !! Template for dbcsr_t_get_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(2), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(2), INTENT(IN) :: sizes
         !! block size
      REAL(kind=real_4), DIMENSION(sizes(1), sizes(2)), &
         INTENT(OUT)                                        :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found

      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      REAL(kind=real_4), DIMENSION(:, :), POINTER, CONTIGUOUS       :: block_2d_ptr
      LOGICAL                                               :: tr
      INTEGER                                               :: i
      REAL(kind=real_4), DIMENSION(:,:), POINTER  :: block_ptr

      NULLIFY (block_2d_ptr)

      ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      ASSOCIATE (map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)
         DBCSR_ASSERT(.NOT. tr)

         IF (found) THEN
            IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 2)/))) THEN
               ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
               block_ptr(LBOUND(block, 1):UBOUND(block, 1), LBOUND(block, 2):UBOUND(block, 2)) => block_2d_ptr(:, :)
               block(:,:) = block_ptr(:,:)
            ELSE
               ! need reshape due to rank reordering
               block(:,:) = RESHAPE(block_2d_ptr, SHAPE=SHAPE(block), ORDER=[map1_2d, map2_2d])
            ENDIF
         ENDIF

      END ASSOCIATE

   END SUBROUTINE
# 636 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_get_3d_block_r_sp (tensor, ind, sizes, block, found)
      !! Template for dbcsr_t_get_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(3), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(3), INTENT(IN) :: sizes
         !! block size
      REAL(kind=real_4), DIMENSION(sizes(1), sizes(2), sizes(3)), &
         INTENT(OUT)                                        :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found

      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      REAL(kind=real_4), DIMENSION(:, :), POINTER, CONTIGUOUS       :: block_2d_ptr
      LOGICAL                                               :: tr
      INTEGER                                               :: i
      REAL(kind=real_4), DIMENSION(:,:,:), POINTER  :: block_ptr

      NULLIFY (block_2d_ptr)

      ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      ASSOCIATE (map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)
         DBCSR_ASSERT(.NOT. tr)

         IF (found) THEN
            IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 3)/))) THEN
               ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
               block_ptr(LBOUND(block, 1):UBOUND(block, 1), LBOUND(block, 2):UBOUND(block, 2), LBOUND(block, 3):UBOUND(block, 3))&
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
                   & => block_2d_ptr(:, :)
               block(:,:,:) = block_ptr(:,:,:)
            ELSE
               ! need reshape due to rank reordering
               block(:,:,:) = RESHAPE(block_2d_ptr, SHAPE=SHAPE(block), ORDER=[map1_2d, map2_2d])
            ENDIF
         ENDIF

      END ASSOCIATE

   END SUBROUTINE
# 636 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_get_4d_block_r_sp (tensor, ind, sizes, block, found)
      !! Template for dbcsr_t_get_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(4), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(4), INTENT(IN) :: sizes
         !! block size
      REAL(kind=real_4), DIMENSION(sizes(1), sizes(2), sizes(3), sizes(4)), &
         INTENT(OUT)                                        :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found

      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      REAL(kind=real_4), DIMENSION(:, :), POINTER, CONTIGUOUS       :: block_2d_ptr
      LOGICAL                                               :: tr
      INTEGER                                               :: i
      REAL(kind=real_4), DIMENSION(:,:,:,:), POINTER  :: block_ptr

      NULLIFY (block_2d_ptr)

      ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      ASSOCIATE (map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)
         DBCSR_ASSERT(.NOT. tr)

         IF (found) THEN
            IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 4)/))) THEN
               ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
               block_ptr(LBOUND(block, 1):UBOUND(block, 1), LBOUND(block, 2):UBOUND(block, 2), LBOUND(block, 3):UBOUND(block, 3),&
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
                   & LBOUND(block, 4):UBOUND(block, 4)) => block_2d_ptr(:, :)
               block(:,:,:,:) = block_ptr(:,:,:,:)
            ELSE
               ! need reshape due to rank reordering
               block(:,:,:,:) = RESHAPE(block_2d_ptr, SHAPE=SHAPE(block), ORDER=[map1_2d, map2_2d])
            ENDIF
         ENDIF

      END ASSOCIATE

   END SUBROUTINE
# 681 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 635 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 636 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_get_2d_block_c_dp (tensor, ind, sizes, block, found)
      !! Template for dbcsr_t_get_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(2), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(2), INTENT(IN) :: sizes
         !! block size
      COMPLEX(kind=real_8), DIMENSION(sizes(1), sizes(2)), &
         INTENT(OUT)                                        :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found

      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      COMPLEX(kind=real_8), DIMENSION(:, :), POINTER, CONTIGUOUS       :: block_2d_ptr
      LOGICAL                                               :: tr
      INTEGER                                               :: i
      COMPLEX(kind=real_8), DIMENSION(:,:), POINTER  :: block_ptr

      NULLIFY (block_2d_ptr)

      ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      ASSOCIATE (map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)
         DBCSR_ASSERT(.NOT. tr)

         IF (found) THEN
            IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 2)/))) THEN
               ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
               block_ptr(LBOUND(block, 1):UBOUND(block, 1), LBOUND(block, 2):UBOUND(block, 2)) => block_2d_ptr(:, :)
               block(:,:) = block_ptr(:,:)
            ELSE
               ! need reshape due to rank reordering
               block(:,:) = RESHAPE(block_2d_ptr, SHAPE=SHAPE(block), ORDER=[map1_2d, map2_2d])
            ENDIF
         ENDIF

      END ASSOCIATE

   END SUBROUTINE
# 636 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_get_3d_block_c_dp (tensor, ind, sizes, block, found)
      !! Template for dbcsr_t_get_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(3), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(3), INTENT(IN) :: sizes
         !! block size
      COMPLEX(kind=real_8), DIMENSION(sizes(1), sizes(2), sizes(3)), &
         INTENT(OUT)                                        :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found

      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      COMPLEX(kind=real_8), DIMENSION(:, :), POINTER, CONTIGUOUS       :: block_2d_ptr
      LOGICAL                                               :: tr
      INTEGER                                               :: i
      COMPLEX(kind=real_8), DIMENSION(:,:,:), POINTER  :: block_ptr

      NULLIFY (block_2d_ptr)

      ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      ASSOCIATE (map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)
         DBCSR_ASSERT(.NOT. tr)

         IF (found) THEN
            IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 3)/))) THEN
               ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
               block_ptr(LBOUND(block, 1):UBOUND(block, 1), LBOUND(block, 2):UBOUND(block, 2), LBOUND(block, 3):UBOUND(block, 3))&
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
                   & => block_2d_ptr(:, :)
               block(:,:,:) = block_ptr(:,:,:)
            ELSE
               ! need reshape due to rank reordering
               block(:,:,:) = RESHAPE(block_2d_ptr, SHAPE=SHAPE(block), ORDER=[map1_2d, map2_2d])
            ENDIF
         ENDIF

      END ASSOCIATE

   END SUBROUTINE
# 636 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_get_4d_block_c_dp (tensor, ind, sizes, block, found)
      !! Template for dbcsr_t_get_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(4), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(4), INTENT(IN) :: sizes
         !! block size
      COMPLEX(kind=real_8), DIMENSION(sizes(1), sizes(2), sizes(3), sizes(4)), &
         INTENT(OUT)                                        :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found

      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      COMPLEX(kind=real_8), DIMENSION(:, :), POINTER, CONTIGUOUS       :: block_2d_ptr
      LOGICAL                                               :: tr
      INTEGER                                               :: i
      COMPLEX(kind=real_8), DIMENSION(:,:,:,:), POINTER  :: block_ptr

      NULLIFY (block_2d_ptr)

      ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      ASSOCIATE (map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)
         DBCSR_ASSERT(.NOT. tr)

         IF (found) THEN
            IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 4)/))) THEN
               ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
               block_ptr(LBOUND(block, 1):UBOUND(block, 1), LBOUND(block, 2):UBOUND(block, 2), LBOUND(block, 3):UBOUND(block, 3),&
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
                   & LBOUND(block, 4):UBOUND(block, 4)) => block_2d_ptr(:, :)
               block(:,:,:,:) = block_ptr(:,:,:,:)
            ELSE
               ! need reshape due to rank reordering
               block(:,:,:,:) = RESHAPE(block_2d_ptr, SHAPE=SHAPE(block), ORDER=[map1_2d, map2_2d])
            ENDIF
         ENDIF

      END ASSOCIATE

   END SUBROUTINE
# 681 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 635 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 636 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_get_2d_block_c_sp (tensor, ind, sizes, block, found)
      !! Template for dbcsr_t_get_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(2), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(2), INTENT(IN) :: sizes
         !! block size
      COMPLEX(kind=real_4), DIMENSION(sizes(1), sizes(2)), &
         INTENT(OUT)                                        :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found

      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      COMPLEX(kind=real_4), DIMENSION(:, :), POINTER, CONTIGUOUS       :: block_2d_ptr
      LOGICAL                                               :: tr
      INTEGER                                               :: i
      COMPLEX(kind=real_4), DIMENSION(:,:), POINTER  :: block_ptr

      NULLIFY (block_2d_ptr)

      ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      ASSOCIATE (map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)
         DBCSR_ASSERT(.NOT. tr)

         IF (found) THEN
            IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 2)/))) THEN
               ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
               block_ptr(LBOUND(block, 1):UBOUND(block, 1), LBOUND(block, 2):UBOUND(block, 2)) => block_2d_ptr(:, :)
               block(:,:) = block_ptr(:,:)
            ELSE
               ! need reshape due to rank reordering
               block(:,:) = RESHAPE(block_2d_ptr, SHAPE=SHAPE(block), ORDER=[map1_2d, map2_2d])
            ENDIF
         ENDIF

      END ASSOCIATE

   END SUBROUTINE
# 636 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_get_3d_block_c_sp (tensor, ind, sizes, block, found)
      !! Template for dbcsr_t_get_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(3), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(3), INTENT(IN) :: sizes
         !! block size
      COMPLEX(kind=real_4), DIMENSION(sizes(1), sizes(2), sizes(3)), &
         INTENT(OUT)                                        :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found

      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      COMPLEX(kind=real_4), DIMENSION(:, :), POINTER, CONTIGUOUS       :: block_2d_ptr
      LOGICAL                                               :: tr
      INTEGER                                               :: i
      COMPLEX(kind=real_4), DIMENSION(:,:,:), POINTER  :: block_ptr

      NULLIFY (block_2d_ptr)

      ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      ASSOCIATE (map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)
         DBCSR_ASSERT(.NOT. tr)

         IF (found) THEN
            IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 3)/))) THEN
               ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
               block_ptr(LBOUND(block, 1):UBOUND(block, 1), LBOUND(block, 2):UBOUND(block, 2), LBOUND(block, 3):UBOUND(block, 3))&
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
                   & => block_2d_ptr(:, :)
               block(:,:,:) = block_ptr(:,:,:)
            ELSE
               ! need reshape due to rank reordering
               block(:,:,:) = RESHAPE(block_2d_ptr, SHAPE=SHAPE(block), ORDER=[map1_2d, map2_2d])
            ENDIF
         ENDIF

      END ASSOCIATE

   END SUBROUTINE
# 636 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
   SUBROUTINE dbcsr_t_get_4d_block_c_sp (tensor, ind, sizes, block, found)
      !! Template for dbcsr_t_get_block.

      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(4), INTENT(IN) :: ind
         !! block index
      INTEGER, DIMENSION(4), INTENT(IN) :: sizes
         !! block size
      COMPLEX(kind=real_4), DIMENSION(sizes(1), sizes(2), sizes(3), sizes(4)), &
         INTENT(OUT)                                        :: block
         !! block to get
      LOGICAL, INTENT(OUT)                                  :: found
         !! whether block was found

      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      COMPLEX(kind=real_4), DIMENSION(:, :), POINTER, CONTIGUOUS       :: block_2d_ptr
      LOGICAL                                               :: tr
      INTEGER                                               :: i
      COMPLEX(kind=real_4), DIMENSION(:,:,:,:), POINTER  :: block_ptr

      NULLIFY (block_2d_ptr)

      ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

      ASSOCIATE (map1_2d => tensor%nd_index_blk%map1_2d, &
                 map2_2d => tensor%nd_index_blk%map2_2d)

         CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)
         DBCSR_ASSERT(.NOT. tr)

         IF (found) THEN
            IF (ALL([map1_2d, map2_2d] == (/(i, i=1, 4)/))) THEN
               ! to avoid costly reshape can do pointer bounds remapping as long as arrays are equivalent in memory
               block_ptr(LBOUND(block, 1):UBOUND(block, 1), LBOUND(block, 2):UBOUND(block, 2), LBOUND(block, 3):UBOUND(block, 3),&
# 669 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
                   & LBOUND(block, 4):UBOUND(block, 4)) => block_2d_ptr(:, :)
               block(:,:,:,:) = block_ptr(:,:,:,:)
            ELSE
               ! need reshape due to rank reordering
               block(:,:,:,:) = RESHAPE(block_2d_ptr, SHAPE=SHAPE(block), ORDER=[map1_2d, map2_2d])
            ENDIF
         ENDIF

      END ASSOCIATE

   END SUBROUTINE
# 681 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"
# 682 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_block.F"

END MODULE
