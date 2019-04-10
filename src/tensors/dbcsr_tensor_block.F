!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Methods to operate on n-dimensional tensor blocks.
!> \author Patrick Seewald
! **************************************************************************************************
MODULE dbcsr_tensor_block

#:include "dbcsr_tensor.fypp"
#:set maxdim = maxrank
#:set ndims = range(2,maxdim+1)

   USE dbcsr_allocate_wrap, ONLY: allocate_any
   USE dbcsr_api, ONLY: ${uselist(dtype_float_param)}$, &
                        dbcsr_iterator_type, &
                        dbcsr_iterator_blocks_left, &
                        dbcsr_iterator_next_block, &
                        dbcsr_iterator_start, &
                        dbcsr_iterator_stop, &
                        dbcsr_type, &
                        dbcsr_reserve_blocks, &
                        dbcsr_scalar_type, &
                        dbcsr_finalize, &
                        dbcsr_get_num_blocks
   USE dbcsr_tas_types, ONLY: dbcsr_tas_iterator
   USE dbcsr_tas_base, ONLY: dbcsr_tas_iterator_next_block, &
                             dbcsr_tas_iterator_blocks_left, &
                             dbcsr_tas_iterator_start, &
                             dbcsr_tas_iterator_stop, &
                             dbcsr_tas_get_block_p, &
                             dbcsr_tas_put_block, &
                             dbcsr_tas_reserve_blocks
   USE dbcsr_kinds, ONLY: ${uselist(dtype_float_prec)}$, &
                          int_8
   USE dbcsr_tensor_index, ONLY: nd_to_2d_mapping, &
                                 ndims_mapping, &
                                 get_nd_indices, &
                                 destroy_nd_to_2d_mapping, &
                                 get_2d_indices, &
                                 get_mapping_info, &
                                 create_nd_to_2d_mapping
   USE dbcsr_array_list_methods, ONLY: array_list, &
                                       get_array_elements, &
                                       destroy_array_list, &
                                       sizes_of_arrays, &
                                       create_array_list, &
                                       get_arrays
   USE dbcsr_tensor_types, ONLY: dbcsr_t_type, &
                                 ndims_tensor, &
                                 dbcsr_t_get_data_type, &
                                 dbcsr_t_blk_sizes, &
                                 dbcsr_t_get_num_blocks, &
                                 dbcsr_t_finalize
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
      destroy_block, &
      ndims_iterator, &
      reshape_2d_to_nd_block, &
      reshape_nd_to_2d_block

   TYPE dbcsr_t_iterator_type
      TYPE(dbcsr_tas_iterator)      :: iter
      TYPE(nd_to_2d_mapping)      :: nd_index_blk
      TYPE(nd_to_2d_mapping)      :: nd_index
      TYPE(array_list)            :: blk_sizes, blk_offsets
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
      TYPE(block_nd_${dsuffix}$) :: ${dsuffix}$
#:endfor
      INTEGER          :: data_type = -1
   END TYPE

   INTERFACE create_block
#:for dparam, dtype, dsuffix in dtype_float_list
      MODULE PROCEDURE create_block_data_${dsuffix}$
#:endfor
      MODULE PROCEDURE create_block_nodata
   END INTERFACE

   INTERFACE reshape_nd_to_2d_block
#:for dparam, dtype, dsuffix in dtype_float_list
#:for ndim in ndims
      MODULE PROCEDURE reshape_block_t2m_${ndim}$d_${dsuffix}$
#:endfor
#:endfor
   END INTERFACE reshape_nd_to_2d_block

   INTERFACE reshape_2d_to_nd_block
#:for dparam, dtype, dsuffix in dtype_float_list
#:for ndim in ndims
      MODULE PROCEDURE reshape_block_m2t_${ndim}$d_${dsuffix}$
#:endfor
#:endfor
   END INTERFACE reshape_2d_to_nd_block

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
      MODULE PROCEDURE dbcsr_t_reserve_blocks_index_list
      MODULE PROCEDURE dbcsr_t_reserve_blocks_index_array
      MODULE PROCEDURE dbcsr_t_reserve_blocks_template
      MODULE PROCEDURE dbcsr_t_reserve_blocks_tensor_to_matrix
      MODULE PROCEDURE dbcsr_t_reserve_blocks_matrix_to_tensor
   END INTERFACE

CONTAINS

! **************************************************************************************************
!> \brief Create block without data
!> \param block ...
!> \param sizes ...
!> \param data_type ...
! **************************************************************************************************
   SUBROUTINE create_block_nodata(block, sizes, data_type)
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

! **************************************************************************************************
!> \brief Destroy block
!> \param block ...
! **************************************************************************************************
   SUBROUTINE destroy_block(block)
      TYPE(block_nd), INTENT(INOUT) :: block

      SELECT CASE (block%data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
      CASE (${dparam}$)
         CALL destroy_block_${dsuffix}$ (block%${dsuffix}$)
#:endfor
      END SELECT

   END SUBROUTINE

! **************************************************************************************************
!> \brief block size
!> \param block ...
!> \return ...
! **************************************************************************************************
   FUNCTION block_size(block)
      TYPE(block_nd), INTENT(IN)         :: block
      INTEGER, ALLOCATABLE, DIMENSION(:) :: block_size

      SELECT CASE (block%data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
      CASE (${dparam}$)
         CALL allocate_any(block_size, source=block%${dsuffix}$%sizes)
#:endfor
      END SELECT
   END FUNCTION

! **************************************************************************************************
!> \brief Generalization of dbcsr_iterator_start for tensors.
!> \param iterator ...
!> \param tensor ...
! **************************************************************************************************
   SUBROUTINE dbcsr_t_iterator_start(iterator, tensor)
      TYPE(dbcsr_t_iterator_type), INTENT(OUT)           :: iterator
      TYPE(dbcsr_t_type), INTENT(IN)                     :: tensor
      INTEGER                                            :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_iterator_start', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)
      DBCSR_ASSERT(tensor%valid)

      CALL dbcsr_tas_iterator_start(iterator%iter, tensor%matrix_rep)
      iterator%nd_index_blk = tensor%nd_index_blk
      iterator%nd_index = tensor%nd_index
      iterator%blk_sizes = tensor%blk_sizes
      iterator%blk_offsets = tensor%blk_offsets

      CALL timestop(handle)
   END SUBROUTINE

! **************************************************************************************************
!> \brief Generalization of dbcsr_iterator_stop for tensors.
!> \param iterator ...
! **************************************************************************************************
   SUBROUTINE dbcsr_t_iterator_stop(iterator)
      TYPE(dbcsr_t_iterator_type), INTENT(INOUT) :: iterator
      INTEGER                                    :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_iterator_stop', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)
      CALL dbcsr_tas_iterator_stop(iterator%iter)
      CALL destroy_nd_to_2d_mapping(iterator%nd_index)
      CALL destroy_nd_to_2d_mapping(iterator%nd_index_blk)
      CALL destroy_array_list(iterator%blk_sizes)
      CALL destroy_array_list(iterator%blk_offsets)

      CALL timestop(handle)
   END SUBROUTINE

! **************************************************************************************************
!> \brief Number of dimensions.
!> \param iterator ...
!> \return ...
!> \par Note: specification function below must be defined before it is used in
!>      the source due to a bug in the IBM XL Fortran compiler (compilation fails)
! **************************************************************************************************
   PURE FUNCTION ndims_iterator(iterator)
      TYPE(dbcsr_t_iterator_type), INTENT(IN) :: iterator
      INTEGER                                 :: ndims_iterator

      ndims_iterator = iterator%nd_index%ndim_nd
   END FUNCTION

! **************************************************************************************************
!> \brief iterate over nd blocks of an nd rank tensor, index only (blocks must be retrieved by calling
!>        dbcsr_t_get_block on tensor).
!> \param iterator ...
!> \param ind_nd nd index of block
!> \param blk is this needed?
!> \param blk_p is this needed?
!> \param blk_size blk size in each dimension
!> \param blk_offset blk offset in each dimension
! **************************************************************************************************
   SUBROUTINE dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_p, blk_size, blk_offset)
      TYPE(dbcsr_t_iterator_type), INTENT(INOUT)     :: iterator
      INTEGER, DIMENSION(ndims_iterator(iterator)), &
         INTENT(OUT)                                 :: ind_nd
      INTEGER, INTENT(OUT)                           :: blk
      INTEGER, INTENT(OUT), OPTIONAL                 :: blk_p
      INTEGER, DIMENSION(ndims_iterator(iterator)), &
         INTENT(OUT), OPTIONAL                       :: blk_size, blk_offset

      INTEGER(KIND=int_8), DIMENSION(2)              :: ind_2d
      INTEGER                                        :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_iterator_next_block', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)
      CALL dbcsr_tas_iterator_next_block(iterator%iter, ind_2d(1), ind_2d(2), blk, blk_p=blk_p)

      ind_nd(:) = get_nd_indices(iterator%nd_index_blk, ind_2d)
      IF (PRESENT(blk_size)) blk_size(:) = get_array_elements(iterator%blk_sizes, ind_nd)
      ! note: blk_offset needs to be determined by tensor metadata, can not be derived from 2d row/col
      ! offset since block index mapping is not consistent with element index mapping
      IF (PRESENT(blk_offset)) blk_offset(:) = get_array_elements(iterator%blk_offsets, ind_nd)

      CALL timestop(handle)
   END SUBROUTINE

! **************************************************************************************************
!> \brief Generalization of dbcsr_iterator_blocks_left for tensors.
!> \param iterator ...
!> \return ...
! **************************************************************************************************
   FUNCTION dbcsr_t_iterator_blocks_left(iterator)
      TYPE(dbcsr_t_iterator_type), INTENT(IN) :: iterator
      LOGICAL                                 :: dbcsr_t_iterator_blocks_left
      INTEGER                                 :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_iterator_blocks_left', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)
      dbcsr_t_iterator_blocks_left = dbcsr_tas_iterator_blocks_left(iterator%iter)

      CALL timestop(handle)
   END FUNCTION

! **************************************************************************************************
!> \brief reserve blocks from indices as array_list object
!> \param tensor ...
!> \param blk_ind ...
! **************************************************************************************************
   SUBROUTINE dbcsr_t_reserve_blocks_index_list(tensor, blk_ind)
      TYPE(dbcsr_t_type), INTENT(INOUT)  :: tensor
      TYPE(array_list), INTENT(IN)       :: blk_ind
      INTEGER, DIMENSION(:), ALLOCATABLE :: ${varlist("blk_ind")}$
      INTEGER                            :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_reserve_blocks_index_list', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)
#:for ndim in ndims
      IF (ndims_tensor(tensor) == ${ndim}$) THEN
         CALL get_arrays(blk_ind, ${varlist("blk_ind", nmax=ndim)}$)
         CALL dbcsr_t_reserve_blocks(tensor, ${varlist("blk_ind", nmax=ndim)}$)
      ENDIF
#:endfor
      CALL timestop(handle)
   END SUBROUTINE

! **************************************************************************************************
!> \brief reserve blocks from indices as array object
!> \param tensor ...
!> \param blk_ind ...
! **************************************************************************************************
   SUBROUTINE dbcsr_t_reserve_blocks_index_array(tensor, blk_ind)
      TYPE(dbcsr_t_type), INTENT(INOUT)   :: tensor
      INTEGER, DIMENSION(:, :), INTENT(IN) :: blk_ind
      INTEGER                             :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_reserve_blocks_index_array', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)
#:for ndim in ndims
      IF (ndims_tensor(tensor) == ${ndim}$) THEN
         CALL dbcsr_t_reserve_blocks(tensor, ${arrlist("blk_ind", nmax=ndim, ndim_pre=1)}$)
      ENDIF
#:endfor
      CALL timestop(handle)

   END SUBROUTINE

! **************************************************************************************************
!> \brief reserve tensor blocks using block indices
!> \param tensor ...
!> \param blk_ind_1 index of blocks to reserve in each dimension
!> \param blk_ind_2 ...
!> \param blk_ind_3 ...
! **************************************************************************************************
   SUBROUTINE dbcsr_t_reserve_blocks_index(tensor, ${varlist("blk_ind")}$)
      TYPE(dbcsr_t_type), INTENT(INOUT)           :: tensor
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL :: ${varlist("blk_ind")}$
      INTEGER                                     :: iblk, nblk, handle
      INTEGER(KIND=int_8), ALLOCATABLE, DIMENSION(:)          :: cols, rows
      INTEGER(KIND=int_8), DIMENSION(2)                       :: ind_2d
      TYPE(array_list)                            :: blks
      INTEGER, DIMENSION(ndims_tensor(tensor))   :: iblk_nd, ind_nd, nblk_tmp
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_reserve_blocks_index', &
                                     routineP = moduleN//':'//routineN

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
         ind_2d(:) = get_2d_indices(tensor%nd_index_blk, ind_nd)
         rows(iblk) = ind_2d(1); cols(iblk) = ind_2d(2)
      ENDDO

      CALL dbcsr_tas_reserve_blocks(tensor%matrix_rep, rows=rows, columns=cols)
      CALL dbcsr_t_finalize(tensor)
      CALL timestop(handle)
   END SUBROUTINE

! **************************************************************************************************
!> \brief reserve tensor blocks using template
!> \param tensor_in template tensor
!> \param tensor_out ...
! **************************************************************************************************
   SUBROUTINE dbcsr_t_reserve_blocks_template(tensor_in, tensor_out)
      TYPE(dbcsr_t_type), INTENT(IN)    :: tensor_in
      TYPE(dbcsr_t_type), INTENT(INOUT) :: tensor_out
      INTEGER                           :: handle

      TYPE(array_list)                  :: blk_ind
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_reserve_blocks_template', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)

      CALL dbcsr_t_reserved_block_indices(tensor_in, blk_ind)
      CALL dbcsr_t_reserve_blocks(tensor_out, blk_ind)

      CALL timestop(handle)
   END SUBROUTINE

! **************************************************************************************************
!> \brief reserve tensor blocks using matrix template
!> \param matrix_in ...
!> \param tensor_out ...
! **************************************************************************************************
   SUBROUTINE dbcsr_t_reserve_blocks_matrix_to_tensor(matrix_in, tensor_out)
      TYPE(dbcsr_type), INTENT(IN)        :: matrix_in
      TYPE(dbcsr_t_type), INTENT(INOUT)  :: tensor_out

      INTEGER                            :: blk, iblk, nblk
      INTEGER, ALLOCATABLE, DIMENSION(:) :: blk_ind_1, blk_ind_2
      INTEGER, DIMENSION(2)              :: ind_2d
      TYPE(dbcsr_iterator_type)          :: iter
      INTEGER                            :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_reserve_blocks_matrix_to_tensor', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)

      nblk = dbcsr_get_num_blocks(matrix_in)
      ALLOCATE (blk_ind_1(nblk), blk_ind_2(nblk))
      CALL dbcsr_iterator_start(iter, matrix_in)
      DO iblk = 1, nblk
         CALL dbcsr_iterator_next_block(iter, ind_2d(1), ind_2d(2), blk)
         blk_ind_1(iblk) = ind_2d(1); blk_ind_2(iblk) = ind_2d(2)
      ENDDO
      CALL dbcsr_iterator_stop(iter)

      CALL dbcsr_t_reserve_blocks(tensor_out, blk_ind_1, blk_ind_2)
      CALL timestop(handle)
   END SUBROUTINE

! **************************************************************************************************
!> \brief reserve matrix blocks using tensor template
!> \param tensor_in
!> \param matrix_out
! **************************************************************************************************
   SUBROUTINE dbcsr_t_reserve_blocks_tensor_to_matrix(tensor_in, matrix_out)
      TYPE(dbcsr_t_type), INTENT(IN)        :: tensor_in
      TYPE(dbcsr_type), INTENT(INOUT)       :: matrix_out
      TYPE(dbcsr_t_iterator_type)           :: iter
      INTEGER, ALLOCATABLE, DIMENSION(:) :: blk_ind_1, blk_ind_2

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_reserve_blocks_tensor_to_matrix', &
                                     routineP = moduleN//':'//routineN
      INTEGER :: handle, blk, iblk, nblk
      INTEGER, DIMENSION(2)              :: ind_2d

      CALL timeset(routineN, handle)

      nblk = dbcsr_t_get_num_blocks(tensor_in)
      ALLOCATE (blk_ind_1(nblk), blk_ind_2(nblk))
      CALL dbcsr_t_iterator_start(iter, tensor_in)
      DO iblk = 1, nblk
         CALL dbcsr_t_iterator_next_block(iter, ind_2d, blk)
         blk_ind_1(iblk) = ind_2d(1); blk_ind_2(iblk) = ind_2d(2)
      ENDDO
      CALL dbcsr_t_iterator_stop(iter)

      CALL dbcsr_reserve_blocks(matrix_out, blk_ind_1, blk_ind_2)
      CALL dbcsr_finalize(matrix_out)

      CALL timestop(handle)
   END SUBROUTINE

! **************************************************************************************************
!> \brief indices of non-zero blocks
!> \param tensor ...
!> \param blk_ind ...
! **************************************************************************************************
   SUBROUTINE dbcsr_t_reserved_block_indices(tensor, blk_ind)
      TYPE(dbcsr_t_type), INTENT(IN)            :: tensor
      INTEGER, ALLOCATABLE, DIMENSION(:)        :: ${varlist("blk_ind")}$
      INTEGER                                   :: blk, iblk, nblk
      TYPE(dbcsr_t_iterator_type)               :: iterator
      INTEGER, DIMENSION(ndims_tensor(tensor))  :: ind_nd
      TYPE(array_list), INTENT(OUT)             :: blk_ind

      DBCSR_ASSERT(tensor%valid)

      nblk = dbcsr_t_get_num_blocks(tensor)

#:for dim in range(1, maxdim+1)
      IF (ndims_tensor(tensor) >= ${dim}$) ALLOCATE (blk_ind_${dim}$ (nblk))
#:endfor

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO iblk = 1, nblk
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk)
#:for dim in range(1, maxdim+1)
         IF (ndims_tensor(tensor) >= ${dim}$) blk_ind_${dim}$ (iblk) = ind_nd(${dim}$)
#:endfor
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)

#:for ndim in ndims
      IF (ndims_tensor(tensor) == ${ndim}$) THEN
         CALL create_array_list(blk_ind, ndims_tensor(tensor), ${varlist("blk_ind", nmax=ndim)}$)
      ENDIF
#:endfor

   END SUBROUTINE

#:for dparam, dtype, dsuffix in dtype_float_list
! **************************************************************************************************
!> \brief Create block from array, array can be n-dimensional.
!> \param block ...
!> \param sizes ...
!> \param array ...
! **************************************************************************************************
   SUBROUTINE create_block_data_${dsuffix}$ (block, sizes, array)
      TYPE(block_nd), INTENT(OUT)                       :: block
      INTEGER, DIMENSION(:), INTENT(IN)                 :: sizes
      ${dtype}$, DIMENSION(PRODUCT(sizes)), INTENT(IN) :: array

      ASSOCIATE (blk=>block%${dsuffix}$)
         block%data_type = ${dparam}$
         CALL allocate_any(blk%sizes, source=sizes)
         CALL allocate_any(blk%blk, source=array)
      END ASSOCIATE
   END SUBROUTINE
#:endfor

#:for dparam, dtype, dsuffix in dtype_float_list
! **************************************************************************************************
!> \brief Create and allocate block, but no data.
!> \param block ...
!> \param sizes ...
! **************************************************************************************************
   SUBROUTINE create_block_nodata_${dsuffix}$ (block, sizes)
      INTEGER, INTENT(IN), DIMENSION(:)       :: sizes
      TYPE(block_nd_${dsuffix}$), INTENT(OUT) :: block
      CALL allocate_any(block%sizes, source=sizes)
      ALLOCATE (block%blk(PRODUCT(sizes)))
   END SUBROUTINE
#:endfor

#:for dparam, dtype, dsuffix in dtype_float_list
! **************************************************************************************************
!> \brief ...
!> \param block ...
! **************************************************************************************************
   SUBROUTINE destroy_block_${dsuffix}$ (block)
      TYPE(block_nd_${dsuffix}$), INTENT(INOUT) :: block
      DEALLOCATE (block%blk)
      DEALLOCATE (block%sizes)
   END SUBROUTINE
#:endfor

! **************************************************************************************************
!> \brief Generic implementation of dbcsr_t_get_block (arbitrary tensor rank and arbitrary datatype)
!> \param tensor ...
!> \param ind block index
!> \param block block to get
!> \param found whether block was found
! **************************************************************************************************
   SUBROUTINE dbcsr_t_get_anyd_block(tensor, ind, block, found)
      TYPE(dbcsr_t_type), INTENT(INOUT)            :: tensor
      INTEGER, DIMENSION(ndims_tensor(tensor)), &
         INTENT(IN)                                :: ind
      TYPE(block_nd), INTENT(OUT)                  :: block
      LOGICAL, INTENT(OUT)                         :: found
      INTEGER                                   :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_get_anyd_block', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)
      SELECT CASE (dbcsr_t_get_data_type(tensor))
#:for dparam, dtype, dsuffix in dtype_float_list
      CASE (${dparam}$)
         CALL dbcsr_t_get_anyd_block_${dsuffix}$ (tensor, ind, block, found)
#:endfor
      END SELECT
      CALL timestop(handle)
   END SUBROUTINE

! **************************************************************************************************
!> \brief Generic implementation of dbcsr_t_put_block (arbitrary tensor rank and arbitrary datatype)
!> \param tensor ...
!> \param ind block index
!> \param block block to put
!> \param summation whether block should be summed to existing block
!> \param scale scaling factor
! **************************************************************************************************
   SUBROUTINE dbcsr_t_put_anyd_block(tensor, ind, block, summation, scale)
      TYPE(dbcsr_t_type), INTENT(INOUT)            :: tensor
      INTEGER, DIMENSION(ndims_tensor(tensor)), &
         INTENT(IN)                                :: ind
      TYPE(block_nd), INTENT(IN)                   :: block
      LOGICAL, INTENT(IN), OPTIONAL                :: summation
      TYPE(dbcsr_scalar_type), INTENT(IN), OPTIONAL :: scale
      INTEGER                                   :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_put_anyd_block', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)
      SELECT CASE (block%data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
      CASE (${dparam}$)
         IF (.NOT. PRESENT(scale)) THEN
            CALL dbcsr_t_put_anyd_block_${dsuffix}$ (tensor, ind, block%${dsuffix}$, summation)
         ELSE
            CALL dbcsr_t_put_anyd_block_${dsuffix}$ (tensor, ind, block%${dsuffix}$, summation, scale=scale%${dsuffix}$)
         ENDIF
#:endfor
      END SELECT

      CALL timestop(handle)
   END SUBROUTINE

#:for dparam, dtype, dsuffix in dtype_float_list
! **************************************************************************************************
!> \brief Generic implementation of dbcsr_t_put_block, template for datatype
!> \param tensor ...
!> \param ind block index
!> \param block block to put
!> \param summation whether block should be summed to existing block
!> \param scale scaling factor
! **************************************************************************************************
   SUBROUTINE dbcsr_t_put_anyd_block_${dsuffix}$ (tensor, ind, block, summation, scale)
      TYPE(block_nd_${dsuffix}$), INTENT(IN)       :: block
      TYPE(dbcsr_t_type), INTENT(INOUT)            :: tensor
      LOGICAL, INTENT(IN), OPTIONAL                :: summation
      ${dtype}$, INTENT(IN), OPTIONAL :: scale
      INTEGER, DIMENSION(ndims_tensor(tensor)), &
         INTENT(IN)                                :: ind

      SELECT CASE (ndims_tensor(tensor))
#:for ndim in ndims
      CASE (${ndim}$)
         CALL dbcsr_t_put_${ndim}$d_block_${dsuffix}$ (tensor, ind, block%sizes, block%blk, summation=summation, scale=scale)
#:endfor
      END SELECT
   END SUBROUTINE
#:endfor

#:for dparam, dtype, dsuffix in dtype_float_list
! **************************************************************************************************
!> \brief Generic implementation of dbcsr_t_get_block (arbitrary tensor rank)
!> \param tensor ...
!> \param ind block index
!> \param block block to get
!> \param found whether block was found
! **************************************************************************************************
   SUBROUTINE dbcsr_t_get_anyd_block_${dsuffix}$ (tensor, ind, block, found)
      TYPE(block_nd), INTENT(OUT)                  :: block
      LOGICAL, INTENT(OUT)                         :: found
      TYPE(dbcsr_t_type), INTENT(INOUT)            :: tensor
      INTEGER, DIMENSION(ndims_tensor(tensor)), &
         INTENT(IN)                                :: ind
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
! **************************************************************************************************
!> \brief Template for dbcsr_t_put_block.
!> \param tensor ...
!> \param ind block index
!> \param sizes block size
!> \param block block to put
!> \param summation whether block should be summed to existing block
!> \param scale scaling factor
! **************************************************************************************************
   SUBROUTINE dbcsr_t_put_${ndim}$d_block_${dsuffix}$ (tensor, ind, sizes, block, summation, scale)
      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(ndims_tensor(tensor)), INTENT(IN) :: ind
      INTEGER, DIMENSION(ndims_tensor(tensor)), INTENT(IN) :: sizes
      ${dtype}$, DIMENSION(${arrlist("sizes", nmax=ndim)}$), &
         INTENT(IN)                                         :: block
      LOGICAL, INTENT(IN), OPTIONAL                         :: summation
      ${dtype}$, INTENT(IN), OPTIONAL                       :: scale

      INTEGER, ALLOCATABLE, DIMENSION(:)                    :: map1_2d, map2_2d
      INTEGER(KIND=int_8), DIMENSION(2)                     :: ind_2d
      INTEGER, DIMENSION(2)                                 :: dims_2d
      ${dtype}$, ALLOCATABLE, DIMENSION(:, :)               :: block_2d
      TYPE(nd_to_2d_mapping)                                :: map_blk
      INTEGER                                   :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_put_${ndim}$d_block_${dsuffix}$', &
                                     routineP = moduleN//':'//routineN
      LOGICAL :: found
      ${dtype}$, DIMENSION(${arrlist("sizes", nmax=ndim)}$) &
         :: block_check
      LOGICAL, PARAMETER :: debug = .FALSE.

      IF (debug) THEN
         CALL dbcsr_t_get_block(tensor, ind, sizes, block_check, found=found)
         DBCSR_ASSERT(found)
      ENDIF

      CALL timeset(routineN, handle)
      ! reshape block
      CALL get_mapping_info(tensor%nd_index_blk, map1_2d=map1_2d, map2_2d=map2_2d)
      CALL create_nd_to_2d_mapping(map_blk, sizes, map1_2d, map2_2d)
      CALL get_mapping_info(map_blk, dims_2d=dims_2d)
      CALL allocate_any(block_2d, shape_spec=dims_2d)
      CALL reshape_nd_to_2d_block(map_blk, block, block_2d)

      ! convert block index
      ind_2d(:) = get_2d_indices(tensor%nd_index_blk, ind)

      CALL dbcsr_tas_put_block(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d, summation=summation, &
                               scale=scale)

      CALL timestop(handle)
   END SUBROUTINE
#:endfor
#:endfor

#:for dparam, dtype, dsuffix in dtype_float_list
#:for ndim in ndims
! **************************************************************************************************
!> \brief allocate and get block
!> \param tensor ...
!> \param ind block index
!> \param block block to get
!> \param found whether block was found
! **************************************************************************************************
   SUBROUTINE dbcsr_t_allocate_and_get_${ndim}$d_block_${dsuffix}$ (tensor, ind, block, found)
      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(ndims_tensor(tensor)), INTENT(IN)  :: ind
      ${dtype}$, DIMENSION(${shape_colon(ndim)}$), &
         ALLOCATABLE, INTENT(OUT)                           :: block
      LOGICAL, INTENT(OUT)                                  :: found
      INTEGER, DIMENSION(${ndim}$)                          :: blk_size

      CALL dbcsr_t_blk_sizes(tensor, ind, blk_size)
      CALL allocate_any(block, shape_spec=blk_size)
      CALL dbcsr_t_get_${ndim}$d_block_${dsuffix}$ (tensor, ind, blk_size, block, found)

   END SUBROUTINE
#:endfor
#:endfor

#:for dparam, dtype, dsuffix in dtype_float_list
#:for ndim in ndims
! **************************************************************************************************
!> \brief Template for dbcsr_t_get_block.
!> \param tensor ...
!> \param ind block index
!> \param sizes block size
!> \param block block to get
!> \param found whether block was found
! **************************************************************************************************
   SUBROUTINE dbcsr_t_get_${ndim}$d_block_${dsuffix}$ (tensor, ind, sizes, block, found)
      TYPE(dbcsr_t_type), INTENT(INOUT)                     :: tensor
      INTEGER, DIMENSION(ndims_tensor(tensor)), INTENT(IN) :: ind
      INTEGER, DIMENSION(ndims_tensor(tensor)), INTENT(IN) :: sizes
      ${dtype}$, DIMENSION(${arrlist("sizes", nmax=ndim)}$), &
         INTENT(OUT)                                        :: block
      LOGICAL, INTENT(OUT)                                  :: found

      INTEGER, ALLOCATABLE, DIMENSION(:)                    :: map1_2d, map2_2d
      INTEGER(KIND=int_8), DIMENSION(2)                                 :: ind_2d
      ${dtype}$, DIMENSION(:, :), POINTER                    :: block_2d_ptr
      ${dtype}$, DIMENSION(:, :), ALLOCATABLE                :: block_2d
      TYPE(nd_to_2d_mapping)                                :: map_blk
      LOGICAL                                               :: tr
      INTEGER                                   :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_get_${ndim}$d_block_${dsuffix}$', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)

      NULLIFY (block_2d_ptr)

      ! convert block index
      ind_2d(:) = get_2d_indices(tensor%nd_index_blk, ind)
      CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)
      DBCSR_ASSERT(.NOT. tr)

      IF (found) THEN
         ! convert pointer to allocatable
         CALL allocate_any(block_2d, source=block_2d_ptr)

         CALL get_mapping_info(tensor%nd_index_blk, map1_2d=map1_2d, map2_2d=map2_2d)
         CALL create_nd_to_2d_mapping(map_blk, sizes, map1_2d, map2_2d)
         CALL reshape_2d_to_nd_block(map_blk, block_2d, block)
      ENDIF

      CALL timestop(handle)
   END SUBROUTINE
#:endfor
#:endfor

#:for dparam, dtype, dsuffix in dtype_float_list
#:for ndim in ndims

! **************************************************************************************************
!> \brief reshape matrix to tensor block
!> \param map ...
!> \param block_2d ...
!> \param block ...
! **************************************************************************************************
   SUBROUTINE reshape_block_m2t_${ndim}$d_${dsuffix}$ (map, block_2d, block)
      TYPE(nd_to_2d_mapping), INTENT(IN)                        :: map
      ${dtype}$, DIMENSION(:, :), INTENT(IN)                    :: block_2d
      ${dtype}$, DIMENSION(${arrlist("map%dims_nd", nmax=ndim)}$), &
         INTENT(OUT)                                            :: block
      INTEGER, DIMENSION(ndims_mapping(map))                    :: map_tmp

      map_tmp(:) = [map%map1_2d, map%map2_2d]
      block(${shape_colon(ndim)}$) = RESHAPE(block_2d, SHAPE=SHAPE(block), ORDER=map_tmp)
   END SUBROUTINE
#:endfor
#:endfor

#:for dparam, dtype, dsuffix in dtype_float_list
#:for ndim in ndims
! **************************************************************************************************
!> \brief reshape tensor to matrix block
!> \param map ...
!> \param block ...
!> \param block_2d ...
! **************************************************************************************************
   SUBROUTINE reshape_block_t2m_${ndim}$d_${dsuffix}$ (map, block, block_2d)
      TYPE(nd_to_2d_mapping), INTENT(IN)                                 :: map
      ${dtype}$, DIMENSION(${arrlist("map%dims_nd", nmax=ndim)}$), INTENT(IN) :: block
      ${dtype}$, DIMENSION(map%dims_2d(1), map%dims_2d(2)), INTENT(OUT)  :: block_2d

      INTEGER, DIMENSION(ndims_mapping(map))                             :: shape_in, shape_reordered
      ${dtype}$, ALLOCATABLE, DIMENSION(${shape_colon(ndim)}$)           :: block_tmp

      shape_in = SHAPE(block)
      shape_reordered(map%map_nd) = shape_in
      CALL allocate_any(block_tmp, source=block, order=map%map_nd)
      block_2d(:, :) = RESHAPE(block_tmp, SHAPE=SHAPE(block_2d))
   END SUBROUTINE
#:endfor
#:endfor

END MODULE
