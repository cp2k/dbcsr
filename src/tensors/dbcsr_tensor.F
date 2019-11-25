!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tensor
   !! DBCSR tensor framework for block-sparse tensor contraction.
   !! Representation of n-rank tensors as DBCSR tall-and-skinny matrices.
   !! Support for arbitrary redistribution between different representations.
   !! Support for arbitrary tensor contractions
   !! \todo implement checks and error messages

#:include "dbcsr_tensor.fypp"
#:set maxdim = maxrank
#:set ndims = range(2,maxdim+1)

   USE dbcsr_allocate_wrap, ONLY: &
      allocate_any
   USE dbcsr_array_list_methods, ONLY: &
      get_arrays, reorder_arrays, get_ith_array, array_list, array_sublist, check_equal, array_eq_i
   USE dbcsr_api, ONLY: &
      dbcsr_type, dbcsr_iterator_type, dbcsr_iterator_blocks_left, &
      dbcsr_iterator_next_block, dbcsr_iterator_start, dbcsr_iterator_stop, &
      dbcsr_transpose, dbcsr_no_transpose, dbcsr_scalar, dbcsr_put_block, &
      ${uselist(dtype_float_param)}$, dbcsr_clear, &
      dbcsr_release, dbcsr_desymmetrize, dbcsr_has_symmetry
   USE dbcsr_tas_types, ONLY: &
      dbcsr_tas_split_info
   USE dbcsr_tas_base, ONLY: &
      dbcsr_tas_copy, dbcsr_tas_finalize, dbcsr_tas_get_data_type, dbcsr_tas_get_info, dbcsr_tas_info
   USE dbcsr_tas_mm, ONLY: dbcsr_tas_multiply
   USE dbcsr_tensor_block, ONLY: &
      dbcsr_t_iterator_type, dbcsr_t_get_block, dbcsr_t_put_block, dbcsr_t_iterator_start, &
      dbcsr_t_iterator_blocks_left, dbcsr_t_iterator_stop, dbcsr_t_iterator_next_block, &
      ndims_iterator, dbcsr_t_reserve_blocks, block_nd
   USE dbcsr_tensor_index, ONLY: &
      dbcsr_t_get_mapping_info, nd_to_2d_mapping, dbcsr_t_inverse_order, permute_index
   USE dbcsr_tensor_types, ONLY: &
      dbcsr_t_create, dbcsr_t_get_data_type, dbcsr_t_type, ndims_tensor, dims_tensor, &
      dbcsr_t_distribution_type, dbcsr_t_distribution, dbcsr_t_nd_mp_comm, dbcsr_t_destroy, &
      dbcsr_t_distribution_destroy, dbcsr_t_distribution_new, dbcsr_t_get_stored_coordinates, &
      blk_dims_tensor, dbcsr_t_hold, dbcsr_t_pgrid_type, mp_environ_pgrid, dbcsr_t_filter, &
      dbcsr_t_clear, dbcsr_t_finalize, dbcsr_t_get_num_blocks, dbcsr_t_scale, &
      dbcsr_t_get_num_blocks_total, dbcsr_t_get_info
   USE dbcsr_kinds, ONLY: &
      ${uselist(dtype_float_prec)}$, default_string_length, int_8
   USE dbcsr_mpiwrap, ONLY: &
      mp_environ, mp_max, mp_sum
   USE dbcsr_toollib, ONLY: &
      sort
   USE dbcsr_tensor_reshape, ONLY: &
      dbcsr_t_reshape
   USE dbcsr_mpiwrap, ONLY: &
      mp_comm_free
   USE dbcsr_tas_split, ONLY: &
      dbcsr_tas_mp_comm, rowsplit, colsplit, dbcsr_tas_info_hold, dbcsr_tas_release_info
   USE dbcsr_data_types, ONLY: &
      dbcsr_scalar_type
   USE dbcsr_tensor_split, ONLY: &
      dbcsr_t_split_copyback, dbcsr_t_make_compatible_blocks, dbcsr_t_crop
   USE dbcsr_tensor_io, ONLY: &
      dbcsr_t_write_tensor_info, dbcsr_t_write_tensor_dist
   USE dbcsr_dist_operations, ONLY: &
      checker_tr
   USE dbcsr_toollib, ONLY: &
      swap

#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tensor'

   PUBLIC :: &
      dbcsr_t_contract, &
      dbcsr_t_copy, &
      dbcsr_t_dims, &
      dbcsr_t_get_block, &
      dbcsr_t_get_stored_coordinates, &
      dbcsr_t_inverse_order, &
      dbcsr_t_iterator_blocks_left, &
      dbcsr_t_iterator_next_block, &
      dbcsr_t_iterator_start, &
      dbcsr_t_iterator_stop, &
      dbcsr_t_iterator_type, &
      dbcsr_t_ndims, &
      dbcsr_t_put_block, &
      dbcsr_t_reserve_blocks, &
      dbcsr_t_copy_matrix_to_tensor, &
      dbcsr_t_copy_tensor_to_matrix, &
      dbcsr_t_need_contract

   INTERFACE dbcsr_t_ndims
      MODULE PROCEDURE ndims_tensor
   END INTERFACE

   INTERFACE dbcsr_t_dims
      MODULE PROCEDURE dims_tensor
   END INTERFACE

CONTAINS

   SUBROUTINE dbcsr_t_copy(tensor_in, tensor_out, order, summation, move_data, unit_nr)
      !! Copy tensor data.
      !! Redistributes tensor data according to distributions of target and source tensor.
      !! Permutes tensor index according to `order` argument (if present).
      !! Source and target tensor formats are arbitrary as long as the following requirements are met:
      !! * source and target tensors have the same number of blocks in each dimension and the same block sizes.
      !! If `order` argument is present, this must be the case after index permutation.
      !! OR
      !! * target tensor is not yet created, in this case an exact copy of source tensor is returned.

      TYPE(dbcsr_t_type), INTENT(INOUT), TARGET      :: tensor_in, tensor_out
         !! Source
         !! Target
      INTEGER, DIMENSION(ndims_tensor(tensor_in)), &
         INTENT(IN), OPTIONAL                        :: order
         !! Permutation of target tensor index. Exact same convention as order argument of RESHAPE intrinsic
      LOGICAL, INTENT(IN), OPTIONAL                  :: summation, move_data
      INTEGER, INTENT(IN), OPTIONAL                  :: unit_nr

      TYPE(dbcsr_t_type), POINTER                    :: in_tmp_1 => NULL(), in_tmp_2 => NULL(), out_tmp_1 => NULL()
      INTEGER                                        :: handle
      INTEGER, DIMENSION(:), ALLOCATABLE             :: map1_in_1, map1_in_2, map2_in_1, map2_in_2

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_copy', &
                                     routineP = moduleN//':'//routineN
      LOGICAL                                        :: dist_compatible, summation_prv, new_in_1, &
                                                        new_in_2, new_out_1, block_compatible, &
                                                        move_prv

      CALL timeset(routineN, handle)

      IF (PRESENT(move_data)) THEN
         move_prv = move_data
      ELSE
         move_prv = .FALSE.
      ENDIF

      DBCSR_ASSERT(tensor_out%valid)

      dist_compatible = .FALSE.
      block_compatible = .FALSE.
      new_in_1 = .FALSE.
      new_in_2 = .FALSE.
      new_out_1 = .FALSE.

      IF (PRESENT(summation)) THEN
         summation_prv = summation
      ELSE
         summation_prv = .FALSE.
      ENDIF

      IF (PRESENT(order)) THEN
         ALLOCATE (in_tmp_1)
         CALL dbcsr_t_permute_index(tensor_in, in_tmp_1, order)
         new_in_1 = .TRUE.
      ELSE
         in_tmp_1 => tensor_in
      ENDIF

      block_compatible = check_equal(in_tmp_1%blk_sizes, tensor_out%blk_sizes)
      IF (.NOT. block_compatible) THEN
         ALLOCATE (in_tmp_2, out_tmp_1)
         new_in_2 = .TRUE.; new_out_1 = .TRUE.
         CALL dbcsr_t_make_compatible_blocks(in_tmp_1, tensor_out, in_tmp_2, out_tmp_1, &
                                             nodata2=.NOT. summation_prv, move_data=move_data)
      ELSE
         in_tmp_2 => in_tmp_1
         out_tmp_1 => tensor_out
      ENDIF

      CALL dbcsr_t_get_mapping_info(in_tmp_2%nd_index, map1_2d=map1_in_1, map2_2d=map1_in_2)
      CALL dbcsr_t_get_mapping_info(out_tmp_1%nd_index, map1_2d=map2_in_1, map2_2d=map2_in_2)

      IF (.NOT. PRESENT(order)) THEN
         IF (array_eq_i(map1_in_1, map2_in_1) .AND. array_eq_i(map1_in_2, map2_in_2)) THEN
            dist_compatible = check_equal(in_tmp_2%nd_dist, out_tmp_1%nd_dist)
         ENDIF
      ENDIF

      IF (dist_compatible) THEN
         CALL dbcsr_tas_copy(out_tmp_1%matrix_rep, in_tmp_2%matrix_rep, summation)
         IF (move_prv) CALL dbcsr_t_clear(in_tmp_2)
      ELSE
         CALL dbcsr_t_reshape(in_tmp_2, out_tmp_1, summation, move_data=move_data)
      ENDIF

      IF (new_in_1) THEN
         CALL dbcsr_t_destroy(in_tmp_1)
         DEALLOCATE (in_tmp_1)
      ENDIF

      IF (new_in_2) THEN
         CALL dbcsr_t_destroy(in_tmp_2)
         DEALLOCATE (in_tmp_2)
      ENDIF

      IF (new_out_1) THEN
         IF (PRESENT(unit_nr)) THEN
            CALL dbcsr_t_write_tensor_dist(out_tmp_1, unit_nr)
         ENDIF
         CALL dbcsr_t_split_copyback(out_tmp_1, tensor_out, summation)
         CALL dbcsr_t_destroy(out_tmp_1)
         DEALLOCATE (out_tmp_1)
      ENDIF

      CALL timestop(handle)

   END SUBROUTINE

   SUBROUTINE dbcsr_t_copy_matrix_to_tensor(matrix_in, tensor_out, summation)
      !! copy matrix to tensor.

      TYPE(dbcsr_type), TARGET, INTENT(IN)               :: matrix_in
      TYPE(dbcsr_t_type), INTENT(INOUT)                  :: tensor_out
      LOGICAL, INTENT(IN), OPTIONAL                      :: summation
         !! tensor_out = tensor_out + matrix_in
      TYPE(dbcsr_type), POINTER                          :: matrix_in_desym

      INTEGER, DIMENSION(2)                              :: ind_2d
      REAL(KIND=real_8), ALLOCATABLE, DIMENSION(:, :)    :: block_arr
      REAL(KIND=real_8), DIMENSION(:, :), POINTER        :: block => NULL()
      TYPE(dbcsr_iterator_type)                          :: iter
      LOGICAL                                            :: tr

      INTEGER                                            :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_copy_matrix_to_tensor', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)
      DBCSR_ASSERT(tensor_out%valid)

      IF(dbcsr_has_symmetry(matrix_in)) THEN
         ALLOCATE(matrix_in_desym)
         CALL dbcsr_desymmetrize(matrix_in, matrix_in_desym)
      ELSE
         matrix_in_desym => matrix_in
      ENDIF

      IF (PRESENT(summation)) THEN
         IF (.NOT. summation) CALL dbcsr_t_clear(tensor_out)
      ELSE
         CALL dbcsr_t_clear(tensor_out)
      ENDIF

      CALL dbcsr_t_reserve_blocks(matrix_in_desym, tensor_out)

      CALL dbcsr_iterator_start(iter, matrix_in_desym)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, ind_2d(1), ind_2d(2), block, tr)
         CALL allocate_any(block_arr, source=block)
         CALL dbcsr_t_put_block(tensor_out, ind_2d, SHAPE(block_arr), block_arr, summation=summation)
         DEALLOCATE (block_arr)
      ENDDO
      CALL dbcsr_iterator_stop(iter)

      IF(dbcsr_has_symmetry(matrix_in)) THEN
         CALL dbcsr_release(matrix_in_desym)
         DEALLOCATE(matrix_in_desym)
      ENDIF

      CALL timestop(handle)

   END SUBROUTINE

   SUBROUTINE dbcsr_t_copy_tensor_to_matrix(tensor_in, matrix_out, summation)
      !! copy tensor to matrix

      TYPE(dbcsr_t_type), INTENT(INOUT)      :: tensor_in
      TYPE(dbcsr_type), INTENT(INOUT)           :: matrix_out
      LOGICAL, INTENT(IN), OPTIONAL          :: summation
         !! matrix_out = matrix_out + tensor_in
      TYPE(dbcsr_t_iterator_type)            :: iter
      INTEGER                                :: blk, handle
      INTEGER, DIMENSION(2)                  :: ind_2d
      REAL(KIND=real_8), DIMENSION(:, :), ALLOCATABLE :: block
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_copy_tensor_to_matrix', &
                                     routineP = moduleN//':'//routineN
      LOGICAL :: found

      CALL timeset(routineN, handle)

      IF (PRESENT(summation)) THEN
         IF (.NOT. summation) CALL dbcsr_clear(matrix_out)
      ELSE
         CALL dbcsr_clear(matrix_out)
      ENDIF

      CALL dbcsr_t_reserve_blocks(tensor_in, matrix_out)

      CALL dbcsr_t_iterator_start(iter, tensor_in)
      DO WHILE (dbcsr_t_iterator_blocks_left(iter))
         CALL dbcsr_t_iterator_next_block(iter, ind_2d, blk)
         IF(dbcsr_has_symmetry(matrix_out) .AND. checker_tr(ind_2d(1), ind_2d(2))) CYCLE

         CALL dbcsr_t_get_block(tensor_in, ind_2d, block, found)
         DBCSR_ASSERT(found)

         IF (dbcsr_has_symmetry(matrix_out) .AND. ind_2d(1) > ind_2d(2)) THEN
            CALL dbcsr_put_block(matrix_out, ind_2d(2), ind_2d(1), TRANSPOSE(block), summation=summation)
         ELSE
            CALL dbcsr_put_block(matrix_out, ind_2d(1), ind_2d(2), block, summation=summation)
         ENDIF
         DEALLOCATE (block)
      ENDDO
      CALL dbcsr_t_iterator_stop(iter)

      CALL timestop(handle)

   END SUBROUTINE

   SUBROUTINE dbcsr_t_contract(alpha, tensor_1, tensor_2, beta, tensor_3, &
                               contract_1, notcontract_1, &
                               contract_2, notcontract_2, &
                               map_1, map_2, &
                               bounds_1, bounds_2, bounds_3, &
                               optimize_dist, pgrid_opt_1, pgrid_opt_2, pgrid_opt_3, &
                               filter_eps, flop, move_data, unit_nr, log_verbose)
      !! Contract tensors by multiplying matrix representations.
      !! tensor_3(map_1, map_2) := alpha * tensor_1(notcontract_1, contract_1)
      !! * tensor_2(contract_2, notcontract_2)
      !! + beta * tensor_3(map_1, map_2)

      TYPE(dbcsr_scalar_type), INTENT(IN)            :: alpha
      TYPE(dbcsr_t_type), INTENT(INOUT), TARGET      :: tensor_1, tensor_2
         !! first tensor (in)
         !! second tensor (in)
      TYPE(dbcsr_scalar_type), INTENT(IN)            :: beta
      INTEGER, DIMENSION(:), INTENT(IN)              :: contract_1, contract_2, map_1, map_2
         !! indices of tensor_1 to contract
         !! indices of tensor_2 to contract (1:1 with contract_1)
         !! which indices of tensor_3 map to non-contracted indices of tensor_1 (1:1 with notcontract_1)
         !! which indices of tensor_3 map to non-contracted indices of tensor_2 (1:1 with notcontract_2)
      INTEGER, DIMENSION(:), INTENT(IN)              :: notcontract_1, notcontract_2
         !! indices of tensor_1 not to contract
         !! indices of tensor_2 not to contract
      TYPE(dbcsr_t_type), INTENT(INOUT), TARGET      :: tensor_3
         !! contracted tensor (out)
      INTEGER, DIMENSION(2, SIZE(contract_1)), &
         INTENT(IN), OPTIONAL                        :: bounds_1
         !! bounds corresponding to contract_1 AKA contract_2
      INTEGER, DIMENSION(2, SIZE(notcontract_1)), &
         INTENT(IN), OPTIONAL                        :: bounds_2
         !! bounds corresponding to notcontract_1
      INTEGER, DIMENSION(2, SIZE(notcontract_2)), &
         INTENT(IN), OPTIONAL                        :: bounds_3
         !! bounds corresponding to notcontract_2
      LOGICAL, INTENT(IN), OPTIONAL                  :: optimize_dist
         !! Whether distribution should be optimized internally. In the current implementation this guarantees optimal parameters
         !! only for dense matrices.
      TYPE(dbcsr_t_pgrid_type), INTENT(OUT), &
         POINTER, OPTIONAL                           :: pgrid_opt_1, pgrid_opt_2, pgrid_opt_3
         !! Optionally return optimal process grid for tensor_1. This can be used to choose optimal process grids for subsequent
         !! tensor contractions with tensors of similar shape and sparsity. Under some conditions, pgrid_opt_1 can not be returned,
         !! in this case the pointer is not associated.
         !! Optionally return optimal process grid for tensor_2.
         !! Optionally return optimal process grid for tensor_3.
      REAL(KIND=real_8), INTENT(IN), OPTIONAL        :: filter_eps
         !! As in DBCSR mm
      INTEGER(KIND=int_8), INTENT(OUT), OPTIONAL     :: flop
         !! As in DBCSR mm
      LOGICAL, INTENT(IN), OPTIONAL                  :: move_data
         !! memory optimization: transfer data such that tensor_1 and tensor_2 may be empty on return
      INTEGER, OPTIONAL, INTENT(IN)                  :: unit_nr
         !! output unit for logging
      LOGICAL, INTENT(IN), OPTIONAL                  :: log_verbose
         !! verbose logging (for testing only)

      TYPE(dbcsr_t_type), POINTER                    :: tensor_contr_1, tensor_contr_2, tensor_contr_3
      TYPE(dbcsr_t_type)                             :: tensor_algn_1, tensor_algn_2, tensor_algn_3
      TYPE(dbcsr_t_type), POINTER                    :: tensor_crop_1, tensor_crop_2
      LOGICAL                                        :: assert_stmt
      INTEGER                                        :: data_type, handle, max_mm_dim, max_tensor, mp_comm
      INTEGER, DIMENSION(SIZE(contract_1))           :: contract_1_mod
      INTEGER, DIMENSION(SIZE(notcontract_1))        :: notcontract_1_mod
      INTEGER, DIMENSION(SIZE(contract_2))           :: contract_2_mod
      INTEGER, DIMENSION(SIZE(notcontract_2))        :: notcontract_2_mod
      INTEGER, DIMENSION(SIZE(map_1))                :: map_1_mod
      INTEGER, DIMENSION(SIZE(map_2))                :: map_2_mod
      CHARACTER(LEN=1)                               :: trans_1, trans_2, trans_3
      LOGICAL                                        :: new_1, new_2, new_3, move_data_1, move_data_2
      INTEGER                                        :: ndims1, ndims2, ndims3
      INTEGER                                        :: occ_1, occ_2
      INTEGER, DIMENSION(:), ALLOCATABLE             :: dims1, dims2, dims3

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_contract', &
                                     routineP = moduleN//':'//routineN
      CHARACTER(LEN=1), DIMENSION(:), ALLOCATABLE    :: indchar1, indchar2, indchar3, indchar1_mod, &
                                                        indchar2_mod, indchar3_mod
      CHARACTER(LEN=1), DIMENSION(15) :: alph = &
                                         ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
      INTEGER, DIMENSION(2, dbcsr_t_ndims(tensor_1)) :: bounds_t1
      INTEGER, DIMENSION(2, dbcsr_t_ndims(tensor_2)) :: bounds_t2
      LOGICAL                                        :: do_crop_1, do_crop_2
      TYPE(dbcsr_tas_split_info), POINTER            :: split_opt

      CALL timeset(routineN, handle)

      NULLIFY (tensor_contr_1, tensor_contr_2, tensor_contr_3, tensor_crop_1, tensor_crop_2, split_opt)

      DBCSR_ASSERT(tensor_1%valid)
      DBCSR_ASSERT(tensor_2%valid)
      DBCSR_ASSERT(tensor_3%valid)

      assert_stmt = SIZE(contract_1) .EQ. SIZE(contract_2)
      DBCSR_ASSERT(assert_stmt)

      assert_stmt = SIZE(map_1) .EQ. SIZE(notcontract_1)
      DBCSR_ASSERT(assert_stmt)

      assert_stmt = SIZE(map_2) .EQ. SIZE(notcontract_2)
      DBCSR_ASSERT(assert_stmt)

      assert_stmt = SIZE(notcontract_1) + SIZE(contract_1) .EQ. dbcsr_t_ndims(tensor_1)
      DBCSR_ASSERT(assert_stmt)

      assert_stmt = SIZE(notcontract_2) + SIZE(contract_2) .EQ. dbcsr_t_ndims(tensor_2)
      DBCSR_ASSERT(assert_stmt)

      assert_stmt = SIZE(map_1) + SIZE(map_2) .EQ. dbcsr_t_ndims(tensor_3)
      DBCSR_ASSERT(assert_stmt)

      assert_stmt = dbcsr_t_get_data_type(tensor_1) .EQ. dbcsr_t_get_data_type(tensor_2)
      DBCSR_ASSERT(assert_stmt)

      IF (PRESENT(move_data)) THEN
         move_data_1 = move_data
         move_data_2 = move_data
      ELSE
         move_data_1 = .FALSE.
         move_data_2 = .FALSE.
      ENDIF

      CALL dbcsr_t_map_bounds_to_tensors(tensor_1, tensor_2, &
                                         contract_1, notcontract_1, &
                                         contract_2, notcontract_2, &
                                         bounds_t1, bounds_t2, &
                                         bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                         do_crop_1=do_crop_1, do_crop_2=do_crop_2)

      IF (do_crop_1) THEN
         ALLOCATE (tensor_crop_1)
         CALL dbcsr_t_crop(tensor_1, tensor_crop_1, bounds_t1, move_data=move_data_1)
         move_data_1 = .TRUE.
      ELSE
         tensor_crop_1 => tensor_1
      ENDIF

      IF (do_crop_2) THEN
         ALLOCATE (tensor_crop_2)
         CALL dbcsr_t_crop(tensor_2, tensor_crop_2, bounds_t2, move_data=move_data_2)
         move_data_2 = .TRUE.
      ELSE
         tensor_crop_2 => tensor_2
      ENDIF

      ! shortcut for empty tensors
      ! this is needed to avoid unnecessary work in case user contracts different portions of a
      ! tensor consecutively to save memory
      mp_comm = tensor_crop_1%pgrid%mp_comm_2d
      occ_1 = dbcsr_t_get_num_blocks(tensor_crop_1)
      CALL mp_max(occ_1, mp_comm)
      occ_2 = dbcsr_t_get_num_blocks(tensor_crop_2)
      CALL mp_max(occ_2, mp_comm)

      IF (occ_1 == 0 .OR. occ_2 == 0) THEN
         CALL dbcsr_t_scale(tensor_3, beta)
         IF (do_crop_1) THEN
            CALL dbcsr_t_destroy(tensor_crop_1)
            DEALLOCATE (tensor_crop_1)
         ENDIF
         IF (do_crop_2) THEN
            CALL dbcsr_t_destroy(tensor_crop_2)
            DEALLOCATE (tensor_crop_2)
         ENDIF
         CALL timestop(handle)
         RETURN
      ENDIF

      IF (PRESENT(unit_nr)) THEN
         IF (unit_nr > 0) THEN
            WRITE (unit_nr, '(A)') repeat("-", 80)
            WRITE (unit_nr, '(A,1X,A,1X,A,1X,A,1X,A,1X,A)') "DBCSR TENSOR CONTRACTION:", &
               TRIM(tensor_crop_1%name), 'x', TRIM(tensor_crop_2%name), '=', TRIM(tensor_3%name)
            WRITE (unit_nr, '(A)') repeat("-", 80)
         ENDIF
         CALL dbcsr_t_write_tensor_info(tensor_crop_1, unit_nr, full_info=log_verbose)
         CALL dbcsr_t_write_tensor_dist(tensor_crop_1, unit_nr)
         CALL dbcsr_t_write_tensor_info(tensor_crop_2, unit_nr, full_info=log_verbose)
         CALL dbcsr_t_write_tensor_dist(tensor_crop_2, unit_nr)
      ENDIF

      data_type = dbcsr_t_get_data_type(tensor_crop_1)

      ! align tensor index with data, tensor data is not modified
      ndims1 = dbcsr_t_ndims(tensor_crop_1)
      ndims2 = dbcsr_t_ndims(tensor_crop_2)
      ndims3 = dbcsr_t_ndims(tensor_3)
      ALLOCATE (indchar1(ndims1), indchar1_mod(ndims1))
      ALLOCATE (indchar2(ndims2), indchar2_mod(ndims2))
      ALLOCATE (indchar3(ndims3), indchar3_mod(ndims3))

      ! labeling tensor index with letters

      indchar1([notcontract_1, contract_1]) = alph(1:ndims1) ! arb. choice
      indchar2(notcontract_2) = alph(ndims1 + 1:ndims1 + SIZE(notcontract_2)) ! arb. choice
      indchar2(contract_2) = indchar1(contract_1)
      indchar3(map_1) = indchar1(notcontract_1)
      indchar3(map_2) = indchar2(notcontract_2)

      IF (PRESENT(unit_nr)) CALL dbcsr_t_print_contraction_index(tensor_crop_1, indchar1, &
                                                                 tensor_crop_2, indchar2, &
                                                                 tensor_3, indchar3, unit_nr)
      IF (PRESENT(unit_nr)) THEN
         IF (unit_nr > 0) THEN
            WRITE (unit_nr, '(T2,A)') "aligning tensor index with data"
         ENDIF
      ENDIF

      CALL align_tensor(tensor_crop_1, contract_1, notcontract_1, &
                        tensor_algn_1, contract_1_mod, notcontract_1_mod, indchar1, indchar1_mod)

      CALL align_tensor(tensor_crop_2, contract_2, notcontract_2, &
                        tensor_algn_2, contract_2_mod, notcontract_2_mod, indchar2, indchar2_mod)

      CALL align_tensor(tensor_3, map_1, map_2, &
                        tensor_algn_3, map_1_mod, map_2_mod, indchar3, indchar3_mod)

      IF (PRESENT(unit_nr)) CALL dbcsr_t_print_contraction_index(tensor_algn_1, indchar1_mod, &
                                                                 tensor_algn_2, indchar2_mod, &
                                                                 tensor_algn_3, indchar3_mod, unit_nr)

      ALLOCATE (dims1(ndims1))
      ALLOCATE (dims2(ndims2))
      ALLOCATE (dims3(ndims3))

      ! ideally we should consider block sizes and occupancy to measure tensor sizes but current solution should work for most
      ! cases and is more elegant. Note that we can not easily consider occupancy since it is unknown for result tensor
      CALL blk_dims_tensor(tensor_crop_1, dims1)
      CALL blk_dims_tensor(tensor_crop_2, dims2)
      CALL blk_dims_tensor(tensor_3, dims3)

      max_mm_dim = MAXLOC([PRODUCT(INT(dims1(notcontract_1), int_8)), &
                           PRODUCT(INT(dims1(contract_1), int_8)), &
                           PRODUCT(INT(dims2(notcontract_2), int_8))], DIM=1)
      max_tensor = MAXLOC([PRODUCT(INT(dims1, int_8)), PRODUCT(INT(dims2, int_8)), PRODUCT(INT(dims3, int_8))], DIM=1)
      SELECT CASE (max_mm_dim)
      CASE (1)
         IF (PRESENT(unit_nr)) THEN
            IF (unit_nr > 0) THEN
               WRITE (unit_nr, '(T2,A)') "large tensors: 1, 3; small tensor: 2"
               WRITE (unit_nr, '(T2,A)') "sorting contraction indices"
            ENDIF
         ENDIF
         CALL index_linked_sort(contract_1_mod, contract_2_mod)
         CALL index_linked_sort(map_2_mod, notcontract_2_mod)
         SELECT CASE (max_tensor)
         CASE (1)
            CALL index_linked_sort(notcontract_1_mod, map_1_mod)
         CASE (3)
            CALL index_linked_sort(map_1_mod, notcontract_1_mod)
         CASE DEFAULT
            DBCSR_ABORT("should not happen")
         END SELECT

         CALL reshape_mm_compatible(tensor_algn_1, tensor_algn_3, tensor_contr_1, tensor_contr_3, &
                                    contract_1_mod, notcontract_1_mod, map_2_mod, map_1_mod, &
                                    trans_1, trans_3, new_1, new_3, nodata2=.TRUE., optimize_dist=optimize_dist, &
                                    move_data_1=move_data_1, unit_nr=unit_nr)

         CALL reshape_mm_small(tensor_algn_2, contract_2_mod, notcontract_2_mod, tensor_contr_2, trans_2, &
                               new_2, move_data=move_data_2, unit_nr=unit_nr)

      CASE (2)
         IF (PRESENT(unit_nr)) THEN
            IF (unit_nr > 0) THEN
               WRITE (unit_nr, '(T2,A)') "large tensors: 1, 2; small tensor: 3"
               WRITE (unit_nr, '(T2,A)') "sorting contraction indices"
            ENDIF
         ENDIF

         CALL index_linked_sort(notcontract_1_mod, map_1_mod)
         CALL index_linked_sort(notcontract_2_mod, map_2_mod)
         SELECT CASE (max_tensor)
         CASE (1)
            CALL index_linked_sort(contract_1_mod, contract_2_mod)
         CASE (2)
            CALL index_linked_sort(contract_2_mod, contract_1_mod)
         CASE DEFAULT
            DBCSR_ABORT("should not happen")
         END SELECT

         CALL reshape_mm_compatible(tensor_algn_1, tensor_algn_2, tensor_contr_1, tensor_contr_2, &
                                    notcontract_1_mod, contract_1_mod, notcontract_2_mod, contract_2_mod, &
                                    trans_1, trans_2, new_1, new_2, optimize_dist=optimize_dist, &
                                    move_data_1=move_data_1, move_data_2=move_data_2, unit_nr=unit_nr)
         CALL invert_transpose_flag(trans_1)

         CALL reshape_mm_small(tensor_algn_3, map_1_mod, map_2_mod, tensor_contr_3, trans_3, &
                               new_3, nodata=.TRUE., unit_nr=unit_nr)

      CASE (3)
         IF (PRESENT(unit_nr)) THEN
            IF (unit_nr > 0) THEN
               WRITE (unit_nr, '(T2,A)') "large tensors: 2, 3; small tensor: 1"
               WRITE (unit_nr, '(T2,A)') "sorting contraction indices"
            ENDIF
         ENDIF
         CALL index_linked_sort(map_1_mod, notcontract_1_mod)
         CALL index_linked_sort(contract_2_mod, contract_1_mod)
         SELECT CASE (max_tensor)
         CASE (2)
            CALL index_linked_sort(notcontract_2_mod, map_2_mod)
         CASE (3)
            CALL index_linked_sort(map_2_mod, notcontract_2_mod)
         CASE DEFAULT
            DBCSR_ABORT("should not happen")
         END SELECT

         CALL reshape_mm_compatible(tensor_algn_2, tensor_algn_3, tensor_contr_2, tensor_contr_3, &
                                    contract_2_mod, notcontract_2_mod, map_1_mod, map_2_mod, &
                                    trans_2, trans_3, new_2, new_3, nodata2=.TRUE., optimize_dist=optimize_dist, &
                                    move_data_1=move_data_2, unit_nr=unit_nr)

         CALL invert_transpose_flag(trans_2)
         CALL invert_transpose_flag(trans_3)

         CALL reshape_mm_small(tensor_algn_1, notcontract_1_mod, contract_1_mod, tensor_contr_1, &
                               trans_1, new_1, move_data=move_data_1, unit_nr=unit_nr)

      END SELECT

      IF (PRESENT(unit_nr)) CALL dbcsr_t_print_contraction_index(tensor_contr_1, indchar1_mod, &
                                                                 tensor_contr_2, indchar2_mod, &
                                                                 tensor_contr_3, indchar3_mod, unit_nr)
      IF (PRESENT(unit_nr)) THEN
         IF (new_1) CALL dbcsr_t_write_tensor_info(tensor_contr_1, unit_nr, full_info=log_verbose)
         IF (new_1) CALL dbcsr_t_write_tensor_dist(tensor_contr_1, unit_nr)
         IF (new_2) CALL dbcsr_t_write_tensor_info(tensor_contr_2, unit_nr, full_info=log_verbose)
         IF (new_2) CALL dbcsr_t_write_tensor_dist(tensor_contr_2, unit_nr)
      ENDIF

      CALL dbcsr_tas_multiply(trans_1, trans_2, trans_3, alpha, &
                              tensor_contr_1%matrix_rep, tensor_contr_2%matrix_rep, &
                              beta, &
                              tensor_contr_3%matrix_rep, filter_eps=filter_eps, flop=flop, &
                              io_unit=unit_nr, log_verbose=log_verbose, &
                              split_opt=split_opt, &
                              move_data_a=move_data_1, move_data_b=move_data_2)

      IF (PRESENT(pgrid_opt_1) .AND. ASSOCIATED(split_opt)) THEN
         IF (.NOT. new_1) THEN
            ALLOCATE (pgrid_opt_1)
            pgrid_opt_1 = opt_pgrid(tensor_1, split_opt)
         ENDIF
      ENDIF

      IF (PRESENT(pgrid_opt_2) .AND. ASSOCIATED(split_opt)) THEN
         IF (.NOT. new_2) THEN
            ALLOCATE (pgrid_opt_2)
            pgrid_opt_2 = opt_pgrid(tensor_2, split_opt)
         ENDIF
      ENDIF

      IF (PRESENT(pgrid_opt_3) .AND. ASSOCIATED(split_opt)) THEN
         IF (.NOT. new_3) THEN
            ALLOCATE (pgrid_opt_3)
            pgrid_opt_3 = opt_pgrid(tensor_3, split_opt)
         ENDIF
      ENDIF

      IF (ASSOCIATED(split_opt)) THEN
         CALL dbcsr_tas_release_info(split_opt)
         DEALLOCATE (split_opt)
      ENDIF

      IF (PRESENT(unit_nr)) THEN
         CALL dbcsr_t_write_tensor_info(tensor_contr_3, unit_nr, full_info=log_verbose)
         CALL dbcsr_t_write_tensor_dist(tensor_contr_3, unit_nr)
      ENDIF

      IF (new_3) THEN
         ! need redistribute if we created new tensor for tensor 3
         CALL dbcsr_t_scale(tensor_algn_3, beta)
         CALL dbcsr_t_copy(tensor_contr_3, tensor_algn_3, summation=.TRUE., move_data=.TRUE.)
         IF (PRESENT(filter_eps)) CALL dbcsr_t_filter(tensor_algn_3, filter_eps)
         ! tensor_3 automatically has correct data because tensor_algn_3 contains a matrix
         ! pointer to data of tensor_3
      ENDIF

      IF (PRESENT(unit_nr)) THEN
         IF (new_3) CALL dbcsr_t_write_tensor_info(tensor_3, unit_nr, full_info=log_verbose)
         IF (new_3) CALL dbcsr_t_write_tensor_dist(tensor_3, unit_nr)
      ENDIF

      CALL dbcsr_t_destroy(tensor_algn_1)
      CALL dbcsr_t_destroy(tensor_algn_2)
      CALL dbcsr_t_destroy(tensor_algn_3)

      IF (do_crop_1) THEN
         CALL dbcsr_t_destroy(tensor_crop_1)
         DEALLOCATE (tensor_crop_1)
      ENDIF

      IF (do_crop_2) THEN
         CALL dbcsr_t_destroy(tensor_crop_2)
         DEALLOCATE (tensor_crop_2)
      ENDIF

      IF (new_1) THEN
         CALL dbcsr_t_destroy(tensor_contr_1)
         DEALLOCATE (tensor_contr_1)
      ENDIF
      IF (new_2) THEN
         CALL dbcsr_t_destroy(tensor_contr_2)
         DEALLOCATE (tensor_contr_2)
      ENDIF
      IF (new_3) THEN
         CALL dbcsr_t_destroy(tensor_contr_3)
         DEALLOCATE (tensor_contr_3)
      ENDIF

      IF (PRESENT(unit_nr)) THEN
         IF (unit_nr > 0) THEN
            WRITE (unit_nr, '(A)') repeat("-", 80)
            WRITE (unit_nr, '(A)') "TENSOR CONTRACTION DONE"
            WRITE (unit_nr, '(A)') repeat("-", 80)
         ENDIF
      ENDIF

      CALL timestop(handle)

   END SUBROUTINE

   SUBROUTINE align_tensor(tensor_in, contract_in, notcontract_in, &
      !! align tensor index with data
                           tensor_out, contract_out, notcontract_out, indp_in, indp_out)
      TYPE(dbcsr_t_type), INTENT(INOUT)               :: tensor_in
      INTEGER, DIMENSION(:), INTENT(IN)            :: contract_in, notcontract_in
      TYPE(dbcsr_t_type), INTENT(OUT)              :: tensor_out
      INTEGER, DIMENSION(SIZE(contract_in)), &
         INTENT(OUT)                               :: contract_out
      INTEGER, DIMENSION(SIZE(notcontract_in)), &
         INTENT(OUT)                               :: notcontract_out
      CHARACTER(LEN=1), DIMENSION(dbcsr_t_ndims(tensor_in)), INTENT(IN) :: indp_in
      CHARACTER(LEN=1), DIMENSION(dbcsr_t_ndims(tensor_in)), INTENT(OUT) :: indp_out
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor_in)) :: align

      CALL dbcsr_t_align_index(tensor_in, tensor_out, order=align)
      contract_out = align(contract_in)
      notcontract_out = align(notcontract_in)
      indp_out(align) = indp_in

   END SUBROUTINE

   SUBROUTINE reshape_mm_compatible(tensor1, tensor2, tensor1_out, tensor2_out, ind1_free, ind1_linked, &
                                    ind2_free, ind2_linked, trans1, trans2, new1, new2, nodata1, nodata2, move_data_1, &
                                    move_data_2, optimize_dist, unit_nr)
      !! Prepare tensor for contraction: redistribute to a 2d format which can be contracted by
      !! matrix multiplication. This routine reshapes the two largest of the three tensors. Redistribution
      !! is avoided if tensors already in a consistent layout.

      TYPE(dbcsr_t_type), TARGET, INTENT(INOUT)   :: tensor1
         !! tensor 1 in
      TYPE(dbcsr_t_type), TARGET, INTENT(INOUT)   :: tensor2
         !! tensor 2 in
      TYPE(dbcsr_t_type), POINTER, INTENT(OUT)    :: tensor1_out, tensor2_out
         !! tensor 1 out
         !! tensor 2 out
      INTEGER, DIMENSION(:), INTENT(IN)           :: ind1_free, ind2_free
         !! indices of tensor 1 that are "free" (not linked to any index of tensor 2)
      INTEGER, DIMENSION(:), INTENT(IN)           :: ind1_linked, ind2_linked
         !! indices of tensor 1 that are linked to indices of tensor 2
         !! 1:1 correspondence with ind1_linked
      CHARACTER(LEN=1), INTENT(OUT)               :: trans1, trans2
         !! transpose flag of matrix rep. of tensor 1
         !! transpose flag of matrix rep. tensor 2
      LOGICAL, INTENT(OUT)                        :: new1, new2
         !! whether a new tensor 1 was created
         !! whether a new tensor 2 was created
      LOGICAL, INTENT(IN), OPTIONAL               :: nodata1, nodata2
         !! don't copy data of tensor 1
         !! don't copy data of tensor 2
      LOGICAL, INTENT(INOUT), OPTIONAL            :: move_data_1, move_data_2
         !! memory optimization: transfer data s.t. tensor1 may be empty on return
         !! memory optimization: transfer data s.t. tensor2 may be empty on return
      LOGICAL, INTENT(IN), OPTIONAL               :: optimize_dist
         !! experimental: optimize distribution
      INTEGER, INTENT(IN), OPTIONAL               :: unit_nr
         !! output unit
      INTEGER                                     :: ref_tensor, compat1, compat1_old, compat2, compat2_old, &
                                                     comm_2d, io_unit
      TYPE(array_list)                            :: dist_list
      INTEGER, DIMENSION(:), ALLOCATABLE          :: mp_dims, dims
      TYPE(dbcsr_t_distribution_type)             :: dist_in
      INTEGER(KIND=int_8)                         :: nblkrows, nblkcols
      LOGICAL                                     :: optimize_dist_prv

      NULLIFY (tensor1_out, tensor2_out)
      IF (PRESENT(unit_nr)) THEN
         io_unit = unit_nr
      ELSE
         io_unit = 0
      ENDIF

      IF (SIZE(ind1_free) .GE. SIZE(ind2_free)) THEN
         ref_tensor = 1
      ELSE
         ref_tensor = 2
      ENDIF

      IF (PRESENT(optimize_dist)) THEN
         optimize_dist_prv = optimize_dist
      ELSE
         optimize_dist_prv = .FALSE.
      ENDIF

      compat1 = compat_map(tensor1%nd_index, ind1_linked)
      compat2 = compat_map(tensor2%nd_index, ind2_linked)
      compat1_old = compat1
      compat2_old = compat2

      IF (io_unit > 0) THEN
         WRITE (io_unit, '(T2,A,1X,A,A,1X)', advance='no') "compatibility of", TRIM(tensor1%name), ":"
         SELECT CASE (compat1)
         CASE (0)
            WRITE (io_unit, '(A)') "Not compatible"
         CASE (1)
            WRITE (io_unit, '(A)') "Normal"
         CASE (2)
            WRITE (io_unit, '(A)') "Transposed"
         END SELECT
         WRITE (io_unit, '(T2,A,1X,A,A,1X)', advance='no') "compatibility of", TRIM(tensor2%name), ":"
         SELECT CASE (compat2)
         CASE (0)
            WRITE (io_unit, '(A)') "Not compatible"
         CASE (1)
            WRITE (io_unit, '(A)') "Normal"
         CASE (2)
            WRITE (io_unit, '(A)') "Transposed"
         END SELECT
      ENDIF

      new1 = .FALSE.
      new2 = .FALSE.

      IF (compat1 == 0 .OR. optimize_dist_prv) THEN
         new1 = .TRUE.
      ENDIF

      IF (compat2 == 0 .OR. optimize_dist_prv) THEN
         new2 = .TRUE.
      ENDIF

      IF (ref_tensor == 1) THEN ! tensor 1 is reference and tensor 2 is reshaped compatible with tensor 1
         IF (compat1 == 0 .OR. optimize_dist_prv) THEN ! tensor 1 is not contraction compatible --> reshape
            IF (io_unit > 0) WRITE (unit_nr, '(T2,A,1X,A)') "Redistribution of", TRIM(tensor1%name)
            ALLOCATE (dims(dbcsr_t_ndims(tensor1)))
            CALL blk_dims_tensor(tensor1, dims)
            nblkrows = PRODUCT(INT(dims(ind1_linked), KIND=int_8))
            nblkcols = PRODUCT(INT(dims(ind1_free), KIND=int_8))
            comm_2d = dbcsr_tas_mp_comm(tensor1%pgrid%mp_comm_2d, nblkrows, nblkcols)
            ALLOCATE (tensor1_out)
            CALL dbcsr_t_remap(tensor1, ind1_linked, ind1_free, tensor1_out, comm_2d=comm_2d, &
                               nodata=nodata1, move_data=move_data_1)
            CALL mp_comm_free(comm_2d)
            compat1 = 1
         ELSE
            IF (io_unit > 0) WRITE (unit_nr, '(T2,A,1X,A)') "No redistribution of", TRIM(tensor1%name)
            tensor1_out => tensor1
         ENDIF
         IF (compat2 == 0 .OR. optimize_dist_prv) THEN ! tensor 2 is not contraction compatible --> reshape
            IF (io_unit > 0) WRITE (unit_nr, '(T2,A,1X,A,1X,A,1X,A)') "Redistribution of", &
               TRIM(tensor2%name), "compatible with", TRIM(tensor1%name)
            dist_in = dbcsr_t_distribution(tensor1_out)
            dist_list = array_sublist(dist_in%nd_dist, ind1_linked)
            IF (compat1 == 1) THEN ! linked index is first 2d dimension
               ! get distribution of linked index, tensor 2 must adopt this distribution
               ! get grid dimensions of linked index
               CALL dbcsr_t_get_mapping_info(dist_in%pgrid%nd_index_grid, dims1_2d=mp_dims)
               ALLOCATE (tensor2_out)
               CALL dbcsr_t_remap(tensor2, ind2_linked, ind2_free, tensor2_out, comm_2d=dist_in%pgrid%mp_comm_2d, &
                                  dist1=dist_list, mp_dims_1=mp_dims, nodata=nodata2, move_data=move_data_2)
            ELSEIF (compat1 == 2) THEN ! linked index is second 2d dimension
               ! get distribution of linked index, tensor 2 must adopt this distribution
               ! get grid dimensions of linked index
               CALL dbcsr_t_get_mapping_info(dist_in%pgrid%nd_index_grid, dims2_2d=mp_dims)
               ALLOCATE (tensor2_out)
               CALL dbcsr_t_remap(tensor2, ind2_free, ind2_linked, tensor2_out, comm_2d=dist_in%pgrid%mp_comm_2d, &
                                  dist2=dist_list, mp_dims_2=mp_dims, nodata=nodata2, move_data=move_data_2)
            ELSE
               DBCSR_ABORT("should not happen")
            ENDIF
            compat2 = compat1
         ELSE
            IF (io_unit > 0) WRITE (unit_nr, '(T2,A,1X,A)') "No redistribution of", TRIM(tensor2%name)
            tensor2_out => tensor2
         ENDIF
      ELSE ! tensor 2 is reference and tensor 1 is reshaped compatible with tensor 2
         IF (compat2 == 0 .OR. optimize_dist_prv) THEN ! tensor 2 is not contraction compatible --> reshape
            IF (io_unit > 0) WRITE (unit_nr, '(T2,A,1X,A)') "Redistribution of", TRIM(tensor2%name)
            ALLOCATE (dims(dbcsr_t_ndims(tensor2)))
            CALL blk_dims_tensor(tensor2, dims)
            nblkrows = PRODUCT(INT(dims(ind2_linked), KIND=int_8))
            nblkcols = PRODUCT(INT(dims(ind2_free), KIND=int_8))
            comm_2d = dbcsr_tas_mp_comm(tensor2%pgrid%mp_comm_2d, nblkrows, nblkcols)
            ALLOCATE (tensor2_out)
            CALL dbcsr_t_remap(tensor2, ind2_linked, ind2_free, tensor2_out, nodata=nodata2, move_data=move_data_2)
            CALL mp_comm_free(comm_2d)
            compat2 = 1
         ELSE
            IF (io_unit > 0) WRITE (unit_nr, '(T2,A,1X,A)') "No redistribution of", TRIM(tensor2%name)
            tensor2_out => tensor2
         ENDIF
         IF (compat1 == 0 .OR. optimize_dist_prv) THEN ! tensor 1 is not contraction compatible --> reshape
            IF (io_unit > 0) WRITE (unit_nr, '(T2,A,1X,A,1X,A,1X,A)') "Redistribution of", TRIM(tensor1%name), &
               "compatible with", TRIM(tensor2%name)
            dist_in = dbcsr_t_distribution(tensor2_out)
            dist_list = array_sublist(dist_in%nd_dist, ind2_linked)
            IF (compat2 == 1) THEN
               CALL dbcsr_t_get_mapping_info(dist_in%pgrid%nd_index_grid, dims1_2d=mp_dims)
               ALLOCATE (tensor1_out)
               CALL dbcsr_t_remap(tensor1, ind1_linked, ind1_free, tensor1_out, comm_2d=dist_in%pgrid%mp_comm_2d, &
                                  dist1=dist_list, mp_dims_1=mp_dims, nodata=nodata1, move_data=move_data_1)
            ELSEIF (compat2 == 2) THEN
               CALL dbcsr_t_get_mapping_info(dist_in%pgrid%nd_index_grid, dims2_2d=mp_dims)
               ALLOCATE (tensor1_out)
               CALL dbcsr_t_remap(tensor1, ind1_free, ind1_linked, tensor1_out, comm_2d=dist_in%pgrid%mp_comm_2d, &
                                  dist2=dist_list, mp_dims_2=mp_dims, nodata=nodata1, move_data=move_data_1)
            ELSE
               DBCSR_ABORT("should not happen")
            ENDIF
            compat1 = compat2
         ELSE
            IF (io_unit > 0) WRITE (unit_nr, '(T2,A,1X,A)') "No redistribution of", TRIM(tensor1%name)
            tensor1_out => tensor1
         ENDIF
      ENDIF

      SELECT CASE (compat1)
      CASE (1)
         trans1 = dbcsr_no_transpose
      CASE (2)
         trans1 = dbcsr_transpose
      CASE DEFAULT
         DBCSR_ABORT("should not happen")
      END SELECT

      SELECT CASE (compat2)
      CASE (1)
         trans2 = dbcsr_no_transpose
      CASE (2)
         trans2 = dbcsr_transpose
      CASE DEFAULT
         DBCSR_ABORT("should not happen")
      END SELECT

      IF (io_unit > 0) THEN
         IF (compat1 .NE. compat1_old) THEN
            WRITE (io_unit, '(T2,A,1X,A,A,1X)', advance='no') "compatibility of", TRIM(tensor1_out%name), ":"
            SELECT CASE (compat1)
            CASE (0)
               WRITE (io_unit, '(A)') "Not compatible"
            CASE (1)
               WRITE (io_unit, '(A)') "Normal"
            CASE (2)
               WRITE (io_unit, '(A)') "Transposed"
            END SELECT
         ENDIF
         IF (compat2 .NE. compat2_old) THEN
            WRITE (io_unit, '(T2,A,1X,A,A,1X)', advance='no') "compatibility of", TRIM(tensor2_out%name), ":"
            SELECT CASE (compat2)
            CASE (0)
               WRITE (io_unit, '(A)') "Not compatible"
            CASE (1)
               WRITE (io_unit, '(A)') "Normal"
            CASE (2)
               WRITE (io_unit, '(A)') "Transposed"
            END SELECT
         ENDIF
      ENDIF

      IF (new1 .AND. PRESENT(move_data_1)) move_data_1 = .TRUE.
      IF (new2 .AND. PRESENT(move_data_2)) move_data_2 = .TRUE.

   END SUBROUTINE

   SUBROUTINE reshape_mm_small(tensor_in, ind1, ind2, tensor_out, trans, new, nodata, move_data, unit_nr)
      !! Prepare tensor for contraction: redistribute to a 2d format which can be contracted by
      !! matrix multiplication. This routine reshapes the smallest of the three tensors.

      TYPE(dbcsr_t_type), TARGET, INTENT(INOUT)   :: tensor_in
         !! tensor in
      INTEGER, DIMENSION(:), INTENT(IN)           :: ind1, ind2
         !! index that should be mapped to first matrix dimension
         !! index that should be mapped to second matrix dimension
      TYPE(dbcsr_t_type), POINTER, INTENT(OUT)    :: tensor_out
         !! tensor out
      CHARACTER(LEN=1), INTENT(OUT)               :: trans
         !! transpose flag of matrix rep.
      LOGICAL, INTENT(OUT)                        :: new
         !! whether a new tensor was created for tensor_out
      LOGICAL, INTENT(IN), OPTIONAL               :: nodata, move_data
         !! don't copy tensor data
         !! memory optimization: transfer data s.t. tensor_in may be empty on return
      INTEGER, INTENT(IN), OPTIONAL               :: unit_nr
         !! output unit
      INTEGER                                     :: compat1, compat2, compat1_old, compat2_old, io_unit
      LOGICAL                                     :: nodata_prv

      NULLIFY (tensor_out)
      IF (PRESENT(nodata)) THEN
         nodata_prv = nodata
      ELSE
         nodata_prv = .FALSE.
      ENDIF

      IF (PRESENT(unit_nr)) THEN
         io_unit = unit_nr
      ELSE
         io_unit = 0
      ENDIF

      new = .FALSE.
      compat1 = compat_map(tensor_in%nd_index, ind1)
      compat2 = compat_map(tensor_in%nd_index, ind2)
      compat1_old = compat1; compat2_old = compat2
      IF (io_unit > 0) THEN
         WRITE (io_unit, '(T2,A,1X,A,A,1X)', advance='no') "compatibility of", TRIM(tensor_in%name), ":"
         IF (compat1 == 1 .AND. compat2 == 2) THEN
            WRITE (io_unit, '(A)') "Normal"
         ELSEIF (compat1 == 2 .AND. compat2 == 1) THEN
            WRITE (io_unit, '(A)') "Transposed"
         ELSE
            WRITE (io_unit, '(A)') "Not compatible"
         ENDIF
      ENDIF
      IF (compat1 == 0 .or. compat2 == 0) THEN ! index mapping not compatible with contract index

         IF (io_unit > 0) WRITE (unit_nr, '(T2,A,1X,A)') "Redistribution of", TRIM(tensor_in%name)

         ALLOCATE (tensor_out)
         CALL dbcsr_t_remap(tensor_in, ind1, ind2, tensor_out, nodata=nodata, move_data=move_data)
         compat1 = 1
         compat2 = 2
         new = .TRUE.
      ELSE
         IF (io_unit > 0) WRITE (unit_nr, '(T2,A,1X,A)') "No redistribution of", TRIM(tensor_in%name)
         tensor_out => tensor_in
      ENDIF

      IF (compat1 == 1 .AND. compat2 == 2) THEN
         trans = dbcsr_no_transpose
      ELSEIF (compat1 == 2 .AND. compat2 == 1) THEN
         trans = dbcsr_transpose
      ELSE
         DBCSR_ABORT("this should not happen")
      ENDIF

      IF (io_unit > 0) THEN
         IF (compat1_old .NE. compat1 .OR. compat2_old .NE. compat2) THEN
            WRITE (io_unit, '(T2,A,1X,A,A,1X)', advance='no') "compatibility of", TRIM(tensor_out%name), ":"
            IF (compat1 == 1 .AND. compat2 == 2) THEN
               WRITE (io_unit, '(A)') "Normal"
            ELSEIF (compat1 == 2 .AND. compat2 == 1) THEN
               WRITE (io_unit, '(A)') "Transposed"
            ELSE
               WRITE (io_unit, '(A)') "Not compatible"
            ENDIF
         ENDIF
      ENDIF

   END SUBROUTINE

   FUNCTION compat_map(nd_index, compat_ind)
      !! Check if 2d index is compatible with tensor index
      TYPE(nd_to_2d_mapping), INTENT(IN) :: nd_index
      INTEGER, DIMENSION(:), INTENT(IN)  :: compat_ind
      INTEGER, DIMENSION(:), ALLOCATABLE :: map1, map2
      INTEGER                            :: compat_map

      CALL dbcsr_t_get_mapping_info(nd_index, map1_2d=map1, map2_2d=map2)

      compat_map = 0
      IF (array_eq_i(map1, compat_ind)) THEN
         compat_map = 1
      ELSEIF (array_eq_i(map2, compat_ind)) THEN
         compat_map = 2
      ENDIF

   END FUNCTION

   SUBROUTINE invert_transpose_flag(trans_flag)
      CHARACTER(LEN=1), INTENT(INOUT)                    :: trans_flag

      IF (trans_flag == dbcsr_transpose) THEN
         trans_flag = dbcsr_no_transpose
      ELSEIF (trans_flag == dbcsr_no_transpose) THEN
         trans_flag = dbcsr_transpose
      ENDIF
   END SUBROUTINE

   SUBROUTINE index_linked_sort(ind_ref, ind_dep)
      INTEGER, DIMENSION(:), INTENT(INOUT) :: ind_ref, ind_dep
      INTEGER, DIMENSION(SIZE(ind_ref))    :: sort_indices

      CALL sort(ind_ref, SIZE(ind_ref), sort_indices)
      ind_dep(:) = ind_dep(sort_indices)

   END SUBROUTINE

   FUNCTION opt_pgrid(tensor, tas_split_info)
      TYPE(dbcsr_t_type), INTENT(IN) :: tensor
      TYPE(dbcsr_tas_split_info), INTENT(IN) :: tas_split_info
      INTEGER, DIMENSION(:), ALLOCATABLE :: map1, map2
      TYPE(dbcsr_t_pgrid_type) :: opt_pgrid

      CALL dbcsr_t_get_mapping_info(tensor%pgrid%nd_index_grid, map1_2d=map1, map2_2d=map2)
      opt_pgrid = dbcsr_t_nd_mp_comm(tas_split_info%mp_comm, map1, map2)

      ALLOCATE (opt_pgrid%tas_split_info, SOURCE=tas_split_info)
      CALL dbcsr_tas_info_hold(opt_pgrid%tas_split_info)
   END FUNCTION

   SUBROUTINE new_default_dist(nel, nbin, weights, dist)
      !! Default distribution that is more or less cyclic (round robin) and load balanced with different
      !! weights for each element.
      !! This is used for creating adhoc distributions whenever tensors are mapped to new grids.

      INTEGER, INTENT(IN) :: nel
      INTEGER, INTENT(IN) :: nbin
      INTEGER, DIMENSION(nel), INTENT(IN)            :: weights
      INTEGER, DIMENSION(:), ALLOCATABLE, INTENT(OUT)            :: dist
      INTEGER, DIMENSION(nbin)                        :: occup
      INTEGER                :: iel, ibin
      INTEGER                :: niter

      ALLOCATE (dist(nel))
      occup(:) = 0
      ibin = 0
      DO iel = 1, nel
         niter = 0
         ibin = MOD(ibin + 1, nbin)
         DO WHILE (occup(ibin + 1) + weights(iel) .GE. MAXVAL(occup))
            IF (MINLOC(occup, DIM=1) == ibin + 1) EXIT
            ibin = MOD(ibin + 1, nbin)
            niter = niter + 1
         ENDDO
         dist(iel) = ibin
         occup(ibin + 1) = occup(ibin + 1) + weights(iel)
      ENDDO

   END SUBROUTINE

   SUBROUTINE dbcsr_t_remap(tensor_in, map1_2d, map2_2d, tensor_out, comm_2d, dist1, dist2, &
                            mp_dims_1, mp_dims_2, name, nodata, move_data)
      !! Copy tensor to tensor with modified index mapping

      TYPE(dbcsr_t_type), INTENT(INOUT)      :: tensor_in
      INTEGER, DIMENSION(:), INTENT(IN)      :: map1_2d, map2_2d
         !! new index mapping
         !! new index mapping
      TYPE(dbcsr_t_type), INTENT(OUT)        :: tensor_out
      CHARACTER(len=*), INTENT(IN), OPTIONAL :: name
      LOGICAL, INTENT(IN), OPTIONAL          :: nodata, move_data
      INTEGER, INTENT(IN), OPTIONAL          :: comm_2d
      TYPE(array_list), INTENT(IN), OPTIONAL :: dist1, dist2
      INTEGER, DIMENSION(SIZE(map1_2d)), OPTIONAL :: mp_dims_1
      INTEGER, DIMENSION(SIZE(map2_2d)), OPTIONAL :: mp_dims_2
      CHARACTER(len=default_string_length)   :: name_tmp
      INTEGER, DIMENSION(:), ALLOCATABLE     :: ${varlist("blk_sizes")}$, &
                                                ${varlist("nd_dist")}$
      TYPE(dbcsr_t_distribution_type)        :: dist
      INTEGER                                :: comm_2d_prv, handle, i
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor_in)) :: pdims, myploc
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_remap', &
                                     routineP = moduleN//':'//routineN
      LOGICAL                               :: nodata_prv
      TYPE(dbcsr_t_pgrid_type)              :: comm_nd

      CALL timeset(routineN, handle)

      IF (PRESENT(name)) THEN
         name_tmp = name
      ELSE
         name_tmp = tensor_in%name
      ENDIF
      IF (PRESENT(dist1)) THEN
         DBCSR_ASSERT(PRESENT(mp_dims_1))
      ENDIF

      IF (PRESENT(dist2)) THEN
         DBCSR_ASSERT(PRESENT(mp_dims_2))
      ENDIF

      IF (PRESENT(comm_2d)) THEN
         comm_2d_prv = comm_2d
      ELSE
         comm_2d_prv = tensor_in%pgrid%mp_comm_2d
      ENDIF

      comm_nd = dbcsr_t_nd_mp_comm(comm_2d_prv, map1_2d, map2_2d, dims1_nd=mp_dims_1, dims2_nd=mp_dims_2)
      CALL mp_environ_pgrid(comm_nd, pdims, myploc)

#:for ndim in ndims
      IF (ndims_tensor(tensor_in) == ${ndim}$) THEN
         CALL get_arrays(tensor_in%blk_sizes, ${varlist("blk_sizes", nmax=ndim)}$)
      ENDIF
#:endfor

#:for ndim in ndims
      IF (ndims_tensor(tensor_in) == ${ndim}$) THEN
#:for idim in range(1, ndim+1)
         IF (PRESENT(dist1)) THEN
            IF (ANY(map1_2d == ${idim}$)) THEN
               i = MINLOC(map1_2d, dim=1, mask=map1_2d == ${idim}$) ! i is location of idim in map1_2d
               CALL get_ith_array(dist1, i, nd_dist_${idim}$)
            ENDIF
         ENDIF

         IF (PRESENT(dist2)) THEN
            IF (ANY(map2_2d == ${idim}$)) THEN
               i = MINLOC(map2_2d, dim=1, mask=map2_2d == ${idim}$) ! i is location of idim in map2_2d
               CALL get_ith_array(dist2, i, nd_dist_${idim}$)
            ENDIF
         ENDIF

         IF (.NOT. ALLOCATED(nd_dist_${idim}$)) THEN
            CALL new_default_dist(SIZE(blk_sizes_${idim}$), pdims(${idim}$), blk_sizes_${idim}$, nd_dist_${idim}$)
         ENDIF
#:endfor
         CALL dbcsr_t_distribution_new(dist, comm_nd, map1_2d, map2_2d, &
                                       ${varlist("nd_dist", nmax=ndim)}$, own_comm=.TRUE.)
      ENDIF
#:endfor

#:for ndim in ndims
      IF (ndims_tensor(tensor_in) == ${ndim}$) THEN
         CALL dbcsr_t_create(tensor_out, name_tmp, dist, &
                             map1_2d, map2_2d, dbcsr_tas_get_data_type(tensor_in%matrix_rep), &
                             ${varlist("blk_sizes", nmax=ndim)}$)
      ENDIF
#:endfor

      IF (PRESENT(nodata)) THEN
         nodata_prv = nodata
      ELSE
         nodata_prv = .FALSE.
      ENDIF

      IF (.NOT. nodata_prv) CALL dbcsr_t_copy(tensor_in, tensor_out, move_data=move_data)
      CALL dbcsr_t_distribution_destroy(dist)

      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE dbcsr_t_align_index(tensor_in, tensor_out, order)
      !! Align index with data

      TYPE(dbcsr_t_type), INTENT(INOUT)               :: tensor_in
      TYPE(dbcsr_t_type), INTENT(OUT)                 :: tensor_out
      INTEGER, DIMENSION(:), ALLOCATABLE              :: map1_2d, map2_2d
      INTEGER, DIMENSION(ndims_tensor(tensor_in)), &
         INTENT(OUT), OPTIONAL                        :: order
         !! permutation resulting from alignment
      INTEGER, DIMENSION(ndims_tensor(tensor_in))     :: order_prv
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_align_index', &
                                     routineP = moduleN//':'//routineN
      INTEGER                                         :: handle

      CALL timeset(routineN, handle)

      CALL dbcsr_t_get_mapping_info(tensor_in%nd_index_blk, map1_2d=map1_2d, map2_2d=map2_2d)
      order_prv = dbcsr_t_inverse_order([map1_2d, map2_2d])
      CALL dbcsr_t_permute_index(tensor_in, tensor_out, order=order_prv)

      IF (PRESENT(order)) order = order_prv

      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE dbcsr_t_permute_index(tensor_in, tensor_out, order)
      !! Create new tensor by reordering index, data is copied exactly (shallow copy)
      TYPE(dbcsr_t_type), INTENT(INOUT)                  :: tensor_in
      TYPE(dbcsr_t_type), INTENT(OUT)                 :: tensor_out
      INTEGER, DIMENSION(ndims_tensor(tensor_in)), &
         INTENT(IN)                                   :: order

      TYPE(nd_to_2d_mapping)                          :: nd_index_blk_rs, nd_index_rs
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_permute_index', &
                                     routineP = moduleN//':'//routineN
      INTEGER                                         :: handle
      INTEGER                                         :: ndims

      CALL timeset(routineN, handle)

      ndims = ndims_tensor(tensor_in)

      CALL permute_index(tensor_in%nd_index, nd_index_rs, order)
      CALL permute_index(tensor_in%nd_index_blk, nd_index_blk_rs, order)
      CALL permute_index(tensor_in%pgrid%nd_index_grid, tensor_out%pgrid%nd_index_grid, order)

      tensor_out%matrix_rep => tensor_in%matrix_rep
      tensor_out%owns_matrix = .FALSE.

      tensor_out%nd_index = nd_index_rs
      tensor_out%nd_index_blk = nd_index_blk_rs
      tensor_out%pgrid%mp_comm_2d = tensor_in%pgrid%mp_comm_2d
      IF (ALLOCATED(tensor_in%pgrid%tas_split_info)) THEN
         ALLOCATE (tensor_out%pgrid%tas_split_info, SOURCE=tensor_in%pgrid%tas_split_info)
      ENDIF
      tensor_out%refcount => tensor_in%refcount
      CALL dbcsr_t_hold(tensor_out)

      CALL reorder_arrays(tensor_in%blk_sizes, tensor_out%blk_sizes, order)
      CALL reorder_arrays(tensor_in%blk_offsets, tensor_out%blk_offsets, order)
      CALL reorder_arrays(tensor_in%nd_dist, tensor_out%nd_dist, order)
      CALL reorder_arrays(tensor_in%blks_local, tensor_out%blks_local, order)
      ALLOCATE (tensor_out%nblks_local(ndims))
      ALLOCATE (tensor_out%nfull_local(ndims))
      tensor_out%nblks_local(order) = tensor_in%nblks_local(:)
      tensor_out%nfull_local(order) = tensor_in%nfull_local(:)
      tensor_out%name = tensor_in%name
      tensor_out%valid = .TRUE.

      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE dbcsr_t_get_nonzero_index(tensor, dim, bounds, ind)
      !! Get all indices along one dimension that have at least one occupied block

      TYPE(dbcsr_t_type), INTENT(IN) :: tensor
      INTEGER, INTENT(IN) :: dim
         !! which dimension
      INTEGER, INTENT(IN), DIMENSION(2, ndims_tensor(tensor)) :: bounds
         !! bounds of (full) tensor index
      LOGICAL, ALLOCATABLE, DIMENSION(:), INTENT(OUT) :: ind
         !! indices with occupied blocks
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor)) :: bdims, blk_index, blk_size, blk_offset
      INTEGER :: bdim, blk, idim
      TYPE(dbcsr_t_iterator_type)                     :: iter

      CALL blk_dims_tensor(tensor, bdims)
      bdim = bdims(dim)
      ALLOCATE (ind(bdim))
      ind = .FALSE.

      CALL dbcsr_t_iterator_start(iter, tensor)

      blk_loop: DO WHILE (dbcsr_t_iterator_blocks_left(iter))
         CALL dbcsr_t_iterator_next_block(iter, blk_index, blk, blk_size=blk_size, blk_offset=blk_offset)
         DO idim = 1, ndims_tensor(tensor)
            IF (bounds(1, idim) > blk_offset(idim) - 1 + blk_size(idim)) CYCLE blk_loop
            IF (bounds(2, idim) < blk_offset(idim)) CYCLE blk_loop
         ENDDO

         ind(blk_index(dim)) = .TRUE.
      ENDDO blk_loop

      CALL dbcsr_t_iterator_stop(iter)
   END SUBROUTINE

   FUNCTION dbcsr_t_need_contract(tensor_1, tensor_2, contract_1, notcontract_1, &
                                  contract_2, notcontract_2, &
                                  bounds_1, bounds_2, bounds_3)
      !! check whether we need to contract two tensors (whether the contraction indices of both tensors
      !! have common occupied blocks). See documentation of dbcsr_t_contract.

      TYPE(dbcsr_t_type), INTENT(INOUT)  :: tensor_1, tensor_2
      INTEGER, DIMENSION(:), INTENT(IN)  :: contract_1, contract_2, &
                                            notcontract_1, notcontract_2
      INTEGER, DIMENSION(2, SIZE(contract_1)), &
         OPTIONAL                                    :: bounds_1
      INTEGER, DIMENSION(2, SIZE(notcontract_1)), &
         OPTIONAL                                    :: bounds_2
      INTEGER, DIMENSION(2, SIZE(notcontract_2)), &
         OPTIONAL                                    :: bounds_3
      INTEGER :: i
      LOGICAL, DIMENSION(:), ALLOCATABLE :: ind1, ind2
      INTEGER, DIMENSION(2, dbcsr_t_ndims(tensor_1)) :: bounds_t1
      INTEGER, DIMENSION(2, dbcsr_t_ndims(tensor_2)) :: bounds_t2
      LOGICAL :: dbcsr_t_need_contract
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_need_contract', &
                                     routineP = moduleN//':'//routineN
      INTEGER :: handle, mp_comm

      CALL timeset(routineN, handle)

      mp_comm = tensor_1%pgrid%mp_comm_2d

      CALL dbcsr_t_map_bounds_to_tensors(tensor_1, tensor_2, &
                                         contract_1, notcontract_1, &
                                         contract_2, notcontract_2, &
                                         bounds_t1, bounds_t2, &
                                         bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3)

      dbcsr_t_need_contract = .TRUE.

      DO i = 1, SIZE(contract_1)
         CALL dbcsr_t_get_nonzero_index(tensor_1, contract_1(i), bounds_t1, ind1)
         CALL dbcsr_t_get_nonzero_index(tensor_2, contract_2(i), bounds_t2, ind2)

         CALL mp_sum(ind1, mp_comm)
         CALL mp_sum(ind2, mp_comm)

         IF (.NOT. ANY(ind1 .AND. ind2)) THEN
            dbcsr_t_need_contract = .FALSE.
            CALL timestop(handle)
            RETURN
         ENDIF

         DEALLOCATE (ind1, ind2)
      ENDDO

      CALL timestop(handle)
   END FUNCTION

   SUBROUTINE dbcsr_t_map_bounds_to_tensors(tensor_1, tensor_2, &
                                            contract_1, notcontract_1, &
                                            contract_2, notcontract_2, &
                                            bounds_t1, bounds_t2, &
                                            bounds_1, bounds_2, bounds_3, &
                                            do_crop_1, do_crop_2)
      !! Map contraction bounds to bounds referring to tensor indices
      !! see dbcsr_t_contract for docu of dummy arguments

      TYPE(dbcsr_t_type), INTENT(IN)      :: tensor_1, tensor_2
      INTEGER, DIMENSION(:), INTENT(IN)   :: contract_1, contract_2, &
                                             notcontract_1, notcontract_2
      INTEGER, DIMENSION(2, dbcsr_t_ndims(tensor_1)), &
         INTENT(OUT)                                 :: bounds_t1
         !! bounds mapped to tensor_1
      INTEGER, DIMENSION(2, dbcsr_t_ndims(tensor_2)), &
         INTENT(OUT)                                 :: bounds_t2
         !! bounds mapped to tensor_2
      INTEGER, DIMENSION(2, SIZE(contract_1)), &
         INTENT(IN), OPTIONAL                        :: bounds_1
      INTEGER, DIMENSION(2, SIZE(notcontract_1)), &
         INTENT(IN), OPTIONAL                        :: bounds_2
      INTEGER, DIMENSION(2, SIZE(notcontract_2)), &
         INTENT(IN), OPTIONAL                        :: bounds_3
      LOGICAL, INTENT(OUT), OPTIONAL                 :: do_crop_1, do_crop_2
         !! whether tensor 1 should be cropped
         !! whether tensor 2 should be cropped
      LOGICAL, DIMENSION(2)                          :: do_crop

      do_crop = .FALSE.

      bounds_t1(1, :) = 1
      CALL dbcsr_t_get_info(tensor_1, nfull_total=bounds_t1(2, :))

      bounds_t2(1, :) = 1
      CALL dbcsr_t_get_info(tensor_2, nfull_total=bounds_t2(2, :))

      IF (PRESENT(bounds_1)) THEN
         bounds_t1(:, contract_1) = bounds_1
         do_crop(1) = .TRUE.
         bounds_t2(:, contract_2) = bounds_1
         do_crop(2) = .TRUE.
      ENDIF

      IF (PRESENT(bounds_2)) THEN
         bounds_t1(:, notcontract_1) = bounds_2
         do_crop(1) = .TRUE.
      ENDIF

      IF (PRESENT(bounds_3)) THEN
         bounds_t2(:, notcontract_2) = bounds_3
         do_crop(2) = .TRUE.
      ENDIF

      IF (PRESENT(do_crop_1)) do_crop_1 = do_crop(1)
      IF (PRESENT(do_crop_2)) do_crop_2 = do_crop(2)

   END SUBROUTINE

   SUBROUTINE dbcsr_t_print_contraction_index(tensor_1, indchar1, tensor_2, indchar2, tensor_3, indchar3, unit_nr)
      !! print tensor contraction indices in a human readable way

      TYPE(dbcsr_t_type), INTENT(IN) :: tensor_1, tensor_2, tensor_3
      CHARACTER(LEN=1), DIMENSION(dbcsr_t_ndims(tensor_1)), INTENT(IN) :: indchar1
         !! characters printed for index of tensor 1
      CHARACTER(LEN=1), DIMENSION(dbcsr_t_ndims(tensor_2)), INTENT(IN) :: indchar2
         !! characters printed for index of tensor 2
      CHARACTER(LEN=1), DIMENSION(dbcsr_t_ndims(tensor_3)), INTENT(IN) :: indchar3
         !! characters printed for index of tensor 3
      INTEGER, INTENT(IN) :: unit_nr
         !! output unit
      INTEGER, DIMENSION(:), ALLOCATABLE :: map11, map12, map21, map22, map31, map32
      INTEGER :: ichar1, ichar2, ichar3

      CALL dbcsr_t_get_mapping_info(tensor_1%nd_index_blk, map1_2d=map11, map2_2d=map12)
      CALL dbcsr_t_get_mapping_info(tensor_2%nd_index_blk, map1_2d=map21, map2_2d=map22)
      CALL dbcsr_t_get_mapping_info(tensor_3%nd_index_blk, map1_2d=map31, map2_2d=map32)

      IF (unit_nr .GT. 0) THEN
         WRITE (unit_nr, '(T2,A)') "INDEX INFO"
         WRITE (unit_nr, '(T15,A)', advance='no') "tensor index: ("
         DO ichar1 = 1, SIZE(indchar1)
            WRITE (unit_nr, '(A1)', advance='no') indchar1(ichar1)
         ENDDO
         WRITE (unit_nr, '(A)', advance='no') ") x ("
         DO ichar2 = 1, SIZE(indchar2)
            WRITE (unit_nr, '(A1)', advance='no') indchar2(ichar2)
         ENDDO
         WRITE (unit_nr, '(A)', advance='no') ") = ("
         DO ichar3 = 1, SIZE(indchar3)
            WRITE (unit_nr, '(A1)', advance='no') indchar3(ichar3)
         ENDDO
         WRITE (unit_nr, '(A)') ")"

         WRITE (unit_nr, '(T15,A)', advance='no') "matrix index: ("
         DO ichar1 = 1, SIZE(map11)
            WRITE (unit_nr, '(A1)', advance='no') indchar1(map11(ichar1))
         ENDDO
         WRITE (unit_nr, '(A1)', advance='no') "|"
         DO ichar1 = 1, SIZE(map12)
            WRITE (unit_nr, '(A1)', advance='no') indchar1(map12(ichar1))
         ENDDO
         WRITE (unit_nr, '(A)', advance='no') ") x ("
         DO ichar2 = 1, SIZE(map21)
            WRITE (unit_nr, '(A1)', advance='no') indchar2(map21(ichar2))
         ENDDO
         WRITE (unit_nr, '(A1)', advance='no') "|"
         DO ichar2 = 1, SIZE(map22)
            WRITE (unit_nr, '(A1)', advance='no') indchar2(map22(ichar2))
         ENDDO
         WRITE (unit_nr, '(A)', advance='no') ") = ("
         DO ichar3 = 1, SIZE(map31)
            WRITE (unit_nr, '(A1)', advance='no') indchar3(map31(ichar3))
         ENDDO
         WRITE (unit_nr, '(A1)', advance='no') "|"
         DO ichar3 = 1, SIZE(map32)
            WRITE (unit_nr, '(A1)', advance='no') indchar3(map32(ichar3))
         ENDDO
         WRITE (unit_nr, '(A)') ")"
      ENDIF

   END SUBROUTINE

END MODULE
