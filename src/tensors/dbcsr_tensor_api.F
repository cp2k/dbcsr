!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tensor_api
   !! This is the start of a dbcsr_tensor_api, all publically needed functions
   !! are exported here. The others remain private to the library.
   !! Currently, this is the CP2K used set.
   !! Ultimately, a reduced subset and well defined api will remain,
   !! possibly grouped in to standard and expert api.
   !! Currently, this is work in progress.

   USE dbcsr_tensor, ONLY: &
      dbcsr_t_contract, dbcsr_t_get_block, dbcsr_t_get_stored_coordinates, dbcsr_t_put_block, &
      dbcsr_t_reserve_blocks, dbcsr_t_copy_matrix_to_tensor, dbcsr_t_copy, &
      dbcsr_t_copy_tensor_to_matrix, dbcsr_t_batched_contract_init, &
      dbcsr_t_batched_contract_finalize, dbcsr_t_contract_index
   USE dbcsr_tensor_block, ONLY: &
      dbcsr_t_iterator_blocks_left, dbcsr_t_iterator_next_block, dbcsr_t_iterator_start, &
      dbcsr_t_iterator_stop, dbcsr_t_iterator_type, dbcsr_t_reserved_block_indices
   USE dbcsr_tensor_types, ONLY: &
      dbcsr_t_create, dbcsr_t_destroy, dbcsr_t_distribution_destroy, dbcsr_t_distribution_new, &
      dbcsr_t_distribution_type, dbcsr_t_nd_mp_comm_prv => dbcsr_t_nd_mp_comm, dbcsr_t_nd_mp_free, dbcsr_t_type, &
      dbcsr_t_pgrid_type, dbcsr_t_pgrid_create_prv => dbcsr_t_pgrid_create, &
      dbcsr_t_pgrid_create_expert_prv => dbcsr_t_pgrid_create_expert, dbcsr_t_pgrid_destroy, &
      dbcsr_t_set, dbcsr_t_filter, &
      dbcsr_t_mp_environ_pgrid => mp_environ_pgrid, dbcsr_t_blk_sizes, dbcsr_t_get_info, &
      dbcsr_t_finalize, dbcsr_t_scale, dbcsr_t_get_nze, dbcsr_t_get_nze_total, &
      dbcsr_t_get_num_blocks, dbcsr_t_get_num_blocks_total, dbcsr_t_clear, &
      dbcsr_t_mp_dims_create, dbcsr_t_pgrid_change_dims, dbcsr_t_ndims => ndims_tensor, &
      dbcsr_t_dims => dims_tensor, dbcsr_t_ndims_matrix_row => ndims_matrix_row, &
      dbcsr_t_ndims_matrix_column => ndims_matrix_column, dbcsr_t_blk_size, dbcsr_t_nblks_local, &
      dbcsr_t_nblks_total, dbcsr_t_max_nblks_local, dbcsr_t_default_distvec
   USE dbcsr_tensor_test, ONLY: &
      dbcsr_t_contract_test, dbcsr_t_checksum
   USE dbcsr_tensor_split, ONLY: &
      dbcsr_t_split_blocks
   USE dbcsr_tensor_index, ONLY: &
      dbcsr_t_get_mapping_info
   USE dbcsr_tensor_io, ONLY: &
      dbcsr_t_write_tensor_info, dbcsr_t_write_split_info, dbcsr_t_write_blocks, dbcsr_t_write_tensor_dist
   USE dbcsr_mpiwrap, ONLY: mp_comm_type

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tensor_api'

   PUBLIC :: dbcsr_t_contract, dbcsr_t_contract_test
   PUBLIC :: dbcsr_t_get_block
   PUBLIC :: dbcsr_t_get_stored_coordinates
   PUBLIC :: dbcsr_t_put_block
   PUBLIC :: dbcsr_t_reserve_blocks
   PUBLIC :: dbcsr_t_create
   PUBLIC :: dbcsr_t_destroy
   PUBLIC :: dbcsr_t_distribution_destroy
   PUBLIC :: dbcsr_t_distribution_new
   PUBLIC :: dbcsr_t_distribution_type
   PUBLIC :: dbcsr_t_nd_mp_comm
   PUBLIC :: dbcsr_t_nd_mp_free
   PUBLIC :: dbcsr_t_type
   PUBLIC :: dbcsr_t_iterator_next_block
   PUBLIC :: dbcsr_t_iterator_blocks_left
   PUBLIC :: dbcsr_t_iterator_stop
   PUBLIC :: dbcsr_t_iterator_start
   PUBLIC :: dbcsr_t_iterator_type
   PUBLIC :: dbcsr_t_split_blocks
   PUBLIC :: dbcsr_t_pgrid_type
   PUBLIC :: dbcsr_t_pgrid_create
   PUBLIC :: dbcsr_t_pgrid_create_expert
   PUBLIC :: dbcsr_t_pgrid_destroy
   PUBLIC :: dbcsr_t_set
   PUBLIC :: dbcsr_t_filter
   PUBLIC :: dbcsr_t_mp_environ_pgrid
   PUBLIC :: dbcsr_t_copy_matrix_to_tensor
   PUBLIC :: dbcsr_t_blk_sizes
   PUBLIC :: dbcsr_t_copy
   PUBLIC :: dbcsr_t_copy_tensor_to_matrix
   PUBLIC :: dbcsr_t_get_info
   PUBLIC :: dbcsr_t_checksum
   PUBLIC :: dbcsr_t_finalize
   PUBLIC :: dbcsr_t_scale
   PUBLIC :: dbcsr_t_get_num_blocks, dbcsr_t_get_num_blocks_total
   PUBLIC :: dbcsr_t_get_nze, dbcsr_t_get_nze_total
   PUBLIC :: dbcsr_t_clear
   PUBLIC :: dbcsr_t_get_mapping_info
   PUBLIC :: dbcsr_t_write_split_info
   PUBLIC :: dbcsr_t_write_blocks
   PUBLIC :: dbcsr_t_write_tensor_dist
   PUBLIC :: dbcsr_t_write_tensor_info
   PUBLIC :: dbcsr_t_mp_dims_create
   PUBLIC :: dbcsr_t_batched_contract_init
   PUBLIC :: dbcsr_t_batched_contract_finalize
   PUBLIC :: dbcsr_t_ndims
   PUBLIC :: dbcsr_t_dims
   PUBLIC :: dbcsr_t_pgrid_change_dims
   PUBLIC :: dbcsr_t_reserved_block_indices
   PUBLIC :: dbcsr_t_contract_index
   PUBLIC :: dbcsr_t_ndims_matrix_row
   PUBLIC :: dbcsr_t_ndims_matrix_column
   PUBLIC :: dbcsr_t_nblks_local
   PUBLIC :: dbcsr_t_nblks_total
   PUBLIC :: dbcsr_t_blk_size
   PUBLIC :: dbcsr_t_max_nblks_local
   PUBLIC :: dbcsr_t_default_distvec

CONTAINS

   FUNCTION dbcsr_t_nd_mp_comm(comm_2d, map1_2d, map2_2d, dims_nd, dims1_nd, dims2_nd, pdims_2d, tdims, &
                               nsplit, dimsplit)
      INTEGER, INTENT(IN)                               :: comm_2d
      INTEGER, DIMENSION(:), INTENT(IN)                 :: map1_2d, map2_2d
      INTEGER, DIMENSION(SIZE(map1_2d) + SIZE(map2_2d)), &
         INTENT(IN), OPTIONAL                           :: dims_nd
      INTEGER, DIMENSION(SIZE(map1_2d)), INTENT(IN), OPTIONAL :: dims1_nd
      INTEGER, DIMENSION(SIZE(map2_2d)), INTENT(IN), OPTIONAL :: dims2_nd
      INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL           :: pdims_2d
      INTEGER, DIMENSION(SIZE(map1_2d) + SIZE(map2_2d)), &
         INTENT(IN), OPTIONAL                           :: tdims
      INTEGER, INTENT(IN), OPTIONAL :: nsplit, dimsplit
      TYPE(dbcsr_t_pgrid_type)                          :: dbcsr_t_nd_mp_comm

      TYPE(mp_comm_type)                                :: my_comm_2d

      CALL my_comm_2d%set_handle(comm_2d)

      dbcsr_t_nd_mp_comm = dbcsr_t_nd_mp_comm_prv(my_comm_2d, map1_2d, map2_2d, &
                                                  dims_nd, dims1_nd, dims2_nd, pdims_2d, tdims, &
                                                  nsplit, dimsplit)

   END FUNCTION dbcsr_t_nd_mp_comm

   SUBROUTINE dbcsr_t_pgrid_create_expert(mp_comm, dims, pgrid, map1_2d, map2_2d, tensor_dims, nsplit, dimsplit)
      INTEGER, INTENT(IN) :: mp_comm
      INTEGER, DIMENSION(:), INTENT(INOUT) :: dims
      TYPE(dbcsr_t_pgrid_type), INTENT(OUT) :: pgrid
      INTEGER, DIMENSION(:), INTENT(IN) :: map1_2d, map2_2d
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL :: tensor_dims
      INTEGER, INTENT(IN), OPTIONAL :: nsplit, dimsplit

      TYPE(mp_comm_type)                                :: my_mp_comm

      CALL my_mp_comm%set_handle(mp_comm)

      CALL dbcsr_t_pgrid_create_expert_prv(my_mp_comm, dims, pgrid, map1_2d, map2_2d, tensor_dims, nsplit, dimsplit)

   END SUBROUTINE dbcsr_t_pgrid_create_expert

   SUBROUTINE dbcsr_t_pgrid_create(mp_comm, dims, pgrid, tensor_dims)
      INTEGER, INTENT(IN) :: mp_comm
      INTEGER, DIMENSION(:), INTENT(INOUT) :: dims
      TYPE(dbcsr_t_pgrid_type), INTENT(OUT) :: pgrid
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL :: tensor_dims

      TYPE(mp_comm_type)                                :: my_mp_comm

      CALL my_mp_comm%set_handle(mp_comm)

      CALL dbcsr_t_pgrid_create_prv(my_mp_comm, dims, pgrid, tensor_dims)

   END SUBROUTINE dbcsr_t_pgrid_create

END MODULE dbcsr_tensor_api
