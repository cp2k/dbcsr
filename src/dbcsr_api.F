!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_api
   !! This is the start of a dbcsr_api, all publicly needed functions
   !! are exported here. The others remain private to the library.
   !! Currently, this is the CP2K used set.
   !! Ultimately, a reduced subset and well defined api will remain,
   !! possibly grouped in to standard and expert api.
   !! Currently, this is work in progress.

   USE dbcsr_array_types, ONLY: array_data, &
                                array_exists, &
                                array_size
   USE dbcsr_block_access, ONLY: dbcsr_get_block_p_prv => dbcsr_get_block_p, &
                                 dbcsr_put_block_prv => dbcsr_put_block, &
                                 dbcsr_reserve_all_blocks_prv => dbcsr_reserve_all_blocks, &
                                 dbcsr_reserve_block2d_prv => dbcsr_reserve_block2d, &
                                 dbcsr_reserve_blocks_prv => dbcsr_reserve_blocks, &
                                 dbcsr_reserve_diag_blocks_prv => dbcsr_reserve_diag_blocks
   USE dbcsr_config, ONLY: dbcsr_get_default_config, &
                           dbcsr_print_config, &
                           dbcsr_set_config
   USE dbcsr_csr_conversions, ONLY: &
      convert_csr_to_dbcsr_prv => convert_csr_to_dbcsr, &
      convert_dbcsr_to_csr_prv => convert_dbcsr_to_csr, &
      csr_create_from_dbcsr_prv => csr_create_from_dbcsr, csr_create_new, &
      csr_create_template, dbcsr_csr_dbcsr_blkrow_dist => csr_dbcsr_blkrow_dist, &
      dbcsr_csr_destroy => csr_destroy, dbcsr_csr_eqrow_ceil_dist => csr_eqrow_ceil_dist, &
      dbcsr_csr_eqrow_floor_dist => csr_eqrow_floor_dist, dbcsr_csr_p_type => csr_p_type, &
      dbcsr_csr_print_sparsity => csr_print_sparsity, dbcsr_csr_type => csr_type, &
      dbcsr_csr_write => csr_write, &
      dbcsr_to_csr_filter_prv => dbcsr_to_csr_filter
   USE dbcsr_data_methods, ONLY: dbcsr_get_data_p_prv => dbcsr_get_data_p, &
                                 dbcsr_scalar, &
                                 dbcsr_scalar_fill_all, &
                                 dbcsr_scalar_get_type, &
                                 dbcsr_scalar_get_value, &
                                 dbcsr_scalar_set_type, &
                                 dbcsr_scalar_zero
   USE dbcsr_dist_methods, ONLY: dbcsr_distribution_get_num_images => dbcsr_distribution_get_num_images_1d, &
                                 dbcsr_distribution_hold_prv => dbcsr_distribution_hold, &
                                 dbcsr_distribution_new_prv => dbcsr_distribution_new, &
                                 dbcsr_distribution_release_prv => dbcsr_distribution_release, &
                                 dbcsr_distribution_get_prv => dbcsr_distribution_get
   USE dbcsr_dist_operations, ONLY: dbcsr_get_stored_coordinates_prv => dbcsr_get_stored_coordinates
   USE dbcsr_io, ONLY: dbcsr_binary_read_prv => dbcsr_binary_read, &
                       dbcsr_binary_write_prv => dbcsr_binary_write, &
                       dbcsr_print_block_sum_prv => dbcsr_print_block_sum, &
                       dbcsr_print_prv => dbcsr_print
   USE dbcsr_iterator_operations, ONLY: dbcsr_iterator_blocks_left_prv => dbcsr_iterator_blocks_left, &
                                        dbcsr_iterator_next_block_prv => dbcsr_iterator_next_block, &
                                        dbcsr_iterator_start_prv => dbcsr_iterator_start, &
                                        dbcsr_iterator_stop_prv => dbcsr_iterator_stop
   USE dbcsr_lib, ONLY: dbcsr_clear_mempools, &
                        dbcsr_finalize_lib, &
                        dbcsr_init_lib, dbcsr_print_statistics_prv => dbcsr_print_statistics
   USE dbcsr_methods, ONLY: &
      dbcsr_get_data_size_prv => dbcsr_get_data_size, &
      dbcsr_get_data_type_prv => dbcsr_get_data_type, &
      dbcsr_get_matrix_type_prv => dbcsr_get_matrix_type, &
      dbcsr_get_num_blocks_prv => dbcsr_get_num_blocks, &
      dbcsr_has_symmetry_prv => dbcsr_has_symmetry, &
      dbcsr_nblkcols_total_prv => dbcsr_nblkcols_total, &
      dbcsr_nblkrows_total_prv => dbcsr_nblkrows_total, &
      dbcsr_nblkcols_local_prv => dbcsr_nblkcols_local, &
      dbcsr_nblkrows_local_prv => dbcsr_nblkrows_local, &
      dbcsr_nfullcols_total_prv => dbcsr_nfullcols_total, &
      dbcsr_nfullrows_total_prv => dbcsr_nfullrows_total, &
      dbcsr_release_prv => dbcsr_release, &
      dbcsr_setname_prv => dbcsr_setname, &
      dbcsr_valid_index_prv => dbcsr_valid_index, dbcsr_wm_use_mutable
   USE dbcsr_mp_methods, ONLY: dbcsr_mp_grid_setup_prv => dbcsr_mp_grid_setup, &
                               dbcsr_mp_group, &
                               dbcsr_mp_new_prv => dbcsr_mp_new, &
                               dbcsr_mp_release
   USE dbcsr_multiply_api, ONLY: dbcsr_multiply_prv => dbcsr_multiply
   USE dbcsr_operations, ONLY: &
      dbcsr_add_on_diag_prv => dbcsr_add_on_diag, &
      dbcsr_add_prv => dbcsr_add, &
      dbcsr_copy_into_existing_prv => dbcsr_copy_into_existing, &
      dbcsr_copy_prv => dbcsr_copy, &
      dbcsr_filter_prv => dbcsr_filter, &
      dbcsr_frobenius_norm_prv => dbcsr_frobenius_norm, &
      dbcsr_function_of_elements_prv => dbcsr_function_of_elements, &
      dbcsr_gershgorin_norm_prv => dbcsr_gershgorin_norm, &
      dbcsr_get_block_diag_prv => dbcsr_get_block_diag, &
      dbcsr_get_diag_prv => dbcsr_get_diag, &
      dbcsr_get_info_prv => dbcsr_get_info, &
      dbcsr_get_occupation_prv => dbcsr_get_occupation, &
      dbcsr_hadamard_product_prv => dbcsr_hadamard_product, &
      dbcsr_init_random_prv => dbcsr_init_random, &
      dbcsr_maxabs_prv => dbcsr_maxabs, &
      dbcsr_norm_scalar_prv => dbcsr_norm_scalar, &
      dbcsr_norm_r8_vec_prv => dbcsr_norm_r8_vec, &
      dbcsr_scale_by_vector_prv => dbcsr_scale_by_vector, &
      dbcsr_scale_prv => dbcsr_scale, &
      dbcsr_set_diag_prv => dbcsr_set_diag, &
      dbcsr_set_prv => dbcsr_set, &
      dbcsr_sum_replicated_prv => dbcsr_sum_replicated, &
      dbcsr_trace_prv => dbcsr_trace, &
      dbcsr_dot_prv => dbcsr_dot, &
      dbcsr_triu_prv => dbcsr_triu, &
      dbcsr_clear_prv => dbcsr_clear, &
      dbcsr_add_block_node_prv => dbcsr_add_block_node, &
      dbcsr_conform_scalar_prv => dbcsr_conform_scalar
   USE dbcsr_test_methods, ONLY: dbcsr_reset_randmat_seed
   USE dbcsr_tests, ONLY: dbcsr_run_tests, &
                          dbcsr_test_binary_io, &
                          dbcsr_test_mm
   USE dbcsr_string_utilities, ONLY: uppercase
   USE dbcsr_transformations, ONLY: dbcsr_complete_redistribute_prv => dbcsr_complete_redistribute, &
                                    dbcsr_desymmetrize_deep_prv => dbcsr_desymmetrize_deep, &
                                    dbcsr_distribute_prv => dbcsr_distribute, &
                                    dbcsr_new_transposed_prv => dbcsr_new_transposed, &
                                    dbcsr_replicate_all_prv => dbcsr_replicate_all, &
                                    dbcsr_transposed_prv => dbcsr_transposed
   USE dbcsr_types, ONLY: &
      dbcsr_dist_prv_obj => dbcsr_distribution_obj, dbcsr_func_artanh, dbcsr_func_dtanh, &
      dbcsr_func_inverse, dbcsr_func_tanh, dbcsr_iterator_prv => dbcsr_iterator, dbcsr_mp_obj, &
      dbcsr_no_transpose, dbcsr_norm_column, dbcsr_norm_frobenius, dbcsr_norm_maxabsnorm, &
      dbcsr_prv_type => dbcsr_type, dbcsr_scalar_type, dbcsr_transpose, &
      dbcsr_type_antisymmetric, dbcsr_type_complex_4, dbcsr_type_complex_8, &
      dbcsr_type_complex_default, dbcsr_type_no_symmetry, dbcsr_type_real_4, dbcsr_type_real_8, &
      dbcsr_type_real_default, dbcsr_type_symmetric
   USE dbcsr_dist_util, ONLY: dbcsr_convert_offsets_to_sizes => convert_offsets_to_sizes, &
                              dbcsr_convert_sizes_to_offsets => convert_sizes_to_offsets, &
                              dbcsr_checksum_prv => dbcsr_checksum, &
                              dbcsr_verify_matrix_prv => dbcsr_verify_matrix
   USE dbcsr_work_operations, ONLY: add_work_coordinate_prv => add_work_coordinate, &
                                    dbcsr_create_prv => dbcsr_create, &
                                    dbcsr_finalize_prv => dbcsr_finalize, &
                                    dbcsr_work_create_prv => dbcsr_work_create
   USE dbcsr_kinds, ONLY: default_string_length, &
                          dp, &
                          int_8, &
                          real_4, &
                          real_8
   USE dbcsr_mpiwrap, ONLY: mp_cart_rank, &
                            mp_environ

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_api'

   ! constants
   PUBLIC :: dbcsr_type_no_symmetry
   PUBLIC :: dbcsr_type_symmetric
   PUBLIC :: dbcsr_type_antisymmetric
   PUBLIC :: dbcsr_transpose
   PUBLIC :: dbcsr_no_transpose
   PUBLIC :: dbcsr_type_complex_8
   PUBLIC :: dbcsr_type_real_4
   PUBLIC :: dbcsr_type_real_8
   PUBLIC :: dbcsr_type_complex_4
   PUBLIC :: dbcsr_type_complex_default
   PUBLIC :: dbcsr_type_real_default

   ! types
   PUBLIC :: dbcsr_type
   PUBLIC :: dbcsr_p_type
   PUBLIC :: dbcsr_distribution_type
   PUBLIC :: dbcsr_iterator_type
   PUBLIC :: dbcsr_scalar_type

   ! lib init/finalize
   PUBLIC :: dbcsr_clear_mempools
   PUBLIC :: dbcsr_init_lib
   PUBLIC :: dbcsr_finalize_lib
   PUBLIC :: dbcsr_set_config
   PUBLIC :: dbcsr_get_default_config
   PUBLIC :: dbcsr_print_config
   PUBLIC :: dbcsr_reset_randmat_seed
   PUBLIC :: dbcsr_mp_grid_setup
   PUBLIC :: dbcsr_print_statistics

   ! create / release
   PUBLIC :: dbcsr_distribution_hold
   PUBLIC :: dbcsr_distribution_release
   PUBLIC :: dbcsr_distribution_new
   PUBLIC :: dbcsr_create
   PUBLIC :: dbcsr_init_p
   PUBLIC :: dbcsr_release
   PUBLIC :: dbcsr_release_p
   PUBLIC :: dbcsr_deallocate_matrix

   ! primitive matrix operations
   PUBLIC :: dbcsr_set
   PUBLIC :: dbcsr_add
   PUBLIC :: dbcsr_scale
   PUBLIC :: dbcsr_scale_by_vector
   PUBLIC :: dbcsr_transposed
   PUBLIC :: dbcsr_multiply
   PUBLIC :: dbcsr_copy
   PUBLIC :: dbcsr_copy_into_existing
   PUBLIC :: dbcsr_desymmetrize
   PUBLIC :: dbcsr_add_on_diag
   PUBLIC :: dbcsr_get_block_diag
   PUBLIC :: dbcsr_set_diag
   PUBLIC :: dbcsr_get_diag
   PUBLIC :: dbcsr_filter
   PUBLIC :: dbcsr_trace
   PUBLIC :: dbcsr_dot
   PUBLIC :: dbcsr_complete_redistribute
   PUBLIC :: dbcsr_get_block_p
   PUBLIC :: dbcsr_clear

   ! block reservation
   PUBLIC :: dbcsr_reserve_diag_blocks
   PUBLIC :: dbcsr_reserve_block2d
   PUBLIC :: dbcsr_reserve_blocks
   PUBLIC :: dbcsr_reserve_all_blocks

   ! iterator
   PUBLIC :: dbcsr_iterator_start
   PUBLIC :: dbcsr_iterator_stop
   PUBLIC :: dbcsr_iterator_blocks_left
   PUBLIC :: dbcsr_iterator_next_block

   ! getters / setters
   PUBLIC :: dbcsr_get_info
   PUBLIC :: dbcsr_distribution_get
   PUBLIC :: dbcsr_setname
   PUBLIC :: dbcsr_get_matrix_type
   PUBLIC :: dbcsr_get_occupation
   PUBLIC :: dbcsr_nblkrows_total
   PUBLIC :: dbcsr_nblkcols_total
   PUBLIC :: dbcsr_nblkrows_local
   PUBLIC :: dbcsr_nblkcols_local
   PUBLIC :: dbcsr_get_num_blocks
   PUBLIC :: dbcsr_get_data_size
   PUBLIC :: dbcsr_has_symmetry
   PUBLIC :: dbcsr_nfullrows_total
   PUBLIC :: dbcsr_nfullcols_total
   PUBLIC :: dbcsr_get_stored_coordinates
   PUBLIC :: dbcsr_valid_index
   PUBLIC :: dbcsr_get_data_type

   ! work operations
   PUBLIC :: dbcsr_add_block_node
   PUBLIC :: dbcsr_put_block
   PUBLIC :: dbcsr_work_create
   PUBLIC :: dbcsr_verify_matrix
   PUBLIC :: dbcsr_add_work_coordinate
   PUBLIC :: dbcsr_get_wms_data_p
   PUBLIC :: dbcsr_get_data_p
   PUBLIC :: dbcsr_set_work_size
   PUBLIC :: dbcsr_finalize

   ! replication
   PUBLIC :: dbcsr_replicate_all
   PUBLIC :: dbcsr_sum_replicated
   PUBLIC :: dbcsr_distribute

   ! misc
   PUBLIC :: dbcsr_distribution_get_num_images
   PUBLIC :: dbcsr_convert_offsets_to_sizes
   PUBLIC :: dbcsr_convert_sizes_to_offsets
   PUBLIC :: dbcsr_run_tests
   PUBLIC :: dbcsr_test_mm
   PUBLIC :: dbcsr_scalar

   ! high level matrix functions
   PUBLIC :: dbcsr_norm_frobenius
   PUBLIC :: dbcsr_norm_maxabsnorm
   PUBLIC :: dbcsr_norm_column
   PUBLIC :: dbcsr_hadamard_product
   PUBLIC :: dbcsr_func_artanh
   PUBLIC :: dbcsr_func_dtanh
   PUBLIC :: dbcsr_func_inverse
   PUBLIC :: dbcsr_func_tanh
   PUBLIC :: dbcsr_print
   PUBLIC :: dbcsr_print_block_sum
   PUBLIC :: dbcsr_checksum
   PUBLIC :: dbcsr_maxabs
   PUBLIC :: dbcsr_norm
   PUBLIC :: dbcsr_gershgorin_norm
   PUBLIC :: dbcsr_frobenius_norm
   PUBLIC :: dbcsr_init_random
   PUBLIC :: dbcsr_function_of_elements
   PUBLIC :: dbcsr_triu

   ! csr conversion
   PUBLIC :: dbcsr_csr_type
   PUBLIC :: dbcsr_csr_p_type
   PUBLIC :: dbcsr_convert_csr_to_dbcsr
   PUBLIC :: dbcsr_convert_dbcsr_to_csr
   PUBLIC :: dbcsr_csr_create_from_dbcsr
   PUBLIC :: dbcsr_csr_destroy
   PUBLIC :: dbcsr_csr_create
   PUBLIC :: dbcsr_csr_eqrow_floor_dist
   PUBLIC :: dbcsr_csr_eqrow_ceil_dist
   PUBLIC :: dbcsr_csr_dbcsr_blkrow_dist
   PUBLIC :: dbcsr_csr_print_sparsity
   PUBLIC :: dbcsr_to_csr_filter
   PUBLIC :: dbcsr_csr_write

   ! binary io
   PUBLIC :: dbcsr_binary_write
   PUBLIC :: dbcsr_binary_read
   PUBLIC :: dbcsr_test_binary_io

   ! -----------------------------------------------------------------------------------------------
   TYPE dbcsr_p_type
      TYPE(dbcsr_type), POINTER :: matrix => Null()
   END TYPE dbcsr_p_type

   ! the components of this type must remain private to encapsulate better the internals
   ! of the dbcsr library.
   TYPE dbcsr_type
      TYPE(dbcsr_prv_type), PRIVATE        :: prv
   END TYPE dbcsr_type

   TYPE dbcsr_distribution_type
      TYPE(dbcsr_dist_prv_obj), PRIVATE        :: prv
   END TYPE dbcsr_distribution_type

   TYPE dbcsr_iterator_type
      TYPE(dbcsr_iterator_prv), PRIVATE        :: prv
   END TYPE dbcsr_iterator_type

   INTERFACE dbcsr_create
      MODULE PROCEDURE dbcsr_create_new, dbcsr_create_template
   END INTERFACE

   INTERFACE dbcsr_trace
      MODULE PROCEDURE dbcsr_trace_d, dbcsr_trace_s
      MODULE PROCEDURE dbcsr_trace_z, dbcsr_trace_c
   END INTERFACE

   INTERFACE dbcsr_dot
      MODULE PROCEDURE dbcsr_dot_d, dbcsr_dot_s
      MODULE PROCEDURE dbcsr_dot_z, dbcsr_dot_c
   END INTERFACE

   INTERFACE dbcsr_set
      MODULE PROCEDURE dbcsr_set_d, dbcsr_set_s, dbcsr_set_c, dbcsr_set_z
   END INTERFACE

   INTERFACE dbcsr_add
      MODULE PROCEDURE dbcsr_add_d, dbcsr_add_s, dbcsr_add_c, dbcsr_add_z
   END INTERFACE

   INTERFACE dbcsr_add_on_diag
      MODULE PROCEDURE dbcsr_add_on_diag_d, dbcsr_add_on_diag_s
      MODULE PROCEDURE dbcsr_add_on_diag_c, dbcsr_add_on_diag_z
   END INTERFACE

   INTERFACE dbcsr_get_diag
      MODULE PROCEDURE dbcsr_get_diag_d, dbcsr_get_diag_s
      MODULE PROCEDURE dbcsr_get_diag_c, dbcsr_get_diag_z
   END INTERFACE

   INTERFACE dbcsr_set_diag
      MODULE PROCEDURE dbcsr_set_diag_d, dbcsr_set_diag_s
      MODULE PROCEDURE dbcsr_set_diag_c, dbcsr_set_diag_z
   END INTERFACE

   INTERFACE dbcsr_scale
      MODULE PROCEDURE dbcsr_scale_d, dbcsr_scale_s, dbcsr_scale_c, dbcsr_scale_z
   END INTERFACE

   INTERFACE dbcsr_scale_by_vector
      MODULE PROCEDURE dbcsr_scale_by_vector_d, dbcsr_scale_by_vector_s
      MODULE PROCEDURE dbcsr_scale_by_vector_c, dbcsr_scale_by_vector_z
   END INTERFACE

   INTERFACE dbcsr_multiply
      MODULE PROCEDURE dbcsr_multiply_d, dbcsr_multiply_s, dbcsr_multiply_c, dbcsr_multiply_z
   END INTERFACE

   INTERFACE dbcsr_get_block_p
      MODULE PROCEDURE dbcsr_get_block_p_d, dbcsr_get_block_p_s
      MODULE PROCEDURE dbcsr_get_block_p_z, dbcsr_get_block_p_c
      MODULE PROCEDURE dbcsr_get_2d_block_p_d, dbcsr_get_2d_block_p_s
      MODULE PROCEDURE dbcsr_get_2d_block_p_z, dbcsr_get_2d_block_p_c
      MODULE PROCEDURE dbcsr_get_block_notrans_p_d, dbcsr_get_block_notrans_p_s
      MODULE PROCEDURE dbcsr_get_block_notrans_p_z, dbcsr_get_block_notrans_p_c
      MODULE PROCEDURE dbcsr_get_2d_block_notrans_p_d, dbcsr_get_2d_block_notrans_p_s
      MODULE PROCEDURE dbcsr_get_2d_block_notrans_p_z, dbcsr_get_2d_block_notrans_p_c
   END INTERFACE

   INTERFACE dbcsr_put_block
      MODULE PROCEDURE dbcsr_put_block_d, dbcsr_put_block_s, dbcsr_put_block_z, dbcsr_put_block_c
      MODULE PROCEDURE dbcsr_put_block2d_d, dbcsr_put_block2d_s, dbcsr_put_block2d_z, dbcsr_put_block2d_c
   END INTERFACE

   INTERFACE dbcsr_iterator_next_block
      MODULE PROCEDURE dbcsr_iterator_next_block_index
      MODULE PROCEDURE dbcsr_iterator_next_2d_block_d, dbcsr_iterator_next_2d_block_s
      MODULE PROCEDURE dbcsr_iterator_next_2d_block_c, dbcsr_iterator_next_2d_block_z
      MODULE PROCEDURE dbcsr_iterator_next_1d_block_d, dbcsr_iterator_next_1d_block_s
      MODULE PROCEDURE dbcsr_iterator_next_1d_block_c, dbcsr_iterator_next_1d_block_z
      MODULE PROCEDURE dbcsr_iterator_next_2d_block_notrans_d, dbcsr_iterator_next_2d_block_notrans_s
      MODULE PROCEDURE dbcsr_iterator_next_2d_block_notrans_c, dbcsr_iterator_next_2d_block_notrans_z
      MODULE PROCEDURE dbcsr_iterator_next_1d_block_notrans_d, dbcsr_iterator_next_1d_block_notrans_s
      MODULE PROCEDURE dbcsr_iterator_next_1d_block_notrans_c, dbcsr_iterator_next_1d_block_notrans_z
   END INTERFACE

   INTERFACE dbcsr_reserve_block2d
      MODULE PROCEDURE dbcsr_reserve_block2d_d, dbcsr_reserve_block2d_s
      MODULE PROCEDURE dbcsr_reserve_block2d_c, dbcsr_reserve_block2d_z
   END INTERFACE

   INTERFACE dbcsr_csr_create
      MODULE PROCEDURE csr_create_new, csr_create_template
   END INTERFACE

   INTERFACE dbcsr_get_wms_data_p
      MODULE PROCEDURE dbcsr_get_wms_data_s, dbcsr_get_wms_data_c
      MODULE PROCEDURE dbcsr_get_wms_data_d, dbcsr_get_wms_data_z
   END INTERFACE

   INTERFACE dbcsr_get_data_p
      MODULE PROCEDURE dbcsr_get_data_s, dbcsr_get_data_c, dbcsr_get_data_d, dbcsr_get_data_z
   END INTERFACE

   INTERFACE dbcsr_norm
      MODULE PROCEDURE dbcsr_norm_scalar
      MODULE PROCEDURE dbcsr_norm_r8_vec
   END INTERFACE dbcsr_norm

   PRIVATE

CONTAINS

   SUBROUTINE dbcsr_mp_grid_setup(dist)
      TYPE(dbcsr_distribution_type), INTENT(INOUT)       :: dist

      CALL dbcsr_mp_grid_setup_prv(dist%prv%d%mp_env)
   END SUBROUTINE dbcsr_mp_grid_setup

   SUBROUTINE dbcsr_setname(matrix, newname)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      CHARACTER(len=*), INTENT(IN)                       :: newname

      CALL dbcsr_setname_prv(matrix%prv, newname)
   END SUBROUTINE dbcsr_setname

   FUNCTION dbcsr_gershgorin_norm(matrix) RESULT(norm)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      REAL(KIND=real_8)                                  :: norm

      norm = dbcsr_gershgorin_norm_prv(matrix%prv)
   END FUNCTION dbcsr_gershgorin_norm

   FUNCTION dbcsr_frobenius_norm(matrix, local) RESULT(norm)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      LOGICAL, INTENT(in), OPTIONAL                      :: local
      REAL(KIND=real_8)                                  :: norm

      norm = dbcsr_frobenius_norm_prv(matrix%prv, local)
   END FUNCTION dbcsr_frobenius_norm

   FUNCTION dbcsr_maxabs(matrix) RESULT(norm)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      REAL(KIND=real_8)                                  :: norm

      norm = dbcsr_maxabs_prv(matrix%prv)
   END FUNCTION dbcsr_maxabs

   SUBROUTINE dbcsr_complete_redistribute(matrix, redist, keep_sparsity, summation)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      TYPE(dbcsr_type), INTENT(INOUT)                    :: redist
      LOGICAL, INTENT(IN), OPTIONAL                      :: keep_sparsity, summation

      CALL dbcsr_complete_redistribute_prv(matrix%prv, redist%prv, keep_sparsity, summation)
   END SUBROUTINE dbcsr_complete_redistribute

   SUBROUTINE dbcsr_reserve_blocks(matrix, rows, cols, blk_pointers)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      INTEGER, DIMENSION(:), INTENT(IN)                  :: rows, cols
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL        :: blk_pointers

      CALL dbcsr_reserve_blocks_prv(matrix%prv, rows, cols, blk_pointers)
   END SUBROUTINE dbcsr_reserve_blocks

   SUBROUTINE dbcsr_reserve_all_blocks(matrix)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix

      CALL dbcsr_reserve_all_blocks_prv(matrix%prv)
   END SUBROUTINE dbcsr_reserve_all_blocks

   SUBROUTINE dbcsr_reserve_diag_blocks(matrix)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix

      CALL dbcsr_reserve_diag_blocks_prv(matrix%prv)
   END SUBROUTINE dbcsr_reserve_diag_blocks

   SUBROUTINE dbcsr_add_work_coordinate(matrix, index_matrix, row, col, blk, index)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      INTEGER, INTENT(IN)                                :: index_matrix, row, col
      INTEGER, INTENT(IN), OPTIONAL                      :: blk
      INTEGER, INTENT(OUT), OPTIONAL                     :: index

      CALL add_work_coordinate_prv(matrix%prv%wms(index_matrix), row, col, blk, index)
   END SUBROUTINE dbcsr_add_work_coordinate

   SUBROUTINE dbcsr_set_work_size(matrix, index_matrix, newvalue)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      INTEGER, INTENT(IN)                                :: index_matrix, newvalue

      matrix%prv%wms(index_matrix)%datasize = newvalue
   END SUBROUTINE dbcsr_set_work_size

   SUBROUTINE dbcsr_init_random(matrix, keep_sparsity)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      LOGICAL, OPTIONAL                                  :: keep_sparsity

      CALL dbcsr_init_random_prv(matrix%prv, keep_sparsity=keep_sparsity)
   END SUBROUTINE dbcsr_init_random

   PURE FUNCTION dbcsr_get_data_type(matrix) RESULT(data_type)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: data_type

      data_type = dbcsr_get_data_type_prv(matrix%prv)
   END FUNCTION dbcsr_get_data_type

   PURE FUNCTION dbcsr_valid_index(matrix) RESULT(valid_index)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      LOGICAL                                            :: valid_index

      valid_index = dbcsr_valid_index_prv(matrix%prv)
   END FUNCTION dbcsr_valid_index

   SUBROUTINE dbcsr_get_stored_coordinates(matrix, row, column, processor)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER, INTENT(IN)                                :: row, column
      INTEGER, INTENT(OUT), OPTIONAL                     :: processor

      CALL dbcsr_get_stored_coordinates_prv(matrix%prv, row, column, processor)
   END SUBROUTINE dbcsr_get_stored_coordinates

   PURE FUNCTION dbcsr_get_num_blocks(matrix) RESULT(num_blocks)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: num_blocks

      num_blocks = dbcsr_get_num_blocks_prv(matrix%prv)
   END FUNCTION dbcsr_get_num_blocks

   FUNCTION dbcsr_get_data_size(matrix) RESULT(data_size)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: data_size

      data_size = dbcsr_get_data_size_prv(matrix%prv)
   END FUNCTION dbcsr_get_data_size

   PURE FUNCTION dbcsr_get_matrix_type(matrix) RESULT(matrix_type)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      CHARACTER                                          :: matrix_type

      matrix_type = dbcsr_get_matrix_type_prv(matrix%prv)
   END FUNCTION dbcsr_get_matrix_type

   FUNCTION dbcsr_get_occupation(matrix) RESULT(occupation)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      REAL(KIND=real_8)                                  :: occupation

      occupation = dbcsr_get_occupation_prv(matrix%prv)
   END FUNCTION dbcsr_get_occupation

   FUNCTION dbcsr_nblkrows_total(matrix) RESULT(nblkrows_total)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: nblkrows_total

      nblkrows_total = dbcsr_nblkrows_total_prv(matrix%prv)
   END FUNCTION dbcsr_nblkrows_total

   FUNCTION dbcsr_nblkcols_total(matrix) RESULT(nblkcols_total)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: nblkcols_total

      nblkcols_total = dbcsr_nblkcols_total_prv(matrix%prv)
   END FUNCTION dbcsr_nblkcols_total
   
   FUNCTION dbcsr_nblkrows_local(matrix) RESULT(nblkrows_local)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: nblkrows_local

      nblkrows_local = dbcsr_nblkrows_local_prv(matrix%prv)
   END FUNCTION dbcsr_nblkrows_local

   FUNCTION dbcsr_nblkcols_local(matrix) RESULT(nblkcols_local)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: nblkcols_local

      nblkcols_local = dbcsr_nblkcols_local_prv(matrix%prv)
   END FUNCTION dbcsr_nblkcols_local

   FUNCTION dbcsr_nfullrows_total(matrix) RESULT(nfullrows_total)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: nfullrows_total

      nfullrows_total = dbcsr_nfullrows_total_prv(matrix%prv)
   END FUNCTION dbcsr_nfullrows_total

   FUNCTION dbcsr_nfullcols_total(matrix) RESULT(nfullcols_total)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: nfullcols_total

      nfullcols_total = dbcsr_nfullcols_total_prv(matrix%prv)
   END FUNCTION dbcsr_nfullcols_total

   PURE FUNCTION dbcsr_iterator_blocks_left(iterator) RESULT(blocks_left)
      TYPE(dbcsr_iterator_type), INTENT(IN)              :: iterator
      LOGICAL                                            :: blocks_left

      blocks_left = dbcsr_iterator_blocks_left_prv(iterator%prv)
   END FUNCTION dbcsr_iterator_blocks_left

   SUBROUTINE dbcsr_iterator_stop(iterator)
      TYPE(dbcsr_iterator_type), INTENT(INOUT)           :: iterator

      CALL dbcsr_iterator_stop_prv(iterator%prv)
   END SUBROUTINE dbcsr_iterator_stop

   SUBROUTINE dbcsr_iterator_start(iterator, matrix, shared, dynamic, &
                                   dynamic_byrows, contiguous_pointers, read_only)
      TYPE(dbcsr_iterator_type), INTENT(OUT)             :: iterator
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      LOGICAL, INTENT(IN), OPTIONAL                      :: shared, dynamic, dynamic_byrows, &
                                                            contiguous_pointers, read_only

      CALL dbcsr_iterator_start_prv(iterator%prv, matrix%prv, shared, dynamic, &
                                    dynamic_byrows, contiguous_pointers, read_only)
   END SUBROUTINE dbcsr_iterator_start

   SUBROUTINE dbcsr_iterator_next_block_index(iterator, row, column, blk, blk_p)
      !! Gets the index information of the next block, no data.

      TYPE(dbcsr_iterator_type), INTENT(INOUT)           :: iterator
         !! the iterator
      INTEGER, INTENT(OUT)                               :: row, column, blk
         !! row of the data block
         !! column of the data block
         !! block number
      INTEGER, INTENT(OUT), OPTIONAL                     :: blk_p
         !! index into block data array

      CALL dbcsr_iterator_next_block_prv(iterator%prv, row=row, column=column, blk=blk, blk_p=blk_p)
   END SUBROUTINE dbcsr_iterator_next_block_index

   SUBROUTINE dbcsr_get_info(matrix, nblkrows_total, nblkcols_total, &
                             nfullrows_total, nfullcols_total, &
                             nblkrows_local, nblkcols_local, &
                             nfullrows_local, nfullcols_local, &
                             my_prow, my_pcol, &
                             local_rows, local_cols, proc_row_dist, proc_col_dist, &
                             row_blk_size, col_blk_size, row_blk_offset, col_blk_offset, &
                             distribution, name, matrix_type, data_type, &
                             group)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER, INTENT(OUT), OPTIONAL :: nblkrows_total, nblkcols_total, nfullrows_total, &
                                        nfullcols_total, nblkrows_local, nblkcols_local, nfullrows_local, nfullcols_local, &
                                        my_prow, my_pcol
      INTEGER, DIMENSION(:), OPTIONAL, POINTER           :: local_rows, local_cols, proc_row_dist, &
                                                            proc_col_dist, &
                                                            row_blk_size, col_blk_size, &
                                                            row_blk_offset, col_blk_offset
      TYPE(dbcsr_distribution_type), INTENT(OUT), &
         OPTIONAL                                        :: distribution
      CHARACTER(len=*), INTENT(OUT), OPTIONAL            :: name
      CHARACTER, INTENT(OUT), OPTIONAL                   :: matrix_type
      INTEGER, INTENT(OUT), OPTIONAL                     :: data_type, group

      TYPE(dbcsr_dist_prv_obj)                           :: dist

      CALL dbcsr_get_info_prv(matrix=matrix%prv, &
                              nblkrows_total=nblkrows_total, &
                              nblkcols_total=nblkcols_total, &
                              nfullrows_total=nfullrows_total, &
                              nfullcols_total=nfullcols_total, &
                              nblkrows_local=nblkrows_local, &
                              nblkcols_local=nblkcols_local, &
                              nfullrows_local=nfullrows_local, &
                              nfullcols_local=nfullcols_local, &
                              my_prow=my_prow, &
                              my_pcol=my_pcol, &
                              local_rows=local_rows, &
                              local_cols=local_cols, &
                              proc_row_dist=proc_row_dist, &
                              proc_col_dist=proc_col_dist, &
                              row_blk_size=row_blk_size, &
                              col_blk_size=col_blk_size, &
                              row_blk_offset=row_blk_offset, &
                              col_blk_offset=col_blk_offset, &
                              distribution=dist, &
                              name=name, &
                              matrix_type=matrix_type, &
                              data_type=data_type, &
                              group=group)

      IF (PRESENT(distribution)) THEN
         distribution%prv = dist
      ENDIF

   END SUBROUTINE dbcsr_get_info

   SUBROUTINE dbcsr_distribution_get(dist, row_dist, col_dist, &
                                     nrows, ncols, has_threads, &
                                     group, mynode, numnodes, nprows, npcols, myprow, mypcol, pgrid, &
                                     subgroups_defined, prow_group, pcol_group)
      TYPE(dbcsr_distribution_type), INTENT(IN)          :: dist
      INTEGER, DIMENSION(:), OPTIONAL, POINTER           :: row_dist, col_dist
      INTEGER, INTENT(OUT), OPTIONAL                     :: nrows, ncols
      LOGICAL, INTENT(OUT), OPTIONAL                     :: has_threads
      INTEGER, INTENT(OUT), OPTIONAL                     :: group, mynode, numnodes, nprows, npcols, &
                                                            myprow, mypcol
      INTEGER, DIMENSION(:, :), OPTIONAL, POINTER        :: pgrid
      LOGICAL, INTENT(OUT), OPTIONAL                     :: subgroups_defined
      INTEGER, INTENT(OUT), OPTIONAL                     :: prow_group, pcol_group

      call dbcsr_distribution_get_prv(dist%prv, row_dist, col_dist, &
                                      nrows, ncols, has_threads, &
                                      group, mynode, numnodes, nprows, npcols, myprow, mypcol, pgrid, &
                                      subgroups_defined, prow_group, pcol_group)
   END SUBROUTINE dbcsr_distribution_get

   SUBROUTINE dbcsr_distribution_hold(dist)
      TYPE(dbcsr_distribution_type)                      :: dist

      CALL dbcsr_distribution_hold_prv(dist%prv)
   END SUBROUTINE dbcsr_distribution_hold

   SUBROUTINE dbcsr_distribution_release(dist)
      TYPE(dbcsr_distribution_type)                      :: dist

      CALL dbcsr_distribution_release_prv(dist%prv)
   END SUBROUTINE dbcsr_distribution_release

   SUBROUTINE dbcsr_norm_scalar(matrix, which_norm, norm_scalar)

      TYPE(dbcsr_type), INTENT(INOUT), TARGET            :: matrix
      INTEGER, INTENT(IN)                                :: which_norm
      REAL(dp), INTENT(OUT)                              :: norm_scalar

      CALL dbcsr_norm_scalar_prv(matrix%prv, which_norm, norm_scalar)
   END SUBROUTINE dbcsr_norm_scalar

   SUBROUTINE dbcsr_norm_r8_vec(matrix, which_norm, norm_vector)

      TYPE(dbcsr_type), INTENT(INOUT), TARGET            :: matrix
      INTEGER, INTENT(IN)                                :: which_norm
      REAL(dp), DIMENSION(:), INTENT(OUT)                :: norm_vector

      CALL dbcsr_norm_r8_vec_prv(matrix%prv, which_norm, norm_vector)
   END SUBROUTINE dbcsr_norm_r8_vec

   SUBROUTINE dbcsr_replicate_all(matrix)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix

      CALL dbcsr_replicate_all_prv(matrix%prv)
   END SUBROUTINE dbcsr_replicate_all

   SUBROUTINE dbcsr_distribute(matrix, fast)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      LOGICAL, INTENT(in), OPTIONAL                      :: fast

      CALL dbcsr_distribute_prv(matrix%prv, fast)
   END SUBROUTINE dbcsr_distribute

   SUBROUTINE dbcsr_release_p(matrix)
      TYPE(dbcsr_type), POINTER                          :: matrix

      IF (ASSOCIATED(matrix)) THEN
         CALL dbcsr_release(matrix)
         DEALLOCATE (matrix)
      ENDIF
   END SUBROUTINE dbcsr_release_p

   SUBROUTINE dbcsr_release(matrix)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix

      CALL dbcsr_release_prv(matrix%prv)
   END SUBROUTINE dbcsr_release

   SUBROUTINE dbcsr_init_p(matrix)
      TYPE(dbcsr_type), POINTER                          :: matrix

      IF (ASSOCIATED(matrix)) THEN
         CALL dbcsr_release(matrix)
         DEALLOCATE (matrix)
      ENDIF

      ALLOCATE (matrix)
   END SUBROUTINE dbcsr_init_p

   SUBROUTINE dbcsr_print(matrix, nodata, matlab_format, variable_name, unit_nr)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      LOGICAL, INTENT(IN), OPTIONAL                      :: nodata, matlab_format
      CHARACTER(*), INTENT(in), OPTIONAL                 :: variable_name
      INTEGER, OPTIONAL                                  :: unit_nr

      CALL dbcsr_print_prv(matrix%prv, nodata, matlab_format, variable_name, unit_nr)
   END SUBROUTINE dbcsr_print

   SUBROUTINE dbcsr_print_block_sum(matrix, unit_nr)
      !! Prints the sum of the elements in each block
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER, OPTIONAL                                  :: unit_nr

      CALL dbcsr_print_block_sum_prv(matrix%prv, unit_nr)
   END SUBROUTINE dbcsr_print_block_sum

   FUNCTION dbcsr_checksum(matrix, local, pos) RESULT(checksum)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      LOGICAL, INTENT(IN), OPTIONAL                      :: local, pos
      REAL(KIND=dp)                                      :: checksum

      checksum = dbcsr_checksum_prv(matrix%prv, local=local, pos=pos)
   END FUNCTION dbcsr_checksum

   SUBROUTINE dbcsr_sum_replicated(matrix)
      TYPE(dbcsr_type), INTENT(inout)                    :: matrix

      CALL dbcsr_sum_replicated_prv(matrix%prv)
   END SUBROUTINE dbcsr_sum_replicated

   SUBROUTINE dbcsr_triu(matrix)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix

      CALL dbcsr_triu_prv(matrix%prv)
   END SUBROUTINE dbcsr_triu

   SUBROUTINE dbcsr_verify_matrix(matrix, verbosity, local)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER, INTENT(IN), OPTIONAL                      :: verbosity
      LOGICAL, INTENT(IN), OPTIONAL                      :: local

      CALL dbcsr_verify_matrix_prv(matrix%prv, verbosity, local)
   END SUBROUTINE dbcsr_verify_matrix

   SUBROUTINE dbcsr_distribution_new(dist, template, group, pgrid, row_dist, col_dist, &
                                     reuse_arrays)
      !! Creates new distribution from blockr distributions

      TYPE(dbcsr_distribution_type), INTENT(OUT)         :: dist
         !! distribution
      TYPE(dbcsr_distribution_type), INTENT(IN), &
         OPTIONAL                                        :: template
      INTEGER, INTENT(IN), OPTIONAL                      :: group
      INTEGER, DIMENSION(:, :), OPTIONAL, POINTER        :: pgrid
      INTEGER, DIMENSION(:), INTENT(INOUT), POINTER      :: row_dist, col_dist
      LOGICAL, INTENT(IN), OPTIONAL                      :: reuse_arrays

      INTEGER, DIMENSION(:), POINTER, CONTIGUOUS         :: cont_row_dist, cont_col_dist

      ! Make the arrays contiguous, avoid change in the API
      ALLOCATE (cont_row_dist(SIZE(row_dist)), cont_col_dist(SIZE(col_dist)))
      cont_row_dist(:) = row_dist(:)
      cont_col_dist(:) = col_dist(:)

      IF (PRESENT(reuse_arrays)) THEN
         IF (reuse_arrays) THEN
            DEALLOCATE (row_dist, col_dist)
            NULLIFY (row_dist, col_dist)
         ENDIF
      ENDIF

      IF (PRESENT(template)) THEN
         call dbcsr_distribution_new_prv(dist%prv, template%prv, group, pgrid, cont_row_dist, cont_col_dist, &
                                         reuse_arrays=.TRUE.)
      ELSE
         call dbcsr_distribution_new_prv(dist%prv, group=group, pgrid=pgrid, row_dist=cont_row_dist, col_dist=cont_col_dist, &
                                         reuse_arrays=.TRUE.)
      ENDIF
   END SUBROUTINE dbcsr_distribution_new

   SUBROUTINE dbcsr_print_statistics(print_timers, callgraph_filename)
      !! Print statistics
      LOGICAL, INTENT(IN), OPTIONAL          :: print_timers
      CHARACTER(len=*), INTENT(IN), OPTIONAL :: callgraph_filename

      CALL dbcsr_print_statistics_prv(print_timers, callgraph_filename)
   END SUBROUTINE dbcsr_print_statistics

   SUBROUTINE dbcsr_finalize(matrix, reshuffle)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      LOGICAL, INTENT(IN), OPTIONAL                      :: reshuffle

      CALL dbcsr_finalize_prv(matrix%prv, reshuffle)
   END SUBROUTINE dbcsr_finalize

   SUBROUTINE dbcsr_work_create(matrix, nblks_guess, sizedata_guess, n, work_mutable)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      INTEGER, INTENT(IN), OPTIONAL                      :: nblks_guess, sizedata_guess, n
      LOGICAL, INTENT(in), OPTIONAL                      :: work_mutable

      CALL dbcsr_work_create_prv(matrix%prv, nblks_guess, sizedata_guess, n, work_mutable)
   END SUBROUTINE dbcsr_work_create

   SUBROUTINE dbcsr_create_new(matrix, name, dist, matrix_type, &
                               row_blk_size, col_blk_size, nze, data_type, reuse, &
                               reuse_arrays, mutable_work, replication_type)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      CHARACTER(len=*), INTENT(IN)                       :: name
      TYPE(dbcsr_distribution_type), INTENT(IN)          :: dist
      CHARACTER, INTENT(IN)                              :: matrix_type
      INTEGER, DIMENSION(:), INTENT(INOUT), POINTER      :: row_blk_size, col_blk_size
      INTEGER, INTENT(IN), OPTIONAL                      :: nze, data_type
      LOGICAL, INTENT(IN), OPTIONAL                      :: reuse, reuse_arrays, mutable_work
      CHARACTER, INTENT(IN), OPTIONAL                    :: replication_type

      INTEGER, DIMENSION(:), POINTER, CONTIGUOUS         :: cont_row_blk_size, cont_col_blk_size

      ! Make the array contiguous, avoid to change API
      ALLOCATE (cont_row_blk_size(SIZE(row_blk_size)), cont_col_blk_size(SIZE(col_blk_size)))
      cont_row_blk_size(:) = row_blk_size(:)
      cont_col_blk_size(:) = col_blk_size(:)
      IF (PRESENT(reuse_arrays)) THEN
         IF (reuse_arrays) THEN
            DEALLOCATE (row_blk_size, col_blk_size)
            NULLIFY (row_blk_size, col_blk_size)
         ENDIF
      ENDIF
      CALL dbcsr_create_prv(matrix%prv, name, dist%prv, &
                            matrix_type, &
                            cont_row_blk_size, cont_col_blk_size, nze=nze, &
                            data_type=data_type, reuse=reuse, &
                            reuse_arrays=.TRUE., &
                            mutable_work=mutable_work, replication_type=replication_type)
   END SUBROUTINE dbcsr_create_new

   SUBROUTINE dbcsr_create_template(matrix, name, template, &
                                    dist, matrix_type, &
                                    row_blk_size, col_blk_size, nze, data_type, &
                                    reuse_arrays, mutable_work, replication_type)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      CHARACTER(len=*), INTENT(IN), OPTIONAL             :: name
      TYPE(dbcsr_type), INTENT(IN)                       :: template
      TYPE(dbcsr_distribution_type), INTENT(IN), &
         OPTIONAL                                        :: dist
      CHARACTER, INTENT(IN), OPTIONAL                    :: matrix_type
      INTEGER, DIMENSION(:), INTENT(INOUT), OPTIONAL, &
         POINTER                                         :: row_blk_size, col_blk_size
      INTEGER, INTENT(IN), OPTIONAL                      :: nze, data_type
      LOGICAL, INTENT(IN), OPTIONAL                      :: reuse_arrays, mutable_work
      CHARACTER, INTENT(IN), OPTIONAL                    :: replication_type

      INTEGER, DIMENSION(:), POINTER, CONTIGUOUS         :: cont_row_blk_size, cont_col_blk_size

      IF (PRESENT(row_blk_size) .NEQV. PRESENT(col_blk_size)) THEN
         DBCSR_ABORT("Both row_blk_size and col_blk_size must be provided!")
      ENDIF

      ! Make the array contiguous, avoid to change API
      IF (PRESENT(row_blk_size)) THEN
         ! Avoid to change API
         ALLOCATE (cont_row_blk_size(SIZE(row_blk_size)), cont_col_blk_size(SIZE(col_blk_size)))
         cont_row_blk_size(:) = row_blk_size(:)
         cont_col_blk_size(:) = col_blk_size(:)
         IF (PRESENT(reuse_arrays)) THEN
            IF (reuse_arrays) THEN
               DEALLOCATE (row_blk_size, col_blk_size)
               NULLIFY (row_blk_size, col_blk_size)
            ENDIF
         ENDIF
      ENDIF

      IF (PRESENT(dist)) THEN
         IF (PRESENT(row_blk_size)) THEN
            CALL dbcsr_create_prv(matrix%prv, template%prv, name, &
                                  dist%prv, matrix_type, &
                                  row_blk_size=cont_row_blk_size, col_blk_size=cont_col_blk_size, &
                                  nze=nze, data_type=data_type, &
                                  reuse_arrays=.TRUE., mutable_work=mutable_work, &
                                  replication_type=replication_type)
         ELSE
            CALL dbcsr_create_prv(matrix%prv, template%prv, name, &
                                  dist%prv, matrix_type, &
                                  nze=nze, data_type=data_type, &
                                  reuse_arrays=reuse_arrays, mutable_work=mutable_work, &
                                  replication_type=replication_type)
         ENDIF
      ELSE
         IF (PRESENT(row_blk_size)) THEN
            CALL dbcsr_create_prv(matrix%prv, template%prv, name, &
                                  matrix_type=matrix_type, &
                                  row_blk_size=cont_row_blk_size, col_blk_size=cont_col_blk_size, &
                                  nze=nze, data_type=data_type, &
                                  reuse_arrays=.TRUE., mutable_work=mutable_work, &
                                  replication_type=replication_type)
         ELSE
            CALL dbcsr_create_prv(matrix%prv, template%prv, name, &
                                  matrix_type=matrix_type, &
                                  nze=nze, data_type=data_type, &
                                  reuse_arrays=reuse_arrays, mutable_work=mutable_work, &
                                  replication_type=replication_type)
         ENDIF
      ENDIF
   END SUBROUTINE dbcsr_create_template

   SUBROUTINE dbcsr_filter(matrix, eps, method, use_absolute, filter_diag)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      REAL(dp), INTENT(IN)                               :: eps
      INTEGER, INTENT(IN), OPTIONAL                      :: method
      LOGICAL, INTENT(in), OPTIONAL                      :: use_absolute, filter_diag

      CALL dbcsr_filter_prv(matrix%prv, dbcsr_conform_scalar_prv(eps, matrix%prv), &
                            method, use_absolute, filter_diag)
   END SUBROUTINE dbcsr_filter

   SUBROUTINE dbcsr_get_block_diag(matrix, diag)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      TYPE(dbcsr_type), INTENT(INOUT)                    :: diag

      CALL dbcsr_get_block_diag_prv(matrix%prv, diag%prv)
   END SUBROUTINE dbcsr_get_block_diag

   SUBROUTINE dbcsr_binary_write(matrix, filepath)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      CHARACTER(LEN=*), INTENT(IN)                       :: filepath

      CALL dbcsr_binary_write_prv(matrix%prv, filepath)
   END SUBROUTINE dbcsr_binary_write

   SUBROUTINE dbcsr_binary_read(filepath, distribution, groupid, matrix_new)
      CHARACTER(len=*), INTENT(IN)                       :: filepath
      TYPE(dbcsr_distribution_type), INTENT(IN)          :: distribution
      INTEGER, INTENT(IN), OPTIONAL                      :: groupid
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_new

      CALL dbcsr_binary_read_prv(filepath, distribution%prv, groupid, matrix_new%prv)
   END SUBROUTINE dbcsr_binary_read

   SUBROUTINE dbcsr_copy(matrix_b, matrix_a, name, keep_sparsity, &
                         shallow_data, keep_imaginary, matrix_type)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_b
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_a
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL             :: name
      LOGICAL, INTENT(IN), OPTIONAL                      :: keep_sparsity, shallow_data, &
                                                            keep_imaginary
      CHARACTER, INTENT(IN), OPTIONAL                    :: matrix_type

      CALL dbcsr_copy_prv(matrix_b%prv, matrix_a%prv, name, keep_sparsity, &
                          shallow_data, keep_imaginary, matrix_type)
   END SUBROUTINE dbcsr_copy

   SUBROUTINE dbcsr_copy_into_existing(matrix_b, matrix_a)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_b
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_a

      CALL dbcsr_copy_into_existing_prv(matrix_b%prv, matrix_a%prv)
   END SUBROUTINE dbcsr_copy_into_existing

   SUBROUTINE dbcsr_desymmetrize(matrix_a, matrix_b)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_a
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_b

      CALL dbcsr_desymmetrize_deep_prv(matrix_a%prv, matrix_b%prv, untransposed_data=.TRUE.)
   END SUBROUTINE dbcsr_desymmetrize

   SUBROUTINE dbcsr_transposed(transposed, normal, shallow_data_copy, &
                               transpose_data, transpose_distribution, use_distribution)

      TYPE(dbcsr_type), INTENT(INOUT)                    :: transposed
      TYPE(dbcsr_type), INTENT(IN)                       :: normal
      LOGICAL, INTENT(IN), OPTIONAL                      :: shallow_data_copy, transpose_data, &
                                                            transpose_distribution
      TYPE(dbcsr_distribution_type), INTENT(IN), &
         OPTIONAL                                        :: use_distribution

      IF (PRESENT(use_distribution)) THEN
         CALL dbcsr_transposed_prv(transposed%prv, normal%prv, shallow_data_copy, &
                                   transpose_data, transpose_distribution, &
                                   use_distribution%prv)
      ELSE
         CALL dbcsr_transposed_prv(transposed%prv, normal%prv, shallow_data_copy, &
                                   transpose_data, transpose_distribution)
      ENDIF
   END SUBROUTINE dbcsr_transposed

   SUBROUTINE dbcsr_function_of_elements(matrix_a, func, a0, a1, a2)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_a
      INTEGER, INTENT(IN)                                :: func
      REAL(kind=dp), INTENT(IN), OPTIONAL                :: a0, a1, a2

      CALL dbcsr_function_of_elements_prv(matrix_a%prv, func, a0, a1, a2)
   END SUBROUTINE dbcsr_function_of_elements

   SUBROUTINE dbcsr_hadamard_product(matrix_a, matrix_b, matrix_c, b_assume_value)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_a, matrix_b
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_c
      REAL(KIND=dp), INTENT(IN), OPTIONAL                :: b_assume_value

      CALL dbcsr_hadamard_product_prv(matrix_a%prv, matrix_b%prv, matrix_c%prv, b_assume_value)
   END SUBROUTINE dbcsr_hadamard_product

   SUBROUTINE dbcsr_deallocate_matrix(matrix)
      !! Deallocates a DBCSR matrix for compatibility with CP2K

      TYPE(dbcsr_type), POINTER                          :: matrix
         !! DBCSR matrix

      CALL dbcsr_release(matrix)
      IF (dbcsr_valid_index(matrix)) &
         CALL dbcsr_abort(__LOCATION__, &
                          'You should not "deallocate" a referenced matrix. '// &
                          'Avoid pointers to DBCSR matrices.')
      DEALLOCATE (matrix)
   END SUBROUTINE dbcsr_deallocate_matrix

   PURE FUNCTION dbcsr_has_symmetry(matrix) RESULT(has_symmetry)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      LOGICAL                                            :: has_symmetry

      has_symmetry = dbcsr_has_symmetry_prv(matrix%prv)
   END FUNCTION dbcsr_has_symmetry

   SUBROUTINE dbcsr_csr_create_from_dbcsr(dbcsr_mat, csr_mat, dist_format, csr_sparsity, numnodes)

      TYPE(dbcsr_type), INTENT(IN)                       :: dbcsr_mat
      TYPE(dbcsr_csr_type), INTENT(OUT)                  :: csr_mat
      INTEGER                                            :: dist_format
      TYPE(dbcsr_type), INTENT(IN), OPTIONAL             :: csr_sparsity
      INTEGER, INTENT(IN), OPTIONAL                      :: numnodes

      IF (PRESENT(csr_sparsity)) THEN
         CALL csr_create_from_dbcsr_prv(dbcsr_mat%prv, csr_mat, dist_format, csr_sparsity%prv, numnodes)
      ELSE
         CALL csr_create_from_dbcsr_prv(dbcsr_mat%prv, csr_mat, dist_format, numnodes=numnodes)
      ENDIF
   END SUBROUTINE dbcsr_csr_create_from_dbcsr

   SUBROUTINE dbcsr_convert_csr_to_dbcsr(dbcsr_mat, csr_mat)

      TYPE(dbcsr_type), INTENT(INOUT)                    :: dbcsr_mat
      TYPE(dbcsr_csr_type), INTENT(INOUT)                :: csr_mat

      CALL convert_csr_to_dbcsr_prv(dbcsr_mat%prv, csr_mat)
   END SUBROUTINE dbcsr_convert_csr_to_dbcsr

   SUBROUTINE dbcsr_convert_dbcsr_to_csr(dbcsr_mat, csr_mat)
      TYPE(dbcsr_type), INTENT(IN)                       :: dbcsr_mat
      TYPE(dbcsr_csr_type), INTENT(INOUT)                :: csr_mat

      CALL convert_dbcsr_to_csr_prv(dbcsr_mat%prv, csr_mat)
   END SUBROUTINE dbcsr_convert_dbcsr_to_csr

   SUBROUTINE dbcsr_to_csr_filter(dbcsr_mat, csr_sparsity, eps)
      !! Apply filtering threshold eps to DBCSR blocks in order to improve
      !! CSR sparsity (currently only used for testing purposes)

      TYPE(dbcsr_type), INTENT(IN)                       :: dbcsr_mat
      TYPE(dbcsr_type), INTENT(OUT)                      :: csr_sparsity
      REAL(kind=real_8), INTENT(IN)                      :: eps

      CALL dbcsr_to_csr_filter_prv(dbcsr_mat%prv, csr_sparsity%prv, eps)
   END SUBROUTINE dbcsr_to_csr_filter

   SUBROUTINE dbcsr_clear(dbcsr_mat)
      !! Clear a matrix
      TYPE(dbcsr_type), INTENT(INOUT) :: dbcsr_mat

      CALL dbcsr_clear_prv(dbcsr_mat%prv)
   END SUBROUTINE

   SUBROUTINE dbcsr_add_block_node(matrix, block_row, block_col, block)
      !! Emulation of sparse_matrix_types/add_block_node mapped
      !! to add_real_matrix_block.... should not be used any longer
      !! It adds a block to the dbcsr matrix and returns a rank-2 pointer to the
      !! block. Currently it only and always uses the mutable data.

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
         !! DBCSR matrix
      INTEGER, INTENT(IN)                                :: block_row, block_col
         !! the row
         !! the column
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: block
         !! the block to put

      call dbcsr_add_block_node_prv(matrix%prv, block_row, block_col, block)
   END SUBROUTINE dbcsr_add_block_node

#:include 'data/dbcsr.fypp'
#:for n, nametype1, base1, prec1, kind1, type1, dkind1 in inst_params_float
   SUBROUTINE dbcsr_reserve_block2d_${nametype1}$ (matrix, row, col, block, transposed, existed)
      TYPE(dbcsr_type), INTENT(INOUT)          :: matrix
      INTEGER, INTENT(IN)                      :: row, col
      ${type1}$, DIMENSION(:, :), POINTER        :: block
      LOGICAL, INTENT(IN), OPTIONAL            :: transposed
      LOGICAL, INTENT(OUT), OPTIONAL           :: existed

      CALL dbcsr_reserve_block2d_prv(matrix%prv, row, col, block, &
                                     transposed, existed)
   END SUBROUTINE dbcsr_reserve_block2d_${nametype1}$

   SUBROUTINE dbcsr_iterator_next_2d_block_${nametype1}$ (iterator, row, column, block, &
                                                          transposed, block_number, &
                                                          row_size, col_size, row_offset, col_offset)
      TYPE(dbcsr_iterator_type), INTENT(INOUT) :: iterator
      INTEGER, INTENT(OUT)                     :: row, column
      ${type1}$, DIMENSION(:, :), POINTER        :: block
      LOGICAL, INTENT(OUT)                     :: transposed
      INTEGER, INTENT(OUT), OPTIONAL           :: block_number, row_size, &
                                                  col_size, row_offset, &
                                                  col_offset

      CALL dbcsr_iterator_next_block_prv(iterator%prv, row, column, block, transposed, &
                                         block_number, row_size, col_size, row_offset, col_offset)
   END SUBROUTINE dbcsr_iterator_next_2d_block_${nametype1}$

   SUBROUTINE dbcsr_iterator_next_2d_block_notrans_${nametype1}$ (iterator, row, column, block, &
                                                                  block_number, &
                                                                  row_size, col_size, row_offset, col_offset)
      TYPE(dbcsr_iterator_type), INTENT(INOUT) :: iterator
      INTEGER, INTENT(OUT)                     :: row, column
      ${type1}$, DIMENSION(:, :), POINTER        :: block
      INTEGER, INTENT(OUT), OPTIONAL           :: block_number, row_size, &
                                                  col_size, row_offset, &
                                                  col_offset

      LOGICAL                                  :: tr

      CALL dbcsr_iterator_next_block_prv(iterator%prv, row, column, block, tr, &
                                         block_number, row_size, col_size, row_offset, col_offset)
      IF (tr) DBCSR_ABORT("Block is transposed!")
   END SUBROUTINE dbcsr_iterator_next_2d_block_notrans_${nametype1}$

! **************************************************************************************************
! *************************************************************************************************
   SUBROUTINE dbcsr_iterator_next_1d_block_${nametype1}$ (iterator, row, column, block, &
                                                          transposed, block_number, &
                                                          row_size, col_size, row_offset, col_offset)
      TYPE(dbcsr_iterator_type), INTENT(INOUT)  :: iterator
      INTEGER, INTENT(OUT)                      :: row, column
      ${type1}$, DIMENSION(:), POINTER            :: block
      LOGICAL, INTENT(OUT)                      :: transposed
      INTEGER, INTENT(OUT), OPTIONAL            :: block_number, row_size, &
                                                   col_size, row_offset, &
                                                   col_offset

      CALL dbcsr_iterator_next_block_prv(iterator%prv, row, column, block, &
                                         transposed, block_number, row_size, col_size, row_offset, col_offset)
   END SUBROUTINE dbcsr_iterator_next_1d_block_${nametype1}$

! **************************************************************************************************
! *************************************************************************************************
   SUBROUTINE dbcsr_iterator_next_1d_block_notrans_${nametype1}$ (iterator, row, column, block, &
                                                                  block_number, &
                                                                  row_size, col_size, row_offset, col_offset)
      TYPE(dbcsr_iterator_type), INTENT(INOUT)  :: iterator
      INTEGER, INTENT(OUT)                      :: row, column
      ${type1}$, DIMENSION(:), POINTER            :: block
      INTEGER, INTENT(OUT), OPTIONAL            :: block_number, row_size, &
                                                   col_size, row_offset, &
                                                   col_offset

      LOGICAL                                   :: tr

      CALL dbcsr_iterator_next_block_prv(iterator%prv, row, column, block, &
                                         tr, block_number, row_size, col_size, row_offset, col_offset)
      IF (tr) DBCSR_ABORT("Block is transposed!")
   END SUBROUTINE dbcsr_iterator_next_1d_block_notrans_${nametype1}$

   SUBROUTINE dbcsr_put_block2d_${nametype1}$ (matrix, row, col, block, &
                                               summation, scale)
      TYPE(dbcsr_type), INTENT(INOUT)          :: matrix
      INTEGER, INTENT(IN)                      :: row, col
      ${type1}$, DIMENSION(:, :), INTENT(IN)     :: block
      LOGICAL, INTENT(IN), OPTIONAL            :: summation
      ${type1}$, INTENT(IN), OPTIONAL            :: scale

      CALL dbcsr_put_block_prv(matrix%prv, row, col, block, summation=summation, scale=scale)
   END SUBROUTINE dbcsr_put_block2d_${nametype1}$

   SUBROUTINE dbcsr_put_block_${nametype1}$ (matrix, row, col, block, &
                                             summation, scale)
      TYPE(dbcsr_type), INTENT(INOUT)          :: matrix
      INTEGER, INTENT(IN)                      :: row, col
      ${type1}$, DIMENSION(:), INTENT(IN)        :: block
      LOGICAL, INTENT(IN), OPTIONAL            :: summation
      ${type1}$, INTENT(IN), OPTIONAL            :: scale

      CALL dbcsr_put_block_prv(matrix%prv, row, col, block, summation=summation, scale=scale)
   END SUBROUTINE dbcsr_put_block_${nametype1}$

   SUBROUTINE dbcsr_get_2d_block_p_${nametype1}$ (matrix, row, col, block, tr, found, row_size, col_size)
      TYPE(dbcsr_type), INTENT(INOUT)          :: matrix
      INTEGER, INTENT(IN)                      :: row, col
      ${type1}$, DIMENSION(:, :), POINTER        :: block
      LOGICAL, INTENT(OUT)                     :: tr
      LOGICAL, INTENT(OUT)                     :: found
      INTEGER, INTENT(OUT), OPTIONAL           :: row_size, col_size

      CALL dbcsr_get_block_p_prv(matrix%prv, row, col, block, tr, found, row_size, col_size)
   END SUBROUTINE dbcsr_get_2d_block_p_${nametype1}$

   SUBROUTINE dbcsr_get_2d_block_notrans_p_${nametype1}$ (matrix, row, col, block, found, row_size, col_size)
      TYPE(dbcsr_type), INTENT(INOUT)          :: matrix
      INTEGER, INTENT(IN)                      :: row, col
      ${type1}$, DIMENSION(:, :), POINTER        :: block
      LOGICAL, INTENT(OUT)                     :: found
      INTEGER, INTENT(OUT), OPTIONAL           :: row_size, col_size

      LOGICAL                                  :: tr

      CALL dbcsr_get_block_p_prv(matrix%prv, row, col, block, tr, found, row_size, col_size)
      IF (tr) DBCSR_ABORT("Block is transposed!")
   END SUBROUTINE dbcsr_get_2d_block_notrans_p_${nametype1}$

   SUBROUTINE dbcsr_get_block_p_${nametype1}$ (matrix, row, col, block, tr, found, row_size, col_size)
      TYPE(dbcsr_type), INTENT(IN)              :: matrix
      INTEGER, INTENT(IN)                       :: row, col
      ${type1}$, DIMENSION(:), POINTER            :: block
      LOGICAL, INTENT(OUT)                      :: tr
      LOGICAL, INTENT(OUT)                      :: found
      INTEGER, INTENT(OUT), OPTIONAL            :: row_size, col_size

      CALL dbcsr_get_block_p_prv(matrix%prv, row, col, block, tr, found, row_size, col_size)
   END SUBROUTINE dbcsr_get_block_p_${nametype1}$

   SUBROUTINE dbcsr_get_block_notrans_p_${nametype1}$ (matrix, row, col, block, found, row_size, col_size)
      TYPE(dbcsr_type), INTENT(IN)              :: matrix
      INTEGER, INTENT(IN)                       :: row, col
      ${type1}$, DIMENSION(:), POINTER            :: block
      LOGICAL, INTENT(OUT)                      :: found
      INTEGER, INTENT(OUT), OPTIONAL            :: row_size, col_size

      LOGICAL                                   :: tr

      CALL dbcsr_get_block_p_prv(matrix%prv, row, col, block, tr, found, row_size, col_size)
      IF (tr) DBCSR_ABORT("Block is transposed!")
   END SUBROUTINE dbcsr_get_block_notrans_p_${nametype1}$

   SUBROUTINE dbcsr_trace_${nametype1}$ (matrix_a, trace)
      TYPE(dbcsr_type), INTENT(IN)              :: matrix_a
      ${type1}$, INTENT(OUT)                     :: trace

      CALL dbcsr_trace_prv(matrix_a%prv, trace)
   END SUBROUTINE dbcsr_trace_${nametype1}$

   SUBROUTINE dbcsr_dot_${nametype1}$ (matrix_a, matrix_b, result)
      TYPE(dbcsr_type), INTENT(IN)              :: matrix_a, matrix_b
      ${type1}$, INTENT(INOUT)                  :: result

      CALL dbcsr_dot_prv(matrix_a%prv, matrix_b%prv, result)
   END SUBROUTINE dbcsr_dot_${nametype1}$

   SUBROUTINE dbcsr_multiply_${nametype1}$ (transa, transb, &
                                            alpha, matrix_a, matrix_b, beta, matrix_c, &
                                            first_row, last_row, first_column, last_column, first_k, last_k, &
                                            retain_sparsity, filter_eps, flop)
      CHARACTER(LEN=1), INTENT(IN)             :: transa, transb
      ${type1}$, INTENT(IN)                      :: alpha
      TYPE(dbcsr_type), INTENT(IN)             :: matrix_a, matrix_b
      ${type1}$, INTENT(IN)                      :: beta
      TYPE(dbcsr_type), INTENT(INOUT)          :: matrix_c
      INTEGER, INTENT(IN), OPTIONAL            :: first_row, last_row, &
                                                  first_column, last_column, &
                                                  first_k, last_k
      LOGICAL, INTENT(IN), OPTIONAL            :: retain_sparsity
      REAL(kind=${kind2[n]}$), INTENT(IN), OPTIONAL :: filter_eps
      INTEGER(int_8), INTENT(OUT), OPTIONAL    :: flop

      CALL dbcsr_multiply_prv(transa, transb, &
                              alpha, matrix_a%prv, matrix_b%prv, beta, matrix_c%prv, &
                              first_row, last_row, first_column, last_column, first_k, last_k, &
                              retain_sparsity, &
                              filter_eps=filter_eps, &
                              flop=flop)
   END SUBROUTINE dbcsr_multiply_${nametype1}$

   SUBROUTINE dbcsr_scale_by_vector_${nametype1}$ (matrix_a, alpha, side)
      TYPE(dbcsr_type), INTENT(INOUT)           :: matrix_a
      ${type1}$, DIMENSION(:), INTENT(IN), TARGET :: alpha
      CHARACTER(LEN=*), INTENT(IN)              :: side

      CALL dbcsr_scale_by_vector_prv(matrix_a%prv, alpha, side)
   END SUBROUTINE dbcsr_scale_by_vector_${nametype1}$

   SUBROUTINE dbcsr_scale_${nametype1}$ (matrix_a, alpha_scalar, last_column)
      TYPE(dbcsr_type), INTENT(INOUT)          :: matrix_a
      ${type1}$, INTENT(IN)                      :: alpha_scalar
      INTEGER, INTENT(IN), OPTIONAL            :: last_column

      CALL dbcsr_scale_prv(matrix_a%prv, alpha_scalar, last_column)
   END SUBROUTINE dbcsr_scale_${nametype1}$

   SUBROUTINE dbcsr_set_${nametype1}$ (matrix, alpha)
      TYPE(dbcsr_type), INTENT(INOUT)       :: matrix
      ${type1}$, INTENT(IN)                      :: alpha

      CALL dbcsr_set_prv(matrix%prv, alpha)
   END SUBROUTINE dbcsr_set_${nametype1}$

   SUBROUTINE dbcsr_add_${nametype1}$ (matrix_a, matrix_b, alpha_scalar, beta_scalar)
      TYPE(dbcsr_type), INTENT(INOUT)          :: matrix_a
      TYPE(dbcsr_type), INTENT(IN)             :: matrix_b
      ${type1}$, INTENT(IN)                      :: alpha_scalar, beta_scalar

      CALL dbcsr_add_prv(matrix_a%prv, matrix_b%prv, alpha_scalar, beta_scalar)
   END SUBROUTINE dbcsr_add_${nametype1}$

   SUBROUTINE dbcsr_add_on_diag_${nametype1}$ (matrix, alpha_scalar)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      ${type1}$, INTENT(IN)                                :: alpha_scalar

      CALL dbcsr_add_on_diag_prv(matrix%prv, alpha_scalar)
   END SUBROUTINE dbcsr_add_on_diag_${nametype1}$

   SUBROUTINE dbcsr_set_diag_${nametype1}$ (matrix, diag)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      ${type1}$, DIMENSION(:), INTENT(IN)                  :: diag

      CALL dbcsr_set_diag_prv(matrix%prv, diag)
   END SUBROUTINE dbcsr_set_diag_${nametype1}$

   SUBROUTINE dbcsr_get_diag_${nametype1}$ (matrix, diag)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      ${type1}$, DIMENSION(:), INTENT(OUT)                 :: diag

      CALL dbcsr_get_diag_prv(matrix%prv, diag)
   END SUBROUTINE dbcsr_get_diag_${nametype1}$

   FUNCTION dbcsr_get_wms_data_${nametype1}$ (matrix, index_matrix, select_data_type, lb, ub) RESULT(DATA)
      TYPE(dbcsr_type), INTENT(IN)     :: matrix
      INTEGER, INTENT(IN)              :: index_matrix
      ${type1}$, INTENT(IN)              :: select_data_type
      ${type1}$, DIMENSION(:), POINTER   :: DATA
      INTEGER, INTENT(IN), OPTIONAL    :: lb, ub

      DATA => dbcsr_get_data_p_prv(matrix%prv%wms(index_matrix)%data_area, select_data_type, lb, ub)
   END FUNCTION dbcsr_get_wms_data_${nametype1}$

   FUNCTION dbcsr_get_data_${nametype1}$ (matrix, select_data_type, lb, ub) RESULT(DATA)
      TYPE(dbcsr_type), INTENT(IN)     :: matrix
      ${type1}$, INTENT(IN)              :: select_data_type
      ${type1}$, DIMENSION(:), POINTER   :: DATA
      INTEGER, INTENT(IN), OPTIONAL    :: lb, ub

      DATA => dbcsr_get_data_p_prv(matrix%prv%data_area, select_data_type, lb, ub)
   END FUNCTION dbcsr_get_data_${nametype1}$
#:endfor

END MODULE dbcsr_api
