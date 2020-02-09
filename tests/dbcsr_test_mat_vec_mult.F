!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

module dbcsr_test_mat_vec_mult
   !! Tests for DBCSR multiply
   USE dbcsr_data_methods, ONLY: dbcsr_data_get_sizes, &
                                 dbcsr_data_init, &
                                 dbcsr_data_new, &
                                 dbcsr_data_release, &
                                 dbcsr_scalar_negative, &
                                 dbcsr_scalar_one, &
                                 dbcsr_type_1d_to_2d
   USE dbcsr_dist_methods, ONLY: dbcsr_distribution_new, &
                                 dbcsr_distribution_release

   USE dbcsr_dist_operations, ONLY: dbcsr_dist_bin
   USE dbcsr_machine, ONLY: default_output_unit

   USE dbcsr_io, ONLY: dbcsr_print
   USE dbcsr_kinds, ONLY: real_4, &
                          real_8
   USE dbcsr_methods, ONLY: &
      dbcsr_col_block_offsets, dbcsr_col_block_sizes, dbcsr_get_data_type, &
      dbcsr_get_matrix_type, dbcsr_name, dbcsr_nblkcols_total, dbcsr_nblkrows_total, &
      dbcsr_nfullcols_total, dbcsr_nfullrows_total, dbcsr_release, dbcsr_row_block_offsets, &
      dbcsr_row_block_sizes,dbcsr_mp_grid_remove
   USE dbcsr_mpiwrap, ONLY: mp_bcast, mp_environ, add_mp_perf_env, &
                            mp_cart_create, mp_cart_rank, mp_comm_free,&
                            mp_world_finalize, mp_world_init, rm_mp_perf_env, &
                            mp_sum

   USE dbcsr_mp_methods, ONLY: dbcsr_mp_new, &
                               dbcsr_mp_release,dbcsr_mp_grid_setup

   USE dbcsr_multiply_api, ONLY: dbcsr_multiply
   USE dbcsr_operations, ONLY: dbcsr_copy, &
                               dbcsr_get_occupation, &
                               dbcsr_scale,&
                               dbcsr_get_info
   USE dbcsr_test_methods, ONLY: compx_to_dbcsr_scalar, &
                                 dbcsr_impose_sparsity, &
                                 dbcsr_make_random_block_sizes, &
                                 dbcsr_make_random_matrix, &
                                 dbcsr_random_dist, &
                                 dbcsr_to_dense_local,&
                                 dbcsr_reset_randmat_seed
   USE dbcsr_transformations, ONLY: dbcsr_redistribute, &
                                    dbcsr_replicate_all
   USE dbcsr_types, ONLY: &
      dbcsr_conjugate_transpose, dbcsr_data_obj, dbcsr_distribution_obj, dbcsr_mp_obj, &
      dbcsr_no_transpose, dbcsr_scalar_type, dbcsr_transpose, dbcsr_type, &
      dbcsr_type_antisymmetric, dbcsr_type_complex_4, dbcsr_type_complex_4_2d, &
      dbcsr_type_complex_8, dbcsr_type_complex_8_2d, dbcsr_type_no_symmetry, dbcsr_type_real_4, &
      dbcsr_type_real_4_2d, dbcsr_type_real_8, dbcsr_type_real_8_2d, dbcsr_type_symmetric,&
      dbcsr_iterator
   USE dbcsr_work_operations, ONLY: dbcsr_create

   use dbcsr_vector

   USE dbcsr_iterator_operations,       ONLY: dbcsr_iterator_blocks_left, &
                                              dbcsr_iterator_next_block, &
                                              dbcsr_iterator_stop, &
                                              dbcsr_iterator_start

   USE dbcsr_kinds,                           ONLY: real_4,&
                                                    real_8, sp,dp

   USE dbcsr_config, ONLY: dbcsr_print_config
   USE dbcsr_error_handling, ONLY: dbcsr_error_handling_setup

   USE dbcsr_timings, ONLY: add_timer_env, &
                            rm_timer_env, &
                            timings_register_hooks
   USE dbcsr_timings_report, ONLY: cost_type_time, &
                                   timings_report_callgraph, &
                                   timings_report_print
   USE dbcsr_lib, ONLY: dbcsr_finalize_lib, &
                        dbcsr_init_lib

   USE dbcsr_log_handling, ONLY: dbcsr_add_default_logger, &
                                 dbcsr_logger_create, &
                                 dbcsr_logger_release, &
                                 dbcsr_logger_type, &
                                 dbcsr_rm_default_logger

   use dbcsr_test_multiply, only: dbcsr_check_multiply

#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE

contains

   subroutine dbcsr_mat_vec_multiplies(test_name, mp_comm, mp_env, npdims,&
                                       io_unit, matrix_size, bs_m, sparsity, alpha, beta, mtype)

      CHARACTER(len=*), INTENT(IN)                       :: test_name
      INTEGER, INTENT(IN)                                :: mp_comm
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      INTEGER, DIMENSION(2), INTENT(in)                  :: npdims
      INTEGER, INTENT(IN)                                :: io_unit
      integer, intent(in) :: matrix_size, bs_m
      real(real_8), intent(in)                           :: sparsity
      complex(real_8), INTENT(in)                        :: alpha, beta
      INTEGER, intent(in)                                :: mtype

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_mat_vec_multiplies'

      LOGICAL, PARAMETER :: debug_mod = .FALSE.

      INTEGER                                  :: group, numnodes, mynode, &
                                                  handle
      LOGICAL                                            :: success

      integer                             ::  m,n, k, i ,j, nb, no, nloc, numthreads
      INTEGER, DIMENSION(2)               :: bs
      TYPE(dbcsr_type)                    :: matrix_a, input_vec, result_vec, rep_row_vec, rep_col_vec
      CHARACTER                           :: a_symm
      TYPE(dbcsr_iterator)                          :: dbcsr_iter
      INTEGER, DIMENSION(:), POINTER                     :: rb_offset, rb_size, sizes_m, sizes_k
      complex(8), DIMENSION(:, :), POINTER            :: xvec
      logical                                          :: transposed

      INTEGER, DIMENSION(:), POINTER :: col_dist, row_dist
      TYPE(dbcsr_distribution_obj)                       :: dist
      TYPE(dbcsr_data_obj)                               :: data_a, inp_vec, res_vec, res_vec_init
      INTEGER                                            :: a_c, a_r, v_c, v_r
      !real(real_8) :: residual

      INTEGER, DIMENSION(6)                  :: limits
      TYPE(dbcsr_scalar_type)                            :: alpha_obj, beta_obj

      a_symm = dbcsr_type_no_symmetry

      n = 1
      bs(1) = 1
      m = matrix_size
      k = m
      bs(2) = bs_m

      limits(:) = 1
      limits(2) = m
      limits(4) = n
      limits(6) = k

      !
      ! Number of nodes and rankid
      CALL mp_environ(numnodes, mynode, mp_comm)

      IF (io_unit .GT. 0) THEN
         WRITE (io_unit, *) 'test_name ', test_name
         numthreads = 1
!$OMP PARALLEL
!$OMP MASTER
!$       numthreads = omp_get_num_threads()
!$OMP END MASTER
!$OMP END PARALLEL
         WRITE (io_unit, *) 'numthreads', numthreads
         WRITE (io_unit, *) 'numnodes', numnodes
         WRITE (io_unit, *) 'matrix_size', matrix_size
         WRITE (io_unit, *) 'sparsity', sparsity
         WRITE (io_unit, *) 'alpha', alpha
         WRITE (io_unit, *) 'beta', beta
         WRITE (io_unit, *) 'limits', limits
         WRITE (io_unit, *) 'retain_sparsity', .false.
         WRITE (io_unit, *) 'bs_m', bs_m
         WRITE (io_unit, *) 'bs_n', bs_m
         WRITE (io_unit, *) 'bs_k', bs_m
      END IF

      CALL timeset(routineN, handle)

      CALL dbcsr_make_random_block_sizes(sizes_m, m, bs)
      CALL dbcsr_make_random_block_sizes(sizes_k, k, bs)

      CALL dbcsr_dist_bin(row_dist, SIZE(sizes_m), npdims(1), &
                          sizes_m)
      CALL dbcsr_dist_bin(col_dist, SIZE(sizes_k), npdims(2), &
                          sizes_k)

      CALL dbcsr_distribution_new(dist, mp_env, row_dist, col_dist)

      CALL dbcsr_reset_randmat_seed()

      CALL dbcsr_make_random_matrix(matrix_a, sizes_m, sizes_k, "Matrix A", &
                                    sparsity, &
                                    group, data_type=mtype, dist=dist)

      ! create vectors
      CALL dbcsr_create_col_vec_from_matrix( input_vec , matrix_a , 1)
      CALL dbcsr_create_replicated_col_vec_from_matrix(rep_col_vec, matrix_a, 1)
      CALL dbcsr_create_replicated_row_vec_from_matrix(rep_row_vec, matrix_a, 1)

      ! init input vector
      CALL dbcsr_get_info(input_vec, nfullrows_local=nloc, row_blk_size=rb_size, row_blk_offset=rb_offset)

      CALL dbcsr_iterator_start(dbcsr_iter, input_vec)
      DO WHILE (dbcsr_iterator_blocks_left(dbcsr_iter))
         CALL dbcsr_iterator_next_block(dbcsr_iter, i, j, xvec, transposed)
         nb = rb_size(i)
         no = rb_offset(i)
         xvec(1:nb, 1) = 1  ! test values
      END DO
      CALL dbcsr_iterator_stop(dbcsr_iter)

      ! copy input vector to results vector just to not to be
      CALL dbcsr_copy(result_vec, input_vec)

      ! densify matrix
      a_r = dbcsr_nfullrows_total(matrix_a)
      a_c = dbcsr_nfullcols_total(matrix_a)
      CALL dbcsr_data_init(data_a)
      CALL dbcsr_data_new(data_a, dbcsr_type_1d_to_2d(mtype), data_size=a_r, data_size2=a_c)
      call dbcsr_to_dense_local(matrix_a, data_a)

      ! densify input vector
      v_r = dbcsr_nfullrows_total(input_vec)
      v_c = dbcsr_nfullcols_total(input_vec)
      CALL dbcsr_data_init(inp_vec)
      CALL dbcsr_data_new(inp_vec, dbcsr_type_1d_to_2d(mtype), data_size=v_r, data_size2=v_c)
      call dbcsr_to_dense_local(input_vec, inp_vec)

      ! create result vectors
      CALL dbcsr_data_init(res_vec)
      CALL dbcsr_data_new(res_vec, dbcsr_type_1d_to_2d(mtype), data_size=v_r, data_size2=v_c)
      call dbcsr_to_dense_local(input_vec, res_vec)

      CALL dbcsr_data_init(res_vec_init)
      CALL dbcsr_data_new(res_vec_init, dbcsr_type_1d_to_2d(mtype), data_size=v_r, data_size2=v_c)

      ! multiply mat x vec
      CALL dbcsr_matrix_colvec_multiply(matrix_a, input_vec, result_vec, alpha, &
                                        beta, rep_row_vec, rep_col_vec)

      ! densify result vector
      call dbcsr_to_dense_local(result_vec, res_vec_init)

      alpha_obj = compx_to_dbcsr_scalar(alpha, mtype)
      beta_obj = compx_to_dbcsr_scalar(beta, mtype)

      SELECT CASE (mtype)
      CASE (dbcsr_type_real_8)
         call mp_sum(res_vec_init % d % r2_dp, mp_comm)
         call mp_sum(data_a       % d % r2_dp, mp_comm)
         call mp_sum(inp_vec      % d % r2_dp, mp_comm)
         call mp_sum(res_vec      % d % r2_dp, mp_comm)
      CASE (dbcsr_type_real_4)
         call mp_sum(res_vec_init % d % r2_sp, mp_comm)
         call mp_sum(data_a       % d % r2_sp, mp_comm)
         call mp_sum(inp_vec      % d % r2_sp, mp_comm)
         call mp_sum(res_vec      % d % r2_sp, mp_comm)
      CASE (dbcsr_type_complex_8)
         call mp_sum(res_vec_init % d % c2_dp, mp_comm)
         call mp_sum(data_a       % d % c2_dp, mp_comm)
         call mp_sum(inp_vec      % d % c2_dp, mp_comm)
         call mp_sum(res_vec      % d % c2_dp, mp_comm)
      CASE (dbcsr_type_complex_4)
         call mp_sum(res_vec_init % d % c2_sp, mp_comm)
         call mp_sum(data_a       % d % c2_sp, mp_comm)
         call mp_sum(inp_vec      % d % c2_sp, mp_comm)
         call mp_sum(res_vec      % d % c2_sp, mp_comm)
      END SELECT

      ! check multiply
      CALL dbcsr_check_multiply(test_name, result_vec, res_vec_init, data_a, inp_vec, res_vec, &
                                'N', 'N', alpha_obj, beta_obj, limits, .false., io_unit, mp_comm, &
                                success)

      call dbcsr_release(matrix_a)
      call dbcsr_release(input_vec)
      call dbcsr_release(result_vec)
      call dbcsr_release(rep_col_vec)
      call dbcsr_release(rep_row_vec)
      call dbcsr_data_release(res_vec_init)
      call dbcsr_data_release(res_vec)
      call dbcsr_data_release(data_a)
      call dbcsr_data_release(inp_vec)

      CALL timestop(handle)

   end subroutine dbcsr_mat_vec_multiplies

END module dbcsr_test_mat_vec_mult
