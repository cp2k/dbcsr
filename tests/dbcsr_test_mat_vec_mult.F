!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief   Tests for DBCSR multiply
!> \author  VW
!> \date    2010
!> \version 1.0
!>
!> <b>Modification history:</b>
!> - Created 2010
! **************************************************************************************************
PROGRAM dbcsr_test_mat_vec_mult
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
                            mp_world_finalize, mp_world_init, rm_mp_perf_env

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

   !PRIVATE

   !PUBLIC :: dbcsr_test_multiplies

   CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_test_multiply'

   LOGICAL, PARAMETER :: debug_mod = .FALSE.

   TYPE(dbcsr_logger_type), POINTER            :: logger

   INTEGER                                  :: mp_comm, group, numnodes, mynode, &
                                               prow, pcol, io_unit, handle
   INTEGER, DIMENSION(2)                    :: npdims, myploc
   INTEGER, DIMENSION(:, :), POINTER        :: pgrid
   TYPE(dbcsr_mp_obj)                       :: mp_env

    !INTEGER                                            :: c_a, c_b, c_c, handle, r_a, r_b, r_c
    !INTEGER, DIMENSION(:), POINTER                     :: blk_offsets, col_dist_a, col_dist_b, &
                                                          !col_dist_c, row_dist_a, row_dist_b, &
                                                          !row_dist_c
    LOGICAL                                            :: success
    !REAL(real_8)                                       :: occ_a, occ_b, occ_c_in, occ_c_out
    !TYPE(dbcsr_distribution_obj)                       :: dist_a, dist_b, dist_c
    !TYPE(dbcsr_type)                                   :: m_a, m_b, m_c

    integer                             ::  m,n, k, i ,j, nb, no, nloc
    INTEGER, DIMENSION(2)               :: bs
    real(real_8)                        :: sparsity
    TYPE(dbcsr_type)                    :: matrix_a, input_vec, result_vec, rep_row_vec, rep_col_vec
    CHARACTER                           :: a_symm
    INTEGER                             :: mtype 
    TYPE(dbcsr_iterator)                          :: dbcsr_iter
    INTEGER, DIMENSION(:), POINTER                     :: rb_offset, rb_size, sizes_m, sizes_k
    REAL(KIND=real_8), DIMENSION(:, :), POINTER            :: xvec
    logical                                          :: transposed
        
    INTEGER, DIMENSION(:), POINTER :: col_dist, row_dist
    TYPE(dbcsr_distribution_obj)                       :: dist
    TYPE(dbcsr_data_obj)                               :: data_a, data_b, data_c, inp_vec, &
                                                          res_vec, res_vec_init, diff
    INTEGER                                            :: a_c, a_r, v_c, v_r
    !real(real_8) :: residual
  
    INTEGER, DIMENSION(6)                  :: limits  
    
    TYPE(dbcsr_scalar_type)                            :: alpha_obj, beta_obj

    a_symm = dbcsr_type_no_symmetry
    mtype = dbcsr_type_real_8
  
     n = 1 
     bs(1) = 1 
     open(unit=123,file="mnkb.dat",status="old",form="formatted")
      read(123,*) m
      read(123,*) k
      read(123,*) bs(2)
      read(123,*) sparsity
      close(123)


    limits(:) = 1
    limits(2) = m
    limits(4) = n
    limits(6) = k

    CALL mp_world_init(mp_comm)

   !
   ! Number of nodes and rankid
   CALL mp_environ(numnodes, mynode, mp_comm)

   npdims(:) = 0
   CALL mp_cart_create(mp_comm, 2, npdims, myploc, group)


  
  ALLOCATE (pgrid(0:npdims(1) - 1, 0:npdims(2) - 1))
   DO prow = 0, npdims(1) - 1
      DO pcol = 0, npdims(2) - 1
         CALL mp_cart_rank(group, (/prow, pcol/), pgrid(prow, pcol))
      ENDDO
   ENDDO
   CALL dbcsr_mp_new(mp_env, pgrid, group, mynode, numnodes, &
                     myprow=myploc(1), mypcol=myploc(2))
   
   call dbcsr_mp_grid_setup(mp_env)              
   DEALLOCATE (pgrid)

   !
   ! set standard output parameters
   io_unit = 0
   IF (mynode .EQ. mp_env%mp%source) io_unit = default_output_unit
   
   NULLIFY (logger)
   CALL dbcsr_logger_create(logger, mp_env=mp_env, &
                            default_global_unit_nr=io_unit, &
                            close_global_unit_on_dealloc=.FALSE.)
   CALL dbcsr_add_default_logger(logger)
   CALL dbcsr_logger_release(logger)
   CALL dbcsr_error_handling_setup()
   CALL timings_register_hooks()
   CALL add_mp_perf_env()
   CALL add_timer_env()

   CALL timeset(routineN, handle)

   CALL dbcsr_init_lib()


   CALL dbcsr_print_config(io_unit)


    CALL dbcsr_make_random_block_sizes(sizes_m, m, bs) 
    CALL dbcsr_make_random_block_sizes(sizes_k, k, bs) 

    CALL dbcsr_dist_bin(row_dist, SIZE(sizes_m), npdims(1), &
                          sizes_m)
    CALL dbcsr_dist_bin(col_dist, SIZE(sizes_k), npdims(2), &
                          sizes_k)

    CALL dbcsr_distribution_new(dist, mp_env, row_dist, col_dist)

    CALL dbcsr_reset_randmat_seed()
 
!    print *, m,k,bs,sparsity
!    print *, sizes_m
    CALL dbcsr_make_random_matrix(matrix_a, sizes_m, sizes_k, "Matrix A", &
                                  sparsity, &
                                  group, data_type=mtype, dist=dist)

    ! create vectors
    CALL dbcsr_create_col_vec_from_matrix( input_vec , matrix_a , 1) 
    CALL dbcsr_copy(result_vec, input_vec)   
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

    CALL dbcsr_data_init(res_vec_init)
    CALL dbcsr_data_new(res_vec_init, dbcsr_type_1d_to_2d(mtype), data_size=v_r, data_size2=v_c)
    
     
    ! multiply mat x vec 
    CALL dbcsr_matrix_colvec_multiply(matrix_a, input_vec, result_vec, 1.0_dp, &
                                               0.0_dp, rep_row_vec, rep_col_vec)

                                           
    ! densify result vector                                       
    call dbcsr_to_dense_local(result_vec, res_vec_init)

    alpha_obj = compx_to_dbcsr_scalar((1.0_dp,0.0_dp), mtype)
    beta_obj = compx_to_dbcsr_scalar((0.0_dp,0.0_dp), mtype)

    ! check multiply
    CALL dbcsr_check_multiply("mat_vec_mult", result_vec, res_vec_init, data_a, inp_vec, res_vec, &
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

      CALL dbcsr_finalize_lib(mp_comm, io_unit)
  
      CALL timestop(handle)

   ! Print timers
   CALL timings_report_print(io_unit, 0.0_dp, .FALSE., cost_type_time, .TRUE., mp_env)
   ! Dump callgraph
   CALL timings_report_callgraph("test.callgraph")

   call dbcsr_mp_grid_remove(mp_env)
   !
   ! clean mp environment
   CALL dbcsr_mp_release(mp_env)
  

   ! Remove logger
   CALL dbcsr_rm_default_logger()

   !
   ! free comm
   CALL mp_comm_free(group)

   !
   ! Remove timers
   CALL rm_mp_perf_env()
   CALL rm_timer_env()
   !
   ! finalize mpi
   CALL mp_world_finalize()

      

END PROGRAM dbcsr_test_mat_vec_mult
