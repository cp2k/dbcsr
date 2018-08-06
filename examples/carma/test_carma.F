!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright (C) 2000 - 2018  CP2K developers group                                               !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief   DBCSR example 3
!>          This example shows how to multiply two dbcsr matrices
!> \author  V. Weber
!> \date    2010
!> \version 1.0
!>
!> <b>Modification history:</b>
!> - Created 2010
! **************************************************************************************************
PROGRAM dbcsr_example_3
 USE dbcsr_machine,                         ONLY: m_walltime,&
                                                  default_output_unit
   USE dbcsr_api,                       ONLY: &
        dbcsr_create, dbcsr_distribution_get, dbcsr_distribution_new, dbcsr_distribution_release, &
        dbcsr_distribution_type, dbcsr_finalize, dbcsr_finalize_lib, dbcsr_get_stored_coordinates, &
        dbcsr_init_lib, dbcsr_multiply, dbcsr_nblkcols_total, dbcsr_nblkrows_total, dbcsr_print, &
        dbcsr_put_block, dbcsr_release, dbcsr_type_loc=>dbcsr_type, dbcsr_type_no_symmetry, &
        dbcsr_checksum, dbcsr_type_real_8
   USE dbcsr_kinds,                           ONLY: real_8
   USE dbcsr_machine,                         ONLY: default_output_unit
   USE dbcsr_message_passing,                 ONLY: mp_cart_create,&
                                              mp_comm_free,&
                                              mp_environ,&
                                              mp_world_finalize,&
                                              mp_world_init, mp_sync, mp_max,&
                                              add_mp_perf_env,&
                                              rm_mp_perf_env
   USE dbcsr_error_handling,               ONLY: dbcsr_error_handling_setup
   USE dbcsr_string_table,                    ONLY: string_table_allocate,&
                                              string_table_deallocate
   USE dbcsr_timings,                         ONLY: add_timer_env,&
                                              timings_register_hooks,&
                                              rm_timer_env
   USE dbcsr_timings_report,                  ONLY: cost_type_time,&
                                              timings_report_print
   USE dbcsr_para_types,                   ONLY: dbcsr_para_env_type
   USE dbcsr_para_env,                     ONLY: dbcsr_para_env_create
   USE dbcsr_kinds,                           ONLY: dp

   use dbcsr_mm_carma, only: carma_mm
   
    use dbcsr_util
!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE

   TYPE(dbcsr_type_loc)                          :: matrix_a, matrix_b, matrix_c, matrix_c_cannon

   INTEGER, DIMENSION(:), POINTER           :: blk_sizes_m, blk_sizes_n, blk_sizes_k
   
   INTEGER                                  :: mp_comm, group, numnodes, mynode, &
                                               nblk_total_m, nblk_total_n, nblk_total_k, &

                                               node_holds_blk, max_nze, nze, row, col, row_s, col_s, &
                                               io_unit, max_row_size, max_col_size
   
   INTEGER, DIMENSION(2)                    :: npdims, myploc
   INTEGER, DIMENSION(:), POINTER           :: dist_m, dist_n, dist_k, dist_m_cannon
   TYPE(dbcsr_distribution_type)             ::  dist_a, dist_b, dist_c, dist_c_cannon
   REAL(real_8), DIMENSION(:), ALLOCATABLE  :: values
   LOGICAL                                  :: tr

  character(len=32) :: cdimx, cdimy, run_order_char, rnd_thr_char, niter_char
  integer :: m,n,k,nb, run_order, niter
   real(8)                              :: t1, t2, total_time_dbcsr, max_times(2)
   TYPE(dbcsr_para_env_type), POINTER          :: para_env
   
   real                                 :: rnd, rnd_thr
   !***************************************************************************************

   !
   ! initialize mpi
   CALL mp_world_init(mp_comm)

   NULLIFY(para_env)
   CALL dbcsr_para_env_create(para_env, group=mp_comm, owns_group=.FALSE.)
   CALL dbcsr_error_handling_setup()
   CALL timings_register_hooks()
   CALL string_table_allocate()
   CALL add_mp_perf_env()
   CALL add_timer_env()

   !
   ! setup the mp enviroment
   npdims(:) = 0

  ! sparsity level 1.0 = dense, 0.0 = empty
  rnd_thr=0.3

  niter=10

  if (command_argument_count() >=2) then
    call get_command_argument(1, cdimx)
    call get_command_argument(2, cdimy)
    call get_command_argument(3, run_order_char)
    call get_command_argument(4, rnd_thr_char)
    call get_command_argument(5, niter_char)

    read(cdimx,*) npdims(1)
    read(cdimy,*) npdims(2)
    read(run_order_char,*) run_order
    read(rnd_thr_char,*) rnd_thr
    read(niter_char,*) niter

  endif

   CALL mp_cart_create(mp_comm, 2, npdims, myploc, group)
   CALL mp_environ(numnodes, mynode, group)
   WRITE (*, *) 'mynode ', mynode, ' numnodes', numnodes

  
  open(unit=123,file="mnkb.dat",status="old",form="formatted")
  read(123,*) m
  read(123,*) n
  read(123,*) k
  read(123,*) nb
  close(123)
  !
  !if (((n/nb) < prow) .or. ((n/nb) < pcol)) then
  !   print *, "Problem size too small for processor set!"
  !   stop 100
  !endif
   !***************************************************************************************
   !
   ! initialize libdbcsr
   CALL dbcsr_init_lib()

   !
   ! set standard output parameters
   io_unit = 0
   IF (mynode .EQ. 0) io_unit = default_output_unit

   !
   ! the matrix will contain nblkrows_total row blocks and nblkcols_total column blocks
   nblk_total_m = m / nb
   nblk_total_n = n / nb
   nblk_total_k = k / nb

   !
   ! set the block size for each row and column
   ALLOCATE (blk_sizes_m(nblk_total_m), blk_sizes_n(nblk_total_n), blk_sizes_k(nblk_total_k))
   blk_sizes_m(:) = nb
   blk_sizes_n(:) = nb
   blk_sizes_k(:) = nb

   !
   ! set the row and column distributions (here the distribution is set randomly)
   CALL random_dist(dist_k, nblk_total_k, npdims(1))
   CALL random_dist(dist_m, nblk_total_m, npdims(2))
   CALL random_dist(dist_n, nblk_total_n, npdims(2))
   
   CALL random_dist(dist_m_cannon, nblk_total_m, npdims(1))
   
   !
   ! set the dbcsr distribution object
   CALL dbcsr_distribution_new(dist_a, group=group, row_dist=dist_k, col_dist=dist_m, reuse_arrays=.false.)
   CALL dbcsr_distribution_new(dist_b, group=group, row_dist=dist_k, col_dist=dist_n, reuse_arrays=.false.)
   CALL dbcsr_distribution_new(dist_c, group=group, row_dist=dist_m_cannon, col_dist=dist_n, reuse_arrays=.false.)
   
   CALL dbcsr_distribution_new(dist_c_cannon, group=group, row_dist=dist_m_cannon, col_dist=dist_n, &
                               reuse_arrays=.false.)
   !
   ! create the dbcsr matrices, i.e. a double precision non symmetric matrix
   ! with nblk_total_m x nblkcols_total blocks and
   ! sizes "sum(blk_sizes_k)" x "sum(col_blk_sizes)", distributed as
   ! specified by the dist object
   CALL dbcsr_create(matrix=matrix_a, &
                     name="this is my matrix a", &
                     dist=dist_a, &
                     matrix_type=dbcsr_type_no_symmetry, &
                     row_blk_size=blk_sizes_k, &
                     col_blk_size=blk_sizes_m, &
                     data_type=dbcsr_type_real_8)

   CALL dbcsr_create(matrix=matrix_b, &
                     name="this is my matrix b", &
                     dist=dist_b, &
                     matrix_type=dbcsr_type_no_symmetry, &
                     row_blk_size=blk_sizes_k, &
                     col_blk_size=blk_sizes_n, &
                     data_type=dbcsr_type_real_8)

   CALL dbcsr_create(matrix=matrix_c, &
                     name="this is my matrix c", &
                     dist=dist_c, &
                     matrix_type=dbcsr_type_no_symmetry, &
                     row_blk_size=blk_sizes_m, &
                     col_blk_size=blk_sizes_n, &
                     data_type=dbcsr_type_real_8)


   CALL dbcsr_create(matrix=matrix_c_cannon, &
                     name="this is my matrix c", &
                     dist=dist_c_cannon, &
                     matrix_type=dbcsr_type_no_symmetry, &
                     row_blk_size=blk_sizes_m, &
                     col_blk_size=blk_sizes_n, &
                     data_type=dbcsr_type_real_8)
                     


   !
   ! set up the a matrix
   CALL dbcsr_distribution_get(dist_a, mynode=mynode)
   ALLOCATE (values( MAXVAL(blk_sizes_k) * MAXVAL(blk_sizes_m) ))
   DO row = 1, dbcsr_nblkrows_total(matrix_a)
      DO col = 1, dbcsr_nblkcols_total(matrix_a)
         call random_number(rnd)
         if (rnd <= rnd_thr) then
             tr = .FALSE.
             row_s = row; col_s = col
             CALL dbcsr_get_stored_coordinates(matrix_a, row_s, col_s, node_holds_blk)
             IF (node_holds_blk .EQ. mynode) THEN
                nze = blk_sizes_k(row_s)*blk_sizes_m(col_s)
                CALL RANDOM_NUMBER(values(1:nze))
                CALL dbcsr_put_block(matrix_a, row_s, col_s, values(1:nze))
             ENDIF
         endif
      ENDDO
   ENDDO
   DEALLOCATE (values)

   !
   ! set up the b matrix
   CALL dbcsr_distribution_get(dist_b, mynode=mynode)
   ALLOCATE (values( MAXVAL(blk_sizes_k) * MAXVAL(blk_sizes_n) ))
   DO row = 1, dbcsr_nblkrows_total(matrix_b)
      DO col = 1, dbcsr_nblkcols_total(matrix_b)
         call random_number(rnd)
         if (rnd <= rnd_thr) then
             tr = .FALSE.
             row_s = row; col_s = col
             CALL dbcsr_get_stored_coordinates(matrix_b, row_s, col_s, node_holds_blk)
             IF (node_holds_blk .EQ. mynode) THEN
                nze = blk_sizes_k(row_s)*blk_sizes_n(col_s)
                CALL RANDOM_NUMBER(values(1:nze))
                CALL dbcsr_put_block(matrix_b, row_s, col_s, values(1:nze))
             ENDIF
         endif
      ENDDO
   ENDDO
   DEALLOCATE (values)

   !
   ! finalize the dbcsr matrices
   CALL dbcsr_finalize(matrix_a)
   CALL dbcsr_finalize(matrix_b)
   CALL dbcsr_finalize(matrix_c)
   CALL dbcsr_finalize(matrix_c_cannon)

   !

   print *,"checksum matrix_a matrix_b", dbcsr_checksum(matrix_a), dbcsr_checksum(matrix_b)
    
   max_times(:) = 0
   if (run_order == 0) then
       
       do row = 1, niter
           CALL mp_sync(group)
           t1 = m_walltime()
           
           ! multiply the matrices
           CALL dbcsr_multiply('T', 'N', 1.0_real_8, matrix_a, matrix_b, 0.0_real_8, matrix_c_cannon)
           
           CALL mp_sync(group)
           
           max_times(2) = max_times(2) +  m_walltime()-t1
       enddo

       !print *,"+++++ cannon time :", m_walltime()-t1

       !print *,"cannon result: nze", matrix_c_cannon % prv % nze
       
       !CALL dbcsr_print(matrix_c_cannon)
       
       print *,"checksum cannon", dbcsr_checksum(matrix_c_cannon)
   
   else

       do row = 1, niter    
           CALL mp_sync(group)
           t1 = m_walltime()
           
           call carma_mm(matrix_a%prv, matrix_b%prv, matrix_c%prv)
           
           CALL mp_sync(group)
           
           max_times(1) = max_times(1) + m_walltime()-t1
       enddo

       !CALL dbcsr_multiply('T', 'N', 1.0_real_8, matrix_a, matrix_b, 0.0_real_8, matrix_c_cannon)
       print *,"checksum carma", dbcsr_checksum(matrix_c)!, dbcsr_checksum(matrix_c_cannon)
       !print *,"+++++ carma time :", m_walltime()-t1
       
  endif 
  
   call mp_max(max_times, group)
   
   if (mynode == 0) print *,"!!!!!!!!!!!!!!!!!!!! max times: carma, cannon: ", max_times/niter
   
       !call dbcsr_print(matrix_c)
   
   !
   ! print the matrices
   !CALL dbcsr_print(matrix_a)
   !CALL dbcsr_print(matrix_b)
   !CALL dbcsr_print(matrix_c)
   

   !
   ! release the matrices
   CALL dbcsr_release(matrix_a)
   CALL dbcsr_release(matrix_b)
   CALL dbcsr_release(matrix_c)
   CALL dbcsr_release(matrix_c_cannon)
   
   CALL dbcsr_distribution_release(dist_a)
   CALL dbcsr_distribution_release(dist_b)
   CALL dbcsr_distribution_release(dist_c)
   CALL dbcsr_distribution_release(dist_c_cannon)
   
   DEALLOCATE (blk_sizes_m, blk_sizes_n, blk_sizes_k, dist_m, dist_m_cannon, dist_n, dist_k)

   !***************************************************************************************

   !
   ! finalize libdbcsr
   CALL dbcsr_finalize_lib(mp_comm, io_unit)

   !
   ! finalize mpi
   CALL mp_comm_free(group)
   
      ! Print timers
   CALL timings_report_print(io_unit, 0.0_dp, .FALSE., cost_type_time, .TRUE., para_env)

   ! Remove timers
   CALL rm_timer_env()
   CALL rm_mp_perf_env()
   CALL string_table_deallocate(0)
   
   CALL mp_world_finalize()

   !***************************************************************************************

CONTAINS

! **************************************************************************************************
!> \brief ...
!> \param dist_array ...
!> \param dist_size ...
!> \param nbins ...
! **************************************************************************************************
   SUBROUTINE random_dist(dist_array, dist_size, nbins)
      INTEGER, DIMENSION(:), INTENT(out), POINTER        :: dist_array
      INTEGER, INTENT(in)                                :: dist_size, nbins

      INTEGER                                            :: i

      ALLOCATE (dist_array(dist_size))
      DO i = 1, dist_size
         dist_array(i) = MODULO(nbins-i, nbins)
      END DO

   END SUBROUTINE random_dist

END PROGRAM dbcsr_example_3
