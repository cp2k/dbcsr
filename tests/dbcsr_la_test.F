
!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

PROGRAM dbcsr_la_test
   !! Tests for DBCSR operations

   USE dbcsr_acc_device, ONLY: dbcsr_acc_get_ndevices, &
                               dbcsr_acc_set_active_device
   USE dbcsr_kinds, ONLY: dp
   USE dbcsr_lib, ONLY: dbcsr_finalize_lib, &
                        dbcsr_init_lib, &
                        dbcsr_print_statistics
   USE dbcsr_machine, ONLY: default_output_unit
   USE dbcsr_mp_methods, ONLY: dbcsr_mp_new, &
                               dbcsr_mp_release,&
                               dbcsr_mp_grid_setup,&
                               dbcsr_mp_grid_remove
   USE dbcsr_mpiwrap, ONLY: mp_cart_create, &
                            mp_cart_rank, &
                            mp_comm_free, &
                            mp_environ, &
                            mp_world_finalize, &
                            mp_world_init, mp_sum


   USE dbcsr_test_methods, ONLY: compx_to_dbcsr_scalar, &
                                 dbcsr_impose_sparsity, &
                                 dbcsr_make_random_block_sizes, &
                                 dbcsr_random_dist, &
                                 dbcsr_reset_randmat_seed

   USE dbcsr_types, ONLY: &
      dbcsr_conjugate_transpose, dbcsr_data_obj, dbcsr_mp_obj, &
      dbcsr_no_transpose, dbcsr_scalar_type, dbcsr_transpose, &
      dbcsr_type_antisymmetric, dbcsr_type_complex_4, dbcsr_type_complex_4_2d, &
      dbcsr_type_complex_8, dbcsr_type_complex_8_2d, dbcsr_type_no_symmetry, dbcsr_type_real_4, &
      dbcsr_type_real_4_2d, dbcsr_type_real_8, dbcsr_type_real_8_2d, dbcsr_type_symmetric,&
      dbcsr_iterator





   USE dbcsr_methods, ONLY: &
      dbcsr_col_block_offsets, dbcsr_col_block_sizes, dbcsr_get_data_type, &
      dbcsr_get_matrix_type, dbcsr_name, dbcsr_nblkcols_total, dbcsr_nblkrows_total, &
      dbcsr_row_block_offsets, &
      dbcsr_row_block_sizes,dbcsr_mp_grid_remove
   
  USE dbcsr_data_methods, ONLY: dbcsr_data_get_sizes, &
                                 dbcsr_data_init, &
                                 dbcsr_data_new, &
                                 dbcsr_data_release, &
                                 dbcsr_scalar_negative, &
                                 dbcsr_scalar_one, &
                                 dbcsr_type_1d_to_2d
   USE dbcsr_dist_operations, ONLY: dbcsr_dist_bin

   use dbcsr_api, only: dbcsr_type, dbcsr_distribution_type, dbcsr_make_random_matrix, dbcsr_to_dense_local, &
                        dbcsr_desymmetrize, dbcsr_nfullrows_total, dbcsr_nfullcols_total, dbcsr_release,&
                        dbcsr_create, dbcsr_distribution_new, dbcsr_distribution_release

  use dbcsr_arnoldi_api 
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   INTEGER                                  :: mp_comm, io_unit, group, numnodes, mynode, &
                                               handle 
   INTEGER, DIMENSION(2)                    :: npdims, myploc

   CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_unittest'

   ! test parameters
   TYPE(dbcsr_type)                    :: matrix, matrix_symm
   INTEGER, DIMENSION(:), POINTER :: col_dist, row_dist
   TYPE(dbcsr_distribution_type)                       :: dist
   TYPE(dbcsr_data_obj)                               :: matrix_data
   INTEGER, DIMENSION(:), POINTER                     ::  sizes_m, sizes_n


   integer                             ::  m,n, bs_m, mtype, a_r, a_c, matrix_size
   INTEGER, DIMENSION(2)               :: bs
   real(8) :: sparsity

   ! arnoldi
   integer :: max_iter
   logical :: converged
   real(8) :: threshold, max_ev, min_ev

   ! lapack
   real(8), allocatable ::eigval(:),  work(:) , iwork(:)  
   real(8) :: zarr(1)
   integer lwork
   integer info, ifail, ld, num_ev

   !***************************************************************************************



   ! initialize mpi
   CALL mp_world_init(mp_comm)

   !! setup the mp environment
   npdims(:) = 0
   CALL mp_cart_create(mp_comm, 2, npdims, myploc, group)
   CALL mp_environ(numnodes, mynode, group)
   
   ! set standard output parameters
   io_unit = 0
   IF (mynode .EQ. 0) io_unit = default_output_unit

   !
   ! initialize libdbcsr
   IF (dbcsr_acc_get_ndevices() > 0) &
      CALL dbcsr_acc_set_active_device(MOD(mynode, dbcsr_acc_get_ndevices()))

   CALL dbcsr_init_lib(mp_comm, io_unit)

   ! initialize libdbcsr errors
   CALL timeset(routineN, handle)

   CALL dbcsr_reset_randmat_seed()
   

   !**********************************************************

   !params

   bs_m = 10
   matrix_size = 1000
   mtype = dbcsr_type_real_8
   sparsity = 0.1

   bs(1) = 1
   m = matrix_size
   n = m
   bs(2) = bs_m

   !setup arrays, matrices

   CALL dbcsr_make_random_block_sizes(sizes_m, m, bs)
   allocate(sizes_n(size(sizes_m)))
   sizes_n(:) = sizes_m(:)


    CALL dbcsr_dist_bin(row_dist, SIZE(sizes_m), npdims(1), &
                        sizes_m)
    CALL dbcsr_dist_bin(col_dist, SIZE(sizes_n), npdims(2), &
                        sizes_n)

    CALL dbcsr_distribution_new(dist, group=group, row_dist=row_dist, col_dist=col_dist, reuse_arrays=.TRUE.)
 
    CALL dbcsr_make_random_matrix(matrix_symm, sizes_m, sizes_n, "symm", &
                                  sparsity, &
                                  group, data_type=mtype, symmetry=dbcsr_type_symmetric, dist=dist)


    CALL dbcsr_create(matrix, "unsymm", &
                      dist, dbcsr_type_no_symmetry , &
                      sizes_n, &
                      sizes_m, &
                      data_type=dbcsr_type_real_8)
    
    call dbcsr_desymmetrize(matrix_symm, matrix)


   ! setup arnoldi
   threshold=1.0E-8_dp
   max_iter=20

   call arnoldi_extremal(matrix, max_ev, min_ev, converged, threshold, max_iter)

   if (converged .eqv. .false.) then
       error stop "arnoldi is not converged"
   endif
   
   ! check arnoldi
   a_r = dbcsr_nfullrows_total(matrix)
   a_c = dbcsr_nfullcols_total(matrix)
   CALL dbcsr_data_init(matrix_data)
   CALL dbcsr_data_new(matrix_data, dbcsr_type_1d_to_2d(mtype), data_size=a_r, data_size2=a_c)
   call dbcsr_to_dense_local(matrix, matrix_data)

   call dbcsr_release(matrix)
   call dbcsr_release(matrix_symm)

   call dbcsr_distribution_release(dist)
   ! collect to one array
   call mp_sum(matrix_data % d % r2_dp, mp_comm)

   ! set parameters for lapack eigensolver 
    info=0
    ld = matrix_size
    ! 5*n according to lapack manual
    lwork = 8 * matrix_size
    allocate (eigval(1),work(lwork),iwork(5*matrix_size))


    call dsyevx('N','I', 'U', matrix_size, matrix_data % d % r2_dp, ld, 0, 0, 1, 1, threshold, &
               num_ev, eigval, zarr, 1, work, lwork, iwork, ifail, info)


    call dbcsr_data_release(matrix_data)

    if (info.ne.0) then
    write(*,*)
    write(*,'("Error: diagonalisation failed")')
    write(*,'(" ZHEEV returned INFO = ",I8)') info
    write(*,*)
    error stop
    end if

   
   if ( abs(min_ev-eigval(1)) > threshold ) then
      if (io_unit /= 0) then
         WRITE (io_unit, *) " -- TESTING DBCSR_LA arnoldi_extremal ... FAILED"
         WRITE (io_unit, *) "Arnoldi eigenvalue: ", min_ev
         WRITE (io_unit, *) "LAPACK eigenvalue: ", eigval(1)
      endif
      DBCSR_ABORT('Test failed')
   else
       if (io_unit /= 0 ) WRITE (io_unit, *) " -- TESTING DBCSR_LA arnoldi_extremal ... PASSED"
   endif 

   deallocate(eigval, work, iwork)

   !*********************************************************

   CALL timestop(handle)
   !
   !
   ! finalize mpi
   CALL mp_comm_free(group)


   !! finalize libdbcsr
   CALL dbcsr_finalize_lib()
   
   CALL mp_world_finalize()

END PROGRAM dbcsr_la_test
