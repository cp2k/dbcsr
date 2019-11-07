!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

PROGRAM dbcsr_example_1
   !! DBCSR example 1
   !! This example shows how to create a dbcsr matrix

   USE mpi
   USE dbcsr_api,                       ONLY: &
        dbcsr_create, dbcsr_distribution_new, dbcsr_distribution_release, dbcsr_distribution_type, &
        dbcsr_finalize, dbcsr_finalize_lib, dbcsr_init_lib, dbcsr_print, dbcsr_release, &
        dbcsr_type, dbcsr_type_no_symmetry, dbcsr_type_real_8

   IMPLICIT NONE

   TYPE(dbcsr_type)                         :: matrix_a

   INTEGER, DIMENSION(:), POINTER           :: col_blk_sizes, row_blk_sizes
   INTEGER                                  :: group, numnodes, mynode, nblkrows_total, &
                                               nblkcols_total, ierr
   INTEGER, DIMENSION(2)                    :: npdims
   INTEGER, DIMENSION(:), POINTER           :: col_dist, row_dist
   TYPE(dbcsr_distribution_type)            :: dist
   LOGICAL, DIMENSION(2)                    :: period = .TRUE.
!$ INTEGER                                  :: provided_tsl

   !***************************************************************************************

   !
   ! initialize mpi
!$ IF ( .FALSE. ) THEN
      CALL mpi_init(ierr)
      IF (ierr /= 0) STOP "Error in MPI_Init"
!$ ELSE
!$    CALL mpi_init_thread(MPI_THREAD_FUNNELED, provided_tsl, ierr)
!$    IF (ierr /= 0) STOP "Error in MPI_Init_thread"
!$    IF (provided_tsl .LT. MPI_THREAD_FUNNELED) THEN
!$       STOP "MPI library does not support the requested level of threading (MPI_THREAD_FUNNELED)."
!$    ENDIF
!$ ENDIF

   !
   ! setup the mp environment
   CALL mpi_comm_size(MPI_COMM_WORLD, numnodes, ierr)
   IF (ierr /= 0) STOP "Error in MPI_Comm_size"
   npdims(:) = 0
   CALL mpi_dims_create(numnodes, 2, npdims, ierr)
   IF (ierr /= 0) STOP "Error in MPI_Dims_create"
   CALL mpi_cart_create(MPI_COMM_WORLD, 2, npdims, period, .FALSE., group, ierr)
   IF (ierr /= 0) STOP "Error in MPI_Cart_create"

   CALL mpi_comm_rank(MPI_COMM_WORLD, mynode, ierr)
   IF (ierr /= 0) STOP "Error in MPI_Comm_rank"
   WRITE (*, *) 'mynode ', mynode, ' numnodes', numnodes

   !***************************************************************************************
   !
   ! initialize libdbcsr
   CALL dbcsr_init_lib(MPI_COMM_WORLD)

   !
   ! the matrix will contain nblkrows_total row blocks and nblkcols_total column blocks
   nblkrows_total = 4
   nblkcols_total = 3

   !
   ! set the block size for each row and column
   ALLOCATE (row_blk_sizes(nblkrows_total), col_blk_sizes(nblkcols_total))
   row_blk_sizes(:) = 2
   col_blk_sizes(:) = 3

   !
   ! set the row and column distributions (here the distribution is set randomly)
   CALL random_dist(row_dist, nblkrows_total, npdims(1))
   CALL random_dist(col_dist, nblkcols_total, npdims(2))

   !
   ! set the dbcsr distribution object
   CALL dbcsr_distribution_new(dist, group=group, row_dist=row_dist, col_dist=col_dist, reuse_arrays=.TRUE.)

   !
   ! create the dbcsr matrix, i.e. a double precision non symmetric matrix
   ! with nblkrows_total x nblkcols_total blocks and
   ! sizes "sum(row_blk_sizes)" x "sum(col_blk_sizes)", distributed as
   ! specified by the dist object
   CALL dbcsr_create(matrix=matrix_a, &
                     name="this is my matrix a", &
                     dist=dist, &
                     matrix_type=dbcsr_type_no_symmetry, &
                     row_blk_size=row_blk_sizes, &
                     col_blk_size=col_blk_sizes, &
                     data_type=dbcsr_type_real_8, &
                     reuse_arrays=.TRUE.)

   !
   ! finalize the dbcsr matrix
   CALL dbcsr_finalize(matrix_a)

   !
   ! print the *empty* matrix
   CALL dbcsr_print(matrix_a)

   !
   ! release the matrix
   CALL dbcsr_release(matrix_a)

   !
   ! release the distribution
   CALL dbcsr_distribution_release(dist)

   !***************************************************************************************
   !

   ! free comm
   CALL mpi_comm_free(group, ierr)
   IF (ierr /= 0) STOP "Error in MPI_Comm_free"

   ! finalize libdbcsr
   CALL dbcsr_finalize_lib()
   !
   ! finalize mpi
   CALL mpi_finalize(ierr)
   IF (ierr /= 0) STOP "Error in MPI_finalize"

   !***************************************************************************************

CONTAINS

   SUBROUTINE random_dist(dist_array, dist_size, nbins)
      INTEGER, DIMENSION(:), INTENT(out), POINTER        :: dist_array
      INTEGER, INTENT(in)                                :: dist_size, nbins

      INTEGER                                            :: i

      ALLOCATE (dist_array(dist_size))
      DO i = 1, dist_size
         dist_array(i) = MODULO(nbins-i, nbins)
      END DO

   END SUBROUTINE random_dist

END PROGRAM dbcsr_example_1
