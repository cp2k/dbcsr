!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
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
PROGRAM tight_binding
   USE mpi
   USE dbcsr_api,                       ONLY: &
        dbcsr_create, dbcsr_distribution_get, dbcsr_distribution_new, dbcsr_distribution_release, &
        dbcsr_distribution_type, dbcsr_finalize, dbcsr_finalize_lib, dbcsr_get_stored_coordinates, &
        dbcsr_init_lib, dbcsr_multiply, dbcsr_nblkcols_total, dbcsr_nblkrows_total, dbcsr_print, &
        dbcsr_put_block, dbcsr_release, dbcsr_type, dbcsr_type_no_symmetry, dbcsr_type_real_8

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   implicit none

   type(dbcsr_type)                         :: matrix_a, matrix_b, matrix_c

   integer, dimension(:), pointer           :: col_blk_sizes, row_blk_sizes
   integer                                  :: group, numnodes, mynode, ierr, &
                                               nblkrows_total, nblkcols_total, &
                                               node_holds_blk, max_nze, nze, row, col, row_s, col_s, &
                                               io_unit, max_row_size, max_col_size
   integer, dimension(2)                    :: npdims, myploc
   integer, dimension(:), pointer           :: col_dist, row_dist
   type(dbcsr_distribution_type)            :: dist
   real(kind=kind(0.0d0)), dimension(:), allocatable :: values
   logical                                  :: tr
   logical, dimension(2)                    :: period = .true.

  real(8), allocatable :: rr(:,:), cell_vec(:,:) 
  real(8), allocatable :: rft(:,:)
  real(8), allocatable :: kpathc(:,:)
  integer :: nr, nrft, nkp, i, j, sys_dim, ntrans, cnt

!$ integer                                  :: provided_tsl

   !***************************************************************************************

   !
   ! initialize mpi
!$ if ( .false. ) then
      call mpi_init(ierr)
      if (ierr /= 0) stop "error in mpi_init"
!$ else
!$    call mpi_init_thread(mpi_thread_funneled, provided_tsl, ierr)
!$    if (ierr /= 0) stop "error in mpi_init_thread"
!$    if (provided_tsl .lt. mpi_thread_funneled) then
!$       stop "mpi library does not support the requested level of threading (mpi_thread_funneled)."
!$    endif
!$ endif

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


   !------- Read atom info ------------------
   open (unit=2, file="coords.dat")
   read (2, *) nr
   allocate(rr(3,nr))
   do j=1,nr
       read (2, *) rr(:,j) 
   enddo
   close(2)

   open (unit=2, file="super_cell_vec.dat")
   read (2, *) sys_dim
   allocate(cell_vec(sys_dim,sys_dim))
   do j=1,sys_dim
       read (2, *) cell_vec(:,j) 
   enddo
   close(2)

   
   !----- k points -------------
   ! temp 1 kpoint
   allocate(kpathc(2,1))   
   kpathc(:,1) = 0.0d0
   nkp = 1


   ! set cell translation, test value ntrans = 2
   ntrans = 2
   nrft = (2 * ntrans + 1)**2

   allocate(rft(sys_dim, nrft))
   
   cnt = 0
   do i = - ntrans, ntrans 
     do j = - ntrans, ntrans 
       cnt = cnt + 1
         rft(:, cnt) = i * cell_vec(:,1) + j * cell_vec(:,2)
     enddo
   enddo

   !***************************************************************************************
   !
   ! initialize libdbcsr
   CALL dbcsr_init_lib()

   !
   ! the matrix will contain nblkrows_total row blocks and nblkcols_total column blocks
   nblkrows_total = nr
   nblkcols_total = nr

   !
   ! set the block size for each row and column
   ! for first TB version will be 1
   ALLOCATE (row_blk_sizes(nblkrows_total), col_blk_sizes(nblkcols_total))
   row_blk_sizes(:) = 1
   col_blk_sizes(:) = 1

   !
   ! set the row and column distributions (here the distribution is set randomly)
   CALL random_dist(row_dist, nblkrows_total, npdims(1))
   CALL random_dist(col_dist, nblkcols_total, npdims(2))

   !
   ! set the dbcsr distribution object
   CALL dbcsr_distribution_new(dist, group=group, row_dist=row_dist, col_dist=col_dist, reuse_arrays=.TRUE.)

   !
   ! create the dbcsr matrices, i.e. a double precision non symmetric matrix
   ! with nblkrows_total x nblkcols_total blocks and
   ! sizes "sum(row_blk_sizes)" x "sum(col_blk_sizes)", distributed as
   ! specified by the dist object
   CALL dbcsr_create(matrix=matrix_a, &
                     name="this is my matrix a", &
                     dist=dist, &
                     matrix_type=dbcsr_type_no_symmetry, &
                     row_blk_size=row_blk_sizes, &
                     col_blk_size=col_blk_sizes, &
                     data_type=dbcsr_type_real_8)

   !CALL dbcsr_create(matrix=matrix_b, &
                     !name="this is my matrix b", &
                     !dist=dist, &
                     !matrix_type=dbcsr_type_no_symmetry, &
                     !row_blk_size=row_blk_sizes, &
                     !col_blk_size=col_blk_sizes, &
                     !data_type=dbcsr_type_real_8)

   !CALL dbcsr_create(matrix=matrix_c, &
                     !name="this is my matrix c", &
                     !dist=dist, &
                     !matrix_type=dbcsr_type_no_symmetry, &
                     !row_blk_size=row_blk_sizes, &
                     !col_blk_size=col_blk_sizes, &
                     !data_type=dbcsr_type_real_8)

   ! get the maximum block size of the matrix
   max_row_size = MAXVAL(row_blk_sizes)
   max_col_size = MAXVAL(col_blk_sizes)
   max_nze = max_row_size*max_col_size

   !
   ! set up the a matrix
   CALL dbcsr_distribution_get(dist, mynode=mynode)
   
   !ALLOCATE (values(max_nze))
   !DO row = 1, dbcsr_nblkrows_total(matrix_a)
      !DO col = MAX(row-1, 1), MIN(row+1, dbcsr_nblkcols_total(matrix_a))
         !tr = .FALSE.
         !row_s = row; col_s = col
         !CALL dbcsr_get_stored_coordinates(matrix_a, row_s, col_s, node_holds_blk)
         !IF (node_holds_blk .EQ. mynode) THEN
            !nze = row_blk_sizes(row_s)*col_blk_sizes(col_s)
            !CALL RANDOM_NUMBER(values(1:nze))
            !CALL dbcsr_put_block(matrix_a, row_s, col_s, values(1:nze))
         !ENDIF
      !ENDDO
   !ENDDO
   !DEALLOCATE (values)

   !
   ! set up the b matrix
   !CALL dbcsr_distribution_get(dist, mynode=mynode)
   !ALLOCATE (values(max_nze))
   !DO row = 1, dbcsr_nblkrows_total(matrix_b)
      !DO col = MAX(row-1, 1), MIN(row+1, dbcsr_nblkcols_total(matrix_b))
         !tr = .FALSE.
         !row_s = row; col_s = col
         !CALL dbcsr_get_stored_coordinates(matrix_b, row_s, col_s, node_holds_blk)
         !IF (node_holds_blk .EQ. mynode) THEN
            !nze = row_blk_sizes(row_s)*col_blk_sizes(col_s)
            !CALL RANDOM_NUMBER(values(1:nze))
            !CALL dbcsr_put_block(matrix_b, row_s, col_s, values(1:nze))
         !ENDIF
      !ENDDO
   !ENDDO
   !DEALLOCATE (values)

   !
   ! finalize the dbcsr matrices
   !CALL dbcsr_finalize(matrix_a)
   !CALL dbcsr_finalize(matrix_b)
   !CALL dbcsr_finalize(matrix_c)

   !
   ! multiply the matrices
   !CALL dbcsr_multiply('N', 'N', 1.0D0, matrix_a, matrix_b, 0.0D0, matrix_c)

   !
   ! print the matrices
   !CALL dbcsr_print(matrix_a)
   !CALL dbcsr_print(matrix_b)
   !CALL dbcsr_print(matrix_c)

   !
   ! release the matrices
   CALL dbcsr_release(matrix_a)
   !CALL dbcsr_release(matrix_b)
   !CALL dbcsr_release(matrix_c)

   CALL dbcsr_distribution_release(dist)
   !DEALLOCATE (row_blk_sizes, col_blk_sizes)

   ! finalize libdbcsr
   CALL dbcsr_finalize_lib(mpi_comm_world)

   ! free comm
   CALL mpi_comm_free(group, ierr)
   IF (ierr /= 0) STOP "Error in MPI_Comm_free"

   !
   ! finalize mpi
   CALL mpi_finalize(ierr)
   IF (ierr /= 0) STOP "Error in MPI_finalize"

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

real(8) function vpp_pi(d)
   real(8), intent(in) :: d
   real(8), parameter :: a0=1.42028166221d0
   real(8), parameter :: r0=0.45264d0
   real(8), parameter :: vpp_pi0=-2.7d0
   vpp_pi=vpp_pi0*exp(-abs(d-a0)/r0)
end function

real(8) function vpp_sig(d)
   real(8), intent(in) :: d
   real(8), parameter :: d0=3.34d0
   real(8), parameter :: r0=0.45264d0
   real(8), parameter :: vpp_sig0=0.48d0
   vpp_sig=vpp_sig0*exp(-abs(d-d0)/r0)
end function 

real(8) function t_of_d(dvec)
   real(8), intent(in) :: dvec(3)
   real(8), parameter :: eps=0.00001
   real(8) vpp_pi,vpp_sig
   real(8) t1
   t1=sqrt(dvec(1)**2+dvec(2)**2+dvec(3)**2)
   if (abs(dvec(3)).lt.eps) then
     t_of_d=-vpp_pi(t1)
   else
     t_of_d=-vpp_sig(t1)
   end if
end function




subroutine read_input(rr, rft, kpathc, nr, nrft, nkp)
  real(8), intent(inout) :: rr(:,:) 
  real(8), intent(inout) :: rft(:,:)
  real(8), intent(inout) :: kpathc(:,:)
  integer, intent(inout) :: nr, nrft, nkp

end subroutine



subroutine build_ham(ham, rr, rft, kc, nr, nrft)
  implicit none
  type(dbcsr_type), intent(inout)    :: ham
  integer, intent(in) :: nr,nrft
  real(8), intent(in) :: rr(3,nr)
  real(8), intent(in) :: rft(2,nrft)
  real(8), intent(in) :: kc(2)

  !double precision, intent(out) :: eig(nr)
  ! cutoff to assime tij=0
  real(8), parameter :: rcut=10.d0
  integer ii,jj,irft
  real(8) t1,z2
  real(8) r01(2),r02(2),r12(2)
  real(8) r1(3),r2(3),dvec(3)
  complex(8) tpk,tpr(nrft)
  real(8) t_of_d
  integer info,n,ld
  integer lwork
  real(8), allocatable :: rwork(:)
  complex(8), allocatable :: work(:)
  complex(8) :: matrix_element
  !complex(8), allocatable :: ham(:,:)
  
  !allocate(ham(nr,nr))
   ALLOCATE (values(max_nze))

  tpr=0.d0
  do irft=1,nrft
    ! FT phase
    t1=dot_product(kc,rft(:,irft))
    tpr(irft)=cmplx(cos(t1),sin(t1),kind=8)
  end do

  !ham=0.d0
  do ii=1,nr
    r01=rr(1:2,ii)
    r1=rr(:,ii)
    
    do jj=ii,nr
       r02=rr(1:2,jj)
       ! bloch phase missing in the tb model
       r12(:)=r02(:)-r01(:)
       t1=dot_product(kc,r12)
       tpk=cmplx(cos(t1),sin(t1),kind=8)
   
       matrix_element = (0.d0, 0.d0)
       ! loop over neighboring cells
       do irft=1,nrft
          r2(1)=r02(1)+rft(1,irft)
          r2(2)=r02(2)+rft(2,irft)
          z2=rr(3,jj)
          r2=(/r2(1),r2(2),z2/)
          dvec(:)=r2(:)-r1(:)
          t1=sqrt(dvec(1)**2+dvec(2)**2+dvec(3)**2)
          
          if (t1>rcut) cycle
 
          ! Do FT
          matrix_element = matrix_element - t_of_d(dvec) * tpr(irft) !*tpk
       end do

       if ( abs(matrix_element) > 1.0d-10) then
   !do row = 1, dbcsr_nblkrows_total(matrix_a)
      !do col = max(row-1, 1), min(row+1, dbcsr_nblkcols_total(matrix_a))
         !tr = .false.
         !row_s = row; col_s = col
         call dbcsr_get_stored_coordinates(ham, ii, jj, node_holds_blk)
         if (node_holds_blk .eq. mynode) then
            
            !nze = row_blk_sizes(ii)*col_blk_sizes(jj)
            values(1) = matrix_element
            call dbcsr_put_block(ham, ii, jj, values(1:nze))
         endif
      !enddo
   !enddo
          
       endif
    end do
  end do 


  !! use the LAPACK routine zheev
  !n=nr
  !ld=nr
  !lwork=2*n
  !allocate(rwork(3*n),work(lwork))
  !call zheev('V','U',n,ham,ld,eig,work,lwork,rwork,info)
  !if (info.ne.0) then
    !write(*,*)
    !write(*,'("Error: diagonalisation failed")')
    !write(*,'(" ZHEEV returned INFO = ",I8)') info
    !write(*,*)
    !stop
  !end if
  !deallocate(ham,rwork,work)
  
   DEALLOCATE (values)
  return
end subroutine


END PROGRAM 
