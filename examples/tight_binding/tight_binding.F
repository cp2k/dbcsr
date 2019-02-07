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
        dbcsr_put_block, dbcsr_release, dbcsr_type, dbcsr_type_no_symmetry, &
        dbcsr_type_real_8, dbcsr_type_complex_8, dbcsr_add_on_diag, dbcsr_trace
   
    use dm_ls_scf_methods, only: density_matrix_sign
!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   implicit none

   type(dbcsr_type)                         ::  matrix_p, matrix_ks, matrix_s, matrix_s_inv, rho_ham

   integer, dimension(:), pointer           :: col_blk_sizes, row_blk_sizes
   integer                                  :: group, numnodes, mynode, ierr, &
                                               nblkrows_total, nblkcols_total, &
                                               node_holds_blk, max_nze, nze, row, col, row_s, col_s, &
                                               io_unit, max_row_size, max_col_size
   integer, dimension(2)                    :: npdims, myploc
   integer, dimension(:), pointer           :: col_dist, row_dist
   type(dbcsr_distribution_type)            :: dist
   complex(8), dimension(:), allocatable    :: values
   logical                                  :: tr
   logical, dimension(2)                    :: period = .true.

  real(8), allocatable :: rr(:,:), cell_vec(:,:) 
  real(8), allocatable :: rft(:,:)
  real(8), allocatable :: kpathc(:,:)
  real(8) :: kc(2), mu, dbcsr_energy
  integer :: nr, nrft, nkp, i, j, sys_dim, ntrans, cnt

  ! parameters for TB parameters
  real(8) :: energies(4), vsss(4), vsps(4), vpps(4), vppp(4)
  real(8), allocatable :: ham_dense(:,:)
  character(len=20) :: exFmt
  integer :: block_size, nelectrons
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

   block_size = 4
      
   ! occupation is 4 spins per atoms, taking spin into account will reveal 2 per atom
   nelectrons = nr * 2 
      
   print *,"Number of atoms ", nr
   print *,"Number of orbitals (block size) ", block_size
   print *,"Occupation (number of electrons / 2)", nelectrons
   print *,"Problem dimension ", nr * block_size
   
   if (nr < 100) then
      print *,"Diagonalize dense hamiltonian for reference"
   !    open (unit=2, file="super_cell_vec.dat")
   !    read (2, *) sys_dim
   !    allocate(cell_vec(sys_dim,sys_dim))
   !    do j=1,sys_dim
   !        read (2, *) cell_vec(:,j) 
   !    enddo
   !    close(2)


      !--- energies for TB ----------
      ! s
      energies(1) = 0.1027
      ! p
      energies(2) = 0.5426
      
      vsss = (/219.56,       -16.2132459618,  -15.5048968097, 1.26439940008/)
      vsps = (/10.127687621, -4.4036811240, 0.2266767834, 0.92267194054/)
      vpps = (/ -22.959028107, 1.7207707741, 1.4191307713, 1.03136916513/)
      vppp = (/10.265449263, 4.6718241428, -2.2161562721, 1.11134828469/)

      allocate(ham_dense(nr*block_size, nr*block_size))
      
      call build_ham_blocked_dense(ham_dense, rr, nr, block_size)
      
   !    write(exFmt,'("(",I0,"(F8.4))")') nr*block_size
   !    do i=1,nr*block_size
   !      write(*,exFmt) (ham_dense(i,j), j=1,nr*block_size)
   !    end do
      
      call diag_dense(ham_dense, nelectrons, nr*block_size)
      
   
      !----- k points -------------
      ! temp 1 kpoint
   !    allocate(kpathc(2,1))   
   !    kpathc(:,1) = 0.0d0
   !    kc(:) = 0.0d0
   !    nkp = 1
   !    
   ! 
   !    ! set cell translation, 0 - no translation
   !    ntrans = 0  
   !    nrft = (2 * ntrans + 1)**2
   ! 
   !    allocate(rft(sys_dim, nrft))
   !    
   !    cnt = 0
   !    do i = - ntrans, ntrans 
   !      do j = - ntrans, ntrans 
   !        cnt = cnt + 1
   !          rft(:, cnt) = i * cell_vec(:,1) + j * cell_vec(:,2)
   !      enddo
   !    enddo
   
   endif
   
   print *,"DBCSR case"
   
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
   row_blk_sizes(:) = block_size
   col_blk_sizes(:) = block_size

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
   CALL dbcsr_create(matrix=matrix_ks, &
                     name="ks", &
                     dist=dist, &
                     matrix_type=dbcsr_type_no_symmetry, &
                     row_blk_size=row_blk_sizes, &
                     col_blk_size=col_blk_sizes, &
                     data_type=dbcsr_type_real_8)

   CALL dbcsr_create(matrix=matrix_p, &
                     name="p", &
                     dist=dist, &
                     matrix_type=dbcsr_type_no_symmetry, &
                     row_blk_size=row_blk_sizes, &
                     col_blk_size=col_blk_sizes, &
                     data_type=dbcsr_type_real_8)

   CALL dbcsr_create(matrix=matrix_s, &
                     name="s", &
                     dist=dist, &
                     matrix_type=dbcsr_type_no_symmetry, &
                     row_blk_size=row_blk_sizes, &
                     col_blk_size=col_blk_sizes, &
                     data_type=dbcsr_type_real_8)

   CALL dbcsr_create(matrix=matrix_s_inv, &
                     name="s_inv", &
                     dist=dist, &
                     matrix_type=dbcsr_type_no_symmetry, &
                     row_blk_size=row_blk_sizes, &
                     col_blk_size=col_blk_sizes, &
                     data_type=dbcsr_type_real_8)
                     
   CALL dbcsr_create(matrix=rho_ham, &
                  name="rho*ham", &
                  dist=dist, &
                  matrix_type=dbcsr_type_no_symmetry, &
                  row_blk_size=row_blk_sizes, &
                  col_blk_size=col_blk_sizes, &
                  data_type=dbcsr_type_real_8)
   
   ! get the maximum block size of the matrix
   max_row_size = MAXVAL(row_blk_sizes)
   max_col_size = MAXVAL(col_blk_sizes)
   max_nze = max_row_size*max_col_size

   
   !
   ! set up the a matrix
   CALL dbcsr_distribution_get(dist, mynode=mynode)

   !call build_ham(matrix_ks, rr, rft, kc, nr, nrft)

      
   ! DBCSR version, sparse and blocked   
   call build_ham_blocked(matrix_ks, rr, nr)
   
   CALL dbcsr_finalize(matrix_ks)
   CALL dbcsr_finalize(matrix_p)

   call dbcsr_add_on_diag(matrix_s, 1.0d0)
   call dbcsr_add_on_diag(matrix_s_inv, 1.0d0)
   
   
   CALL dbcsr_finalize(matrix_s)
   CALL dbcsr_finalize(matrix_s_inv)

  
!    call dbcsr_print(matrix_ks) 

   ! TODO fix
   mu = 0.0d0
   call density_matrix_sign(matrix_p, mu, .false., matrix_ks, matrix_s, &
       matrix_s_inv, nelectrons, 0.0001d0)

!    call dbcsr_print(matrix_p)


   CALL dbcsr_multiply('N', 'N', 1.0D0, matrix_ks, matrix_p, 0.0D0, rho_ham)
   
   
   call dbcsr_trace(rho_ham, dbcsr_energy)
   
     print *,"DBCSR Energy: ", dbcsr_energy
   
   
   ! Release all
   deallocate(rr)

   !
   ! release the matrices
   CALL dbcsr_release(matrix_ks)
   CALL dbcsr_release(matrix_p)
   CALL dbcsr_release(matrix_s)
   CALL dbcsr_release(matrix_s_inv)
   CALL dbcsr_release(rho_ham)

   CALL dbcsr_distribution_release(dist)
   !DEALLOCATE (row_blk_sizes, col_blk_sizes)

   deallocate(ham_dense)


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

   
! Funcions for interatomic interaction 
! for hamiltonian with block size 1, based on paper 
! PHYSICAL REVIEW B 96, 075311 (2017)
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
   !real(8) vpp_pi,vpp_sig
   real(8) t1
   t1=sqrt(dvec(1)**2+dvec(2)**2+dvec(3)**2)
   if (abs(dvec(3)).lt.eps) then
     t_of_d=-vpp_pi(t1)
   else
     t_of_d=-vpp_sig(t1)
   end if
end function




! Builds hamiltonian with block size 1, based on paper 
! PHYSICAL REVIEW B 96, 075311 (2017)
subroutine build_ham(ham, rr, rft, kc, nr, nrft)
  implicit none
  type(dbcsr_type), intent(inout)    :: ham
  integer, intent(in) :: nr,nrft
  real(8), intent(in) :: rr(3,nr)
  real(8), intent(in) :: rft(2,nrft)
  real(8), intent(in) :: kc(2)

  real(8), parameter :: rcut=10.d0
  integer ii,jj,irft
  real(8) t1,z2
  real(8) r01(2),r02(2),r12(2)
  real(8) r1(3),r2(3),dvec(3)
  complex(8) tpk,tpr(nrft)

  integer info,n,ld
  integer lwork
  real(8), allocatable :: rwork(:)
  complex(8), allocatable :: work(:)
  real(8) :: matrix_element

  integer :: nzz
  real(8), dimension(:), allocatable    :: values
  !allocate(ham(nr,nr))
   ALLOCATE (values(max_nze))

  nzz = 0
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
          print *,ii,jj, t_of_d(dvec) * tpr(irft)
       end do

       if ( abs(matrix_element) > 1.0d-10) then
          nzz = nzz + 1

         call dbcsr_get_stored_coordinates(ham, ii, jj, node_holds_blk)
         if (node_holds_blk .eq. mynode) then
            
            nze = row_blk_sizes(ii)*col_blk_sizes(jj)
            values(1) = matrix_element
            
            call dbcsr_put_block(ham, ii, jj, values(1:nze))
         endif

          
       endif
    end do
  end do 

  print *, "nonzeros ", nzz
  
   DEALLOCATE (values)
  return
end subroutine


! The same as before but without k-points and translation
subroutine build_ham_no_trans(ham, rr, nr)
  implicit none
  type(dbcsr_type), intent(inout)    :: ham
  integer, intent(in) :: nr
  real(8), intent(in) :: rr(3,nr)

  real(8), parameter :: rcut=10.d0
  integer ii,jj,irft
  real(8) t1,z2
  real(8) r01(2),r02(2),r12(2)
  real(8) r1(3),r2(3),dvec(3)

  integer info,n,ld
  integer lwork
  real(8), allocatable :: rwork(:)
  real(8), allocatable :: work(:)
  real(8) :: matrix_element

  integer :: nzz
  real(8), dimension(:), allocatable    :: values

   ALLOCATE (values(max_nze))
  
   nzz = 0

  do ii=1,nr
    r01=rr(1:2,ii)
    r1=rr(:,ii)
    
    do jj=ii,nr
       r02=rr(1:2,jj)

          z2=rr(3,jj)
          r2=(/r02(1),r02(2),z2/)
          dvec(:)=r2(:)-r1(:)
          t1=sqrt(dvec(1)**2+dvec(2)**2+dvec(3)**2)
          
          if (t1>rcut) cycle
 
          matrix_element = - t_of_d(dvec) 
          print *,ii,jj, t_of_d(dvec) 

       if ( abs(matrix_element) > 1.0d-10) then
          nzz = nzz + 1
         call dbcsr_get_stored_coordinates(ham, ii, jj, node_holds_blk)
         if (node_holds_blk .eq. mynode) then
            
            nze = row_blk_sizes(ii)*col_blk_sizes(jj)
            print *,nze
            values(:) = matrix_element
            call dbcsr_put_block(ham, ii, jj, values(1:nze))
            if (ii /= jj) then
              call dbcsr_put_block(ham, jj, ii, values(1:nze))
            endif
         endif
          
       endif
    end do
  end do 

  print *, "nonzeros ", nzz
  
   DEALLOCATE (values)
  return
end subroutine



! ------------------------------------------------
! functions for blocked s px py pz hamiltonian
! Based on PRB VOLUME 54, NUMBER 7 15 AUGUST 1996-I
! and PRB VOLUME 54, NUMBER 7
! ------------------------------------------------

! vec_norm should be normalized vector between two atoms
real(8) function vpp_nondiag(i, j, vec_norm, vpp_p, vpp_s)
   integer, intent(in) :: i, j
   real(8), intent(in) :: vec_norm(3), vpp_p, vpp_s
   real(8) :: cos_sq
   
   if (i /= j) then
      vpp_nondiag = vec_norm(i) * vec_norm(j) * ( vpp_s - vpp_p)
   else
      cos_sq = vec_norm(i) ** 2
      vpp_nondiag = cos_sq * vpp_s + (1 - cos_sq) * vpp_p
   endif
end function

! vec_norm should be normalized vector between two atoms
real(8) function vsp_nondiag(i, vec_norm, vsp_s)
   integer, intent(in) :: i
   real(8), intent(in) :: vec_norm(3), vsp_s
   
   vsp_nondiag = vec_norm(i) *  vsp_s
      
end function



real(8) function ff(dist)
  real(8), intent(in) :: dist
  ! paramterization from PRB VOLUME 54, NUMBER 7 15 AUGUST 1996-I
  ff = 1.0d0 / (1 + exp ( (dist - 14.0d0) / 0.5d0 ) )
end function


! par - params a b c d from PRB VOLUME 54, NUMBER 7
real(8) function sk(dist, par)
  real(8), intent(in) :: dist, par(4)
  
  sk = (par(1) + par(2) * dist + par(3) * dist**2 ) * exp(- par(4)**2 * dist) * ff(dist)
end function


subroutine build_ham_blocked(ham, rr, nr)
  implicit none
  type(dbcsr_type), intent(inout)    :: ham
  integer, intent(in) :: nr
  real(8), intent(in) :: rr(3,nr)

  real(8), parameter :: rcut=10.d0
  integer ii,jj,irft, ib,jb
  real(8) t1,z2
  real(8) r01(2),r02(2),r12(2)
  real(8) r1(3),r2(3),dvec(3)

  integer info,n,ld
  integer lwork
  real(8), allocatable :: rwork(:)
  real(8), allocatable :: work(:)
  real(8) :: vec_norm(3)

  integer :: nzz
  real(8), dimension(:), allocatable    :: values
  integer :: h, w

  ALLOCATE (values(max_nze))
  
  nzz = 0
   
  do ii=1,nr
    r1=rr(:,ii)
    
    do jj=ii,nr
      r2=rr(:,jj)
      dvec(:)=r2(:)-r1(:)
      t1=sqrt(dvec(1)**2+dvec(2)**2+dvec(3)**2)
      vec_norm(:) = dvec(:) / t1
      
      call dbcsr_get_stored_coordinates(ham, ii, jj, node_holds_blk)
      
      if (node_holds_blk .eq. mynode) then
      
      w = row_blk_sizes(ii)
      h = col_blk_sizes(jj)
      
      nze = w * h
      nzz = nzz + nze
      
      values(:) = 0            
      
      if (ii == jj) then
         ! calc onsite
         ! Es
         values(1) = energies(1)
         ! Ep
         do ib=2,4
            values( ib + h * (ib - 1)  ) = energies(2)
         enddo

        call dbcsr_put_block(ham, ii, jj, values(1:nze))
      else
         ! calc non-diag
         values(1) = sk(t1, vsss)
         do jb = 2,4
            values(jb) = vsp_nondiag(jb-1, vec_norm, sk(t1,vsps)) 
            values(1 + h * (jb-1)) = values(jb)
            
            do ib = jb,4
            values(ib + h * (jb-1)) = vpp_nondiag(ib-1, jb-1, vec_norm, sk(t1, vppp), sk(t1, vpps))
            values(jb + h * (ib-1)) = values(ib + h * (jb-1))
            enddo
         enddo
         call dbcsr_put_block(ham, jj, ii, values(1:nze))
         call dbcsr_put_block(ham, ii, jj, values(1:nze))
      endif
      

      endif
         
    end do
  end do 

  print *, "nonzeros ", nzz
  
  DEALLOCATE (values)
  return
end subroutine



subroutine build_ham_blocked_dense(ham, rr, nr, block_size)
  implicit none
  integer, intent(in) :: nr, block_size
  real(8), intent(in) :: rr(3,nr)
  real(8), allocatable, intent(inout) :: ham(:,:)

  real(8), parameter :: rcut=10.d0
  integer ii,jj,irft, ib,jb, i,j
  real(8) t1,z2
  real(8) r01(2),r02(2),r12(2)
  real(8) r1(3),r2(3),dvec(3)
  integer info,n,ld
  integer lwork
  real(8), allocatable :: rwork(:)
  real(8), allocatable :: work(:)
  real(8) :: vec_norm(3), matrix_element
  integer :: nzz
  integer :: h, w
  
  ham(:,:) = 0.0d0
  nzz = 0
   
  do ii=1,nr
    r1=rr(:,ii)
    
    do jj=ii,nr
      r2=rr(:,jj)
      dvec(:)=r2(:)-r1(:)
      t1=sqrt(dvec(1)**2+dvec(2)**2+dvec(3)**2)
      vec_norm(:) = dvec(:) / t1
      
      i = (ii - 1) * block_size + 1
      j = (jj - 1) * block_size + 1
      
      
      if (ii == jj) then
               
         ! calc onsite
         ! Es
         ham(i,i) = energies(1)

         ! Ep
         do ib= 1,3
            ham(i + ib, i + ib ) = energies(2)
         enddo

      else
         ! calc non-diag
         ham(i,j) = sk(t1, vsss)
         ham(j,i) = ham(i,j)
         do jb = 1,3
            matrix_element = vsp_nondiag(jb, vec_norm, sk(t1,vsps)) 
            
            ham(i, j+jb) = matrix_element 
            ham(i+jb, j) = matrix_element
            ham(j+jb, i) = matrix_element
            ham(j, i+jb) = matrix_element
            
            
            do ib = jb,3
            ham(i+ib, j+jb) = vpp_nondiag(ib, jb, vec_norm, sk(t1, vppp), sk(t1, vpps))
            ham(i+jb, j+ib) = ham(i+ib, j+jb)
            ham(j+jb, i+ib) = ham(i+ib, j+jb)
            ham(j+ib, i+jb) = ham(i+ib, j+jb)
            enddo
         enddo
      endif
      
         
    end do
  end do 

  return
end subroutine


subroutine diag_dense(ham, nelec, matrix_size)
  implicit none
  integer, intent(in) :: nelec, matrix_size
  real(8), allocatable, intent(inout) :: ham(:,:)

  real(8), allocatable :: rwork(:)
  real(8), allocatable :: work(:)
  real(8), allocatable :: eigval(:)
  integer lwork, i
  integer info,n,ld
  real(8) energy
    
    info=0
    n = matrix_size
    ld = matrix_size
    lwork=3*n
    allocate (eigval(n),work(lwork))
    
    call dsyev('V','U',n,ham,ld,eigval,work,lwork,info)
    
  if (info.ne.0) then
    write(*,*)
    write(*,'("Error: diagonalisation failed")')
    write(*,'(" ZHEEV returned INFO = ",I8)') info
    write(*,*)
    stop
  end if
  
  energy=0
  do i=1,nelec
    energy = energy + eigval(i)
  enddo
  
  print *,"Dense Energy: ", energy
  
    deallocate(eigval, work)
end subroutine

END PROGRAM 
