!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!
PROGRAM tight_binding
   USE mpi
   USE dbcsr_api,                       ONLY: &
        dbcsr_create, dbcsr_distribution_get, dbcsr_distribution_new, dbcsr_distribution_release, &
        dbcsr_distribution_type, dbcsr_finalize, dbcsr_finalize_lib, dbcsr_get_stored_coordinates, &
        dbcsr_init_lib, dbcsr_multiply, dbcsr_nblkcols_total, dbcsr_nblkrows_total, dbcsr_print, &
        dbcsr_put_block, dbcsr_release, dbcsr_type, dbcsr_type_no_symmetry, dbcsr_print_statistics, &
        dbcsr_type_real_8, dbcsr_type_complex_8, dbcsr_add_on_diag, dbcsr_trace,dbcsr_work_create
   
    use dm_ls_scf_methods, only: density_matrix_sign
  
    use dbcsr_scalapack  
    USE dbcsr_log_handling, ONLY: dbcsr_add_default_logger, &
                                 dbcsr_logger_create, &
                                 dbcsr_logger_release, &
                                 dbcsr_logger_type, &
                                 dbcsr_rm_default_logger
    USE dbcsr_machine, ONLY: default_output_unit, m_walltime
    use dbcsr_base_hooks
!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   implicit none

   type(dbcsr_type)                         ::  matrix_p, matrix_ks, matrix_s, matrix_s_inv, rho_ham

   integer, dimension(:), pointer           :: col_blk_sizes, row_blk_sizes
   integer                                  :: group, numnodes, mynode, ierr, &
                                               nblkrows_total, nblkcols_total, &
                                               node_holds_blk, max_nze, nze, row, col, row_s, col_s, &
                                               io_unit, max_row_size, max_col_size
   integer, dimension(2)                    :: npdims, myploc,coord
   integer, dimension(:), pointer           :: col_dist, row_dist
   type(dbcsr_distribution_type)            :: dist
   complex(8), dimension(:), allocatable    :: values
   logical                                  :: tr
   logical, dimension(2)                    :: period = .true.
integer                             :: numroc  ! blacs routine
  real(8), allocatable :: rr(:,:), cell_vec(:,:) 
  real(8), allocatable :: rft(:,:)
  real(8), allocatable :: kpathc(:,:)
  real(8) :: kc(2), mu, dbcsr_energy, hnondiag_cutoff, dist_cutoff
  integer :: nr, nrft, nkp, i, j, k, sys_dim, ntrans, cnt

  ! parameters for TB parameters
  real(8) :: energies(4), vsss(4), vsps(4), vpps(4), vppp(4)
  real(8) :: ssss(5), ssps(5), spps(5), sppp(5)
  real(8), allocatable :: ham_dense(:,:), s_dense(:,:), ham_dense2(:,:)
  character(len=20) :: exFmt
  integer :: block_size, nelectrons
     TYPE(dbcsr_logger_type), POINTER            :: logger
     CHARACTER(len=100) :: arg, cell_vecs, cutoffs, thrs

  real(8) :: lambda, energiesS(4), energiesP(4), cutoff, thr

  integer, dimension(9)         :: ides_a, ides_b, ides_c ! scalapack array desc

  integer :: icontxt, prow, pcol, myrow, mycol, n, nb, unit
  REAL(8), dimension(:,:), allocatable :: myA, myB, myC
  INTEGER, DIMENSION(:, :), POINTER   :: pgrid
  integer                             :: myrows, mycols

 CHARACTER(LEN=*), PARAMETER :: scalapack_diag_fn = 'scalapack_pdsyev'  
 CHARACTER(LEN=*), PARAMETER :: ls_fn = 'density_matrix_sign'  
 integer :: scalapack_fhandle, ls_fhandle

 real(8) :: t1 , t2
 
 real(8), parameter :: bohr_rad = 0.529 ! Angstrom
 integer, parameter :: max_atoms_for_printing_matrix = 10

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
   IF (ierr /= 0) ERROR STOP "Error in MPI_Comm_size"
   npdims(:) = 0
   CALL mpi_dims_create(numnodes, 2, npdims, ierr)
   IF (ierr /= 0) ERROR STOP "Error in MPI_Dims_create"
   CALL mpi_cart_create(MPI_COMM_WORLD, 2, npdims, period, .FALSE., group, ierr)
   IF (ierr /= 0) ERROR STOP "Error in MPI_Cart_create"

   CALL mpi_comm_rank(MPI_COMM_WORLD, mynode, ierr)
   IF (ierr /= 0) ERROR STOP "Error in MPI_Comm_rank"

   if (mynode == 0) WRITE (*, *) ' numnodes ', numnodes

  prow = npdims(1)
  pcol = npdims(2)

   ALLOCATE(pgrid(0:prow-1, 0:pcol-1))
  DO myrow = 0, prow-1
     DO mycol = 0, pcol-1
        coord = [myrow, mycol]
        CALL mpi_cart_rank(group, coord, pgrid(myrow, mycol), ierr)
     ENDDO
  ENDDO

    if (command_argument_count() < 4) ERROR STOP "too few arguments"

    CALL get_command_argument(1, arg)
    CALL get_command_argument(2, cell_vecs)
    CALL get_command_argument(3, cutoffs)
    CALL get_command_argument(4, thrs)
    read (cutoffs,*) cutoff
    read (thrs,*) thr
   
   !------- Read atom info ------------------
   open (newunit=unit, file=arg)
   read (unit, *) nr
   allocate(rr(3,nr))
   do j=1,nr
       read (unit, *) rr(:,j) 
   enddo
   close(unit)

   rr = rr / bohr_rad
   
   open (newunit=unit, file=cell_vecs)
   
   allocate(cell_vec(sys_dim,sys_dim))
   do j=1,3
       read (unit, *) cell_vec(:,j) 
   enddo
   close(unit)

   cell_vec = cell_vec / bohr_rad
  
   ! In case the example uses k-points to build Hamiltonian 
   !----- k points -------------
   ! temp 1 kpoint
   !    allocate(kpathc(2,1))   
   !    kpathc(:,1) = 0.0d0
   !    kc(:) = 0.0d0
   !    nkp = 1
   !    
   ! 
   ! set cell translation, 0 - no translation
   ntrans = 1  
   nrft = (2 * ntrans + 1)**sys_dim

   allocate(rft(sys_dim, nrft))
   
   cnt = 0
   do i = - ntrans, ntrans 
     do j = - ntrans, ntrans 
       do k = -ntrans, ntrans
         cnt = cnt + 1
         rft(:, cnt) = i * cell_vec(:,1) + j * cell_vec(:,2) + k * cell_vec(:,3)
       enddo  
     enddo
   enddo
   
   ! carbon has 4 orbitals
   block_size = 4
   ! occupation is 4 spins per atoms, taking spin into account will reveal 2 per atom
   nelectrons = nr * 2 
  
   if (mynode == 0) then   
     print *,"Number of atoms ", nr
     print *,"Number of orbitals (block size) ", block_size
     print *,"Occupation (number of electrons / 2)", nelectrons
     print *,"Problem dimension ", nr * block_size
     print *,"cutoff, filter_eps ", cutoff, thr
   endif
   !--- energies for TB ----------
   ! Carbon ---------------------------------------
   lambda = 1.59901905594
   ! s
   energies(1) = 0.1027
   energiesS = (/ -0.102789972814, -1.62604640052, -178.884826119, 4516.11342028 /)
   ! p
   energies(2) = 0.5426
   energiesP = (/ 0.542619178314, 2.73454062799, -67.139709883, 438.52883145 /)
   
   vsss = (/ 74.0837449667,  -18.3225697598,   -12.5253007169,  1.41100521808/)
   vsps = (/ -7.9172955767, 3.6163510241, 1.0416715714,  1.16878908431 /)
   vpps = (/ -5.7016933899, 1.0450894823, 1.5062731505,  1.13627440135 /)
   vppp = (/ 24.9104111573, -5.0603652530, -3.6844386855, 1.36548919302/)
  
   ssss = (/ 1.0,  0.18525064246, 1.56010486948, -0.308751658739, 1.13700564649 /)
   ssps = (/ 0.0, 1.85250642463, -2.50183774417, 0.178540723033, 1.12900344616 /)
   spps = (/ 1.0, -1.29666913067, 0.28270660019, -0.022234235553, 0.76177690688 /)
   sppp = (/ 1.0,  0.74092406925, -0.07310263856, 0.016694077196, 1.02148246334 /)

   ! If Silicon tests are needed, then uncomment and comment for Carbon
   ! Silicon ----------------------
   !lambda = 1.10356625153
   !! s
   !!energies(1) = 0.1027
   !energiesS = (/  -0.053233461902, -0.907642743185, -8.83084913674,   56.5661321469 /)  
  
   !! p
   !!energies(2) = 0.5426
   !energiesP = (/ 0.357859715265, 0.303647693101, 7.09222903560,  -77.4785508399   /)
   
   !vsss = (/ 219.560813651, -16.2132459618, -15.5048968097,    1.26439940008    /)
   !vsps = (/ 10.127687621, -4.4036811240, 0.2266767834,   0.92267194054   /)
   !vpps = (/ -22.959028107,   1.7207707741, 1.4191307713,   1.03136916513   /)
   !vppp = (/ 10.265449263,    4.6718241428, -2.2161562721,     1.11134828469     /)
  
   !ssss = (/ 1.0,  5.157587186,   0.660009308, -0.0815441307,   1.10814448800   /)
   !ssps = (/ 0.0, 8.873646665, -16.240770475, 5.1822969049,   1.24065238343   /)
   !spps = (/ 1.0, 11.250489009, -1.1701322929, -1.0591485021,   1.13762861032   /)
   !sppp = (/ 1.0,  -692.184231145, 396.153248956, -13.8172106270,   1.57248559510   /)
   !--------------------------------------

   hnondiag_cutoff = 1.0D-7 ! Ry
   dist_cutoff     = cutoff   ! Bohr
  
   
   if (nr < max_atoms_for_printing_matrix) then
      
       if (mynode == 0)  print *,"Diagonalize dense hamiltonian for reference"

      allocate(ham_dense(nr*block_size, nr*block_size),&
          s_dense(nr*block_size, nr*block_size))
      
      call build_ham_blocked_dense(ham_dense, rr, nr, block_size, dist_cutoff, rft, nrft)
     
      !can be uncommented to create overlapping matrix or without translation symmetry
      !call build_ham_blocked_dense_no_trans(ham_dense2, rr, nr, block_size, dist_cutoff)
      !call build_S_blocked_dense(s_dense, rr, nr, block_size, dist_cutoff)
      
      if (mynode == 0) then
         write(exFmt,'("(",I0,"(F8.4))")') nr*block_size
         do i=1,nr*block_size
         write(*,exFmt) (ham_dense(i,j), j=1,nr*block_size)
         end do
      endif
      
      !call diag_dense_gen(ham_dense, s_dense, nelectrons, nr*block_size)
      call diag_dense(ham_dense, nelectrons, nr*block_size)

      deallocate( ham_dense, s_dense)
   endif
   
   if (mynode == 0) print *,"DBCSR case"

   
   !***************************************************************************************
   !
   ! initialize libdbcsr
   CALL dbcsr_init_lib(MPI_COMM_WORLD)

   

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
   call build_ham_blocked(matrix_ks, rr, nr, hnondiag_cutoff, dist_cutoff, rft, nrft)
   !call build_ham_blocked_no_trans(matrix_ks, rr, nr, hnondiag_cutoff, dist_cutoff)
   
   CALL dbcsr_finalize(matrix_ks)
   CALL dbcsr_finalize(matrix_p)

   call dbcsr_add_on_diag(matrix_s, 1.0d0)
   call dbcsr_add_on_diag(matrix_s_inv, 1.0d0)
   
   
   CALL dbcsr_finalize(matrix_s)
   CALL dbcsr_finalize(matrix_s_inv)

  
   if (nr <= max_atoms_for_printing_matrix .and. mynode == 0) then
     print *,"!!!!!!!!!!!!!!!! Sparse Hamiltonian matrix !!!!!!!!!!!!!!!!!!!!!"
     call dbcsr_print(matrix_ks) 
   endif
  

  ! SCALAPACK environment
  icontxt = group
  ! Initialize blacs processor grid
  call blacs_gridinit(icontxt, 'R', prow, pcol)
  call blacs_gridinfo(icontxt, prow, pcol, myrow, mycol)

  n = nr * block_size
  nb = block_size

  myrows = numroc(n, nb, myrow, 0, prow)
  mycols = numroc(n, nb, mycol, 0, pcol)

  ! Prepare array descriptors for ScaLAPACK 
  ides_a(1) = 1         ! descriptor type
  ides_a(2) = icontxt   ! blacs context
  ides_a(3) = n         ! global number of rows
  ides_a(4) = n         ! global number of columns
  ides_a(5) = nb        ! row block size
  ides_a(6) = nb        ! column block size
  ides_a(7) = 0         ! initial process row
  ides_a(8) = 0         ! initial process column
  ides_a(9) = myrows   ! leading dimension of local array

  do i=1,9
     ides_b(i) = ides_a(i)
     ides_c(i) = ides_a(i)
  enddo

  ! scalapack
  allocate(myA(myrows,mycols))

  if (mynode==0) print *,"copy dbcsr to blacs matrix"

  call copy_dbcsr_to_blacs(matrix_ks, myA, nb, nb)
  

  if (nr < max_atoms_for_printing_matrix) then
         write(exFmt,'("(",I0,"(F8.4))")') myrows
         do i=1,mycols
         write(*,exFmt) (myA(i,j), j=1,myrows)
         end do
  endif


  if (mynode==0) print *,"diagonalize with scalapack"
  call timeset(scalapack_diag_fn, scalapack_fhandle)
  t1 = m_walltime()
  call diag_dense_par(myA, nelectrons, n, 1, 1, ides_a)
  t2 = m_walltime()
  call timestop(scalapack_fhandle)

  if (mynode == 0 ) print *, "scalapack_pdsyev time", t2 - t1

  deallocate(myA)

 ! call sign matrix solver   

  call timeset(ls_fn, ls_fhandle)
  t1 = m_walltime()

   ! Chemical potential for Diamond is taken to be inside of a gap
   mu = 0.58d0
   call density_matrix_sign(matrix_p, mu, .true., matrix_ks, matrix_s, &
       matrix_s_inv, nelectrons, thr)

   t2 = m_walltime() 
   call timestop(ls_fhandle)
   
   if (mynode == 0 ) print *, "DBCSR density_matrix_sign time", t2 - t1
   
   if (nr <= max_atoms_for_printing_matrix .and. mynode == 0) then
     print *,"!!!!!!!!!!!!!!!! Sparse density matrix !!!!!!!!!!!!!!!!!!!!!"
     call dbcsr_print(matrix_p)
   endif

   CALL dbcsr_multiply('N', 'N', 1.0D0, matrix_ks, matrix_p, 0.0D0, rho_ham)
     
   call dbcsr_trace(rho_ham, dbcsr_energy)
   
   if (mynode == 0)  print *,"DBCSR Energy: ", dbcsr_energy
    

  
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

   ! free comm
   CALL mpi_comm_free(group, ierr)
   IF (ierr /= 0) STOP "Error in MPI_Comm_free"

   call dbcsr_print_statistics(.true.)

   ! finalize libdbcsr
   CALL dbcsr_finalize_lib()
   
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


real(8) function rhoPart(lambda, dist)
    real(8), intent(in) :: lambda, dist
    real(8) :: rho

    rhoPart = exp(- lambda**2 * dist) * ff(dist)
end function


real(8) function diagH(rho, energies)
    real(8), intent(in) :: rho, energies(4)

    diagH = energies(1) + rho**(2.0/3.0) * energies(2) + rho**(4.0/3.0) * energies(3) + rho**2 * energies(4)
end function


real(8) function ff(dist)
  real(8), intent(in) :: dist
  ! paramterization from PRB VOLUME 54, NUMBER 7 15 AUGUST 1996-I
  ff = 1.0d0 / (1 + exp ( (dist - 14.0d0) / 0.5d0 ) )
end function


! for Slater Koster hopping term, params a b c d from PRB VOLUME 54, NUMBER 7
real(8) function Helem(dist, par)
  real(8), intent(in) :: dist, par(4)
  
  Helem = (par(1) + par(2) * dist + par(3) * dist**2 ) * exp(- par(4)**2 * dist) * ff(dist)
end function

! for Slater Koster overlapping integral , PRB VOLUME 54, NUMBER 7
real(8) function Selem(dist,par)
    real(8), intent(in) :: dist, par(5)
    
    Selem = (par(1) + par(2) * dist + par(3) * dist**2 + par(4) * dist**3 ) * exp(- par(5)**2 * dist) * ff(dist)
end function


subroutine build_ham_blocked_no_trans(ham, rr, nr, hnondiag_cutoff, distance_cutoff)
  implicit none
  type(dbcsr_type), intent(inout)    :: ham
  integer, intent(in) :: nr
  real(8), intent(in) :: rr(3,nr), hnondiag_cutoff, distance_cutoff

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

  integer :: nzz, nzz_per_site
  real(8), dimension(:), pointer    :: values
  integer :: h, w, datasize
  
  real(8) :: rho, diagVal
  
  
  nzz = 0
  datasize = 0
  
  ! estimate number of non-zero blocks based on distance_cutoff radius
  do ii=1,nr
    r1=rr(:,ii)
    nzz_per_site = 0
    do jj=ii,nr
      r2=rr(:,jj)
      dvec(:)=r2(:)-r1(:)
      t1=sqrt(dvec(1)**2+dvec(2)**2+dvec(3)**2)
      
      if (t1 < distance_cutoff) then
         call dbcsr_get_stored_coordinates(ham, ii, jj, node_holds_blk)
      
         if (node_holds_blk .eq. mynode) then
            
            w = row_blk_sizes(ii)
            h = col_blk_sizes(jj)
            nze = w * h
            if (ii == jj) then
               nzz = nzz + 1
               datasize = datasize + nze
            else
               nzz = nzz + 2
               datasize = datasize + 2 * nze
            endif
         endif
      endif
      
    end do
  end do 
   
  print *,"estimated nonzero blocks", nzz
  print *,"estimated nonzero elements", datasize

! allocate work array with specified numbers of blocks and total elements
!$OMP        PARALLEL &
!$OMP        PRIVATE (rho, node_holds_blk, r1, r2, dvec, t1, vec_norm, values, ib, jb) &
!$OMP        SHARED (ham, mynode) 
  call dbcsr_work_create(matrix = ham, nblks_guess = nzz, &
                         sizedata_guess = datasize, work_mutable=.TRUE.)
!$OMP        BARRIER
  ALLOCATE (values(max_nze))
!$OMP DO
  do ii=1,nr
    r1=rr(:,ii)

    rho = 0.0

    do jj=1,nr
      r2=rr(:,jj)
      dvec(:)=r2(:)-r1(:)
      t1=sqrt(dvec(1)**2+dvec(2)**2+dvec(3)**2)
      
      if (t1 < distance_cutoff .and. ii /= jj) then
        
         rho = rho + rhoPart(lambda, t1) 
         
             vec_norm(:) = dvec(:) / t1
             call dbcsr_get_stored_coordinates(ham, ii, jj, node_holds_blk)
             
             if (node_holds_blk .eq. mynode) then
             
                w = row_blk_sizes(jj)
                h = col_blk_sizes(ii)
                nze = w * h
                
               ! calc non-diag               
                values(1) = Helem(t1, vsss)
                do jb = 2,w
                   values(jb) = - vsp_nondiag(jb-1, vec_norm, Helem(t1,vsps))  ! minus because jb goes along column, see SK integrals
                   values(1 + h * (jb-1)) =  - values(jb)
                   
                   do ib = jb,h ! SK integrals are even
                      values(ib + h * (jb-1)) = vpp_nondiag(ib-1, jb-1, vec_norm, Helem(t1, vppp), Helem(t1, vpps))
                      values(jb + h * (ib-1)) =  values(ib + h * (jb-1))
                   enddo
                enddo
                
                if (abs(sum(values))/nze > hnondiag_cutoff) then
                   call dbcsr_put_block(ham, ii, jj, values(1:nze))
                endif
             endif
      endif
         
    end do

    call dbcsr_get_stored_coordinates(ham, ii, ii, node_holds_blk)
    
    if (node_holds_blk .eq. mynode) then
       w = row_blk_sizes(ii)
       h = col_blk_sizes(ii)
       nze = w * h
       values(:) = 0
       ! calc onsite
       ! Es
       values(1) = diagH(rho, energiesS)
       diagVal = diagH(rho, energiesP)

       ! Ep
       do ib=2,w
          values(ib + h * (ib - 1)) = diagVal
       enddo
       call dbcsr_put_block(ham, ii, ii, values(1:nze))
    endif

  end do 
!$OMP END DO
  DEALLOCATE (values)
  
!$OMP END PARALLEL
  return
end subroutine



subroutine build_ham_blocked(ham, rr, nr, hnondiag_cutoff, distance_cutoff, rft, nrft)
  implicit none
  type(dbcsr_type), intent(inout)    :: ham
  integer, intent(in) :: nr, nrft
  real(8), intent(in) :: rr(3,nr), hnondiag_cutoff, distance_cutoff, rft(3, nrft)

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

  integer :: nzz, nzz_per_site
  real(8), dimension(:), pointer    :: values
  integer :: h, w, datasize
  
  real(8) :: rho, diagVal, matrix_element
  logical :: add_block 
  
  nzz = 0
  datasize = 0
  
  ! estimate number of non-zero blocks based on distance_cutoff radius
  do ii=1,nr
    r1=rr(:,ii)
    nzz_per_site = 0
    
    do jj=ii,nr
      do irft = 1, nrft
        r2 = rr(:,jj) + rft(:,irft)
        dvec(:)=r2(:)-r1(:)
        t1=norm2(dvec)
    
        if (t1 < distance_cutoff) then
           call dbcsr_get_stored_coordinates(ham, ii, jj, node_holds_blk)
        
           if (node_holds_blk .eq. mynode) then
              w = row_blk_sizes(ii)
              h = col_blk_sizes(jj)
              nze = w * h
              
              if (ii == jj) then
                 nzz = nzz + 1
                 datasize = datasize + nze
              else
                 nzz = nzz + 2
                 datasize = datasize + 2 * nze
              endif
           endif
        endif
      enddo
      
    end do
  end do 
  
  if (mynode == 0) then 
    print *,"estimated nonzero blocks", nzz
    print *,"estimated nonzero elements", datasize
  endif
! allocate work array with specified numbers of blocks and total elements
!$OMP        PARALLEL &
!$OMP        PRIVATE (add_block, matrix_element, rho, node_holds_blk, r1, r2, dvec, t1, vec_norm, values, ib, jb) &
!$OMP        SHARED (ham, mynode) 
  call dbcsr_work_create(matrix = ham, nblks_guess = nzz, &
                         sizedata_guess = datasize, work_mutable=.TRUE.)
!$OMP        BARRIER
  ALLOCATE (values(max_nze))
!$OMP DO
  do ii=1,nr
    r1=rr(:,ii)
    rho = 0.0

    do jj=1,nr
      add_block = .false. 
      values(:) = 0.0

      do irft = 1, nrft
        r2 = rr(:,jj) + rft(:,irft)
        dvec(:)=r2(:)-r1(:)
        t1=norm2(dvec)
      
        if (t1 < distance_cutoff .and. ii /= jj) then
           rho = rho + rhoPart(lambda, t1) 
           vec_norm(:) = dvec(:) / t1
           call dbcsr_get_stored_coordinates(ham, ii, jj, node_holds_blk)
           
           if (node_holds_blk .eq. mynode) then
              w = row_blk_sizes(jj)
              h = col_blk_sizes(ii)
              nze = w * h
              add_block = .true.

              ! calc non-diag               
              values(1) =  values(1) + Helem(t1, vsss)
              
              do jb = 2,w
                 ! minus because jb goes along column, see SK integrals
                 matrix_element = vsp_nondiag(jb-1, vec_norm, Helem(t1,vsps))
                 values(jb) = values(jb) - matrix_element
                 values(1 + h * (jb-1)) =  values(1 + h * (jb-1)) + matrix_element
                 
                 do ib = jb,h ! SK integrals are even
                    matrix_element = vpp_nondiag(ib-1, jb-1, vec_norm, Helem(t1, vppp), Helem(t1, vpps))
                    values(ib + h * (jb-1)) = values(ib + h * (jb-1)) + matrix_element
                    values(jb + h * (ib-1)) =  values(ib + h * (jb-1))
                 enddo
              enddo
              
           endif
        endif
      enddo ! irft       
      
      if (add_block) then
         call dbcsr_put_block(ham, ii, jj, values(1:nze))
      endif
    end do

    call dbcsr_get_stored_coordinates(ham, ii, ii, node_holds_blk)
    
    if (node_holds_blk .eq. mynode) then
       w = row_blk_sizes(ii)
       h = col_blk_sizes(ii)
       nze = w * h
       values(:) = 0
       ! calc onsite
       ! Es
       values(1) = diagH(rho, energiesS)
       diagVal = diagH(rho, energiesP)

       ! Ep
       do ib=2,w
          values(ib + h * (ib - 1)) = diagVal
       enddo
       call dbcsr_put_block(ham, ii, ii, values(1:nze))
    endif

  end do 
!$OMP END DO
  DEALLOCATE (values)
  
!$OMP END PARALLEL
  return
end subroutine


subroutine build_ham_blocked_dense_no_trans(ham, rr, nr, block_size, distance_cutoff)
  implicit none
  integer, intent(in) :: nr, block_size
  real(8), intent(in) :: rr(3,nr), distance_cutoff
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
  real(8) :: vec_norm(3), matrix_element,diagVal, rho
  integer :: nzz
  integer :: h, w
  
  ham(:,:) = 0.0d0
  nzz = 0
   
  do ii=1,nr
    r1=rr(:,ii)
    rho = 0
    i = (ii - 1) * block_size + 1
    
    do jj=1,nr
      r2=rr(:,jj)
      dvec(:)=r2(:)-r1(:)
      t1=sqrt(dvec(1)**2+dvec(2)**2+dvec(3)**2)
      vec_norm(:) = dvec(:) / t1
      
      j = (jj - 1) * block_size + 1
      
      if (t1 < distance_cutoff .and.  ii /= jj) then
         rho = rho + rhoPart(lambda, t1) 
         
         ! calc non-diag
         ham(i,j) = Helem(t1, vsss)
         ham(j,i) = ham(i,j)
         do jb = 1,3
            matrix_element = vsp_nondiag(jb, vec_norm, Helem(t1,vsps)) 
            
            ham(i, j+jb) =   matrix_element 
            ham(i+jb, j) = - matrix_element
            
            
            do ib = jb,3
            matrix_element = vpp_nondiag(ib, jb, vec_norm, Helem(t1, vppp), Helem(t1, vpps))
            ham(i+ib, j+jb) =  matrix_element
            ham(i+jb, j+ib) =  matrix_element
            enddo
         enddo
      endif
      
            
      ! calc onsite
      ! Es
      ham(i,i) = diagH(rho, energiesS)
      diagVal = diagH(rho, energiesP)

      ! Ep
      do ib= 1,3
         ham(i + ib, i + ib ) = diagVal
      enddo

      
    end do
  end do 

  return
end subroutine


subroutine build_ham_blocked_dense(ham, rr, nr, block_size, distance_cutoff, rft, nrft)
  implicit none
  integer, intent(in) :: nr, block_size, nrft
  real(8), intent(in) :: rr(3,nr), distance_cutoff, rft(3,nrft)
  real(8), allocatable, intent(inout) :: ham(:,:)

  real(8), parameter :: rcut=10.d0
  integer ii,jj,irft, ib,jb, i,j
  real(8) t1,z2
  real(8) r01(2),r02(2),r12(2)
  real(8) r1(3),r2(3),dvec(3)
  integer info,n,ld
  integer lwork
  real(8), allocatable :: values(:,:)
  real(8), allocatable :: work(:)
  real(8) :: vec_norm(3), matrix_element,diagVal, rho
  integer :: nzz
  integer :: h, w
  logical :: add_block

  ham(:,:) = 0.0d0
  nzz = 0
   
  allocate(values(block_size,block_size))

  do ii=1,nr
    r1=rr(:,ii)
    rho = 0
    i = (ii - 1) * block_size + 1
    
    do jj=1,nr
      j = (jj - 1) * block_size + 1
      values(:,:) = 0.0
      add_block = .false.
      
      do irft = 1, nrft
        r2 = rr(:,jj) + rft(:,irft)
        dvec(:)=r2(:)-r1(:)
        t1=sqrt(dvec(1)**2+dvec(2)**2+dvec(3)**2)
        vec_norm(:) = dvec(:) / t1

         if (t1 < distance_cutoff .and.  ii /= jj) then
            add_block = .true.
            rho = rho + rhoPart(lambda, t1)             
            values(1,1) = values(1,1) + Helem(t1, vsss)
            
            do jb = 1,3
               matrix_element = vsp_nondiag(jb, vec_norm, Helem(t1,vsps)) 
               values(1,jb+1) = values(1,jb+1) + matrix_element
               values(jb+1,1) = values(jb+1,1) - matrix_element
               
               do ib = jb,3
                 matrix_element = vpp_nondiag(ib, jb, vec_norm, Helem(t1, vppp), Helem(t1, vpps))
                 values(1+ib,1+jb) = values(1+ib,1+jb) + matrix_element
                 values(1+jb,1+ib) = values(1+ib,1+jb)
               enddo
            enddo
         endif
      enddo
   
      if (add_block) then
          ham(i:i+block_size, j:j+block_size) = values(:,:)
      endif

    enddo
      
      ! calc onsite
      ! Es
    ham(i,i) = diagH(rho, energiesS)
    diagVal = diagH(rho, energiesP)

      ! Ep
    do ib= 1,3
       ham(i + ib, i + ib ) = diagVal
    enddo
      
  enddo 

  deallocate(values)
  return
end subroutine



subroutine build_S_blocked_dense(ham, rr, nr, block_size, distance_cutoff)
  implicit none
  integer, intent(in) :: nr, block_size
  real(8), intent(in) :: rr(3,nr), distance_cutoff
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
  real(8) :: vec_norm(3), matrix_element,diagVal, rho
  integer :: nzz
  integer :: h, w
  
  ham(:,:) = 0.0d0
  nzz = 0
   
  do ii=1,nr
    r1=rr(:,ii)
    rho = 0
    i = (ii - 1) * block_size + 1
    
    do jj=1,nr
      r2=rr(:,jj)
      dvec(:)=r2(:)-r1(:)
      t1=sqrt(dvec(1)**2+dvec(2)**2+dvec(3)**2)
      vec_norm(:) = dvec(:) / t1
      
      j = (jj - 1) * block_size + 1
      
      if (t1 < distance_cutoff .and.  ii /= jj) then
         
         ! calc non-diag
         ham(i,j) = Selem(t1, ssss)
         ham(j,i) = ham(i,j)
         do jb = 1,3
            matrix_element = vsp_nondiag(jb, vec_norm, Selem(t1,ssps)) 
            
            ham(i, j+jb) =   matrix_element 
            ham(i+jb, j) = -  matrix_element
            
            
            do ib = jb,3
            matrix_element = vpp_nondiag(ib, jb, vec_norm, Selem(t1, sppp),Selem(t1, spps))
            ham(i+ib, j+jb) =  matrix_element
            ham(i+jb, j+ib) =  matrix_element
            enddo
         enddo
      endif
      
            
      ! S orbital, calc onsite
      ham(i,i) = 1.0 
      

      ! P orbital
      do ib= 1,3
         ham(i + ib, i + ib ) = 1.0 
      enddo

      
    end do
  end do 

  return
end subroutine


subroutine diag_dense_par(ham, nelec, matrix_size, first_i, first_j, desc_ham)
  implicit none
  integer, intent(in) :: nelec, matrix_size, first_i, first_j, desc_ham(9)
  real(8), allocatable, intent(inout) :: ham(:,:)

  real(8), allocatable :: work(:)
  real(8) :: testwork(1)
  real(8), allocatable :: eigval(:)
  integer lwork, i
  integer info,n,ld, desczdef(9)
  real(8) energy , zdef(2)
  !real(8), allocatable :: Cmatr(:,:)
    
    info=0
    n = matrix_size
    ld = matrix_size

    
    allocate (eigval(n))
   
    ! get lwork
    call pdsyev('N','U', n, ham, first_i, first_j, desc_ham, eigval, zdef, 0, 0, desczdef, testwork, -1, info)
    
    lwork = testwork(1)
    allocate(work(lwork))
    ! diagonalize
    call pdsyev('N','U', n, ham, first_i, first_j, desc_ham, eigval, zdef, 0, 0, desczdef, work, lwork, info)

    
  if (info.ne.0) then
    write(*,*)
    write(*,'("Error: diagonalisation failed")')
    write(*,'(" ZHEEV returned INFO = ",I8)') info
    write(*,*)
    stop
  end if
  
  if (mynode == 0) then
    energy=0
    do i=1,nelec
      energy = energy + eigval(i)
    enddo
 
    print *,"Scalapack Dense Energy, HOMO, LUMO: ", energy, eigval(nelec),eigval(nelec+1)
  endif

    deallocate(eigval, work)
end subroutine



subroutine diag_dense(ham, nelec, matrix_size)
  implicit none
  integer, intent(in) :: nelec, matrix_size
  real(8), allocatable, intent(inout) :: ham(:,:)

  real(8), allocatable :: work(:)
  real(8), allocatable :: eigval(:)
  integer lwork, i
  integer info,n,ld
  real(8) energy
  !real(8), allocatable :: Cmatr(:,:)
    
    info=0
    n = matrix_size
    ld = matrix_size
    lwork=3*n
    allocate (eigval(n),work(lwork))
   

    call dsyev('N','U',n,ham,ld,eigval,work,lwork,info)
    
  if (info.ne.0) then
    write(*,*)
    write(*,'("Error: diagonalisation failed")')
    write(*,'(" ZHEEV returned INFO = ",I8)') info
    write(*,*)
    stop
  end if
  
  if (mynode == 0) then
    energy=0
    do i=1,nelec
      energy = energy + eigval(i)
    enddo
 
 
    print *,"Dense Energy, HOMO, LUMO: ", energy, eigval(nelec),eigval(nelec+1)
  endif 
    deallocate(eigval, work)
end subroutine


subroutine diag_dense_gen(ham, Smatr, nelec, matrix_size)
  implicit none
  integer, intent(in) :: nelec, matrix_size
  real(8), allocatable, intent(inout) :: ham(:,:), Smatr(:,:)

  real(8), allocatable :: work(:), iwork(:)
  real(8), allocatable :: eigval(:)
  integer lwork, i, liwork
  integer info,n,ld
  real(8) energy
    
    info=0
    n = matrix_size
    ld = matrix_size
    lwork=3*n
    liwork = 10
    allocate (eigval(n),work(lwork), iwork(liwork))
   
    call dsygvd(1, 'N', 'U', n, ham, ld, Smatr, ld, eigval, work, lwork,&
               iwork, liwork, info )

    
  if (info.ne.0) then
    write(*,*)
    write(*,'("Error: diagonalisation failed")')
    write(*,'(" ZHEEV returned INFO = ",I8)') info
    write(*,*)
    stop
  end if
  
  if (mynode == 0) then
    energy=0
    do i=1,nelec
      energy = energy + eigval(i)
    enddo
  
    print *,"Dense Energy, HOMO, LUMO: ", energy, eigval(nelec),eigval(nelec+1)
  endif 
  
  deallocate(eigval, work,  iwork)
end subroutine

END PROGRAM 
