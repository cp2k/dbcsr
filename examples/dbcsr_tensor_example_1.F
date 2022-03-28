!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

program dbcsr_tensor_example_1
   !! Sparse tensor contraction example
   use mpi
   use dbcsr_api, only: &
      dbcsr_type, dbcsr_distribution_type, dbcsr_init_lib, dbcsr_distribution_new, &
      dbcsr_type_no_symmetry, dbcsr_create, dbcsr_iterator_start, dbcsr_iterator_blocks_left, &
      dbcsr_iterator_stop, dbcsr_iterator_next_block, dbcsr_iterator_type, dbcsr_put_block, &
      dbcsr_reserve_blocks, dbcsr_scalar, dbcsr_finalize_lib, dbcsr_distribution_release, &
      dbcsr_nblkrows_total, dbcsr_type_real_8, dbcsr_release, dbcsr_nblkcols_total, dbcsr_finalize, &
      dbcsr_get_stored_coordinates, dbcsr_get_info, dbcsr_filter, dbcsr_checksum
   use dbcsr_tensor_api, only: &
      dbcsr_t_create, dbcsr_t_copy_matrix_to_tensor, &
      dbcsr_t_pgrid_type, dbcsr_t_type, dbcsr_t_distribution_type, dbcsr_t_nblks_total, &
      dbcsr_t_reserve_blocks, dbcsr_t_iterator_start, dbcsr_t_iterator_blocks_left, &
      dbcsr_t_iterator_next_block, dbcsr_t_iterator_stop, dbcsr_t_default_distvec, dbcsr_t_put_block, &
      dbcsr_t_copy, dbcsr_t_distribution_new, dbcsr_t_distribution_destroy, dbcsr_t_write_blocks, dbcsr_t_contract, &
      dbcsr_t_copy_tensor_to_matrix, dbcsr_t_destroy, dbcsr_t_pgrid_destroy, dbcsr_t_nblks_total, &
      dbcsr_t_pgrid_create, dbcsr_t_iterator_type, dbcsr_t_get_stored_coordinates, dbcsr_t_get_info, dbcsr_t_filter, &
      dbcsr_t_checksum, dbcsr_t_clear, dbcsr_t_batched_contract_init, dbcsr_t_batched_contract_finalize
   use iso_fortran_env, only: &
      output_unit, real64, int64

! --------------------------------------------------------------------------------------------------
! this example implements the sparse tensor contraction (einstein notation)
! c(n,o) = c(n,o) + a(i,j,k) x a(l,m,k) x b(i,l,n) x (b(m,o,j) + b(o,m,j))
!
! the tensors have the following shape and entries:
! a: n x n x 2n: a(i,j,k) = exp(-1/3*alpha*((i-j)**2+(i-k)**2+(j-k)**2))
! b: n x n x n: b(i,j,k) = exp(-1/3*beta*((i-j)**2+(i-k)**2+(j-k)**2))
! c: n x n: c(i,j) = exp(-1/2*gamma*(i-j)**2)
!
! due to the exponential decay of the tensor elements w.r.t. difference between two indices,
! all tensors are sparse. neglect of small tensor elements is controlled by threshold 'filter_eps':
! tensor blocks with frobenius norm < filter_eps are neglected.
!
! block sizes are set randomly in this example to demonstrate a heterogeneous sparsity pattern,
! these should ideally be adapted to the natural sparsity pattern of the problem
! (e.g. blocks corresponding to a set of gaussian basis functions with same exponent)
!
! DBCSR provides two basic operations in terms of which any tensor contraction can be expressed:
! dbcsr_t_contract: contraction of a pair of tensors
! dbcsr_t_copy: copy supporting redistribution and index permutation
!
! by default, DBCSR supports tensors of ranks between 2 and 4.
! higher ranks can be enabled by adapting 'maxrank' in 'dbcsr_tensor.fypp'.
!
! the above contraction is executed in the following order:
! 1) d(i,j,l,m) = a(i,j,k) x a(l,m,k)
! 2) e(j,m,n) = d(i,j,l,m) x b(i,l,n)
! 3) f(j,m,o) = b(m,o,j) + b(o,m,j)
! 4) c(n,o) = c(n,o) + e(j,m,n) x f(j,m,o)
!
! how to run (this example and DBCSR for tensors in general):
! - best performance is obtained by running with mpi and one openmp thread per rank.
! - ideally number of mpi ranks should be composed of small prime factors (e.g. powers of 2).
! - for sparse data & heterogeneous block sizes, DBCSR should be run on CPUs with libxsmm backend.
! - for dense data best performance is obtained by choosing homogeneous block sizes of 64 and by
!   compiling with GPU support.
! --------------------------------------------------------------------------------------------------

! ------ Parameters ------

   ! example type:
   ! - 1: debug (small & verbose)
   ! - 2: default (medium size)
   ! - 3: large (requires parallelism)
   ! - 4: large, batched contraction to reduce memory (does not require parallelism)
   integer, parameter :: example_type = 2

   ! filter threshold (larger value means more sparse but less accurate)
   real(real64), parameter :: filter_eps = 1.0e-08_real64

   ! number of batches in one dimension (to reduce memory footprint)
   integer, parameter :: nbatch = 8

   ! exponents for gaussians
   real(real64) :: alpha, beta, gamma

   ! maximum block size (actual block sizes are random between 1 and this number)
   integer :: max_bsize

   ! tensor size in one dimension (n)
   integer :: nel

   ! tune sparsity by scaling exponent for calculation of tensor elements
   real(real64) :: scale_exp

   ! contract all tensors at once
   logical :: contract_direct

   ! contract in batches (memory saving)
   logical :: contract_batched

   ! verbosity level
   ! 0: essential output
   ! 1: tensor log
   ! 2: verbose tensor log
   ! 3: verbose tensor log and print all tensor data
   integer :: verbosity

   integer :: &
      ierr, numnodes, mynode, node_holds_blk, io_unit, io_unit_dbcsr, ind, row, col, blk, group, &
      i, j, k, l, n, o, i_arr, j_arr, k_arr, l_arr, n_arr, o_arr, blk_size, &
      min_exp, min_exp_ij, min_exp_ik, min_exp_jk, min_exp_il, min_exp_in, min_exp_ln, &
      ibatch, jbatch, lbatch, mbatch
   integer, dimension(:), allocatable :: &
      offset_i, offset_j, offset_l, offset_k, offset_n, tmp, &
      start_batch_i, start_batch_j, start_batch_l, start_batch_m, &
      end_batch_i, end_batch_j, end_batch_l, end_batch_m
   integer, dimension(:), allocatable, target :: &
      blk_ind_1, blk_ind_2, blk_ind_3, &
      blk_size_i, blk_size_j, blk_size_k, blk_size_l, blk_size_m, blk_size_n, blk_size_o, &
      dist_1, dist_2, dist_3, dist_4
   integer, dimension(:, :), allocatable :: bounds_1, bounds_2, bounds_3
   integer, dimension(:), pointer :: &
      row_dist, col_dist, row_blk_size, col_blk_size, row_offset, col_offset
   integer, dimension(2) :: shape_2d, blk_ind_2d, blk_size_2d, blk_offset_2d, pdims_2d
   integer, dimension(3) :: blk_ind_3d, pdims_3d, shape_3d, blk_size_3d, blk_offset_3d
   integer, dimension(4) :: shape_4d, pdims_4d
   integer, dimension(7) :: shape_ijklmno
   integer(int64) :: nflop_sum, nflop
   real(real64) :: cs, t1, t0, time, flop_rate
   real(real64), dimension(:, :), pointer :: blk_values_2d
   real(real64), dimension(:, :, :), allocatable :: blk_values_3d
   logical :: tr
   logical, dimension(2) :: period = .true.
   type(dbcsr_type) :: c_matrix
   type(dbcsr_distribution_type) :: dist_matrix
   type(dbcsr_iterator_type) :: iter_matrix
   type(dbcsr_t_pgrid_type) :: pgrid_3d, pgrid_4d
   type(dbcsr_t_distribution_type) :: dist_tensor
   type(dbcsr_t_type) :: a_ijk, a_lmk, b_iln, c_no, d_ijlm, e_jmn, f_jmo
   type(dbcsr_t_iterator_type) :: iter_tensor

   ! prefactor in exponent for tensor data
   alpha = 1.0_real64; beta = 0.5_real64; gamma = 2.0_real64

   ! parameters for different example types
   select case (example_type)
   case (1)
      nel = 10
      max_bsize = 3
      verbosity = 3
      scale_exp = 10.0_real64
      contract_direct = .true.
      contract_batched = .false.
   case (2)
      nel = 200
      max_bsize = 10
      verbosity = 1
      scale_exp = 0.01_real64
      contract_direct = .true.
      contract_batched = .false.
   case (3)
      nel = 2000
      max_bsize = 10
      verbosity = 1
      scale_exp = 0.01_real64
      contract_direct = .true.
      contract_batched = .false.
   case (4)
      nel = 2000
      max_bsize = 10
      verbosity = 0
      scale_exp = 0.01_real64
      contract_direct = .false.
      contract_batched = .true.
   end select

   alpha = alpha*scale_exp
   beta = beta*scale_exp
   gamma = gamma*scale_exp

   ! initialize mpi
   call mpi_init(ierr)
   if (ierr /= 0) stop "error in mpi_init"

   call mpi_comm_size(mpi_comm_world, numnodes, ierr)
   if (ierr /= 0) stop "error in mpi_comm_size"

   call mpi_comm_rank(mpi_comm_world, mynode, ierr)
   if (ierr /= 0) stop "error in mpi_comm_rank"

   ! initialize DBCSR
   call dbcsr_init_lib(mpi_comm_world)

   ! prepare output
   io_unit_dbcsr = -1
   io_unit = -1
   if (mynode == 0 .and. verbosity > 0) io_unit_dbcsr = output_unit
   if (mynode == 0) io_unit = output_unit

   ! create block sizes
   call random_blk_sizes(nel, shape_ijklmno(1), blk_size_i)
   call random_blk_sizes(nel, shape_ijklmno(2), blk_size_j)
   call random_blk_sizes(2*nel, shape_ijklmno(3), blk_size_k)
   call random_blk_sizes(nel, shape_ijklmno(4), blk_size_l)
   call random_blk_sizes(nel, shape_ijklmno(5), blk_size_m)
   call random_blk_sizes(nel, shape_ijklmno(6), blk_size_n)
   call random_blk_sizes(nel, shape_ijklmno(7), blk_size_o)

! ------ create matrix c[no] ------

   ! shape (number of blocks in each dimension)
   shape_2d = shape_ijklmno(6:7)

   ! set up 2-dimensional process grid
   pdims_2d(:) = 0
   call mpi_dims_create(numnodes, 2, pdims_2d, ierr)
   if (ierr /= 0) stop "error in mpi_dims_create"
   call mpi_cart_create(mpi_comm_world, 2, pdims_2d, period, .false., group, ierr)
   if (ierr /= 0) stop "error in mpi_cart_create"

   ! row and column distribution (mapping blocks in each dimension to process grid coordinate)
   ! this routine creates a load-balanced distribution for heterogeneous block sizes, alternatively
   ! any custom distribution can be used
   allocate (dist_1(shape_2d(1)))
   call dbcsr_t_default_distvec(shape_2d(1), pdims_2d(1), blk_size_n, dist_1)
   allocate (dist_2(shape_2d(2)))
   call dbcsr_t_default_distvec(shape_2d(2), pdims_2d(2), blk_size_o, dist_2)

   ! convert to pointers because DBCSR matrix api only accepts pointers
   row_dist => dist_1
   col_dist => dist_2

   ! create distribution
   call dbcsr_distribution_new(dist_matrix, group=group, row_dist=row_dist, col_dist=col_dist)
   deallocate (dist_1, dist_2)

   ! convert to pointers since DBCSR matrix api only accepts pointers
   row_blk_size => blk_size_n
   col_blk_size => blk_size_o

   ! create DBCSR matrix
   call dbcsr_create(matrix=c_matrix, name="c[n|o]", dist=dist_matrix, matrix_type=dbcsr_type_no_symmetry, &
                     row_blk_size=row_blk_size, col_blk_size=col_blk_size, data_type=dbcsr_type_real_8)

   call dbcsr_distribution_release(dist_matrix)

! ------ fill matrix c[no] ------

   ! reserve non-zero blocks. for performance it is important to first reserve all present blocks
   ! before calculating them and inserting them into DBCSR matrix.
   call dbcsr_get_info(c_matrix, row_blk_offset=row_offset, col_blk_offset=col_offset)

   ind = 0
   allocate (blk_ind_1(0), blk_ind_2(0))
   do row = 1, dbcsr_nblkrows_total(c_matrix)
      do col = 1, dbcsr_nblkcols_total(c_matrix)

         ! only consider blocks that are local to this rank (according to distribution)
         call dbcsr_get_stored_coordinates(c_matrix, row, col, node_holds_blk)
         if (node_holds_blk /= mynode) cycle

         ! calculate largest matrix element to determine an upper bound for block frobenius norm
         ! block is reserved only if this estimate is larger than the filter_eps parameter
         min_exp = block_minabsdiff(row_offset(row), col_offset(col), row_blk_size(row), col_blk_size(col))
         blk_size = row_blk_size(row)*col_blk_size(col)
         if (blk_size*exp(-0.5*gamma*real(min_exp**2)) < filter_eps) cycle

         ind = ind + 1

         ! store index of block to be reserved
         call move_alloc(blk_ind_1, tmp)
         allocate (blk_ind_1(ind))
         blk_ind_1(:ind - 1) = tmp; deallocate (tmp)

         call move_alloc(blk_ind_2, tmp)
         allocate (blk_ind_2(ind))
         blk_ind_2(:ind - 1) = tmp; deallocate (tmp)

         blk_ind_1(ind) = row
         blk_ind_2(ind) = col

      end do
   end do

   ! reserve blocks
   call dbcsr_reserve_blocks(c_matrix, blk_ind_1, blk_ind_2)
   deallocate (blk_ind_1, blk_ind_2)

   ! iterate over reserved matrix blocks to fill them with data
   call dbcsr_iterator_start(iter_matrix, c_matrix)
   do while (dbcsr_iterator_blocks_left(iter_matrix))
      call dbcsr_iterator_next_block(iter_matrix, blk_ind_2d(1), blk_ind_2d(2), blk_values_2d, tr, &
                                     row_size=blk_size_2d(1), col_size=blk_size_2d(2), &
                                     row_offset=blk_offset_2d(1), col_offset=blk_offset_2d(2))
      do n_arr = 1, blk_size_2d(1)
         do o_arr = 1, blk_size_2d(2)
            ! get matrix element index n & o from block offset
            n = n_arr + blk_offset_2d(1) - 1
            o = o_arr + blk_offset_2d(2) - 1
            ! calculate matrix element
            blk_values_2d(n_arr, o_arr) = exp(-0.5*gamma*real((n - o)**2))
         end do
      end do
   end do
   call dbcsr_iterator_stop(iter_matrix)

   ! finalize the DBCSR matrix
   call dbcsr_finalize(c_matrix)

   ! sparsity refinement by removing small blocks
   call dbcsr_filter(c_matrix, filter_eps)

   ! create tensor from DBCSR matrix for tensor contraction and copy data
   ! (alternatively we could have directly created c_matrix as a tensor)
   call dbcsr_t_create(c_matrix, c_no)
   call dbcsr_t_copy_matrix_to_tensor(c_matrix, c_no)

! ------ create tensor a[ijk] ------

   ! note: tensor api is analogous to matrix api with a few differences of technical and historical nature

   shape_3d = shape_ijklmno(1:3)

   ! n-rank tensor requires an n-dimensional process grid:
   ! 'dbcsr_t_pgrid_create' is analogous to 'mpi_cart_create' but comes with some additional defaults.
   ! If the tensor dimensions vary significantly in size, it's important for performance to use the
   ! optional argument 'tensor_dims' to specify the tensor (block) dimensions.
   pdims_3d(:) = 0
   call dbcsr_t_pgrid_create(mpi_comm_world, pdims_3d, pgrid_3d)

   allocate (dist_1(shape_3d(1)))
   call dbcsr_t_default_distvec(shape_3d(1), pdims_3d(1), blk_size_i, dist_1)

   allocate (dist_2(shape_3d(2)))
   call dbcsr_t_default_distvec(shape_3d(2), pdims_3d(2), blk_size_j, dist_2)

   allocate (dist_3(shape_3d(3)))
   call dbcsr_t_default_distvec(shape_3d(3), pdims_3d(3), blk_size_k, dist_3)

   call dbcsr_t_distribution_new(dist_tensor, pgrid_3d, dist_1, dist_2, dist_3)
   deallocate (dist_1, dist_2, dist_3)

   ! create tensor. Compared with dbcsr_create this takes 2 additional arguments to control how the
   ! tensor is internally represented as a matrix:
   ! - map1_2d: which tensor dimensions are mapped to the first matrix dimension (in this case i & j)
   ! - map2_2d: which tensor dimensions are mapped to the second matrix dimension (in this case k)
   ! (these arguments need to be given for performance reasons, see documentation of dbcsr_t_contract
   ! for more info)
   call dbcsr_t_create(a_ijk, "a[ij|k]", dist_tensor, &
                       map1_2d=[1, 2], map2_2d=[3], &
                       data_type=dbcsr_type_real_8, &
                       blk_size_1=blk_size_i, blk_size_2=blk_size_j, blk_size_3=blk_size_k)
   call dbcsr_t_distribution_destroy(dist_tensor)

! ------ create a[lmk]  ------
   ! note: normally we can just create an exact copy by calling:
   !    call dbcsr_t_create(a_ijk, a_lmk)
   !    call dbcsr_t_copy(a_ijk, a_lmk)
   ! here we need to create from scratch since the tensors have different block sizes
   shape_3d = shape_ijklmno([4, 5, 3])

   allocate (dist_1(shape_3d(1)))
   call dbcsr_t_default_distvec(shape_3d(1), pdims_3d(1), blk_size_l, dist_1)

   allocate (dist_2(shape_3d(2)))
   call dbcsr_t_default_distvec(shape_3d(2), pdims_3d(2), blk_size_m, dist_2)

   allocate (dist_3(shape_3d(3)))
   call dbcsr_t_default_distvec(shape_3d(3), pdims_3d(3), blk_size_k, dist_3)

   call dbcsr_t_distribution_new(dist_tensor, pgrid_3d, dist_1, dist_2, dist_3)
   deallocate (dist_1, dist_2, dist_3)
   call dbcsr_t_create(a_lmk, "a[lm|k]", dist_tensor, [1, 2], [3], dbcsr_type_real_8, &
                       blk_size_l, blk_size_m, blk_size_k)
   call dbcsr_t_distribution_destroy(dist_tensor)

! ------ fill tensor a[ijk] and copy to a[lmk] ------

   allocate (offset_i(dbcsr_t_nblks_total(a_ijk, 1)))
   allocate (offset_j(dbcsr_t_nblks_total(a_ijk, 2)))
   allocate (offset_k(dbcsr_t_nblks_total(a_ijk, 3)))
   call dbcsr_t_get_info(a_ijk, blk_offset_1=offset_i, blk_offset_2=offset_j, blk_offset_3=offset_k)

   ind = 0
   allocate (blk_ind_1(0), blk_ind_2(0), blk_ind_3(0))
   do i = 1, dbcsr_t_nblks_total(a_ijk, 1)
      do j = 1, dbcsr_t_nblks_total(a_ijk, 2)
         do k = 1, dbcsr_t_nblks_total(a_ijk, 3)

            call dbcsr_t_get_stored_coordinates(a_ijk, [i, j, k], node_holds_blk)
            if (node_holds_blk /= mynode) cycle

            min_exp_ij = block_minabsdiff(offset_i(i), offset_j(j), blk_size_i(i), blk_size_j(j))
            min_exp_ik = block_minabsdiff(offset_i(i), offset_k(k), blk_size_i(i), blk_size_k(k))
            min_exp_jk = block_minabsdiff(offset_j(j), offset_k(k), blk_size_j(j), blk_size_k(k))

            blk_size = blk_size_i(i)*blk_size_j(j)*blk_size_k(k)

            if (blk_size*exp(-1./3*alpha*real(min_exp_ij**2 + min_exp_ik**2 + min_exp_jk**2)) < filter_eps) cycle

            ind = ind + 1

            call move_alloc(blk_ind_1, tmp)
            allocate (blk_ind_1(ind))
            blk_ind_1(:ind - 1) = tmp; deallocate (tmp)

            call move_alloc(blk_ind_2, tmp)
            allocate (blk_ind_2(ind))
            blk_ind_2(:ind - 1) = tmp; deallocate (tmp)

            call move_alloc(blk_ind_3, tmp)
            allocate (blk_ind_3(ind))
            blk_ind_3(:ind - 1) = tmp; deallocate (tmp)

            blk_ind_1(ind) = i
            blk_ind_2(ind) = j
            blk_ind_3(ind) = k
         end do
      end do
   end do

   call dbcsr_t_reserve_blocks(a_ijk, blk_ind_1, blk_ind_2, blk_ind_3)
   deallocate (blk_ind_1, blk_ind_2, blk_ind_3)

   call dbcsr_t_iterator_start(iter_tensor, a_ijk)
   do while (dbcsr_t_iterator_blocks_left(iter_tensor))
      ! direct access to block pointers via iterator is not possible in the tensor api
      ! the iterator goes over indices and then we call 'put_block'
      call dbcsr_t_iterator_next_block(iter_tensor, blk_ind_3d, blk, blk_size=blk_size_3d, blk_offset=blk_offset_3d)
      allocate (blk_values_3d(blk_size_3d(1), blk_size_3d(2), blk_size_3d(3)))
      do i_arr = 1, blk_size_3d(1)
         do j_arr = 1, blk_size_3d(2)
            do k_arr = 1, blk_size_3d(3)
               i = i_arr + blk_offset_3d(1) - 1
               j = j_arr + blk_offset_3d(2) - 1
               k = k_arr + blk_offset_3d(3) - 1
               blk_values_3d(i_arr, j_arr, k_arr) = exp(-1./3*alpha*real((i - j)**2 + (i - k)**2 + (j - k)**2))
            end do
         end do
      end do
      call dbcsr_t_put_block(a_ijk, blk_ind_3d, blk_size_3d, blk_values_3d)
      deallocate (blk_values_3d)
   end do
   call dbcsr_t_iterator_stop(iter_tensor)

   call dbcsr_t_filter(a_ijk, filter_eps)

   ! no need to finalize for tensors, this is done internally

   ! fill tensor (lmk) by copying from a[ijk]
   call dbcsr_t_copy(a_ijk, a_lmk)
   call dbcsr_t_filter(a_lmk, filter_eps)

! ------ create tensor b[iln] ------
   shape_3d = shape_ijklmno([1, 4, 6])

   allocate (dist_1(shape_3d(1)))
   call dbcsr_t_default_distvec(shape_3d(1), pdims_3d(1), blk_size_i, dist_1)

   allocate (dist_2(shape_3d(2)))
   call dbcsr_t_default_distvec(shape_3d(2), pdims_3d(2), blk_size_l, dist_2)

   allocate (dist_3(shape_3d(3)))
   call dbcsr_t_default_distvec(shape_3d(3), pdims_3d(3), blk_size_n, dist_3)

   call dbcsr_t_distribution_new(dist_tensor, pgrid_3d, dist_1, dist_2, dist_3)
   deallocate (dist_1, dist_2, dist_3)

   call dbcsr_t_create(b_iln, "b[il|n]", dist_tensor, [1, 2], [3], dbcsr_type_real_8, &
                       blk_size_i, blk_size_l, blk_size_n)
   call dbcsr_t_distribution_destroy(dist_tensor)

! ------ fill tensor b[iln] ------
   allocate (offset_l(dbcsr_t_nblks_total(b_iln, 2)))
   allocate (offset_n(dbcsr_t_nblks_total(b_iln, 3)))
   call dbcsr_t_get_info(b_iln, blk_offset_2=offset_l, blk_offset_3=offset_n)

   ind = 0
   allocate (blk_ind_1(0), blk_ind_2(0), blk_ind_3(0))
   do i = 1, dbcsr_t_nblks_total(b_iln, 1)
      do l = 1, dbcsr_t_nblks_total(b_iln, 2)
         do n = 1, dbcsr_t_nblks_total(b_iln, 3)

            call dbcsr_t_get_stored_coordinates(b_iln, [i, l, n], node_holds_blk)
            if (node_holds_blk /= mynode) cycle

            min_exp_il = block_minabsdiff(offset_i(i), offset_l(l), blk_size_i(i), blk_size_l(l))
            min_exp_in = block_minabsdiff(offset_i(i), offset_n(n), blk_size_i(i), blk_size_n(n))
            min_exp_ln = block_minabsdiff(offset_l(l), offset_n(n), blk_size_l(l), blk_size_n(n))

            blk_size = blk_size_i(i)*blk_size_l(l)*blk_size_n(n)

            if (blk_size*exp(-1./3*beta*real(min_exp_il**2 + min_exp_in**2 + min_exp_ln**2)) < filter_eps) cycle

            ind = ind + 1

            call move_alloc(blk_ind_1, tmp)
            allocate (blk_ind_1(ind))
            blk_ind_1(:ind - 1) = tmp; deallocate (tmp)

            call move_alloc(blk_ind_2, tmp)
            allocate (blk_ind_2(ind))
            blk_ind_2(:ind - 1) = tmp; deallocate (tmp)

            call move_alloc(blk_ind_3, tmp)
            allocate (blk_ind_3(ind))
            blk_ind_3(:ind - 1) = tmp; deallocate (tmp)

            blk_ind_1(ind) = i
            blk_ind_2(ind) = l
            blk_ind_3(ind) = n
         end do
      end do
   end do

   call dbcsr_t_reserve_blocks(b_iln, blk_ind_1, blk_ind_2, blk_ind_3)
   deallocate (blk_ind_1, blk_ind_2, blk_ind_3)

   call dbcsr_t_iterator_start(iter_tensor, b_iln)
   do while (dbcsr_t_iterator_blocks_left(iter_tensor))
      call dbcsr_t_iterator_next_block(iter_tensor, blk_ind_3d, blk, blk_size=blk_size_3d, blk_offset=blk_offset_3d)
      allocate (blk_values_3d(blk_size_3d(1), blk_size_3d(2), blk_size_3d(3)))
      do i_arr = 1, blk_size_3d(1)
         do l_arr = 1, blk_size_3d(2)
            do n_arr = 1, blk_size_3d(3)
               i = i_arr + blk_offset_3d(1) - 1
               l = l_arr + blk_offset_3d(2) - 1
               n = n_arr + blk_offset_3d(3) - 1
               blk_values_3d(i_arr, l_arr, n_arr) = exp(-1./3*beta*real((i - l)**2 + (i - n)**2 + (l - n)**2))
            end do
         end do
      end do
      call dbcsr_t_put_block(b_iln, blk_ind_3d, blk_size_3d, blk_values_3d)
      deallocate (blk_values_3d)
   end do
   call dbcsr_t_iterator_stop(iter_tensor)

   call dbcsr_t_filter(b_iln, filter_eps)

! ------ create tensor e[jmn] ------
   shape_3d = shape_ijklmno([2, 5, 6])

   allocate (dist_1(shape_3d(1)))
   call dbcsr_t_default_distvec(shape_3d(1), pdims_3d(1), blk_size_j, dist_1)

   allocate (dist_2(shape_3d(2)))
   call dbcsr_t_default_distvec(shape_3d(2), pdims_3d(2), blk_size_m, dist_2)

   allocate (dist_3(shape_3d(3)))
   call dbcsr_t_default_distvec(shape_3d(3), pdims_3d(3), blk_size_n, dist_3)

   call dbcsr_t_distribution_new(dist_tensor, pgrid_3d, dist_1, dist_2, dist_3)
   deallocate (dist_1, dist_2, dist_3)
   call dbcsr_t_create(e_jmn, "e[jm|n]", dist_tensor, [1, 2], [3], dbcsr_type_real_8, &
                       blk_size_j, blk_size_m, blk_size_n)
   call dbcsr_t_distribution_destroy(dist_tensor)

! ------ create tensor f[jmo] ------
   shape_3d = shape_ijklmno([2, 5, 7])

   allocate (dist_1(shape_3d(1)))
   call dbcsr_t_default_distvec(shape_3d(1), pdims_3d(1), blk_size_j, dist_1)

   allocate (dist_2(shape_3d(2)))
   call dbcsr_t_default_distvec(shape_3d(2), pdims_3d(2), blk_size_m, dist_2)

   allocate (dist_3(shape_3d(3)))
   call dbcsr_t_default_distvec(shape_3d(3), pdims_3d(3), blk_size_o, dist_3)

   call dbcsr_t_distribution_new(dist_tensor, pgrid_3d, dist_1, dist_2, dist_3)
   deallocate (dist_1, dist_2, dist_3)
   call dbcsr_t_create(f_jmo, "f[jm|o]", dist_tensor, [1, 2], [3], dbcsr_type_real_8, &
                       blk_size_j, blk_size_m, blk_size_o)
   call dbcsr_t_distribution_destroy(dist_tensor)

! ------ create and fill tensor f[jmo] ------
! ------ f(j,m,o) = b(m,o,j) + b(o,m,j) ------

   ! note: order argument of dbcsr_t_copy allows for arbitrary index permutations
   ! (same convention as fortran reshape intrinsic)

   ! f(j,m,o) = b(m,o,j)
   call dbcsr_t_copy(b_iln, f_jmo, order=[2, 3, 1])

   ! f(j,m,o) = f(j,m,o) + b(o,m,j)
   call dbcsr_t_copy(b_iln, f_jmo, order=[3, 2, 1], summation=.true.)

   call dbcsr_t_filter(f_jmo, filter_eps)

! ------ create tensor d[i,j,l,m] ------
   shape_4d = shape_ijklmno([1, 2, 4, 5])

   pdims_4d(:) = 0
   call dbcsr_t_pgrid_create(mpi_comm_world, pdims_4d, pgrid_4d)

   allocate (dist_1(shape_4d(1)))
   call dbcsr_t_default_distvec(shape_4d(1), pdims_4d(1), blk_size_i, dist_1)

   allocate (dist_2(shape_4d(2)))
   call dbcsr_t_default_distvec(shape_4d(2), pdims_4d(2), blk_size_j, dist_2)

   allocate (dist_3(shape_4d(3)))
   call dbcsr_t_default_distvec(shape_4d(3), pdims_4d(3), blk_size_l, dist_3)

   allocate (dist_4(shape_4d(4)))
   call dbcsr_t_default_distvec(shape_4d(4), pdims_4d(4), blk_size_m, dist_4)

   call dbcsr_t_distribution_new(dist_tensor, pgrid_4d, dist_1, dist_2, dist_3, dist_4)
   deallocate (dist_1, dist_2, dist_3, dist_4)

   call dbcsr_t_create(d_ijlm, "d[ij|lm]", dist_tensor, [1, 2], [3, 4], dbcsr_type_real_8, &
                       blk_size_i, blk_size_j, blk_size_l, blk_size_m)
   call dbcsr_t_distribution_destroy(dist_tensor)

! ------ write tensors (for debugging purposes only) ------
   if (verbosity == 3) call dbcsr_t_write_blocks(a_ijk, io_unit_dbcsr, output_unit)
   if (verbosity == 3) call dbcsr_t_write_blocks(b_iln, io_unit_dbcsr, output_unit)
   if (verbosity == 3) call dbcsr_t_write_blocks(c_no, io_unit_dbcsr, output_unit)

   if (contract_direct) then

! ------ d(i,j,l,m) = a(i,j,k) x a(l,m,k) ------

      ! performance measurement
      nflop_sum = 0
      call cpu_time(t0)

      ! contract_1: indices of first tensor to sum
      ! notcontract_1: all other indices of first tensor
      ! contract_2: indices of second tensor to sum (corresponding to contract_1)
      ! notcontract_2: all other indices of second tensor
      ! map_1: indices of result tensor corresponding to notcontract_1
      ! map_2: indices of result tensor corresponding to notcontract_2

      call dbcsr_t_contract(alpha=dbcsr_scalar(1.0_real64), tensor_1=a_ijk, tensor_2=a_lmk, &
                            beta=dbcsr_scalar(0.0_real64), tensor_3=d_ijlm, &
                            contract_1=[3], notcontract_1=[1, 2], &
                            contract_2=[3], notcontract_2=[1, 2], &
                            map_1=[1, 2], map_2=[3, 4], &
                            filter_eps=filter_eps, &
                            unit_nr=io_unit_dbcsr, log_verbose=verbosity >= 2, &
                            flop=nflop)
      nflop_sum = nflop_sum + nflop

! ------ e(j,m,n) = d(i,j,l,m) x b(i,l,n) ------

      ! note: tensor d was created with map1_2d, map2_2d arguments inconsistent with
      ! contract_1 and notcontract_1 since this tensor was created with the previous contraction in mind.
      ! in this case tensor will be redistributed to the correct layout automatically.
      call dbcsr_t_contract(dbcsr_scalar(1.0_real64), d_ijlm, b_iln, dbcsr_scalar(0.0_real64), e_jmn, &
                            contract_1=[1, 3], notcontract_1=[2, 4], &
                            contract_2=[1, 2], notcontract_2=[3], &
                            map_1=[1, 2], map_2=[3], &
                            filter_eps=filter_eps, &
                            unit_nr=io_unit_dbcsr, log_verbose=verbosity >= 2, &
                            flop=nflop)

      nflop_sum = nflop_sum + nflop

      ! free memory
      call dbcsr_t_clear(d_ijlm)

! ------ c(n,o) = c(n,o) + e(j,m,n) x f(j,m,o) ------

      ! summation to c is done by setting beta parameter to 1
      call dbcsr_t_contract(dbcsr_scalar(1.0_real64), e_jmn, f_jmo, dbcsr_scalar(1.0_real64), c_no, &
                            contract_1=[1, 2], notcontract_1=[3], &
                            contract_2=[1, 2], notcontract_2=[3], &
                            map_1=[1], map_2=[2], &
                            filter_eps=filter_eps, &
                            unit_nr=io_unit_dbcsr, log_verbose=verbosity >= 2, &
                            flop=nflop)

      nflop_sum = nflop_sum + nflop

      ! free memory
      call dbcsr_t_clear(e_jmn)

      call cpu_time(t1)

! ------ verify result by calculating checksum of c ------

      cs = dbcsr_t_checksum(c_no)
      if (io_unit > 0) write (io_unit, "(a, e20.13)") "checksum matrix c", cs

! ------ write contraction result (for debugging purposes only) ------
      if (verbosity == 3) call dbcsr_t_write_blocks(c_no, io_unit_dbcsr, output_unit)

! ------ output performance measurements ------
! useful to test strong scaling & overhead of batched contraction

      time = t1 - t0
      flop_rate = real(nflop_sum, real64)/(1.0e09_real64*time)

      if (io_unit > 0) then
         write (io_unit, "(a,t73,es8.2)") "performance: total number of flops:", real(nflop_sum*numnodes)
         write (io_unit, "(a,t66,f15.2)") "performance: total execution time:", time
         write (io_unit, "(a,t66,f15.2)") "performance: contraction flop rate (gflops / mpi rank):", flop_rate
      end if

   end if

   if (contract_batched) then

! ------ batched contraction ------
! reduce memory by contracting over batches (such that intermediate tensors are never fully held in memory)
! indices i,j,l,m are split into n batches each (these indices belong to largest tensor d[ijlm])

      ! performance measurement
      nflop_sum = 0
      call cpu_time(t0)

      call create_batches(blk_size_i, nbatch, start_batch_i, end_batch_i)
      call create_batches(blk_size_j, nbatch, start_batch_j, end_batch_j)
      call create_batches(blk_size_l, nbatch, start_batch_l, end_batch_l)
      call create_batches(blk_size_m, nbatch, start_batch_m, end_batch_m)

      call dbcsr_t_copy_matrix_to_tensor(c_matrix, c_no)

      ! for better performance (avoiding communications) call init routine on all tensors that appear
      ! in multiple contraction calls with the same bounds:
      call dbcsr_t_batched_contract_init(c_no)

      ! iterate over index batches
      do jbatch = 1, nbatch
         do mbatch = 1, nbatch
            do ibatch = 1, nbatch
               call dbcsr_t_batched_contract_init(a_ijk)
               do lbatch = 1, nbatch

! ------ d(i,j,l,m) = a(i,j,k) x a(l,m,k) ------

                  ! specify bounds corresponding to the contraction index sets
                  allocate (bounds_2(2, 2), bounds_3(2, 2))

                  ! bounds corresponding to notcontract_1 indices i,j
                  bounds_2(:, 1) = [start_batch_i(ibatch), end_batch_i(ibatch)]
                  bounds_2(:, 2) = [start_batch_j(jbatch), end_batch_j(jbatch)]

                  ! bounds corresponding to notcontract_2 indices l,m
                  bounds_3(:, 1) = [start_batch_l(lbatch), end_batch_l(lbatch)]
                  bounds_3(:, 2) = [start_batch_m(mbatch), end_batch_m(mbatch)]

                  call dbcsr_t_contract(dbcsr_scalar(1.0_real64), a_ijk, a_lmk, &
                                        dbcsr_scalar(0.0_real64), d_ijlm, &
                                        contract_1=[3], notcontract_1=[1, 2], &
                                        contract_2=[3], notcontract_2=[1, 2], &
                                        map_1=[1, 2], map_2=[3, 4], &
                                        bounds_2=bounds_2, &
                                        bounds_3=bounds_3, &
                                        filter_eps=filter_eps, &
                                        unit_nr=io_unit_dbcsr, &
                                        flop=nflop)
                  nflop_sum = nflop_sum + nflop
                  deallocate (bounds_2, bounds_3)

! ------ e(j,m,n) = d(i,j,l,m) x b(i,l,n) ------

                  allocate (bounds_1(2, 2), bounds_2(2, 2))

                  ! bounds corresponding to contract indices i,l
                  bounds_1(:, 1) = [start_batch_i(ibatch), end_batch_i(ibatch)]
                  bounds_1(:, 2) = [start_batch_l(lbatch), end_batch_l(lbatch)]

                  ! bounds corresponding to notcontract_1 indices j,m
                  bounds_2(:, 1) = [start_batch_j(jbatch), end_batch_j(jbatch)]
                  bounds_2(:, 2) = [start_batch_m(mbatch), end_batch_m(mbatch)]

                  ! note: we sum up contributions from batches i & l, thus beta parameter set to 1
                  call dbcsr_t_contract(dbcsr_scalar(1.0_real64), d_ijlm, b_iln, dbcsr_scalar(1.0_real64), e_jmn, &
                                        contract_1=[1, 3], notcontract_1=[2, 4], &
                                        contract_2=[1, 2], notcontract_2=[3], &
                                        map_1=[1, 2], map_2=[3], &
                                        bounds_1=bounds_1, bounds_2=bounds_2, &
                                        filter_eps=filter_eps, &
                                        unit_nr=io_unit_dbcsr, &
                                        flop=nflop)

                  nflop_sum = nflop_sum + nflop
                  deallocate (bounds_1, bounds_2)

                  ! free memory
                  call dbcsr_t_clear(d_ijlm)

               end do

               ! complete batched contraction of a
               call dbcsr_t_batched_contract_finalize(a_ijk)
            end do

! ------ c(n,o) = c(n,o) + e(j,m,n) x f(j,m,o) ------

            allocate (bounds_1(2, 2))

            ! bounds corresponding to contract indices j,m
            bounds_1(:, 1) = [start_batch_j(jbatch), end_batch_j(jbatch)]
            bounds_1(:, 2) = [start_batch_m(mbatch), end_batch_m(mbatch)]

            call dbcsr_t_contract(dbcsr_scalar(1.0_real64), e_jmn, f_jmo, dbcsr_scalar(1.0_real64), c_no, &
                                  contract_1=[1, 2], notcontract_1=[3], &
                                  contract_2=[1, 2], notcontract_2=[3], &
                                  map_1=[1], map_2=[2], &
                                  bounds_1=bounds_1, &
                                  filter_eps=filter_eps, &
                                  unit_nr=io_unit_dbcsr, &
                                  flop=nflop)

            nflop_sum = nflop_sum + nflop
            deallocate (bounds_1)

            ! free memory
            call dbcsr_t_clear(e_jmn)
         end do
      end do

      ! complete batched contraction of c
      call dbcsr_t_batched_contract_finalize(c_no)

      call cpu_time(t1)

! ------ verify result by calculating checksum of c ------
      cs = dbcsr_t_checksum(c_no)
      if (io_unit > 0) write (io_unit, "(a, e20.13)") "checksum matrix c", cs

! ------ output performance measurements ------
! useful to test strong scaling & overhead of batched contraction

      time = t1 - t0
      flop_rate = real(nflop_sum, real64)/(1.0e09_real64*time)

      if (io_unit > 0) then
         write (io_unit, "(a,t73,es8.2)") "performance (batched): total number of flops:", real(nflop_sum*numnodes)
         write (io_unit, "(a,t66,f15.2)") "performance (batched): total execution time:", time
         write (io_unit, "(a,t66,f15.2)") "performance (batched): contraction flop rate (gflops / mpi rank):", flop_rate
      end if

      deallocate (start_batch_i, start_batch_j, start_batch_l, start_batch_m, &
                  end_batch_i, end_batch_j, end_batch_l, end_batch_m)

   end if

! ------ copy tensor c to matrix c ------
   call dbcsr_t_copy_tensor_to_matrix(c_no, c_matrix)

! ------ cleanup ------

   call dbcsr_t_pgrid_destroy(pgrid_3d)
   call dbcsr_t_pgrid_destroy(pgrid_4d)

   call dbcsr_release(c_matrix)

   call dbcsr_t_destroy(c_no)
   call dbcsr_t_destroy(a_ijk)
   call dbcsr_t_destroy(e_jmn)
   call dbcsr_t_destroy(a_lmk)
   call dbcsr_t_destroy(b_iln)
   call dbcsr_t_destroy(f_jmo)
   call dbcsr_t_destroy(d_ijlm)

   deallocate (blk_size_i, blk_size_j, blk_size_k, blk_size_l, blk_size_m, blk_size_n, blk_size_o, &
               offset_i, offset_j, offset_k, offset_l, offset_n)

   call mpi_comm_free(group, ierr)
   if (ierr /= 0) stop "error in mpi_comm_free"

   ! finalize libdbcsr
   call dbcsr_finalize_lib()

   ! finalize mpi
   call mpi_finalize(ierr)
   if (ierr /= 0) stop "error in mpi_finalize"

contains

   subroutine random_blk_sizes(total_size, nblk, blk_sizes)
      ! random block sizes such that sum is equal to total_size
      integer, intent(in) :: total_size
      integer, intent(out) :: nblk
      integer, intent(out), allocatable :: blk_sizes(:)
      integer, allocatable :: tmp(:)
      integer :: mynode, ierr, blk_sum, bsize
      real :: rand

      call mpi_comm_rank(mpi_comm_world, mynode, ierr)
      if (ierr /= 0) stop "error in mpi_comm_rank"

      if (mynode == 0) then
         blk_sum = 0
         allocate (blk_sizes(0))
         nblk = 0
         do while (blk_sum < total_size)
            call random_number(rand)
            bsize = int(rand*max_bsize + 1)
            if (blk_sum + bsize > total_size) bsize = total_size - blk_sum
            blk_sum = blk_sum + bsize
            nblk = nblk + 1
            call move_alloc(blk_sizes, tmp)
            allocate (blk_sizes(nblk))
            blk_sizes(1:nblk - 1) = tmp; deallocate (tmp)
            blk_sizes(nblk) = bsize
         end do
      end if
      call mpi_bcast(nblk, 1, mpi_integer, 0, mpi_comm_world, ierr)
      if (ierr /= 0) stop "error in mpi_bcast"
      if (mynode /= 0) allocate (blk_sizes(nblk))
      call mpi_bcast(blk_sizes, nblk, mpi_integer, 0, mpi_comm_world, ierr)
      if (ierr /= 0) stop "error in mpi_bcast"

   end subroutine

   function block_minabsdiff(offset_1, offset_2, size_1, size_2)
      ! get minimum difference between row and column indices belonging to a block defined by its
      ! size and offset
      integer, intent(in) :: offset_1, offset_2, size_1, size_2
      integer :: block_minabsdiff
      integer, dimension(2) :: limits_1, limits_2

      limits_1 = offset_1 - 1 + [1, size_1]
      limits_2 = offset_2 - 1 + [1, size_2]

      if (limits_1(2) < limits_2(1)) then
         block_minabsdiff = limits_2(1) - limits_1(2)
      elseif (limits_2(2) < limits_1(1)) then
         block_minabsdiff = limits_1(1) - limits_2(2)
      else
         block_minabsdiff = 0
      end if

   end function

   subroutine create_batches(blk_sizes, nbatch, start_batch, end_batch)
      ! create tensor batches: split index at block boundaries such that each batch contains approximately
      ! the same number of tensor elements.
      integer, dimension(:), intent(in) :: blk_sizes
      integer, intent(in) :: nbatch
      integer, dimension(:), allocatable, intent(out) :: start_batch, end_batch
      integer :: nel, nel_batch, nblk, blk_sum, batch_sum, iblk
      integer, dimension(:), allocatable :: tmp

      nblk = size(blk_sizes)
      nel = sum(blk_sizes)
      nel_batch = nel/nbatch
      ibatch = 0
      blk_sum = 0; batch_sum = nel_batch
      allocate (end_batch(0:nbatch))
      allocate (start_batch(1:nbatch))
      end_batch(0) = 0
      do iblk = 1, nblk
         blk_sum = blk_sum + blk_sizes(iblk)
         if (blk_sum >= batch_sum) then
            ibatch = ibatch + 1
            end_batch(ibatch) = blk_sum
            start_batch(ibatch) = end_batch(ibatch - 1) + 1
            batch_sum = min(batch_sum + nel_batch, nel)
         end if
      end do

      call move_alloc(end_batch, tmp)
      allocate (end_batch(1:nbatch))
      end_batch(:) = tmp(1:)

   end subroutine

end program
