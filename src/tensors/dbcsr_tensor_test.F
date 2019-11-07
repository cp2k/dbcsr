!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tensor_test
   !! General methods for testing DBCSR tensors.

#:include "dbcsr_tensor.fypp"
#:set maxdim = maxrank
#:set ndims = range(2,maxdim+1)

   USE dbcsr_api, ONLY: ${uselist(dtype_float_param)}$
   USE dbcsr_tas_base, ONLY: dbcsr_tas_info
   USE dbcsr_tensor, ONLY: &
      dbcsr_t_copy, dbcsr_t_get_block, dbcsr_t_iterator_type, dbcsr_t_iterator_blocks_left, &
      dbcsr_t_iterator_next_block, dbcsr_t_iterator_start, dbcsr_t_iterator_stop, dbcsr_t_ndims, &
      dbcsr_t_reserve_blocks, dbcsr_t_get_stored_coordinates, dbcsr_t_put_block, &
      dbcsr_t_contract, dbcsr_t_inverse_order, dbcsr_t_dims
   USE dbcsr_tensor_block, ONLY: block_nd
   USE dbcsr_tensor_types, ONLY: dbcsr_t_create, &
                                 dbcsr_t_destroy, &
                                 dbcsr_t_type, &
                                 dbcsr_t_distribution_type, &
                                 dbcsr_t_distribution_destroy, &
                                 dims_tensor, &
                                 dbcsr_t_distribution_new, &
                                 dbcsr_t_nd_mp_comm, &
                                 dbcsr_t_get_data_type, &
                                 mp_environ_pgrid, &
                                 dbcsr_t_pgrid_type, &
                                 dbcsr_t_pgrid_create, &
                                 dbcsr_t_pgrid_destroy, &
                                 dbcsr_t_get_info
   USE dbcsr_tensor_io, ONLY: &
      dbcsr_t_write_blocks, dbcsr_t_write_block_indices
   USE dbcsr_kinds, ONLY: ${uselist(dtype_float_prec)}$, &
                          default_string_length, &
                          int_8
   USE dbcsr_mpiwrap, ONLY: mp_environ, &
                            mp_bcast, &
                            mp_comm_free, &
                            mp_comm_free, &
                            mp_sum, &
                            mp_cart_create
   USE dbcsr_allocate_wrap, ONLY: allocate_any
   USE dbcsr_tensor_index, ONLY: combine_index, &
                                 dbcsr_t_get_mapping_info
   USE dbcsr_tas_test, ONLY: dbcsr_tas_checksum
   USE dbcsr_data_types, ONLY: dbcsr_scalar_type
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tensor_test'

   PUBLIC :: &
      dbcsr_t_setup_test_tensor, &
      dbcsr_t_random_dist, &
      dbcsr_t_contract_test, &
      dbcsr_t_test_formats, &
      dbcsr_t_checksum

   INTERFACE repl_dense_array_to_dist_sparse_tensor
#:for dparam, dtype, dsuffix in dtype_float_list
#:for ndim in ndims
      MODULE PROCEDURE repl_dense_${ndim}$d_array_to_dist_sparse_tensor_${dsuffix}$
#:endfor
#:endfor
   END INTERFACE

   INTERFACE write_array
#:for ndim in ndims
      MODULE PROCEDURE write_${ndim}$d_array
#:endfor
   END INTERFACE

   INTERFACE dist_sparse_tensor_to_repl_dense_array
#:for dparam, dtype, dsuffix in dtype_float_list
#:for ndim in ndims
      MODULE PROCEDURE dist_sparse_tensor_to_repl_dense_${ndim}$d_array_${dsuffix}$
#:endfor
#:endfor
   END INTERFACE
CONTAINS

   FUNCTION dbcsr_t_equal(tensor1, tensor2)
      !! check if two (arbitrarily mapped and distributed) tensors are equal.
      TYPE(dbcsr_t_type), INTENT(INOUT)          :: tensor1, tensor2
      LOGICAL                                    :: dbcsr_t_equal

      INTEGER                                    :: blk
      TYPE(dbcsr_t_type)                         :: tensor2_tmp
      TYPE(dbcsr_t_iterator_type)                :: iter
      TYPE(block_nd)                             :: blk_data1, blk_data2
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor1)) :: blk_size, ind_nd
      LOGICAL :: found

      ! create a copy of tensor2 that has exact same data format as tensor1
      CALL dbcsr_t_create(tensor1, tensor2_tmp)

      CALL dbcsr_t_reserve_blocks(tensor1, tensor2_tmp)
      CALL dbcsr_t_copy(tensor2, tensor2_tmp)

      dbcsr_t_equal = .TRUE.

      CALL dbcsr_t_iterator_start(iter, tensor1)

      DO WHILE (dbcsr_t_iterator_blocks_left(iter))
         CALL dbcsr_t_iterator_next_block(iter, ind_nd, blk, blk_size=blk_size)
         CALL dbcsr_t_get_block(tensor1, ind_nd, blk_data1, found)
         DBCSR_ASSERT(found)
         CALL dbcsr_t_get_block(tensor2_tmp, ind_nd, blk_data2, found)
         DBCSR_ASSERT(found)

         IF (.NOT. blocks_equal(blk_data1, blk_data2)) THEN
            dbcsr_t_equal = .FALSE.
         ENDIF
      ENDDO

      CALL dbcsr_t_iterator_stop(iter)

      CALL dbcsr_t_destroy(tensor2_tmp)
   END FUNCTION

   PURE FUNCTION blocks_equal(block1, block2)
      !! check if two blocks are equal
      TYPE(block_nd), INTENT(IN) :: block1, block2
      LOGICAL                    :: blocks_equal

      SELECT CASE (block1%data_type)
#:for dprec, dparam, dtype, dsuffix in dtype_float_list_prec
      CASE (${dparam}$)
         blocks_equal = MAXVAL(ABS(block1%${dsuffix}$%blk - block2%${dsuffix}$%blk)) .LT. 1.0E-12_${dprec}$
#:endfor
      END SELECT

   END FUNCTION

   PURE FUNCTION factorial(n)
      !! Compute factorial
      INTEGER, INTENT(IN) :: n
      INTEGER             :: k
      INTEGER             :: factorial
      factorial = PRODUCT((/(k, k=1, n)/))
   END FUNCTION

   SUBROUTINE permute(n, p)
      !! Compute all permutations p of (1, 2, ..., n)
      INTEGER, INTENT(IN)                              :: n
      INTEGER                                          :: i, c
      INTEGER, DIMENSION(n)                            :: pp
      INTEGER, DIMENSION(n, factorial(n)), INTENT(OUT) :: p

      pp = [(i, i=1, n)]
      c = 1
      CALL perm(1)
   CONTAINS
      RECURSIVE SUBROUTINE perm(i)
         INTEGER, INTENT(IN) :: i
         INTEGER :: j, t
         IF (i == n) THEN
            p(:, c) = pp(:)
            c = c + 1
         ELSE
            DO j = i, n
               t = pp(i)
               pp(i) = pp(j)
               pp(j) = t
               call perm(i + 1)
               t = pp(i)
               pp(i) = pp(j)
               pp(j) = t
            END DO
         END IF
      END SUBROUTINE
   END SUBROUTINE

   SUBROUTINE dbcsr_t_test_formats(ndims, mp_comm, unit_nr, verbose, &
                                   ${varlist("blk_size")}$, &
                                   ${varlist("blk_ind")}$)
      !! Test equivalence of all tensor formats, using a random distribution.
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL :: ${varlist("blk_size")}$
         !! block sizes along respective dimension
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL :: ${varlist("blk_ind")}$
         !! index along respective dimension of non-zero blocks
      INTEGER, INTENT(IN)                         :: ndims
         !! tensor rank
      INTEGER, INTENT(IN)                         :: unit_nr
         !! output unit, needs to be a valid unit number on all mpi ranks
      LOGICAL, INTENT(IN)                         :: verbose
         !! if .TRUE., print all tensor blocks
      INTEGER, INTENT(IN)                         :: mp_comm
      TYPE(dbcsr_t_distribution_type)             :: dist1, dist2
      TYPE(dbcsr_t_type)                          :: tensor1, tensor2
      INTEGER                                     :: isep, iblk
      INTEGER, DIMENSION(:), ALLOCATABLE          :: ${varlist("dist1")}$, &
                                                     ${varlist("dist2")}$
      INTEGER                                     :: nblks, imap
      INTEGER, DIMENSION(ndims)                   :: pdims, myploc
      LOGICAL                                     :: eql
      INTEGER                                     :: iperm, idist, icount
      INTEGER, DIMENSION(:), ALLOCATABLE          :: map1, map2, map1_ref, map2_ref
      INTEGER, DIMENSION(ndims, factorial(ndims)) :: perm
      INTEGER                                     :: io_unit
      INTEGER                                     :: mynode, numnodes
      TYPE(dbcsr_t_pgrid_type)                    :: comm_nd
      CHARACTER(LEN=default_string_length)        :: tensor_name

      ! Process grid
      pdims(:) = 0
      CALL dbcsr_t_pgrid_create(mp_comm, pdims, comm_nd)
      CALL mp_environ(numnodes, mynode, mp_comm)

      io_unit = 0
      IF (mynode .EQ. 0) io_unit = unit_nr

      CALL permute(ndims, perm)
      CALL allocate_any(map1_ref, source=perm(1:ndims/2, 1))
      CALL allocate_any(map2_ref, source=perm(ndims/2 + 1:ndims, 1))

      IF (io_unit > 0) THEN
         WRITE (io_unit, *)
         WRITE (io_unit, '(A)') repeat("-", 80)
         WRITE (io_unit, '(A,1X,I1)') "Testing matrix representations of tensor rank", ndims
         WRITE (io_unit, '(A)') repeat("-", 80)
         WRITE (io_unit, '(A)') "Block sizes:"

#:for dim in range(1, maxdim+1)
         IF (ndims >= ${dim}$) THEN
            WRITE (io_unit, '(T4,A,1X,I1,A,1X)', advance='no') 'Dim', ${dim}$, ':'
            DO iblk = 1, SIZE(blk_size_${dim}$)
               WRITE (io_unit, '(I2,1X)', advance='no') blk_size_${dim}$ (iblk)
            ENDDO
            WRITE (io_unit, *)
         ENDIF
#:endfor

         WRITE (io_unit, '(A)') "Non-zero blocks:"
         DO iblk = 1, SIZE(blk_ind_1)
#:for ndim in ndims
            IF (ndims == ${ndim}$) THEN
               WRITE (io_unit, '(T4,A, I3, A, ${ndim}$I3, 1X, A)') &
                  'Block', iblk, ': (', ${varlist("blk_ind", nmax=ndim, suffix='(iblk)')}$, ')'
            ENDIF
#:endfor
         ENDDO

         WRITE (io_unit, *)
         WRITE (io_unit, '(A,1X)', advance='no') "Reference map:"
         WRITE (io_unit, '(A1,1X)', advance='no') "("
         DO imap = 1, SIZE(map1_ref)
            WRITE (io_unit, '(I1,1X)', advance='no') map1_ref(imap)
         ENDDO
         WRITE (io_unit, '(A1,1X)', advance='no') "|"
         DO imap = 1, SIZE(map2_ref)
            WRITE (io_unit, '(I1,1X)', advance='no') map2_ref(imap)
         ENDDO
         WRITE (io_unit, '(A1)') ")"

      ENDIF

      icount = 0
      DO iperm = 1, factorial(ndims)
         DO isep = 1, ndims - 1
            icount = icount + 1

            CALL allocate_any(map1, source=perm(1:isep, iperm))
            CALL allocate_any(map2, source=perm(isep + 1:ndims, iperm))

            CALL mp_environ(numnodes, mynode, mp_comm)
            CALL mp_environ_pgrid(comm_nd, pdims, myploc)

#:for dim in range(1, maxdim+1)
            IF (${dim}$ <= ndims) THEN
               nblks = SIZE(blk_size_${dim}$)
               CALL dbcsr_t_random_dist(dist1_${dim}$, nblks, pdims(${dim}$), mp_comm)
               CALL dbcsr_t_random_dist(dist2_${dim}$, nblks, pdims(${dim}$), mp_comm)
            ENDIF
#:endfor

            WRITE (tensor_name, '(A,1X,I3,1X)') "Test", icount

            IF (io_unit > 0) THEN
               WRITE (io_unit, *)
               WRITE (io_unit, '(A,A,1X)', advance='no') TRIM(tensor_name), ':'
               WRITE (io_unit, '(A1,1X)', advance='no') "("
               DO imap = 1, SIZE(map1)
                  WRITE (io_unit, '(I1,1X)', advance='no') map1(imap)
               ENDDO
               WRITE (io_unit, '(A1,1X)', advance='no') "|"
               DO imap = 1, SIZE(map2)
                  WRITE (io_unit, '(I1,1X)', advance='no') map2(imap)
               ENDDO
               WRITE (io_unit, '(A1)') ")"

               WRITE (io_unit, '(T4,A)') "Reference distribution:"
#:for dim in range(1, maxdim+1)
               IF (${dim}$ <= ndims) THEN
                  WRITE (io_unit, '(T7,A,1X)', advance='no') "Dist vec ${dim}$:"
                  DO idist = 1, SIZE(dist2_${dim}$)
                     WRITE (io_unit, '(I2,1X)', advance='no') dist2_${dim}$ (idist)
                  ENDDO
                  WRITE (io_unit, *)
               ENDIF
#:endfor

               WRITE (io_unit, '(T4,A)') "Test distribution:"
#:for dim in range(1, maxdim+1)
               IF (${dim}$ <= ndims) THEN
                  WRITE (io_unit, '(T7,A,1X)', advance='no') "Dist vec ${dim}$:"
                  DO idist = 1, SIZE(dist2_${dim}$)
                     WRITE (io_unit, '(I2,1X)', advance='no') dist1_${dim}$ (idist)
                  ENDDO
                  WRITE (io_unit, *)
               ENDIF
#:endfor
            ENDIF

#:for ndim in ndims
            IF (ndims == ${ndim}$) THEN
               CALL dbcsr_t_distribution_new(dist2, comm_nd, map1_ref, map2_ref, ${varlist("dist2", nmax=ndim)}$)
               CALL dbcsr_t_create(tensor2, "Ref", dist2, map1_ref, map2_ref, &
                                   dbcsr_type_real_8, ${varlist("blk_size", nmax=ndim)}$)
               CALL dbcsr_t_setup_test_tensor(tensor2, comm_nd%mp_comm_2d, .TRUE., ${varlist("blk_ind", nmax=ndim)}$)
            ENDIF
#:endfor

            IF (verbose) CALL dbcsr_t_write_blocks(tensor2, io_unit, unit_nr)

#:for ndim in ndims
            IF (ndims == ${ndim}$) THEN
               CALL dbcsr_t_distribution_new(dist1, comm_nd, map1, map2, ${varlist("dist1", nmax=ndim)}$)
               CALL dbcsr_t_create(tensor1, tensor_name, dist1, map1, map2, &
                                   dbcsr_type_real_8, ${varlist("blk_size", nmax=ndim)}$)
               CALL dbcsr_t_setup_test_tensor(tensor1, comm_nd%mp_comm_2d, .TRUE., ${varlist("blk_ind", nmax=ndim)}$)
            ENDIF
#:endfor

            IF (verbose) CALL dbcsr_t_write_blocks(tensor1, io_unit, unit_nr)

            eql = dbcsr_t_equal(tensor1, tensor2)

            IF (.NOT. eql) THEN
               IF (io_unit > 0) WRITE (io_unit, '(A,1X,A)') TRIM(tensor_name), 'Test failed!'
               DBCSR_ABORT('')
            ELSE
               IF (io_unit > 0) WRITE (io_unit, '(A,1X,A)') TRIM(tensor_name), 'Test passed!'
            ENDIF
            DEALLOCATE (map1, map2)

            CALL dbcsr_t_destroy(tensor1)
            CALL dbcsr_t_distribution_destroy(dist1)

            CALL dbcsr_t_destroy(tensor2)
            CALL dbcsr_t_distribution_destroy(dist2)

#:for dim in range(1, maxdim+1)
            IF (${dim}$ <= ndims) THEN
               DEALLOCATE (dist1_${dim}$, dist2_${dim}$)
            ENDIF
#:endfor

         ENDDO
      ENDDO
      CALL dbcsr_t_pgrid_destroy(comm_nd)
   END SUBROUTINE

   SUBROUTINE dbcsr_t_random_dist(dist_array, dist_size, nbins, mp_comm)
      !! Create test distribution

      INTEGER, ALLOCATABLE, DIMENSION(:), INTENT(OUT) :: dist_array
      INTEGER, INTENT(IN)                             :: dist_size, nbins, mp_comm
      REAL, DIMENSION(dist_size)                      :: rn
      INTEGER, DIMENSION(dist_size)                   :: rn_int
      INTEGER                                         :: numnodes, mynode

      CALL mp_environ(numnodes, mynode, mp_comm)

      IF (mynode .EQ. 0) THEN
         CALL RANDOM_NUMBER(rn)
      ENDIF
      CALL mp_bcast(rn, 0, mp_comm)

      rn_int = FLOOR(rn*nbins)
      CALL allocate_any(dist_array, source=rn_int)

   END SUBROUTINE dbcsr_t_random_dist

   SUBROUTINE dbcsr_t_setup_test_tensor(tensor, mp_comm, enumerate, ${varlist("blk_ind")}$)
      !! Allocate and fill test tensor - entries are enumerated by their index s.t. they only depend
      !! on global properties of the tensor but not on distribution, matrix representation, etc.
      TYPE(dbcsr_t_type), INTENT(INOUT)                  :: tensor
      INTEGER, INTENT(IN)                                :: mp_comm
         !! communicator
      LOGICAL, INTENT(IN)                                :: enumerate
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL        :: ${varlist("blk_ind")}$
         !! index along respective dimension of non-zero blocks
      INTEGER                                            :: blk, numnodes, mynode

      INTEGER                                            :: i, ib, my_nblks_alloc, nblks_alloc, proc
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: ${varlist("my_blk_ind")}$
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))          :: blk_index, blk_offset, blk_size, &
                                                            tensor_dims
      INTEGER, DIMENSION(:, :), ALLOCATABLE               :: ind_nd
#:for ndim in ndims
      REAL(KIND=real_8), ALLOCATABLE, &
         DIMENSION(${shape_colon(ndim)}$)                :: blk_values_${ndim}$
#:endfor
      TYPE(dbcsr_t_iterator_type)                        :: iterator

      nblks_alloc = SIZE(blk_ind_1)
      CALL mp_environ(numnodes, mynode, mp_comm)

      ALLOCATE (ind_nd(nblks_alloc, dbcsr_t_ndims(tensor)))
      my_nblks_alloc = 0
      DO ib = 1, nblks_alloc
#:for ndim in ndims
         IF (dbcsr_t_ndims(tensor) == ${ndim}$) THEN
            ind_nd(ib, :) = [${varlist("blk_ind", nmax=ndim, suffix="(ib)")}$]
         ENDIF
#:endfor
         CALL dbcsr_t_get_stored_coordinates(tensor, ind_nd(ib, :), proc)
         IF (proc == mynode) THEN
            my_nblks_alloc = my_nblks_alloc + 1
         ENDIF
      ENDDO

#:for dim in range(1, maxdim+1)
      IF (dbcsr_t_ndims(tensor) >= ${dim}$) THEN
         ALLOCATE (my_blk_ind_${dim}$ (my_nblks_alloc))
      ENDIF
#:endfor

      i = 0
      DO ib = 1, nblks_alloc
         CALL dbcsr_t_get_stored_coordinates(tensor, ind_nd(ib, :), proc)
         IF (proc == mynode) THEN
            i = i + 1
#:for dim in range(1, maxdim+1)
            IF (dbcsr_t_ndims(tensor) >= ${dim}$) THEN
               my_blk_ind_${dim}$ (i) = blk_ind_${dim}$ (ib)
            ENDIF
#:endfor
         ENDIF
      ENDDO

#:for ndim in ndims
      IF (dbcsr_t_ndims(tensor) == ${ndim}$) THEN
         CALL dbcsr_t_reserve_blocks(tensor, ${varlist("my_blk_ind", nmax=ndim)}$)
      ENDIF
#:endfor

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, blk_index, blk, blk_size=blk_size, blk_offset=blk_offset)

#:for ndim in ndims
         IF (dbcsr_t_ndims(tensor) == ${ndim}$) THEN
            CALL allocate_any(blk_values_${ndim}$, shape_spec=blk_size)
            CALL dims_tensor(tensor, tensor_dims)
            IF (enumerate) THEN
               CALL enumerate_block_elements(blk_size, blk_offset, tensor_dims, blk_${ndim}$=blk_values_${ndim}$)
            ELSE
               CALL random_number(blk_values_${ndim}$)
            ENDIF
            CALL dbcsr_t_put_block(tensor, blk_index, blk_size, blk_values_${ndim}$)
            DEALLOCATE (blk_values_${ndim}$)
         ENDIF
#:endfor
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)

   END SUBROUTINE

   SUBROUTINE enumerate_block_elements(blk_size, blk_offset, tensor_size, ${varlist("blk", nmin=2)}$)
      !! Enumerate tensor entries in block
      !! \blk_2 block values for 2 dimensions
      !! \blk_3 block values for 3 dimensions

      INTEGER, DIMENSION(:), INTENT(IN)                  :: blk_size, blk_offset, tensor_size
         !! size of block
         !! block offset (indices of first element)
         !! global tensor sizes
#:for ndim in ndims
      REAL(KIND=real_8), DIMENSION(${shape_colon(ndim)}$), &
         OPTIONAL, INTENT(OUT)                           :: blk_${ndim}$
#:endfor
      INTEGER                                            :: ndim
      INTEGER, DIMENSION(SIZE(blk_size))                 :: arr_ind, tens_ind
      INTEGER                                            :: ${varlist("i")}$

      ndim = SIZE(tensor_size)

#:for ndim in ndims
      IF (ndim == ${ndim}$) THEN
#:for idim in range(ndim,0,-1)
         DO i_${idim}$ = 1, blk_size(${idim}$)
#:endfor
            arr_ind(:) = [${varlist("i", nmax=ndim)}$]
            tens_ind(:) = arr_ind(:) + blk_offset(:) - 1
            blk_${ndim}$ (${arrlist("arr_ind", nmax=ndim)}$) = combine_index(tens_ind, tensor_size)
#:for idim in range(ndim,0,-1)
         ENDDO
#:endfor
      ENDIF
#:endfor

   END SUBROUTINE

#:for dprec, dparam, dtype, dsuffix in dtype_float_list_prec
#:for ndim in ndims
   SUBROUTINE dist_sparse_tensor_to_repl_dense_${ndim}$d_array_${dsuffix}$ (tensor, array)
      !! Transform a distributed sparse tensor to a replicated dense array. This is only useful for
      !! testing tensor contraction by matrix multiplication of dense arrays.

      TYPE(dbcsr_t_type), INTENT(INOUT)                          :: tensor
      ${dtype}$, ALLOCATABLE, DIMENSION(${shape_colon(ndim)}$), &
         INTENT(OUT)                                             :: array
      ${dtype}$, ALLOCATABLE, DIMENSION(${shape_colon(ndim)}$)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end
      LOGICAL                                                    :: found

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. ${ndim}$)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)
      CALL allocate_any(array, shape_spec=dims_nd)
      array(${shape_colon(ndim)}$) = 0.0_${dprec}$

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL dbcsr_t_get_block(tensor, ind_nd, block, found)
         DBCSR_ASSERT(found)

         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         array(${", ".join(["blk_start("+str(idim)+"):blk_end("+str(idim)+")" for idim in range(1, ndim + 1)])}$) = &
            block(${shape_colon(ndim)}$)

         DEALLOCATE (block)
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)
      CALL mp_sum(array, tensor%pgrid%mp_comm_2d)

   END SUBROUTINE
#:endfor
#:endfor

#:for dprec, dparam, dtype, dsuffix in dtype_float_list_prec
#:for ndim in ndims
   SUBROUTINE repl_dense_${ndim}$d_array_to_dist_sparse_tensor_${dsuffix}$ (tensor, array)
      TYPE(dbcsr_t_type), INTENT(INOUT)                             :: tensor
      ${dtype}$, ALLOCATABLE, DIMENSION(${shape_colon(ndim)}$), &
         INTENT(INOUT)                                           :: array
      ${dtype}$, ALLOCATABLE, DIMENSION(${shape_colon(ndim)}$)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. ${ndim}$)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL allocate_any(block, shape_spec=blk_size)
         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         block(${shape_colon(ndim)}$) = &
            array(${", ".join(["blk_start("+str(idim)+"):blk_end("+str(idim)+")" for idim in range(1, ndim + 1)])}$)
         CALL dbcsr_t_put_block(tensor, ind_nd, blk_size, block)
         DEALLOCATE (block)
      ENDDO

   END SUBROUTINE
#:endfor
#:endfor

   SUBROUTINE dbcsr_t_contract_test(alpha, tensor_1, tensor_2, beta, tensor_3, &
                                    contract_1, notcontract_1, &
                                    contract_2, notcontract_2, &
                                    map_1, map_2, &
                                    unit_nr, &
                                    bounds_1, bounds_2, bounds_3, &
                                    log_verbose, write_int)
      !! test tensor contraction
      !! @note for testing/debugging, simply replace a call to dbcsr_t_contract with a call to this routine
      !! @endnote

      TYPE(dbcsr_scalar_type), INTENT(IN) :: alpha
      TYPE(dbcsr_t_type), INTENT(INOUT)    :: tensor_1, tensor_2, tensor_3
      TYPE(dbcsr_scalar_type), INTENT(IN) :: beta
      INTEGER, DIMENSION(:), INTENT(IN)    :: contract_1, contract_2, &
                                              notcontract_1, notcontract_2, &
                                              map_1, map_2
      INTEGER, INTENT(IN)                  :: unit_nr
      INTEGER, DIMENSION(2, SIZE(contract_1)), &
         OPTIONAL                          :: bounds_1
      INTEGER, DIMENSION(2, SIZE(notcontract_1)), &
         OPTIONAL                          :: bounds_2
      INTEGER, DIMENSION(2, SIZE(notcontract_2)), &
         OPTIONAL                          :: bounds_3
      LOGICAL, INTENT(IN), OPTIONAL        :: log_verbose
      LOGICAL, INTENT(IN), OPTIONAL        :: write_int
      INTEGER                              :: io_unit, mynode, numnodes, mp_comm
      INTEGER, DIMENSION(:), ALLOCATABLE   :: size_1, size_2, size_3, &
                                              order_t1, order_t2, order_t3
      INTEGER, DIMENSION(2, dbcsr_t_ndims(tensor_1)) :: bounds_t1
      INTEGER, DIMENSION(2, dbcsr_t_ndims(tensor_2)) :: bounds_t2

#:for ndim in ndims
      REAL(KIND=real_8), ALLOCATABLE, &
         DIMENSION(${shape_colon(ndim)}$) :: array_1_${ndim}$d, &
                                             array_2_${ndim}$d, &
                                             array_3_${ndim}$d, &
                                             array_1_${ndim}$d_full, &
                                             array_2_${ndim}$d_full, &
                                             array_3_0_${ndim}$d, &
                                             array_1_rs${ndim}$d, &
                                             array_2_rs${ndim}$d, &
                                             array_3_rs${ndim}$d, &
                                             array_3_0_rs${ndim}$d
#:endfor
      REAL(KIND=real_8), ALLOCATABLE, &
         DIMENSION(:, :)                   :: array_1_mm, &
                                              array_2_mm, &
                                              array_3_mm, &
                                              array_3_test_mm
      LOGICAL                             :: eql, notzero
      LOGICAL, PARAMETER                  :: debug = .FALSE.
      REAL(KIND=real_8)                   :: cs_1, cs_2, cs_3, eql_diff
      LOGICAL                             :: do_crop_1, do_crop_2

      mp_comm = tensor_1%pgrid%mp_comm_2d
      CALL mp_environ(numnodes, mynode, mp_comm)
      io_unit = 0
      IF (mynode .EQ. 0) io_unit = unit_nr

      cs_1 = dbcsr_t_checksum(tensor_1)
      cs_2 = dbcsr_t_checksum(tensor_2)
      cs_3 = dbcsr_t_checksum(tensor_3)

      IF (io_unit > 0) THEN
         WRITE (io_unit, *)
         WRITE (io_unit, '(A)') repeat("-", 80)
         WRITE (io_unit, '(A,1X,A,1X,A,1X,A,1X,A,1X,A)') "Testing tensor contraction", &
            TRIM(tensor_1%name), "x", TRIM(tensor_2%name), "=", TRIM(tensor_3%name)
         WRITE (io_unit, '(A)') repeat("-", 80)
      ENDIF

      IF (debug) THEN
         IF (io_unit > 0) THEN
            WRITE (io_unit, *) "checksum ", TRIM(tensor_1%name), cs_1
            WRITE (io_unit, *) "checksum ", TRIM(tensor_2%name), cs_2
            WRITE (io_unit, *) "checksum ", TRIM(tensor_3%name), cs_3
         ENDIF
      ENDIF

      IF (debug) THEN
         CALL dbcsr_t_write_block_indices(tensor_1, io_unit, unit_nr)
         CALL dbcsr_t_write_blocks(tensor_1, io_unit, unit_nr, write_int)
      ENDIF

      SELECT CASE (dbcsr_t_ndims(tensor_3))
#:for ndim in ndims
      CASE (${ndim}$)
         CALL dist_sparse_tensor_to_repl_dense_array(tensor_3, array_3_0_${ndim}$d)
#:endfor
      END SELECT

      CALL dbcsr_t_contract(alpha, tensor_1, tensor_2, beta, tensor_3, &
                            contract_1, notcontract_1, &
                            contract_2, notcontract_2, &
                            map_1, map_2, &
                            bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                            filter_eps=1.0E-12_real_8, &
                            unit_nr=io_unit, log_verbose=log_verbose)

      cs_3 = dbcsr_t_checksum(tensor_3)

      IF (debug) THEN
         IF (io_unit > 0) THEN
            WRITE (io_unit, *) "checksum ", TRIM(tensor_3%name), cs_3
         ENDIF
      ENDIF

      do_crop_1 = .FALSE.; do_crop_2 = .FALSE.!; do_crop_3 = .FALSE.

      ! crop tensor as first step
      bounds_t1(1, :) = 1
      CALL dbcsr_t_get_info(tensor_1, nfull_total=bounds_t1(2, :))

      bounds_t2(1, :) = 1
      CALL dbcsr_t_get_info(tensor_2, nfull_total=bounds_t2(2, :))

      IF (PRESENT(bounds_1)) THEN
         bounds_t1(:, contract_1) = bounds_1
         do_crop_1 = .TRUE.
         bounds_t2(:, contract_2) = bounds_1
         do_crop_2 = .TRUE.
      ENDIF

      IF (PRESENT(bounds_2)) THEN
         bounds_t1(:, notcontract_1) = bounds_2
         do_crop_1 = .TRUE.
      ENDIF

      IF (PRESENT(bounds_3)) THEN
         bounds_t2(:, notcontract_2) = bounds_3
         do_crop_2 = .TRUE.
      ENDIF

      ! Convert tensors to simple multidimensional arrays
#:for i in range(1,4)
      SELECT CASE (dbcsr_t_ndims(tensor_${i}$))
#:for ndim in ndims
      CASE (${ndim}$)
#:if i < 3
         CALL dist_sparse_tensor_to_repl_dense_array(tensor_${i}$, array_${i}$_${ndim}$d_full)
         CALL allocate_any(array_${i}$_${ndim}$d, shape_spec=SHAPE(array_${i}$_${ndim}$d_full))
         array_${i}$_${ndim}$d = 0.0_real_8
         array_${i}$_${ndim}$d(${", ".join(["bounds_t" + str(i) + "(1, " + str(idim) + "):bounds_t" + str(i) + "(2, " + str(idim) + ")" for idim in range(1, ndim+1)])}$) = &
         array_${i}$_${ndim}$d_full(${", ".join(["bounds_t" + str(i) + "(1, " + str(idim) + "):bounds_t" + str(i) + "(2, " + str(idim) + ")" for idim in range(1, ndim+1)])}$)
#:else
         CALL dist_sparse_tensor_to_repl_dense_array(tensor_${i}$, array_${i}$_${ndim}$d)
#:endif

#:endfor
      END SELECT
#:endfor

      ! Get array sizes

#:for i in range(1,4)
      SELECT CASE (dbcsr_t_ndims(tensor_${i}$))
#:for ndim in ndims
      CASE (${ndim}$)
         CALL allocate_any(size_${i}$, source=SHAPE(array_${i}$_${ndim}$d))

#:endfor
      END SELECT
#:endfor

#:for i in range(1,4)
      ALLOCATE (order_t${i}$ (dbcsr_t_ndims(tensor_${i}$)))
#:endfor

      ASSOCIATE (map_t1_1=>notcontract_1, map_t1_2=>contract_1, &
                 map_t2_1=>notcontract_2, map_t2_2=>contract_2, &
                 map_t3_1=>map_1, map_t3_2=>map_2)

#:for i in range(1,4)
         order_t${i}$ (:) = dbcsr_t_inverse_order([map_t${i}$_1, map_t${i}$_2])

         SELECT CASE (dbcsr_t_ndims(tensor_${i}$))
#:for ndim in ndims
         CASE (${ndim}$)
            CALL allocate_any(array_${i}$_rs${ndim}$d, source=array_${i}$_${ndim}$d, order=order_t${i}$)
            CALL allocate_any(array_${i}$_mm, sizes_2d(size_${i}$, map_t${i}$_1, map_t${i}$_2))
            array_${i}$_mm(:, :) = RESHAPE(array_${i}$_rs${ndim}$d, SHAPE(array_${i}$_mm))
#:endfor
         END SELECT
#:endfor

         SELECT CASE (dbcsr_t_ndims(tensor_3))
#:for ndim in ndims
         CASE (${ndim}$)
            CALL allocate_any(array_3_0_rs${ndim}$d, source=array_3_0_${ndim}$d, order=order_t3)
            CALL allocate_any(array_3_test_mm, sizes_2d(size_3, map_t3_1, map_t3_2))
            array_3_test_mm(:, :) = RESHAPE(array_3_0_rs${ndim}$d, SHAPE(array_3_mm))
#:endfor
         END SELECT

         array_3_test_mm(:, :) = beta%r_dp*array_3_test_mm(:, :) + alpha%r_dp*MATMUL(array_1_mm, transpose(array_2_mm))

      END ASSOCIATE

      eql_diff = MAXVAL(ABS(array_3_test_mm(:, :) - array_3_mm(:, :)))
      notzero = MAXVAL(ABS(array_3_test_mm(:, :))) .GT. 1.0E-12_${dprec}$

      eql = eql_diff .LT. 1.0E-11_${dprec}$

      IF (.NOT. eql .OR. .NOT. notzero) THEN
         IF (io_unit > 0) WRITE (io_unit, *) 'Test failed!', eql_diff
         DBCSR_ABORT('')
      ELSE
         IF (io_unit > 0) WRITE (io_unit, *) 'Test passed!', eql_diff
      ENDIF

   END SUBROUTINE

#:for ndim in ndims
   SUBROUTINE write_${ndim}$d_array(array, unit_nr, array_compare)
      !! Write nd array
      REAL(KIND=real_8), DIMENSION(${shape_colon(ndim)}$), INTENT(IN) :: array
      INTEGER, INTENT(IN)                                             :: unit_nr
      REAL(KIND=real_8), DIMENSION(${shape_colon(ndim)}$), INTENT(IN), OPTIONAL :: array_compare
      INTEGER                                            :: ${varlist("i", nmax=ndim)}$
#:for idim in range(ndim,0,-1)
      DO i_${idim}$ = 1, SIZE(array, ${idim}$)
#:endfor
         IF (INT(array(${varlist("i", nmax=ndim)}$), KIND=int_8) .GT. 0) THEN
            WRITE (unit_nr, '(${ndim+1}$I20)', advance='no') ${varlist("i", nmax=ndim)}$, INT(array(${varlist("i", nmax=ndim)}$), &
                                                                                              KIND=int_8)
            IF (PRESENT(array_compare)) THEN
               WRITE (unit_nr, '(I20)', advance='no') INT(array_compare(${varlist("i", nmax=ndim)}$), KIND=int_8)
            ENDIF
            WRITE (unit_nr, *)
         ENDIF
#:for idim in range(ndim,0,-1)
      ENDDO
#:endfor
   END SUBROUTINE
#:endfor

   FUNCTION sizes_2d(nd_sizes, map1, map2)
      !! mapped sizes in 2d
      INTEGER, DIMENSION(:), INTENT(IN) :: nd_sizes, map1, map2
      INTEGER, DIMENSION(2)             :: sizes_2d
      sizes_2d(1) = PRODUCT(nd_sizes(map1))
      sizes_2d(2) = PRODUCT(nd_sizes(map2))
   END FUNCTION

   FUNCTION dbcsr_t_checksum(tensor, local, pos)
      !! checksum of a tensor consistent with dbcsr_checksum
      TYPE(dbcsr_t_type), INTENT(IN) :: tensor
      REAL(KIND=real_8) :: dbcsr_t_checksum
      LOGICAL, INTENT(IN), OPTIONAL     :: local, pos
      dbcsr_t_checksum = dbcsr_tas_checksum(tensor%matrix_rep, local, pos)
   END FUNCTION

END MODULE
