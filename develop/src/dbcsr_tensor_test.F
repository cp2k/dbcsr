# 1 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F" 1
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

# 1 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp" 1
# 9 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp"

# 66 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp"
# 14 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F" 2
# 15 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 16 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

   USE dbcsr_api, ONLY: dbcsr_type_complex_4, dbcsr_type_complex_8, dbcsr_type_real_8, dbcsr_type_real_4
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
   USE dbcsr_kinds, ONLY: real_8, real_4, &
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
# 70 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 71 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE repl_dense_2d_array_to_dist_sparse_tensor_r_dp
# 71 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE repl_dense_3d_array_to_dist_sparse_tensor_r_dp
# 71 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE repl_dense_4d_array_to_dist_sparse_tensor_r_dp
# 73 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 70 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 71 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE repl_dense_2d_array_to_dist_sparse_tensor_r_sp
# 71 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE repl_dense_3d_array_to_dist_sparse_tensor_r_sp
# 71 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE repl_dense_4d_array_to_dist_sparse_tensor_r_sp
# 73 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 70 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 71 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE repl_dense_2d_array_to_dist_sparse_tensor_c_dp
# 71 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE repl_dense_3d_array_to_dist_sparse_tensor_c_dp
# 71 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE repl_dense_4d_array_to_dist_sparse_tensor_c_dp
# 73 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 70 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 71 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE repl_dense_2d_array_to_dist_sparse_tensor_c_sp
# 71 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE repl_dense_3d_array_to_dist_sparse_tensor_c_sp
# 71 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE repl_dense_4d_array_to_dist_sparse_tensor_c_sp
# 73 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 74 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   END INTERFACE

   INTERFACE write_array
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE write_2d_array
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE write_3d_array
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE write_4d_array
# 80 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   END INTERFACE

   INTERFACE dist_sparse_tensor_to_repl_dense_array
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 85 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE dist_sparse_tensor_to_repl_dense_2d_array_r_dp
# 85 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE dist_sparse_tensor_to_repl_dense_3d_array_r_dp
# 85 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE dist_sparse_tensor_to_repl_dense_4d_array_r_dp
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 85 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE dist_sparse_tensor_to_repl_dense_2d_array_r_sp
# 85 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE dist_sparse_tensor_to_repl_dense_3d_array_r_sp
# 85 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE dist_sparse_tensor_to_repl_dense_4d_array_r_sp
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 85 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE dist_sparse_tensor_to_repl_dense_2d_array_c_dp
# 85 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE dist_sparse_tensor_to_repl_dense_3d_array_c_dp
# 85 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE dist_sparse_tensor_to_repl_dense_4d_array_c_dp
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 85 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE dist_sparse_tensor_to_repl_dense_2d_array_c_sp
# 85 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE dist_sparse_tensor_to_repl_dense_3d_array_c_sp
# 85 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      MODULE PROCEDURE dist_sparse_tensor_to_repl_dense_4d_array_c_sp
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
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
# 137 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (dbcsr_type_real_8)
         blocks_equal = MAXVAL(ABS(block1%r_dp%blk - block2%r_dp%blk)) .LT. 1.0E-12_real_8
# 137 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (dbcsr_type_real_4)
         blocks_equal = MAXVAL(ABS(block1%r_sp%blk - block2%r_sp%blk)) .LT. 1.0E-12_real_4
# 137 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (dbcsr_type_complex_8)
         blocks_equal = MAXVAL(ABS(block1%c_dp%blk - block2%c_dp%blk)) .LT. 1.0E-12_real_8
# 137 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (dbcsr_type_complex_4)
         blocks_equal = MAXVAL(ABS(block1%c_sp%blk - block2%c_sp%blk)) .LT. 1.0E-12_real_4
# 140 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
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
                                   blk_size_1, blk_size_2, blk_size_3, blk_size_4, &
                                   blk_ind_1, blk_ind_2, blk_ind_3, blk_ind_4)
      !! Test equivalence of all tensor formats, using a random distribution.
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL :: blk_size_1, blk_size_2, blk_size_3, blk_size_4
         !! block sizes along respective dimension
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL :: blk_ind_1, blk_ind_2, blk_ind_3, blk_ind_4
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
      INTEGER, DIMENSION(:), ALLOCATABLE          :: dist1_1, dist1_2, dist1_3, dist1_4, &
                                                     dist2_1, dist2_2, dist2_3, dist2_4
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

# 234 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         IF (ndims >= 1) THEN
            WRITE (io_unit, '(T4,A,1X,I1,A,1X)', advance='no') 'Dim', 1, ':'
            DO iblk = 1, SIZE(blk_size_1)
               WRITE (io_unit, '(I2,1X)', advance='no') blk_size_1 (iblk)
            ENDDO
            WRITE (io_unit, *)
         ENDIF
# 234 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         IF (ndims >= 2) THEN
            WRITE (io_unit, '(T4,A,1X,I1,A,1X)', advance='no') 'Dim', 2, ':'
            DO iblk = 1, SIZE(blk_size_2)
               WRITE (io_unit, '(I2,1X)', advance='no') blk_size_2 (iblk)
            ENDDO
            WRITE (io_unit, *)
         ENDIF
# 234 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         IF (ndims >= 3) THEN
            WRITE (io_unit, '(T4,A,1X,I1,A,1X)', advance='no') 'Dim', 3, ':'
            DO iblk = 1, SIZE(blk_size_3)
               WRITE (io_unit, '(I2,1X)', advance='no') blk_size_3 (iblk)
            ENDDO
            WRITE (io_unit, *)
         ENDIF
# 234 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         IF (ndims >= 4) THEN
            WRITE (io_unit, '(T4,A,1X,I1,A,1X)', advance='no') 'Dim', 4, ':'
            DO iblk = 1, SIZE(blk_size_4)
               WRITE (io_unit, '(I2,1X)', advance='no') blk_size_4 (iblk)
            ENDDO
            WRITE (io_unit, *)
         ENDIF
# 242 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

         WRITE (io_unit, '(A)') "Non-zero blocks:"
         DO iblk = 1, SIZE(blk_ind_1)
# 246 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (ndims == 2) THEN
               WRITE (io_unit, '(T4,A, I3, A, 2I3, 1X, A)') &
                  'Block', iblk, ': (', blk_ind_1(iblk), blk_ind_2(iblk), ')'
            ENDIF
# 246 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (ndims == 3) THEN
               WRITE (io_unit, '(T4,A, I3, A, 3I3, 1X, A)') &
                  'Block', iblk, ': (', blk_ind_1(iblk), blk_ind_2(iblk), blk_ind_3(iblk), ')'
            ENDIF
# 246 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (ndims == 4) THEN
               WRITE (io_unit, '(T4,A, I3, A, 4I3, 1X, A)') &
                  'Block', iblk, ': (', blk_ind_1(iblk), blk_ind_2(iblk), blk_ind_3(iblk), blk_ind_4(iblk), ')'
            ENDIF
# 251 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
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

# 279 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (1 <= ndims) THEN
               nblks = SIZE(blk_size_1)
               CALL dbcsr_t_random_dist(dist1_1, nblks, pdims(1), mp_comm)
               CALL dbcsr_t_random_dist(dist2_1, nblks, pdims(1), mp_comm)
            ENDIF
# 279 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (2 <= ndims) THEN
               nblks = SIZE(blk_size_2)
               CALL dbcsr_t_random_dist(dist1_2, nblks, pdims(2), mp_comm)
               CALL dbcsr_t_random_dist(dist2_2, nblks, pdims(2), mp_comm)
            ENDIF
# 279 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (3 <= ndims) THEN
               nblks = SIZE(blk_size_3)
               CALL dbcsr_t_random_dist(dist1_3, nblks, pdims(3), mp_comm)
               CALL dbcsr_t_random_dist(dist2_3, nblks, pdims(3), mp_comm)
            ENDIF
# 279 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (4 <= ndims) THEN
               nblks = SIZE(blk_size_4)
               CALL dbcsr_t_random_dist(dist1_4, nblks, pdims(4), mp_comm)
               CALL dbcsr_t_random_dist(dist2_4, nblks, pdims(4), mp_comm)
            ENDIF
# 285 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

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
# 303 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
               IF (1 <= ndims) THEN
                  WRITE (io_unit, '(T7,A,1X)', advance='no') "Dist vec 1:"
                  DO idist = 1, SIZE(dist2_1)
                     WRITE (io_unit, '(I2,1X)', advance='no') dist2_1 (idist)
                  ENDDO
                  WRITE (io_unit, *)
               ENDIF
# 303 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
               IF (2 <= ndims) THEN
                  WRITE (io_unit, '(T7,A,1X)', advance='no') "Dist vec 2:"
                  DO idist = 1, SIZE(dist2_2)
                     WRITE (io_unit, '(I2,1X)', advance='no') dist2_2 (idist)
                  ENDDO
                  WRITE (io_unit, *)
               ENDIF
# 303 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
               IF (3 <= ndims) THEN
                  WRITE (io_unit, '(T7,A,1X)', advance='no') "Dist vec 3:"
                  DO idist = 1, SIZE(dist2_3)
                     WRITE (io_unit, '(I2,1X)', advance='no') dist2_3 (idist)
                  ENDDO
                  WRITE (io_unit, *)
               ENDIF
# 303 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
               IF (4 <= ndims) THEN
                  WRITE (io_unit, '(T7,A,1X)', advance='no') "Dist vec 4:"
                  DO idist = 1, SIZE(dist2_4)
                     WRITE (io_unit, '(I2,1X)', advance='no') dist2_4 (idist)
                  ENDDO
                  WRITE (io_unit, *)
               ENDIF
# 311 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

               WRITE (io_unit, '(T4,A)') "Test distribution:"
# 314 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
               IF (1 <= ndims) THEN
                  WRITE (io_unit, '(T7,A,1X)', advance='no') "Dist vec 1:"
                  DO idist = 1, SIZE(dist2_1)
                     WRITE (io_unit, '(I2,1X)', advance='no') dist1_1 (idist)
                  ENDDO
                  WRITE (io_unit, *)
               ENDIF
# 314 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
               IF (2 <= ndims) THEN
                  WRITE (io_unit, '(T7,A,1X)', advance='no') "Dist vec 2:"
                  DO idist = 1, SIZE(dist2_2)
                     WRITE (io_unit, '(I2,1X)', advance='no') dist1_2 (idist)
                  ENDDO
                  WRITE (io_unit, *)
               ENDIF
# 314 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
               IF (3 <= ndims) THEN
                  WRITE (io_unit, '(T7,A,1X)', advance='no') "Dist vec 3:"
                  DO idist = 1, SIZE(dist2_3)
                     WRITE (io_unit, '(I2,1X)', advance='no') dist1_3 (idist)
                  ENDDO
                  WRITE (io_unit, *)
               ENDIF
# 314 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
               IF (4 <= ndims) THEN
                  WRITE (io_unit, '(T7,A,1X)', advance='no') "Dist vec 4:"
                  DO idist = 1, SIZE(dist2_4)
                     WRITE (io_unit, '(I2,1X)', advance='no') dist1_4 (idist)
                  ENDDO
                  WRITE (io_unit, *)
               ENDIF
# 322 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            ENDIF

# 325 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (ndims == 2) THEN
               CALL dbcsr_t_distribution_new(dist2, comm_nd, dist2_1, dist2_2)
               CALL dbcsr_t_create(tensor2, "Ref", dist2, map1_ref, map2_ref, &
                                   dbcsr_type_real_8, blk_size_1, blk_size_2)
               CALL dbcsr_t_setup_test_tensor(tensor2, comm_nd%mp_comm_2d, .TRUE., blk_ind_1, blk_ind_2)
            ENDIF
# 325 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (ndims == 3) THEN
               CALL dbcsr_t_distribution_new(dist2, comm_nd, dist2_1, dist2_2, dist2_3)
               CALL dbcsr_t_create(tensor2, "Ref", dist2, map1_ref, map2_ref, &
                                   dbcsr_type_real_8, blk_size_1, blk_size_2, blk_size_3)
               CALL dbcsr_t_setup_test_tensor(tensor2, comm_nd%mp_comm_2d, .TRUE., blk_ind_1, blk_ind_2, blk_ind_3)
            ENDIF
# 325 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (ndims == 4) THEN
               CALL dbcsr_t_distribution_new(dist2, comm_nd, dist2_1, dist2_2, dist2_3, dist2_4)
               CALL dbcsr_t_create(tensor2, "Ref", dist2, map1_ref, map2_ref, &
                                   dbcsr_type_real_8, blk_size_1, blk_size_2, blk_size_3, blk_size_4)
               CALL dbcsr_t_setup_test_tensor(tensor2, comm_nd%mp_comm_2d, .TRUE., blk_ind_1, blk_ind_2, blk_ind_3, blk_ind_4)
            ENDIF
# 332 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

            IF (verbose) CALL dbcsr_t_write_blocks(tensor2, io_unit, unit_nr)

# 336 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (ndims == 2) THEN
               CALL dbcsr_t_distribution_new(dist1, comm_nd, dist1_1, dist1_2)
               CALL dbcsr_t_create(tensor1, tensor_name, dist1, map1, map2, &
                                   dbcsr_type_real_8, blk_size_1, blk_size_2)
               CALL dbcsr_t_setup_test_tensor(tensor1, comm_nd%mp_comm_2d, .TRUE., blk_ind_1, blk_ind_2)
            ENDIF
# 336 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (ndims == 3) THEN
               CALL dbcsr_t_distribution_new(dist1, comm_nd, dist1_1, dist1_2, dist1_3)
               CALL dbcsr_t_create(tensor1, tensor_name, dist1, map1, map2, &
                                   dbcsr_type_real_8, blk_size_1, blk_size_2, blk_size_3)
               CALL dbcsr_t_setup_test_tensor(tensor1, comm_nd%mp_comm_2d, .TRUE., blk_ind_1, blk_ind_2, blk_ind_3)
            ENDIF
# 336 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (ndims == 4) THEN
               CALL dbcsr_t_distribution_new(dist1, comm_nd, dist1_1, dist1_2, dist1_3, dist1_4)
               CALL dbcsr_t_create(tensor1, tensor_name, dist1, map1, map2, &
                                   dbcsr_type_real_8, blk_size_1, blk_size_2, blk_size_3, blk_size_4)
               CALL dbcsr_t_setup_test_tensor(tensor1, comm_nd%mp_comm_2d, .TRUE., blk_ind_1, blk_ind_2, blk_ind_3, blk_ind_4)
            ENDIF
# 343 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

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

# 363 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (1 <= ndims) THEN
               DEALLOCATE (dist1_1, dist2_1)
            ENDIF
# 363 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (2 <= ndims) THEN
               DEALLOCATE (dist1_2, dist2_2)
            ENDIF
# 363 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (3 <= ndims) THEN
               DEALLOCATE (dist1_3, dist2_3)
            ENDIF
# 363 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (4 <= ndims) THEN
               DEALLOCATE (dist1_4, dist2_4)
            ENDIF
# 367 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

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

   SUBROUTINE dbcsr_t_setup_test_tensor(tensor, mp_comm, enumerate, blk_ind_1, blk_ind_2, blk_ind_3, blk_ind_4)
      !! Allocate and fill test tensor - entries are enumerated by their index s.t. they only depend
      !! on global properties of the tensor but not on distribution, matrix representation, etc.
      TYPE(dbcsr_t_type), INTENT(INOUT)                  :: tensor
      INTEGER, INTENT(IN)                                :: mp_comm
         !! communicator
      LOGICAL, INTENT(IN)                                :: enumerate
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL        :: blk_ind_1, blk_ind_2, blk_ind_3, blk_ind_4
         !! index along respective dimension of non-zero blocks
      INTEGER                                            :: blk, numnodes, mynode

      INTEGER                                            :: i, ib, my_nblks_alloc, nblks_alloc, proc
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: my_blk_ind_1, my_blk_ind_2, my_blk_ind_3, my_blk_ind_4
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))          :: blk_index, blk_offset, blk_size, &
                                                            tensor_dims
      INTEGER, DIMENSION(:, :), ALLOCATABLE               :: ind_nd
# 411 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      REAL(KIND=real_8), ALLOCATABLE, &
         DIMENSION(:,:)                :: blk_values_2
# 411 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      REAL(KIND=real_8), ALLOCATABLE, &
         DIMENSION(:,:,:)                :: blk_values_3
# 411 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      REAL(KIND=real_8), ALLOCATABLE, &
         DIMENSION(:,:,:,:)                :: blk_values_4
# 414 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      TYPE(dbcsr_t_iterator_type)                        :: iterator

      nblks_alloc = SIZE(blk_ind_1)
      CALL mp_environ(numnodes, mynode, mp_comm)

      ALLOCATE (ind_nd(nblks_alloc, dbcsr_t_ndims(tensor)))
      my_nblks_alloc = 0
      DO ib = 1, nblks_alloc
# 423 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         IF (dbcsr_t_ndims(tensor) == 2) THEN
            ind_nd(ib, :) = [blk_ind_1(ib), blk_ind_2(ib)]
         ENDIF
# 423 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         IF (dbcsr_t_ndims(tensor) == 3) THEN
            ind_nd(ib, :) = [blk_ind_1(ib), blk_ind_2(ib), blk_ind_3(ib)]
         ENDIF
# 423 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         IF (dbcsr_t_ndims(tensor) == 4) THEN
            ind_nd(ib, :) = [blk_ind_1(ib), blk_ind_2(ib), blk_ind_3(ib), blk_ind_4(ib)]
         ENDIF
# 427 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CALL dbcsr_t_get_stored_coordinates(tensor, ind_nd(ib, :), proc)
         IF (proc == mynode) THEN
            my_nblks_alloc = my_nblks_alloc + 1
         ENDIF
      ENDDO

# 434 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      IF (dbcsr_t_ndims(tensor) >= 1) THEN
         ALLOCATE (my_blk_ind_1 (my_nblks_alloc))
      ENDIF
# 434 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      IF (dbcsr_t_ndims(tensor) >= 2) THEN
         ALLOCATE (my_blk_ind_2 (my_nblks_alloc))
      ENDIF
# 434 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      IF (dbcsr_t_ndims(tensor) >= 3) THEN
         ALLOCATE (my_blk_ind_3 (my_nblks_alloc))
      ENDIF
# 434 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      IF (dbcsr_t_ndims(tensor) >= 4) THEN
         ALLOCATE (my_blk_ind_4 (my_nblks_alloc))
      ENDIF
# 438 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

      i = 0
      DO ib = 1, nblks_alloc
         CALL dbcsr_t_get_stored_coordinates(tensor, ind_nd(ib, :), proc)
         IF (proc == mynode) THEN
            i = i + 1
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (dbcsr_t_ndims(tensor) >= 1) THEN
               my_blk_ind_1 (i) = blk_ind_1 (ib)
            ENDIF
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (dbcsr_t_ndims(tensor) >= 2) THEN
               my_blk_ind_2 (i) = blk_ind_2 (ib)
            ENDIF
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (dbcsr_t_ndims(tensor) >= 3) THEN
               my_blk_ind_3 (i) = blk_ind_3 (ib)
            ENDIF
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            IF (dbcsr_t_ndims(tensor) >= 4) THEN
               my_blk_ind_4 (i) = blk_ind_4 (ib)
            ENDIF
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         ENDIF
      ENDDO

# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      IF (dbcsr_t_ndims(tensor) == 2) THEN
         CALL dbcsr_t_reserve_blocks(tensor, my_blk_ind_1, my_blk_ind_2)
      ENDIF
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      IF (dbcsr_t_ndims(tensor) == 3) THEN
         CALL dbcsr_t_reserve_blocks(tensor, my_blk_ind_1, my_blk_ind_2, my_blk_ind_3)
      ENDIF
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      IF (dbcsr_t_ndims(tensor) == 4) THEN
         CALL dbcsr_t_reserve_blocks(tensor, my_blk_ind_1, my_blk_ind_2, my_blk_ind_3, my_blk_ind_4)
      ENDIF
# 457 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, blk_index, blk, blk_size=blk_size, blk_offset=blk_offset)

# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         IF (dbcsr_t_ndims(tensor) == 2) THEN
            CALL allocate_any(blk_values_2, shape_spec=blk_size)
            CALL dims_tensor(tensor, tensor_dims)
            IF (enumerate) THEN
               CALL enumerate_block_elements(blk_size, blk_offset, tensor_dims, blk_2=blk_values_2)
            ELSE
               CALL random_number(blk_values_2)
            ENDIF
            CALL dbcsr_t_put_block(tensor, blk_index, blk_size, blk_values_2)
            DEALLOCATE (blk_values_2)
         ENDIF
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         IF (dbcsr_t_ndims(tensor) == 3) THEN
            CALL allocate_any(blk_values_3, shape_spec=blk_size)
            CALL dims_tensor(tensor, tensor_dims)
            IF (enumerate) THEN
               CALL enumerate_block_elements(blk_size, blk_offset, tensor_dims, blk_3=blk_values_3)
            ELSE
               CALL random_number(blk_values_3)
            ENDIF
            CALL dbcsr_t_put_block(tensor, blk_index, blk_size, blk_values_3)
            DEALLOCATE (blk_values_3)
         ENDIF
# 463 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         IF (dbcsr_t_ndims(tensor) == 4) THEN
            CALL allocate_any(blk_values_4, shape_spec=blk_size)
            CALL dims_tensor(tensor, tensor_dims)
            IF (enumerate) THEN
               CALL enumerate_block_elements(blk_size, blk_offset, tensor_dims, blk_4=blk_values_4)
            ELSE
               CALL random_number(blk_values_4)
            ENDIF
            CALL dbcsr_t_put_block(tensor, blk_index, blk_size, blk_values_4)
            DEALLOCATE (blk_values_4)
         ENDIF
# 475 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)

   END SUBROUTINE

   SUBROUTINE enumerate_block_elements(blk_size, blk_offset, tensor_size, blk_2, blk_3, blk_4)
      !! Enumerate tensor entries in block
      !! \blk_2 block values for 2 dimensions
      !! \blk_3 block values for 3 dimensions

      INTEGER, DIMENSION(:), INTENT(IN)                  :: blk_size, blk_offset, tensor_size
         !! size of block
         !! block offset (indices of first element)
         !! global tensor sizes
# 490 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      REAL(KIND=real_8), DIMENSION(:,:), &
         OPTIONAL, INTENT(OUT)                           :: blk_2
# 490 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      REAL(KIND=real_8), DIMENSION(:,:,:), &
         OPTIONAL, INTENT(OUT)                           :: blk_3
# 490 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      REAL(KIND=real_8), DIMENSION(:,:,:,:), &
         OPTIONAL, INTENT(OUT)                           :: blk_4
# 493 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      INTEGER                                            :: ndim
      INTEGER, DIMENSION(SIZE(blk_size))                 :: arr_ind, tens_ind
      INTEGER                                            :: i_1, i_2, i_3, i_4

      ndim = SIZE(tensor_size)

# 500 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      IF (ndim == 2) THEN
# 502 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         DO i_2 = 1, blk_size(2)
# 502 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         DO i_1 = 1, blk_size(1)
# 504 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            arr_ind(:) = [i_1, i_2]
            tens_ind(:) = arr_ind(:) + blk_offset(:) - 1
            blk_2 (arr_ind(1), arr_ind(2)) = combine_index(tens_ind, tensor_size)
# 508 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         ENDDO
# 508 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         ENDDO
# 510 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      ENDIF
# 500 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      IF (ndim == 3) THEN
# 502 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         DO i_3 = 1, blk_size(3)
# 502 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         DO i_2 = 1, blk_size(2)
# 502 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         DO i_1 = 1, blk_size(1)
# 504 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            arr_ind(:) = [i_1, i_2, i_3]
            tens_ind(:) = arr_ind(:) + blk_offset(:) - 1
            blk_3 (arr_ind(1), arr_ind(2), arr_ind(3)) = combine_index(tens_ind, tensor_size)
# 508 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         ENDDO
# 508 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         ENDDO
# 508 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         ENDDO
# 510 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      ENDIF
# 500 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      IF (ndim == 4) THEN
# 502 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         DO i_4 = 1, blk_size(4)
# 502 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         DO i_3 = 1, blk_size(3)
# 502 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         DO i_2 = 1, blk_size(2)
# 502 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         DO i_1 = 1, blk_size(1)
# 504 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
            arr_ind(:) = [i_1, i_2, i_3, i_4]
            tens_ind(:) = arr_ind(:) + blk_offset(:) - 1
            blk_4 (arr_ind(1), arr_ind(2), arr_ind(3), arr_ind(4)) = combine_index(tens_ind, tensor_size)
# 508 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         ENDDO
# 508 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         ENDDO
# 508 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         ENDDO
# 508 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         ENDDO
# 510 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      ENDIF
# 512 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

   END SUBROUTINE

# 516 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 517 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE dist_sparse_tensor_to_repl_dense_2d_array_r_dp (tensor, array)
      !! Transform a distributed sparse tensor to a replicated dense array. This is only useful for
      !! testing tensor contraction by matrix multiplication of dense arrays.

      TYPE(dbcsr_t_type), INTENT(INOUT)                          :: tensor
      REAL(kind=real_8), ALLOCATABLE, DIMENSION(:,:), &
         INTENT(OUT)                                             :: array
      REAL(kind=real_8), ALLOCATABLE, DIMENSION(:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end
      LOGICAL                                                    :: found

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 2)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)
      CALL allocate_any(array, shape_spec=dims_nd)
      array(:,:) = 0.0_real_8

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL dbcsr_t_get_block(tensor, ind_nd, block, found)
         DBCSR_ASSERT(found)

         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         array(blk_start(1):blk_end(1), blk_start(2):blk_end(2)) = &
            block(:,:)

         DEALLOCATE (block)
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)
      CALL mp_sum(array, tensor%pgrid%mp_comm_2d)

   END SUBROUTINE
# 517 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE dist_sparse_tensor_to_repl_dense_3d_array_r_dp (tensor, array)
      !! Transform a distributed sparse tensor to a replicated dense array. This is only useful for
      !! testing tensor contraction by matrix multiplication of dense arrays.

      TYPE(dbcsr_t_type), INTENT(INOUT)                          :: tensor
      REAL(kind=real_8), ALLOCATABLE, DIMENSION(:,:,:), &
         INTENT(OUT)                                             :: array
      REAL(kind=real_8), ALLOCATABLE, DIMENSION(:,:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end
      LOGICAL                                                    :: found

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 3)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)
      CALL allocate_any(array, shape_spec=dims_nd)
      array(:,:,:) = 0.0_real_8

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL dbcsr_t_get_block(tensor, ind_nd, block, found)
         DBCSR_ASSERT(found)

         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         array(blk_start(1):blk_end(1), blk_start(2):blk_end(2), blk_start(3):blk_end(3)) = &
            block(:,:,:)

         DEALLOCATE (block)
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)
      CALL mp_sum(array, tensor%pgrid%mp_comm_2d)

   END SUBROUTINE
# 517 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE dist_sparse_tensor_to_repl_dense_4d_array_r_dp (tensor, array)
      !! Transform a distributed sparse tensor to a replicated dense array. This is only useful for
      !! testing tensor contraction by matrix multiplication of dense arrays.

      TYPE(dbcsr_t_type), INTENT(INOUT)                          :: tensor
      REAL(kind=real_8), ALLOCATABLE, DIMENSION(:,:,:,:), &
         INTENT(OUT)                                             :: array
      REAL(kind=real_8), ALLOCATABLE, DIMENSION(:,:,:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end
      LOGICAL                                                    :: found

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 4)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)
      CALL allocate_any(array, shape_spec=dims_nd)
      array(:,:,:,:) = 0.0_real_8

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL dbcsr_t_get_block(tensor, ind_nd, block, found)
         DBCSR_ASSERT(found)

         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         array(blk_start(1):blk_end(1), blk_start(2):blk_end(2), blk_start(3):blk_end(3), blk_start(4):blk_end(4)) = &
            block(:,:,:,:)

         DEALLOCATE (block)
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)
      CALL mp_sum(array, tensor%pgrid%mp_comm_2d)

   END SUBROUTINE
# 556 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 516 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 517 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE dist_sparse_tensor_to_repl_dense_2d_array_r_sp (tensor, array)
      !! Transform a distributed sparse tensor to a replicated dense array. This is only useful for
      !! testing tensor contraction by matrix multiplication of dense arrays.

      TYPE(dbcsr_t_type), INTENT(INOUT)                          :: tensor
      REAL(kind=real_4), ALLOCATABLE, DIMENSION(:,:), &
         INTENT(OUT)                                             :: array
      REAL(kind=real_4), ALLOCATABLE, DIMENSION(:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end
      LOGICAL                                                    :: found

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 2)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)
      CALL allocate_any(array, shape_spec=dims_nd)
      array(:,:) = 0.0_real_4

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL dbcsr_t_get_block(tensor, ind_nd, block, found)
         DBCSR_ASSERT(found)

         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         array(blk_start(1):blk_end(1), blk_start(2):blk_end(2)) = &
            block(:,:)

         DEALLOCATE (block)
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)
      CALL mp_sum(array, tensor%pgrid%mp_comm_2d)

   END SUBROUTINE
# 517 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE dist_sparse_tensor_to_repl_dense_3d_array_r_sp (tensor, array)
      !! Transform a distributed sparse tensor to a replicated dense array. This is only useful for
      !! testing tensor contraction by matrix multiplication of dense arrays.

      TYPE(dbcsr_t_type), INTENT(INOUT)                          :: tensor
      REAL(kind=real_4), ALLOCATABLE, DIMENSION(:,:,:), &
         INTENT(OUT)                                             :: array
      REAL(kind=real_4), ALLOCATABLE, DIMENSION(:,:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end
      LOGICAL                                                    :: found

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 3)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)
      CALL allocate_any(array, shape_spec=dims_nd)
      array(:,:,:) = 0.0_real_4

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL dbcsr_t_get_block(tensor, ind_nd, block, found)
         DBCSR_ASSERT(found)

         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         array(blk_start(1):blk_end(1), blk_start(2):blk_end(2), blk_start(3):blk_end(3)) = &
            block(:,:,:)

         DEALLOCATE (block)
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)
      CALL mp_sum(array, tensor%pgrid%mp_comm_2d)

   END SUBROUTINE
# 517 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE dist_sparse_tensor_to_repl_dense_4d_array_r_sp (tensor, array)
      !! Transform a distributed sparse tensor to a replicated dense array. This is only useful for
      !! testing tensor contraction by matrix multiplication of dense arrays.

      TYPE(dbcsr_t_type), INTENT(INOUT)                          :: tensor
      REAL(kind=real_4), ALLOCATABLE, DIMENSION(:,:,:,:), &
         INTENT(OUT)                                             :: array
      REAL(kind=real_4), ALLOCATABLE, DIMENSION(:,:,:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end
      LOGICAL                                                    :: found

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 4)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)
      CALL allocate_any(array, shape_spec=dims_nd)
      array(:,:,:,:) = 0.0_real_4

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL dbcsr_t_get_block(tensor, ind_nd, block, found)
         DBCSR_ASSERT(found)

         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         array(blk_start(1):blk_end(1), blk_start(2):blk_end(2), blk_start(3):blk_end(3), blk_start(4):blk_end(4)) = &
            block(:,:,:,:)

         DEALLOCATE (block)
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)
      CALL mp_sum(array, tensor%pgrid%mp_comm_2d)

   END SUBROUTINE
# 556 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 516 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 517 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE dist_sparse_tensor_to_repl_dense_2d_array_c_dp (tensor, array)
      !! Transform a distributed sparse tensor to a replicated dense array. This is only useful for
      !! testing tensor contraction by matrix multiplication of dense arrays.

      TYPE(dbcsr_t_type), INTENT(INOUT)                          :: tensor
      COMPLEX(kind=real_8), ALLOCATABLE, DIMENSION(:,:), &
         INTENT(OUT)                                             :: array
      COMPLEX(kind=real_8), ALLOCATABLE, DIMENSION(:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end
      LOGICAL                                                    :: found

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 2)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)
      CALL allocate_any(array, shape_spec=dims_nd)
      array(:,:) = 0.0_real_8

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL dbcsr_t_get_block(tensor, ind_nd, block, found)
         DBCSR_ASSERT(found)

         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         array(blk_start(1):blk_end(1), blk_start(2):blk_end(2)) = &
            block(:,:)

         DEALLOCATE (block)
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)
      CALL mp_sum(array, tensor%pgrid%mp_comm_2d)

   END SUBROUTINE
# 517 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE dist_sparse_tensor_to_repl_dense_3d_array_c_dp (tensor, array)
      !! Transform a distributed sparse tensor to a replicated dense array. This is only useful for
      !! testing tensor contraction by matrix multiplication of dense arrays.

      TYPE(dbcsr_t_type), INTENT(INOUT)                          :: tensor
      COMPLEX(kind=real_8), ALLOCATABLE, DIMENSION(:,:,:), &
         INTENT(OUT)                                             :: array
      COMPLEX(kind=real_8), ALLOCATABLE, DIMENSION(:,:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end
      LOGICAL                                                    :: found

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 3)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)
      CALL allocate_any(array, shape_spec=dims_nd)
      array(:,:,:) = 0.0_real_8

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL dbcsr_t_get_block(tensor, ind_nd, block, found)
         DBCSR_ASSERT(found)

         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         array(blk_start(1):blk_end(1), blk_start(2):blk_end(2), blk_start(3):blk_end(3)) = &
            block(:,:,:)

         DEALLOCATE (block)
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)
      CALL mp_sum(array, tensor%pgrid%mp_comm_2d)

   END SUBROUTINE
# 517 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE dist_sparse_tensor_to_repl_dense_4d_array_c_dp (tensor, array)
      !! Transform a distributed sparse tensor to a replicated dense array. This is only useful for
      !! testing tensor contraction by matrix multiplication of dense arrays.

      TYPE(dbcsr_t_type), INTENT(INOUT)                          :: tensor
      COMPLEX(kind=real_8), ALLOCATABLE, DIMENSION(:,:,:,:), &
         INTENT(OUT)                                             :: array
      COMPLEX(kind=real_8), ALLOCATABLE, DIMENSION(:,:,:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end
      LOGICAL                                                    :: found

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 4)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)
      CALL allocate_any(array, shape_spec=dims_nd)
      array(:,:,:,:) = 0.0_real_8

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL dbcsr_t_get_block(tensor, ind_nd, block, found)
         DBCSR_ASSERT(found)

         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         array(blk_start(1):blk_end(1), blk_start(2):blk_end(2), blk_start(3):blk_end(3), blk_start(4):blk_end(4)) = &
            block(:,:,:,:)

         DEALLOCATE (block)
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)
      CALL mp_sum(array, tensor%pgrid%mp_comm_2d)

   END SUBROUTINE
# 556 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 516 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 517 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE dist_sparse_tensor_to_repl_dense_2d_array_c_sp (tensor, array)
      !! Transform a distributed sparse tensor to a replicated dense array. This is only useful for
      !! testing tensor contraction by matrix multiplication of dense arrays.

      TYPE(dbcsr_t_type), INTENT(INOUT)                          :: tensor
      COMPLEX(kind=real_4), ALLOCATABLE, DIMENSION(:,:), &
         INTENT(OUT)                                             :: array
      COMPLEX(kind=real_4), ALLOCATABLE, DIMENSION(:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end
      LOGICAL                                                    :: found

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 2)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)
      CALL allocate_any(array, shape_spec=dims_nd)
      array(:,:) = 0.0_real_4

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL dbcsr_t_get_block(tensor, ind_nd, block, found)
         DBCSR_ASSERT(found)

         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         array(blk_start(1):blk_end(1), blk_start(2):blk_end(2)) = &
            block(:,:)

         DEALLOCATE (block)
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)
      CALL mp_sum(array, tensor%pgrid%mp_comm_2d)

   END SUBROUTINE
# 517 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE dist_sparse_tensor_to_repl_dense_3d_array_c_sp (tensor, array)
      !! Transform a distributed sparse tensor to a replicated dense array. This is only useful for
      !! testing tensor contraction by matrix multiplication of dense arrays.

      TYPE(dbcsr_t_type), INTENT(INOUT)                          :: tensor
      COMPLEX(kind=real_4), ALLOCATABLE, DIMENSION(:,:,:), &
         INTENT(OUT)                                             :: array
      COMPLEX(kind=real_4), ALLOCATABLE, DIMENSION(:,:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end
      LOGICAL                                                    :: found

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 3)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)
      CALL allocate_any(array, shape_spec=dims_nd)
      array(:,:,:) = 0.0_real_4

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL dbcsr_t_get_block(tensor, ind_nd, block, found)
         DBCSR_ASSERT(found)

         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         array(blk_start(1):blk_end(1), blk_start(2):blk_end(2), blk_start(3):blk_end(3)) = &
            block(:,:,:)

         DEALLOCATE (block)
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)
      CALL mp_sum(array, tensor%pgrid%mp_comm_2d)

   END SUBROUTINE
# 517 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE dist_sparse_tensor_to_repl_dense_4d_array_c_sp (tensor, array)
      !! Transform a distributed sparse tensor to a replicated dense array. This is only useful for
      !! testing tensor contraction by matrix multiplication of dense arrays.

      TYPE(dbcsr_t_type), INTENT(INOUT)                          :: tensor
      COMPLEX(kind=real_4), ALLOCATABLE, DIMENSION(:,:,:,:), &
         INTENT(OUT)                                             :: array
      COMPLEX(kind=real_4), ALLOCATABLE, DIMENSION(:,:,:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end
      LOGICAL                                                    :: found

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 4)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)
      CALL allocate_any(array, shape_spec=dims_nd)
      array(:,:,:,:) = 0.0_real_4

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL dbcsr_t_get_block(tensor, ind_nd, block, found)
         DBCSR_ASSERT(found)

         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         array(blk_start(1):blk_end(1), blk_start(2):blk_end(2), blk_start(3):blk_end(3), blk_start(4):blk_end(4)) = &
            block(:,:,:,:)

         DEALLOCATE (block)
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)
      CALL mp_sum(array, tensor%pgrid%mp_comm_2d)

   END SUBROUTINE
# 556 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 557 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

# 559 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE repl_dense_2d_array_to_dist_sparse_tensor_r_dp (tensor, array)
      TYPE(dbcsr_t_type), INTENT(INOUT)                             :: tensor
      REAL(kind=real_8), ALLOCATABLE, DIMENSION(:,:), &
         INTENT(INOUT)                                           :: array
      REAL(kind=real_8), ALLOCATABLE, DIMENSION(:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 2)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL allocate_any(block, shape_spec=blk_size)
         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         block(:,:) = &
            array(blk_start(1):blk_end(1), blk_start(2):blk_end(2))
         CALL dbcsr_t_put_block(tensor, ind_nd, blk_size, block)
         DEALLOCATE (block)
      ENDDO

   END SUBROUTINE
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE repl_dense_3d_array_to_dist_sparse_tensor_r_dp (tensor, array)
      TYPE(dbcsr_t_type), INTENT(INOUT)                             :: tensor
      REAL(kind=real_8), ALLOCATABLE, DIMENSION(:,:,:), &
         INTENT(INOUT)                                           :: array
      REAL(kind=real_8), ALLOCATABLE, DIMENSION(:,:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 3)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL allocate_any(block, shape_spec=blk_size)
         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         block(:,:,:) = &
            array(blk_start(1):blk_end(1), blk_start(2):blk_end(2), blk_start(3):blk_end(3))
         CALL dbcsr_t_put_block(tensor, ind_nd, blk_size, block)
         DEALLOCATE (block)
      ENDDO

   END SUBROUTINE
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE repl_dense_4d_array_to_dist_sparse_tensor_r_dp (tensor, array)
      TYPE(dbcsr_t_type), INTENT(INOUT)                             :: tensor
      REAL(kind=real_8), ALLOCATABLE, DIMENSION(:,:,:,:), &
         INTENT(INOUT)                                           :: array
      REAL(kind=real_8), ALLOCATABLE, DIMENSION(:,:,:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 4)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL allocate_any(block, shape_spec=blk_size)
         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         block(:,:,:,:) = &
            array(blk_start(1):blk_end(1), blk_start(2):blk_end(2), blk_start(3):blk_end(3), blk_start(4):blk_end(4))
         CALL dbcsr_t_put_block(tensor, ind_nd, blk_size, block)
         DEALLOCATE (block)
      ENDDO

   END SUBROUTINE
# 589 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 559 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE repl_dense_2d_array_to_dist_sparse_tensor_r_sp (tensor, array)
      TYPE(dbcsr_t_type), INTENT(INOUT)                             :: tensor
      REAL(kind=real_4), ALLOCATABLE, DIMENSION(:,:), &
         INTENT(INOUT)                                           :: array
      REAL(kind=real_4), ALLOCATABLE, DIMENSION(:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 2)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL allocate_any(block, shape_spec=blk_size)
         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         block(:,:) = &
            array(blk_start(1):blk_end(1), blk_start(2):blk_end(2))
         CALL dbcsr_t_put_block(tensor, ind_nd, blk_size, block)
         DEALLOCATE (block)
      ENDDO

   END SUBROUTINE
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE repl_dense_3d_array_to_dist_sparse_tensor_r_sp (tensor, array)
      TYPE(dbcsr_t_type), INTENT(INOUT)                             :: tensor
      REAL(kind=real_4), ALLOCATABLE, DIMENSION(:,:,:), &
         INTENT(INOUT)                                           :: array
      REAL(kind=real_4), ALLOCATABLE, DIMENSION(:,:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 3)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL allocate_any(block, shape_spec=blk_size)
         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         block(:,:,:) = &
            array(blk_start(1):blk_end(1), blk_start(2):blk_end(2), blk_start(3):blk_end(3))
         CALL dbcsr_t_put_block(tensor, ind_nd, blk_size, block)
         DEALLOCATE (block)
      ENDDO

   END SUBROUTINE
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE repl_dense_4d_array_to_dist_sparse_tensor_r_sp (tensor, array)
      TYPE(dbcsr_t_type), INTENT(INOUT)                             :: tensor
      REAL(kind=real_4), ALLOCATABLE, DIMENSION(:,:,:,:), &
         INTENT(INOUT)                                           :: array
      REAL(kind=real_4), ALLOCATABLE, DIMENSION(:,:,:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 4)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL allocate_any(block, shape_spec=blk_size)
         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         block(:,:,:,:) = &
            array(blk_start(1):blk_end(1), blk_start(2):blk_end(2), blk_start(3):blk_end(3), blk_start(4):blk_end(4))
         CALL dbcsr_t_put_block(tensor, ind_nd, blk_size, block)
         DEALLOCATE (block)
      ENDDO

   END SUBROUTINE
# 589 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 559 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE repl_dense_2d_array_to_dist_sparse_tensor_c_dp (tensor, array)
      TYPE(dbcsr_t_type), INTENT(INOUT)                             :: tensor
      COMPLEX(kind=real_8), ALLOCATABLE, DIMENSION(:,:), &
         INTENT(INOUT)                                           :: array
      COMPLEX(kind=real_8), ALLOCATABLE, DIMENSION(:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 2)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL allocate_any(block, shape_spec=blk_size)
         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         block(:,:) = &
            array(blk_start(1):blk_end(1), blk_start(2):blk_end(2))
         CALL dbcsr_t_put_block(tensor, ind_nd, blk_size, block)
         DEALLOCATE (block)
      ENDDO

   END SUBROUTINE
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE repl_dense_3d_array_to_dist_sparse_tensor_c_dp (tensor, array)
      TYPE(dbcsr_t_type), INTENT(INOUT)                             :: tensor
      COMPLEX(kind=real_8), ALLOCATABLE, DIMENSION(:,:,:), &
         INTENT(INOUT)                                           :: array
      COMPLEX(kind=real_8), ALLOCATABLE, DIMENSION(:,:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 3)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL allocate_any(block, shape_spec=blk_size)
         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         block(:,:,:) = &
            array(blk_start(1):blk_end(1), blk_start(2):blk_end(2), blk_start(3):blk_end(3))
         CALL dbcsr_t_put_block(tensor, ind_nd, blk_size, block)
         DEALLOCATE (block)
      ENDDO

   END SUBROUTINE
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE repl_dense_4d_array_to_dist_sparse_tensor_c_dp (tensor, array)
      TYPE(dbcsr_t_type), INTENT(INOUT)                             :: tensor
      COMPLEX(kind=real_8), ALLOCATABLE, DIMENSION(:,:,:,:), &
         INTENT(INOUT)                                           :: array
      COMPLEX(kind=real_8), ALLOCATABLE, DIMENSION(:,:,:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 4)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL allocate_any(block, shape_spec=blk_size)
         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         block(:,:,:,:) = &
            array(blk_start(1):blk_end(1), blk_start(2):blk_end(2), blk_start(3):blk_end(3), blk_start(4):blk_end(4))
         CALL dbcsr_t_put_block(tensor, ind_nd, blk_size, block)
         DEALLOCATE (block)
      ENDDO

   END SUBROUTINE
# 589 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 559 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE repl_dense_2d_array_to_dist_sparse_tensor_c_sp (tensor, array)
      TYPE(dbcsr_t_type), INTENT(INOUT)                             :: tensor
      COMPLEX(kind=real_4), ALLOCATABLE, DIMENSION(:,:), &
         INTENT(INOUT)                                           :: array
      COMPLEX(kind=real_4), ALLOCATABLE, DIMENSION(:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 2)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL allocate_any(block, shape_spec=blk_size)
         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         block(:,:) = &
            array(blk_start(1):blk_end(1), blk_start(2):blk_end(2))
         CALL dbcsr_t_put_block(tensor, ind_nd, blk_size, block)
         DEALLOCATE (block)
      ENDDO

   END SUBROUTINE
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE repl_dense_3d_array_to_dist_sparse_tensor_c_sp (tensor, array)
      TYPE(dbcsr_t_type), INTENT(INOUT)                             :: tensor
      COMPLEX(kind=real_4), ALLOCATABLE, DIMENSION(:,:,:), &
         INTENT(INOUT)                                           :: array
      COMPLEX(kind=real_4), ALLOCATABLE, DIMENSION(:,:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 3)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL allocate_any(block, shape_spec=blk_size)
         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         block(:,:,:) = &
            array(blk_start(1):blk_end(1), blk_start(2):blk_end(2), blk_start(3):blk_end(3))
         CALL dbcsr_t_put_block(tensor, ind_nd, blk_size, block)
         DEALLOCATE (block)
      ENDDO

   END SUBROUTINE
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE repl_dense_4d_array_to_dist_sparse_tensor_c_sp (tensor, array)
      TYPE(dbcsr_t_type), INTENT(INOUT)                             :: tensor
      COMPLEX(kind=real_4), ALLOCATABLE, DIMENSION(:,:,:,:), &
         INTENT(INOUT)                                           :: array
      COMPLEX(kind=real_4), ALLOCATABLE, DIMENSION(:,:,:,:)   :: block
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: dims_nd, ind_nd, blk_size, blk_offset
      TYPE(dbcsr_t_iterator_type)                                     :: iterator
      INTEGER                                                    :: blk, idim
      INTEGER, DIMENSION(dbcsr_t_ndims(tensor))                  :: blk_start, blk_end

      DBCSR_ASSERT(dbcsr_t_ndims(tensor) .EQ. 4)
      CALL dbcsr_t_get_mapping_info(tensor%nd_index, dims_nd=dims_nd)

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, ind_nd, blk, blk_size=blk_size, blk_offset=blk_offset)
         CALL allocate_any(block, shape_spec=blk_size)
         DO idim = 1, dbcsr_t_ndims(tensor)
            blk_start(idim) = blk_offset(idim)
            blk_end(idim) = blk_offset(idim) + blk_size(idim) - 1
         ENDDO
         block(:,:,:,:) = &
            array(blk_start(1):blk_end(1), blk_start(2):blk_end(2), blk_start(3):blk_end(3), blk_start(4):blk_end(4))
         CALL dbcsr_t_put_block(tensor, ind_nd, blk_size, block)
         DEALLOCATE (block)
      ENDDO

   END SUBROUTINE
# 589 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
# 590 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

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

# 624 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      REAL(KIND=real_8), ALLOCATABLE, &
         DIMENSION(:,:) :: array_1_2d, &
                                             array_2_2d, &
                                             array_3_2d, &
                                             array_1_2d_full, &
                                             array_2_2d_full, &
                                             array_3_0_2d, &
                                             array_1_rs2d, &
                                             array_2_rs2d, &
                                             array_3_rs2d, &
                                             array_3_0_rs2d
# 624 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      REAL(KIND=real_8), ALLOCATABLE, &
         DIMENSION(:,:,:) :: array_1_3d, &
                                             array_2_3d, &
                                             array_3_3d, &
                                             array_1_3d_full, &
                                             array_2_3d_full, &
                                             array_3_0_3d, &
                                             array_1_rs3d, &
                                             array_2_rs3d, &
                                             array_3_rs3d, &
                                             array_3_0_rs3d
# 624 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      REAL(KIND=real_8), ALLOCATABLE, &
         DIMENSION(:,:,:,:) :: array_1_4d, &
                                             array_2_4d, &
                                             array_3_4d, &
                                             array_1_4d_full, &
                                             array_2_4d_full, &
                                             array_3_0_4d, &
                                             array_1_rs4d, &
                                             array_2_rs4d, &
                                             array_3_rs4d, &
                                             array_3_0_rs4d
# 636 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
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
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (2)
         CALL dist_sparse_tensor_to_repl_dense_array(tensor_3, array_3_0_2d)
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (3)
         CALL dist_sparse_tensor_to_repl_dense_array(tensor_3, array_3_0_3d)
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (4)
         CALL dist_sparse_tensor_to_repl_dense_array(tensor_3, array_3_0_4d)
# 681 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
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
# 727 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      SELECT CASE (dbcsr_t_ndims(tensor_1))
# 729 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (2)
# 731 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CALL dist_sparse_tensor_to_repl_dense_array(tensor_1, array_1_2d_full)
         CALL allocate_any(array_1_2d, shape_spec=SHAPE(array_1_2d_full))
         array_1_2d = 0.0_real_8
         array_1_2d(bounds_t1(1, 1):bounds_t1(2, 1), bounds_t1(1, 2):bounds_t1(2, 2)) = &
         array_1_2d_full(bounds_t1(1, 1):bounds_t1(2, 1), bounds_t1(1, 2):bounds_t1(2, 2))
# 739 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

# 729 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (3)
# 731 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CALL dist_sparse_tensor_to_repl_dense_array(tensor_1, array_1_3d_full)
         CALL allocate_any(array_1_3d, shape_spec=SHAPE(array_1_3d_full))
         array_1_3d = 0.0_real_8
         array_1_3d(bounds_t1(1, 1):bounds_t1(2, 1), bounds_t1(1, 2):bounds_t1(2, 2), bounds_t1(1, 3):bounds_t1(2, 3)) = &
         array_1_3d_full(bounds_t1(1, 1):bounds_t1(2, 1), bounds_t1(1, 2):bounds_t1(2, 2), bounds_t1(1, 3):bounds_t1(2, 3))
# 739 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

# 729 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (4)
# 731 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CALL dist_sparse_tensor_to_repl_dense_array(tensor_1, array_1_4d_full)
         CALL allocate_any(array_1_4d, shape_spec=SHAPE(array_1_4d_full))
         array_1_4d = 0.0_real_8
         array_1_4d(bounds_t1(1, 1):bounds_t1(2, 1), bounds_t1(1, 2):bounds_t1(2, 2), bounds_t1(1, 3):bounds_t1(2, 3),&
# 734 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
             & bounds_t1(1, 4):bounds_t1(2, 4)) = &
         array_1_4d_full(bounds_t1(1, 1):bounds_t1(2, 1), bounds_t1(1, 2):bounds_t1(2, 2), bounds_t1(1, 3):bounds_t1(2, 3),&
# 735 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
             & bounds_t1(1, 4):bounds_t1(2, 4))
# 739 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

# 741 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      END SELECT
# 727 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      SELECT CASE (dbcsr_t_ndims(tensor_2))
# 729 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (2)
# 731 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CALL dist_sparse_tensor_to_repl_dense_array(tensor_2, array_2_2d_full)
         CALL allocate_any(array_2_2d, shape_spec=SHAPE(array_2_2d_full))
         array_2_2d = 0.0_real_8
         array_2_2d(bounds_t2(1, 1):bounds_t2(2, 1), bounds_t2(1, 2):bounds_t2(2, 2)) = &
         array_2_2d_full(bounds_t2(1, 1):bounds_t2(2, 1), bounds_t2(1, 2):bounds_t2(2, 2))
# 739 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

# 729 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (3)
# 731 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CALL dist_sparse_tensor_to_repl_dense_array(tensor_2, array_2_3d_full)
         CALL allocate_any(array_2_3d, shape_spec=SHAPE(array_2_3d_full))
         array_2_3d = 0.0_real_8
         array_2_3d(bounds_t2(1, 1):bounds_t2(2, 1), bounds_t2(1, 2):bounds_t2(2, 2), bounds_t2(1, 3):bounds_t2(2, 3)) = &
         array_2_3d_full(bounds_t2(1, 1):bounds_t2(2, 1), bounds_t2(1, 2):bounds_t2(2, 2), bounds_t2(1, 3):bounds_t2(2, 3))
# 739 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

# 729 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (4)
# 731 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CALL dist_sparse_tensor_to_repl_dense_array(tensor_2, array_2_4d_full)
         CALL allocate_any(array_2_4d, shape_spec=SHAPE(array_2_4d_full))
         array_2_4d = 0.0_real_8
         array_2_4d(bounds_t2(1, 1):bounds_t2(2, 1), bounds_t2(1, 2):bounds_t2(2, 2), bounds_t2(1, 3):bounds_t2(2, 3),&
# 734 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
             & bounds_t2(1, 4):bounds_t2(2, 4)) = &
         array_2_4d_full(bounds_t2(1, 1):bounds_t2(2, 1), bounds_t2(1, 2):bounds_t2(2, 2), bounds_t2(1, 3):bounds_t2(2, 3),&
# 735 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
             & bounds_t2(1, 4):bounds_t2(2, 4))
# 739 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

# 741 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      END SELECT
# 727 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      SELECT CASE (dbcsr_t_ndims(tensor_3))
# 729 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (2)
# 737 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CALL dist_sparse_tensor_to_repl_dense_array(tensor_3, array_3_2d)
# 739 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

# 729 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (3)
# 737 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CALL dist_sparse_tensor_to_repl_dense_array(tensor_3, array_3_3d)
# 739 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

# 729 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (4)
# 737 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CALL dist_sparse_tensor_to_repl_dense_array(tensor_3, array_3_4d)
# 739 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

# 741 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      END SELECT
# 743 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

      ! Get array sizes

# 747 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      SELECT CASE (dbcsr_t_ndims(tensor_1))
# 749 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (2)
         CALL allocate_any(size_1, source=SHAPE(array_1_2d))

# 749 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (3)
         CALL allocate_any(size_1, source=SHAPE(array_1_3d))

# 749 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (4)
         CALL allocate_any(size_1, source=SHAPE(array_1_4d))

# 753 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      END SELECT
# 747 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      SELECT CASE (dbcsr_t_ndims(tensor_2))
# 749 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (2)
         CALL allocate_any(size_2, source=SHAPE(array_2_2d))

# 749 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (3)
         CALL allocate_any(size_2, source=SHAPE(array_2_3d))

# 749 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (4)
         CALL allocate_any(size_2, source=SHAPE(array_2_4d))

# 753 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      END SELECT
# 747 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      SELECT CASE (dbcsr_t_ndims(tensor_3))
# 749 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (2)
         CALL allocate_any(size_3, source=SHAPE(array_3_2d))

# 749 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (3)
         CALL allocate_any(size_3, source=SHAPE(array_3_3d))

# 749 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      CASE (4)
         CALL allocate_any(size_3, source=SHAPE(array_3_4d))

# 753 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      END SELECT
# 755 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

# 757 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      ALLOCATE (order_t1 (dbcsr_t_ndims(tensor_1)))
# 757 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      ALLOCATE (order_t2 (dbcsr_t_ndims(tensor_2)))
# 757 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      ALLOCATE (order_t3 (dbcsr_t_ndims(tensor_3)))
# 759 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

      ASSOCIATE (map_t1_1=>notcontract_1, map_t1_2=>contract_1, &
                 map_t2_1=>notcontract_2, map_t2_2=>contract_2, &
                 map_t3_1=>map_1, map_t3_2=>map_2)

# 765 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         order_t1 (:) = dbcsr_t_inverse_order([map_t1_1, map_t1_2])

         SELECT CASE (dbcsr_t_ndims(tensor_1))
# 769 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CASE (2)
            CALL allocate_any(array_1_rs2d, source=array_1_2d, order=order_t1)
            CALL allocate_any(array_1_mm, sizes_2d(size_1, map_t1_1, map_t1_2))
            array_1_mm(:, :) = RESHAPE(array_1_rs2d, SHAPE(array_1_mm))
# 769 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CASE (3)
            CALL allocate_any(array_1_rs3d, source=array_1_3d, order=order_t1)
            CALL allocate_any(array_1_mm, sizes_2d(size_1, map_t1_1, map_t1_2))
            array_1_mm(:, :) = RESHAPE(array_1_rs3d, SHAPE(array_1_mm))
# 769 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CASE (4)
            CALL allocate_any(array_1_rs4d, source=array_1_4d, order=order_t1)
            CALL allocate_any(array_1_mm, sizes_2d(size_1, map_t1_1, map_t1_2))
            array_1_mm(:, :) = RESHAPE(array_1_rs4d, SHAPE(array_1_mm))
# 774 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         END SELECT
# 765 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         order_t2 (:) = dbcsr_t_inverse_order([map_t2_1, map_t2_2])

         SELECT CASE (dbcsr_t_ndims(tensor_2))
# 769 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CASE (2)
            CALL allocate_any(array_2_rs2d, source=array_2_2d, order=order_t2)
            CALL allocate_any(array_2_mm, sizes_2d(size_2, map_t2_1, map_t2_2))
            array_2_mm(:, :) = RESHAPE(array_2_rs2d, SHAPE(array_2_mm))
# 769 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CASE (3)
            CALL allocate_any(array_2_rs3d, source=array_2_3d, order=order_t2)
            CALL allocate_any(array_2_mm, sizes_2d(size_2, map_t2_1, map_t2_2))
            array_2_mm(:, :) = RESHAPE(array_2_rs3d, SHAPE(array_2_mm))
# 769 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CASE (4)
            CALL allocate_any(array_2_rs4d, source=array_2_4d, order=order_t2)
            CALL allocate_any(array_2_mm, sizes_2d(size_2, map_t2_1, map_t2_2))
            array_2_mm(:, :) = RESHAPE(array_2_rs4d, SHAPE(array_2_mm))
# 774 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         END SELECT
# 765 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         order_t3 (:) = dbcsr_t_inverse_order([map_t3_1, map_t3_2])

         SELECT CASE (dbcsr_t_ndims(tensor_3))
# 769 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CASE (2)
            CALL allocate_any(array_3_rs2d, source=array_3_2d, order=order_t3)
            CALL allocate_any(array_3_mm, sizes_2d(size_3, map_t3_1, map_t3_2))
            array_3_mm(:, :) = RESHAPE(array_3_rs2d, SHAPE(array_3_mm))
# 769 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CASE (3)
            CALL allocate_any(array_3_rs3d, source=array_3_3d, order=order_t3)
            CALL allocate_any(array_3_mm, sizes_2d(size_3, map_t3_1, map_t3_2))
            array_3_mm(:, :) = RESHAPE(array_3_rs3d, SHAPE(array_3_mm))
# 769 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CASE (4)
            CALL allocate_any(array_3_rs4d, source=array_3_4d, order=order_t3)
            CALL allocate_any(array_3_mm, sizes_2d(size_3, map_t3_1, map_t3_2))
            array_3_mm(:, :) = RESHAPE(array_3_rs4d, SHAPE(array_3_mm))
# 774 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         END SELECT
# 776 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

         SELECT CASE (dbcsr_t_ndims(tensor_3))
# 779 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CASE (2)
            CALL allocate_any(array_3_0_rs2d, source=array_3_0_2d, order=order_t3)
            CALL allocate_any(array_3_test_mm, sizes_2d(size_3, map_t3_1, map_t3_2))
            array_3_test_mm(:, :) = RESHAPE(array_3_0_rs2d, SHAPE(array_3_mm))
# 779 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CASE (3)
            CALL allocate_any(array_3_0_rs3d, source=array_3_0_3d, order=order_t3)
            CALL allocate_any(array_3_test_mm, sizes_2d(size_3, map_t3_1, map_t3_2))
            array_3_test_mm(:, :) = RESHAPE(array_3_0_rs3d, SHAPE(array_3_mm))
# 779 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         CASE (4)
            CALL allocate_any(array_3_0_rs4d, source=array_3_0_4d, order=order_t3)
            CALL allocate_any(array_3_test_mm, sizes_2d(size_3, map_t3_1, map_t3_2))
            array_3_test_mm(:, :) = RESHAPE(array_3_0_rs4d, SHAPE(array_3_mm))
# 784 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         END SELECT

         array_3_test_mm(:, :) = beta%r_dp*array_3_test_mm(:, :) + alpha%r_dp*MATMUL(array_1_mm, transpose(array_2_mm))

      END ASSOCIATE

      eql_diff = MAXVAL(ABS(array_3_test_mm(:, :) - array_3_mm(:, :)))
      notzero = MAXVAL(ABS(array_3_test_mm(:, :))) .GT. 1.0E-12_real_4

      eql = eql_diff .LT. 1.0E-11_real_4

      IF (.NOT. eql .OR. .NOT. notzero) THEN
         IF (io_unit > 0) WRITE (io_unit, *) 'Test failed!', eql_diff
         DBCSR_ABORT('')
      ELSE
         IF (io_unit > 0) WRITE (io_unit, *) 'Test passed!', eql_diff
      ENDIF

   END SUBROUTINE

# 805 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE write_2d_array(array, unit_nr, array_compare)
      !! Write nd array
      REAL(KIND=real_8), DIMENSION(:,:), INTENT(IN) :: array
      INTEGER, INTENT(IN)                                             :: unit_nr
      REAL(KIND=real_8), DIMENSION(:,:), INTENT(IN), OPTIONAL :: array_compare
      INTEGER                                            :: i_1, i_2
# 812 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      DO i_2 = 1, SIZE(array, 2)
# 812 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      DO i_1 = 1, SIZE(array, 1)
# 814 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         IF (INT(array(i_1, i_2), KIND=int_8) .GT. 0) THEN
            WRITE (unit_nr, '(3I20)', advance='no') i_1, i_2, INT(array(i_1, i_2), &
                                                                                              KIND=int_8)
            IF (PRESENT(array_compare)) THEN
               WRITE (unit_nr, '(I20)', advance='no') INT(array_compare(i_1, i_2), KIND=int_8)
            ENDIF
            WRITE (unit_nr, *)
         ENDIF
# 823 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      ENDDO
# 823 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      ENDDO
# 825 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   END SUBROUTINE
# 805 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE write_3d_array(array, unit_nr, array_compare)
      !! Write nd array
      REAL(KIND=real_8), DIMENSION(:,:,:), INTENT(IN) :: array
      INTEGER, INTENT(IN)                                             :: unit_nr
      REAL(KIND=real_8), DIMENSION(:,:,:), INTENT(IN), OPTIONAL :: array_compare
      INTEGER                                            :: i_1, i_2, i_3
# 812 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      DO i_3 = 1, SIZE(array, 3)
# 812 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      DO i_2 = 1, SIZE(array, 2)
# 812 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      DO i_1 = 1, SIZE(array, 1)
# 814 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         IF (INT(array(i_1, i_2, i_3), KIND=int_8) .GT. 0) THEN
            WRITE (unit_nr, '(4I20)', advance='no') i_1, i_2, i_3, INT(array(i_1, i_2, i_3), &
                                                                                              KIND=int_8)
            IF (PRESENT(array_compare)) THEN
               WRITE (unit_nr, '(I20)', advance='no') INT(array_compare(i_1, i_2, i_3), KIND=int_8)
            ENDIF
            WRITE (unit_nr, *)
         ENDIF
# 823 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      ENDDO
# 823 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      ENDDO
# 823 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      ENDDO
# 825 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   END SUBROUTINE
# 805 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   SUBROUTINE write_4d_array(array, unit_nr, array_compare)
      !! Write nd array
      REAL(KIND=real_8), DIMENSION(:,:,:,:), INTENT(IN) :: array
      INTEGER, INTENT(IN)                                             :: unit_nr
      REAL(KIND=real_8), DIMENSION(:,:,:,:), INTENT(IN), OPTIONAL :: array_compare
      INTEGER                                            :: i_1, i_2, i_3, i_4
# 812 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      DO i_4 = 1, SIZE(array, 4)
# 812 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      DO i_3 = 1, SIZE(array, 3)
# 812 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      DO i_2 = 1, SIZE(array, 2)
# 812 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      DO i_1 = 1, SIZE(array, 1)
# 814 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
         IF (INT(array(i_1, i_2, i_3, i_4), KIND=int_8) .GT. 0) THEN
            WRITE (unit_nr, '(5I20)', advance='no') i_1, i_2, i_3, i_4, INT(array(i_1, i_2, i_3, i_4), &
                                                                                              KIND=int_8)
            IF (PRESENT(array_compare)) THEN
               WRITE (unit_nr, '(I20)', advance='no') INT(array_compare(i_1, i_2, i_3, i_4), KIND=int_8)
            ENDIF
            WRITE (unit_nr, *)
         ENDIF
# 823 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      ENDDO
# 823 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      ENDDO
# 823 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      ENDDO
# 823 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
      ENDDO
# 825 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"
   END SUBROUTINE
# 827 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_test.F"

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
