!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

PROGRAM dbcsr_test_csr_conversions
   !! Testing DBCSR to CSR conversion with random matrices
   USE dbcsr_kinds, ONLY: dp, real_8
   USE dbcsr_api, ONLY: &
      dbcsr_convert_csr_to_dbcsr, dbcsr_convert_dbcsr_to_csr, &
      dbcsr_csr_create_from_dbcsr, dbcsr_csr_destroy, &
      dbcsr_csr_eqrow_ceil_dist, dbcsr_csr_type, dbcsr_add, dbcsr_copy, dbcsr_create, &
      dbcsr_distribution_get, dbcsr_distribution_new, dbcsr_distribution_release, &
      dbcsr_distribution_type, dbcsr_finalize, dbcsr_finalize_lib, dbcsr_get_stored_coordinates, &
      dbcsr_init_lib, dbcsr_nblkcols_total, dbcsr_nblkrows_total, dbcsr_norm, &
      dbcsr_norm_maxabsnorm, dbcsr_put_block, dbcsr_release, dbcsr_to_csr_filter, dbcsr_type, &
      dbcsr_type_no_symmetry, dbcsr_type_real_8, dbcsr_print_statistics
   USE dbcsr_machine, ONLY: default_output_unit
   USE dbcsr_mpiwrap, ONLY: mp_bcast, &
                            mp_cart_create, &
                            mp_comm_free, &
                            mp_environ, &
                            mp_world_finalize, &
                            mp_world_init, mp_comm_type
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   TYPE(dbcsr_type)              :: matrix_a
   TYPE(dbcsr_csr_type)          :: matrix_b

   INTEGER, DIMENSION(:), POINTER :: col_blk_sizes, row_blk_sizes
   INTEGER                        :: nblkrows_total, nblkcols_total

   INTEGER, DIMENSION(:), POINTER :: col_dist, row_dist

   INTEGER                      :: numnodes, mynode, io_unit

   INTEGER, DIMENSION(2)                    :: npdims, myploc

   INTEGER                      :: max_blks_total, max_blk_size, k, seedsz
   INTEGER, ALLOCATABLE, DIMENSION(:)        ::seed

   REAL                         :: rn
   REAL, ALLOCATABLE, DIMENSION(:)        :: rn_array

   REAL(KIND=real_8)            :: norm, norm_eps, sparsity, eps

   CHARACTER(LEN=10)            :: k_str, mynode_str

   TYPE(mp_comm_type)           :: mp_comm, group

   ! Set up everything as in the dbcsr example codes
   CALL mp_world_init(mp_comm)

   CALL mp_environ(numnodes, mynode, mp_comm)

   io_unit = 0
   IF (mynode .EQ. 0) io_unit = default_output_unit

   CALL dbcsr_init_lib(mp_comm%get_handle(), io_unit)

   npdims(:) = 0
   CALL mp_cart_create(mp_comm, 2, npdims, myploc, group)
   CALL mp_environ(numnodes, mynode, group)

   ! Set seed for random number generator
   CALL RANDOM_SEED(size=seedsz)
   ALLOCATE (seed(seedsz))
   seed = 434358235

   ! Maximum number of blocks and maximum block sizes (in 1 dimension)
   max_blks_total = 50
   max_blk_size = 10

   eps = 0.1_dp ! Filter threshold

   DO k = 1, 100 ! test 100 matrices

      CALL RANDOM_SEED(get=seed)
      CALL mp_bcast(seed, 0, mp_comm)
      CALL RANDOM_SEED(put=seed)

      CALL RANDOM_NUMBER(rn)
      nblkrows_total = FLOOR(rn*(max_blks_total)) + 1

      CALL RANDOM_NUMBER(rn)
      nblkcols_total = FLOOR(rn*(max_blks_total)) + 1

      ALLOCATE (rn_array(MAX(nblkcols_total, nblkrows_total)))
      ALLOCATE (col_blk_sizes(nblkcols_total))
      ALLOCATE (row_blk_sizes(nblkrows_total))
      ALLOCATE (row_dist(nblkrows_total))
      ALLOCATE (col_dist(nblkcols_total))

      CALL RANDOM_NUMBER(rn_array)
      col_blk_sizes = FLOOR(rn_array(1:nblkcols_total)*(max_blk_size)) + 1

      CALL RANDOM_NUMBER(rn_array)
      row_blk_sizes = FLOOR(rn_array(1:nblkrows_total)*(max_blk_size)) + 1

      CALL RANDOM_NUMBER(rn)
      sparsity = rn

      CALL RANDOM_NUMBER(rn_array)
      row_dist = FLOOR(rn_array(1:nblkrows_total)*npdims(1))
      CALL RANDOM_NUMBER(rn_array)
      col_dist = FLOOR(rn_array(1:nblkcols_total)*npdims(2))

      CALL make_random_dbcsr_matrix(matrix_a, group, col_blk_sizes, row_blk_sizes, col_dist, row_dist, sparsity)

      WRITE (UNIT=k_str, FMT='(I0)') k
      WRITE (UNIT=mynode_str, FMT='(I0)') mynode

      CALL csr_conversion_test(matrix_a, matrix_b, norm, 0.0_dp)
      CALL dbcsr_csr_destroy(matrix_b)
      CALL csr_conversion_test(matrix_a, matrix_b, norm_eps, eps)
      CALL dbcsr_csr_destroy(matrix_b)

      IF ((norm > EPSILON(norm)) .OR. (norm_eps > eps)) THEN
         IF (io_unit > 0) WRITE (io_unit, *) "Conversion error > 0 for matrix no.", k_str
         DBCSR_ABORT("Error in csr conversion")
      ELSE
         IF (io_unit > 0) WRITE (io_unit, *) "Conversion OK!"
      END IF

      CALL dbcsr_release(matrix_a)
      DEALLOCATE (rn_array)

   END DO

   DEALLOCATE (seed)

   CALL mp_comm_free(group)
   call dbcsr_print_statistics(.true.)
   CALL dbcsr_finalize_lib()
   CALL mp_world_finalize()

CONTAINS

   SUBROUTINE csr_conversion_test(dbcsr_mat, csr_mat, norm, eps)
      !! Test the conversion by converting to CSR format and converting back,
      !! where the CSR sparsity is defined by some filtering threshold eps.
      !! The maximum norm of the differences between the original and the
      !! back-converted matrix is calculated.

      TYPE(dbcsr_type), INTENT(IN)                       :: dbcsr_mat
      TYPE(dbcsr_csr_type), INTENT(OUT)                  :: csr_mat
      REAL(KIND=real_8), INTENT(OUT)                     :: norm
      REAL(KIND=real_8), INTENT(IN)                      :: eps

      TYPE(dbcsr_type)                                   :: csr_sparsity, dbcsr_mat_conv

      CALL dbcsr_to_csr_filter(dbcsr_mat, csr_sparsity, eps)

      CALL dbcsr_csr_create_from_dbcsr(dbcsr_mat, csr_mat, dbcsr_csr_eqrow_ceil_dist, csr_sparsity)
      CALL dbcsr_convert_dbcsr_to_csr(dbcsr_mat, csr_mat)

      CALL dbcsr_copy(dbcsr_mat_conv, dbcsr_mat)

      CALL dbcsr_convert_csr_to_dbcsr(dbcsr_mat_conv, csr_mat)

      CALL dbcsr_add(dbcsr_mat_conv, dbcsr_mat, 1.0_dp, -1.0_dp)
      CALL dbcsr_norm(dbcsr_mat_conv, dbcsr_norm_maxabsnorm, norm_scalar=norm)

      CALL dbcsr_release(dbcsr_mat_conv)
      CALL dbcsr_release(csr_sparsity)
   END SUBROUTINE csr_conversion_test

   SUBROUTINE make_random_dbcsr_matrix(matrix_a, group, &
      !! Create a DBCSR matrix with random values and random blocks
                                       col_blk_sizes, row_blk_sizes, col_dist, row_dist, sparsity)
      TYPE(dbcsr_type), INTENT(OUT)                      :: matrix_a
      TYPE(mp_comm_type), INTENT(IN)                                :: group
      INTEGER, DIMENSION(:), POINTER                     :: col_blk_sizes, row_blk_sizes, col_dist, &
                                                            row_dist
      REAL(real_8), INTENT(IN)                           :: sparsity

      INTEGER                                            :: col, col_s, max_col_size, max_nze, &
                                                            max_row_size, node_holds_blk, nze, &
                                                            row, row_s
      LOGICAL                                            :: tr
      REAL(real_8)                                       :: rn
      REAL(real_8), ALLOCATABLE, DIMENSION(:)            :: values
      TYPE(dbcsr_distribution_type)                      :: dist

      CALL dbcsr_distribution_new(dist, group=group%get_handle(), row_dist=row_dist, col_dist=col_dist, reuse_arrays=.TRUE.)

      CALL dbcsr_create(matrix=matrix_a, &
                        name="this is my matrix a", &
                        dist=dist, &
                        matrix_type=dbcsr_type_no_symmetry, &
                        row_blk_size=row_blk_sizes, &
                        col_blk_size=col_blk_sizes, &
                        data_type=dbcsr_type_real_8)

      CALL dbcsr_distribution_get(dist, mynode=mynode)

      ! get the maximum block size of the matrix
      max_row_size = MAXVAL(row_blk_sizes)
      max_col_size = MAXVAL(col_blk_sizes)
      max_nze = max_row_size*max_col_size

      ALLOCATE (values(max_nze))

      DO row = 1, dbcsr_nblkrows_total(matrix_a)
         DO col = 1, dbcsr_nblkcols_total(matrix_a)
            CALL RANDOM_NUMBER(rn)
            IF (rn .GT. sparsity) THEN
               tr = .FALSE.
               row_s = row; col_s = col
               CALL dbcsr_get_stored_coordinates(matrix_a, row_s, col_s, node_holds_blk)
               IF (node_holds_blk .EQ. mynode) THEN
                  nze = row_blk_sizes(row_s)*col_blk_sizes(col_s)
                  CALL RANDOM_NUMBER(values(1:nze))
                  CALL dbcsr_put_block(matrix_a, row_s, col_s, values(1:nze))
               END IF
            END IF
         END DO
      END DO
      DEALLOCATE (values)

      CALL dbcsr_finalize(matrix_a)
      CALL dbcsr_distribution_release(dist)
      DEALLOCATE (row_blk_sizes, col_blk_sizes)

   END SUBROUTINE make_random_dbcsr_matrix

END PROGRAM dbcsr_test_csr_conversions
