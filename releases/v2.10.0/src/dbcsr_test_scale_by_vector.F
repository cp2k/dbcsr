!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_test_scale_by_vector
   !! Tests for DBCSR scale_by_vector
   USE dbcsr_data_methods, ONLY: dbcsr_data_get_sizes, &
                                 dbcsr_data_init, &
                                 dbcsr_data_new, &
                                 dbcsr_data_release, &
                                 dbcsr_type_1d_to_2d
   USE dbcsr_dist_methods, ONLY: dbcsr_distribution_new, &
                                 dbcsr_distribution_release
   USE dbcsr_kinds, ONLY: real_8
   USE dbcsr_methods, ONLY: &
      dbcsr_get_data_type, &
      dbcsr_get_matrix_type, dbcsr_name, dbcsr_nblkcols_total, dbcsr_nblkrows_total, &
      dbcsr_nfullcols_total, dbcsr_nfullrows_total, dbcsr_release

   USE dbcsr_mpiwrap, ONLY: mp_environ, mp_comm_type
   USE dbcsr_test_methods, ONLY: dbcsr_make_random_block_sizes, &
                                 dbcsr_make_random_matrix, &
                                 dbcsr_random_dist, &
                                 dbcsr_to_dense_local
   USE dbcsr_transformations, ONLY: dbcsr_redistribute, &
                                    dbcsr_new_transposed
   USE dbcsr_types, ONLY: &
      dbcsr_data_obj, dbcsr_distribution_obj, dbcsr_mp_obj, dbcsr_type, &
      dbcsr_type_antisymmetric, dbcsr_type_no_symmetry, dbcsr_type_symmetric, &
      dbcsr_type_real_4, dbcsr_type_real_8, &
      dbcsr_type_complex_4, dbcsr_type_complex_8
   USE dbcsr_work_operations, ONLY: dbcsr_create
   USE dbcsr_operations, ONLY: dbcsr_scale_by_vector
   USE dbcsr_dist_util, ONLY: dbcsr_checksum
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_num_threads

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: dbcsr_test_scale_by_vectors

   LOGICAL, PARAMETER :: debug_mod = .FALSE.

CONTAINS

   FUNCTION dbcsr_test_scale_by_vectors(test_name, mp_group, mp_env, npdims, io_unit, &
                                        matrix_size, bs_m, bs_n, sparsity, do_exact_comparison) RESULT(success)
      !! Performs a variety of matrix multiplies of same matrices on different
      !! processor grids

      CHARACTER(len=*), INTENT(IN)                       :: test_name
      TYPE(mp_comm_type), INTENT(IN)                                :: mp_group
         !! MPI communicator
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      INTEGER, DIMENSION(2), INTENT(IN)                  :: npdims
      INTEGER, INTENT(IN)                                :: io_unit
         !! which unit to write to, if not negative
      INTEGER, DIMENSION(2), INTENT(IN)                  :: matrix_size
         !! size of matrix to test
      INTEGER, DIMENSION(:), INTENT(IN)                  :: bs_m, bs_n
         !! block sizes of the 2 dimension
         !! block sizes of the 2 dimension
      REAL(real_8), INTENT(IN)                           :: sparsity
         !! sparsity of the matrix to create
      LOGICAL, INTENT(IN)                                :: do_exact_comparison
         !! whether or not to do exact comparison for the matrix values

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_test_scale_by_vectors'

      CHARACTER, DIMENSION(3), PARAMETER :: symmetries = [dbcsr_type_no_symmetry, dbcsr_type_symmetric, dbcsr_type_antisymmetric]
      INTEGER, DIMENSION(4), PARAMETER :: types = [dbcsr_type_real_4, dbcsr_type_real_8, dbcsr_type_complex_4, dbcsr_type_complex_8]

      CHARACTER                                          :: symm
      INTEGER                                            :: handle, isymm, itype, mynode, &
                                                            numnodes, numthreads, type, nrows
      INTEGER, DIMENSION(:), POINTER, CONTIGUOUS         :: sizes_m, sizes_n, sizes_1
      LOGICAL                                            :: success
      TYPE(dbcsr_data_obj)                               :: vector_data
      TYPE(dbcsr_type)                                   :: matrix, vector

      CALL timeset(routineN, handle)
      NULLIFY (sizes_m, sizes_n, sizes_1)
      !
      ! print
      CALL mp_environ(numnodes, mynode, mp_group)
      IF (io_unit > 0) THEN
         WRITE (io_unit, *) 'test_name ', test_name
         numthreads = 1
!$OMP PARALLEL
!$OMP MASTER
!$       numthreads = omp_get_num_threads()
!$OMP END MASTER
!$OMP END PARALLEL
         WRITE (io_unit, *) 'numthreads', numthreads
         WRITE (io_unit, *) 'numnodes', numnodes
         WRITE (io_unit, *) 'matrix_size', matrix_size
         WRITE (io_unit, *) 'sparsity', sparsity
         WRITE (io_unit, *) 'bs_m', bs_m
         WRITE (io_unit, *) 'bs_n', bs_n
      END IF

      success = .TRUE.
      !
      !
      ! loop over symmetry
      DO isymm = 1, SIZE(symmetries)
         symm = symmetries(isymm)

         IF (matrix_size(1) /= matrix_size(2) .AND. symm /= dbcsr_type_no_symmetry) &
            CYCLE

         !
         ! loop over types
         DO itype = 1, SIZE(types)
            type = types(itype)

            !
            ! Create the row/column block sizes.
            CALL dbcsr_make_random_block_sizes(sizes_m, matrix_size(1), bs_m)
            CALL dbcsr_make_random_block_sizes(sizes_n, matrix_size(2), bs_n)
            ALLOCATE (sizes_1(1))
            sizes_1 = 1

            !
            ! Create the undistributed matrices.
            CALL dbcsr_make_random_matrix(matrix, sizes_m, sizes_n, "Matrix", &
                                          sparsity, mp_group, data_type=type, symmetry=symm)

            !
            ! Use a very skinny matrix to generate our test data
            CALL dbcsr_make_random_matrix(vector, sizes_n, sizes_1, "Vector", 0.0_real_8, mp_group, data_type=type)

            DEALLOCATE (sizes_m, sizes_n, sizes_1)

            !
            ! Densify the the vector
            nrows = dbcsr_nfullrows_total(vector)
            CALL dbcsr_data_init(vector_data)
            CALL dbcsr_data_new(vector_data, type, data_size=nrows)
            CALL dbcsr_to_dense_local(vector, vector_data)

            IF (debug_mod .AND. io_unit > 0) THEN
               CALL write_1d_data_obj(io_unit, vector_data)
               CALL write_matrix_dense(io_unit, matrix)
            END IF

            !
            ! Prepare test parameters
            success = test_scale_by_vector(mp_env, npdims, matrix, vector_data, do_exact_comparison) .AND. success

            IF (io_unit > 0) THEN
               IF (success) THEN
                  WRITE (io_unit, *) REPEAT("*", 70)
                  WRITE (io_unit, *) " -- TESTING dbcsr_scale_by_vector (", &
                     dbcsr_get_data_type(matrix), &
                     dbcsr_get_matrix_type(matrix), &
                     do_exact_comparison, &
                     ") ............... PASSED !"
                  WRITE (io_unit, *) REPEAT("*", 70)
               ELSE
                  WRITE (io_unit, *) REPEAT("*", 70)
                  WRITE (io_unit, *) " -- TESTING dbcsr_scale_by_vector (", &
                     dbcsr_get_data_type(matrix), &
                     dbcsr_get_matrix_type(matrix), &
                     do_exact_comparison, &
                     ") ............... FAILED !"
                  WRITE (io_unit, *) REPEAT("*", 70)
               END IF
            END IF

            !
            ! cleanup
            CALL dbcsr_release(matrix)
            CALL dbcsr_release(vector)
            CALL dbcsr_data_release(vector_data)

         END DO ! itype
      END DO !isymm

      CALL timestop(handle)
   END FUNCTION

   SUBROUTINE write_1d_data_obj(io_unit, vector)
      INTEGER, INTENT(IN)               :: io_unit
      TYPE(dbcsr_data_obj), INTENT(IN)  :: vector

      INTEGER                           :: i, sz
      LOGICAL                           :: valid

      CALL dbcsr_data_get_sizes(vector, sz, valid)

      IF (.NOT. valid) &
         RETURN

      SELECT CASE (vector%d%data_type)
      CASE (dbcsr_type_real_4)
         WRITE (io_unit, "(A,I3)") "Vector dbcsr_type_real_4, size:", sz
         DO i = 1, SIZE(vector%d%r_sp)
            WRITE (io_unit, '(T2,A,I3,A,E15.7,A)') 'vector(', i, ')=', vector%d%r_sp(i), ';'
         END DO
      CASE (dbcsr_type_real_8)
         WRITE (io_unit, "(A,I3)") "Vector dbcsr_type_real_8, size:", sz
         DO i = 1, SIZE(vector%d%r_dp)
            WRITE (io_unit, '(T2,A,I3,A,E15.7,A)') 'vector(', i, ')=', vector%d%r_dp(i), ';'
         END DO
      CASE (dbcsr_type_complex_4)
         WRITE (io_unit, "(A,I3)") "Vector dbcsr_type_complex_4, size:", sz
         DO i = 1, SIZE(vector%d%c_sp)
            WRITE (io_unit, '(T2,A,I3,A,E15.7,SP,E15.7,"i",A)') 'vector(', i, ')=', vector%d%c_sp(i), ';'
         END DO
      CASE (dbcsr_type_complex_8)
         WRITE (io_unit, "(A,I3)") "Vector dbcsr_type_complex_8, size:", sz
         DO i = 1, SIZE(vector%d%c_dp)
            WRITE (io_unit, '(T2,A,I3,A,E15.7,SP,E15.7,"i",A)') 'vector(', i, ')=', vector%d%c_dp(i), ';'
         END DO
      END SELECT
   END SUBROUTINE

   SUBROUTINE write_matrix_dense(io_unit, matrix)
      INTEGER, INTENT(IN)               :: io_unit
      TYPE(dbcsr_type), INTENT(IN)      :: matrix

      TYPE(dbcsr_data_obj)              :: mdata
      INTEGER                           :: i, j, sz(2)
      LOGICAL                           :: valid

      CALL dbcsr_data_init(mdata)
      CALL dbcsr_data_new(mdata, dbcsr_type_1d_to_2d(matrix%data_type), &
                          data_size=dbcsr_nfullrows_total(matrix), data_size2=dbcsr_nfullcols_total(matrix))
      CALL dbcsr_to_dense_local(matrix, mdata)

      CALL dbcsr_data_get_sizes(mdata, sz, valid)

      IF (.NOT. valid) &
         CALL dbcsr_abort(__LOCATION__, &
                          "densification failed?!")

      SELECT CASE (matrix%data_type)
      CASE (dbcsr_type_real_4)
         WRITE (io_unit, "(A,I3,I3)") "Matrix dbcsr_type_real_4, size:", sz
         DO j = 1, SIZE(mdata%d%r2_sp, 2)
            DO i = 1, SIZE(mdata%d%r2_sp, 1)
               WRITE (io_unit, '(T2,A,I3,A,I3,A,E15.7,A)') 'matrix(', i, ',', j, ')=', mdata%d%r2_sp(i, j), ';'
            END DO
         END DO
      CASE (dbcsr_type_real_8)
         WRITE (io_unit, "(A,I3,I3)") "Matrix dbcsr_type_real_8, size:", sz
         DO j = 1, SIZE(mdata%d%r2_dp, 2)
            DO i = 1, SIZE(mdata%d%r2_dp, 1)
               WRITE (io_unit, '(T2,A,I3,A,I3,A,E15.7,A)') 'matrix(', i, ',', j, ')=', mdata%d%r2_dp(i, j), ';'
            END DO
         END DO
      CASE (dbcsr_type_complex_4)
         WRITE (io_unit, "(A,I3,I3)") "Matrix dbcsr_type_complex_4, size:", sz
         DO j = 1, SIZE(mdata%d%c2_sp, 2)
            DO i = 1, SIZE(mdata%d%c2_sp, 1)
               WRITE (io_unit, '(T2,A,I3,A,I3,A,E15.7,SP,E15.7,"i",A)') 'matrix(', i, ',', j, ')=', mdata%d%c2_sp(i, j), ';'
            END DO
         END DO
      CASE (dbcsr_type_complex_8)
         WRITE (io_unit, "(A,I3,I3)") "Matrix dbcsr_type_complex_8, size:", sz
         DO j = 1, SIZE(mdata%d%c2_dp, 2)
            DO i = 1, SIZE(mdata%d%c2_dp, 1)
               WRITE (io_unit, '(T2,A,I3,A,I3,A,E15.7,SP,E15.7,"i",A)') 'matrix(', i, ',', j, ')=', mdata%d%c2_dp(i, j), ';'
            END DO
         END DO
      END SELECT

      CALL dbcsr_data_release(mdata)
   END SUBROUTINE

   FUNCTION test_scale_by_vector(mp_env, npdims, matrix, vector, do_exact_comparison) RESULT(res)
      !! Performs T(v * T(M)) == M*v
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      INTEGER, DIMENSION(2), INTENT(IN)                  :: npdims
         !! processor grid
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! matrix to scale
      TYPE(dbcsr_data_obj), INTENT(IN)                   :: vector
         !! scaling vector
      LOGICAL, INTENT(IN)                                :: do_exact_comparison
         !! whether to do an exact comparison (via densification)

      INTEGER                                            :: handle
      INTEGER, DIMENSION(:), POINTER, CONTIGUOUS         :: col_dist, row_dist
      TYPE(dbcsr_distribution_obj)                       :: dist
      TYPE(dbcsr_type)                                   :: matrix_right, matrix_left, matrix_left_transposed
      LOGICAL                                            :: res

      CHARACTER(len=*), PARAMETER :: routineN = 'test_scale_by_vector'

      CALL timeset(routineN, handle)
      NULLIFY (row_dist, col_dist)

      ! Row & column distributions
      CALL dbcsr_random_dist(row_dist, dbcsr_nblkrows_total(matrix), npdims(1))
      CALL dbcsr_random_dist(col_dist, dbcsr_nblkcols_total(matrix), npdims(2))
      CALL dbcsr_distribution_new(dist, mp_env, row_dist, col_dist, reuse_arrays=.TRUE.)

      ! Create redistributed matrix
      CALL dbcsr_create(matrix_right, "RHS Test for "//TRIM(dbcsr_name(matrix)), &
                        dist, dbcsr_get_matrix_type(matrix), &
                        row_blk_size_obj=matrix%row_blk_size, &
                        col_blk_size_obj=matrix%col_blk_size, &
                        data_type=dbcsr_get_data_type(matrix))

      CALL dbcsr_distribution_release(dist)
      CALL dbcsr_redistribute(matrix, matrix_right)

      CALL dbcsr_new_transposed(matrix_left, matrix_right)

      !
      ! Perform scaling, once from right, once from left
      CALL dbcsr_scale_by_vector(matrix_right, vector, side="right")
      CALL dbcsr_scale_by_vector(matrix_left, vector, side="left")

      ! for the comparison we need the transposed LHS again
      CALL dbcsr_new_transposed(matrix_left_transposed, matrix_left)

      ! now compare either exactly via densification or less exactly via checksums
      IF (do_exact_comparison) THEN
         BLOCK
            TYPE(dbcsr_data_obj)              :: mdata_left, mdata_right

            CALL dbcsr_data_init(mdata_left)
            CALL dbcsr_data_new(mdata_left, dbcsr_type_1d_to_2d(dbcsr_get_data_type(matrix_left_transposed)), &
                                data_size=dbcsr_nfullrows_total(matrix_left_transposed), &
                                data_size2=dbcsr_nfullcols_total(matrix_left_transposed))
            CALL dbcsr_to_dense_local(matrix_left_transposed, mdata_left)

            CALL dbcsr_data_init(mdata_right)
            CALL dbcsr_data_new(mdata_right, dbcsr_type_1d_to_2d(dbcsr_get_data_type(matrix_right)), &
                                data_size=dbcsr_nfullrows_total(matrix_right), data_size2=dbcsr_nfullcols_total(matrix_right))
            CALL dbcsr_to_dense_local(matrix_right, mdata_right)

            SELECT CASE (dbcsr_get_data_type(matrix_right))
            CASE (dbcsr_type_real_4)
               res = ALL(ABS(mdata_right%d%r2_sp - mdata_left%d%r2_sp) < 1.0D-5)
            CASE (dbcsr_type_real_8)
               res = ALL(ABS(mdata_right%d%r2_dp - mdata_left%d%r2_dp) < 1.0D-5)
            CASE (dbcsr_type_complex_4)
               res = ALL(ABS(mdata_right%d%c2_sp - mdata_left%d%c2_sp) < 1.0D-5)
            CASE (dbcsr_type_complex_8)
               res = ALL(ABS(mdata_right%d%c2_dp - mdata_left%d%c2_dp) < 1.0D-5)
            END SELECT

            CALL dbcsr_data_release(mdata_left)
            CALL dbcsr_data_release(mdata_right)
         END BLOCK
      ELSE
         !
         ! Calculate checksums and set result
         res = ABS(dbcsr_checksum(matrix_right, pos=.TRUE.) - dbcsr_checksum(matrix_left_transposed, pos=.TRUE.)) < 1.0D-5
      END IF

      CALL dbcsr_release(matrix_left)
      CALL dbcsr_release(matrix_left_transposed)
      CALL dbcsr_release(matrix_right)

      CALL timestop(handle)
   END FUNCTION
END MODULE
