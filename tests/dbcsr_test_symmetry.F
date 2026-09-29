!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_test_symmetry
   USE dbcsr_api, ONLY: dbcsr_binary_read, dbcsr_binary_write, dbcsr_conjugate_transpose, &
                        dbcsr_copy, dbcsr_create, dbcsr_distribution_new, dbcsr_distribution_release, &
                        dbcsr_distribution_type, dbcsr_dot, dbcsr_filter, dbcsr_finalize, &
                        dbcsr_frobenius_norm, dbcsr_gershgorin_norm, dbcsr_get_data_type, &
                        dbcsr_get_matrix_type, dbcsr_multiply, dbcsr_no_transpose, &
                        dbcsr_put_block, dbcsr_release, dbcsr_transposed, dbcsr_type, &
                        dbcsr_type_antihermitian, dbcsr_type_antisymmetric, dbcsr_type_complex_4, &
                        dbcsr_type_complex_8, dbcsr_type_hermitian, dbcsr_type_no_symmetry, &
                        dbcsr_type_real_4, dbcsr_type_real_8, dbcsr_type_symmetric, dbcsr_work_create
   USE dbcsr_kinds, ONLY: real_8
   USE dbcsr_mpiwrap, ONLY: mp_cart_rank, mp_comm_type
   USE dbcsr_types, ONLY: dbcsr_symmetry_data_type_compatible
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: dbcsr_test_hermitian_symmetry

CONTAINS

   LOGICAL FUNCTION dbcsr_test_hermitian_symmetry(group, npdims, io_unit) RESULT(success)
      TYPE(mp_comm_type), INTENT(IN)             :: group
      INTEGER, DIMENSION(2), INTENT(IN)          :: npdims
      INTEGER, INTENT(IN)                        :: io_unit

      INTEGER, DIMENSION(:), POINTER             :: col_dist, row_dist, col_blk_size, row_blk_size
      INTEGER, DIMENSION(:, :), POINTER          :: pgrid
      INTEGER                                    :: pcol, prow
      COMPLEX(real_8)                             :: dot_value
      TYPE(dbcsr_distribution_type)              :: dist
      TYPE(dbcsr_type)                           :: h, k, identity, h_copy, k_copy, h_transposed, k_transposed, &
                                                    h_product, k_product, h_read, k_read

      success = .TRUE.
      CALL check(dbcsr_api_type_exports(), "high-level symmetry and transpose exports", success, io_unit)
      CALL check_type_combinations(success, io_unit)

      ALLOCATE (pgrid(0:npdims(1) - 1, 0:npdims(2) - 1))
      DO prow = 0, npdims(1) - 1
         DO pcol = 0, npdims(2) - 1
            CALL mp_cart_rank(group, (/prow, pcol/), pgrid(prow, pcol))
         END DO
      END DO
      ALLOCATE (row_dist(2), col_dist(2))
      row_dist = 0
      col_dist = 0
      CALL dbcsr_distribution_new(dist, group=group%get_handle(), pgrid=pgrid, &
                                  row_dist=row_dist, col_dist=col_dist)
      DEALLOCATE (pgrid, row_dist, col_dist)

      CALL make_complex_matrix(h, dist, dbcsr_type_hermitian, &
                               CMPLX(2.0_real_8, 0.0_real_8, KIND=real_8), &
                               CMPLX(1.0_real_8, 2.0_real_8, KIND=real_8), &
                               CMPLX(3.0_real_8, 0.0_real_8, KIND=real_8))
      CALL make_complex_matrix(k, dist, dbcsr_type_antihermitian, &
                               CMPLX(0.0_real_8, 2.0_real_8, KIND=real_8), &
                               CMPLX(1.0_real_8, 2.0_real_8, KIND=real_8), &
                               CMPLX(0.0_real_8, -3.0_real_8, KIND=real_8))
      CALL make_complex_matrix(identity, dist, dbcsr_type_no_symmetry, &
                               CMPLX(1.0_real_8, 0.0_real_8, KIND=real_8), &
                               CMPLX(0.0_real_8, 0.0_real_8, KIND=real_8), &
                               CMPLX(1.0_real_8, 0.0_real_8, KIND=real_8))

      CALL check(dbcsr_get_matrix_type(h) == dbcsr_type_hermitian .AND. &
                 dbcsr_get_data_type(h) == dbcsr_type_complex_8, "Hermitian creation", success, io_unit)
      CALL check(dbcsr_get_matrix_type(k) == dbcsr_type_antihermitian .AND. &
                 dbcsr_get_data_type(k) == dbcsr_type_complex_8, "anti-Hermitian creation", success, io_unit)

      CALL dbcsr_dot(h, h, dot_value)
      CALL check_complex(dot_value, CMPLX(7.0_real_8, 0.0_real_8, KIND=real_8), &
                         "Hermitian bilinear dot", success, io_unit)
      CALL dbcsr_dot(h, h, dot_value, conjugate_a=.TRUE.)
      CALL check_complex(dot_value, CMPLX(23.0_real_8, 0.0_real_8, KIND=real_8), &
                         "Hermitian Frobenius product", success, io_unit)
      CALL dbcsr_dot(k, k, dot_value)
      CALL check_complex(dot_value, CMPLX(-19.0_real_8, 0.0_real_8, KIND=real_8), &
                         "anti-Hermitian bilinear dot", success, io_unit)
      CALL dbcsr_dot(k, k, dot_value, conjugate_a=.TRUE.)
      CALL check_complex(dot_value, CMPLX(23.0_real_8, 0.0_real_8, KIND=real_8), &
                         "anti-Hermitian Frobenius product", success, io_unit)
      CALL dbcsr_dot(h, k, dot_value)
      CALL check_complex(dot_value, CMPLX(0.0_real_8, 3.0_real_8, KIND=real_8), &
                         "mixed H/K bilinear dot", success, io_unit)
      CALL dbcsr_dot(h, k, dot_value, conjugate_a=.TRUE.)
      CALL check_complex(dot_value, CMPLX(0.0_real_8, -5.0_real_8, KIND=real_8), &
                         "mixed H/K Frobenius product", success, io_unit)

      CALL check_real(dbcsr_frobenius_norm(h), SQRT(23.0_real_8), &
                      "Hermitian Frobenius norm", success, io_unit)
      CALL check_real(dbcsr_frobenius_norm(k), SQRT(23.0_real_8), &
                      "anti-Hermitian Frobenius norm", success, io_unit)
      CALL check_real(dbcsr_gershgorin_norm(h), 3.0_real_8 + SQRT(5.0_real_8), &
                      "Hermitian Gershgorin norm", success, io_unit)
      CALL check_real(dbcsr_gershgorin_norm(k), 3.0_real_8 + SQRT(5.0_real_8), &
                      "anti-Hermitian Gershgorin norm", success, io_unit)

      CALL dbcsr_copy(h_copy, h)
      CALL dbcsr_copy(k_copy, k)
      CALL check(dbcsr_get_matrix_type(h_copy) == dbcsr_type_hermitian .AND. &
                 dbcsr_get_matrix_type(k_copy) == dbcsr_type_antihermitian, &
                 "copy preserves H/K symmetry", success, io_unit)

      CALL dbcsr_transposed(h_transposed, h, shallow_data_copy=.FALSE., transpose_distribution=.FALSE.)
      CALL dbcsr_transposed(k_transposed, k, shallow_data_copy=.FALSE., transpose_distribution=.FALSE.)
      CALL dbcsr_dot(h_transposed, h, dot_value)
      CALL check_complex(dot_value, CMPLX(23.0_real_8, 0.0_real_8, KIND=real_8), &
                         "Hermitian transpose", success, io_unit)
      CALL dbcsr_dot(k_transposed, k, dot_value)
      CALL check_complex(dot_value, CMPLX(-23.0_real_8, 0.0_real_8, KIND=real_8), &
                         "anti-Hermitian transpose", success, io_unit)

      CALL dbcsr_filter(h, 0.1_real_8)
      CALL dbcsr_filter(k, 0.1_real_8)
      CALL dbcsr_dot(h, h, dot_value, conjugate_a=.TRUE.)
      CALL check_complex(dot_value, CMPLX(23.0_real_8, 0.0_real_8, KIND=real_8), &
                         "Hermitian filter preserves stored values", success, io_unit)
      CALL dbcsr_dot(k, k, dot_value, conjugate_a=.TRUE.)
      CALL check_complex(dot_value, CMPLX(23.0_real_8, 0.0_real_8, KIND=real_8), &
                         "anti-Hermitian filter preserves stored values", success, io_unit)

      CALL dbcsr_multiply(dbcsr_no_transpose, dbcsr_no_transpose, &
                          CMPLX(1.0_real_8, 0.0_real_8, KIND=real_8), h, identity, &
                          CMPLX(0.0_real_8, 0.0_real_8, KIND=real_8), h_product)
      CALL dbcsr_multiply(dbcsr_conjugate_transpose, dbcsr_no_transpose, &
                          CMPLX(-1.0_real_8, 0.0_real_8, KIND=real_8), k, identity, &
                          CMPLX(0.0_real_8, 0.0_real_8, KIND=real_8), k_product)
      CALL check(dbcsr_get_matrix_type(h_product) == dbcsr_type_hermitian .AND. &
                 dbcsr_get_matrix_type(k_product) == dbcsr_type_antihermitian, &
                 "multiplication preserves requested H/K output types", success, io_unit)
      CALL dbcsr_dot(h_product, h, dot_value, conjugate_a=.TRUE.)
      CALL check_complex(dot_value, CMPLX(23.0_real_8, 0.0_real_8, KIND=real_8), &
                         "Hermitian multiplication and conjugate transpose", success, io_unit)
      CALL dbcsr_dot(k_product, k, dot_value, conjugate_a=.TRUE.)
      CALL check_complex(dot_value, CMPLX(23.0_real_8, 0.0_real_8, KIND=real_8), &
                         "anti-Hermitian multiplication and conjugate transpose", success, io_unit)

      CALL dbcsr_binary_write(h, "dbcsr_test_hermitian_io.bin")
      CALL dbcsr_binary_read("dbcsr_test_hermitian_io.bin", dist, h_read)
      CALL check(dbcsr_get_matrix_type(h_read) == dbcsr_type_hermitian, &
                 "Hermitian binary read preserves symmetry", success, io_unit)
      CALL dbcsr_dot(h_read, h, dot_value, conjugate_a=.TRUE.)
      CALL check_complex(dot_value, CMPLX(23.0_real_8, 0.0_real_8, KIND=real_8), &
                         "Hermitian binary round trip", success, io_unit)
      CALL dbcsr_binary_write(k, "dbcsr_test_antihermitian_io.bin")
      CALL dbcsr_binary_read("dbcsr_test_antihermitian_io.bin", dist, k_read)
      CALL check(dbcsr_get_matrix_type(k_read) == dbcsr_type_antihermitian, &
                 "anti-Hermitian binary read preserves symmetry", success, io_unit)
      CALL dbcsr_dot(k_read, k, dot_value, conjugate_a=.TRUE.)
      CALL check_complex(dot_value, CMPLX(23.0_real_8, 0.0_real_8, KIND=real_8), &
                         "anti-Hermitian binary round trip", success, io_unit)

      CALL dbcsr_release(h_read)
      CALL dbcsr_release(k_read)
      CALL dbcsr_release(h_product)
      CALL dbcsr_release(k_product)
      CALL dbcsr_release(h_transposed)
      CALL dbcsr_release(k_transposed)
      CALL dbcsr_release(h_copy)
      CALL dbcsr_release(k_copy)
      CALL dbcsr_release(identity)
      CALL dbcsr_release(h)
      CALL dbcsr_release(k)
      CALL dbcsr_distribution_release(dist)
   END FUNCTION dbcsr_test_hermitian_symmetry

   SUBROUTINE make_complex_matrix(matrix, dist, matrix_type, diag1, offdiag, diag2)
      TYPE(dbcsr_type), INTENT(INOUT)          :: matrix
      TYPE(dbcsr_distribution_type), INTENT(IN):: dist
      CHARACTER, INTENT(IN)                    :: matrix_type
      COMPLEX(real_8), INTENT(IN)               :: diag1, offdiag, diag2

      INTEGER, DIMENSION(:), POINTER            :: col_blk_size, row_blk_size
      COMPLEX(real_8), DIMENSION(1, 1)          :: block

      ALLOCATE (row_blk_size(2), col_blk_size(2))
      row_blk_size = 1
      col_blk_size = 1
      CALL dbcsr_create(matrix, "symmetry test", dist, matrix_type, row_blk_size, col_blk_size, &
                        data_type=dbcsr_type_complex_8)
      DEALLOCATE (row_blk_size, col_blk_size)
      CALL dbcsr_work_create(matrix, nblks_guess=3, sizedata_guess=3)
      block(1, 1) = diag1
      CALL dbcsr_put_block(matrix, 1, 1, block)
      block(1, 1) = offdiag
      IF (ABS(offdiag) > 0.0_real_8) CALL dbcsr_put_block(matrix, 1, 2, block)
      block(1, 1) = diag2
      CALL dbcsr_put_block(matrix, 2, 2, block)
      CALL dbcsr_finalize(matrix)
   END SUBROUTINE make_complex_matrix

   PURE LOGICAL FUNCTION dbcsr_api_type_exports() RESULT(ok)
      USE dbcsr_api, ONLY: api_conjugate_transpose => dbcsr_conjugate_transpose, &
                           api_antihermitian => dbcsr_type_antihermitian, &
                           api_hermitian => dbcsr_type_hermitian
      ok = api_hermitian == 'H' .AND. api_antihermitian == 'K' .AND. api_conjugate_transpose == 'C'
   END FUNCTION dbcsr_api_type_exports

   SUBROUTINE check_type_combinations(success, io_unit)
      LOGICAL, INTENT(INOUT)                       :: success
      INTEGER, INTENT(IN)                          :: io_unit

      CALL check(dbcsr_symmetry_data_type_compatible(dbcsr_type_no_symmetry, dbcsr_type_real_4), &
                 "real single N combination", success, io_unit)
      CALL check(dbcsr_symmetry_data_type_compatible(dbcsr_type_symmetric, dbcsr_type_real_8), &
                 "real double S combination", success, io_unit)
      CALL check(dbcsr_symmetry_data_type_compatible(dbcsr_type_antisymmetric, dbcsr_type_real_8), &
                 "real double A combination", success, io_unit)
      CALL check(dbcsr_symmetry_data_type_compatible(dbcsr_type_no_symmetry, dbcsr_type_complex_4), &
                 "complex single N combination", success, io_unit)
      CALL check(dbcsr_symmetry_data_type_compatible(dbcsr_type_hermitian, dbcsr_type_complex_8), &
                 "complex double H combination", success, io_unit)
      CALL check(dbcsr_symmetry_data_type_compatible(dbcsr_type_antihermitian, dbcsr_type_complex_8), &
                 "complex double K combination", success, io_unit)
      CALL check(.NOT. dbcsr_symmetry_data_type_compatible(dbcsr_type_hermitian, dbcsr_type_real_8), &
                 "real H combination rejected", success, io_unit)
      CALL check(.NOT. dbcsr_symmetry_data_type_compatible(dbcsr_type_antihermitian, dbcsr_type_real_8), &
                 "real K combination rejected", success, io_unit)
      CALL check(.NOT. dbcsr_symmetry_data_type_compatible(dbcsr_type_symmetric, dbcsr_type_complex_8), &
                 "complex S combination rejected", success, io_unit)
      CALL check(.NOT. dbcsr_symmetry_data_type_compatible(dbcsr_type_antisymmetric, dbcsr_type_complex_8), &
                 "complex A combination rejected", success, io_unit)
   END SUBROUTINE check_type_combinations

   SUBROUTINE check(condition, label, success, io_unit)
      LOGICAL, INTENT(IN)                          :: condition
      CHARACTER(LEN=*), INTENT(IN)                 :: label
      LOGICAL, INTENT(INOUT)                       :: success
      INTEGER, INTENT(IN)                          :: io_unit

      IF (condition) RETURN
      success = .FALSE.
      IF (io_unit > 0) WRITE (io_unit, *) "FAILED: ", TRIM(label)
   END SUBROUTINE check

   SUBROUTINE check_complex(actual, expected, label, success, io_unit)
      COMPLEX(real_8), INTENT(IN)                  :: actual, expected
      CHARACTER(LEN=*), INTENT(IN)                 :: label
      LOGICAL, INTENT(INOUT)                       :: success
      INTEGER, INTENT(IN)                          :: io_unit

      CALL check(ABS(actual - expected) < 1.0E-12_real_8, label, success, io_unit)
   END SUBROUTINE check_complex

   SUBROUTINE check_real(actual, expected, label, success, io_unit)
      REAL(real_8), INTENT(IN)                     :: actual, expected
      CHARACTER(LEN=*), INTENT(IN)                 :: label
      LOGICAL, INTENT(INOUT)                       :: success
      INTEGER, INTENT(IN)                          :: io_unit

      CALL check(ABS(actual - expected) < 1.0E-12_real_8, label, success, io_unit)
   END SUBROUTINE check_real

END MODULE dbcsr_test_symmetry
