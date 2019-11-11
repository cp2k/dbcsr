!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_api_c

   USE, INTRINSIC :: ISO_C_BINDING, ONLY: c_loc, c_ptr, c_double, C_NULL_CHAR, &
      c_f_pointer, c_int, c_char, c_null_ptr, c_bool
   USE dbcsr_api
   USE dbcsr_machine, ONLY: default_output_unit

   IMPLICIT NONE
   PRIVATE

CONTAINS

   SUBROUTINE c_f_string(c_str, str)
      USE, INTRINSIC :: iso_c_binding, ONLY: c_ptr, c_f_pointer, c_char
      TYPE(c_ptr), INTENT(in) :: c_str
      CHARACTER(kind=c_char), POINTER :: arr(:)
      CHARACTER(:, kind=c_char), ALLOCATABLE, INTENT(out) :: str
      INTEGER(8) :: n, i
      INTERFACE
         ! steal std c library function rather than writing our own.
         FUNCTION strlen(s) bind(c, name='strlen')
            USE, INTRINSIC :: iso_c_binding, ONLY: c_ptr, c_size_t
            IMPLICIT NONE
            !----
            TYPE(c_ptr), INTENT(in), value :: s
            INTEGER(c_size_t) :: strlen
         END FUNCTION strlen
      END INTERFACE
      n = strlen(c_str)
      !****
      CALL c_f_pointer(c_str, arr, [n])
      ALLOCATE (CHARACTER(len=n) :: str)
      DO i = 1, n
         str(i:i) = arr(i)
      ENDDO
   END SUBROUTINE c_f_string

   SUBROUTINE c_dbcsr_init_lib(fcomm, io_unit) bind(C, name="c_dbcsr_init_lib_internal")
      INTEGER(kind=c_int), INTENT(in)                    :: fcomm
      INTEGER(kind=c_int), INTENT(in), optional          :: io_unit

      CALL dbcsr_init_lib(fcomm, io_unit)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_finalise_lib() bind(C, name="c_dbcsr_finalize_lib")
      CALL dbcsr_finalize_lib()
   END SUBROUTINE

   SUBROUTINE c_dbcsr_distribution_new(c_dist, fcomm, c_row_dist, row_dist_size, &
                                       c_col_dist, col_dist_size) &
      bind(C, name="c_dbcsr_distribution_new_aux")
      TYPE(c_ptr), INTENT(out)                           :: c_dist
      INTEGER(kind=c_int), INTENT(in)                    :: fcomm
      INTEGER(kind=c_int), INTENT(in), value             :: row_dist_size
      INTEGER(kind=c_int), INTENT(in), TARGET            :: c_row_dist(row_dist_size)
      INTEGER(kind=c_int), INTENT(in), value             :: col_dist_size
      INTEGER(kind=c_int), INTENT(in), TARGET            :: c_col_dist(col_dist_size)

      INTEGER, POINTER                                   :: col_dist(:), row_dist(:)
      TYPE(dbcsr_distribution_type), POINTER             :: dist

      row_dist => c_row_dist
      col_dist => c_col_dist
      ALLOCATE (dist)
      CALL dbcsr_distribution_new(dist, group=fcomm, row_dist=row_dist, &
                                  col_dist=col_dist, reuse_arrays=.FALSE.)
      c_dist = c_loc(dist)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_distribution_release(c_dist) bind(C, name="c_dbcsr_distribution_release")
      TYPE(c_ptr), INTENT(inout)                         :: c_dist

      TYPE(dbcsr_distribution_type), POINTER             :: dist

      CALL c_f_pointer(c_dist, dist)

      CALL dbcsr_distribution_release(dist)

      DEALLOCATE (dist)

      c_dist = c_null_ptr
   END SUBROUTINE

   SUBROUTINE c_dbcsr_create_new_d(c_matrix, c_name, c_dist, c_matrix_type, &
                                   c_row_blk_sizes, c_row_blk_sizes_length, &
                                   c_col_blk_sizes, c_col_blk_sizes_length) &
      bind(C, name="c_dbcsr_create_new_d")
      TYPE(c_ptr), INTENT(out)                           :: c_matrix
      TYPE(c_ptr), INTENT(in), value                     :: c_name, c_dist
      CHARACTER(kind=c_char), INTENT(in), value          :: c_matrix_type
      INTEGER(kind=c_int), INTENT(in), value             :: c_row_blk_sizes_length
      INTEGER(kind=c_int), INTENT(in), TARGET :: c_row_blk_sizes(c_row_blk_sizes_length)
      INTEGER(kind=c_int), INTENT(in), value             :: c_col_blk_sizes_length
      INTEGER(kind=c_int), INTENT(in), TARGET :: c_col_blk_sizes(c_col_blk_sizes_length)

      CHARACTER(:, kind=c_char), ALLOCATABLE             :: fname
      INTEGER, POINTER                                   :: col_blk_sizes(:), row_blk_sizes(:)
      TYPE(dbcsr_distribution_type), POINTER             :: dist
      TYPE(dbcsr_type), POINTER                          :: matrix

      ! associate pointers
      row_blk_sizes => c_row_blk_sizes
      col_blk_sizes => c_col_blk_sizes

      ! convert normal distribution pointer to fortran pointer
      CALL c_f_pointer(c_dist, dist)

      ! allocate heap memory for matrix structure
      ALLOCATE (matrix)

      ! convert normal name to fortran name
      CALL c_f_string(c_name, fname)

      CALL dbcsr_create(matrix=matrix, name=fname, dist=dist, &
                        matrix_type=c_matrix_type, row_blk_size=row_blk_sizes, &
                        col_blk_size=col_blk_sizes, data_type=dbcsr_type_real_8)

      c_matrix = c_loc(matrix)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_finalize(c_matrix) bind(C, name="c_dbcsr_finalize")
      TYPE(c_ptr), INTENT(in), value                     :: c_matrix

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_finalize(matrix)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_release(c_matrix) bind(C, name="c_dbcsr_release")
      TYPE(c_ptr), INTENT(inout)                         :: c_matrix

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_release(matrix)

      DEALLOCATE (matrix)

      c_matrix = c_null_ptr
   END SUBROUTINE

   SUBROUTINE c_dbcsr_print(c_matrix) bind(C, name="c_dbcsr_print")
      TYPE(c_ptr), INTENT(in), value                     :: c_matrix

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_print(matrix)

      ! Fortran and C may use different buffers for I/O, make sure we flush before returning:
      flush(default_output_unit)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_stored_coordinates(c_matrix, row, col, processor) bind(C, name="c_dbcsr_get_stored_coordinates")
      TYPE(c_ptr), INTENT(in), value                     :: c_matrix
      INTEGER(kind=c_int), INTENT(in), value             :: row, col
      INTEGER(kind=c_int), INTENT(out)                   :: processor

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_get_stored_coordinates(matrix, row+1, col+1, processor)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_put_block_d(c_matrix, row, col, block, block_length) bind(C, name="c_dbcsr_put_block_d")
      TYPE(c_ptr), INTENT(in), value                     :: c_matrix
      INTEGER(kind=c_int), INTENT(in), value             :: row, col, block_length
      REAL(kind=c_double), INTENT(in)                    :: block(block_length)

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_put_block(matrix, row+1, col+1, block)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_multiply_d(transa, transb, alpha, c_matrix_a, c_matrix_b, beta, c_matrix_c, retain_sparsity) &
      bind(C, name="c_dbcsr_multiply_d")
      CHARACTER(kind=c_char), INTENT(in), value          :: transa, transb
      REAL(kind=c_double), INTENT(in), value             :: alpha
      TYPE(c_ptr), INTENT(in)                            :: c_matrix_a, c_matrix_b
      REAL(kind=c_double), INTENT(in), value             :: beta
      TYPE(c_ptr), INTENT(inout)                         :: c_matrix_c
      LOGICAL(c_bool), INTENT(in), OPTIONAL              :: retain_sparsity

      LOGICAL                                            :: ret_sp
      TYPE(dbcsr_type), POINTER                          :: matrix_a, matrix_b, matrix_c

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      CALL c_f_pointer(c_matrix_c, matrix_c)

      IF (PRESENT(retain_sparsity)) THEN
         ret_sp = retain_sparsity
         CALL dbcsr_multiply(transa, transb, alpha, matrix_a, matrix_b, beta, matrix_c, retain_sparsity=ret_sp)
      ELSE
         CALL dbcsr_multiply(transa, transb, alpha, matrix_a, matrix_b, beta, matrix_c)
      ENDIF
   END SUBROUTINE

END MODULE
