# 1 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F" 1
!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_api_c

   USE, INTRINSIC :: ISO_C_BINDING, ONLY: c_loc, c_ptr, c_double, c_sizeof, C_NULL_CHAR, &
                                                                             c_float, c_f_pointer, c_int, c_long_long, &
                                                                             c_char, c_null_ptr, c_bool, c_associated, &
                                                                             c_float_complex, c_double_complex
   USE dbcsr_api
   USE dbcsr_machine, ONLY: default_output_unit
   USE dbcsr_kinds, ONLY: default_string_length, &
                          dp, &
                          int_8, &
                          real_4, &
                          real_8

   IMPLICIT NONE
   PRIVATE

# 1 "/__w/dbcsr/dbcsr/src/data/dbcsr.fypp" 1
# 9 "/__w/dbcsr/dbcsr/src/data/dbcsr.fypp"

# 11 "/__w/dbcsr/dbcsr/src/data/dbcsr.fypp"

# 169 "/__w/dbcsr/dbcsr/src/data/dbcsr.fypp"
# 28 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F" 2

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

   !----------------------------------------------------!
   !                    lib init/finalize               !
   !----------------------------------------------------!

   SUBROUTINE c_dbcsr_clear_mempools() BIND(C, name="c_dbcsr_clear_mempools")
      CALL dbcsr_clear_mempools()
   END SUBROUTINE

   SUBROUTINE c_dbcsr_init_lib(fcomm, io_unit) bind(C, name="c_dbcsr_init_lib_internal")
      INTEGER(kind=c_int), INTENT(in)                    :: fcomm
      INTEGER(kind=c_int), INTENT(in), optional          :: io_unit

      CALL dbcsr_init_lib(fcomm, io_unit)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_finalise_lib() bind(C, name="c_dbcsr_finalize_lib")
      CALL dbcsr_finalize_lib()
   END SUBROUTINE

   SUBROUTINE c_dbcsr_mp_grid_setup(c_dist) BIND(C, name="c_dbcsr_mp_grid_setup")
      TYPE(c_ptr), INTENT(IN), VALUE :: c_dist
      TYPE(dbcsr_distribution_type), POINTER :: dist

      CALL c_f_pointer(c_dist, dist)

      CALL dbcsr_mp_grid_setup(dist)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_print_statistics(c_print_timers, c_callgraph_filename) &
      BIND(C, name="c_dbcsr_print_statistics")

      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_print_timers
      TYPE(c_ptr), INTENT(IN), VALUE :: c_callgraph_filename

      LOGICAL :: print_timers
      CHARACTER(:, kind=c_char), ALLOCATABLE :: callgraph_filename

      IF (C_ASSOCIATED(c_callgraph_filename)) CALL c_f_string(c_callgraph_filename, callgraph_filename)

      IF (PRESENT(c_print_timers)) THEN
         print_timers = c_print_timers
         CALL dbcsr_print_statistics(print_timers=print_timers, &
                                     callgraph_filename=callgraph_filename)
      ELSE
         CALL dbcsr_print_statistics(callgraph_filename=callgraph_filename)
      ENDIF

   END SUBROUTINE

   ! create / release
   !PUBLIC :: dbcsr_distribution_hold
   !PUBLIC :: dbcsr_distribution_release
   !PUBLIC :: dbcsr_distribution_new
   !--- PUBLIC :: dbcsr_create
   ! SKIP PUBLIC :: dbcsr_init_p
   !PUBLIC :: dbcsr_release
   ! SKIP PUBLIC :: dbcsr_release_p
   ! SKIP PUBLIC :: dbcsr_deallocate_matrix

   !-------------------------------------------------------!
   !                    create/release                     !
   !-------------------------------------------------------!

   SUBROUTINE c_dbcsr_distribution_hold(c_dist) &
      BIND(C, name="c_dbcsr_distribution_hold")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_dist
      TYPE(dbcsr_distribution_type), POINTER :: dist

      CALL c_f_pointer(c_dist, dist)

      CALL dbcsr_distribution_hold(dist)
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

      ALLOCATE (dist)
      row_dist => c_row_dist
      col_dist => c_col_dist
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

   SUBROUTINE c_dbcsr_release(c_matrix) bind(C, name="c_dbcsr_release")
      TYPE(c_ptr), INTENT(inout)                         :: c_matrix

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_release(matrix)

      DEALLOCATE (matrix)

      c_matrix = c_null_ptr
   END SUBROUTINE

   SUBROUTINE c_dbcsr_create_new(c_matrix, c_name, c_dist, c_matrix_type, &
                                 c_row_blk_size, c_row_size, &
                                 c_col_blk_size, c_col_size, &
                                 c_nze, c_data_type, c_reuse, &
                                 c_reuse_arrays, c_mutable_work, c_replication_type) &
      BIND(C, name="c_dbcsr_create_new")

      TYPE(c_ptr), INTENT(INOUT) :: c_matrix
      TYPE(c_ptr), INTENT(IN), VALUE :: c_name
      TYPE(c_ptr), INTENT(IN), VALUE :: c_dist
      CHARACTER(kind=c_char), INTENT(IN), VALUE :: c_matrix_type
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row_size, &
                                                c_col_size
      INTEGER(kind=c_int), INTENT(IN), TARGET :: c_row_blk_size(c_row_size), &
                                                 c_col_blk_size(c_col_size)
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_nze, c_data_type
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_reuse, c_reuse_arrays, &
                                                    c_mutable_work
      CHARACTER(kind=c_char), INTENT(IN), OPTIONAL :: c_replication_type

      TYPE(dbcsr_type), POINTER :: matrix
      CHARACTER(:, kind=c_char), ALLOCATABLE  :: name
      TYPE(dbcsr_distribution_type), POINTER :: dist
      INTEGER, DIMENSION(:), POINTER :: row_blk_size, col_blk_size
      LOGICAL, POINTER :: reuse, reuse_arrays, mutable_work

      ALLOCATE (matrix)
      CALL c_f_pointer(c_dist, dist)
      CALL c_f_string(c_name, name)

      row_blk_size => c_row_blk_size
      col_blk_size => c_col_blk_size

# 213 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 214 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (reuse)
      IF (PRESENT(c_reuse)) THEN
         ALLOCATE (reuse)
         reuse = c_reuse
      ENDIF
# 214 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (reuse_arrays)
      IF (PRESENT(c_reuse_arrays)) THEN
         ALLOCATE (reuse_arrays)
         reuse_arrays = c_reuse_arrays
      ENDIF
# 214 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (mutable_work)
      IF (PRESENT(c_mutable_work)) THEN
         ALLOCATE (mutable_work)
         mutable_work = c_mutable_work
      ENDIF
# 220 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

      CALL dbcsr_create(matrix, name, dist, c_matrix_type, &
                        row_blk_size, col_blk_size, &
                        c_nze, c_data_type, reuse, &
                        reuse_arrays, mutable_work, c_replication_type)

# 227 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(reuse)) DEALLOCATE (reuse)
# 227 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(reuse_arrays)) DEALLOCATE (reuse_arrays)
# 227 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(mutable_work)) DEALLOCATE (mutable_work)
# 229 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

      c_matrix = c_loc(matrix)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_create_template(c_matrix, c_name, c_template, &
                                      c_dist, c_matrix_type, &
                                      c_row_blk_size, c_row_size, &
                                      c_col_blk_size, c_col_size, &
                                      c_nze, c_data_type, &
                                      c_reuse_arrays, c_mutable_work, c_replication_type) &
      BIND(C, name="c_dbcsr_create_template")

      TYPE(c_ptr), INTENT(INOUT) :: c_matrix
      TYPE(c_ptr), INTENT(IN), VALUE :: c_name
      TYPE(c_ptr), INTENT(IN), VALUE :: c_template
      TYPE(c_ptr), INTENT(IN), VALUE :: c_dist
      CHARACTER(kind=c_char), INTENT(IN), OPTIONAL :: c_matrix_type
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row_size, &
                                                c_col_size
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL, TARGET :: c_row_blk_size(c_row_size), &
                                                           C_col_blk_size(c_col_size)
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_nze, c_data_type
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_reuse_arrays, &
                                                    c_mutable_work
      CHARACTER(kind=c_char), INTENT(IN), OPTIONAL :: c_replication_type

      TYPE(dbcsr_type), POINTER :: matrix, template
      CHARACTER(:, kind=c_char), ALLOCATABLE  :: name
      INTEGER, DIMENSION(:), POINTER :: row_blk_size, col_blk_size
      TYPE(dbcsr_distribution_type), POINTER :: dist
      LOGICAL, POINTER :: reuse_arrays, mutable_work

      ALLOCATE (matrix)
      CALL c_f_pointer(c_template, template)
      CALL c_f_string(c_name, name)

      NULLIFY (row_blk_size)
      NULLIFY (col_blk_size)
      IF (PRESENT(c_row_blk_size)) row_blk_size => c_row_blk_size
      IF (PRESENT(c_col_blk_size)) col_blk_size => c_col_blk_size

      NULLIFY (dist)
      IF (C_ASSOCIATED(c_dist)) CALL c_f_pointer(c_dist, dist)

# 275 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 276 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (reuse_arrays)
      IF (PRESENT(c_reuse_arrays)) THEN
         ALLOCATE (reuse_arrays)
         reuse_arrays = c_reuse_arrays
      ENDIF
# 276 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (mutable_work)
      IF (PRESENT(c_mutable_work)) THEN
         ALLOCATE (mutable_work)
         mutable_work = c_mutable_work
      ENDIF
# 282 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

# 284 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 285 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      

# 288 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF(PRESENT(c_row_blk_size) .AND. PRESENT(c_col_blk_size)) THEN 

      CALL dbcsr_create(matrix, name, template, dist, c_matrix_type &
                        , row_blk_size = row_blk_size, col_blk_size = col_blk_size, &
                        nze=c_nze, data_type=c_data_type, &
                        reuse_arrays=reuse_arrays, mutable_work=mutable_work, &
                        replication_type=c_replication_type)

# 288 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(PRESENT(c_col_blk_size) .AND. .NOT. PRESENT(c_row_blk_size)) THEN 

      CALL dbcsr_create(matrix, name, template, dist, c_matrix_type &
                        , col_blk_size = col_blk_size, &
                        nze=c_nze, data_type=c_data_type, &
                        reuse_arrays=reuse_arrays, mutable_work=mutable_work, &
                        replication_type=c_replication_type)

# 288 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(PRESENT(c_row_blk_size) .AND. .NOT. PRESENT(c_col_blk_size)) THEN 

      CALL dbcsr_create(matrix, name, template, dist, c_matrix_type &
                        , row_blk_size = row_blk_size, &
                        nze=c_nze, data_type=c_data_type, &
                        reuse_arrays=reuse_arrays, mutable_work=mutable_work, &
                        replication_type=c_replication_type)

# 288 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE

      CALL dbcsr_create(matrix, name, template, dist, c_matrix_type &
                        , &
                        nze=c_nze, data_type=c_data_type, &
                        reuse_arrays=reuse_arrays, mutable_work=mutable_work, &
                        replication_type=c_replication_type)

# 297 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
ENDIF

# 300 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(reuse_arrays)) DEALLOCATE (reuse_arrays)
# 300 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(mutable_work)) DEALLOCATE (mutable_work)
# 302 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

      c_matrix = c_loc(matrix)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_finalize(c_matrix) bind(C, name="c_dbcsr_finalize")
      TYPE(c_ptr), INTENT(in), value                     :: c_matrix

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_finalize(matrix)
   END SUBROUTINE

   !----------------------------------------------------------!
   !              primitive matrix operations                 !
   !----------------------------------------------------------!

# 322 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   SUBROUTINE c_dbcsr_set_d (c_matrix, c_alpha) &
      BIND(C, name="c_dbcsr_set_d")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      REAL(kind=c_double), INTENT(IN), VALUE :: c_alpha
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_set(matrix, c_alpha)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_add_d (c_matrix_a, c_matrix_b, c_alpha_scalar, c_beta_scalar) &
      BIND(C, name="c_dbcsr_add_d")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a, c_matrix_b
      REAL(kind=c_double), INTENT(IN), VALUE :: c_alpha_scalar, c_beta_scalar
      TYPE(dbcsr_type), POINTER :: matrix_a, matrix_b

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      CALL dbcsr_add(matrix_a, matrix_b, c_alpha_scalar, c_beta_scalar)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_scale_d (c_matrix_a, c_alpha_scalar, c_last_column) &
      BIND(C, name="c_dbcsr_scale_d")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a
      REAL(kind=c_double), INTENT(IN), VALUE :: c_alpha_scalar
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_last_column
      TYPE(dbcsr_type), POINTER :: matrix_a

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL dbcsr_scale(matrix_a, c_alpha_scalar, c_last_column)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_scale_by_vector_d (c_matrix_a, c_alpha, c_alpha_size, c_side) &
      BIND(C, name="c_dbcsr_scale_by_vector_d")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_alpha_size
      REAL(kind=c_double), INTENT(IN) :: c_alpha(c_alpha_size)
      TYPE(c_ptr), INTENT(IN), VALUE :: c_side
      TYPE(dbcsr_type), POINTER :: matrix_a
      CHARACTER(:, kind=c_char), ALLOCATABLE  :: side

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_string(c_side, side)

      CALL dbcsr_scale_by_vector(matrix_a, c_alpha, side)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_multiply_d (c_transa, c_transb, &
                                             c_alpha, c_matrix_a, c_matrix_b, c_beta, c_matrix_c, &
                                             c_first_row, c_last_row, c_first_column, c_last_column, &
                                             c_first_k, c_last_k, &
                                             c_retain_sparsity, c_filter_eps, c_flop) &
      BIND(C, name="c_dbcsr_multiply_d")

      CHARACTER(kind=c_char), INTENT(in), value :: c_transa, c_transb
      REAL(kind=c_double), INTENT(in), value :: c_alpha, c_beta
      TYPE(c_ptr), INTENT(in), VALUE :: c_matrix_a, c_matrix_b, c_matrix_c
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_first_row, c_last_row, &
                                                   c_first_column, c_last_column, &
                                                   c_first_k, c_last_k
      LOGICAL(c_bool), INTENT(in), OPTIONAL :: c_retain_sparsity
      REAL(kind=c_double), OPTIONAL :: c_filter_eps
      INTEGER(kind=c_long_long), OPTIONAL :: c_flop

      LOGICAL                                            :: ret_sp
      TYPE(dbcsr_type), POINTER                          :: matrix_a, matrix_b, matrix_c
      INTEGER, POINTER :: first_row, last_row, first_column, last_column, first_k, last_k

# 400 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (first_row)
      IF (PRESENT(c_first_row)) THEN
         ALLOCATE (first_row)
         first_row = c_first_row+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (last_row)
      IF (PRESENT(c_last_row)) THEN
         ALLOCATE (last_row)
         last_row = c_last_row+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (first_column)
      IF (PRESENT(c_first_column)) THEN
         ALLOCATE (first_column)
         first_column = c_first_column+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (last_column)
      IF (PRESENT(c_last_column)) THEN
         ALLOCATE (last_column)
         last_column = c_last_column+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (first_k)
      IF (PRESENT(c_first_k)) THEN
         ALLOCATE (first_k)
         first_k = c_first_k+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (last_k)
      IF (PRESENT(c_last_k)) THEN
         ALLOCATE (last_k)
         last_k = c_last_k+1
      ENDIF
# 407 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      CALL c_f_pointer(c_matrix_c, matrix_c)

      IF (PRESENT(c_retain_sparsity)) THEN
         ret_sp = c_retain_sparsity
         CALL dbcsr_multiply(c_transa, c_transb, &
                             c_alpha, matrix_a, matrix_b, c_beta, matrix_c, &
                             first_row, last_row, first_column, last_column, &
                             first_k, last_k, &
                             ret_sp, c_filter_eps, c_flop)
      ELSE
         CALL dbcsr_multiply(c_transa, c_transb, &
                             c_alpha, matrix_a, matrix_b, c_beta, matrix_c, &
                             first_row, last_row, first_column, last_column, &
                             first_k, last_k, &
                             filter_eps=c_filter_eps, flop=c_flop)
      ENDIF

# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_first_row)) DEALLOCATE (first_row)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_last_row)) DEALLOCATE (last_row)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_first_column)) DEALLOCATE (first_column)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_last_column)) DEALLOCATE (last_column)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_first_k)) DEALLOCATE (first_k)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_last_k)) DEALLOCATE (last_k)
# 430 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   END SUBROUTINE

   SUBROUTINE c_dbcsr_add_on_diag_d (c_matrix, c_alpha_scalar) &
      BIND(C, name="c_dbcsr_add_on_diag_d")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      REAL(kind=c_double), INTENT(IN) :: c_alpha_scalar
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_add_on_diag(matrix, c_alpha_scalar)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_set_diag_d (c_matrix, c_diag, c_diag_size) &
      BIND(C, name="c_dbcsr_set_diag_d")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_diag_size
      REAL(kind=c_double), INTENT(IN) :: c_diag(c_diag_size)
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_set_diag(matrix, c_diag)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_diag_d (c_matrix, c_diag, c_diag_size) &
      BIND(C, name="c_dbcsr_get_diag_d")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_diag_size
      REAL(kind=c_double), INTENT(INOUT) :: c_diag(c_diag_size)
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_get_diag(matrix, c_diag)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_trace_d (c_matrix_a, c_trace) &
      BIND(C, name="c_dbcsr_trace_d")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a
      REAL(kind=c_double), INTENT(OUT) :: c_trace
      TYPE(dbcsr_type), POINTER :: matrix_a

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL dbcsr_trace(matrix_a, c_trace)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_dot_d (c_matrix_a, c_matrix_b, c_result) &
      BIND(C, name="c_dbcsr_dot_d")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a, c_matrix_b
      REAL(kind=c_double), INTENT(INOUT) :: c_result
      TYPE(dbcsr_type), POINTER :: matrix_a, matrix_b

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      CALL dbcsr_dot(matrix_a, matrix_b, c_result)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_block_p_d (c_matrix, c_row, c_col, c_block, &
                                                c_tr, c_found, c_row_size, c_col_size) &
      BIND(C, name="c_dbcsr_get_block_p_d")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row, c_col
      TYPE(c_ptr), INTENT(INOUT) :: c_block
      LOGICAL(kind=c_bool), INTENT(OUT) :: c_tr, c_found
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_row_size, c_col_size

      TYPE(dbcsr_type), POINTER :: matrix
      REAL(kind=c_double), DIMENSION(:), POINTER :: block
      LOGICAL :: tr, found

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_get_block_p(matrix, c_row + 1, c_col + 1, block, tr, &
                             found, c_row_size, c_col_size)

      c_tr = tr
      c_found = found
      c_block = c_loc(block)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_block_notrans_p_d (c_matrix, c_row, c_col, &
                                                        c_block, c_found, c_row_size, c_col_size) &
      BIND(C, name="c_dbcsr_get_block_notrans_p_d")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row, c_col
      TYPE(c_ptr), INTENT(INOUT) :: c_block
      LOGICAL(kind=c_bool), INTENT(OUT) :: c_found
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_row_size, c_col_size

      TYPE(dbcsr_type), POINTER :: matrix
      REAL(kind=c_double), DIMENSION(:), POINTER :: block
      LOGICAL :: found

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_get_block_p(matrix, c_row + 1, c_col + 1, block, found, &
                             c_row_size, c_col_size)

      c_block = c_loc(block)
      c_found = found

   END SUBROUTINE

# 322 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   SUBROUTINE c_dbcsr_set_s (c_matrix, c_alpha) &
      BIND(C, name="c_dbcsr_set_s")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      REAL(kind=c_float), INTENT(IN), VALUE :: c_alpha
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_set(matrix, c_alpha)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_add_s (c_matrix_a, c_matrix_b, c_alpha_scalar, c_beta_scalar) &
      BIND(C, name="c_dbcsr_add_s")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a, c_matrix_b
      REAL(kind=c_float), INTENT(IN), VALUE :: c_alpha_scalar, c_beta_scalar
      TYPE(dbcsr_type), POINTER :: matrix_a, matrix_b

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      CALL dbcsr_add(matrix_a, matrix_b, c_alpha_scalar, c_beta_scalar)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_scale_s (c_matrix_a, c_alpha_scalar, c_last_column) &
      BIND(C, name="c_dbcsr_scale_s")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a
      REAL(kind=c_float), INTENT(IN), VALUE :: c_alpha_scalar
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_last_column
      TYPE(dbcsr_type), POINTER :: matrix_a

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL dbcsr_scale(matrix_a, c_alpha_scalar, c_last_column)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_scale_by_vector_s (c_matrix_a, c_alpha, c_alpha_size, c_side) &
      BIND(C, name="c_dbcsr_scale_by_vector_s")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_alpha_size
      REAL(kind=c_float), INTENT(IN) :: c_alpha(c_alpha_size)
      TYPE(c_ptr), INTENT(IN), VALUE :: c_side
      TYPE(dbcsr_type), POINTER :: matrix_a
      CHARACTER(:, kind=c_char), ALLOCATABLE  :: side

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_string(c_side, side)

      CALL dbcsr_scale_by_vector(matrix_a, c_alpha, side)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_multiply_s (c_transa, c_transb, &
                                             c_alpha, c_matrix_a, c_matrix_b, c_beta, c_matrix_c, &
                                             c_first_row, c_last_row, c_first_column, c_last_column, &
                                             c_first_k, c_last_k, &
                                             c_retain_sparsity, c_filter_eps, c_flop) &
      BIND(C, name="c_dbcsr_multiply_s")

      CHARACTER(kind=c_char), INTENT(in), value :: c_transa, c_transb
      REAL(kind=c_float), INTENT(in), value :: c_alpha, c_beta
      TYPE(c_ptr), INTENT(in), VALUE :: c_matrix_a, c_matrix_b, c_matrix_c
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_first_row, c_last_row, &
                                                   c_first_column, c_last_column, &
                                                   c_first_k, c_last_k
      LOGICAL(c_bool), INTENT(in), OPTIONAL :: c_retain_sparsity
      REAL(kind=c_double), OPTIONAL :: c_filter_eps
      INTEGER(kind=c_long_long), OPTIONAL :: c_flop

      LOGICAL                                            :: ret_sp
      TYPE(dbcsr_type), POINTER                          :: matrix_a, matrix_b, matrix_c
      INTEGER, POINTER :: first_row, last_row, first_column, last_column, first_k, last_k

# 400 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (first_row)
      IF (PRESENT(c_first_row)) THEN
         ALLOCATE (first_row)
         first_row = c_first_row+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (last_row)
      IF (PRESENT(c_last_row)) THEN
         ALLOCATE (last_row)
         last_row = c_last_row+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (first_column)
      IF (PRESENT(c_first_column)) THEN
         ALLOCATE (first_column)
         first_column = c_first_column+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (last_column)
      IF (PRESENT(c_last_column)) THEN
         ALLOCATE (last_column)
         last_column = c_last_column+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (first_k)
      IF (PRESENT(c_first_k)) THEN
         ALLOCATE (first_k)
         first_k = c_first_k+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (last_k)
      IF (PRESENT(c_last_k)) THEN
         ALLOCATE (last_k)
         last_k = c_last_k+1
      ENDIF
# 407 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      CALL c_f_pointer(c_matrix_c, matrix_c)

      IF (PRESENT(c_retain_sparsity)) THEN
         ret_sp = c_retain_sparsity
         CALL dbcsr_multiply(c_transa, c_transb, &
                             c_alpha, matrix_a, matrix_b, c_beta, matrix_c, &
                             first_row, last_row, first_column, last_column, &
                             first_k, last_k, &
                             ret_sp, c_filter_eps, c_flop)
      ELSE
         CALL dbcsr_multiply(c_transa, c_transb, &
                             c_alpha, matrix_a, matrix_b, c_beta, matrix_c, &
                             first_row, last_row, first_column, last_column, &
                             first_k, last_k, &
                             filter_eps=c_filter_eps, flop=c_flop)
      ENDIF

# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_first_row)) DEALLOCATE (first_row)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_last_row)) DEALLOCATE (last_row)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_first_column)) DEALLOCATE (first_column)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_last_column)) DEALLOCATE (last_column)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_first_k)) DEALLOCATE (first_k)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_last_k)) DEALLOCATE (last_k)
# 430 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   END SUBROUTINE

   SUBROUTINE c_dbcsr_add_on_diag_s (c_matrix, c_alpha_scalar) &
      BIND(C, name="c_dbcsr_add_on_diag_s")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      REAL(kind=c_float), INTENT(IN) :: c_alpha_scalar
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_add_on_diag(matrix, c_alpha_scalar)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_set_diag_s (c_matrix, c_diag, c_diag_size) &
      BIND(C, name="c_dbcsr_set_diag_s")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_diag_size
      REAL(kind=c_float), INTENT(IN) :: c_diag(c_diag_size)
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_set_diag(matrix, c_diag)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_diag_s (c_matrix, c_diag, c_diag_size) &
      BIND(C, name="c_dbcsr_get_diag_s")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_diag_size
      REAL(kind=c_float), INTENT(INOUT) :: c_diag(c_diag_size)
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_get_diag(matrix, c_diag)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_trace_s (c_matrix_a, c_trace) &
      BIND(C, name="c_dbcsr_trace_s")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a
      REAL(kind=c_float), INTENT(OUT) :: c_trace
      TYPE(dbcsr_type), POINTER :: matrix_a

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL dbcsr_trace(matrix_a, c_trace)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_dot_s (c_matrix_a, c_matrix_b, c_result) &
      BIND(C, name="c_dbcsr_dot_s")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a, c_matrix_b
      REAL(kind=c_float), INTENT(INOUT) :: c_result
      TYPE(dbcsr_type), POINTER :: matrix_a, matrix_b

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      CALL dbcsr_dot(matrix_a, matrix_b, c_result)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_block_p_s (c_matrix, c_row, c_col, c_block, &
                                                c_tr, c_found, c_row_size, c_col_size) &
      BIND(C, name="c_dbcsr_get_block_p_s")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row, c_col
      TYPE(c_ptr), INTENT(INOUT) :: c_block
      LOGICAL(kind=c_bool), INTENT(OUT) :: c_tr, c_found
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_row_size, c_col_size

      TYPE(dbcsr_type), POINTER :: matrix
      REAL(kind=c_float), DIMENSION(:), POINTER :: block
      LOGICAL :: tr, found

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_get_block_p(matrix, c_row + 1, c_col + 1, block, tr, &
                             found, c_row_size, c_col_size)

      c_tr = tr
      c_found = found
      c_block = c_loc(block)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_block_notrans_p_s (c_matrix, c_row, c_col, &
                                                        c_block, c_found, c_row_size, c_col_size) &
      BIND(C, name="c_dbcsr_get_block_notrans_p_s")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row, c_col
      TYPE(c_ptr), INTENT(INOUT) :: c_block
      LOGICAL(kind=c_bool), INTENT(OUT) :: c_found
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_row_size, c_col_size

      TYPE(dbcsr_type), POINTER :: matrix
      REAL(kind=c_float), DIMENSION(:), POINTER :: block
      LOGICAL :: found

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_get_block_p(matrix, c_row + 1, c_col + 1, block, found, &
                             c_row_size, c_col_size)

      c_block = c_loc(block)
      c_found = found

   END SUBROUTINE

# 322 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   SUBROUTINE c_dbcsr_set_z (c_matrix, c_alpha) &
      BIND(C, name="c_dbcsr_set_z")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      COMPLEX(kind=c_double_complex), INTENT(IN), VALUE :: c_alpha
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_set(matrix, c_alpha)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_add_z (c_matrix_a, c_matrix_b, c_alpha_scalar, c_beta_scalar) &
      BIND(C, name="c_dbcsr_add_z")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a, c_matrix_b
      COMPLEX(kind=c_double_complex), INTENT(IN), VALUE :: c_alpha_scalar, c_beta_scalar
      TYPE(dbcsr_type), POINTER :: matrix_a, matrix_b

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      CALL dbcsr_add(matrix_a, matrix_b, c_alpha_scalar, c_beta_scalar)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_scale_z (c_matrix_a, c_alpha_scalar, c_last_column) &
      BIND(C, name="c_dbcsr_scale_z")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a
      COMPLEX(kind=c_double_complex), INTENT(IN), VALUE :: c_alpha_scalar
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_last_column
      TYPE(dbcsr_type), POINTER :: matrix_a

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL dbcsr_scale(matrix_a, c_alpha_scalar, c_last_column)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_scale_by_vector_z (c_matrix_a, c_alpha, c_alpha_size, c_side) &
      BIND(C, name="c_dbcsr_scale_by_vector_z")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_alpha_size
      COMPLEX(kind=c_double_complex), INTENT(IN) :: c_alpha(c_alpha_size)
      TYPE(c_ptr), INTENT(IN), VALUE :: c_side
      TYPE(dbcsr_type), POINTER :: matrix_a
      CHARACTER(:, kind=c_char), ALLOCATABLE  :: side

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_string(c_side, side)

      CALL dbcsr_scale_by_vector(matrix_a, c_alpha, side)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_multiply_z (c_transa, c_transb, &
                                             c_alpha, c_matrix_a, c_matrix_b, c_beta, c_matrix_c, &
                                             c_first_row, c_last_row, c_first_column, c_last_column, &
                                             c_first_k, c_last_k, &
                                             c_retain_sparsity, c_filter_eps, c_flop) &
      BIND(C, name="c_dbcsr_multiply_z")

      CHARACTER(kind=c_char), INTENT(in), value :: c_transa, c_transb
      COMPLEX(kind=c_double_complex), INTENT(in), value :: c_alpha, c_beta
      TYPE(c_ptr), INTENT(in), VALUE :: c_matrix_a, c_matrix_b, c_matrix_c
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_first_row, c_last_row, &
                                                   c_first_column, c_last_column, &
                                                   c_first_k, c_last_k
      LOGICAL(c_bool), INTENT(in), OPTIONAL :: c_retain_sparsity
      REAL(kind=c_double), OPTIONAL :: c_filter_eps
      INTEGER(kind=c_long_long), OPTIONAL :: c_flop

      LOGICAL                                            :: ret_sp
      TYPE(dbcsr_type), POINTER                          :: matrix_a, matrix_b, matrix_c
      INTEGER, POINTER :: first_row, last_row, first_column, last_column, first_k, last_k

# 400 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (first_row)
      IF (PRESENT(c_first_row)) THEN
         ALLOCATE (first_row)
         first_row = c_first_row+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (last_row)
      IF (PRESENT(c_last_row)) THEN
         ALLOCATE (last_row)
         last_row = c_last_row+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (first_column)
      IF (PRESENT(c_first_column)) THEN
         ALLOCATE (first_column)
         first_column = c_first_column+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (last_column)
      IF (PRESENT(c_last_column)) THEN
         ALLOCATE (last_column)
         last_column = c_last_column+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (first_k)
      IF (PRESENT(c_first_k)) THEN
         ALLOCATE (first_k)
         first_k = c_first_k+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (last_k)
      IF (PRESENT(c_last_k)) THEN
         ALLOCATE (last_k)
         last_k = c_last_k+1
      ENDIF
# 407 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      CALL c_f_pointer(c_matrix_c, matrix_c)

      IF (PRESENT(c_retain_sparsity)) THEN
         ret_sp = c_retain_sparsity
         CALL dbcsr_multiply(c_transa, c_transb, &
                             c_alpha, matrix_a, matrix_b, c_beta, matrix_c, &
                             first_row, last_row, first_column, last_column, &
                             first_k, last_k, &
                             ret_sp, c_filter_eps, c_flop)
      ELSE
         CALL dbcsr_multiply(c_transa, c_transb, &
                             c_alpha, matrix_a, matrix_b, c_beta, matrix_c, &
                             first_row, last_row, first_column, last_column, &
                             first_k, last_k, &
                             filter_eps=c_filter_eps, flop=c_flop)
      ENDIF

# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_first_row)) DEALLOCATE (first_row)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_last_row)) DEALLOCATE (last_row)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_first_column)) DEALLOCATE (first_column)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_last_column)) DEALLOCATE (last_column)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_first_k)) DEALLOCATE (first_k)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_last_k)) DEALLOCATE (last_k)
# 430 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   END SUBROUTINE

   SUBROUTINE c_dbcsr_add_on_diag_z (c_matrix, c_alpha_scalar) &
      BIND(C, name="c_dbcsr_add_on_diag_z")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      COMPLEX(kind=c_double_complex), INTENT(IN) :: c_alpha_scalar
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_add_on_diag(matrix, c_alpha_scalar)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_set_diag_z (c_matrix, c_diag, c_diag_size) &
      BIND(C, name="c_dbcsr_set_diag_z")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_diag_size
      COMPLEX(kind=c_double_complex), INTENT(IN) :: c_diag(c_diag_size)
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_set_diag(matrix, c_diag)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_diag_z (c_matrix, c_diag, c_diag_size) &
      BIND(C, name="c_dbcsr_get_diag_z")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_diag_size
      COMPLEX(kind=c_double_complex), INTENT(INOUT) :: c_diag(c_diag_size)
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_get_diag(matrix, c_diag)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_trace_z (c_matrix_a, c_trace) &
      BIND(C, name="c_dbcsr_trace_z")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a
      COMPLEX(kind=c_double_complex), INTENT(OUT) :: c_trace
      TYPE(dbcsr_type), POINTER :: matrix_a

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL dbcsr_trace(matrix_a, c_trace)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_dot_z (c_matrix_a, c_matrix_b, c_result) &
      BIND(C, name="c_dbcsr_dot_z")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a, c_matrix_b
      COMPLEX(kind=c_double_complex), INTENT(INOUT) :: c_result
      TYPE(dbcsr_type), POINTER :: matrix_a, matrix_b

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      CALL dbcsr_dot(matrix_a, matrix_b, c_result)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_block_p_z (c_matrix, c_row, c_col, c_block, &
                                                c_tr, c_found, c_row_size, c_col_size) &
      BIND(C, name="c_dbcsr_get_block_p_z")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row, c_col
      TYPE(c_ptr), INTENT(INOUT) :: c_block
      LOGICAL(kind=c_bool), INTENT(OUT) :: c_tr, c_found
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_row_size, c_col_size

      TYPE(dbcsr_type), POINTER :: matrix
      COMPLEX(kind=c_double_complex), DIMENSION(:), POINTER :: block
      LOGICAL :: tr, found

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_get_block_p(matrix, c_row + 1, c_col + 1, block, tr, &
                             found, c_row_size, c_col_size)

      c_tr = tr
      c_found = found
      c_block = c_loc(block)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_block_notrans_p_z (c_matrix, c_row, c_col, &
                                                        c_block, c_found, c_row_size, c_col_size) &
      BIND(C, name="c_dbcsr_get_block_notrans_p_z")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row, c_col
      TYPE(c_ptr), INTENT(INOUT) :: c_block
      LOGICAL(kind=c_bool), INTENT(OUT) :: c_found
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_row_size, c_col_size

      TYPE(dbcsr_type), POINTER :: matrix
      COMPLEX(kind=c_double_complex), DIMENSION(:), POINTER :: block
      LOGICAL :: found

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_get_block_p(matrix, c_row + 1, c_col + 1, block, found, &
                             c_row_size, c_col_size)

      c_block = c_loc(block)
      c_found = found

   END SUBROUTINE

# 322 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   SUBROUTINE c_dbcsr_set_c (c_matrix, c_alpha) &
      BIND(C, name="c_dbcsr_set_c")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      COMPLEX(kind=c_float_complex), INTENT(IN), VALUE :: c_alpha
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_set(matrix, c_alpha)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_add_c (c_matrix_a, c_matrix_b, c_alpha_scalar, c_beta_scalar) &
      BIND(C, name="c_dbcsr_add_c")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a, c_matrix_b
      COMPLEX(kind=c_float_complex), INTENT(IN), VALUE :: c_alpha_scalar, c_beta_scalar
      TYPE(dbcsr_type), POINTER :: matrix_a, matrix_b

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      CALL dbcsr_add(matrix_a, matrix_b, c_alpha_scalar, c_beta_scalar)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_scale_c (c_matrix_a, c_alpha_scalar, c_last_column) &
      BIND(C, name="c_dbcsr_scale_c")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a
      COMPLEX(kind=c_float_complex), INTENT(IN), VALUE :: c_alpha_scalar
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_last_column
      TYPE(dbcsr_type), POINTER :: matrix_a

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL dbcsr_scale(matrix_a, c_alpha_scalar, c_last_column)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_scale_by_vector_c (c_matrix_a, c_alpha, c_alpha_size, c_side) &
      BIND(C, name="c_dbcsr_scale_by_vector_c")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_alpha_size
      COMPLEX(kind=c_float_complex), INTENT(IN) :: c_alpha(c_alpha_size)
      TYPE(c_ptr), INTENT(IN), VALUE :: c_side
      TYPE(dbcsr_type), POINTER :: matrix_a
      CHARACTER(:, kind=c_char), ALLOCATABLE  :: side

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_string(c_side, side)

      CALL dbcsr_scale_by_vector(matrix_a, c_alpha, side)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_multiply_c (c_transa, c_transb, &
                                             c_alpha, c_matrix_a, c_matrix_b, c_beta, c_matrix_c, &
                                             c_first_row, c_last_row, c_first_column, c_last_column, &
                                             c_first_k, c_last_k, &
                                             c_retain_sparsity, c_filter_eps, c_flop) &
      BIND(C, name="c_dbcsr_multiply_c")

      CHARACTER(kind=c_char), INTENT(in), value :: c_transa, c_transb
      COMPLEX(kind=c_float_complex), INTENT(in), value :: c_alpha, c_beta
      TYPE(c_ptr), INTENT(in), VALUE :: c_matrix_a, c_matrix_b, c_matrix_c
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_first_row, c_last_row, &
                                                   c_first_column, c_last_column, &
                                                   c_first_k, c_last_k
      LOGICAL(c_bool), INTENT(in), OPTIONAL :: c_retain_sparsity
      REAL(kind=c_double), OPTIONAL :: c_filter_eps
      INTEGER(kind=c_long_long), OPTIONAL :: c_flop

      LOGICAL                                            :: ret_sp
      TYPE(dbcsr_type), POINTER                          :: matrix_a, matrix_b, matrix_c
      INTEGER, POINTER :: first_row, last_row, first_column, last_column, first_k, last_k

# 400 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (first_row)
      IF (PRESENT(c_first_row)) THEN
         ALLOCATE (first_row)
         first_row = c_first_row+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (last_row)
      IF (PRESENT(c_last_row)) THEN
         ALLOCATE (last_row)
         last_row = c_last_row+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (first_column)
      IF (PRESENT(c_first_column)) THEN
         ALLOCATE (first_column)
         first_column = c_first_column+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (last_column)
      IF (PRESENT(c_last_column)) THEN
         ALLOCATE (last_column)
         last_column = c_last_column+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (first_k)
      IF (PRESENT(c_first_k)) THEN
         ALLOCATE (first_k)
         first_k = c_first_k+1
      ENDIF
# 401 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (last_k)
      IF (PRESENT(c_last_k)) THEN
         ALLOCATE (last_k)
         last_k = c_last_k+1
      ENDIF
# 407 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      CALL c_f_pointer(c_matrix_c, matrix_c)

      IF (PRESENT(c_retain_sparsity)) THEN
         ret_sp = c_retain_sparsity
         CALL dbcsr_multiply(c_transa, c_transb, &
                             c_alpha, matrix_a, matrix_b, c_beta, matrix_c, &
                             first_row, last_row, first_column, last_column, &
                             first_k, last_k, &
                             ret_sp, c_filter_eps, c_flop)
      ELSE
         CALL dbcsr_multiply(c_transa, c_transb, &
                             c_alpha, matrix_a, matrix_b, c_beta, matrix_c, &
                             first_row, last_row, first_column, last_column, &
                             first_k, last_k, &
                             filter_eps=c_filter_eps, flop=c_flop)
      ENDIF

# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_first_row)) DEALLOCATE (first_row)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_last_row)) DEALLOCATE (last_row)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_first_column)) DEALLOCATE (first_column)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_last_column)) DEALLOCATE (last_column)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_first_k)) DEALLOCATE (first_k)
# 428 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_last_k)) DEALLOCATE (last_k)
# 430 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   END SUBROUTINE

   SUBROUTINE c_dbcsr_add_on_diag_c (c_matrix, c_alpha_scalar) &
      BIND(C, name="c_dbcsr_add_on_diag_c")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      COMPLEX(kind=c_float_complex), INTENT(IN) :: c_alpha_scalar
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_add_on_diag(matrix, c_alpha_scalar)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_set_diag_c (c_matrix, c_diag, c_diag_size) &
      BIND(C, name="c_dbcsr_set_diag_c")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_diag_size
      COMPLEX(kind=c_float_complex), INTENT(IN) :: c_diag(c_diag_size)
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_set_diag(matrix, c_diag)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_diag_c (c_matrix, c_diag, c_diag_size) &
      BIND(C, name="c_dbcsr_get_diag_c")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_diag_size
      COMPLEX(kind=c_float_complex), INTENT(INOUT) :: c_diag(c_diag_size)
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_get_diag(matrix, c_diag)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_trace_c (c_matrix_a, c_trace) &
      BIND(C, name="c_dbcsr_trace_c")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a
      COMPLEX(kind=c_float_complex), INTENT(OUT) :: c_trace
      TYPE(dbcsr_type), POINTER :: matrix_a

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL dbcsr_trace(matrix_a, c_trace)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_dot_c (c_matrix_a, c_matrix_b, c_result) &
      BIND(C, name="c_dbcsr_dot_c")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a, c_matrix_b
      COMPLEX(kind=c_float_complex), INTENT(INOUT) :: c_result
      TYPE(dbcsr_type), POINTER :: matrix_a, matrix_b

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      CALL dbcsr_dot(matrix_a, matrix_b, c_result)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_block_p_c (c_matrix, c_row, c_col, c_block, &
                                                c_tr, c_found, c_row_size, c_col_size) &
      BIND(C, name="c_dbcsr_get_block_p_c")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row, c_col
      TYPE(c_ptr), INTENT(INOUT) :: c_block
      LOGICAL(kind=c_bool), INTENT(OUT) :: c_tr, c_found
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_row_size, c_col_size

      TYPE(dbcsr_type), POINTER :: matrix
      COMPLEX(kind=c_float_complex), DIMENSION(:), POINTER :: block
      LOGICAL :: tr, found

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_get_block_p(matrix, c_row + 1, c_col + 1, block, tr, &
                             found, c_row_size, c_col_size)

      c_tr = tr
      c_found = found
      c_block = c_loc(block)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_block_notrans_p_c (c_matrix, c_row, c_col, &
                                                        c_block, c_found, c_row_size, c_col_size) &
      BIND(C, name="c_dbcsr_get_block_notrans_p_c")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row, c_col
      TYPE(c_ptr), INTENT(INOUT) :: c_block
      LOGICAL(kind=c_bool), INTENT(OUT) :: c_found
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_row_size, c_col_size

      TYPE(dbcsr_type), POINTER :: matrix
      COMPLEX(kind=c_float_complex), DIMENSION(:), POINTER :: block
      LOGICAL :: found

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_get_block_p(matrix, c_row + 1, c_col + 1, block, found, &
                             c_row_size, c_col_size)

      c_block = c_loc(block)
      c_found = found

   END SUBROUTINE

# 541 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   SUBROUTINE c_dbcsr_complete_redistribute(c_matrix, c_redist, c_keep_sparsity, c_summation) &
      BIND(C, name="c_dbcsr_complete_redistribute")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(c_ptr), INTENT(IN), VALUE :: c_redist
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_keep_sparsity, c_summation
      TYPE(dbcsr_type), POINTER :: matrix, redist
      LOGICAL, POINTER :: keep_sparsity, summation

      CALL c_f_pointer(c_matrix, matrix)
      CALL c_f_pointer(c_redist, redist)

# 555 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 556 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (keep_sparsity)
      IF (PRESENT(c_keep_sparsity)) THEN
         ALLOCATE (keep_sparsity)
         keep_sparsity = c_keep_sparsity
      ENDIF
# 556 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (summation)
      IF (PRESENT(c_summation)) THEN
         ALLOCATE (summation)
         summation = c_summation
      ENDIF
# 562 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

      CALL dbcsr_complete_redistribute(matrix, redist, keep_sparsity, summation)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_filter(c_matrix, c_eps, c_method, c_use_absolute, c_filter_diag) &
      BIND(C, name="c_dbcsr_filter")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      REAL(kind=c_double), INTENT(IN) :: c_eps
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_method
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_use_absolute, c_filter_diag
      TYPE(dbcsr_type), POINTER :: matrix
      LOGICAL, POINTER :: use_absolute, filter_diag

      CALL c_f_pointer(c_matrix, matrix)
# 579 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 580 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (use_absolute)
      IF (PRESENT(c_use_absolute)) THEN
         ALLOCATE (use_absolute)
         use_absolute = c_use_absolute
      ENDIF
# 580 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (filter_diag)
      IF (PRESENT(c_filter_diag)) THEN
         ALLOCATE (filter_diag)
         filter_diag = c_filter_diag
      ENDIF
# 586 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      CALL dbcsr_filter(matrix, c_eps, c_method, use_absolute, filter_diag)
# 588 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(use_absolute)) DEALLOCATE (use_absolute)
# 588 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(filter_diag)) DEALLOCATE (filter_diag)
# 590 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_block_diag(c_matrix, c_diag) &
      BIND(C, name="c_dbcsr_get_block_diag")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(c_ptr), INTENT(INOUT) :: c_diag
      TYPE(dbcsr_type), POINTER :: matrix, diag

      CALL c_f_pointer(c_matrix, matrix)
      IF (C_ASSOCIATED(c_diag)) THEN
         CALL c_f_pointer(c_diag, diag)
      ELSE
         ALLOCATE (diag)
      ENDIF

      CALL dbcsr_get_block_diag(matrix, diag)
      IF (.NOT. C_ASSOCIATED(c_diag)) c_diag = c_loc(diag)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_transposed(c_transposed, c_normal, c_shallow_data_copy, &
                                 c_transpose_data, c_transpose_distribution, c_use_distribution) &
      BIND(C, name="c_dbcsr_transposed")

      TYPE(c_ptr), INTENT(INOUT) :: c_transposed
      TYPE(c_ptr), INTENT(IN), VALUE :: c_normal, c_use_distribution
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_shallow_data_copy, c_transpose_data, &
                                                    c_transpose_distribution

      TYPE(dbcsr_type), POINTER :: transposed, normal
      LOGICAL, POINTER :: shallow_data_copy, transpose_data, &
                          transpose_distribution
      TYPE(dbcsr_distribution_type), POINTER :: use_distribution

      ALLOCATE (transposed)
      CALL c_f_pointer(c_normal, normal)
      IF (C_ASSOCIATED(c_use_distribution)) &
         CALL c_f_pointer(c_use_distribution, use_distribution)

# 630 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 631 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (shallow_data_copy)
      IF (PRESENT(c_shallow_data_copy)) THEN
         ALLOCATE (shallow_data_copy)
         shallow_data_copy = c_shallow_data_copy
      ENDIF
# 631 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (transpose_data)
      IF (PRESENT(c_transpose_data)) THEN
         ALLOCATE (transpose_data)
         transpose_data = c_transpose_data
      ENDIF
# 631 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (transpose_distribution)
      IF (PRESENT(c_transpose_distribution)) THEN
         ALLOCATE (transpose_distribution)
         transpose_distribution = c_transpose_distribution
      ENDIF
# 637 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

      CALL dbcsr_transposed(transposed, normal, shallow_data_copy, &
                            transpose_data, transpose_distribution, use_distribution)

      c_transposed = c_loc(transposed)

# 644 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(shallow_data_copy)) DEALLOCATE (shallow_data_copy)
# 644 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(transpose_data)) DEALLOCATE (transpose_data)
# 644 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(transpose_distribution)) DEALLOCATE (transpose_distribution)
# 646 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   END SUBROUTINE

   SUBROUTINE c_dbcsr_copy(c_matrix_b, c_matrix_a, c_name, c_keep_sparsity, &
                           c_shallow_data, c_keep_imaginary, c_matrix_type) &
      BIND(C, name="c_dbcsr_copy")

      TYPE(c_ptr), INTENT(INOUT) :: c_matrix_b
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a, c_name
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_keep_sparsity, c_shallow_data, &
                                                    c_keep_imaginary
      CHARACTER(kind=c_char), INTENT(IN), OPTIONAL :: c_matrix_type

      TYPE(dbcsr_type), POINTER :: matrix_b, matrix_a
      CHARACTER(:, kind=c_char), ALLOCATABLE  :: name
      LOGICAL, POINTER :: keep_sparsity, shallow_data, &
                          keep_imaginary

      IF (C_ASSOCIATED(c_matrix_b)) THEN
         CALL c_f_pointer(c_matrix_b, matrix_b)
      ELSE
         ALLOCATE (matrix_b)
      ENDIF

      IF (C_ASSOCIATED(c_name)) CALL c_f_string(c_name, name)

      CALL c_f_pointer(c_matrix_a, matrix_a)

# 675 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 676 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (keep_sparsity)
      IF (PRESENT(c_keep_sparsity)) THEN
         ALLOCATE (keep_sparsity)
         keep_sparsity = c_keep_sparsity
      ENDIF
# 676 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (shallow_data)
      IF (PRESENT(c_shallow_data)) THEN
         ALLOCATE (shallow_data)
         shallow_data = c_shallow_data
      ENDIF
# 676 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (keep_imaginary)
      IF (PRESENT(c_keep_imaginary)) THEN
         ALLOCATE (keep_imaginary)
         keep_imaginary = c_keep_imaginary
      ENDIF
# 682 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

      CALL dbcsr_copy(matrix_b, matrix_a, name, keep_sparsity, &
                      shallow_data, keep_imaginary, c_matrix_type)

# 687 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(keep_sparsity)) DEALLOCATE (keep_sparsity)
# 687 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(shallow_data)) DEALLOCATE (shallow_data)
# 687 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(keep_imaginary)) DEALLOCATE (keep_imaginary)
# 689 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (.NOT. C_ASSOCIATED(c_matrix_b)) c_matrix_b = c_loc(matrix_b)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_copy_into_existing(c_matrix_b, c_matrix_a) &
      BIND(C, name="c_dbcsr_copy_into_existing")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a, c_matrix_b
      TYPE(dbcsr_type), POINTER :: matrix_b, matrix_a

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)

      CALL dbcsr_copy_into_existing(matrix_b, matrix_a)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_desymmetrize(c_matrix_a, c_matrix_b) &
      BIND(C, name="c_dbcsr_desymmetrize")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a
      TYPE(c_ptr), INTENT(INOUT) :: c_matrix_b
      TYPE(dbcsr_type), POINTER :: matrix_a, matrix_b

      CALL c_f_pointer(c_matrix_a, matrix_a)
      IF (C_ASSOCIATED(c_matrix_b)) THEN
         CALL c_f_pointer(c_matrix_b, matrix_b)
      ELSE
         ALLOCATE (matrix_b)
      ENDIF

      CALL dbcsr_desymmetrize(matrix_a, matrix_b)
      IF (.NOT. C_ASSOCIATED(c_matrix_b)) c_matrix_b = c_loc(matrix_b)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_clear(c_dbcsr_mat) BIND(C, name="c_dbcsr_clear")
      TYPE(c_ptr), INTENT(INOUT) :: c_dbcsr_mat
      TYPE(dbcsr_type), POINTER :: dbcsr_mat
      CALL c_f_pointer(c_dbcsr_mat, dbcsr_mat)
      CALL dbcsr_clear(dbcsr_mat)
      c_dbcsr_mat = c_loc(dbcsr_mat)
   END SUBROUTINE

   ! block reservation
   !PUBLIC :: dbcsr_reserve_diag_blocks
   !PUBLIC :: dbcsr_reserve_block2d
   ! MODIFIED !!! PUBLIC :: dbcsr_reserve_blocks
   !PUBLIC :: dbcsr_reserve_all_blocks

   !-----------------------------------------------------------------!
   !                   block_reservations                            !
   !-----------------------------------------------------------------!

   SUBROUTINE c_dbcsr_reserve_diag_blocks(c_matrix) &
      BIND(C, name="c_dbcsr_reserve_diag_blocks")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix
      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_reserve_diag_blocks(matrix)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_reserve_blocks(c_matrix, c_rows, c_cols, c_size) &
      BIND(C, name="c_dbcsr_reserve_blocks")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_size
      INTEGER(kind=c_int), INTENT(IN) :: c_rows(c_size), c_cols(c_size)
      TYPE(dbcsr_type), POINTER:: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_reserve_blocks(matrix, c_rows + 1, c_cols + 1)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_reserve_all_blocks(c_matrix) &
      BIND(C, name="c_dbcsr_reserve_all_blocks")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_reserve_all_blocks(matrix)
   END SUBROUTINE

# 775 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   SUBROUTINE c_dbcsr_reserve_block2d_d (c_matrix, c_row, c_col, &
                                                    c_block, c_row_size, c_col_size, c_transposed, c_existed) &
      BIND(C, name="c_dbcsr_reserve_block2d_d")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN) :: c_row, c_col, c_row_size, c_col_size
      REAL(kind=c_double), INTENT(IN), DIMENSION(c_row_size, c_col_size), TARGET :: c_block
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_transposed
      LOGICAL(kind=c_bool), INTENT(OUT), OPTIONAL :: c_existed

      TYPE(dbcsr_type), POINTER :: matrix
      REAL(kind=real_8), DIMENSION(:, :), POINTER        :: block
      LOGICAL, POINTER :: transposed
      LOGICAL, POINTER :: existed

      CALL c_f_pointer(c_matrix, matrix)
      block => c_block
      NULLIFY (transposed)
      NULLIFY (existed)
      IF (PRESENT(c_transposed)) THEN
         ALLOCATE (transposed)
         transposed = c_transposed
      ENDIF
      IF (PRESENT(c_existed)) ALLOCATE (existed)

      CALL dbcsr_reserve_block2d(matrix, c_row + 1, c_col + 1, block, transposed, existed)
      IF (PRESENT(c_existed)) c_existed = existed
   END SUBROUTINE

# 775 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   SUBROUTINE c_dbcsr_reserve_block2d_s (c_matrix, c_row, c_col, &
                                                    c_block, c_row_size, c_col_size, c_transposed, c_existed) &
      BIND(C, name="c_dbcsr_reserve_block2d_s")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN) :: c_row, c_col, c_row_size, c_col_size
      REAL(kind=c_float), INTENT(IN), DIMENSION(c_row_size, c_col_size), TARGET :: c_block
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_transposed
      LOGICAL(kind=c_bool), INTENT(OUT), OPTIONAL :: c_existed

      TYPE(dbcsr_type), POINTER :: matrix
      REAL(kind=real_4), DIMENSION(:, :), POINTER        :: block
      LOGICAL, POINTER :: transposed
      LOGICAL, POINTER :: existed

      CALL c_f_pointer(c_matrix, matrix)
      block => c_block
      NULLIFY (transposed)
      NULLIFY (existed)
      IF (PRESENT(c_transposed)) THEN
         ALLOCATE (transposed)
         transposed = c_transposed
      ENDIF
      IF (PRESENT(c_existed)) ALLOCATE (existed)

      CALL dbcsr_reserve_block2d(matrix, c_row + 1, c_col + 1, block, transposed, existed)
      IF (PRESENT(c_existed)) c_existed = existed
   END SUBROUTINE

# 775 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   SUBROUTINE c_dbcsr_reserve_block2d_z (c_matrix, c_row, c_col, &
                                                    c_block, c_row_size, c_col_size, c_transposed, c_existed) &
      BIND(C, name="c_dbcsr_reserve_block2d_z")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN) :: c_row, c_col, c_row_size, c_col_size
      COMPLEX(kind=c_double_complex), INTENT(IN), DIMENSION(c_row_size, c_col_size), TARGET :: c_block
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_transposed
      LOGICAL(kind=c_bool), INTENT(OUT), OPTIONAL :: c_existed

      TYPE(dbcsr_type), POINTER :: matrix
      COMPLEX(kind=real_8), DIMENSION(:, :), POINTER        :: block
      LOGICAL, POINTER :: transposed
      LOGICAL, POINTER :: existed

      CALL c_f_pointer(c_matrix, matrix)
      block => c_block
      NULLIFY (transposed)
      NULLIFY (existed)
      IF (PRESENT(c_transposed)) THEN
         ALLOCATE (transposed)
         transposed = c_transposed
      ENDIF
      IF (PRESENT(c_existed)) ALLOCATE (existed)

      CALL dbcsr_reserve_block2d(matrix, c_row + 1, c_col + 1, block, transposed, existed)
      IF (PRESENT(c_existed)) c_existed = existed
   END SUBROUTINE

# 775 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   SUBROUTINE c_dbcsr_reserve_block2d_c (c_matrix, c_row, c_col, &
                                                    c_block, c_row_size, c_col_size, c_transposed, c_existed) &
      BIND(C, name="c_dbcsr_reserve_block2d_c")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN) :: c_row, c_col, c_row_size, c_col_size
      COMPLEX(kind=c_float_complex), INTENT(IN), DIMENSION(c_row_size, c_col_size), TARGET :: c_block
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_transposed
      LOGICAL(kind=c_bool), INTENT(OUT), OPTIONAL :: c_existed

      TYPE(dbcsr_type), POINTER :: matrix
      COMPLEX(kind=real_4), DIMENSION(:, :), POINTER        :: block
      LOGICAL, POINTER :: transposed
      LOGICAL, POINTER :: existed

      CALL c_f_pointer(c_matrix, matrix)
      block => c_block
      NULLIFY (transposed)
      NULLIFY (existed)
      IF (PRESENT(c_transposed)) THEN
         ALLOCATE (transposed)
         transposed = c_transposed
      ENDIF
      IF (PRESENT(c_existed)) ALLOCATE (existed)

      CALL dbcsr_reserve_block2d(matrix, c_row + 1, c_col + 1, block, transposed, existed)
      IF (PRESENT(c_existed)) c_existed = existed
   END SUBROUTINE

# 806 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   ! iterator
   !PUBLIC :: dbcsr_iterator_start
   !PUBLIC :: dbcsr_iterator_stop
   !PUBLIC :: dbcsr_iterator_blocks_left
   ! SOME MODS NEEDED PUBLIC :: dbcsr_iterator_next_block

   !-------------------------------!
   !        iterator               !
   !-------------------------------!

   SUBROUTINE c_dbcsr_iterator_stop(c_iterator) &
      BIND(C, name="c_dbcsr_iterator_stop")

      TYPE(c_ptr), INTENT(INOUT) :: c_iterator
      TYPE(dbcsr_iterator_type), POINTER :: iterator

      CALL c_f_pointer(c_iterator, iterator)
      CALL dbcsr_iterator_stop(iterator)
      IF (ASSOCIATED(iterator)) DEALLOCATE (iterator)
      c_iterator = c_null_ptr

   END SUBROUTINE

   SUBROUTINE c_dbcsr_iterator_start(c_iterator, c_matrix, c_shared, c_dynamic, &
                                     c_dynamic_byrows, c_contiguous_pointers, c_read_only) &
      BIND(C, name="c_dbcsr_iterator_start")

      TYPE(c_ptr), INTENT(INOUT) :: c_iterator
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_shared, c_dynamic, c_dynamic_byrows, &
                                                    c_contiguous_pointers, c_read_only
      TYPE(dbcsr_iterator_type), POINTER :: iterator
      TYPE(dbcsr_type), POINTER :: matrix
      LOGICAL, POINTER :: shared, dynamic, dynamic_byrows, &
                          contiguous_pointers, read_only

      ALLOCATE (iterator)
      CALL c_f_pointer(c_matrix, matrix)
# 846 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 847 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (shared)
      IF (PRESENT(c_shared)) THEN
         ALLOCATE (shared)
         shared = c_shared
      ENDIF
# 847 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (dynamic)
      IF (PRESENT(c_dynamic)) THEN
         ALLOCATE (dynamic)
         dynamic = c_dynamic
      ENDIF
# 847 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (dynamic_byrows)
      IF (PRESENT(c_dynamic_byrows)) THEN
         ALLOCATE (dynamic_byrows)
         dynamic_byrows = c_dynamic_byrows
      ENDIF
# 847 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (contiguous_pointers)
      IF (PRESENT(c_contiguous_pointers)) THEN
         ALLOCATE (contiguous_pointers)
         contiguous_pointers = c_contiguous_pointers
      ENDIF
# 847 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (read_only)
      IF (PRESENT(c_read_only)) THEN
         ALLOCATE (read_only)
         read_only = c_read_only
      ENDIF
# 853 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

      CALL dbcsr_iterator_start(iterator, matrix, shared, dynamic, &
                                dynamic_byrows, contiguous_pointers, read_only)

      c_iterator = c_loc(iterator)
# 859 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(shared)) DEALLOCATE (shared)
# 859 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(dynamic)) DEALLOCATE (dynamic)
# 859 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(dynamic_byrows)) DEALLOCATE (dynamic_byrows)
# 859 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(contiguous_pointers)) DEALLOCATE (contiguous_pointers)
# 859 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(read_only)) DEALLOCATE (read_only)
# 861 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   END SUBROUTINE

   FUNCTION c_dbcsr_iterator_blocks_left(c_iterator) RESULT(c_blocks_left) &
      BIND(C, name="c_dbcsr_iterator_blocks_left")
      TYPE(c_ptr), INTENT(IN), VALUE :: c_iterator
      TYPE(dbcsr_iterator_type), POINTER :: iterator
      LOGICAL(kind=c_bool) :: c_blocks_left

      CALL c_f_pointer(c_iterator, iterator)
      c_blocks_left = dbcsr_iterator_blocks_left(iterator)
   END FUNCTION

   SUBROUTINE c_dbcsr_iterator_next_block_index(c_iterator, c_row, c_column, c_blk, c_blk_p) &
      BIND(C, name="c_dbcsr_iterator_next_block_index")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_iterator
      INTEGER(kind=c_int), INTENT(OUT) :: c_row, c_column, c_blk
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_blk_p
      TYPE(dbcsr_iterator_type), POINTER :: iterator

      CALL c_f_pointer(c_iterator, iterator)

      CALL dbcsr_iterator_next_block(iterator, c_row, c_column, c_blk, c_blk_p)

      c_row = c_row - 1
      c_column = c_column - 1
      c_blk = c_blk - 1

   END SUBROUTINE

# 893 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   SUBROUTINE c_dbcsr_iterator_next_2d_block_d (c_iterator, c_row, c_column, c_block, &
                                                           c_transposed, c_block_number, &
                                                           c_row_size, c_col_size, c_row_offset, c_col_offset) &
      BIND(C, name="c_dbcsr_iterator_next_2d_block_d")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_iterator
      INTEGER(kind=c_int), INTENT(OUT) :: c_row, c_column
      TYPE(c_ptr), INTENT(INOUT) :: c_block
      LOGICAL(kind=c_bool), INTENT(OUT) :: c_transposed
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_block_number, c_row_size, &
                                                    c_col_size, c_row_offset, &
                                                    c_col_offset
      TYPE(dbcsr_iterator_type), POINTER :: iterator
      REAL(kind=c_double), DIMENSION(:, :), POINTER :: block
      LOGICAL :: transposed

      CALL c_f_pointer(c_iterator, iterator)

      CALL dbcsr_iterator_next_block(iterator, c_row, c_column, block, &
                                     transposed, c_block_number, &
                                     c_row_size, c_col_size, c_row_offset, c_col_offset)

      c_row = c_row - 1
      c_column = c_column - 1

      IF (PRESENT(c_block_number)) c_block_number = c_block_number - 1
      IF (PRESENT(c_row_offset)) c_row_offset = c_row_offset - 1
      IF (PRESENT(c_col_offset)) c_col_offset = c_col_offset - 1

      c_transposed = transposed
      c_block = c_loc(block)

   END SUBROUTINE
# 893 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   SUBROUTINE c_dbcsr_iterator_next_2d_block_s (c_iterator, c_row, c_column, c_block, &
                                                           c_transposed, c_block_number, &
                                                           c_row_size, c_col_size, c_row_offset, c_col_offset) &
      BIND(C, name="c_dbcsr_iterator_next_2d_block_s")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_iterator
      INTEGER(kind=c_int), INTENT(OUT) :: c_row, c_column
      TYPE(c_ptr), INTENT(INOUT) :: c_block
      LOGICAL(kind=c_bool), INTENT(OUT) :: c_transposed
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_block_number, c_row_size, &
                                                    c_col_size, c_row_offset, &
                                                    c_col_offset
      TYPE(dbcsr_iterator_type), POINTER :: iterator
      REAL(kind=c_float), DIMENSION(:, :), POINTER :: block
      LOGICAL :: transposed

      CALL c_f_pointer(c_iterator, iterator)

      CALL dbcsr_iterator_next_block(iterator, c_row, c_column, block, &
                                     transposed, c_block_number, &
                                     c_row_size, c_col_size, c_row_offset, c_col_offset)

      c_row = c_row - 1
      c_column = c_column - 1

      IF (PRESENT(c_block_number)) c_block_number = c_block_number - 1
      IF (PRESENT(c_row_offset)) c_row_offset = c_row_offset - 1
      IF (PRESENT(c_col_offset)) c_col_offset = c_col_offset - 1

      c_transposed = transposed
      c_block = c_loc(block)

   END SUBROUTINE
# 893 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   SUBROUTINE c_dbcsr_iterator_next_2d_block_z (c_iterator, c_row, c_column, c_block, &
                                                           c_transposed, c_block_number, &
                                                           c_row_size, c_col_size, c_row_offset, c_col_offset) &
      BIND(C, name="c_dbcsr_iterator_next_2d_block_z")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_iterator
      INTEGER(kind=c_int), INTENT(OUT) :: c_row, c_column
      TYPE(c_ptr), INTENT(INOUT) :: c_block
      LOGICAL(kind=c_bool), INTENT(OUT) :: c_transposed
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_block_number, c_row_size, &
                                                    c_col_size, c_row_offset, &
                                                    c_col_offset
      TYPE(dbcsr_iterator_type), POINTER :: iterator
      COMPLEX(kind=c_double_complex), DIMENSION(:, :), POINTER :: block
      LOGICAL :: transposed

      CALL c_f_pointer(c_iterator, iterator)

      CALL dbcsr_iterator_next_block(iterator, c_row, c_column, block, &
                                     transposed, c_block_number, &
                                     c_row_size, c_col_size, c_row_offset, c_col_offset)

      c_row = c_row - 1
      c_column = c_column - 1

      IF (PRESENT(c_block_number)) c_block_number = c_block_number - 1
      IF (PRESENT(c_row_offset)) c_row_offset = c_row_offset - 1
      IF (PRESENT(c_col_offset)) c_col_offset = c_col_offset - 1

      c_transposed = transposed
      c_block = c_loc(block)

   END SUBROUTINE
# 893 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   SUBROUTINE c_dbcsr_iterator_next_2d_block_c (c_iterator, c_row, c_column, c_block, &
                                                           c_transposed, c_block_number, &
                                                           c_row_size, c_col_size, c_row_offset, c_col_offset) &
      BIND(C, name="c_dbcsr_iterator_next_2d_block_c")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_iterator
      INTEGER(kind=c_int), INTENT(OUT) :: c_row, c_column
      TYPE(c_ptr), INTENT(INOUT) :: c_block
      LOGICAL(kind=c_bool), INTENT(OUT) :: c_transposed
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_block_number, c_row_size, &
                                                    c_col_size, c_row_offset, &
                                                    c_col_offset
      TYPE(dbcsr_iterator_type), POINTER :: iterator
      COMPLEX(kind=c_float_complex), DIMENSION(:, :), POINTER :: block
      LOGICAL :: transposed

      CALL c_f_pointer(c_iterator, iterator)

      CALL dbcsr_iterator_next_block(iterator, c_row, c_column, block, &
                                     transposed, c_block_number, &
                                     c_row_size, c_col_size, c_row_offset, c_col_offset)

      c_row = c_row - 1
      c_column = c_column - 1

      IF (PRESENT(c_block_number)) c_block_number = c_block_number - 1
      IF (PRESENT(c_row_offset)) c_row_offset = c_row_offset - 1
      IF (PRESENT(c_col_offset)) c_col_offset = c_col_offset - 1

      c_transposed = transposed
      c_block = c_loc(block)

   END SUBROUTINE
# 927 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   !--------------------------------------------------------!
   !                  work operations                       !
   !--------------------------------------------------------!

   ! SKIP PUBLIC :: dbcsr_add_block_node
   !PUBLIC :: dbcsr_put_block
   ! SKIP PUBLIC :: dbcsr_work_create
   ! SKIP PUBLIC :: dbcsr_verify_matrix
   ! SKIP PUBLIC :: dbcsr_add_work_coordinate
   ! SKIP PUBLIC :: dbcsr_get_wms_data_p
   ! MODIFIED PUBLIC :: dbcsr_get_data_p
   ! SKIP PUBLIC :: dbcsr_set_work_size
   ! SKIP PUBLIC :: dbcsr_finalize

# 943 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   SUBROUTINE c_dbcsr_put_block2d_d (c_matrix, c_row, c_col, c_block, &
                                                c_row_size, c_col_size, c_summation, c_scale) &
      BIND(C, name="c_dbcsr_put_block2d_d")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row, c_col, c_row_size, c_col_size
      REAL(kind=c_double), INTENT(IN) :: c_block(c_row_size, c_col_size)
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_summation
      REAL(kind=c_double), INTENT(IN), OPTIONAL :: c_scale

      TYPE(dbcsr_type), POINTER :: matrix
      LOGICAL :: summation

      CALL c_f_pointer(c_matrix, matrix)

      IF (PRESENT(c_summation)) THEN
         summation = c_summation
         CALL dbcsr_put_block(matrix, c_row + 1, c_col + 1, c_block, summation, c_scale)
      ELSE
         CALL dbcsr_put_block(matrix, c_row + 1, c_col + 1, c_block, scale=c_scale)
      ENDIF
   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_data_d (c_matrix, c_data, c_data_size, c_select_data_type, c_lb, c_ub) &
      BIND(C, name="c_dbcsr_get_data_d")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(c_ptr), INTENT(OUT) :: c_data
      INTEGER(kind=c_long_long), INTENT(OUT) :: c_data_size
      REAL(kind=c_double), INTENT(IN) :: c_select_data_type
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_lb, c_ub
      INTEGER(kind=c_int), POINTER :: lb, ub
      TYPE(dbcsr_type), POINTER :: matrix
      REAL(kind=c_double), DIMENSION(:), POINTER :: data

      CALL c_f_pointer(c_matrix, matrix)
# 980 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 981 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (lb)
      IF (PRESENT(c_lb)) THEN
         ALLOCATE (lb)
         lb = c_lb+1
      ENDIF
# 981 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (ub)
      IF (PRESENT(c_ub)) THEN
         ALLOCATE (ub)
         ub = c_ub+1
      ENDIF
# 987 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

      data => dbcsr_get_data_p(matrix, c_select_data_type, lb, ub)
      c_data = c_loc(data)
      c_data_size = SIZE(data)

   END SUBROUTINE

# 943 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   SUBROUTINE c_dbcsr_put_block2d_s (c_matrix, c_row, c_col, c_block, &
                                                c_row_size, c_col_size, c_summation, c_scale) &
      BIND(C, name="c_dbcsr_put_block2d_s")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row, c_col, c_row_size, c_col_size
      REAL(kind=c_float), INTENT(IN) :: c_block(c_row_size, c_col_size)
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_summation
      REAL(kind=c_float), INTENT(IN), OPTIONAL :: c_scale

      TYPE(dbcsr_type), POINTER :: matrix
      LOGICAL :: summation

      CALL c_f_pointer(c_matrix, matrix)

      IF (PRESENT(c_summation)) THEN
         summation = c_summation
         CALL dbcsr_put_block(matrix, c_row + 1, c_col + 1, c_block, summation, c_scale)
      ELSE
         CALL dbcsr_put_block(matrix, c_row + 1, c_col + 1, c_block, scale=c_scale)
      ENDIF
   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_data_s (c_matrix, c_data, c_data_size, c_select_data_type, c_lb, c_ub) &
      BIND(C, name="c_dbcsr_get_data_s")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(c_ptr), INTENT(OUT) :: c_data
      INTEGER(kind=c_long_long), INTENT(OUT) :: c_data_size
      REAL(kind=c_float), INTENT(IN) :: c_select_data_type
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_lb, c_ub
      INTEGER(kind=c_int), POINTER :: lb, ub
      TYPE(dbcsr_type), POINTER :: matrix
      REAL(kind=c_float), DIMENSION(:), POINTER :: data

      CALL c_f_pointer(c_matrix, matrix)
# 980 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 981 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (lb)
      IF (PRESENT(c_lb)) THEN
         ALLOCATE (lb)
         lb = c_lb+1
      ENDIF
# 981 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (ub)
      IF (PRESENT(c_ub)) THEN
         ALLOCATE (ub)
         ub = c_ub+1
      ENDIF
# 987 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

      data => dbcsr_get_data_p(matrix, c_select_data_type, lb, ub)
      c_data = c_loc(data)
      c_data_size = SIZE(data)

   END SUBROUTINE

# 943 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   SUBROUTINE c_dbcsr_put_block2d_z (c_matrix, c_row, c_col, c_block, &
                                                c_row_size, c_col_size, c_summation, c_scale) &
      BIND(C, name="c_dbcsr_put_block2d_z")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row, c_col, c_row_size, c_col_size
      COMPLEX(kind=c_double_complex), INTENT(IN) :: c_block(c_row_size, c_col_size)
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_summation
      COMPLEX(kind=c_double_complex), INTENT(IN), OPTIONAL :: c_scale

      TYPE(dbcsr_type), POINTER :: matrix
      LOGICAL :: summation

      CALL c_f_pointer(c_matrix, matrix)

      IF (PRESENT(c_summation)) THEN
         summation = c_summation
         CALL dbcsr_put_block(matrix, c_row + 1, c_col + 1, c_block, summation, c_scale)
      ELSE
         CALL dbcsr_put_block(matrix, c_row + 1, c_col + 1, c_block, scale=c_scale)
      ENDIF
   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_data_z (c_matrix, c_data, c_data_size, c_select_data_type, c_lb, c_ub) &
      BIND(C, name="c_dbcsr_get_data_z")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(c_ptr), INTENT(OUT) :: c_data
      INTEGER(kind=c_long_long), INTENT(OUT) :: c_data_size
      COMPLEX(kind=c_double_complex), INTENT(IN) :: c_select_data_type
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_lb, c_ub
      INTEGER(kind=c_int), POINTER :: lb, ub
      TYPE(dbcsr_type), POINTER :: matrix
      COMPLEX(kind=c_double_complex), DIMENSION(:), POINTER :: data

      CALL c_f_pointer(c_matrix, matrix)
# 980 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 981 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (lb)
      IF (PRESENT(c_lb)) THEN
         ALLOCATE (lb)
         lb = c_lb+1
      ENDIF
# 981 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (ub)
      IF (PRESENT(c_ub)) THEN
         ALLOCATE (ub)
         ub = c_ub+1
      ENDIF
# 987 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

      data => dbcsr_get_data_p(matrix, c_select_data_type, lb, ub)
      c_data = c_loc(data)
      c_data_size = SIZE(data)

   END SUBROUTINE

# 943 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   SUBROUTINE c_dbcsr_put_block2d_c (c_matrix, c_row, c_col, c_block, &
                                                c_row_size, c_col_size, c_summation, c_scale) &
      BIND(C, name="c_dbcsr_put_block2d_c")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row, c_col, c_row_size, c_col_size
      COMPLEX(kind=c_float_complex), INTENT(IN) :: c_block(c_row_size, c_col_size)
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_summation
      COMPLEX(kind=c_float_complex), INTENT(IN), OPTIONAL :: c_scale

      TYPE(dbcsr_type), POINTER :: matrix
      LOGICAL :: summation

      CALL c_f_pointer(c_matrix, matrix)

      IF (PRESENT(c_summation)) THEN
         summation = c_summation
         CALL dbcsr_put_block(matrix, c_row + 1, c_col + 1, c_block, summation, c_scale)
      ELSE
         CALL dbcsr_put_block(matrix, c_row + 1, c_col + 1, c_block, scale=c_scale)
      ENDIF
   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_data_c (c_matrix, c_data, c_data_size, c_select_data_type, c_lb, c_ub) &
      BIND(C, name="c_dbcsr_get_data_c")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(c_ptr), INTENT(OUT) :: c_data
      INTEGER(kind=c_long_long), INTENT(OUT) :: c_data_size
      COMPLEX(kind=c_float_complex), INTENT(IN) :: c_select_data_type
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_lb, c_ub
      INTEGER(kind=c_int), POINTER :: lb, ub
      TYPE(dbcsr_type), POINTER :: matrix
      COMPLEX(kind=c_float_complex), DIMENSION(:), POINTER :: data

      CALL c_f_pointer(c_matrix, matrix)
# 980 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 981 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (lb)
      IF (PRESENT(c_lb)) THEN
         ALLOCATE (lb)
         lb = c_lb+1
      ENDIF
# 981 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (ub)
      IF (PRESENT(c_ub)) THEN
         ALLOCATE (ub)
         ub = c_ub+1
      ENDIF
# 987 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

      data => dbcsr_get_data_p(matrix, c_select_data_type, lb, ub)
      c_data = c_loc(data)
      c_data_size = SIZE(data)

   END SUBROUTINE

# 995 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   !------------------------------------------------------------!
   !                   replication                              !
   !------------------------------------------------------------!

   !PUBLIC :: dbcsr_replicate_all
   !PUBLIC :: dbcsr_sum_replicated
   !PUBLIC :: dbcsr_distribute

   SUBROUTINE c_dbcsr_replicate_all(c_matrix) &
      BIND(C, name="c_dbcsr_replicate_all")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_replicate_all(matrix)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_distribute(c_matrix, c_fast) &
      BIND(C, name="c_dbcsr_distribute")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_fast
      TYPE(dbcsr_type), POINTER :: matrix
      LOGICAL :: fast

      CALL c_f_pointer(c_matrix, matrix)
      IF (PRESENT(c_fast)) THEN
         fast = c_fast
         CALL dbcsr_distribute(matrix, fast)
      ELSE
         CALL dbcsr_distribute(matrix)
      ENDIF
   END SUBROUTINE

   SUBROUTINE c_dbcsr_sum_replicated(c_matrix) &
      BIND(C, name="c_dbcsr_sum_replicated")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_sum_replicated(matrix)
   END SUBROUTINE

   !PUBLIC :: dbcsr_norm_frobenius
   !PUBLIC :: dbcsr_norm_maxabsnorm
   !PUBLIC :: dbcsr_norm_column
   !PUBLIC :: dbcsr_hadamard_product
   !PUBLIC :: dbcsr_func_artanh
   !PUBLIC :: dbcsr_func_dtanh
   !PUBLIC :: dbcsr_func_inverse
   !PUBLIC :: dbcsr_func_tanh
   !PUBLIC :: dbcsr_print
   !PUBLIC :: dbcsr_print_block_sum
   !PUBLIC :: dbcsr_checksum
   !PUBLIC :: dbcsr_maxabs
   ! VECTOR? PUBLIC :: dbcsr_norm
   !PUBLIC :: dbcsr_gershgorin_norm
   !PUBLIC :: dbcsr_frobenius_norm
   !PUBLIC :: dbcsr_init_random
   !PUBLIC :: dbcsr_function_of_elements
   !PUBLIC :: dbcsr_triu

   !-----------------------------------------!
   !       high level matrix functions       !
   !-----------------------------------------!

   SUBROUTINE c_dbcsr_hadamard_product(c_matrix_a, c_matrix_b, c_matrix_c, c_b_assume_value) &
      BIND(C, name="c_dbcsr_hadamard_product")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a, c_matrix_b, c_matrix_c
      REAL(kind=c_double), INTENT(IN), OPTIONAL :: c_b_assume_value
      TYPE(dbcsr_type), POINTER :: matrix_a, matrix_b, matrix_c

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      CALL c_f_pointer(c_matrix_c, matrix_c)

      CALL dbcsr_hadamard_product(matrix_a, matrix_b, matrix_c, c_b_assume_value)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_print(c_matrix) bind(C, name="c_dbcsr_print")
      TYPE(c_ptr), INTENT(in), value                     :: c_matrix

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_print(matrix)

      ! Fortran and C may use different buffers for I/O, make sure we flush before returning:
      flush (default_output_unit)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_print_block_sum(c_matrix, c_unit_nr) &
      BIND(C, name="c_dbcsr_print_block_sum")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_unit_nr
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_print_block_sum(matrix, c_unit_nr)
      FLUSH (default_output_unit)
   END SUBROUTINE

   FUNCTION c_dbcsr_checksum(c_matrix, c_local, c_pos) RESULT(c_checksum) &
      BIND(C, name="c_dbcsr_checksum")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_local, c_pos
      TYPE(dbcsr_type), POINTER :: matrix
      LOGICAL, POINTER :: local, pos
      REAL(kind=c_double) :: c_checksum

      CALL c_f_pointer(c_matrix, matrix)
# 1115 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 1116 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (local)
      IF (PRESENT(c_local)) THEN
         ALLOCATE (local)
         local = c_local
      ENDIF
# 1116 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (pos)
      IF (PRESENT(c_pos)) THEN
         ALLOCATE (pos)
         pos = c_pos
      ENDIF
# 1122 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      c_checksum = dbcsr_checksum(matrix, local, pos)
# 1124 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(local)) DEALLOCATE (local)
# 1124 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (ASSOCIATED(pos)) DEALLOCATE (pos)
# 1126 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   END FUNCTION

   FUNCTION c_dbcsr_maxabs(c_matrix) RESULT(c_norm) &
      BIND(C, name="c_dbcsr_maxabs")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix
      REAL(kind=c_double) :: c_norm

      CALL c_f_pointer(c_matrix, matrix)
      c_norm = dbcsr_maxabs(matrix)
   END FUNCTION

   FUNCTION c_dbcsr_gershgorin_norm(c_matrix) RESULT(c_norm) &
      BIND(C, name="c_dbcsr_gershgorin_norm")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix
      REAL(kind=c_double) :: c_norm

      CALL c_f_pointer(c_matrix, matrix)
      c_norm = dbcsr_gershgorin_norm(matrix)
   END FUNCTION

   FUNCTION c_dbcsr_frobenius_norm(c_matrix, c_local) RESULT(c_norm) &
      BIND(C, name="c_dbcsr_frobenius_norm")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_local
      TYPE(dbcsr_type), POINTER :: matrix
      LOGICAL :: local
      REAL(kind=c_double) :: c_norm

      CALL c_f_pointer(c_matrix, matrix)
      IF (PRESENT(c_local)) THEN
         local = c_local
         c_norm = dbcsr_frobenius_norm(matrix, local)
      ELSE
         c_norm = dbcsr_frobenius_norm(matrix)
      ENDIF
   END FUNCTION

   SUBROUTINE c_dbcsr_norm_scalar(c_matrix, c_which_norm, c_norm_scalar) &
      BIND(C, name="c_dbcsr_norm_scalar")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_which_norm
      REAL(kind=c_double), INTENT(OUT) :: c_norm_scalar
      TYPE(dbcsr_type), POINTER  :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_norm(matrix, c_which_norm, c_norm_scalar)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_triu(c_matrix) BIND(C, name="c_dbcsr_triu")
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_triu(matrix)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_init_random(c_matrix, c_keep_sparsity) &
      BIND(C, name="c_dbcsr_init_random")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_keep_sparsity
      TYPE(dbcsr_type), POINTER :: matrix
      LOGICAL :: keep_sparsity

      CALL c_f_pointer(c_matrix, matrix)
      IF (PRESENT(c_keep_sparsity)) THEN
         keep_sparsity = c_keep_sparsity
         CALL dbcsr_init_random(matrix, keep_sparsity)
      ELSE
         CALL dbcsr_init_random(matrix)
      ENDIF
   END SUBROUTINE

   SUBROUTINE c_dbcsr_function_of_elements(c_matrix, c_func, c_a0, c_a1, c_a2) &
      BIND(C, name="c_dbcsr_function_of_elements")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_func
      REAL(kind=c_double), INTENT(IN), OPTIONAL :: c_a0, c_a1, c_a2
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)
      CALL dbcsr_function_of_elements(matrix, c_func, c_a0, c_a1, c_a2)
   END SUBROUTINE

   ! ---------------------------------------------------- !
   !                 getters / setters                    !
   ! ---------------------------------------------------- !

   ! getters / setters
   !!PUBLIC :: dbcsr_get_info
   !!PUBLIC :: dbcsr_distribution_get
   !!PUBLIC :: dbcsr_setname
   !!PUBLIC :: dbcsr_get_matrix_type
   !!PUBLIC :: dbcsr_get_occupation
   !!PUBLIC :: dbcsr_nblkrows_total
   !!PUBLIC :: dbcsr_nblkcols_total
   ! ADDED LOCAL TOO
   !!PUBLIC :: dbcsr_get_num_blocks
   !!PUBLIC :: dbcsr_get_data_size
   !!PUBLIC :: dbcsr_has_symmetry
   !!PUBLIC :: dbcsr_nfullrows_total
   !!PUBLIC :: dbcsr_nfullcols_total
   !!PUBLIC :: dbcsr_get_stored_coordinates
   !!PUBLIC :: dbcsr_valid_index
   !!PUBLIC :: dbcsr_get_data_type

   FUNCTION c_dbcsr_nblkrows_total(c_matrix) RESULT(c_nblkrows_tot) &
      BIND(C, name="c_dbcsr_nblkrows_total")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int) :: c_nblkrows_tot

      CALL c_f_pointer(c_matrix, matrix)
      c_nblkrows_tot = dbcsr_nblkrows_total(matrix)

   END FUNCTION

   FUNCTION c_dbcsr_nblkcols_total(c_matrix) RESULT(c_nblkcols_tot) &
      BIND(C, name="c_dbcsr_nblkcols_total")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int) :: c_nblkcols_tot

      CALL c_f_pointer(c_matrix, matrix)
      c_nblkcols_tot = dbcsr_nblkcols_total(matrix)

   END FUNCTION

   FUNCTION c_dbcsr_nblkrows_local(c_matrix) RESULT(c_nblkrows_loc) &
      BIND(C, name="c_dbcsr_nblkrows_local")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int) :: c_nblkrows_loc

      CALL c_f_pointer(c_matrix, matrix)
      c_nblkrows_loc = dbcsr_nblkrows_local(matrix)

   END FUNCTION

   FUNCTION c_dbcsr_nblkcols_local(c_matrix) RESULT(c_nblkcols_loc) &
      BIND(C, name="c_dbcsr_nblkcols_local")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int) :: c_nblkcols_loc

      CALL c_f_pointer(c_matrix, matrix)
      c_nblkcols_loc = dbcsr_nblkcols_local(matrix)

   END FUNCTION

   SUBROUTINE c_dbcsr_get_info(c_matrix, c_nblkrows_total, c_nblkcols_total, &
                               c_nfullrows_total, c_nfullcols_total, &
                               c_nblkrows_local, c_nblkcols_local, &
                               c_nfullrows_local, c_nfullcols_local, &
                               c_my_prow, c_my_pcol, &
                               c_local_rows, c_local_cols, c_proc_row_dist, c_proc_col_dist, &
                               c_row_blk_size, c_col_blk_size, c_row_blk_offset, c_col_blk_offset, &
                               c_distribution, c_name, c_matrix_type, c_data_type, &
                               c_group) BIND(C, name="c_dbcsr_get_info")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_nblkrows_total, c_nblkcols_total, c_nfullrows_total, &
                                                    c_nfullcols_total, c_nblkrows_local, c_nblkcols_local, c_nfullrows_local, &
                                                    c_nfullcols_local, c_my_prow, c_my_pcol
      TYPE(c_ptr), INTENT(IN), VALUE :: c_local_rows, c_local_cols, c_proc_row_dist, &
                                        c_proc_col_dist, c_row_blk_size, c_col_blk_size, &
                                        c_row_blk_offset, c_col_blk_offset
      TYPE(c_ptr), INTENT(OUT), OPTIONAL :: c_distribution
      TYPE(c_ptr), INTENT(OUT), OPTIONAL :: c_name
      CHARACTER(kind=c_char), INTENT(OUT), OPTIONAL :: c_matrix_type
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_data_type, c_group

      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int), DIMENSION(:), POINTER :: f_local_rows, f_local_cols, f_proc_row_dist, &
                                                    f_proc_col_dist, &
                                                    f_row_blk_size, f_col_blk_size, &
                                                    f_row_blk_offset, f_col_blk_offset
      ! copies for the following arrays
      INTEGER(kind=c_int), DIMENSION(:), POINTER :: local_rows, local_cols, proc_row_dist, &
                                                    proc_col_dist, &
                                                    row_blk_size, col_blk_size, &
                                                    row_blk_offset, col_blk_offset
      TYPE(dbcsr_distribution_type), POINTER :: distribution
      CHARACTER(kind=c_char, len=:), POINTER :: name

      CALL c_f_pointer(c_matrix, matrix)

      ! Because the pointers passed are always null at the beginning, we cannot just pass them in this case
      ! we use if/else branches, but reduce their number by grouping some variables together
      ! and splitting the function into several calls

# 1332 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 1333 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      

# 1339 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

      ! This will generate 16 branches
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF(C_ASSOCIATED(c_local_rows) .AND. C_ASSOCIATED(c_local_cols) .AND. C_ASSOCIATED(c_proc_row_dist) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_proc_col_dist) .AND. C_ASSOCIATED(c_row_blk_size) .AND. C_ASSOCIATED(c_col_blk_size) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_row_blk_offset) .AND. C_ASSOCIATED(c_col_blk_offset)) THEN 

      CALL dbcsr_get_info(matrix, c_nblkrows_total, c_nblkcols_total, &
                          c_nfullrows_total, c_nfullcols_total, &
                          c_nblkrows_local, c_nblkcols_local, &
                          c_nfullrows_local, c_nfullcols_local, &
                          c_my_prow, c_my_pcol &
                          , local_rows = local_rows, local_cols = local_cols, proc_row_dist = proc_row_dist, proc_col_dist =&
# 1349 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
                              & proc_col_dist, row_blk_size = row_blk_size, col_blk_size = col_blk_size, row_blk_offset =&
# 1349 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
                              & row_blk_offset, col_blk_offset = col_blk_offset, &
                          matrix_type=c_matrix_type, data_type=c_data_type, &
                          group=c_group)
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(C_ASSOCIATED(c_proc_row_dist) .AND. C_ASSOCIATED(c_proc_col_dist) .AND. C_ASSOCIATED(c_row_blk_size) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_col_blk_size) .AND. C_ASSOCIATED(c_row_blk_offset) .AND. C_ASSOCIATED(c_col_blk_offset) .AND. .NOT.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_local_rows) .AND. .NOT. C_ASSOCIATED(c_local_cols)) THEN 

      CALL dbcsr_get_info(matrix, c_nblkrows_total, c_nblkcols_total, &
                          c_nfullrows_total, c_nfullcols_total, &
                          c_nblkrows_local, c_nblkcols_local, &
                          c_nfullrows_local, c_nfullcols_local, &
                          c_my_prow, c_my_pcol &
                          , proc_row_dist = proc_row_dist, proc_col_dist = proc_col_dist, row_blk_size = row_blk_size,&
# 1349 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
                              & col_blk_size = col_blk_size, row_blk_offset = row_blk_offset, col_blk_offset = col_blk_offset, &
                          matrix_type=c_matrix_type, data_type=c_data_type, &
                          group=c_group)
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(C_ASSOCIATED(c_local_rows) .AND. C_ASSOCIATED(c_local_cols) .AND. C_ASSOCIATED(c_row_blk_size) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_col_blk_size) .AND. C_ASSOCIATED(c_row_blk_offset) .AND. C_ASSOCIATED(c_col_blk_offset) .AND. .NOT.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_proc_row_dist) .AND. .NOT. C_ASSOCIATED(c_proc_col_dist)) THEN 

      CALL dbcsr_get_info(matrix, c_nblkrows_total, c_nblkcols_total, &
                          c_nfullrows_total, c_nfullcols_total, &
                          c_nblkrows_local, c_nblkcols_local, &
                          c_nfullrows_local, c_nfullcols_local, &
                          c_my_prow, c_my_pcol &
                          , local_rows = local_rows, local_cols = local_cols, row_blk_size = row_blk_size, col_blk_size =&
# 1349 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
                              & col_blk_size, row_blk_offset = row_blk_offset, col_blk_offset = col_blk_offset, &
                          matrix_type=c_matrix_type, data_type=c_data_type, &
                          group=c_group)
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(C_ASSOCIATED(c_row_blk_size) .AND. C_ASSOCIATED(c_col_blk_size) .AND. C_ASSOCIATED(c_row_blk_offset) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_col_blk_offset) .AND. .NOT. C_ASSOCIATED(c_local_rows) .AND. .NOT. C_ASSOCIATED(c_local_cols) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & .NOT. C_ASSOCIATED(c_proc_row_dist) .AND. .NOT. C_ASSOCIATED(c_proc_col_dist)) THEN 

      CALL dbcsr_get_info(matrix, c_nblkrows_total, c_nblkcols_total, &
                          c_nfullrows_total, c_nfullcols_total, &
                          c_nblkrows_local, c_nblkcols_local, &
                          c_nfullrows_local, c_nfullcols_local, &
                          c_my_prow, c_my_pcol &
                          , row_blk_size = row_blk_size, col_blk_size = col_blk_size, row_blk_offset = row_blk_offset,&
# 1349 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
                              & col_blk_offset = col_blk_offset, &
                          matrix_type=c_matrix_type, data_type=c_data_type, &
                          group=c_group)
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(C_ASSOCIATED(c_local_rows) .AND. C_ASSOCIATED(c_local_cols) .AND. C_ASSOCIATED(c_proc_row_dist) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_proc_col_dist) .AND. C_ASSOCIATED(c_row_blk_offset) .AND. C_ASSOCIATED(c_col_blk_offset) .AND. .NOT.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_row_blk_size) .AND. .NOT. C_ASSOCIATED(c_col_blk_size)) THEN 

      CALL dbcsr_get_info(matrix, c_nblkrows_total, c_nblkcols_total, &
                          c_nfullrows_total, c_nfullcols_total, &
                          c_nblkrows_local, c_nblkcols_local, &
                          c_nfullrows_local, c_nfullcols_local, &
                          c_my_prow, c_my_pcol &
                          , local_rows = local_rows, local_cols = local_cols, proc_row_dist = proc_row_dist, proc_col_dist =&
# 1349 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
                              & proc_col_dist, row_blk_offset = row_blk_offset, col_blk_offset = col_blk_offset, &
                          matrix_type=c_matrix_type, data_type=c_data_type, &
                          group=c_group)
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(C_ASSOCIATED(c_proc_row_dist) .AND. C_ASSOCIATED(c_proc_col_dist) .AND. C_ASSOCIATED(c_row_blk_offset) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_col_blk_offset) .AND. .NOT. C_ASSOCIATED(c_local_rows) .AND. .NOT. C_ASSOCIATED(c_local_cols) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & .NOT. C_ASSOCIATED(c_row_blk_size) .AND. .NOT. C_ASSOCIATED(c_col_blk_size)) THEN 

      CALL dbcsr_get_info(matrix, c_nblkrows_total, c_nblkcols_total, &
                          c_nfullrows_total, c_nfullcols_total, &
                          c_nblkrows_local, c_nblkcols_local, &
                          c_nfullrows_local, c_nfullcols_local, &
                          c_my_prow, c_my_pcol &
                          , proc_row_dist = proc_row_dist, proc_col_dist = proc_col_dist, row_blk_offset = row_blk_offset,&
# 1349 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
                              & col_blk_offset = col_blk_offset, &
                          matrix_type=c_matrix_type, data_type=c_data_type, &
                          group=c_group)
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(C_ASSOCIATED(c_local_rows) .AND. C_ASSOCIATED(c_local_cols) .AND. C_ASSOCIATED(c_row_blk_offset) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_col_blk_offset) .AND. .NOT. C_ASSOCIATED(c_proc_row_dist) .AND. .NOT. C_ASSOCIATED(c_proc_col_dist)&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & .AND. .NOT. C_ASSOCIATED(c_row_blk_size) .AND. .NOT. C_ASSOCIATED(c_col_blk_size)) THEN 

      CALL dbcsr_get_info(matrix, c_nblkrows_total, c_nblkcols_total, &
                          c_nfullrows_total, c_nfullcols_total, &
                          c_nblkrows_local, c_nblkcols_local, &
                          c_nfullrows_local, c_nfullcols_local, &
                          c_my_prow, c_my_pcol &
                          , local_rows = local_rows, local_cols = local_cols, row_blk_offset = row_blk_offset, col_blk_offset =&
# 1349 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
                              & col_blk_offset, &
                          matrix_type=c_matrix_type, data_type=c_data_type, &
                          group=c_group)
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(C_ASSOCIATED(c_row_blk_offset) .AND. C_ASSOCIATED(c_col_blk_offset) .AND. .NOT. C_ASSOCIATED(c_local_rows) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & .NOT. C_ASSOCIATED(c_local_cols) .AND. .NOT. C_ASSOCIATED(c_proc_row_dist) .AND. .NOT. C_ASSOCIATED(c_proc_col_dist)&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & .AND. .NOT. C_ASSOCIATED(c_row_blk_size) .AND. .NOT. C_ASSOCIATED(c_col_blk_size)) THEN 

      CALL dbcsr_get_info(matrix, c_nblkrows_total, c_nblkcols_total, &
                          c_nfullrows_total, c_nfullcols_total, &
                          c_nblkrows_local, c_nblkcols_local, &
                          c_nfullrows_local, c_nfullcols_local, &
                          c_my_prow, c_my_pcol &
                          , row_blk_offset = row_blk_offset, col_blk_offset = col_blk_offset, &
                          matrix_type=c_matrix_type, data_type=c_data_type, &
                          group=c_group)
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(C_ASSOCIATED(c_local_rows) .AND. C_ASSOCIATED(c_local_cols) .AND. C_ASSOCIATED(c_proc_row_dist) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_proc_col_dist) .AND. C_ASSOCIATED(c_row_blk_size) .AND. C_ASSOCIATED(c_col_blk_size) .AND. .NOT.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_row_blk_offset) .AND. .NOT. C_ASSOCIATED(c_col_blk_offset)) THEN 

      CALL dbcsr_get_info(matrix, c_nblkrows_total, c_nblkcols_total, &
                          c_nfullrows_total, c_nfullcols_total, &
                          c_nblkrows_local, c_nblkcols_local, &
                          c_nfullrows_local, c_nfullcols_local, &
                          c_my_prow, c_my_pcol &
                          , local_rows = local_rows, local_cols = local_cols, proc_row_dist = proc_row_dist, proc_col_dist =&
# 1349 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
                              & proc_col_dist, row_blk_size = row_blk_size, col_blk_size = col_blk_size, &
                          matrix_type=c_matrix_type, data_type=c_data_type, &
                          group=c_group)
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(C_ASSOCIATED(c_proc_row_dist) .AND. C_ASSOCIATED(c_proc_col_dist) .AND. C_ASSOCIATED(c_row_blk_size) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_col_blk_size) .AND. .NOT. C_ASSOCIATED(c_local_rows) .AND. .NOT. C_ASSOCIATED(c_local_cols) .AND. .NOT.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_row_blk_offset) .AND. .NOT. C_ASSOCIATED(c_col_blk_offset)) THEN 

      CALL dbcsr_get_info(matrix, c_nblkrows_total, c_nblkcols_total, &
                          c_nfullrows_total, c_nfullcols_total, &
                          c_nblkrows_local, c_nblkcols_local, &
                          c_nfullrows_local, c_nfullcols_local, &
                          c_my_prow, c_my_pcol &
                          , proc_row_dist = proc_row_dist, proc_col_dist = proc_col_dist, row_blk_size = row_blk_size,&
# 1349 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
                              & col_blk_size = col_blk_size, &
                          matrix_type=c_matrix_type, data_type=c_data_type, &
                          group=c_group)
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(C_ASSOCIATED(c_local_rows) .AND. C_ASSOCIATED(c_local_cols) .AND. C_ASSOCIATED(c_row_blk_size) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_col_blk_size) .AND. .NOT. C_ASSOCIATED(c_proc_row_dist) .AND. .NOT. C_ASSOCIATED(c_proc_col_dist) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & .NOT. C_ASSOCIATED(c_row_blk_offset) .AND. .NOT. C_ASSOCIATED(c_col_blk_offset)) THEN 

      CALL dbcsr_get_info(matrix, c_nblkrows_total, c_nblkcols_total, &
                          c_nfullrows_total, c_nfullcols_total, &
                          c_nblkrows_local, c_nblkcols_local, &
                          c_nfullrows_local, c_nfullcols_local, &
                          c_my_prow, c_my_pcol &
                          , local_rows = local_rows, local_cols = local_cols, row_blk_size = row_blk_size, col_blk_size =&
# 1349 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
                              & col_blk_size, &
                          matrix_type=c_matrix_type, data_type=c_data_type, &
                          group=c_group)
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(C_ASSOCIATED(c_row_blk_size) .AND. C_ASSOCIATED(c_col_blk_size) .AND. .NOT. C_ASSOCIATED(c_local_rows) .AND. .NOT.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_local_cols) .AND. .NOT. C_ASSOCIATED(c_proc_row_dist) .AND. .NOT. C_ASSOCIATED(c_proc_col_dist) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & .NOT. C_ASSOCIATED(c_row_blk_offset) .AND. .NOT. C_ASSOCIATED(c_col_blk_offset)) THEN 

      CALL dbcsr_get_info(matrix, c_nblkrows_total, c_nblkcols_total, &
                          c_nfullrows_total, c_nfullcols_total, &
                          c_nblkrows_local, c_nblkcols_local, &
                          c_nfullrows_local, c_nfullcols_local, &
                          c_my_prow, c_my_pcol &
                          , row_blk_size = row_blk_size, col_blk_size = col_blk_size, &
                          matrix_type=c_matrix_type, data_type=c_data_type, &
                          group=c_group)
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(C_ASSOCIATED(c_local_rows) .AND. C_ASSOCIATED(c_local_cols) .AND. C_ASSOCIATED(c_proc_row_dist) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_proc_col_dist) .AND. .NOT. C_ASSOCIATED(c_row_blk_size) .AND. .NOT. C_ASSOCIATED(c_col_blk_size) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & .NOT. C_ASSOCIATED(c_row_blk_offset) .AND. .NOT. C_ASSOCIATED(c_col_blk_offset)) THEN 

      CALL dbcsr_get_info(matrix, c_nblkrows_total, c_nblkcols_total, &
                          c_nfullrows_total, c_nfullcols_total, &
                          c_nblkrows_local, c_nblkcols_local, &
                          c_nfullrows_local, c_nfullcols_local, &
                          c_my_prow, c_my_pcol &
                          , local_rows = local_rows, local_cols = local_cols, proc_row_dist = proc_row_dist, proc_col_dist =&
# 1349 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
                              & proc_col_dist, &
                          matrix_type=c_matrix_type, data_type=c_data_type, &
                          group=c_group)
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(C_ASSOCIATED(c_proc_row_dist) .AND. C_ASSOCIATED(c_proc_col_dist) .AND. .NOT. C_ASSOCIATED(c_local_rows) .AND. .NOT.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_local_cols) .AND. .NOT. C_ASSOCIATED(c_row_blk_size) .AND. .NOT. C_ASSOCIATED(c_col_blk_size) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & .NOT. C_ASSOCIATED(c_row_blk_offset) .AND. .NOT. C_ASSOCIATED(c_col_blk_offset)) THEN 

      CALL dbcsr_get_info(matrix, c_nblkrows_total, c_nblkcols_total, &
                          c_nfullrows_total, c_nfullcols_total, &
                          c_nblkrows_local, c_nblkcols_local, &
                          c_nfullrows_local, c_nfullcols_local, &
                          c_my_prow, c_my_pcol &
                          , proc_row_dist = proc_row_dist, proc_col_dist = proc_col_dist, &
                          matrix_type=c_matrix_type, data_type=c_data_type, &
                          group=c_group)
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(C_ASSOCIATED(c_local_rows) .AND. C_ASSOCIATED(c_local_cols) .AND. .NOT. C_ASSOCIATED(c_proc_row_dist) .AND. .NOT.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & C_ASSOCIATED(c_proc_col_dist) .AND. .NOT. C_ASSOCIATED(c_row_blk_size) .AND. .NOT. C_ASSOCIATED(c_col_blk_size) .AND.&
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
          & .NOT. C_ASSOCIATED(c_row_blk_offset) .AND. .NOT. C_ASSOCIATED(c_col_blk_offset)) THEN 

      CALL dbcsr_get_info(matrix, c_nblkrows_total, c_nblkcols_total, &
                          c_nfullrows_total, c_nfullcols_total, &
                          c_nblkrows_local, c_nblkcols_local, &
                          c_nfullrows_local, c_nfullcols_local, &
                          c_my_prow, c_my_pcol &
                          , local_rows = local_rows, local_cols = local_cols, &
                          matrix_type=c_matrix_type, data_type=c_data_type, &
                          group=c_group)
# 1342 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE

      CALL dbcsr_get_info(matrix, c_nblkrows_total, c_nblkcols_total, &
                          c_nfullrows_total, c_nfullcols_total, &
                          c_nblkrows_local, c_nblkcols_local, &
                          c_nfullrows_local, c_nfullcols_local, &
                          c_my_prow, c_my_pcol &
                          , &
                          matrix_type=c_matrix_type, data_type=c_data_type, &
                          group=c_group)
# 1353 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
ENDIF

      ! now take care of name and dist

      IF (PRESENT(c_name)) THEN
         ALLOCATE (CHARACTER(len=default_string_length) :: name)
         CALL dbcsr_get_info(matrix, name=name)
         name = TRIM(name)//char(0)
         c_name = c_loc(name)
      ENDIF

      IF (PRESENT(c_distribution)) THEN
         ALLOCATE (distribution)
         CALL dbcsr_get_info(matrix, distribution=distribution)
         c_distribution = c_loc(distribution)
      ENDIF

# 1372 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 1373 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (C_ASSOCIATED(c_local_rows)) THEN
         CALL c_f_pointer(c_local_rows, f_local_rows, SHAPE(local_rows))
# 1376 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
         f_local_rows = local_rows-1
# 1380 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
         NULLIFY (local_rows)
      ENDIF
# 1373 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (C_ASSOCIATED(c_local_cols)) THEN
         CALL c_f_pointer(c_local_cols, f_local_cols, SHAPE(local_cols))
# 1376 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
         f_local_cols = local_cols-1
# 1380 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
         NULLIFY (local_cols)
      ENDIF
# 1373 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (C_ASSOCIATED(c_proc_row_dist)) THEN
         CALL c_f_pointer(c_proc_row_dist, f_proc_row_dist, SHAPE(proc_row_dist))
# 1378 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
         f_proc_row_dist = proc_row_dist
# 1380 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
         NULLIFY (proc_row_dist)
      ENDIF
# 1373 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (C_ASSOCIATED(c_proc_col_dist)) THEN
         CALL c_f_pointer(c_proc_col_dist, f_proc_col_dist, SHAPE(proc_col_dist))
# 1378 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
         f_proc_col_dist = proc_col_dist
# 1380 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
         NULLIFY (proc_col_dist)
      ENDIF
# 1373 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (C_ASSOCIATED(c_row_blk_size)) THEN
         CALL c_f_pointer(c_row_blk_size, f_row_blk_size, SHAPE(row_blk_size))
# 1378 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
         f_row_blk_size = row_blk_size
# 1380 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
         NULLIFY (row_blk_size)
      ENDIF
# 1373 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (C_ASSOCIATED(c_col_blk_size)) THEN
         CALL c_f_pointer(c_col_blk_size, f_col_blk_size, SHAPE(col_blk_size))
# 1378 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
         f_col_blk_size = col_blk_size
# 1380 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
         NULLIFY (col_blk_size)
      ENDIF
# 1373 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (C_ASSOCIATED(c_row_blk_offset)) THEN
         CALL c_f_pointer(c_row_blk_offset, f_row_blk_offset, SHAPE(row_blk_offset))
# 1376 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
         f_row_blk_offset = row_blk_offset-1
# 1380 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
         NULLIFY (row_blk_offset)
      ENDIF
# 1373 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (C_ASSOCIATED(c_col_blk_offset)) THEN
         CALL c_f_pointer(c_col_blk_offset, f_col_blk_offset, SHAPE(col_blk_offset))
# 1376 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
         f_col_blk_offset = col_blk_offset-1
# 1380 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
         NULLIFY (col_blk_offset)
      ENDIF
# 1383 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   END SUBROUTINE

# 1388 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 1389 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   SUBROUTINE c_dbcsr_get_local_rows (c_matrix, c_local_rows, c_size) BIND(C, name="c_dbcsr_get_local_rows")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_size
      INTEGER(kind=c_int), INTENT(INOUT), DIMENSION(c_size) :: c_local_rows
      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int), DIMENSION(:), POINTER :: local_rows
      INTEGER :: i

      CALL c_f_pointer(c_matrix, matrix)

      NULLIFY (local_rows)
      CALL dbcsr_get_info(matrix=matrix, local_rows=local_rows)

# 1404 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      DO i = 1, c_size
         c_local_rows (i) = local_rows (i) - 1
      ENDDO
# 1412 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (local_rows)

   END SUBROUTINE
# 1389 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   SUBROUTINE c_dbcsr_get_local_cols (c_matrix, c_local_cols, c_size) BIND(C, name="c_dbcsr_get_local_cols")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_size
      INTEGER(kind=c_int), INTENT(INOUT), DIMENSION(c_size) :: c_local_cols
      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int), DIMENSION(:), POINTER :: local_cols
      INTEGER :: i

      CALL c_f_pointer(c_matrix, matrix)

      NULLIFY (local_cols)
      CALL dbcsr_get_info(matrix=matrix, local_cols=local_cols)

# 1404 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      DO i = 1, c_size
         c_local_cols (i) = local_cols (i) - 1
      ENDDO
# 1412 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (local_cols)

   END SUBROUTINE
# 1389 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   SUBROUTINE c_dbcsr_get_proc_row_dist (c_matrix, c_proc_row_dist, c_size) BIND(C, name="c_dbcsr_get_proc_row_dist")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_size
      INTEGER(kind=c_int), INTENT(INOUT), DIMENSION(c_size) :: c_proc_row_dist
      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int), DIMENSION(:), POINTER :: proc_row_dist
      INTEGER :: i

      CALL c_f_pointer(c_matrix, matrix)

      NULLIFY (proc_row_dist)
      CALL dbcsr_get_info(matrix=matrix, proc_row_dist=proc_row_dist)

# 1408 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      DO i = 1, c_size
         c_proc_row_dist (i) = proc_row_dist (i); 
      ENDDO
# 1412 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (proc_row_dist)

   END SUBROUTINE
# 1389 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   SUBROUTINE c_dbcsr_get_proc_col_dist (c_matrix, c_proc_col_dist, c_size) BIND(C, name="c_dbcsr_get_proc_col_dist")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_size
      INTEGER(kind=c_int), INTENT(INOUT), DIMENSION(c_size) :: c_proc_col_dist
      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int), DIMENSION(:), POINTER :: proc_col_dist
      INTEGER :: i

      CALL c_f_pointer(c_matrix, matrix)

      NULLIFY (proc_col_dist)
      CALL dbcsr_get_info(matrix=matrix, proc_col_dist=proc_col_dist)

# 1408 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      DO i = 1, c_size
         c_proc_col_dist (i) = proc_col_dist (i); 
      ENDDO
# 1412 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (proc_col_dist)

   END SUBROUTINE
# 1389 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   SUBROUTINE c_dbcsr_get_row_blk_size (c_matrix, c_row_blk_size, c_size) BIND(C, name="c_dbcsr_get_row_blk_size")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_size
      INTEGER(kind=c_int), INTENT(INOUT), DIMENSION(c_size) :: c_row_blk_size
      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int), DIMENSION(:), POINTER :: row_blk_size
      INTEGER :: i

      CALL c_f_pointer(c_matrix, matrix)

      NULLIFY (row_blk_size)
      CALL dbcsr_get_info(matrix=matrix, row_blk_size=row_blk_size)

# 1408 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      DO i = 1, c_size
         c_row_blk_size (i) = row_blk_size (i); 
      ENDDO
# 1412 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (row_blk_size)

   END SUBROUTINE
# 1389 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   SUBROUTINE c_dbcsr_get_col_blk_size (c_matrix, c_col_blk_size, c_size) BIND(C, name="c_dbcsr_get_col_blk_size")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_size
      INTEGER(kind=c_int), INTENT(INOUT), DIMENSION(c_size) :: c_col_blk_size
      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int), DIMENSION(:), POINTER :: col_blk_size
      INTEGER :: i

      CALL c_f_pointer(c_matrix, matrix)

      NULLIFY (col_blk_size)
      CALL dbcsr_get_info(matrix=matrix, col_blk_size=col_blk_size)

# 1408 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      DO i = 1, c_size
         c_col_blk_size (i) = col_blk_size (i); 
      ENDDO
# 1412 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (col_blk_size)

   END SUBROUTINE
# 1389 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   SUBROUTINE c_dbcsr_get_row_blk_offset (c_matrix, c_row_blk_offset, c_size) BIND(C, name="c_dbcsr_get_row_blk_offset")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_size
      INTEGER(kind=c_int), INTENT(INOUT), DIMENSION(c_size) :: c_row_blk_offset
      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int), DIMENSION(:), POINTER :: row_blk_offset
      INTEGER :: i

      CALL c_f_pointer(c_matrix, matrix)

      NULLIFY (row_blk_offset)
      CALL dbcsr_get_info(matrix=matrix, row_blk_offset=row_blk_offset)

# 1404 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      DO i = 1, c_size
         c_row_blk_offset (i) = row_blk_offset (i) - 1
      ENDDO
# 1412 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (row_blk_offset)

   END SUBROUTINE
# 1389 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
   SUBROUTINE c_dbcsr_get_col_blk_offset (c_matrix, c_col_blk_offset, c_size) BIND(C, name="c_dbcsr_get_col_blk_offset")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_size
      INTEGER(kind=c_int), INTENT(INOUT), DIMENSION(c_size) :: c_col_blk_offset
      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int), DIMENSION(:), POINTER :: col_blk_offset
      INTEGER :: i

      CALL c_f_pointer(c_matrix, matrix)

      NULLIFY (col_blk_offset)
      CALL dbcsr_get_info(matrix=matrix, col_blk_offset=col_blk_offset)

# 1404 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      DO i = 1, c_size
         c_col_blk_offset (i) = col_blk_offset (i) - 1
      ENDDO
# 1412 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (col_blk_offset)

   END SUBROUTINE
# 1416 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   ! name, group
   SUBROUTINE c_dbcsr_get_name(c_matrix, c_name) BIND(C, name="c_dbcsr_get_name")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(c_ptr), INTENT(OUT) :: c_name
      TYPE(dbcsr_type), POINTER :: matrix
      CHARACTER(kind=c_char, len=:), POINTER :: name

      CALL c_f_pointer(c_matrix, matrix)

      NULLIFY (name)
      ALLOCATE (CHARACTER(len=default_string_length) :: name)
      CALL dbcsr_get_info(matrix=matrix, name=name)

      name = TRIM(name)//char(0)
      c_name = c_loc(name)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_group(c_matrix, c_group) BIND(C, name="c_dbcsr_get_group_aux")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(OUT) :: c_group
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_get_info(matrix=matrix, group=c_group)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_distribution(c_matrix, c_dist) &
      BIND(C, name="c_dbcsr_get_distribution")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(c_ptr), INTENT(OUT) :: c_dist
      TYPE(dbcsr_type), POINTER :: matrix
      TYPE(dbcsr_distribution_type), POINTER :: dist

      ALLOCATE (dist)
      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_get_info(matrix=matrix, distribution=dist)
      c_dist = c_loc(dist)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_distribution_get(c_dist, c_row_dist, c_col_dist, &
                                       c_nrows, c_ncols, c_has_threads, &
                                       c_group, c_mynode, c_numnodes, c_nprows, &
                                       c_npcols, c_myprow, c_mypcol, c_pgrid, &
                                       c_subgroups_defined, c_prow_group, c_pcol_group) &
      BIND(C, name="c_dbcsr_distribution_get_aux")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_dist
      TYPE(c_ptr), INTENT(OUT), OPTIONAL :: c_row_dist, c_col_dist
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_nrows, c_ncols
      LOGICAL(kind=c_bool), INTENT(OUT), OPTIONAL :: c_has_threads
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_group, c_mynode, c_numnodes, c_nprows, c_npcols, &
                                                    c_myprow, c_mypcol
      TYPE(c_ptr), INTENT(OUT), OPTIONAL :: c_pgrid
      LOGICAL(kind=c_bool), INTENT(OUT), OPTIONAL :: c_subgroups_defined
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_prow_group, c_pcol_group

      TYPE(dbcsr_distribution_type), POINTER :: dist
      INTEGER, DIMENSION(:), POINTER :: row_dist, col_dist
      LOGICAL, POINTER :: has_threads
      INTEGER, DIMENSION(:, :), POINTER :: pgrid
      LOGICAL, POINTER :: subgroups_defined

      CALL c_f_pointer(c_dist, dist)
# 1489 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 1490 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (has_threads)
      IF (PRESENT(c_has_threads)) THEN
         ALLOCATE (has_threads)
      ENDIF
# 1490 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      NULLIFY (subgroups_defined)
      IF (PRESENT(c_subgroups_defined)) THEN
         ALLOCATE (subgroups_defined)
      ENDIF
# 1495 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

# 1497 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 1498 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      

# 1501 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF(PRESENT(c_row_dist) .AND. PRESENT(c_col_dist) .AND. PRESENT(c_pgrid)) THEN 

      CALL dbcsr_distribution_get(dist=dist, nrows=c_nrows, ncols=c_ncols, &
                                  has_threads=has_threads, &
                                  group=c_group, mynode=c_mynode, numnodes=c_numnodes, &
                                  nprows=c_nprows, npcols=c_npcols, myprow=c_myprow, mypcol=c_mypcol, &
                                  subgroups_defined=subgroups_defined, &
                                  prow_group=c_prow_group, pcol_group=c_pcol_group &
                                  , row_dist = row_dist, col_dist = col_dist, pgrid = pgrid)
# 1501 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(PRESENT(c_col_dist) .AND. PRESENT(c_pgrid) .AND. .NOT. PRESENT(c_row_dist)) THEN 

      CALL dbcsr_distribution_get(dist=dist, nrows=c_nrows, ncols=c_ncols, &
                                  has_threads=has_threads, &
                                  group=c_group, mynode=c_mynode, numnodes=c_numnodes, &
                                  nprows=c_nprows, npcols=c_npcols, myprow=c_myprow, mypcol=c_mypcol, &
                                  subgroups_defined=subgroups_defined, &
                                  prow_group=c_prow_group, pcol_group=c_pcol_group &
                                  , col_dist = col_dist, pgrid = pgrid)
# 1501 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(PRESENT(c_row_dist) .AND. PRESENT(c_pgrid) .AND. .NOT. PRESENT(c_col_dist)) THEN 

      CALL dbcsr_distribution_get(dist=dist, nrows=c_nrows, ncols=c_ncols, &
                                  has_threads=has_threads, &
                                  group=c_group, mynode=c_mynode, numnodes=c_numnodes, &
                                  nprows=c_nprows, npcols=c_npcols, myprow=c_myprow, mypcol=c_mypcol, &
                                  subgroups_defined=subgroups_defined, &
                                  prow_group=c_prow_group, pcol_group=c_pcol_group &
                                  , row_dist = row_dist, pgrid = pgrid)
# 1501 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(PRESENT(c_pgrid) .AND. .NOT. PRESENT(c_row_dist) .AND. .NOT. PRESENT(c_col_dist)) THEN 

      CALL dbcsr_distribution_get(dist=dist, nrows=c_nrows, ncols=c_ncols, &
                                  has_threads=has_threads, &
                                  group=c_group, mynode=c_mynode, numnodes=c_numnodes, &
                                  nprows=c_nprows, npcols=c_npcols, myprow=c_myprow, mypcol=c_mypcol, &
                                  subgroups_defined=subgroups_defined, &
                                  prow_group=c_prow_group, pcol_group=c_pcol_group &
                                  , pgrid = pgrid)
# 1501 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(PRESENT(c_row_dist) .AND. PRESENT(c_col_dist) .AND. .NOT. PRESENT(c_pgrid)) THEN 

      CALL dbcsr_distribution_get(dist=dist, nrows=c_nrows, ncols=c_ncols, &
                                  has_threads=has_threads, &
                                  group=c_group, mynode=c_mynode, numnodes=c_numnodes, &
                                  nprows=c_nprows, npcols=c_npcols, myprow=c_myprow, mypcol=c_mypcol, &
                                  subgroups_defined=subgroups_defined, &
                                  prow_group=c_prow_group, pcol_group=c_pcol_group &
                                  , row_dist = row_dist, col_dist = col_dist)
# 1501 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(PRESENT(c_col_dist) .AND. .NOT. PRESENT(c_row_dist) .AND. .NOT. PRESENT(c_pgrid)) THEN 

      CALL dbcsr_distribution_get(dist=dist, nrows=c_nrows, ncols=c_ncols, &
                                  has_threads=has_threads, &
                                  group=c_group, mynode=c_mynode, numnodes=c_numnodes, &
                                  nprows=c_nprows, npcols=c_npcols, myprow=c_myprow, mypcol=c_mypcol, &
                                  subgroups_defined=subgroups_defined, &
                                  prow_group=c_prow_group, pcol_group=c_pcol_group &
                                  , col_dist = col_dist)
# 1501 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE IF(PRESENT(c_row_dist) .AND. .NOT. PRESENT(c_col_dist) .AND. .NOT. PRESENT(c_pgrid)) THEN 

      CALL dbcsr_distribution_get(dist=dist, nrows=c_nrows, ncols=c_ncols, &
                                  has_threads=has_threads, &
                                  group=c_group, mynode=c_mynode, numnodes=c_numnodes, &
                                  nprows=c_nprows, npcols=c_npcols, myprow=c_myprow, mypcol=c_mypcol, &
                                  subgroups_defined=subgroups_defined, &
                                  prow_group=c_prow_group, pcol_group=c_pcol_group &
                                  , row_dist = row_dist)
# 1501 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      ELSE

      CALL dbcsr_distribution_get(dist=dist, nrows=c_nrows, ncols=c_ncols, &
                                  has_threads=has_threads, &
                                  group=c_group, mynode=c_mynode, numnodes=c_numnodes, &
                                  nprows=c_nprows, npcols=c_npcols, myprow=c_myprow, mypcol=c_mypcol, &
                                  subgroups_defined=subgroups_defined, &
                                  prow_group=c_prow_group, pcol_group=c_pcol_group &
                                  )
# 1511 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
ENDIF

# 1514 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_has_threads)) THEN
         c_has_threads = has_threads
         DEALLOCATE (has_threads)
      ENDIF
# 1514 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_subgroups_defined)) THEN
         c_subgroups_defined = subgroups_defined
         DEALLOCATE (subgroups_defined)
      ENDIF
# 1519 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 1520 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
# 1521 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_row_dist)) THEN
         c_row_dist = c_loc(row_dist)
      ENDIF
# 1521 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_col_dist)) THEN
         c_col_dist = c_loc(col_dist)
      ENDIF
# 1521 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"
      IF (PRESENT(c_pgrid)) THEN
         c_pgrid = c_loc(pgrid)
      ENDIF
# 1525 "/__w/dbcsr/dbcsr/src/dbcsr_api_c.F"

   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_stored_coordinates(c_matrix, row, col, processor) &
      bind(C, name="c_dbcsr_get_stored_coordinates")

      TYPE(c_ptr), INTENT(in), value                     :: c_matrix
      INTEGER(kind=c_int), INTENT(in), value             :: row, col
      INTEGER(kind=c_int), INTENT(out)                   :: processor

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_get_stored_coordinates(matrix, row + 1, col + 1, processor)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_setname(c_matrix, c_newname) &
      BIND(C, name="c_dbcsr_setname")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix, c_newname
      TYPE(dbcsr_type), POINTER :: matrix
      CHARACTER(:, kind=c_char), ALLOCATABLE  :: newname

      CALL c_f_pointer(c_matrix, matrix)
      CALL c_f_string(c_newname, newname)
      CALL dbcsr_setname(matrix, newname)
   END SUBROUTINE

   FUNCTION c_dbcsr_get_matrix_type(c_matrix) RESULT(c_matrix_type) &
      BIND(C, name="c_dbcsr_get_matrix_type")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix
      CHARACTER(kind=c_char) :: c_matrix_type

      CALL c_f_pointer(c_matrix, matrix)
      c_matrix_type = dbcsr_get_matrix_type(matrix)
   END FUNCTION

   FUNCTION c_dbcsr_get_occupation(c_matrix) RESULT(c_occupation) &
      BIND(C, name="c_dbcsr_get_occupation")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix
      REAL(KIND=c_double) :: c_occupation

      CALL c_f_pointer(c_matrix, matrix)
      c_occupation = dbcsr_get_occupation(matrix)
   END FUNCTION

   FUNCTION c_dbcsr_get_num_blocks(c_matrix) RESULT(c_num_blocks) &
      BIND(C, name="c_dbcsr_get_num_blocks")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int) :: c_num_blocks

      CALL c_f_pointer(c_matrix, matrix)
      c_num_blocks = dbcsr_get_num_blocks(matrix)

   END FUNCTION

   FUNCTION c_dbcsr_get_data_size(c_matrix) RESULT(c_data_size) &
      BIND(C, name="c_dbcsr_get_data_size")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int) :: c_data_size

      CALL c_f_pointer(c_matrix, matrix)
      c_data_size = dbcsr_get_data_size(matrix)

   END FUNCTION

   FUNCTION c_dbcsr_has_symmetry(c_matrix) RESULT(c_has_symmetry) &
      BIND(C, name="c_dbcsr_has_symmetry")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix
      LOGICAL(kind=c_bool) :: c_has_symmetry

      CALL c_f_pointer(c_matrix, matrix)
      c_has_symmetry = dbcsr_has_symmetry(matrix)

   END FUNCTION

   FUNCTION c_dbcsr_nfullrows_total(c_matrix) RESULT(c_nfullrows_total) &
      BIND(C, name="c_dbcsr_nfullrows_total")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int) :: c_nfullrows_total

      CALL c_f_pointer(c_matrix, matrix)
      c_nfullrows_total = dbcsr_nfullrows_total(matrix)
   END FUNCTION

   FUNCTION c_dbcsr_nfullcols_total(c_matrix) RESULT(c_nfullcols_total) &
      BIND(C, name="c_dbcsr_nfullcols_total")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int) :: c_nfullcols_total

      CALL c_f_pointer(c_matrix, matrix)
      c_nfullcols_total = dbcsr_nfullcols_total(matrix)
   END FUNCTION

   FUNCTION c_dbcsr_valid_index(c_matrix) RESULT(c_valid_index) &
      BIND(C, name="c_dbcsr_valid_index")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix
      LOGICAL(kind=c_bool) :: c_valid_index

      CALL c_f_pointer(c_matrix, matrix)
      c_valid_index = dbcsr_valid_index(matrix)
   END FUNCTION

   FUNCTION c_dbcsr_get_data_type(c_matrix) RESULT(c_data_type) &
      BIND(C, name="c_dbcsr_get_data_type")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix
      INTEGER(kind=c_int) :: c_data_type

      CALL c_f_pointer(c_matrix, matrix)
      c_data_type = dbcsr_get_data_type(matrix)
   END FUNCTION

   ! ---------------------------------------- !
   !              other                       !
   ! ---------------------------------------- !

   SUBROUTINE c_dbcsr_binary_write(c_matrix, c_filepath) &
      BIND(C, name="c_dbcsr_binary_write")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix, c_filepath
      TYPE(dbcsr_type), POINTER :: matrix
      CHARACTER(:, kind=c_char), ALLOCATABLE :: filepath

      CALL c_f_pointer(c_matrix, matrix)
      CALL c_f_string(c_filepath, filepath)

      CALL dbcsr_binary_write(matrix, filepath)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_binary_read(c_filepath, c_distribution, c_groupid, c_matrix_new) &
      BIND(C, name="c_dbcsr_binary_read_aux")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_filepath, c_distribution
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_groupid
      TYPE(c_ptr), INTENT(INOUT) :: c_matrix_new

      CHARACTER(:, kind=c_char), ALLOCATABLE :: filepath
      TYPE(dbcsr_distribution_type), POINTER :: distribution
      TYPE(dbcsr_type), POINTER :: matrix_new

      CALL c_f_string(c_filepath, filepath)
      CALL c_f_pointer(c_distribution, distribution)
      ALLOCATE (matrix_new)

      CALL dbcsr_binary_read(filepath, distribution, c_groupid, matrix_new)
      c_matrix_new = c_loc(matrix_new)
   END SUBROUTINE

   SUBROUTINE c_free_string(c_string) BIND(C, name="c_free_string")
      TYPE(c_ptr), INTENT(INOUT)             :: c_string

      CHARACTER(:, kind=c_char), POINTER :: string

      CALL c_f_pointer(c_string, string)
      DEALLOCATE (string)
      c_string = c_null_ptr

   END SUBROUTINE c_free_string

END MODULE
