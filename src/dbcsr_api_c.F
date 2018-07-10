MODULE dbcsr_api_c

   USE, INTRINSIC :: ISO_C_BINDING, ONLY: c_loc, c_ptr, c_double, C_NULL_CHAR, &
      c_f_pointer, c_int, c_char, c_null_ptr, c_bool
   USE dbcsr_api
   USE dbcsr_machine, ONLY: default_output_unit

   IMPLICIT NONE
   PRIVATE

CONTAINS

! **************************************************************************************************
!> \brief ...
!> \param c_str ...
!> \param str ...
! **************************************************************************************************
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

! **************************************************************************************************
!> \brief ...
!> \param C ...
!> \param name="c_dbcsr_init_lib" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_init_lib() bind(C, name="c_dbcsr_init_lib")
      CALL dbcsr_init_lib()
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param fcomm ...
!> \param C ...
!> \param name="c_dbcsr_finalize_lib_aux" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_finalise_lib(fcomm) bind(C, name="c_dbcsr_finalize_lib_aux")
      INTEGER(kind=c_int), INTENT(in)                    :: fcomm

      INTEGER                                            :: io_stream

      INTEGER:: ierr, rank
      CALL mpi_comm_rank(fcomm, rank, ierr)
      io_stream = 0
      IF (rank .EQ. 0) THEN
         io_stream = default_output_unit
      ENDIF
      CALL dbcsr_finalize_lib(fcomm, io_stream)
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_dist ...
!> \param fcomm ...
!> \param c_row_dist ...
!> \param row_dist_size ...
!> \param c_col_dist ...
!> \param col_dist_size ...
!> \param C ...
!> \param name="c_dbcsr_distribution_new_aux" ...
! **************************************************************************************************
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

      INTEGER:: ierr, rank
      CALL mpi_comm_rank(fcomm, rank, ierr)
      row_dist => c_row_dist
      col_dist => c_col_dist
      ALLOCATE (dist)
      CALL dbcsr_distribution_new(dist, group=fcomm, row_dist=row_dist, &
                                  col_dist=col_dist, reuse_arrays=.FALSE.)
      c_dist = c_loc(dist)
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_dist ...
!> \param C ...
!> \param name="c_dbcsr_distribution_release" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_distribution_release(c_dist) bind(C, name="c_dbcsr_distribution_release")
      TYPE(c_ptr), INTENT(inout)                         :: c_dist

      TYPE(dbcsr_distribution_type), POINTER             :: dist

      CALL c_f_pointer(c_dist, dist)

      CALL dbcsr_distribution_release(dist)

      DEALLOCATE (dist)

      c_dist = c_null_ptr
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_matrix ...
!> \param c_name ...
!> \param c_dist ...
!> \param c_matrix_type ...
!> \param c_row_blk_sizes ...
!> \param c_row_blk_sizes_length ...
!> \param c_col_blk_sizes ...
!> \param c_col_blk_sizes_length ...
!> \param C ...
!> \param name="c_dbcsr_create_new_d" ...
! **************************************************************************************************
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

!character(len=*),pointer                :: fname

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

! **************************************************************************************************
!> \brief ...
!> \param c_matrix ...
!> \param C ...
!> \param name="c_dbcsr_finalize" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_finalize(c_matrix) bind(C, name="c_dbcsr_finalize")
      TYPE(c_ptr), INTENT(in), value                     :: c_matrix

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_finalize(matrix)
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_matrix ...
!> \param C ...
!> \param name="c_dbcsr_release" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_release(c_matrix) bind(C, name="c_dbcsr_release")
      TYPE(c_ptr), INTENT(inout)                         :: c_matrix

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_release(matrix)

      DEALLOCATE (matrix)

      c_matrix = c_null_ptr
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_matrix ...
!> \param C ...
!> \param name="c_dbcsr_print" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_print(c_matrix) bind(C, name="c_dbcsr_print")
      TYPE(c_ptr), INTENT(in), value                     :: c_matrix

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_print(matrix)

      ! Fortran and C may use different buffers for I/O, make sure we flush before returning:
      flush(default_output_unit)
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_matrix ...
!> \param row ...
!> \param col ...
!> \param processor ...
!> \param C ...
!> \param name="c_dbcsr_get_stored_coordinates" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_get_stored_coordinates(c_matrix, row, col, processor) bind(C, name="c_dbcsr_get_stored_coordinates")
      TYPE(c_ptr), INTENT(in), value                     :: c_matrix
      INTEGER(kind=c_int), INTENT(in), value             :: row, col
      INTEGER(kind=c_int), INTENT(out)                   :: processor

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_get_stored_coordinates(matrix, row+1, col+1, processor)
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_matrix ...
!> \param row ...
!> \param col ...
!> \param block ...
!> \param block_length ...
!> \param C ...
!> \param name="c_dbcsr_put_block_d" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_put_block_d(c_matrix, row, col, block, block_length) bind(C, name="c_dbcsr_put_block_d")
      TYPE(c_ptr), INTENT(in), value                     :: c_matrix
      INTEGER(kind=c_int), INTENT(in), value             :: row, col, block_length
      REAL(kind=c_double), INTENT(in)                    :: block(block_length)

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_put_block(matrix, row+1, col+1, block)
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param transa ...
!> \param transb ...
!> \param alpha ...
!> \param c_matrix_a ...
!> \param c_matrix_b ...
!> \param beta ...
!> \param c_matrix_c ...
!> \param retain_sparsity ...
!> \param C ...
!> \param name="c_dbcsr_multiply_d" ...
! **************************************************************************************************
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
