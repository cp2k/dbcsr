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

      CALL dbcsr_finalize_lib(fcomm)
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

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! JKN: additional routines for dp
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! **************************************************************************************************
!> \brief ...
!> \param c_matrix_a ...
!> \param c_matrix_b ...
!> \param C ...
!> \param name="c_dbcsr_copy_d" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_copy_d(c_matrix_a, c_matrix_b) &
      bind(C, name="c_dbcsr_copy_d")
      TYPE(c_ptr), INTENT(in)                            :: c_matrix_b
      TYPE(c_ptr), INTENT(inout)                         :: c_matrix_a

      LOGICAL                                            :: fa
      TYPE(dbcsr_type), POINTER                          :: matrix_a, matrix_b
      CHARACTER                                          :: c_matrix_type

      c_matrix_type = 'N'
      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      fa = .FALSE.
      CALL dbcsr_copy(matrix_a, matrix_b,"matrix-copy", fa,fa,fa,c_matrix_type)
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_matrix_a ...
!> \param c_matrix_b ...
!> \param faca ...
!> \param facb ...
!> \param C ...
!> \param name="c_dbcsr_add_d" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_add_d(c_matrix_a, c_matrix_b, faca, facb) &
      bind(C, name="c_dbcsr_add_d")
      TYPE(c_ptr), INTENT(in)                            :: c_matrix_b
      TYPE(c_ptr), INTENT(inout)                         :: c_matrix_a
      REAL(kind=c_double), INTENT(in), value             :: faca,facb

      TYPE(dbcsr_type), POINTER                          :: matrix_a, matrix_b

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      CALL dbcsr_add(matrix_a, matrix_b, faca, facb)

   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param fcomm ...
!> \param C ...
!> \param name="c_dbcsr_finalize_lib_silent" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_finalise_lib_silent(fcomm) bind(C, name="c_dbcsr_finalize_lib_aux_silent")
      INTEGER(kind=c_int), INTENT(in)                    :: fcomm

      CALL dbcsr_finalize_lib(fcomm,-1)
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_matrix_a ...
!> \param c_matrix_b ...
!> \param retval ...
!> \param C ...
!> \param name="c_dbcsr_trace_ab_d" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_trace_ab_d(c_matrix_a, c_matrix_b, retval) &
      bind(C, name="c_dbcsr_trace_ab_d")
      TYPE(c_ptr), INTENT(in)                            :: c_matrix_b
      TYPE(c_ptr), INTENT(in)                            :: c_matrix_a

      TYPE(dbcsr_type), POINTER                          :: matrix_a, matrix_b
      !REAL(kind=c_double), INTENT(inout), value            :: retval
      REAL(kind=c_double), INTENT(inout)                   :: retval

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      retval = 0.0
      call dbcsr_trace(matrix_a, matrix_b, retval)
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_matrix_a ...
!> \param retval ...
!> \param C ...
!> \param name="c_dbcsr_trace_a_d" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_trace_a_d(c_matrix_a, retval) &
      bind(C, name="c_dbcsr_trace_a_d")
      TYPE(c_ptr), INTENT(in)                            :: c_matrix_a

      TYPE(dbcsr_type), POINTER                          :: matrix_a
      !REAL(kind=c_double), INTENT(inout), value            :: retval
      REAL(kind=c_double), INTENT(inout)                   :: retval

      CALL c_f_pointer(c_matrix_a, matrix_a)
      retval = 0.0
      call dbcsr_trace(matrix_a, retval)
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_matrix_a ...
!> \param diags ...
!> \param tot_dim ...
!> \param C ...
!> \param name="c_dbcsr_set_diag_d" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_set_diag_d(c_matrix_a, diags, tot_dim) &
      bind(C, name="c_dbcsr_set_diag_d")
      TYPE(c_ptr), INTENT(inout)                         :: c_matrix_a
      INTEGER(kind=c_int), INTENT(in), value             :: tot_dim

      TYPE(dbcsr_type), POINTER                          :: matrix_a
      REAL(kind=c_double), INTENT(IN)                    :: diags(tot_dim)

      CALL c_f_pointer(c_matrix_a, matrix_a)
      call dbcsr_set_diag(matrix_a, diags)
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_matrix_a ...
!> \param scl ...
!> \param C ...
!> \param name="c_dbcsr_set_d" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_set_d(c_matrix_a, scl) &
      bind(C, name="c_dbcsr_set_d")
      TYPE(c_ptr), INTENT(inout)                         :: c_matrix_a

      TYPE(dbcsr_type), POINTER                          :: matrix_a
      REAL(kind=c_double),  INTENT(in), value            :: scl

      CALL c_f_pointer(c_matrix_a, matrix_a)
      call dbcsr_set(matrix_a, scl)
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_matrix_a ...
!> \param eps ...
!> \param C ...
!> \param name="c_dbcsr_filter_d" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_filter_d(c_matrix_a, eps) &
      bind(C, name="c_dbcsr_filter_d")

      TYPE(c_ptr), INTENT(inout)                         :: c_matrix_a
      TYPE(dbcsr_type), POINTER                          :: matrix_a
      REAL(kind=c_double),  INTENT(in), value                   :: eps

      CALL c_f_pointer(c_matrix_a, matrix_a)

      call dbcsr_filter(matrix_a, eps)

   END SUBROUTINE
   
! **************************************************************************************************
!> \brief ...
!> \param c_matrix_a ...
!> \param row ...
!> \param col ...
!> \param data_block ...
!> \param found_it ...
!> \param row_size ...
!> \param col_size ...
!> \param C ...
!> \param name="c_dbcsr_get_block_p_d" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_get_block_d(c_matrix_a, row, col, data_block, found_it, row_size, col_size) bind(C, name="c_dbcsr_get_block_d")

      INTEGER(c_int), INTENT(in), value                        :: row, col, row_size, col_size
      TYPE(c_ptr), INTENT(in)                                  :: c_matrix_a
      REAL(kind=c_double), INTENT(INOUT)                       :: data_block(row_size*col_size)
      LOGICAL(kind=c_bool), INTENT(out)                        :: found_it

      TYPE(dbcsr_type), POINTER                                :: matrix_a
      INTEGER                                                  :: rowp1, colp1
      LOGICAL                                                  :: found_loc
      REAL(kind=8), DIMENSION(:), POINTER :: data_ptr
      INTEGER :: iii

      NULLIFY (data_ptr)
      found_loc = .FALSE.
      CALL c_f_pointer(c_matrix_a, matrix_a)
      rowp1 = row+1
      colp1 = col+1
      CALL dbcsr_get_block_p(matrix_a, rowp1, colp1, data_ptr, found_loc)
      if (found_loc) then
        do iii=1,row_size*col_size 
          data_block(iii) = data_ptr(iii)
        enddo
      endif
      found_it = found_loc
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param trans_a ...
!> \param trans_b ...
!> \param alpha ...
!> \param c_matrix_a ...
!> \param c_matrix_b ...
!> \param beta ...
!> \param c_matrix_c ...
!> \param sthr ...
!> \param C ...
!> \param name="c_dbcsr_multiply_eps_d" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_multiply_eps_d(transa, transb, alpha, c_matrix_a, c_matrix_b, beta, c_matrix_c, sthr) &
      bind(C, name="c_dbcsr_multiply_eps_d")
      CHARACTER(kind=c_char), INTENT(in), value          :: transa, transb
      REAL(kind=c_double), INTENT(in), value             :: alpha
      TYPE(c_ptr), INTENT(in)                            :: c_matrix_a, c_matrix_b
      REAL(kind=c_double), INTENT(in), value             :: beta
      TYPE(c_ptr), INTENT(inout)                         :: c_matrix_c
      REAL(kind=c_double), INTENT(in), value             :: sthr

      LOGICAL                                            :: ret_sp
      TYPE(dbcsr_type), POINTER                          :: matrix_a, matrix_b, matrix_c

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      CALL c_f_pointer(c_matrix_c, matrix_c)

      ret_sp = .false.
      CALL dbcsr_multiply(transa, transb, alpha, matrix_a, matrix_b, beta, matrix_c, retain_sparsity=ret_sp, filter_eps=sthr)
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_matrix_a ...
!> \param bdims ...
!> \param nblocks ...
!> \param tot_dim ...
!> \param sums ...
!> \param diags ...
!> \param C ...
!> \param name="c_dbcsr_gershgorin_estimate_d" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_gershgorin_estimate_d(c_matrix_a, bdims, nblocks, tot_dim, sums, diags) &
      bind(C, name="c_dbcsr_gershgorin_estimate_d")
      TYPE(c_ptr), INTENT(in)                            :: c_matrix_a 
      INTEGER(kind=c_int), INTENT(IN), value             :: nblocks, tot_dim
      INTEGER(kind=c_int), INTENT(IN)                    :: bdims(nblocks)
      REAL(kind=c_double), INTENT(INOUT)                 :: sums(tot_dim), diags(tot_dim)

      TYPE(dbcsr_type), POINTER                          :: matrix_a
      REAL(kind=8), DIMENSION(:,:), POINTER              :: data_ptr
      INTEGER                                            :: irow, icol, ii, jj, row_dim, col_dim, orow
      LOGICAL                                            :: found_loc
      !REAL(kind=8), ALLOCATABLE, DIMENSION(:)            :: diags_loc, sums_loc
      !REAL(kind=8)                                       :: diags_loc(tot_dim), sums_loc(tot_dim)

      CALL c_f_pointer(c_matrix_a, matrix_a)

      diags = 0.0
      sums = 0.0

      CALL dbcsr_get_diag(matrix_a, diags)

      orow = 0
      do irow=1,nblocks
        row_dim = bdims(irow)
        do icol=1,nblocks
          NULLIFY (data_ptr)
          found_loc = .FALSE.
          CALL dbcsr_get_block_p(matrix_a, irow, icol, data_ptr, found_loc)
          if (found_loc) then
            col_dim = bdims(icol)
            do jj=1,col_dim
              do ii=1,row_dim
                sums(orow+ii) = sums(orow+ii) + abs(data_ptr(ii,jj))
              enddo
            enddo
          endif
        enddo
        orow = orow + row_dim
      enddo

   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_matrix_a ...
!> \param scl ...
!> \param C ...
!> \param name="c_dbcsr_gershgorin_estimate_d" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_scale_d(c_matrix_a, scl) &
      bind(C, name="c_dbcsr_scale_d")
      TYPE(c_ptr), INTENT(in)                            :: c_matrix_a 
      REAL(kind=c_double), INTENT(IN), value             :: scl

      TYPE(dbcsr_type), POINTER                          :: matrix_a

      CALL c_f_pointer(c_matrix_a, matrix_a)
      call dbcsr_scale(matrix_a,scl)
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_matrix_a ...
!> \param cfname ...
!> \param c_dist ...
!> \param C ...
!> \param name="c_dbcsr_read_d" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_read_d(c_matrix_a, cfname, c_dist) &
      bind(C, name="c_dbcsr_read_d")
      TYPE(c_ptr), INTENT(inout)                         :: c_matrix_a 
      TYPE(c_ptr), INTENT(in), value                     :: cfname
      TYPE(c_ptr), INTENT(in)                            :: c_dist

      TYPE(dbcsr_type), POINTER                          :: matrix_a
      TYPE(dbcsr_distribution_type), POINTER             :: dist
      CHARACTER(:, kind=c_char), ALLOCATABLE             :: ffname

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_dist, dist)
      CALL c_f_string(cfname, ffname)

      CALL dbcsr_binary_read(filepath=ffname, distribution=dist, matrix_new=matrix_a)

      deallocate(ffname)
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_matrix_a ...
!> \param cfname ...
!> \param C ...
!> \param name="c_dbcsr_write_d" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_write_d(c_matrix_a, cfname) &
      bind(C, name="c_dbcsr_write_d")
      TYPE(c_ptr), INTENT(in)                            :: c_matrix_a 
      TYPE(c_ptr), INTENT(in), value                     :: cfname

      TYPE(dbcsr_type), POINTER                          :: matrix_a
      CHARACTER(:, kind=c_char), ALLOCATABLE             :: ffname

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_string(cfname, ffname)

      CALL dbcsr_binary_write(matrix_a, ffname)

      deallocate(ffname)
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param c_matrix_a ...
!> \param amv ...
!> \param C ...
!> \param name="c_dbcsr_maxabs_d" ...
! **************************************************************************************************
   SUBROUTINE c_dbcsr_maxabs_d(c_matrix_a, amv) &
      bind(C, name="c_dbcsr_maxabs_d")
      TYPE(c_ptr), INTENT(in)                            :: c_matrix_a 
      REAL(kind=c_double), INTENT(out)                   :: amv

      TYPE(dbcsr_type), POINTER                          :: matrix_a

      CALL c_f_pointer(c_matrix_a, matrix_a)
      amv = dbcsr_maxabs(matrix_a)
   END SUBROUTINE

END MODULE
