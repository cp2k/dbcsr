module dbcsr_api_c

#if defined(__C_INTERFACE)

   use, intrinsic :: ISO_C_BINDING, only : c_loc, c_ptr, c_double, C_NULL_CHAR, &
                                           c_f_pointer, c_int, c_char, c_null_ptr, c_bool
   USE dbcsr_api
   USE dbcsr_machine,                         ONLY: default_output_unit
   
   implicit none
   private
   
   contains

      
      subroutine c_f_string(c_str, str) 
        use, intrinsic :: iso_c_binding, only: c_ptr, c_f_pointer, c_char
        type(c_ptr), intent(in) :: c_str
        character(kind=c_char), pointer :: arr(:)
        character(:,kind=c_char), allocatable, intent(out) :: str
        integer(8) :: n,i
        interface
          ! steal std c library function rather than writing our own.
          function strlen(s) bind(c, name='strlen')
            use, intrinsic :: iso_c_binding, only: c_ptr, c_size_t
            implicit none
            !----
            type(c_ptr), intent(in), value :: s
            integer(c_size_t) :: strlen
          end function strlen
        end interface
        n = strlen(c_str)
        !****
        call c_f_pointer(c_str, arr, [n])
        allocate(character(len=n) :: str)
        do i=1,n 
          str( i : i ) = arr( i )
        enddo
      end subroutine c_f_string
      

      subroutine c_dbcsr_init_lib() bind(C, name="c_dbcsr_init_lib")
        call dbcsr_init_lib()
      end subroutine

      
      subroutine c_dbcsr_finalise_lib(fcomm) bind(C, name="c_dbcsr_finalize_lib_aux")
        integer(kind=c_int), intent(in) :: fcomm
        integer :: io_stream
        integer:: ierr, rank
        call mpi_comm_rank(fcomm,rank,ierr)
        io_stream = 0
        if (rank .eq. 0) then
            io_stream = default_output_unit
        endif
        call dbcsr_finalize_lib(fcomm, io_stream)
      end subroutine

      
      subroutine c_dbcsr_distribution_new(c_dist, fcomm, c_row_dist, row_dist_size, &
                                    c_col_dist, col_dist_size) &
                                    bind(C, name="c_dbcsr_distribution_new_aux")
        type(c_ptr), intent(out)                :: c_dist
        integer(kind=c_int), intent(in)                     :: fcomm
        integer(kind=c_int), intent(in), value              :: row_dist_size, col_dist_size
        integer(kind=c_int), intent(in), target             :: c_row_dist(row_dist_size), c_col_dist(col_dist_size)
        type(dbcsr_distribution_type), pointer  :: dist
        integer, pointer                        :: row_dist(:), col_dist(:)
        
        integer:: ierr, rank
        call mpi_comm_rank(fcomm,rank,ierr)
        row_dist => c_row_dist
        col_dist => c_col_dist
        allocate(dist)
        call dbcsr_distribution_new(dist, group=fcomm, row_dist=row_dist, &
                                    col_dist=col_dist, reuse_arrays=.false.)
        c_dist = c_loc(dist)
      end subroutine


      subroutine c_dbcsr_distribution_release(c_dist) bind(C, name="c_dbcsr_distribution_release")
        type(c_ptr), intent(inout)      :: c_dist
        type(dbcsr_distribution_type),  pointer          :: dist
        call c_f_pointer(c_dist, dist) 

        call dbcsr_distribution_release(dist)

        deallocate(dist)

        c_dist = c_null_ptr
      end subroutine


      
      subroutine c_dbcsr_create_new_d(c_matrix, c_name, c_dist, c_matrix_type, &
                                      c_row_blk_sizes, c_row_blk_sizes_length, &
                                      c_col_blk_sizes, c_col_blk_sizes_length) &
                                      bind(C, name="c_dbcsr_create_new_d")
        type(c_ptr), intent(out)            :: c_matrix
        type(c_ptr), value, intent(in)                   :: c_name
        type(c_ptr), value, intent(in)                   :: c_dist  
        character(kind=c_char), value, intent(in)        :: c_matrix_type
        integer(kind=c_int), intent(in), value           :: c_row_blk_sizes_length, c_col_blk_sizes_length
        integer(kind=c_int), intent(in), target          :: c_row_blk_sizes(c_row_blk_sizes_length)
        integer(kind=c_int), intent(in), target          :: c_col_blk_sizes(c_col_blk_sizes_length) 
        
        type(dbcsr_type),  pointer              :: matrix
        type(dbcsr_distribution_type), pointer  :: dist
        integer, pointer                        :: row_blk_sizes(:), col_blk_sizes(:) 
        !character(len=*),pointer                :: fname

        character(:,kind=c_char), allocatable :: fname
       
        ! associate pointers
        row_blk_sizes => c_row_blk_sizes
        col_blk_sizes => c_col_blk_sizes

        ! convert normal distribution pointer to fortran pointer
        call c_f_pointer(c_dist, dist)

        ! allocate heap memory for matrix structure
        allocate(matrix)

        ! convert normal name to fortran name
        call c_f_string(c_name, fname)
 
        call dbcsr_create(matrix = matrix, name = fname, dist = dist, &
                          matrix_type = c_matrix_type, row_blk_size = row_blk_sizes, &
                          col_blk_size = col_blk_sizes,  data_type=dbcsr_type_real_8)

        c_matrix = c_loc(matrix)
      end subroutine


      subroutine c_dbcsr_finalize(c_matrix) bind(C, name="c_dbcsr_finalize")
        type(c_ptr), intent(in), value   :: c_matrix
        type(dbcsr_type),  pointer          :: matrix

        call c_f_pointer(c_matrix, matrix) 

        call dbcsr_finalize(matrix)
      end subroutine



      subroutine c_dbcsr_release(c_matrix) bind(C, name="c_dbcsr_release")
        type(c_ptr), intent(inout)      :: c_matrix
        type(dbcsr_type),  pointer          :: matrix

        call c_f_pointer(c_matrix, matrix) 

        call dbcsr_release(matrix)

        deallocate(matrix)

        c_matrix = c_null_ptr
      end subroutine


      subroutine c_dbcsr_print(c_matrix) bind(C, name="c_dbcsr_print") 
        type(c_ptr), intent(in), value   :: c_matrix
        type(dbcsr_type),  pointer          :: matrix

        call c_f_pointer(c_matrix, matrix) 

        call dbcsr_print(matrix)
      end subroutine


      subroutine c_dbcsr_get_stored_coordinates(c_matrix, row, col, processor) bind(C, name="c_dbcsr_get_stored_coordinates")
        type(c_ptr), intent(in), value  :: c_matrix 
        integer(kind=c_int), intent(in), value      :: row, col
        integer(kind=c_int), intent(out)            :: processor

        type(dbcsr_type),  pointer      :: matrix

        call c_f_pointer(c_matrix, matrix)

        call dbcsr_get_stored_coordinates(matrix, row + 1, col + 1, processor)
      end subroutine


      subroutine c_dbcsr_put_block_d(c_matrix, row, col, block, block_length) bind(C, name="c_dbcsr_put_block_d")
        type(c_ptr), intent(in), value           :: c_matrix        
        integer(kind=c_int), intent(in), value               :: row, col, block_length      
        real(kind=c_double), intent(in)                      :: block(block_length)
        
        type(dbcsr_type), pointer                :: matrix 

        call c_f_pointer(c_matrix, matrix)

        call dbcsr_put_block(matrix, row + 1, col + 1, block)
      end subroutine


      subroutine c_dbcsr_multiply_d(transa, transb, alpha, c_matrix_a, c_matrix_b, beta, c_matrix_c, retain_sparsity) &
                    bind(C, name="c_dbcsr_multiply_d")
        character(kind=c_char), intent(in), value          :: transa, transb
        real(kind=c_double), intent(in), value            :: alpha
        type(c_ptr), intent(in)               :: c_matrix_a, c_matrix_b
        real(kind=c_double), intent(in), value            :: beta
        type(c_ptr), intent(inout)            :: c_matrix_c
        logical(c_bool), intent(in), optional         :: retain_sparsity
        logical :: ret_sp
        type(dbcsr_type),  pointer      :: matrix_a, matrix_b, matrix_c
        
        call c_f_pointer(c_matrix_a, matrix_a)
        call c_f_pointer(c_matrix_b, matrix_b)
        call c_f_pointer(c_matrix_c, matrix_c)

        if (present(retain_sparsity) ) then
            ret_sp = retain_sparsity 
            call dbcsr_multiply(transa, transb, alpha, matrix_a, matrix_b, beta, matrix_c, retain_sparsity = ret_sp)
        else
            call dbcsr_multiply(transa, transb, alpha, matrix_a, matrix_b, beta, matrix_c)
        endif
      end subroutine

#endif
end module
