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
   
#:include 'data/dbcsr.fypp'

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

   ! getters / setters
   PUBLIC :: dbcsr_get_info
   PUBLIC :: dbcsr_distribution_get
   PUBLIC :: dbcsr_setname
   PUBLIC :: dbcsr_get_matrix_type
   PUBLIC :: dbcsr_get_occupation
   PUBLIC :: dbcsr_nblkrows_total
   PUBLIC :: dbcsr_nblkcols_total
   PUBLIC :: dbcsr_get_num_blocks
   PUBLIC :: dbcsr_get_data_size
   PUBLIC :: dbcsr_has_symmetry
   PUBLIC :: dbcsr_nfullrows_total
   PUBLIC :: dbcsr_nfullcols_total
   PUBLIC :: dbcsr_get_stored_coordinates
   PUBLIC :: dbcsr_valid_index
   PUBLIC :: dbcsr_get_data_type

   ! work operations
   PUBLIC :: dbcsr_add_block_node
   PUBLIC :: dbcsr_put_block
   PUBLIC :: dbcsr_work_create
   PUBLIC :: dbcsr_verify_matrix
   PUBLIC :: dbcsr_add_work_coordinate
   PUBLIC :: dbcsr_get_wms_data_p
   PUBLIC :: dbcsr_get_data_p
   PUBLIC :: dbcsr_set_work_size
   !PUBLIC :: dbcsr_finalize

   ! replication
   PUBLIC :: dbcsr_replicate_all
   PUBLIC :: dbcsr_sum_replicated
   PUBLIC :: dbcsr_distribute

   ! misc
   PUBLIC :: dbcsr_distribution_get_num_images
   PUBLIC :: dbcsr_convert_offsets_to_sizes
   PUBLIC :: dbcsr_convert_sizes_to_offsets
   PUBLIC :: dbcsr_run_tests
   PUBLIC :: dbcsr_test_mm
   PUBLIC :: dbcsr_scalar

   ! high level matrix functions
   PUBLIC :: dbcsr_norm_frobenius
   PUBLIC :: dbcsr_norm_maxabsnorm
   PUBLIC :: dbcsr_norm_column
   PUBLIC :: dbcsr_hadamard_product
   PUBLIC :: dbcsr_func_artanh
   PUBLIC :: dbcsr_func_dtanh
   PUBLIC :: dbcsr_func_inverse
   PUBLIC :: dbcsr_func_tanh
   !PUBLIC :: dbcsr_print
   PUBLIC :: dbcsr_print_block_sum
   PUBLIC :: dbcsr_checksum
   PUBLIC :: dbcsr_maxabs
   PUBLIC :: dbcsr_norm
   PUBLIC :: dbcsr_gershgorin_norm
   PUBLIC :: dbcsr_frobenius_norm
   PUBLIC :: dbcsr_init_random
   PUBLIC :: dbcsr_function_of_elements
   PUBLIC :: dbcsr_triu

   ! csr conversion
   PUBLIC :: dbcsr_csr_type
   PUBLIC :: dbcsr_csr_p_type
   PUBLIC :: dbcsr_convert_csr_to_dbcsr
   PUBLIC :: dbcsr_convert_dbcsr_to_csr
   PUBLIC :: dbcsr_csr_create_from_dbcsr
   PUBLIC :: dbcsr_csr_destroy
   PUBLIC :: dbcsr_csr_create
   PUBLIC :: dbcsr_csr_eqrow_floor_dist
   PUBLIC :: dbcsr_csr_eqrow_ceil_dist
   PUBLIC :: dbcsr_csr_dbcsr_blkrow_dist
   PUBLIC :: dbcsr_csr_print_sparsity
   PUBLIC :: dbcsr_to_csr_filter
   PUBLIC :: dbcsr_csr_write

   ! binary io
   PUBLIC :: dbcsr_binary_write
   PUBLIC :: dbcsr_binary_read
   PUBLIC :: dbcsr_test_binary_io
   
   ! lib init/finalize
   !PUBLIC :: dbcsr_clear_mempools
   !PUBLIC :: dbcsr_init_lib
   !PUBLIC :: dbcsr_finalize_lib
   !SKIP PUBLIC :: dbcsr_set_config
   !SKIP PUBLIC :: dbcsr_get_default_config
   !SKIP PUBLIC :: dbcsr_print_config
   !SKIP PUBLIC :: dbcsr_reset_randmat_seed
   !PUBLIC :: dbcsr_mp_grid_setup
   !PUBLIC :: dbcsr_print_statistics
   
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !!!!! lib init/finalize !!!!!
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   
   SUBROUTINE c_dbcsr_clear_mempools() BIND(C, name="c_dbcsr_clear_mempools")
      CALL dbcsr_clear_mempools()
   END SUBROUTINE dbcsr_clear_mempools
   
   SUBROUTINE c_dbcsr_init_lib(fcomm, io_unit) bind(C, name="c_dbcsr_init_lib_internal")
      INTEGER(kind=c_int), INTENT(in)                    :: fcomm
      INTEGER(kind=c_int), INTENT(in), optional          :: io_unit

      CALL dbcsr_init_lib(fcomm, io_unit)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_finalise_lib() bind(C, name="c_dbcsr_finalize_lib")
      CALL dbcsr_finalize_lib()
   END SUBROUTINE
   
   SUBROUTINE c_dbcsr_mp_grid_setup(c_dist) BIND(C,name="c_dbcsr_mp_grid_setup")
      TYPE(c_ptr), INTENT(IN), VALUE :: c_dist
      TYPE(dbcsr_distribution_type), POINTER :: dist
      
      CALL c_f_pointer(c_dist,dist)

      CALL dbcsr_mp_grid_setup(dist)
   END SUBROUTINE dbcsr_mp_grid_setup
   
   SUBROUTINE c_dbcsr_print_statistics(c_print_timers, c_callgraph_filename) &
      BIND(C,name="c_dbcsr_print_statistics")
      
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_print_timers
      CHARACTER(len=*), INTENT(IN), OPTIONAL :: c_callgraph_filename

      LOGICAL :: print_timers
      CHARACTER(:, kind=c_char), ALLOCATABLE :: callgraph_filename
      
      IF (PRESENT(c_callgraph_filename)) CALL c_f_string(c_callgraph_filename,callgraph_filename)

      IF (PRESENT(c_print_timers)) THEN 
         print_timers = c_print_timers
         CALL dbcsr_print_statistics_prv(print_timers=print_timers, &
                                         callgraph_filename=callgraph_filename)
      ELSE 
         CALL dbcsr_print_statistics_prv(callgraph_filename=callgraph_filename)
      ENDIF
      
   END SUBROUTINE dbcsr_print_statistics
   
   ! create / release
   !PUBLIC :: dbcsr_distribution_hold
   !PUBLIC :: dbcsr_distribution_release
   !PUBLIC :: dbcsr_distribution_new
   !--- PUBLIC :: dbcsr_create
   ! SKIP PUBLIC :: dbcsr_init_p
   !PUBLIC :: dbcsr_release
   ! SKIP PUBLIC :: dbcsr_release_p
   ! SKIP PUBLIC :: dbcsr_deallocate_matrix
   
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !!!!!!! create/release !!!!!!!!!
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   
   SUBROUTINE c_dbcsr_distribution_hold(c_dist) &
      BIND(C,name="c_dbcsr_distribution_hold") 
   
      TYPE(c_ptr), INTENT(IN), VALUE :: c_dist
      TYPE(dbcsr_distribution_type), POINTER :: dist
      
      CALL c_f_pointer(c_dist,dist)

      CALL dbcsr_distribution_hold(dist)
   END SUBROUTINE dbcsr_distribution_hold

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
                               BIND(C,name="c_dbcsr_create_new")
                               
      TYPE(c_ptr), INTENT(INOUT) :: c_matrix                         
      TYPE(c_ptr), INTENT(IN), VALUE :: c_name
      TYPE(c_ptr), INTENT(IN), VALUE :: c_dist
      CHARACTER(kind=c_char), INTENT(IN), VALUE :: c_matrix_type
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row_blk_size_size, &
                                                c_col_blk_size_size
      INTEGER(kind=c_int), INTENT(IN) :: c_row_blk_size(c_row_size), &
                                         C_col_blk_size(c_col_size)
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_nze, c_data_type
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_reuse, c_reuse_arrays, &
                                                    c_mutable_work
      CHARACTER(kind=c_char), INTENT(IN), OPTIONAL :: c_replication_type
                               
                               
      TYPE(dbcsr_type), POINTER :: matrix
      CHARACTER(:, kind=c_char), ALLOCATABLE  :: name
      TYPE(dbcsr_distribution_type), POINTER :: dist
      LOGICAL, POINTER :: reuse, reuse_arrays, mutable_work
      
      ALLOCATE(matrix)
      CALL c_f_pointer(c_dist,dist)
      CALL c_f_string(c_name,name)
      
#:set list = ['reuse', 'reuse_arrays', 'mutable_work']
#:for var in list
      NULLIFY(${var}$)
      IF PRESENT(c_${var}$) THEN
         ALLOCATE(${var}$)
         ${var}$ = c_${var}$
      ENDIF
#:endfor

      CALL dbcsr_create_new(matrix, name, dist, c_matrix_type, &
                               c_row_blk_size, c_col_blk_size, &
                               c_nze, c_data_type, reuse, &
                               reuse_arrays, mutable_work, replication_type)
#:for var in list
      IF (ASSOCIATED(${var}$)) DEALLOCATE (${var}$)
#:endfor

      c_matrix = c_loc(matrix)
     
   END SUBROUTINE

   SUBROUTINE c_dbcsr_create_template(c_matrix, c_name, c_template, &
                                    c_dist, c_matrix_type, &
                                    c_row_blk_size, c_row_size, &
                                    c_col_blk_size, c_col_size, &
                                    c_nze, c_data_type, &
                                    c_reuse_arrays, c_mutable_work, c_replication_type)
                                    BIND(C,name="c_dbcsr_create_template")
                                    
      TYPE(c_ptr), INTENT(INOUT) :: c_matrix                         
      TYPE(c_ptr), INTENT(IN), VALUE :: c_name
      TYPE(c_ptr), INTENT(IN), VALUE :: c_template
      TYPE(c_ptr), INTENT(IN), VALUE :: c_dist
      CHARACTER(kind=c_char), INTENT(IN), VALUE :: c_matrix_type
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row_blk_size_size, &
                                                c_col_blk_size_size
      INTEGER(kind=c_int), INTENT(IN) :: c_row_blk_size(c_row_size), &
                                         C_col_blk_size(c_col_size)
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_nze, c_data_type
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_reuse, c_reuse_arrays, &
                                                    c_mutable_work
      CHARACTER(kind=c_char), INTENT(IN), OPTIONAL :: c_replication_type
                                    
      TYPE(dbcsr_type), POINTER :: matrix, template
      CHARACTER(:, kind=c_char), ALLOCATABLE  :: name
      TYPE(dbcsr_distribution_type), POINTER :: dist
      LOGICAL, POINTER :: reuse, reuse_arrays, mutable_work
      
      ALLOCATE(matrix)
      CALL c_f_pointer(c_template,template)
      CALL c_f_string(c_name,name)
      
#:set list = ['reuse', 'reuse_arrays', 'mutable_work']
#:for var in list
      NULLIFY(${var}$)
      IF PRESENT(c_${var}$) THEN
         ALLOCATE(${var}$)
         ${var}$ = c_${var}$
      ENDIF
#:endfor

      CALL dbcsr_create_template(matrix, name, template, dist, c_matrix_type, &
                               c_row_blk_size, c_col_blk_size, &
                               c_nze, c_data_type, reuse, &
                               reuse_arrays, mutable_work, replication_type)
#:for var in list
      IF (ASSOCIATED(${var}$)) DEALLOCATE (${var}$)
#:endfor

      c_matrix = c_loc(matrix)
      
   END SUBROUTINE 

   SUBROUTINE c_dbcsr_finalize(c_matrix) bind(C, name="c_dbcsr_finalize")
      TYPE(c_ptr), INTENT(in), value                     :: c_matrix

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_finalize(matrix)
   END SUBROUTINE


   !PUBLIC :: dbcsr_set
   !PUBLIC :: dbcsr_add
   !PUBLIC :: dbcsr_scale
   !PUBLIC :: dbcsr_scale_by_vector
   !PUBLIC :: dbcsr_transposed
   !PUBLIC :: dbcsr_multiply
   !PUBLIC :: dbcsr_copy
   !PUBLIC :: dbcsr_copy_into_existing
   !PUBLIC :: dbcsr_desymmetrize
   !PUBLIC :: dbcsr_add_on_diag
   !PUBLIC :: dbcsr_get_block_diag
   !PUBLIC :: dbcsr_set_diag
   !PUBLIC :: dbcsr_get_diag
   !PUBLIC :: dbcsr_filter
   !PUBLIC :: dbcsr_trace
   !PUBLIC :: dbcsr_dot
   !PUBLIC :: dbcsr_complete_redistribute
   !PUBLIC :: dbcsr_get_block_p
   !PUBLIC :: dbcsr_clear

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !!!!!! primitive matrix operations !!!!!!!
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#:for n, nametype, base, prec, kind, type, dkind, normname, ctype in c_inst_params_float   
   
   SUBROUTINE c_dbcsr_set_${nametype}$ (c_matrix, c_alpha) &
      BIND(C,name="c_dbcsr_set_${nametype}$")
      
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      ${ctype}$, INTENT(IN), VALUE :: c_alpha
      TYPE(dbcsr_type), POINTER :: matrix
      
      CALL c_f_pointer(c_matrix,matrix)
     
      CALL dbcsr_set_${nametype}$(matrix, c_alpha)
      
   END SUBROUTINE 
   
   SUBROUTINE c_dbcsr_add_${nametype}$ (c_matrix_a, c_matrix_b, c_alpha_scalar, c_beta_scalar) &
      BIND(C,name="c_dbcsr_add_${nametype}$")
     
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a, c_matrix_b
      ${ctype}$, INTENT(IN), VALUE :: c_alpha_scalar, c_beta_scalar
      TYPE(dbcsr_type), POINTER :: matrix_a, matrix_b
     
      CALL c_f_pointer(c_matrix_a,matrix_a)
      CALL c_f_pointer(c_matrix_b,matrix_b)
      CALL dbcsr_add_${nametype}$ (matrix_a, matrix_b, c_alpha_scalar, c_beta_scalar)
      
   END SUBROUTINE 

   SUBROUTINE c_dbcsr_scale_${nametype}$ (c_matrix_a, c_alpha_scalar, c_last_column)
      BIND(C,name="c_dbcsr_scale_${nametype}$")
   
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a
      ${ctype}$, INTENT(IN), VALUE :: c_alpha_scalar
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_last_column
      TYPE(dbcsr_type), POINTER :: matrix_a
      
      CALL c_f_pointer(c_matrix_a,matrix_a)
      CALL dbcsr_scale_${nametype1}$ (matrix_a, alpha_scalar, last_column)
      
   END SUBROUTINE 
   
   SUBROUTINE c_dbcsr_scale_by_vector_${nametype}$ (c_matrix_a, c_alpha, c_side)
      BIND(C,name="c_dbcsr_scale_by_vector_${nametype}$")
      
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_alpha_size
      ${ctype}$, INTENT(IN) :: c_alpha(c_alpha_size)
      TYPE(c_ptr), INTENT(IN), VALUE :: c_side
      TYPE(dbcsr_type), INTENT(INOUT) :: matrix_a
      CHARACTER(:, kind=c_char), ALLOCATABLE  :: side
      
      CALL c_f_pointer(c_matrix_a,matrix_a)
      CALL c_f_string(c_side,side)
 
      CALL dbcsr_scale_by_vector_${nametype}$ (matrix_a, c_alpha, side)
   END SUBROUTINE 
   
   SUBROUTINE c_dbcsr_multiply_${nametype}$ (c_transa, c_transb, &
                                           c_alpha, c_matrix_a, c_matrix_b, c_beta, c_matrix_c, &
                                           c_first_row, c_last_row, c_first_column, c_last_column, &
                                           c_first_k, c_last_k, &
                                           c_retain_sparsity, c_filter_eps, c_flop)
      BIND(C,name="c_dbcsr_multiply_${nametype}$")
                                            
      CHARACTER(kind=c_char), INTENT(in), value :: c_transa, c_transb
      ${ctype}$, INTENT(in), value :: c_alpha, c_beta
      TYPE(c_ptr), INTENT(in), VALUE :: c_matrix_a, c_matrix_b, c_matrix_c
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_first_row, c_last_row, &
                                                   c_first_column, c_last_column, &
                                                   c_first_k, c_last_k
      LOGICAL(c_bool), INTENT(in), OPTIONAL :: c_retain_sparsity
      REAL(kind=c_double), OPTIONAL :: c_filter_eps
      REAL(kind=c_long_long), OPTIONAL :: c_flop

      LOGICAL                                            :: ret_sp
      TYPE(dbcsr_type), POINTER                          :: matrix_a, matrix_b, matrix_c

      CALL c_f_pointer(c_matrix_a, matrix_a)
      CALL c_f_pointer(c_matrix_b, matrix_b)
      CALL c_f_pointer(c_matrix_c, matrix_c)

      IF (PRESENT(c_retain_sparsity) THEN 
           ret_sp = c_retain_sparsity
           CALL dbcsr_multiply_${nametype}$ (c_transa, c_transb, &
                                            c_alpha, matrix_a, matrix_b, c_beta, matrix_c, &
                                            c_first_row, c_last_row, c_first_column, c_last_column, &
                                            c_first_k, c_last_k, &
                                            retain_sparsity, c_filter_eps, c_flop)
     ELSE 
          CALL dbcsr_multiply_${nametype}$ (c_transa, c_transb, &
                                            c_alpha, matrix_a, matrix_b, c_beta, matrix_c, &
                                            c_first_row, c_last_row, c_first_column, c_last_column, &
                                            c_first_k, c_last_k, &
                                            filter_eps=c_filter_eps, flop=c_flop)
     ENDIF
                                       
   END SUBROUTINE
   
   SUBROUTINE c_dbcsr_add_on_diag_${nametype}$ (c_matrix, c_alpha_scalar) &
      BIND(C,name="c_dbcsr_add_on_diag_${nametype}$")
      
      TYPE(c_ptr), INTENT(IN), VALUE :: matrix
      ${ctype}$, INTENT(IN) :: c_alpha_scalar
      TYPE(dbcsr_type), POINTER :: matrix
      
      CALL c_f_pointer(c_matrix,matrix)
      CALL dbcsr_add_on_diag_${nametype}$ (matrix, c_alpha_scalar)
   END SUBROUTINE 
   
   SUBROUTINE c_dbcsr_set_diag_${nametype}$ (c_matrix, c_diag, c_diag_size) &
      BIND(C,name="c_dbcsr_set_diag_${nametype}$")
      
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_diag_size
      ${ctype}$, INTENT(IN) :: c_diag(c_diag_size)
      TYPE(dbcsr_type), POINTER :: matrix
     
      CALL c_f_pointer(c_matrix,matrix)
      CALL dbcsr_set_diag_${nametype}$ (matrix, c_diag)
   END SUBROUTINE 
   
   SUBROUTINE c_dbcsr_get_diag_${nametype}$ (c_matrix, c_diag, c_diag_size) &
      BIND(C,name="c_dbcsr_get_diag_${nametype}$")
      
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_diag_size
      ${ctype}$, INTENT(INOUT) :: c_diag(c_diag_size)
      TYPE(dbcsr_type), POINTER :: matrix
     
      CALL c_f_pointer(c_matrix,matrix)
      CALL dbcsr_get_diag_${nametype}$ (matrix, c_diag)
   END SUBROUTINE 
   
   SUBROUTINE c_dbcsr_trace_${nametype}$ (c_matrix_a, c_trace) &
      BIND(C,name="c_dbcsr_trace_${nametype}$")
      
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a
      ${ctype}$, INTENT(OUT) :: c_trace
      TYPE(dbcsr_type), POINTER :: matrix_a

      CALL c_f_pointer(c_matrix_a,matrix_a)
      CALL dbcsr_trace_${nametype}$(matrix_a, c_trace)
   END SUBROUTINE 
   
   SUBROUTINE c_dbcsr_dot_${nametype}$ (c_matrix_a, c_matrix_b, c_result) &
      BIND(C,name="c_dbcsr_dot_${nametype}$")
      
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a, c_matrix_b
      ${ctype}$, INTENT(INOUT) :: c_result
      TYPE(dbcsr_type), POINTER :: matrix_a, matrix_b
      
      CALL c_f_pointer(c_matrix_a,matrix_a)
      CALL c_f_pointer(c_matrix_b,matrix_b)
      CALL dbcsr_dot_${nametype}$ (matrix_a, matrix_b, c_result)
   END SUBROUTINE
   
   SUBROUTINE c_dbcsr_get_block_${nametype}$ (c_matrix, c_row, c_col, c_block, &
              c_tr, c_found, c_row_size, c_col_size) &
      BIND(C,name="c_dbcsr_get_block_${nametype}$")
      
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row, c_col
      TYPE(c_ptr), INTENT(INOUT) :: c_block
      LOGICAL(kind=c_bool), INTENT(OUT) :: c_tr, c_found
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_row_size, c_col_size 
     
      TYPE(dbcsr_type), POINTER :: matrix
      ${ctype}$, DIMENSION(:), POINTER :: block
      LOGICAL :: tr, found
      
      CALL c_f_pointer(c_matrix,matrix)

      CALL dbcsr_get_block_p_${nametype}$ (matrix, c_row, c_col, block, tr, &
           found, c_row_size, c_col_size)
      
      c_tr = tr
      c_found = found
      c_block = c_loc(block)
      
   END SUBROUTINE 

   SUBROUTINE c_dbcsr_get_block_notrans_p_${nametype}$ (c_matrix, c_row, c_col, &
              c_block, c_found, c_row_size, c_col_size) &
      BIND(C,name="c_dbcsr_get_block_notrans_p_${nametype}$")
      
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_row, c_col
      TYPE(c_ptr), INTENT(INOUT) :: c_block
      LOGICAL(kind=c_bool), INTENT(OUT) :: c_found
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_row_size, c_col_size 
     
      TYPE(dbcsr_type), POINTER :: matrix
      ${ctype}$, DIMENSION(:), POINTER :: block
      LOGICAL :: found
      
      CALL c_f_pointer(c_matrix,matrix)

      CALL dbcsr_get_block_notrans_p_${nametype}$(matrix, row, col, block, found, row_size, col_size)
      
      c_block = c_loc(block)
      c_found = found
      
   END SUBROUTINE 
   
#:endfor

   SUBROUTINE c_dbcsr_complete_redistribute(c_matrix, c_redist, c_keep_sparsity, c_summation) &
      BIND(C,name="c_dbcsr_complete_redistribute")
      
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(c_ptr), INTENT(INOUT) :: c_redist
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_keep_sparsity, c_summation
      TYPE(dbcsr_type), POINTER :: matrix, redist
      LOGICAL, POINTER :: keep_sparsity, summation
      
      CALL c_f_pointer(c_matrix, matrix)
      IF (C_ASSOCIATED(c_redist)) THEN
         CALL c_f_pointer(c_redist, redist)
      ELSE
         ALLOCATE(redist)
      ENDIF
#:set vars = ['keep_sparsity', 'summation']
#:for var in vars 
      NULLIFY(${var}$)
      IF (PRESENT(c_${var}$)) THEN
         ALLOCATE(${var}$)
         ${var}$ = c_${var}$
      ENDIF
#:endfor

      CALL dbcsr_complete_redistribute(matrix, redist, keep_sparsity, summation)
      IF (C_ASSOCIATED(c_redist)) c_redist = c_loc(redist)
   END SUBROUTINE dbcsr_complete_redistribute

   SUBROUTINE c_dbcsr_filter(c_matrix, c_eps, c_method, c_use_absolute, c_filter_diag) &
      BIND(C,name="c_dbcsr_filter")
      
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      REAL(kind=c_double), INTENT(IN) :: c_eps
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_method
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_use_absolute, c_filter_diag
      TYPE(dbcsr_type), POINTER :: matrix
      LOGICAL, POINTER :: use_absolute, filter_diag

#:set vars = ['use_absolute', 'filter_diag']
#:for var in vars
      NULLIFY(${var}$)
      IF (PRESENT(c_${var}$)) THEN
         ALLOCATE(${var}$)
         ${var}$ = c_${var}$
      ENDIF
#:endfor
      CALL dbcsr_filter(matrix, eps, method, use_absolute, filter_diag)
#:for var in vars
      IF (ASSOCIATED(${var}$)) DEALLOCATE(${var}$)

   END SUBROUTINE dbcsr_filter

   SUBROUTINE c_dbcsr_get_block_diag(c_matrix, c_diag) &
      BIND(C,name="c_dbcsr_get_block_diag")
      
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(c_ptr), INTENT(INOUT) :: c_diag
      TYPE(dbcsr_type), POINTER :: matrix, diag
      
      IF (C_ASSOCIATED(c_diag)) THEN
         CALL c_f_pointer(c_diag,diag)
      ELSE 
         ALLOCATE(diag)
      ENDIF

      CALL dbcsr_get_block_diag(matrix, diag)
      IF (.NOT. C_ASSOCIATED(c_diag)) c_diag = c_loc(diag)
   END SUBROUTINE dbcsr_get_block_diag

   SUBROUTINE c_dbcsr_transposed(c_transposed, c_normal, c_shallow_data_copy, &
                               c_transpose_data, c_transpose_distribution, c_use_distribution) &
      BIND(C,name="c_dbcsr_transposed")
                               
      TYPE(c_ptr), INTENT(INOUT) :: c_transposed
      TYPE(c_ptr), INTENT(IN), VALUE :: c_normal, c_use_distribution
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_shallow_data_copy, c_transpose_data, &
                                                    c_transpose_distribution

      TYPE(dbcsr_type), POINTER :: transposed, normal
      LOGICAL, POINTER :: shallow_data_copy, transpose_data, &
                          transpose_distribution
      TYPE(dbcsr_distribution_type), POINTER :: use_distribution
      
      ALLOCATE(transposed)
      CALL c_f_pointer(c_normal,normal)
      IF (C_ASSOCIATED(c_use_distribution)) &
          CALL c_f_pointer(c_use_distribution,use_distribution)
          
#:set vars = ['shallow_data_copy', 'transpose_data', 'transpose_distribution']
#:for var in vars
      NULLIFY(${var}$)
      IF (PRESENT(c_${var}$) THEN
         ALLOCATE(${var}$)
         ${var}$ = c_${var}$
      ENDIF
#:endfor

      CALL dbcsr_transposed(transposed, normal, shallow_data_copy, &
                               transpose_data, transpose_distribution, use_distribution)
    
      c_transposed = c_loc(transposed)
      
#:for var in vars
      IF (ASSOCIATED(${var}$)) DEALLOCATE(${var}$)
#:endfor
    
   END SUBROUTINE
   
   SUBROUTINE c_dbcsr_copy(c_matrix_b, c_matrix_a, c_name, c_keep_sparsity, &
                           c_shallow_data, c_keep_imaginary, c_matrix_type) &
      BIND(C,name="c_dbcsr_copy")
      
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
        CALL c_f_pointer(c_matrix_b,matrix_b)
     ELSE
        ALLOCATE(matrix_b)
     ENDIF
     
     IF (C_ASSOCIATED(c_name)) CALL c_f_string(c_name,name)
     
     CALL c_f_pointer(c_matrix_a,matrix_a)
     
#:set vars = ['keep_sparsity', 'shallow_data', 'keep_imaginary']
#:for var in vars
     NULLIFY(${var}$)
     IF (PRESENT(c_${var}$)) THEN
        ALLOCATE(${var}$)
        ${var}$ = c_${var}$
     ENDIF
#:endif

     CALL dbcsr_copy(matrix_b, matrix_a, name, keep_sparsity, &
                         shallow_data, keep_imaginary, c_matrix_type)
                         
#:for var in vars
     IF (ASSOCIATED(${var}$) DEALLOCATE(${var}$)
#:endfor
     IF (.NOT. C_ASSOCIATED(c_matrix_b)) c_matrix_b = c_loc(matrix_b)
                         
   END SUBROUTINE

   SUBROUTINE c_dbcsr_copy_into_existing(c_matrix_b, c_matrix_a) &
      BIND(C,name="c_dbcsr_copy_into_existing")
   
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a, c_matrix_b
      TYPE(dbcsr_type), POINTER :: matrix_b, matrix_a

      CALL c_f_pointer(c_matrix_a,matrix_a)
      CALL c_f_pointer(c_matrix_b,matrix_b)

      CALL dbcsr_copy_into_existing(matrix_b, matrix_a)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_desymmetrize(c_matrix_a, c_matrix_b) &
      BIND(C,name="c_dbcsr_desymmetrize")
      
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix_a
      TYPE(c_ptr), INTENT(INOUT) :: c_matrix_b
      TYPE(dbcsr_type), POINTER :: matrix_a, matrix_b
      
      CALL c_f_pointer(c_matrix_a,matrix_a)
      IF (C_ASSOCIATED(c_matrix_b)) THEN
         CALL c_f_pointer(c_matrix_b,matrix_b)
      ELSE
         ALLOCATE(matrix_b)
      ENDIF

      CALL dbcsr_desymmetrize(matrix_a, matrix_b)
      IF (.NOT. C_ASSOCIATED(c_matrix_b)) c_matrix_b = c_loc(matrix_b)
      
   END SUBROUTINE

   SUBROUTINE c_dbcsr_clear(c_dbcsr_mat) BIND(C,name="c_dbcsr_clear")
      TYPE(c_ptr), INTENT(IN) :: c_dbcsr_mat
      TYPE(dbcsr_type), POINTER :: dbcsr_mat
      CALL c_f_pointer(c_dbcsr_mat,dbcsr_mat)
      CALL dbcsr_clear(dbcsr_mat)
   END SUBROUTINE

   ! block reservation
   !PUBLIC :: dbcsr_reserve_diag_blocks
   !PUBLIC :: dbcsr_reserve_block2d
   ! MODIFIED !!! PUBLIC :: dbcsr_reserve_blocks
   !PUBLIC :: dbcsr_reserve_all_blocks

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !!!!!!!! block_reservations !!!!!!!!!!!!!!!!
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
   SUBROUTINE c_dbcsr_reserve_diag_blocks(c_matrix) &
      BIND(C,name="c_dbcsr_reserve_diag_blocks")
      
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix
      CALL c_f_pointer(c_matrix,matrix)
      CALL dbcsr_reserve_diag_blocks(matrix)
   END SUBROUTINE dbcsr_reserve_diag_blocks
   
   SUBROUTINE c_dbcsr_reserve_blocks(c_matrix, c_rows, c_cols, c_size) &
      BIND(C,name="c_dbcsr_reserve_blocks")
      
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN), VALUE :: c_size
      INTEGER(kind=c_int), INTENT(IN) :: c_rows(c_size), c_cols(c_size)
      TYPE(dbcsr_type), POINTER:: matrix
      INTEGER, DIMENSION(c_size) :: rows, cols
      
      CALL c_f_pointer(c_matrix,matrix)
      rows = c_rows + 1
      cols = c_cols + 1

      CALL dbcsr_reserve_blocks(matrix, rows, cols)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_reserve_all_blocks(c_matrix) &
      BIND(C,name="c_dbcsr_reserve_all_blocks")
      
      TYPE(c_ptr), INTENT(IN); VALUE :: c_matrix
      TYPE(dbcsr_type), POINTER :: matrix

      CALL c_f_pointer(c_matrix,matrix)
      CALL dbcsr_reserve_all_blocks(matrix)
   END SUBROUTINE 
   
#:for n, nametype, base, prec, kind, type, dkind, normname, ctype in c_inst_params_float
   
   SUBROUTINE c_dbcsr_reserve_block2d_${nametype}$ (c_matrix, c_row, c_col, 
              c_block, c_row_size, c_col_size, c_transposed, c_existed) &
      BIND(C,name="c_dbcsr_reserve_block2d_${nametype}$")
      
      TYPE(c_ptr), INTENT(IN), VALUE :: c_matrix
      INTEGER(kind=c_int), INTENT(IN) :: c_row, c_col, c_row_size, c_col_size
      ${ctype}$, INTENT(IN), DIMENSION(c_row_size, c_col_size), TARGET :: c_block
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_transposed
      LOGICAL(kind=c_bool), INTENT(OUT), OPTIONAL :: c_existed
      
      TYPE(dbcsr_type), POINTER :: matrix
      ${type}$, DIMENSION(:, :), POINTER        :: block
      LOGICAL, POINTER :: transposed
      LOGICAL, POINTER :: existed
      
      CALL c_f_pointer(c_matrix, matrix)
      block => c_block
      NULLIFY(transposed)
      NULLIFY(existed)
      IF (PRESENT(c_transposed)) THEN
        ALLOCATE(transposed)
        transposed = c_transposed
      ENDIF
      IF (PRESENT(c_existed)) ALLOCATE(existed)

      CALL dbcsr_reserve_block2d_${nametype}$ (matrix, row, col, block, transposed, existed)
      IF (PRESENT(c_existed)) c_existed = existed
   END SUBROUTINE 
   
#:endfor

    ! iterator
   !PUBLIC :: dbcsr_iterator_start
   !PUBLIC :: dbcsr_iterator_stop
   PUBLIC :: dbcsr_iterator_blocks_left
   PUBLIC :: dbcsr_iterator_next_block
   
   !-------------------------------!
   !        iterator               !
   !-------------------------------!

   SUBROUTINE c_dbcsr_iterator_stop(c_iterator)
      BIND(C,name="c_dbcsr_iterator_stop")
      
      TYPE(c_ptr), INTENT(INOUT) :: c_iterator
      TYPE(dbcsr_iterator_type), POINTER :: iterator
      
      CALL c_f_pointer(c_iterator,iterator)
      CALL dbcsr_iterator_stop(iterator)
      IF (ASSOCIATED(iterator)) DEALLOCATE(iterator)
      c_iterator = c_null_ptr
      
   END SUBROUTINE

   SUBROUTINE c_dbcsr_iterator_start(c_iterator, c_matrix, c_shared, c_dynamic, &
                                   c_dynamic_byrows, c_contiguous_pointers, c_read_only)
      BIND(C,name="c_dbcsr_iterator_start")                               
    
      TYPE(c_ptr), INTENT(INOUT) :: c_iterator
      TYPE(c_ptr), INTENT(IN) :: c_matrix
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL :: c_shared, c_dynamic, c_dynamic_byrows, &
                                                    c_contiguous_pointers, c_read_only
      TYPE(dbcsr_iterator_type), POINTER :: iterator
      TYPE(dbcsr_type), POINTER :: matrix
      LOGICAL, POINTER :: shared, dynamic, dynamic_byrows, &
                          contiguous_pointers, read_only

      ALLOCATE(iterator)
      CALL c_f_pointer(c_matrix,matrix)
#:set vars = ['shared', 'dynamic', 'dynamic_byrows', 'contiguous_pointers', 'read_only']
#:for var in vars
      NULLIFY(${var}$)
      IF (PRESENT(c_${var}$)) THEN
         ALLOCATE(${var}$)
         ${var}$ = c_${var}$
      ENDIF
#:endfor

      CALL dbcsr_iterator_start(iterator, matrix, shared, dynamic, &
                                   dynamic_byrows, contiguous_pointers, read_only)
                                   
      c_iterator = c_loc(iterator)
#:for var in vars
      IF (ASSOCIATED(${var}$)) DEALLOCATE(${var}$)
    
   END SUBROUTINE 

   FUNCTION c_dbcsr_iterator_blocks_left(iterator) RESULT(c_blocks_left) &
   BIND(C,name="c_dbcsr_iterator_blocks_left")
      TYPE(c_ptr), INTENT(IN), VALUE :: c_iterator
      TYPE(dbcsr_iterator_type), POINTER :: iterator
      LOGICAL(kind=c_bool) :: c_blocks_left

      CALL c_f_pointer(c_iterator,iterator)
      blocks_left = dbcsr_iterator_blocks_left(iterator)
   END FUNCTION dbcsr_iterator_blocks_left

   SUBROUTINE c_dbcsr_print(c_matrix) bind(C, name="c_dbcsr_print")
      TYPE(c_ptr), INTENT(in), value                     :: c_matrix

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_print(matrix)

      ! Fortran and C may use different buffers for I/O, make sure we flush before returning:
      flush (default_output_unit)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_get_stored_coordinates(c_matrix, row, col, processor) bind(C, name="c_dbcsr_get_stored_coordinates")
      TYPE(c_ptr), INTENT(in), value                     :: c_matrix
      INTEGER(kind=c_int), INTENT(in), value             :: row, col
      INTEGER(kind=c_int), INTENT(out)                   :: processor

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_get_stored_coordinates(matrix, row + 1, col + 1, processor)
   END SUBROUTINE

   SUBROUTINE c_dbcsr_put_block_d(c_matrix, row, col, block, block_length) bind(C, name="c_dbcsr_put_block_d")
      TYPE(c_ptr), INTENT(in), value                     :: c_matrix
      INTEGER(kind=c_int), INTENT(in), value             :: row, col, block_length
      REAL(kind=c_double), INTENT(in)                    :: block(block_length)

      TYPE(dbcsr_type), POINTER                          :: matrix

      CALL c_f_pointer(c_matrix, matrix)

      CALL dbcsr_put_block(matrix, row + 1, col + 1, block)
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

   SUBROUTINE c_free_string(c_string) BIND(C, name="c_free_string")
      TYPE(c_ptr), INTENT(INOUT)             :: c_string

      CHARACTER(:, kind=c_char), POINTER :: string

      CALL c_f_pointer(c_string, string)
      DEALLOCATE (string)
      c_string = c_null_ptr

   END SUBROUTINE c_free_string

END MODULE
