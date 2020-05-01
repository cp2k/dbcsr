/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#ifndef DBCSR_H
#define DBCSR_H

#include <mpi.h>
#include <stdbool.h> /* we need bool from C99 */
#:include 'data/dbcsr.fypp'

static const int dbcsr_type_real_4 = 1;
static const int dbcsr_type_real_8 = 3;
static const int dbcsr_type_complex_4 = 5;
static const int dbcsr_type_complex_8 = 7;

static const int dbcsr_filter_frobenius = 1;

static const int dbcsr_norm_frobenius = 1;
static const int dbcsr_norm_maxabsnorm = 2;
static const int dbcsr_norm_gershgorin = 3;
static const int dbcsr_norm_column = 4;
 
static const int dbcsr_func_inverse = 0;
static const int dbcsr_func_tanh = 1;
static const int dbcsr_func_dtanh = 2;
static const int dbcsr_func_ddtanh = 3;
static const int dbcsr_func_artanh = 4;
static const int dbcsr_func_inverse_special = 5;
static const int dbcsr_func_spread_from_zero = 6;
static const int dbcsr_func_sin = 7;
static const int dbcsr_func_dsin = 8;
static const int dbcsr_func_ddsin = 9;
static const int dbcsr_func_asin = 10;
static const int dbcsr_func_cos = 11;
static const int dbcsr_func_truncate = 12;

static const char dbcsr_type_invalid = '0';
static const char dbcsr_type_no_symmetry = 'N';
static const char dbcsr_type_symmetric = 'S';
static const char dbcsr_type_antisymmetric = 'A';
static const char dbcsr_type_hermitian = 'H';
static const char dbcsr_type_antihermitian = 'K';

static const char dbcsr_no_transpose = 'N';
static const char dbcsr_transpose = 'T';
static const char dbcsr_conjugate_transpose = 'C';

static const char dbcsr_repl_none = 'N';
static const char dbcsr_repl_row = 'R';
static const char dbcsr_repl_col = 'C';
static const char dbcsr_repl_full = 'A';

#if defined(__cplusplus)
extern "C" {
#endif

    //----------------------------------------------------!
    //                    lib init/finalize               !
    //----------------------------------------------------!

    void c_dbcsr_clear_mempools();
    
    void c_dbcsr_mp_grid_setup(void* c_dist);

    void c_dbcsr_init_lib_internal(MPI_Fint* fcomm, int* io_unit);

    inline void c_dbcsr_init_lib(MPI_Comm comm, int* io_unit)
    {
        MPI_Fint fcomm = MPI_Comm_c2f(comm);
        c_dbcsr_init_lib_internal(&fcomm, io_unit);
    }
    
    void c_dbcsr_print_statistics(const bool* c_print_timers, const char** c_callgraph_filename);

    void c_dbcsr_finalize_lib(void);
    
   //-------------------------------------------------------!
   //                    create/release                     !
   //-------------------------------------------------------!
   
    void c_dbcsr_distribution_hold(void* c_dist);

    void c_dbcsr_distribution_new_aux(void** dist, MPI_Fint* fcomm, int* row_dist, int row_dist_size,
                                      int* col_dist, int col_dist_size);

    inline void c_dbcsr_distribution_new(void** dist, MPI_Comm comm, int* row_dist, int row_dist_size,
                                         int* col_dist, int col_dist_size)
    {
        MPI_Fint fcomm = MPI_Comm_c2f(comm);
        c_dbcsr_distribution_new_aux(dist, &fcomm, row_dist, row_dist_size, col_dist, col_dist_size);
    }

    void c_dbcsr_distribution_release(void** dist);

    void c_dbcsr_create_new(void** c_matrix, const char* c_name, void* c_dist, const char c_matrix_type, 
                               const int* c_row_blk_size, const int c_row_size, 
                               const int* c_col_blk_size, const int c_col_size, 
                               const int* c_nze, const int* c_data_type, const bool* c_reuse,
                               const bool* c_reuse_arrays, const bool* c_mutable_work, 
                               const char* c_replication_type);

    void c_dbcsr_create_template(void** c_matrix, char* c_name, void* c_template, 
                               void* c_dist, const char c_matrix_type, 
                               const int* c_row_blk_size, const int c_row_size, 
                               const int* c_col_blk_size, const int c_col_size, 
                               const int* c_nze, const int* c_data_type, 
                               const bool* c_reuse_arrays, const bool* c_mutable_work, 
                               const char* c_replication_type);
                               
    void c_dbcsr_finalize(void* matrix);

    void c_dbcsr_release(void** matrix);
    
   //----------------------------------------------------------!
   //              primitive matrix operations                 !
   //----------------------------------------------------------!

#:for n_inst, nametype, base, prec, ctype, extype in c_exparams  
   
    void c_dbcsr_set_${nametype}$ (void* c_matrix, const ${extype}$ c_alpha);
   
    void c_dbcsr_add_${nametype}$ (void* c_matrix_a, void* c_matrix_b, 
                                  const ${extype}$ c_alpha_scalar, const ${extype}$ c_beta_scalar);

    void c_dbcsr_scale_${nametype}$ (void* c_matrix_a, const ${extype}$ c_alpha_scalar, int c_last_column);
    
    void c_dbcsr_scale_by_vector_${nametype}$ (void* c_matrix_a, const ${extype}$* c_alpha, 
                                               const int c_alpha_size, const char* c_side);
    
    void c_dbcsr_multiply_${nametype}$ (char c_transa, char c_transb,
                                        const ${extype}$ c_alpha, void* c_matrix_a, void* c_matrix_b, 
                                        const ${extype}$ c_beta, void* c_matrix_c,
                                        const int* c_first_row, const int* c_last_row, 
                                        const int* c_first_column, const int* c_last_column,
                                        const int* c_first_k, const int* c_last_k,
                                        const bool* c_retain_sparsity, const double* c_filter_eps, 
                                        long long int* c_flop);
                                        
    void c_dbcsr_add_on_diag_${nametype}$ (void* c_matrix, const ${extype}$ c_alpha_scalar);
   
// CHECK SIZES !!!!!!!!!!!!!!
   
   
    void c_dbcsr_set_diag_${nametype}$ (void* c_matrix, const ${extype}$* c_diag, const int c_diag_size);
   
    void c_dbcsr_get_diag_${nametype}$ (void* c_matrix, ${extype}$* c_diag, const int c_diag_size);
     
    void c_dbcsr_trace_${nametype}$ (void* c_matrix_a, ${extype}$* c_trace);
     
    void c_dbcsr_dot_${nametype}$ (void* c_matrix_a, void* c_matrix_b, ${extype}$* c_result);
    
    void c_dbcsr_get_block_${nametype}$ (void* c_matrix, const int c_row, const int c_col, 
                                         ${extype}$** c_block, bool* c_tr, bool* c_found, 
                                         int* c_row_size, int* c_col_size);

    void c_dbcsr_get_block_notrans_p_${nametype}$ (void* c_matrix, const int c_row, const int c_col, 
                                                   ${extype}$** c_block, bool* c_found, 
                                                   int* c_row_size, int* c_col_size);

#:endfor
   
    void c_dbcsr_complete_redistribute(void* c_matrix, void** c_redist, 
           const bool* c_keep_sparsity, const bool* c_summation);

    void c_dbcsr_filter(void* c_matrix, const double* c_eps, const int* c_method, 
                        const bool* c_use_absolute, const bool* c_filter_diag);

    void c_dbcsr_get_block_diag(void* c_matrix, void** c_diag);
   
    void c_dbcsr_transposed(void** c_transposed, void* c_normal, const bool* c_shallow_data_copy,
                               const bool* c_transpose_data, const bool* c_transpose_distribution, 
                               const bool* c_use_distribution);
   
    void c_dbcsr_copy(void** c_matrix_b, void* c_matrix_a, const char* c_name, 
                      const bool* c_keep_sparsity, const bool* c_shallow_data, 
                      const bool* c_keep_imaginary, const int c_matrix_type);

    void c_dbcsr_copy_into_existing(void* c_matrix_b, void* c_matrix_a);
      
    void c_dbcsr_desymmetrize(void* c_matrix_a, void** c_matrix_b);

    void c_dbcsr_clear(void* c_dbcsr_mat); 

    //-----------------------------------------------------------------!
    //                   block_reservations                            !
    //-----------------------------------------------------------------!
  
    void c_dbcsr_reserve_diag_blocks(void* c_matrix);
   
    void c_dbcsr_reserve_blocks(void* c_matrix, const int* c_rows, const int* c_cols, const int c_size);

    void c_dbcsr_reserve_all_blocks(void* c_matrix);
   
#:for n_inst, nametype, base, prec, ctype, extype in c_exparams
   
    void c_dbcsr_reserve_block2d_${nametype}$ (void* c_matrix, const int c_row, const int c_col, 
              const ${extype}$* c_block, const int c_row_size, const int c_col_size, 
              const bool* c_transposed, bool* c_existed);
   
#:endfor
   
     //-------------------------------!
     //        iterator               !
     //-------------------------------!

     void* c_dbcsr_iterator_stop(void** c_iterator);

     void* c_dbcsr_iterator_start(void** c_iterator, void* c_matrix, const bool* c_shared, 
                                  const bool* c_dynamic, const bool* c_dynamic_byrows, 
                                  const bool* c_contiguous_pointers, const bool* c_read_only);

     bool c_dbcsr_iterator_blocks_left(void* c_iterator);
   
     void c_dbcsr_iterator_next_block_index(void* c_iterator, int* c_row, int* c_column, int* c_blk, int* c_blk_p);

#:for n_inst, nametype, base, prec, ctype, extype in c_exparams

     void c_dbcsr_iterator_next_2d_block_${nametype}$ (void* c_iterator, int* c_row, int* c_column, 
             ${extype}$** c_block, bool* c_transposed, int* c_block_number, 
             int* c_row_size, int* c_col_size, int* c_row_offset, int* c_col_offset);
 
#:endfor
   
   //--------------------------------------------------------!
   //                  work operations                       !
   //--------------------------------------------------------!
  
#:for n_inst, nametype, base, prec, ctype, extype in c_exparams
     void c_dbcsr_put_block2d_${nametype}$ (void* c_matrix, const int c_row, const int c_col, 
                                            const ${extype}$* c_block, const int c_row_size, 
                                            const int c_col_size, const bool* c_summation, 
                                            const ${extype}$* c_scale);
#:endfor
     
   //------------------------------------------------------------!
   //                   replication                              !
   //------------------------------------------------------------!  
    
   void c_dbcsr_replicate_all(void* c_matrix);

   void c_dbcsr_distribute(void* c_matrix, bool* c_fast);

   void c_dbcsr_sum_replicated(void* c_matrix);
   
   //-----------------------------------------!
   //       high level matrix functions       !
   //-----------------------------------------!

   void c_dbcsr_hadamard_product(void* c_matrix_a, void* c_matrix_b, void* c_matrix_c, const double* c_b_assume_value);
   
   void c_dbcsr_print(void* matrix);
    
   void c_dbcsr_print_block_sum(void* c_matrix, const int* c_unit_nr);
   
   double c_dbcsr_checksum(void* c_matrix, const bool* c_local, const bool* c_pos);
   
   double c_dbcsr_maxabs(void* c_matrix);
   
   double c_dbcsr_gershgorin_norm(void* c_matrix);

   double c_dbcsr_frobenius_norm(void* c_matrix, const bool* c_local);
   
   void c_dbcsr_norm_scalar(void* c_matrix, const int c_which_norm, double* c_norm_scalar);
   
   void c_dbcsr_triu(void* c_matrix);

   void c_dbcsr_init_random(void* c_matrix, const bool* c_keep_sparsity);
   
   void c_dbcsr_function_of_elements(void* c_matrix, const int c_func, const double* c_a0, 
           const double* c_a1, const double* c_a2);

   //--------------------------------------------------!
   //           setters/getters                        !
   //--------------------------------------------------!
   
   int c_dbcsr_nblkrows_total(void* c_matrix);
   
   int c_dbcsr_nblkcols_total(void* c_matrix);
   
   int c_dbcsr_nblkrows_local(void* c_matrix);
   
   int c_dbcsr_nblkcols_local(void* c_matrix);
   
   void c_dbcsr_get_info(void* c_matrix, int* c_nblkrows_total, int* c_nblkcols_total,
                             int* c_nfullrows_total, int* c_nfullcols_total, 
                             int* c_nblkrows_local, int* c_nblkcols_local, 
                             int* c_nfullrows_local, int* c_nfullcols_local, 
                             int* c_my_prow, int* c_my_pcol, 
                             int* c_local_rows, int* c_local_cols, 
                             int* c_proc_row_dist, int* c_proc_col_dist, 
                             int* c_row_blk_size, int* c_col_blk_size, 
                             int* c_row_blk_offset, int* c_col_blk_offset, 
                             void** c_distribution, char** c_name, char* c_matrix_type, 
                             int* c_data_type, int* c_group);
                             
    void c_dbcsr_distribution_get(void* c_dist, int** c_row_dist, int** c_col_dist, 
                                  int* c_nrows, int* c_ncols, bool* c_has_threads, 
                                  int* c_group, int* c_mynode, int* c_numnodes, int* c_nprows, 
                                  int* c_npcols, int* c_myprow, int* c_mypcol, int** c_pgrid, 
                                  bool* c_subgroups_defined, int* c_prow_group, int* c_pcol_group);

    void c_dbcsr_get_stored_coordinates(void* matrix, int row, int col, int* processor);
    
    void c_dbcsr_setname(void* c_matrix, char* c_newname);
   
    char c_dbcsr_get_matrix_type(void* c_matrix);
    
    double c_dbcsr_get_occupation(void* c_matrix);
   
    int c_dbcsr_get_num_blocks(void* c_matrix);
   
    int c_dbcsr_get_data_size(void* c_matrix);
   
    bool c_dbcsr_has_symmetry(void* c_matrix);
   
    int c_dbcsr_nfullrows_total(void* c_matrix);
   
    int c_dbcsr_nfullcols_total(void* c_matrix);
   
    bool c_dbcsr_valid_index(void* c_matrix);
   
    int c_dbcsr_get_data_type(void* c_matrix);

    void c_free_string(char** c_string);

#if defined(__cplusplus)
}
#endif

#endif /*DBCSR_H*/
