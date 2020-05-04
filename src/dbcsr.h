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

typedef void* dbcsr_matrix_t;
typedef void* dbcsr_dist_t;
typedef void* dbcsr_iterator_t;

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
    
    void c_dbcsr_print_statistics(const bool* c_print_timers, const char* c_callgraph_filename);

    void c_dbcsr_finalize_lib(void);
    
   //-------------------------------------------------------!
   //                    create/release                     !
   //-------------------------------------------------------!
   
    void c_dbcsr_distribution_hold(const dbcsr_dist_t c_dist);

    void c_dbcsr_distribution_new_aux(dbcsr_dist_t* dist, MPI_Fint* fcomm, int* row_dist, int row_dist_size,
                                      int* col_dist, int col_dist_size);

    inline void c_dbcsr_distribution_new(dbcsr_dist_t* dist, MPI_Comm comm, int* row_dist, int row_dist_size,
                                         int* col_dist, int col_dist_size)
    {
        MPI_Fint fcomm = MPI_Comm_c2f(comm);
        c_dbcsr_distribution_new_aux(dist, &fcomm, row_dist, row_dist_size, col_dist, col_dist_size);
    }

    void c_dbcsr_distribution_release(dbcsr_dist_t* dist);

    void c_dbcsr_create_new(dbcsr_matrix_t* c_matrix, const char* c_name, 
							   const dbcsr_dist_t, const char c_matrix_type, 
                               const int* c_row_blk_size, const int c_row_size, 
                               const int* c_col_blk_size, const int c_col_size, 
                               const int* c_nze, const int* c_data_type, const bool* c_reuse,
                               const bool* c_reuse_arrays, const bool* c_mutable_work, 
                               const char* c_replication_type);

    void c_dbcsr_create_template(dbcsr_matrix_t* c_matrix, char* c_name, 
							   const dbcsr_matrix_t c_template, 
                               const dbcsr_dist_t c_dist, const char* c_matrix_type, 
                               const int* c_row_blk_size, const int c_row_size, 
                               const int* c_col_blk_size, const int c_col_size, 
                               const int* c_nze, const int* c_data_type, 
                               const bool* c_reuse_arrays, const bool* c_mutable_work, 
                               const char* c_replication_type);
                               
    void c_dbcsr_finalize(const dbcsr_matrix_t matrix);

    void c_dbcsr_release(dbcsr_matrix_t* matrix);
    
   //----------------------------------------------------------!
   //              primitive matrix operations                 !
   //----------------------------------------------------------!

#:for n_inst, nametype, base, prec, ctype, extype in c_exparams  
   
    void c_dbcsr_set_${nametype}$ (dbcsr_matrix_t c_matrix, const ${extype}$ c_alpha);
   
    void c_dbcsr_add_${nametype}$ (const dbcsr_matrix_t c_matrix_a, dbcsr_matrix_t c_matrix_b, 
                                  const ${extype}$ c_alpha_scalar, const ${extype}$ c_beta_scalar);

    void c_dbcsr_scale_${nametype}$ (dbcsr_matrix_t c_matrix_a, const ${extype}$ c_alpha_scalar, const int c_last_column);
    
    void c_dbcsr_scale_by_vector_${nametype}$ (const dbcsr_matrix_t c_matrix_a, const ${extype}$* c_alpha, 
                                               const int c_alpha_size, const char* c_side);
    
    void c_dbcsr_multiply_${nametype}$ (char c_transa, char c_transb,
                                        const ${extype}$ c_alpha, const dbcsr_matrix_t c_matrix_a, 
                                        const dbcsr_matrix_t c_matrix_b, 
                                        const ${extype}$ c_beta, dbcsr_matrix_t c_matrix_c,
                                        const int* c_first_row, const int* c_last_row, 
                                        const int* c_first_column, const int* c_last_column,
                                        const int* c_first_k, const int* c_last_k,
                                        const bool* c_retain_sparsity, const double* c_filter_eps, 
                                        long long int* c_flop);
                                        
    void c_dbcsr_add_on_diag_${nametype}$ (dbcsr_matrix_t c_matrix, const ${extype}$ c_alpha_scalar);
   
    void c_dbcsr_set_diag_${nametype}$ (dbcsr_matrix_t c_matrix, const ${extype}$* c_diag, const int c_diag_size);
   
    void c_dbcsr_get_diag_${nametype}$ (const dbcsr_matrix_t c_matrix, ${extype}$* c_diag, const int c_diag_size);
     
    void c_dbcsr_trace_${nametype}$ (const dbcsr_matrix_t c_matrix_a, ${extype}$* c_trace);
     
    void c_dbcsr_dot_${nametype}$ (const dbcsr_matrix_t c_matrix_a, const dbcsr_matrix_t c_matrix_b, ${extype}$* c_result);
    
    void c_dbcsr_get_block_p_${nametype}$ (const dbcsr_matrix_t c_matrix, const int c_row, const int c_col, 
                                         ${extype}$** c_block, bool* c_tr, bool* c_found, 
                                         int* c_row_size, int* c_col_size);

    void c_dbcsr_get_block_notrans_p_${nametype}$ (const dbcsr_matrix_t c_matrix, const int c_row, const int c_col, 
                                                   ${extype}$** c_block, bool* c_found, 
                                                   int* c_row_size, int* c_col_size);

#:endfor
   
    void c_dbcsr_complete_redistribute(dbcsr_matrix_t c_matrix, void** c_redist, 
           const bool* c_keep_sparsity, const bool* c_summation);

    void c_dbcsr_filter(dbcsr_matrix_t c_matrix, const double* c_eps, const int* c_method, 
                        const bool* c_use_absolute, const bool* c_filter_diag);

    void c_dbcsr_get_block_diag(const dbcsr_matrix_t c_matrix, void** c_diag);
   
    void c_dbcsr_transposed(dbcsr_matrix_t* c_transposed, dbcsr_matrix_t c_normal, const bool* c_shallow_data_copy,
                               const bool* c_transpose_data, const bool* c_transpose_distribution, 
                               const bool* c_use_distribution);
   
    void c_dbcsr_copy(dbcsr_matrix_t* c_matrix_b, const dbcsr_matrix_t c_matrix_a, const char* c_name, 
                      const bool* c_keep_sparsity, const bool* c_shallow_data, 
                      const bool* c_keep_imaginary, const int c_matrix_type);

    void c_dbcsr_copy_into_existing(dbcsr_matrix_t c_matrix_b, const dbcsr_matrix_t c_matrix_a);
      
    void c_dbcsr_desymmetrize(const dbcsr_matrix_t c_matrix_a, dbcsr_matrix_t* c_matrix_b);

    void c_dbcsr_clear(dbcsr_matrix_t c_dbcsr_mat); 

    //-----------------------------------------------------------------!
    //                   block_reservations                            !
    //-----------------------------------------------------------------!
  
    void c_dbcsr_reserve_diag_blocks(dbcsr_matrix_t c_matrix);
   
    void c_dbcsr_reserve_blocks(dbcsr_matrix_t c_matrix, const int* c_rows, const int* c_cols, const int c_size);

    void c_dbcsr_reserve_all_blocks(dbcsr_matrix_t c_matrix);
   
#:for n_inst, nametype, base, prec, ctype, extype in c_exparams
   
    void c_dbcsr_reserve_block2d_${nametype}$ (dbcsr_matrix_t c_matrix, const int c_row, const int c_col, 
              const ${extype}$* c_block, const int c_row_size, const int c_col_size, 
              const bool* c_transposed, bool* c_existed);
   
#:endfor
   
     //-------------------------------!
     //        iterator               !
     //-------------------------------!

     void* c_dbcsr_iterator_stop(dbcsr_iterator_t* c_iterator);

     void* c_dbcsr_iterator_start(dbcsr_iterator_t* c_iterator, const dbcsr_matrix_t c_matrix, const bool* c_shared, 
                                  const bool* c_dynamic, const bool* c_dynamic_byrows, 
                                  const bool* c_contiguous_pointers, const bool* c_read_only);

     bool c_dbcsr_iterator_blocks_left(const dbcsr_iterator_t c_iterator);
   
     void c_dbcsr_iterator_next_block_index(const dbcsr_iterator_t c_iterator, 
					int* c_row, int* c_column, int* c_blk, int* c_blk_p);

#:for n_inst, nametype, base, prec, ctype, extype in c_exparams

     void c_dbcsr_iterator_next_2d_block_${nametype}$ (const dbcsr_iterator_t c_iterator, 
             int* c_row, int* c_column, ${extype}$** c_block, bool* c_transposed, int* c_block_number, 
             int* c_row_size, int* c_col_size, int* c_row_offset, int* c_col_offset);
 
#:endfor
   
   //--------------------------------------------------------!
   //                  work operations                       !
   //--------------------------------------------------------!
  
#:for n_inst, nametype, base, prec, ctype, extype in c_exparams
     void c_dbcsr_put_block2d_${nametype}$ (dbcsr_matrix_t c_matrix, const int c_row, const int c_col, 
                                            const ${extype}$* c_block, const int c_row_size, 
                                            const int c_col_size, const bool* c_summation, 
                                            const ${extype}$* c_scale);
                                            
     void c_dbcsr_get_data_${nametype}$ (const dbcsr_matrix_t c_matrix, ${extype}$** c_data, int* c_data_size, 
                                         ${extype}$* c_select_data_type, int* c_lb, int* c_ub);
                                            
#:endfor
     
   //------------------------------------------------------------!
   //                   replication                              !
   //------------------------------------------------------------!  
    
   void c_dbcsr_replicate_all(dbcsr_matrix_t c_matrix);

   void c_dbcsr_distribute(dbcsr_matrix_t c_matrix, bool* c_fast);

   void c_dbcsr_sum_replicated(dbcsr_matrix_t c_matrix);
   
   //-----------------------------------------!
   //       high level matrix functions       !
   //-----------------------------------------!

   void c_dbcsr_hadamard_product(const dbcsr_matrix_t c_matrix_a, const dbcsr_matrix_t c_matrix_b, 
						dbcsr_matrix_t c_matrix_c, const double* c_b_assume_value);
   
   void c_dbcsr_print(const dbcsr_matrix_t matrix);
    
   void c_dbcsr_print_block_sum(const dbcsr_matrix_t c_matrix, const int* c_unit_nr);
   
   double c_dbcsr_checksum(const dbcsr_matrix_t c_matrix, const bool* c_local, const bool* c_pos);
   
   double c_dbcsr_maxabs(const dbcsr_matrix_t c_matrix);
   
   double c_dbcsr_gershgorin_norm(const dbcsr_matrix_t c_matrix);

   double c_dbcsr_frobenius_norm(const dbcsr_matrix_t c_matrix, const bool* c_local);
   
   void c_dbcsr_norm_scalar(const dbcsr_matrix_t c_matrix, const int c_which_norm, double* c_norm_scalar);
   
   void c_dbcsr_triu(const dbcsr_matrix_t c_matrix);

   void c_dbcsr_init_random(dbcsr_matrix_t c_matrix, const bool* c_keep_sparsity);
   
   void c_dbcsr_function_of_elements(dbcsr_matrix_t c_matrix, const int c_func, const double* c_a0, 
           const double* c_a1, const double* c_a2);

   //--------------------------------------------------!
   //           setters/getters                        !
   //--------------------------------------------------!
   
   int c_dbcsr_nblkrows_total(const dbcsr_matrix_t c_matrix);
   
   int c_dbcsr_nblkcols_total(const dbcsr_matrix_t c_matrix);
   
   int c_dbcsr_nblkrows_local(const dbcsr_matrix_t c_matrix);
   
   int c_dbcsr_nblkcols_local(const dbcsr_matrix_t c_matrix);
   
   void c_dbcsr_get_info(const dbcsr_matrix_t c_matrix, int* c_nblkrows_total, int* c_nblkcols_total,
                             int* c_nfullrows_total, int* c_nfullcols_total, 
                             int* c_nblkrows_local, int* c_nblkcols_local, 
                             int* c_nfullrows_local, int* c_nfullcols_local, 
                             int* c_my_prow, int* c_my_pcol, 
                             int* c_local_rows, int* c_local_cols, 
                             int* c_proc_row_dist, int* c_proc_col_dist, 
                             int* c_row_blk_size, int* c_col_blk_size, 
                             int* c_row_blk_offset, int* c_col_blk_offset, 
                             dbcsr_dist_t* c_distribution, char** c_name, char* c_matrix_type, 
                             int* c_data_type, int* c_group);
                            
    void c_dbcsr_distribution_get_aux(const dbcsr_dist_t c_dist, int** c_row_dist, int** c_col_dist, 
                                  int* c_nrows, int* c_ncols, bool* c_has_threads, 
                                  MPI_Fint* c_group, int* c_mynode, int* c_numnodes, int* c_nprows, 
                                  int* c_npcols, int* c_myprow, int* c_mypcol, int** c_pgrid, 
                                  bool* c_subgroups_defined, int* c_prow_group, int* c_pcol_group);
                                  
    inline void c_dbcsr_distribution_get(const dbcsr_dist_t c_dist, int** c_row_dist, int** c_col_dist, 
                                  int* c_nrows, int* c_ncols, bool* c_has_threads, 
                                  MPI_Comm* c_group, int* c_mynode, int* c_numnodes, int* c_nprows, 
                                  int* c_npcols, int* c_myprow, int* c_mypcol, int** c_pgrid, 
                                  bool* c_subgroups_defined, int* c_prow_group, int* c_pcol_group)
    {
		MPI_Fint fgroup;
		c_dbcsr_distribution_get_aux(c_dist, c_row_dist, c_col_dist, 
                                  c_nrows,c_ncols, c_has_threads, 
                                  &fgroup, c_mynode, c_numnodes, c_nprows, 
                                  c_npcols, c_myprow, c_mypcol, c_pgrid, 
                                  c_subgroups_defined,c_prow_group, c_pcol_group);
                             
        if (c_group != nullptr) *c_group = MPI_Comm_f2c(fgroup);
	}

    void c_dbcsr_get_stored_coordinates(const dbcsr_matrix_t matrix, const int row, const int col, int* processor);
    
    void c_dbcsr_setname(const dbcsr_matrix_t c_matrix, const char* c_newname);
   
    char c_dbcsr_get_matrix_type(const dbcsr_matrix_t c_matrix);
    
    double c_dbcsr_get_occupation(const dbcsr_matrix_t c_matrix);
   
    int c_dbcsr_get_num_blocks(const dbcsr_matrix_t c_matrix);
   
    int c_dbcsr_get_data_size(const dbcsr_matrix_t c_matrix);
   
    bool c_dbcsr_has_symmetry(const dbcsr_matrix_t c_matrix);
   
    int c_dbcsr_nfullrows_total(const dbcsr_matrix_t c_matrix);
   
    int c_dbcsr_nfullcols_total(const dbcsr_matrix_t c_matrix);
   
    bool c_dbcsr_valid_index(const dbcsr_matrix_t c_matrix);
   
    int c_dbcsr_get_data_type(const dbcsr_matrix_t c_matrix);
    
    //-----------------------------------------------!
    //                  other                        !
    //-----------------------------------------------!
    
    void c_dbcsr_binary_write(const dbcsr_matrix_t c_matrix, const char* c_filepath);

    void c_dbcsr_binary_read(const char* c_filepath, dbcsr_dist_t c_distribution, MPI_Fint* c_groupid, dbcsr_matrix_t* c_matrix_new);

    void c_free_string(char** c_string);

#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
	//---------------------------------------------------------!
	//                  overloaded functions                   !
	//---------------------------------------------------------!

#:for n_inst, nametype, base, prec, ctype, extype in c_exparams

    inline void c_dbcsr_set (dbcsr_matrix_t c_matrix, const ${extype}$ c_alpha)
	{
		c_dbcsr_set_${nametype}$ (c_matrix, c_alpha);
	}
   
    inline void c_dbcsr_add (const dbcsr_matrix_t c_matrix_a, dbcsr_matrix_t c_matrix_b, 
                             const ${extype}$ c_alpha_scalar, const ${extype}$ c_beta_scalar) 
    {
		c_dbcsr_add_${nametype}$ (c_matrix_a, c_matrix_b, c_alpha_scalar, c_beta_scalar);
	}

    inline void c_dbcsr_scale (dbcsr_matrix_t c_matrix_a, const ${extype}$ c_alpha_scalar, 
                               const int c_last_column)
    {
		c_dbcsr_scale_${nametype}$ (c_matrix_a, c_alpha_scalar, c_last_column);
	}
    
    inline void c_dbcsr_scale_by_vector (const dbcsr_matrix_t c_matrix_a, const ${extype}$* c_alpha, 
                                         const int c_alpha_size, const char* c_side) 
    {
        c_dbcsr_scale_by_vector_${nametype}$ (c_matrix_a, c_alpha, c_alpha_size, c_side);
	}
    
    inline void c_dbcsr_multiply (char c_transa, char c_transb,
									const ${extype}$ c_alpha, const dbcsr_matrix_t c_matrix_a, 
									const dbcsr_matrix_t c_matrix_b, 
									const ${extype}$ c_beta, dbcsr_matrix_t c_matrix_c,
									const int* c_first_row, const int* c_last_row, 
									const int* c_first_column, const int* c_last_column,
									const int* c_first_k, const int* c_last_k,
									const bool* c_retain_sparsity, const double* c_filter_eps, 
									long long int* c_flop) 
	{
		c_dbcsr_multiply_${nametype}$ (c_transa, c_transb, c_alpha, c_matrix_a, 
                                       c_matrix_b, c_beta, c_matrix_c,
                                       c_first_row, c_last_row, c_first_column, c_last_column,
                                       c_first_k, c_last_k, c_retain_sparsity, c_filter_eps, 
                                       c_flop);
    }
                                        
    inline void c_dbcsr_add_on_diag (dbcsr_matrix_t c_matrix, const ${extype}$ c_alpha_scalar) 
    {
		c_dbcsr_add_on_diag_${nametype}$ (c_matrix, c_alpha_scalar);
	}
   
    inline void c_dbcsr_set_diag (dbcsr_matrix_t c_matrix, const ${extype}$* c_diag, const int c_diag_size) 
    {
		c_dbcsr_set_diag_${nametype}$ (c_matrix, c_diag, c_diag_size);
	}
   
    inline void c_dbcsr_get_diag (const dbcsr_matrix_t c_matrix, ${extype}$* c_diag, const int c_diag_size) 
    {
		c_dbcsr_get_diag_${nametype}$ (c_matrix, c_diag, c_diag_size);
	}
     
    inline void c_dbcsr_trace (const dbcsr_matrix_t c_matrix_a, ${extype}$* c_trace) 
    {
		c_dbcsr_trace_${nametype}$ (c_matrix_a, c_trace);
	}
     
    inline void c_dbcsr_dot (const dbcsr_matrix_t c_matrix_a, const dbcsr_matrix_t c_matrix_b, ${extype}$* c_result) 
    {
		c_dbcsr_dot_${nametype}$ (c_matrix_a, c_matrix_b, c_result);
	}
	
    inline void c_dbcsr_get_block_p (const dbcsr_matrix_t c_matrix, const int c_row, const int c_col, 
                                     ${extype}$** c_block, bool* c_tr, bool* c_found, 
                                     int* c_row_size, int* c_col_size) 
    {
        c_dbcsr_get_block_p_${nametype}$ (c_matrix, c_row, c_col, c_block, c_tr, c_found, c_row_size, c_col_size);
	}

    inline void c_dbcsr_get_block_p (const dbcsr_matrix_t c_matrix, const int c_row, const int c_col, 
                                     ${extype}$** c_block, bool* c_found, int* c_row_size, int* c_col_size) 
    {
		c_dbcsr_get_block_notrans_p_${nametype}$ (c_matrix, c_row, c_col, c_block, c_found, c_row_size, c_col_size);
	}
	
	inline void c_dbcsr_reserve_block2d (dbcsr_matrix_t c_matrix, const int c_row, const int c_col, 
                                         const ${extype}$* c_block, const int c_row_size, const int c_col_size, 
                                         const bool* c_transposed, bool* c_existed) 
    {
		c_dbcsr_reserve_block2d_${nametype}$ (c_matrix, c_row, c_col, c_block, c_row_size, c_col_size, c_transposed, c_existed);
	}
	
	inline void c_dbcsr_iterator_next_2d_block (const dbcsr_iterator_t c_iterator, int* c_row, int* c_column, 
	                                            ${extype}$** c_block, bool* c_transposed, int* c_block_number, 
                                                int* c_row_size, int* c_col_size, int* c_row_offset, int* c_col_offset) 
    {
		c_dbcsr_iterator_next_2d_block_${nametype}$ (c_iterator, c_row, c_column, c_block, c_transposed, c_block_number, 
             c_row_size, c_col_size, c_row_offset, c_col_offset);
	}
	
	void c_dbcsr_put_block2d (dbcsr_matrix_t c_matrix, const int c_row, const int c_col, 
                              const ${extype}$* c_block, const int c_row_size, const int c_col_size, 
                              const bool* c_summation, const ${extype}$* c_scale) 
    {
		c_dbcsr_put_block2d_${nametype}$ (c_matrix, c_row, c_col, c_block, c_row_size, 
                                          c_col_size, c_summation, c_scale);
    }
                                            
    void c_dbcsr_get_data (const dbcsr_matrix_t c_matrix, ${extype}$** c_data, int* c_data_size, 
                           ${extype}$* c_select_data_type, int* c_lb, int* c_ub) 
    {
		c_dbcsr_get_data_${nametype}$ (c_matrix, c_data, c_data_size, c_select_data_type, c_lb, c_ub);
	}
													 
	inline void c_dbcsr_iterator_next_block (void* c_iterator, int* c_row, int* c_column, 
				 ${extype}$** c_block, bool* c_transposed, int* c_block_number, 
				 int* c_row_size, int* c_col_size, int* c_row_offset, int* c_col_offset) 
		 {
			 c_dbcsr_iterator_next_2d_block_${nametype}$ (c_iterator, c_row, c_column, 
				 c_block, c_transposed, c_block_number, c_row_size, c_col_size, c_row_offset, c_col_offset);
		 }
		 
	inline void c_dbcsr_put(void* c_matrix, const int c_row, const int c_col, 
                             const ${extype}$* c_block, const int c_row_size, 
                             const int c_col_size, const bool* c_summation, 
                             const ${extype}$* c_scale)
     {
		 c_dbcsr_put_block2d_${nametype}$ (c_matrix, c_row, c_col, c_block, c_row_size, 
                                           c_col_size, c_summation, c_scale);
     }

#:endfor

#endif
