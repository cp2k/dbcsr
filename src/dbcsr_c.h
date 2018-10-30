#ifndef DBCSR_H
#define DBCSR_H

#define dm_dbcsr   void*

extern "C" {
  void c_dbcsr_add_d(dm_dbcsr* mat_a, dm_dbcsr* mat_b, double pa, double pb);
  void c_dbcsr_init_lib();
  void c_dbcsr_distribution_new_aux(void** dist, MPI_Fint* comm, int* row_dist, int row_dist_size,
                                    int* col_dist, int col_dist_size);
  void c_dbcsr_create_new_d(dm_dbcsr* matrix, const char* name, void* dist, char matrix_type, int* row_blk_sizes,
                            int row_blk_sizes_length, int* col_blk_sizes, int col_blk_sizes_length);
  void c_dbcsr_release(dm_dbcsr* matrix);
  void c_dbcsr_multiply_d(char transa, char transb, double alpha, dm_dbcsr* c_matrix_a, dm_dbcsr* c_matrix_b,
                          double beta, dm_dbcsr* c_matrix_c, bool* retain_sparsity);
  void c_dbcsr_multiply_eps_d(char transa, char transb, double alpha, dm_dbcsr* c_matrix_a, dm_dbcsr* c_matrix_b,
                          double beta, dm_dbcsr* c_matrix_c, double sthr);
  void c_dbcsr_get_stored_coordinates(dm_dbcsr matrix, int row, int col, int* processor);
  void c_dbcsr_distribution_release(void** dist);
 
  void c_dbcsr_put_block_d(dm_dbcsr matrix, int row, int col, double* block, int block_length);
  void c_dbcsr_copy_d(dm_dbcsr* c_matrix_a, dm_dbcsr* c_matrix_b);
  void c_dbcsr_finalize_lib_aux(MPI_Fint* fcomm);
  void c_dbcsr_finalize_lib_aux_silent(MPI_Fint* fcomm);
  void c_dbcsr_finalize(dm_dbcsr matrix);
  void c_dbcsr_trace_ab_d(dm_dbcsr* mat_a, dm_dbcsr* mat_b, double& tr);
  void c_dbcsr_trace_a_d(dm_dbcsr* mat_a, double& tr);
  void c_dbcsr_set_diag_d(dm_dbcsr* mat_a, double* diags, int dim);
  void c_dbcsr_set_d(dm_dbcsr* mat_a, double scl);
  void c_dbcsr_get_block_d(dm_dbcsr* mat_a, int row, int col, double* block, bool& found, int row_size, int col_size);
  void c_dbcsr_filter_d(dm_dbcsr* mat_a, double eps);
  void c_dbcsr_gershgorin_estimate_d(dm_dbcsr*, int* bdims, int nblocks, int tot_dim, double* sums, double* diags);
  void c_dbcsr_scale_d(dm_dbcsr* mat_a, double eps);
  void c_dbcsr_print(dm_dbcsr matrix);
  void c_dbcsr_read_d(dm_dbcsr* matrix, char* cfname, void** fdist);
  void c_dbcsr_write_d(dm_dbcsr* matrix, char* cfname);
  void c_dbcsr_maxabs_d(dm_dbcsr* matrix, double* amv);

}
#endif

