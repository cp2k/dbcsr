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

namespace dbcsr {

  void add(dm_dbcsr& mat_a, dm_dbcsr& mat_b, double pa, double pb){
    c_dbcsr_add_d(&mat_a,&mat_b,pa,pb);
  }

  void init_lib(){
    c_dbcsr_init_lib();
  }

  void distribution_new(void*& dist, MPI_Comm comm, int* row_dist, int row_dist_size,
                               int* col_dist, int col_dist_size){
    MPI_Fint fcomm = MPI_Comm_c2f(comm);
    c_dbcsr_distribution_new_aux(&dist,&fcomm,row_dist,row_dist_size,col_dist,col_dist_size);
  }

  void create_new(dm_dbcsr& matrix, const char* name, void* dist, char matrix_type, int* row_blk_sizes,
                         int row_blk_sizes_length, int* col_blk_sizes, int col_blk_sizes_length){
    c_dbcsr_create_new_d(&matrix,name,dist,matrix_type,row_blk_sizes,
                         row_blk_sizes_length,col_blk_sizes,col_blk_sizes_length);
  }

  void release(dm_dbcsr& matrix){
    c_dbcsr_release(&matrix);
  }

  void multiply(char transa, char transb, double alpha, dm_dbcsr& c_matrix_a, dm_dbcsr& c_matrix_b,
                  double beta, dm_dbcsr& c_matrix_c){
    c_dbcsr_multiply_d(transa,transb,alpha,&c_matrix_a,&c_matrix_b,beta,&c_matrix_c,nullptr);
  }

  void multiply_eps(char transa, char transb, double alpha, dm_dbcsr& c_matrix_a, dm_dbcsr& c_matrix_b,
                  double beta, dm_dbcsr& c_matrix_c, double sthr){
    c_dbcsr_multiply_eps_d(transa,transb,alpha,&c_matrix_a,&c_matrix_b,beta,&c_matrix_c,sthr);
  }

  void get_stored_coordinates(dm_dbcsr& matrix, int row, int col, int* processor){
    c_dbcsr_get_stored_coordinates(matrix,row,col,processor);
  }

  void distribution_release(void*& dist){
    c_dbcsr_distribution_release(&dist);
  }

 
  void put_block(dm_dbcsr& matrix, int row, int col, double* block, int block_length){
    c_dbcsr_put_block_d(matrix,row,col,block,block_length);
  }

  void copy(dm_dbcsr& c_matrix_a, dm_dbcsr& c_matrix_b){
    c_dbcsr_copy_d(&c_matrix_a,&c_matrix_b);
  }

  void finalize_lib(MPI_Comm comm){
    MPI_Fint fcomm = MPI_Comm_c2f(comm);
    c_dbcsr_finalize_lib_aux(&fcomm);
  }

  void finalize_lib_silent(MPI_Comm comm){
    MPI_Fint fcomm = MPI_Comm_c2f(comm);
    c_dbcsr_finalize_lib_aux_silent(&fcomm);
  }

  void finalize(dm_dbcsr matrix){
    c_dbcsr_finalize(matrix);
  }

  void trace_ab(dm_dbcsr& mat_a, dm_dbcsr& mat_b, double& tr){
    c_dbcsr_trace_ab_d(&mat_a,&mat_b,tr);
  }

  void trace_a(dm_dbcsr& mat_a, double& tr){
    c_dbcsr_trace_a_d(&mat_a,tr);
  }

  void set_diag(dm_dbcsr& mat_a, double* diags, int dim){
    c_dbcsr_set_diag_d(&mat_a,diags,dim);
  }

  void set(dm_dbcsr& mat_a, double scl){
    c_dbcsr_set_d(&mat_a,scl);
  }

  void get_block(dm_dbcsr& mat_a, int row, int col, double* block, bool& found, int row_size, int col_size){
    c_dbcsr_get_block_d(&mat_a,row,col,block,found,row_size,col_size);
  }

  void filter(dm_dbcsr& mat_a, double eps){
    c_dbcsr_filter_d(&mat_a,eps);
  }

  void gershgorin_estimate(dm_dbcsr& mat, int* bdims, int nblocks, int tot_dim, double* sums, double* diags){
    c_dbcsr_gershgorin_estimate_d(&mat,bdims,nblocks,tot_dim,sums,diags);
  }

  void scale(dm_dbcsr& mat_a, double eps){
    c_dbcsr_scale_d(&mat_a,eps);
  }

  void print(dm_dbcsr& matrix){
    c_dbcsr_print(matrix);
  }

  void read(dm_dbcsr& matrix, char* cfname, void** fdist){
    c_dbcsr_read_d(&matrix,cfname,fdist);
  }

  void write(dm_dbcsr& matrix, char* cfname){
    c_dbcsr_write_d(&matrix,cfname);
  }

  void maxabs(dm_dbcsr& matrix, double& amv){
    c_dbcsr_maxabs_d(&matrix,&amv);
  }

}

#endif

