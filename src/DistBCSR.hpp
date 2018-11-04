#ifndef DISTBCSR_H
#define DISTBCSR_H

#include <climits>
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

  inline void add(dm_dbcsr& mat_a, dm_dbcsr& mat_b, double pa, double pb){
    c_dbcsr_add_d(&mat_a,&mat_b,pa,pb);
  }

  inline void init_lib(){
    c_dbcsr_init_lib();
  }

  inline void distribution_new(void*& dist, MPI_Comm comm, int* row_dist, int row_dist_size,
                               int* col_dist, int col_dist_size){
    MPI_Fint fcomm = MPI_Comm_c2f(comm);
    c_dbcsr_distribution_new_aux(&dist,&fcomm,row_dist,row_dist_size,col_dist,col_dist_size);
  }

  inline void create_new(dm_dbcsr& matrix, const char* name, void* dist, char matrix_type, int* row_blk_sizes,
                         int row_blk_sizes_length, int* col_blk_sizes, int col_blk_sizes_length){
    c_dbcsr_create_new_d(&matrix,name,dist,matrix_type,row_blk_sizes,
                         row_blk_sizes_length,col_blk_sizes,col_blk_sizes_length);
  }

  inline void release(dm_dbcsr& matrix){
    c_dbcsr_release(&matrix);
  }

  inline void multiply(char transa, char transb, double alpha, dm_dbcsr& c_matrix_a, dm_dbcsr& c_matrix_b,
                  double beta, dm_dbcsr& c_matrix_c){
    c_dbcsr_multiply_d(transa,transb,alpha,&c_matrix_a,&c_matrix_b,beta,&c_matrix_c,nullptr);
  }

  inline void multiply_eps(char transa, char transb, double alpha, dm_dbcsr& c_matrix_a, dm_dbcsr& c_matrix_b,
                  double beta, dm_dbcsr& c_matrix_c, double sthr){
    c_dbcsr_multiply_eps_d(transa,transb,alpha,&c_matrix_a,&c_matrix_b,beta,&c_matrix_c,sthr);
  }

  inline void get_stored_coordinates(dm_dbcsr& matrix, int row, int col, int* processor){
    c_dbcsr_get_stored_coordinates(matrix,row,col,processor);
  }

  inline void distribution_release(void*& dist){
    c_dbcsr_distribution_release(&dist);
  }

 
  inline void put_block(dm_dbcsr& matrix, int row, int col, double* block, int block_length){
    c_dbcsr_put_block_d(matrix,row,col,block,block_length);
  }

  inline void copy(dm_dbcsr& c_matrix_a, dm_dbcsr& c_matrix_b){
    c_dbcsr_copy_d(&c_matrix_a,&c_matrix_b);
  }

  inline void finalize_lib(MPI_Comm comm){
    MPI_Fint fcomm = MPI_Comm_c2f(comm);
    c_dbcsr_finalize_lib_aux(&fcomm);
  }

  inline void finalize_lib_silent(MPI_Comm comm){
    MPI_Fint fcomm = MPI_Comm_c2f(comm);
    c_dbcsr_finalize_lib_aux_silent(&fcomm);
  }

  inline void finalize(dm_dbcsr matrix){
    c_dbcsr_finalize(matrix);
  }

  inline void trace_ab(dm_dbcsr& mat_a, dm_dbcsr& mat_b, double& tr){
    c_dbcsr_trace_ab_d(&mat_a,&mat_b,tr);
  }

  inline void trace_a(dm_dbcsr& mat_a, double& tr){
    c_dbcsr_trace_a_d(&mat_a,tr);
  }

  inline void set_diag(dm_dbcsr& mat_a, double* diags, int dim){
    c_dbcsr_set_diag_d(&mat_a,diags,dim);
  }

  inline void set(dm_dbcsr& mat_a, double scl){
    c_dbcsr_set_d(&mat_a,scl);
  }

  inline void get_block(dm_dbcsr& mat_a, int row, int col, double* block, bool& found, int row_size, int col_size){
    c_dbcsr_get_block_d(&mat_a,row,col,block,found,row_size,col_size);
  }

  inline void filter(dm_dbcsr& mat_a, double eps){
    c_dbcsr_filter_d(&mat_a,eps);
  }

  inline void gershgorin_estimate(dm_dbcsr& mat, int* bdims, int nblocks, int tot_dim, double* sums, double* diags){
    c_dbcsr_gershgorin_estimate_d(&mat,bdims,nblocks,tot_dim,sums,diags);
  }

  inline void scale(dm_dbcsr& mat_a, double eps){
    c_dbcsr_scale_d(&mat_a,eps);
  }

  inline void print(dm_dbcsr& matrix){
    c_dbcsr_print(matrix);
  }

  inline void read(dm_dbcsr& matrix, char* cfname, void** fdist){
    c_dbcsr_read_d(&matrix,cfname,fdist);
  }

  inline void write(dm_dbcsr& matrix, char* cfname){
    c_dbcsr_write_d(&matrix,cfname);
  }

  inline void maxabs(dm_dbcsr& matrix, double& amv){
    c_dbcsr_maxabs_d(&matrix,&amv);
  }

}


/*
 * Class to hold MPI-setup for DBCSR
 */
class DBCSR_Environment {
  public:
    std::string env_id;
    int dbcsr_dims[2];
    int dbcsr_periods[2];
    int dbcsr_coords[2];
    int dbcsr_reorder;
    void* dbcsr_dist;
    MPI_Comm dbcsr_group;
    std::vector<int> dbcsr_row_dist;
    std::vector<int> dbcsr_col_dist;
    double dbcsr_default_thresh;

    int mpi_rank;
    int mpi_size;
    int nblk_row;
    int nblk_col;

    DBCSR_Environment(int nblk_row_in, int nblk_col_in, const std::string& id_in="dbcsr_env_default"){
      env_id = id_in;
      nblk_row = nblk_row_in;
      nblk_col = nblk_col_in;
      MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

      // Make 2D grid
      dbcsr_dims[0] = 0;
      dbcsr_dims[1] = 0;
      MPI_Dims_create(mpi_size, 2, dbcsr_dims);
      dbcsr_periods[0] = 1;
      dbcsr_periods[1] = 1;
      dbcsr_reorder = 0;
      MPI_Cart_create(MPI_COMM_WORLD, 2, dbcsr_dims, dbcsr_periods, dbcsr_reorder, &dbcsr_group);

      MPI_Cart_coords(dbcsr_group, mpi_rank, 2, dbcsr_coords);

      c_dbcsr_init_lib();

      dbcsr_row_dist.resize(nblk_row);
      dbcsr_col_dist.resize(nblk_col);

      for(int i=0; i < nblk_row; i++){
       dbcsr_row_dist[i] = (i+1) % dbcsr_dims[0];
      }
      for(int i=0; i < nblk_col; i++){
       dbcsr_col_dist[i] = (i+1) % dbcsr_dims[1];
      }

      dbcsr_dist = nullptr;

      int dr = (int)dbcsr_row_dist.size();
      int dc = (int)dbcsr_col_dist.size();
      MPI_Fint fcomm = MPI_Comm_c2f(dbcsr_group);
      c_dbcsr_distribution_new_aux(&dbcsr_dist, &fcomm,
                                   &dbcsr_row_dist[0], dr,
                                   &dbcsr_col_dist[0], dc);

    };

    ~DBCSR_Environment(){
      if (dbcsr_dist == nullptr) return;
      c_dbcsr_distribution_release(&dbcsr_dist);
      MPI_Fint fcomm = MPI_Comm_c2f(dbcsr_group);
    //if (verbose)
    //  c_dbcsr_finalize_lib_aux(&fcomm);
    //else
        c_dbcsr_finalize_lib_aux_silent(&fcomm);
      MPI_Comm_free(&dbcsr_group);
      MPI_Finalize();
    };

};

/*
 * Wrapper-class for dbcsr
 */
class DistBCSR {

  private:
    DBCSR_Environment* dbcsr_env;

    dm_dbcsr dbcsr_matrix;
    std::string mname;
    size_t nrow;
    size_t ncol;
    std::vector<int> row_dims;
    std::vector<int> col_dims;
    double dbcsr_thresh;

  public:
    DistBCSR(){
      dbcsr_matrix = nullptr;
      dbcsr_env    = nullptr;
    };

    DistBCSR(DBCSR_Environment* dbcsr_env_in, const std::string& mname_in="default matrix name"){
      mname = mname_in;
      dbcsr_env = dbcsr_env_in;
      dbcsr_matrix = nullptr;
      nrow = 0;
      ncol = 0;
    };

    DistBCSR(DistBCSR& ref){
      mname = ref.mname;
      dbcsr_env = ref.dbcsr_env;
      nrow = ref.nrow;
      ncol = ref.ncol;
      row_dims = ref.row_dims;
      col_dims = ref.col_dims;

      dbcsr_matrix = nullptr;
      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_env->dbcsr_dist, 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());
      c_dbcsr_finalize(dbcsr_matrix);
    };

    DistBCSR(const DistBCSR& ref){ ///< Constructor
      mname = ref.mname;
      dbcsr_env = ref.dbcsr_env;
      nrow = ref.nrow;
      ncol = ref.ncol;
      row_dims = ref.row_dims;
      col_dims = ref.col_dims;

      dbcsr_matrix = nullptr;
      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_env->dbcsr_dist, 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());
      c_dbcsr_finalize(dbcsr_matrix);
      this->copy(ref);
    };

    DistBCSR(size_t nrow_in, size_t ncol_in, std::vector<int>& row_dims_in, std::vector<int>& col_dims_in, DBCSR_Environment* dbcsr_env_in,
          const std::string& mname_in="default matrix name"){
      mname = mname_in;
      dbcsr_env = dbcsr_env_in;
      nrow = nrow_in;
      ncol = ncol_in;
      row_dims = row_dims_in;
      col_dims = col_dims_in;

      dbcsr_matrix = nullptr;
      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_env->dbcsr_dist, 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());
      c_dbcsr_finalize(dbcsr_matrix);

    };

    DistBCSR(size_t ldim, std::vector<int>& dims_in, DBCSR_Environment* dbcsr_env_in, bool add_zero_diag=false, const std::string& mname_in="default matrix name"){
      mname = mname_in;
      dbcsr_env = dbcsr_env_in;
      nrow = ldim;
      ncol = ldim;
      row_dims = dims_in;
      col_dims = dims_in;

      dbcsr_matrix = nullptr;
      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_env->dbcsr_dist, 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());

      if (add_zero_diag){
        int max_dim = *(std::max_element(row_dims.begin(),row_dims.end()));
        std::vector<double> block(max_dim*max_dim,0.e0);
        for(int i = 0; i < (int)row_dims.size(); i++){
          int blk_proc = -1;
          c_dbcsr_get_stored_coordinates(dbcsr_matrix, i, i, &blk_proc);

          if(blk_proc == dbcsr_env->mpi_rank){
            int idim = row_dims[i];
            c_dbcsr_put_block_d(dbcsr_matrix, i, i, block.data(), idim*idim);
          }
        }
      }

      c_dbcsr_finalize(dbcsr_matrix);

    };

    ~DistBCSR(){
       c_dbcsr_release(&dbcsr_matrix);
       dbcsr_matrix = nullptr;
    };

    // Load from dense array
    void load(double const* src, double cthr=-1.e0){
      if (dbcsr_matrix != nullptr){
        c_dbcsr_release(&(this->dbcsr_matrix));
        dbcsr_matrix = nullptr;
      }

      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_env->dbcsr_dist, 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());

      bool do_compress = (cthr > 0.e0);

      int ioff = 0;
      std::vector<double> block;
      for(int i = 0; i < (int)row_dims.size(); i++)
      {
          int idim = row_dims[i];
          int joff = 0;
          for(int j = 0; j < (int)col_dims.size(); j++)
          {
              int jdim = col_dims[j];
              int blk_proc = -1;
              c_dbcsr_get_stored_coordinates(dbcsr_matrix, i, j, &blk_proc);
  
              if(blk_proc == dbcsr_env->mpi_rank)
              {
                  block.resize(idim*jdim);
                  double sum = 0.e0;
                  for(int cc=0;cc<jdim;cc++){
                    for(int rr=0;rr<idim;rr++){
                      double ddd = src[rr+ioff+(cc+joff)*nrow];
                      block[rr+cc*idim] = ddd;
                      sum += ddd*ddd;
                    }
                  }
                  if (!do_compress || sqrt(sum/((double)(idim*jdim))) > cthr)
                    c_dbcsr_put_block_d(dbcsr_matrix, i, j, block.data(), block.size());
              }
              joff += jdim;
          }
          ioff += idim;
      }
      c_dbcsr_finalize(dbcsr_matrix);

    };

    void load(std::vector<double>& src, double cthr=-1.e0){
      this->load(src.data(),cthr);
    };

    // Collect matrix in local vector
    std::vector<double> gather(){
      assert(dbcsr_matrix != nullptr);
      std::vector<double> ret(nrow*ncol,0.e0);
      std::vector<double> loc(nrow*ncol,0.e0);

      std::vector<double> block;

      int ioff = 0;
      for(int i = 0; i < (int)row_dims.size(); i++)
      {
          int idim = row_dims[i];
          int joff = 0;
          for(int j = 0; j < (int)col_dims.size(); j++)
          {
              int jdim = col_dims[j];
              int blk_proc = -1;
              c_dbcsr_get_stored_coordinates(dbcsr_matrix, i, j, &blk_proc);
    
              if(blk_proc == dbcsr_env->mpi_rank)
              {
                  block.resize(idim*jdim);
                  std::fill(block.begin(),block.end(),0);
                  double* bptr = &block[0];
                  bool foundit = false;
                  c_dbcsr_get_block_d(&dbcsr_matrix,i,j,bptr,foundit,idim,jdim);
                  if (foundit){
                    for(int cc=0;cc<jdim;cc++){
                      for(int rr=0;rr<idim;rr++){
                        loc[rr+ioff+(cc+joff)*nrow] = block[rr+cc*idim];
                      }
                    }
                  }
              }
              joff += jdim;
          }
          ioff += idim;
      }

      size_t tot_dim = nrow * ncol;
      if (tot_dim > INT_MAX){
        size_t off = 0;
        int    idim_act = INT_MAX;
        while((off+idim_act) < tot_dim){
          idim_act = INT_MAX;
          if ((off+idim_act) > tot_dim) idim_act = (int)(tot_dim - off);
          MPI_Allreduce(loc.data() + off, ret.data() + off, idim_act,
                        MPI_DOUBLE, MPI_SUM, dbcsr_env->dbcsr_group);
          off += idim_act;
        }
      }else{
        int idim = (int) tot_dim;
        MPI_Allreduce(loc.data(), ret.data(), idim,
                      MPI_DOUBLE, MPI_SUM, dbcsr_env->dbcsr_group);
      }

      return ret;
    };

    void mult(char mA, char mB, const DistBCSR& A, const DistBCSR& B, double alpha=1.e0, double beta=0.e0, double cthr=-1.e0){
      if (cthr > 0.e0){
        c_dbcsr_multiply_eps_d(mA,mB,alpha,(dm_dbcsr*)&(A.dbcsr_matrix),(dm_dbcsr*)&(B.dbcsr_matrix),beta,&(this->dbcsr_matrix),cthr);
      }else{
        bool retain_sparsity = false; // not sure what that is...
        c_dbcsr_multiply_d(mA,mB,alpha,(dm_dbcsr*)&(A.dbcsr_matrix),(dm_dbcsr*)&(B.dbcsr_matrix),beta,&(this->dbcsr_matrix),&retain_sparsity);
      }
    };

    void copy(const DistBCSR& src){
      c_dbcsr_copy_d(&(this->dbcsr_matrix),(dm_dbcsr*)&(src.dbcsr_matrix));
    }
    
    void add(const DistBCSR& A){
      double done = 1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), (dm_dbcsr*)&(A.dbcsr_matrix), done, done);
    }

    void sub(const DistBCSR& A){
      double done = 1.e0;
      double mdone = -1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), (dm_dbcsr*)&(A.dbcsr_matrix), done, mdone);
    }

    void add(const DistBCSR& A, const DistBCSR& B){
      this->copy(A);
      double done = 1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), (dm_dbcsr*)&(B.dbcsr_matrix), done, done);
    }

    void sub(const DistBCSR& A, const DistBCSR& B){
      this->copy(A);
      double done = 1.e0;
      double mdone = -1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), (dm_dbcsr*)&(B.dbcsr_matrix), done, mdone);
    }

    void axpy(const DistBCSR& A, const double fac){
      double done = 1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), (dm_dbcsr*)&(A.dbcsr_matrix), done, fac);
    }

    void axpy(const DistBCSR& A, const DistBCSR& B, const double fac){
      this->copy(A);
      double done = 1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), (dm_dbcsr*)&(B.dbcsr_matrix), done, fac);
    }

    double trace(){
      assert(nrow == ncol);
      double ret = 0.e0;
      c_dbcsr_trace_a_d(&(this->dbcsr_matrix),ret);
      return ret;
    }

    void set_diag(const double scl){
      assert(nrow == ncol);
      int dim = this->nrow;
      std::vector<double> dvals(nrow,scl);
      c_dbcsr_set_diag_d(&(this->dbcsr_matrix),dvals.data(),dim);
    }

    void set_diag(double const* dvals){
      assert(nrow == ncol);
      int dim = this->nrow;
      c_dbcsr_set_diag_d(&(this->dbcsr_matrix),(double*)dvals,dim);
    }

    void set_diag(const std::vector<double>& dvals){
      this->set_diag(dvals.data());
    }

    void add_diag(double const* dvals){
      assert(nrow == ncol);
      int max_dim = *(std::max_element(row_dims.begin(),row_dims.end()));
      std::vector<double> block1(max_dim*max_dim,0.e0);
    
      int ioff = 0;
      for(int i = 0; i < (int)row_dims.size(); i++){
        int idim = row_dims[i];
        bool found1 = false;
        c_dbcsr_get_block_d(&(dbcsr_matrix), i, i, block1.data(), found1, idim, idim);
        if (found1){
          for(int ii=0;ii<idim;ii++) block1[ii+ii*idim] += dvals[ii+ioff];
          c_dbcsr_put_block_d(dbcsr_matrix, i, i, block1.data(), idim*idim);
        }
        ioff += idim;
      }
    }

    void add_diag(const std::vector<double>& dvals){
      this->add_diag(dvals.data());
    }

    void add_diag(const double scl){
      assert(nrow == ncol);
      int max_dim = *(std::max_element(row_dims.begin(),row_dims.end()));
      std::vector<double> block1(max_dim*max_dim,0.e0);
    
      for(int i = 0; i < (int)row_dims.size(); i++){
        int idim = row_dims[i];
        bool found1 = false;
        c_dbcsr_get_block_d(&(dbcsr_matrix), i, i, block1.data(), found1, idim, idim);
        if (found1){
          for(int ii=0;ii<idim;ii++) block1[ii+ii*idim] += scl;
          c_dbcsr_put_block_d(dbcsr_matrix, i, i, block1.data(), idim*idim);
        }
      }

    }

    void set(const double scl){
      c_dbcsr_set_d(&(this->dbcsr_matrix),scl);
    }

    void zero(){
      // best to remove old matrix and generate new one...
      if (dbcsr_matrix != nullptr){
        c_dbcsr_release(&(this->dbcsr_matrix));
        dbcsr_matrix = nullptr;
      }

      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_env->dbcsr_dist, 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());

      int max_dim = *(std::max_element(row_dims.begin(),row_dims.end()));
      std::vector<double> block(max_dim*max_dim,0.e0);
      for(int i = 0; i < (int)row_dims.size(); i++){
        int blk_proc = -1;
        c_dbcsr_get_stored_coordinates(dbcsr_matrix, i, i, &blk_proc);

        if(blk_proc == dbcsr_env->mpi_rank){
          int idim = row_dims[i];
          c_dbcsr_put_block_d(dbcsr_matrix, i, i, block.data(), idim*idim);
        }
      }
      

      c_dbcsr_finalize(dbcsr_matrix);

    }

    double dot(const DistBCSR& A){
      double tr = 0.e0;
      c_dbcsr_trace_ab_d(&(this->dbcsr_matrix),(dm_dbcsr*)&(A.dbcsr_matrix),tr);
      return tr;
    }

    void filter(double eps=-1.e0){
      if (eps < 0.e0) eps = dbcsr_thresh;
      c_dbcsr_filter_d(&(this->dbcsr_matrix),eps);
    }

    void scale(const double fac){
      c_dbcsr_scale_d(&(this->dbcsr_matrix),fac);
    }

    void load(const std::string& cfname){
      c_dbcsr_read_d(&(this->dbcsr_matrix),(char*)cfname.c_str(),&(dbcsr_env->dbcsr_dist));
    }

    void write(const std::string& cfname){
      c_dbcsr_write_d(&(this->dbcsr_matrix),(char*)cfname.c_str());
    }

    double maxabs(){
      double amv = 0.e0;
      c_dbcsr_maxabs_d(&(this->dbcsr_matrix),&amv);
      return amv;
    }

    void hadamard(const DistBCSR& rhs){
      int max_dimr = *(std::max_element(row_dims.begin(),row_dims.end()));
      int max_dimc = *(std::max_element(col_dims.begin(),col_dims.end()));

      std::vector<double> block1(max_dimr*max_dimc,0.e0);
      std::vector<double> block2(max_dimr*max_dimc,0.e0);
      
      size_t rowoff = 0;
      for(int irow = 0; irow < (int)row_dims.size(); irow++){
        int irowdim = row_dims[irow];
        size_t coloff = 0;
        for(int icol = irow; icol < (int)col_dims.size(); icol++){
          int icoldim = col_dims[icol];
          bool found1 = false;
          c_dbcsr_get_block_d(&(this->dbcsr_matrix), irow, icol, block1.data(), found1, irowdim, icoldim);
          if (found1){
            c_dbcsr_get_block_d((dm_dbcsr*)&(rhs.dbcsr_matrix), irow, icol, block2.data(), found1, irowdim, icoldim);
            if (!found1){
              printf("\n  Error: Different distribution for dbcsr?!?\n\n");
              exit(1);
            }
            size_t ctot = coloff;
            for(int col=0;col<icoldim;col++){
              size_t rtot = rowoff;
              for(int row=0;row<irowdim;row++){
                block1[row+col*irowdim] *= block2[row+col*irowdim];
                rtot++;
              }
              ctot++;
            }
            c_dbcsr_put_block_d(this->dbcsr_matrix, irow, icol, block1.data(), irowdim*icoldim);

          }
          coloff += icoldim;
        }
        rowoff += irowdim;
      }
    };

    void hadamard_inv(const DistBCSR& rhs){
      int max_dimr = *(std::max_element(row_dims.begin(),row_dims.end()));
      int max_dimc = *(std::max_element(col_dims.begin(),col_dims.end()));

      std::vector<double> block1(max_dimr*max_dimc,0.e0);
      std::vector<double> block2(max_dimr*max_dimc,0.e0);
      
      size_t rowoff = 0;
      for(int irow = 0; irow < (int)row_dims.size(); irow++){
        int irowdim = row_dims[irow];
        size_t coloff = 0;
        for(int icol = irow; icol < (int)col_dims.size(); icol++){
          int icoldim = col_dims[icol];
          bool found1 = false;
          c_dbcsr_get_block_d(&(this->dbcsr_matrix), irow, icol, block1.data(), found1, irowdim, icoldim);
          if (found1){
            c_dbcsr_get_block_d((dm_dbcsr*)&(rhs.dbcsr_matrix), irow, icol, block2.data(), found1, irowdim, icoldim);
            if (!found1){
              printf("\n  Error: Different distribution for dbcsr?!?\n\n");
              exit(1);
            }
            size_t ctot = coloff;
            for(int col=0;col<icoldim;col++){
              size_t rtot = rowoff;
              for(int row=0;row<irowdim;row++){
                double s2act = block2[row+col*irowdim];
                if (fabs(s2act) < 1e-13)
                  block1[row+col*irowdim]  = 0.e0;
                else
                  block1[row+col*irowdim] /= s2act;
                rtot++;
              }
              ctot++;
            }
            c_dbcsr_put_block_d(this->dbcsr_matrix, irow, icol, block1.data(), irowdim*icoldim);
          }
          coloff += icoldim;
        }
        rowoff += irowdim;
      }
    };

    void gershgorin_estimate(double& eps0, double& epsn){
      assert(nrow == ncol);
      std::vector<double> sums(nrow,0.e0);
      std::vector<double> diags(nrow,0.e0);
      std::vector<double> red_sums(nrow,0.e0);
      std::vector<double> red_diags(nrow,0.e0);
    
      dbcsr::gershgorin_estimate(this->dbcsr_matrix,row_dims.data(),col_dims.size(),(int)nrow,sums.data(),diags.data());
    
      // gather results
      MPI_Allreduce(sums.data(),red_sums.data(), ((int)nrow), MPI_DOUBLE, MPI_SUM, dbcsr_env->dbcsr_group);
      MPI_Allreduce(diags.data(),red_diags.data(), ((int)nrow), MPI_DOUBLE, MPI_SUM, dbcsr_env->dbcsr_group);
    
      double disc_min[2];
      disc_min[0] =  9999.e0;
      disc_min[1] = -9999.e0;
      double disc_max[2];
      disc_max[0] = -9999.e0;
      disc_max[1] = -9999.e0;
    
      for(size_t i=0;i<nrow;++i){
        // remove diagonal value from row-sum
        red_sums[i] -= fabs(red_diags[i]);
        if((disc_max[0] + disc_max[1]) < (red_diags[i] + red_sums[i])){
          disc_max[0] = red_diags[i];
          disc_max[1] = red_sums[i];
        }
        if((disc_min[0] - disc_min[1]) > (red_diags[i] - red_sums[i])){
          disc_min[0] = red_diags[i];
          disc_min[1] = red_sums[i];
        }
      }
      eps0 = disc_min[0] - disc_min[1];
      epsn = disc_max[0] + disc_max[1];

    }


    // todo: transpose,...

    // some operators...
    DistBCSR& operator=(const DistBCSR& rhs){
      if (dbcsr_matrix != nullptr){
        c_dbcsr_release(&dbcsr_matrix);
        dbcsr_matrix = nullptr;
      }

      mname = rhs.mname;
      dbcsr_env = rhs.dbcsr_env;
      nrow = rhs.nrow;
      ncol = rhs.ncol;
      row_dims = rhs.row_dims;
      col_dims = rhs.col_dims;

      dbcsr_matrix = nullptr;
      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_env->dbcsr_dist, 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());
      c_dbcsr_finalize(dbcsr_matrix);
      this->copy((DistBCSR&)rhs);
      return *this;
    };

    DistBCSR& operator=(DistBCSR&& rhs){
      if (dbcsr_matrix != nullptr){
        c_dbcsr_release(&dbcsr_matrix);
        dbcsr_matrix = nullptr;
      }
      mname = rhs.mname;
      dbcsr_env = rhs.dbcsr_env;
      nrow = rhs.nrow;
      ncol = rhs.ncol;
      row_dims = rhs.row_dims;
      col_dims = rhs.col_dims;

      dbcsr_matrix = nullptr;
      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_env->dbcsr_dist, 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());
      c_dbcsr_finalize(dbcsr_matrix);
      this->copy((DistBCSR&)rhs);
      return *this;
    };

    DistBCSR& operator+=(const DistBCSR& rhs){
      this->add((DistBCSR&)rhs);
      return *this;
    };

    DistBCSR& operator-=(const DistBCSR& rhs){
      this->sub((DistBCSR&)rhs);
      return *this;
    };

    DistBCSR& operator*=(const double& fac){
      this->scale(fac);
      return *this;
    };

    DistBCSR operator*(const double& fac){
      DistBCSR ret = *this;
      ret.scale(fac);
      return ret;
    };

    DistBCSR& operator/=(const double& fac){
      assert(fabs(fac) > 1e-13);
      this->scale(1.e0/fac);
      return *this;
    };

    DistBCSR operator+(const DistBCSR& rhs) const{
      DistBCSR ret = *this;
      ret.add(rhs);
      return std::move(ret);
    };

    DistBCSR operator-(const DistBCSR& rhs) const{
      DistBCSR ret = *this;
      ret.sub(rhs);
      return std::move(ret);
    };

    DistBCSR operator*(const DistBCSR& rhs) const{
      DistBCSR ret(*this);
      ret.mult('N','N',*this,rhs);
      return std::move(ret);
    };

    void print(const std::string& tit){
      if (dbcsr_env->mpi_rank == 0) printf("  %s\n",tit.c_str());
      MPI_Barrier(dbcsr_env->dbcsr_group);
      dbcsr::print(this->dbcsr_matrix);
    };

    dm_dbcsr& get_dbcsr(){return dbcsr_matrix;};
    DBCSR_Environment* get_env(){return dbcsr_env;};

    void      set_dbcsr(dm_dbcsr& din){
      if (dbcsr_matrix != nullptr){
        c_dbcsr_release(&dbcsr_matrix);
        dbcsr_matrix = nullptr;
      }
      dbcsr_matrix = din;
    };
    
    std::vector<int>& get_row_dims(){return row_dims;};
    std::vector<int>& get_col_dims(){return col_dims;};

    size_t get_nrow(){return nrow;};
    size_t get_ncol(){return ncol;};

};

#endif

