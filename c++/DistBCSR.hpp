#ifndef DBCSR_CLASS_H
#define DBCSR_CLASS_H

#include <dbcsr_c.h>

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
    int dbcsr_tot_dim;
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
      dbcsr_col_dist.resize(nblk_row);

      for(int i=0; i < nblk_row; i++){
       dbcsr_row_dist[i] = (i+1) % dbcsr_dims[0];
      }
      for(int i=0; i < nblk_col; i++){
       dbcsr_col_dist[i] = (i+1) % dbcsr_dims[1];
      }

      dbcsr_dist = nullptr;

      int dr = (int)dbcsr_row_dist.size();
      int dc = (int)dbcsr_col_dist.size();
      c_dbcsr_distribution_new(&dbcsr_dist, dbcsr_group,
                               &dbcsr_row_dist[0], dr,
                               &dbcsr_col_dist[0], dc);

    };

    ~DBCSR_Environment(){
      if (dbcsr_dist == nullptr) return;
      c_dbcsr_distribution_release(&dbcsr_dist);
      MPI_Fint fcomm = MPI_Comm_c2f(dbcsr_group);
      if (verbose)
        c_dbcsr_finalize_lib_aux(&fcomm);
      else
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
    DistBCSR(DBCSR_Environment* dbcsr_renv_in, const std::string mname_in="default matrix name"){
      mname = mname_in;
      dbcsr_env = dbcsr_env_in;

    };

    DistBCSR(DistBCSR& ref){
      mname = ref.mname_in;
      dbcsr_env = ref.dbcsr_env_in;
      nrow = ref.nrow_in;
      ncol = ref.ncol_in;
      row_dims = ref.row_dims_in;
      col_dims = ref.col_dims_in;

      dbcsr_matrix = nullptr;
      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_dist, 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());
      c_dbcsr_finalize(dbcsr_matrix);
    };

    DistBCSR(size_t nrow_in, size_t ncol_int, std::vector<int>& row_dims_in, std::vector<int>& col_dims_in, DBCSR_Environment* dbcsr_renv_in,
          const std::string mname_in="default matrix name"){
      mname = mname_in;
      dbcsr_env = dbcsr_env_in;
      nrow = nrow_in;
      ncol = ncol_in;
      row_dims = row_dims_in;
      col_dims = col_dims_in;

      dbcsr_matrix = nullptr;
      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_dist, 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());
      c_dbcsr_finalize(dbcsr_matrix);

    };

    DistBCSR(size_t ldim, std::vector<int>& dims_in, bool add_zero_diag=false, const std::string mname_in="default matrix name"){
      mname = mname_in;
      dbcsr_env = dbcsr_env_in;
      nrow = ldim;
      ncol = ldim;
      row_dims = dims_in;
      col_dims = dims_in;

      dbcsr_matrix = nullptr;
      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_dist, 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());

      if (add_zero_diag){
        int max_dim = *(std::max_element(row_dims.begin(),row_dims.end()));
        std::vector<double> block(max_dim*max_dim,0.e0);
        for(int i = 0; i < (int)row_dims.size(); i++){
          int blk_proc = -1;
          c_dbcsr_get_stored_coordinates(dbcsr_matrix, i, i, &blk_proc);

          if(blk_proc == dbcsr_env.mpi_rank){
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
      assert(dbcsr_matrix != nullptr);
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
  
              if(blk_proc == dbcsr_env.mpi_rank)
              {
                  block.resize(idim*jdim);
                  double sum = 0.e0;
                  for(int cc=0;cc<jdim;cc++){
                    for(int rr=0;rr<idim;rr++){
                      double ddd = src[rr+ioff+(cc+joff)*ldim];
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

    };

    // Collect matrix in local vector
    std::vector<double> gather(){
      std::vector<double> ret(nrow*ncol,0.e0);
      std::vector<double> loc(nrow*ncol,0.e0);

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
    
              if(blk_proc == mpi_rank)
              {
                  block.resize(idim*jdim);
                  bzero(&block[0],idim*jdim*sizeof(double));
                  double* bptr = &block[0];
                  bool foundit = false;
                  c_dbcsr_get_block_d(&dbcsr_matrix,i,j,bptr,foundit,idim,jdim);
                  if (foundit){
                    for(int cc=0;cc<jdim;cc++){
                      for(int rr=0;rr<idim;rr++){
                        loc[rr+ioff+(cc+joff)*ldim] = block[rr+cc*idim];
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
                        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          off += idim_act;
        }
      }else{
        int idim = (int) tot_dim;
        MPI_Allreduce(loc.data(), ret.data(), idim,
                      MPI_DOUBLE, MPI_SUM, LA4HPC_globals.dbcsr_group);
      }

      return ret;
    };

    void mult(char mA, char mB, DistBCSR& A, DistBCSR& B, double alpha=1.e0, double beta=0.e0, double cthr=-1.e0){
      if (sthr > 0.e0){
        c_dbcsr_multiply_eps_d(mA,mB,alpha,&(A.dbcsr_matrix),&(B.dbcsr_matrix),beta,&(C.dbcsr_matrix),sthr);
      }else{
        bool retain_sparsity = false; // not sure what that is...
        c_dbcsr_multiply_d(mA,mB,alpha,&(A.dbcsr_matrix),&(B.dbcsr_matrix),beta,&(C.dbcsr_matrix),&retain_sparsity);
      }
    };

    void copy(DistBCSR& src){
      c_dbcsr_copy_d(&(this->dbcsr_matrix),&(src.dbcsr_matrix));
    }
    
    void add(DistBCSR& A){
      double done = 1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), &(A.dbcsr_matrix), done, done);
    }

    void sub(DistBCSR& A){
      double done = 1.e0;
      double mdone = -1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), &(A.dbcsr_matrix), done, mdone);
    }

    void add(DistBCSR& A, DistBCSR& B){
      this->copy(A);
      double done = 1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), &(B.dbcsr_matrix), done, done);
    }

    void sub(DistBCSR& A, DistBCSR& B){
      this->copy(A);
      double done = 1.e0;
      double mdone = -1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), &(B.dbcsr_matrix), done, mdone);
    }

    void axpy(DistBCSR& A, double fac){
      double done = 1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), &(A.dbcsr_matrix), done, fac);
    }

    void axpy(DistBCSR& A, DistBCSR& B, double fac){
      this->copy(A);
      double done = 1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), &(B.dbcsr_matrix), done, fac);
    }

    double trace(){
      assert(nrow == ncol);
      double ret = 0.e0;
      c_dbcsr_trace_a_d(&(this->dbcsr_matrix),ret);
      return ret;
    }

    void set_diag(dm_dbcsr& mat_a, double scl){
      assert(nrow == ncol);
      int dim = this->nrow;
      std::vector<double> dvals(nrow,scl);
      c_dbcsr_set_diag_d(&(this->dbcsr_matrix),dvals.data(),dim);
    }

    void set_values(double scl){
      c_dbcsr_set_d(&(this->dbcsr_matrix),scl);
    }

    void zero(){
      // best to remove old matrix and generate new one...
      c_dbcsr_release(&(this->dbcsr_matrix));
      dbcsr_matrix = nullptr;
      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_dist, 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());

      int max_dim = *(std::max_element(row_dims.begin(),row_dims.end()));
      std::vector<double> block(max_dim*max_dim,0.e0);
      for(int i = 0; i < (int)row_dims.size(); i++){
        int blk_proc = -1;
        c_dbcsr_get_stored_coordinates(dbcsr_matrix, i, i, &blk_proc);

        if(blk_proc == dbcsr_env.mpi_rank){
          int idim = row_dims[i];
          c_dbcsr_put_block_d(dbcsr_matrix, i, i, block.data(), idim*idim);
        }
      }
      

      c_dbcsr_finalize(dbcsr_matrix);

    }

    double dot(DistBCSR& A){
      double tr = 0.e0;
      c_dbcsr_trace_ab_d(&(this->dbcsr_matrix),&(A.dbcsr_matrix),tr);
      return tr;
    }

    void filter(double eps=-1.e0){
      if (eps < 0.e0) eps = dbcsr_thresh;
      c_dbcsr_filter_d(&(this->dbcsr_matrix),eps);
    }

    void scale(double fac){
      c_dbcsr_scale_d(&(this->dbcsr_matrix),fac);
    }

    void load(const std::string& cfname){
      c_dbcsr_read_d(&(this->dbcsr_matrix),(char*)cfname.c_str(),&(dbcsr_env.dbcsr_dist));
    }

    void write(const std::string& cfname){
      c_dbcsr_write_d(&(this->dbcsr_matrix),(char*)cfname.c_str());
    }

    double maxabs(){
      double amv = 0.e0;
      c_dbcsr_maxabs_d(&(this->dbcsr_matrix),&amv);
      return amv;
    }

    void hadamard(DistBCSR& rhs){
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
            c_dbcsr_get_block_d(&(rhs.dbcsr_matrix), irow, icol, block2.data(), found1, irowdim, icoldim);
            if (!found1) die("Different distribution for dbcsr?!?");
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

    void hadamard_inv(DistBCSR& rhs){
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
            c_dbcsr_get_block_d(&(rhs.dbcsr_matrix), irow, icol, block2.data(), found1, irowdim, icoldim);
            if (!found1) die("Different distribution for dbcsr?!?");
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
    // todo: transpose,...

    // some operators...
    DistBCSR& operator=(const DistBCSR& rhs){
      mname = rhs.mname_in;
      dbcsr_env = rhs.dbcsr_env_in;
      nrow = rhs.nrow_in;
      ncol = rhs.ncol_in;
      row_dims = rhs.row_dims_in;
      col_dims = rhs.col_dims_in;

      dbcsr_matrix = nullptr;
      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_dist, 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());
      c_dbcsr_finalize(dbcsr_matrix);
      this->copy(rhs);
      return *this;
    };

    DistBCSR& operator=(DistBCSR&& rhs){
      mname = rhs.mname_in;
      dbcsr_env = rhs.dbcsr_env_in;
      nrow = rhs.nrow_in;
      ncol = rhs.ncol_in;
      row_dims = rhs.row_dims_in;
      col_dims = rhs.col_dims_in;

      dbcsr_matrix = nullptr;
      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_dist, 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());
      c_dbcsr_finalize(dbcsr_matrix);
      this->copy(rhs);
      return *this;
    };

    DistBCSR& operator+=(const DistBCSR& rhs){
      this->add(rhs);
      return *this;
    };

    DistBCSR& operator-=(const DistBCSR& rhs){
      this->sub(rhs);
      return *this;
    };

    DistBCSR& operator*=(const double& fac){
      this->scale(fac);
      return *this;
    };

    DistBCSR operator*(const double& fac){
      DistBCSR ret = *this;
      ret.scale(fac);
      return std::move(ret);
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


};

#endif

