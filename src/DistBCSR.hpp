#ifndef DISTBCSR_H
#define DISTBCSR_H

#include <limits>
#include <memory>
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

  void c_dbcsr_remove_blk_d(dm_dbcsr* matrix, int row, int col, int block_dim);

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

  inline void remove_blk(dm_dbcsr& mat_a, int row, int col, int block_dim){
    c_dbcsr_remove_blk_d(&mat_a,row,col,block_dim);
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


/*! \brief Class to hold specific MPI-setup for DistBCSR
 * 
 *  'DBCSR_Environment' holds the required MPI-information (comm-group, node-distribution)
 *  for a specific DistBCSR matrix-setup.
 *  It can not be modified by objects of 'DistBCSR'.
 */
class DBCSR_Environment {

  private:

    std::string env_id;              ///< ID-string for MPI-setup
    std::string dist_pattern;        ///< Node distribution pattern
    int dbcsr_dims[2];               ///< # of nodes in 2D-grid
    int dbcsr_periods[2];            ///< Holds periodic-flags of 2D-grid (true)
    int dbcsr_coords[2];             ///< Holds cartesian coordinates of this process
    int dbcsr_reorder;               ///< reorder-flag (false)
    void* dbcsr_dist;                ///< Handle to dbcsr-distribution (FORTRAN)
    MPI_Comm dbcsr_group;            ///< MPI communication group
    std::vector<int> dbcsr_row_dist; ///< table of node-distribution (rows)
    std::vector<int> dbcsr_col_dist; ///< table of node-distribution (columns)

    int mpi_rank;                    ///< rank of this process
    int mpi_size;                    ///< # of mpi-processes
    int nblk_row;                    ///< # of blocks (row)
    int nblk_col;                    ///< # of blocks (columns)

  public:
    
    /*! \brief Constructor
     * Requires the number of blocks for rows and columns.
     * Sets up the comm-group and dbcsr-distribution.
     * 
     *  @param nlk_row_in # of row-blocks in 2D MPI-grid
     *  @param nlk_col_in # of column-blocks in 2D MPI-grid
     *  @param id_in string-identifier
     *  @param pattern ID of distribution pattern
     *  @param mpi_comm MPI-communicator
     *
     */
    DBCSR_Environment(const int nblk_row_in, const int nblk_col_in, const std::string& id_in="dbcsr_env_default",
                      const std::string& pattern="default", MPI_Comm mpi_comm=MPI_COMM_WORLD){
      env_id = id_in;
      dist_pattern = pattern;
      nblk_row = nblk_row_in;
      nblk_col = nblk_col_in;
      MPI_Comm_size(mpi_comm, &mpi_size);
      MPI_Comm_rank(mpi_comm, &mpi_rank);

      // Make 2D grid
      dbcsr_dims[0] = 0;
      dbcsr_dims[1] = 0;
      MPI_Dims_create(mpi_size, 2, dbcsr_dims);
      dbcsr_periods[0] = 1;
      dbcsr_periods[1] = 1;
      dbcsr_reorder = 0;
      MPI_Cart_create(mpi_comm, 2, dbcsr_dims, dbcsr_periods, dbcsr_reorder, &dbcsr_group);

      MPI_Cart_coords(dbcsr_group, mpi_rank, 2, dbcsr_coords);

      c_dbcsr_init_lib();

      dbcsr_row_dist.resize(nblk_row);
      dbcsr_col_dist.resize(nblk_col);

      if (dist_pattern == "default"){
        for(int i=0; i < nblk_row; i++){
         dbcsr_row_dist[i] = (i+1) % dbcsr_dims[0];
        }
        for(int i=0; i < nblk_col; i++){
         dbcsr_col_dist[i] = (i+1) % dbcsr_dims[1];
        }
      }else{
        printf("\n  Error: illegal option for distribution-pattern!\n\n");
        exit(1);
      }

      dbcsr_dist = nullptr;

      int dr = (int)dbcsr_row_dist.size();
      int dc = (int)dbcsr_col_dist.size();
      MPI_Fint fcomm = MPI_Comm_c2f(dbcsr_group);
      c_dbcsr_distribution_new_aux(&dbcsr_dist, &fcomm,
                                   &dbcsr_row_dist[0], dr,
                                   &dbcsr_col_dist[0], dc);

    };

    /*! \brief Destructor
     * Release comm/dist by calling 'free'. Must be called before 'MPI_Finalize()'
     */
    ~DBCSR_Environment(){
      if (dbcsr_dist == nullptr) return;
      c_dbcsr_distribution_release((void**)&dbcsr_dist);
      dbcsr_dist = nullptr;
      MPI_Fint fcomm = MPI_Comm_c2f(dbcsr_group);
      c_dbcsr_finalize_lib_aux_silent(&fcomm);
      MPI_Comm_free(&dbcsr_group);
    };

    /*! \brief Access-functions
     */
    std::string      get_id()       const {return env_id;};
    void*            get_dist()     const {return dbcsr_dist;};
    void**           get_dist_ptr() const {return ((void**)&dbcsr_dist);};
    int              get_rank()     const {return mpi_rank;};
    int              get_size()     const {return mpi_size;};
    MPI_Comm         get_comm()     const {return dbcsr_group;};
    int              get_nblk_row() const {return nblk_row;};
    int              get_nblk_col() const {return nblk_col;};
    std::vector<int> get_row_dist() const {return dbcsr_row_dist;};
    std::vector<int> get_col_dist() const {return dbcsr_col_dist;};
    int              get_dims(int which) const {
      assert(which == 0 || which == 1);
      return dbcsr_dims[which];
    }
    int              get_coords(int which) const {
      assert(which == 0 || which == 1);
      return dbcsr_coords[which];
    }

};

/*! \brief Wrapper-class for dbcsr
 *
 * A self-contained class that provides an interface to dbcsr-matrices.
 * Requires a valid object of DBCSR_Environment that holds a distribution
 * in accordance to the blocking-structure.
 *
 */
class DistBCSR {

  private:
    std::shared_ptr<const DBCSR_Environment> dbcsr_env; ///< const/shared pointer to MPI-setup

    dm_dbcsr dbcsr_matrix;     ///< Handle to FORTRAN-object of 'dbcsr'
    std::string mname;         ///< Name of matrix
    size_t nrow;               ///< # of rows in dense matrix
    size_t ncol;               ///< # of columns in dense matrix
    std::vector<int> row_dims; ///< block-dimensions (row)
    std::vector<int> col_dims; ///< block-dimensions (columns)
    double dbcsr_thresh;       ///< compression threshold (filter/recompress)

  public:
    /*! \brief Default constructor
     *
     *  Before usage, the matrix must be initialized (see 'init()').
     *  Allows the use of regular class-objects as members of another class.
     */
    DistBCSR(){
      dbcsr_matrix = nullptr;
      this->free();
    };

    /*! \brief Constructor from reference
     */
    DistBCSR(DistBCSR& ref){
      mname = ref.mname;
      dbcsr_env = ref.dbcsr_env;
      nrow = ref.nrow;
      ncol = ref.ncol;
      row_dims = ref.row_dims;
      col_dims = ref.col_dims;
      dbcsr_thresh = ref.dbcsr_thresh;

      dbcsr_matrix = nullptr;
      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_env->get_dist(), 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());
      c_dbcsr_finalize(dbcsr_matrix);
      this->copy(ref);
    };

    /*! \brief Constructor from const-reference
     */
    DistBCSR(const DistBCSR& ref){
      assert(ref.dbcsr_env != nullptr);
      mname = ref.mname;
      mname += ":copy";
      dbcsr_env = ref.dbcsr_env;
      nrow = ref.nrow;
      ncol = ref.ncol;
      row_dims = ref.row_dims;
      col_dims = ref.col_dims;
      dbcsr_thresh = ref.dbcsr_thresh;

      dbcsr_matrix = nullptr;
      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_env->get_dist(), 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());
      c_dbcsr_finalize(dbcsr_matrix);
      this->copy(ref);
    };

    /*! \brief Move-constructor
     */
    DistBCSR(DistBCSR&& ref){ ///< move Constructor
      assert(ref.dbcsr_env != nullptr);
      mname = ref.mname;
      mname += ":copy";
      dbcsr_env = ref.dbcsr_env;
      nrow = ref.nrow;
      ncol = ref.ncol;
      row_dims = ref.row_dims;
      col_dims = ref.col_dims;
      dbcsr_thresh = ref.dbcsr_thresh;

      dbcsr_matrix = nullptr;
      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_env->get_dist(), 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());
      c_dbcsr_finalize(dbcsr_matrix);
      this->copy(ref);
      // free ref...
      ref.free();
    };


    /*! \brief Constructor w/ initialization
     * Allows to create non-square matrices.
     *
     * @param nrow_in # of rows/columns in dense matrix
     * @param ncol_in # of rows/columns in dense matrix
     * @param row_dims_in dimensions of blocks in rows
     * @param col_dims_in dimensions of blocks in columns
     * @param dbcsr_env DBCSR MPI-environment
     * @param thr_in sparse compression threshold
     * @param mname_in matrix-ID
     *
     */
    DistBCSR(const size_t nrow_in, const size_t ncol_in, const std::vector<int>& row_dims_in, const std::vector<int>& col_dims_in,
             std::shared_ptr<const DBCSR_Environment> dbcsr_env_in, const double thr_in=0.e0, const std::string& mname_in="default matrix name"){
      this->init(nrow_in,ncol_in,row_dims_in,col_dims_in,dbcsr_env_in,thr_in,false,mname_in);
    };

    /*! \brief Constructor w/ initialization of square matrix
     *
     * @param ldim leading dimension of square matrix
     * @param dims_in dimensions of blocks in rows/columns
     * @param dbcsr_env DBCSR MPI-environment
     * @param thr_in sparse compression threshold
     * @param mname_in matrix-ID
     *
     */
    DistBCSR(const size_t ldim, const std::vector<int>& dims_in, std::shared_ptr<const DBCSR_Environment> dbcsr_env_in, const double thr_in=0.e0,
             const bool add_zero_diag=false, const std::string& mname_in="default matrix name"){
      this->init(ldim,dims_in,dbcsr_env_in,thr_in,add_zero_diag,mname_in);
    };

    /*! \brief Initialization (non-square matrix)
     *
     * @param nrow_in # of rows/columns in dense matrix
     * @param ncol_in # of rows/columns in dense matrix
     * @param row_dims_in dimensions of blocks in rows
     * @param col_dims_in dimensions of blocks in columns
     * @param dbcsr_env DBCSR MPI-environment
     * @param thr_in sparse compression threshold
     * @param mname_in matrix-ID
     *
     */
    void init(const size_t nrow_in, const size_t ncol_in, const std::vector<int>& row_dims_in, const std::vector<int>& col_dims_in,
              std::shared_ptr<const DBCSR_Environment> dbcsr_env_in, const double thr_in=0.e0, const bool add_zero_diag=false,
              const std::string& mname_in="default matrix name"){
      assert(dbcsr_env_in->get_id() != "invalid");
      mname = mname_in;
      dbcsr_env = dbcsr_env_in;
      nrow = nrow_in;
      ncol = ncol_in;
      row_dims = row_dims_in;
      col_dims = col_dims_in;
      dbcsr_thresh = thr_in;

      dbcsr_matrix = nullptr;
      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_env->get_dist(), 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());
      if (add_zero_diag){ // only valid for square matrices...
        bool roweqcol = false;
        if (row_dims.size() == col_dims.size()){
          roweqcol = true;
          for(size_t ii=0;ii<row_dims.size();ii++){
            if (row_dims[ii] != col_dims[ii]){
              roweqcol = false;
              break;
            }
          }
        }
        if (!roweqcol){
          printf("\n  Error: zero-diag only for square matrices\n\n");
          exit(1);
        }
        int max_dim = *(std::max_element(row_dims.begin(),row_dims.end()));
        std::vector<double> block(max_dim*max_dim,0.e0);
        for(int i = 0; i < (int)row_dims.size(); i++){
          int blk_proc = -1;
          c_dbcsr_get_stored_coordinates(dbcsr_matrix, i, i, &blk_proc);

          if(blk_proc == dbcsr_env->get_rank()){
            int idim = row_dims[i];
            c_dbcsr_put_block_d(dbcsr_matrix, i, i, block.data(), idim*idim);
          }
        }
      }
      c_dbcsr_finalize(dbcsr_matrix);

    };

    /*! \brief Initialization (square matrix)
     *
     * @param ldim leading dimension of square matrix
     * @param dims_in dimensions of blocks in rows/columns
     * @param dbcsr_env DBCSR MPI-environment
     * @param thr_in sparse compression threshold
     * @param mname_in matrix-ID
     *
     */
    void init(const size_t ldim, const std::vector<int>& block_dims, std::shared_ptr<const DBCSR_Environment> dbcsr_env_in,
              const double thr_in=0.e0, const bool add_zero_diag=false, const std::string& mname_in="default matrix name"){
      this->init(ldim,ldim,block_dims,block_dims,dbcsr_env_in,thr_in,add_zero_diag,mname_in);
    };

    /*! \brief Destructor
     */
    ~DistBCSR(){
      this->free();
    };

    /*! \brief Release object
     */
    void free(){
       if (dbcsr_matrix != nullptr){
         c_dbcsr_release(&dbcsr_matrix);
       }
       dbcsr_matrix = nullptr;
       nrow = 0L;
       ncol = 0L;
       row_dims.clear();
       col_dims.clear();
       dbcsr_thresh = 0.e0;
       dbcsr_env = nullptr;
    };

    /*! \brief Load values from dense/local array
     *
     * @param src values of dense matrix of appropriate dimension
     * @param cthr sparse compression threshold
     *
     */
    void load(double const* src, const double cthr=-1.e0){
      assert(this->dbcsr_env != nullptr);
      if (dbcsr_matrix != nullptr){
        c_dbcsr_release(&(this->dbcsr_matrix));
        dbcsr_matrix = nullptr;
      }

      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_env->get_dist(), 'N',
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
  
              if(blk_proc == dbcsr_env->get_rank())
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
                    c_dbcsr_put_block_d(dbcsr_matrix, i, j, block.data(), (int)block.size());
              }
              joff += jdim;
          }
          ioff += idim;
      }
      c_dbcsr_finalize(dbcsr_matrix);

    };

    /*! \brief Load values from dense/local array
     *
     * @param src values of dense matrix of appropriate dimension
     * @param cthr sparse compression threshold
     *
     */
    void load(const std::vector<double>& src, const double cthr=-1.e0){
      assert(src.size() >= nrow*ncol);
      this->load(src.data(),cthr);
    };

    /*! \brief Gather distributed matrix in dense/local array
     */
    std::vector<double> gather(){
      assert(this->dbcsr_env != nullptr);
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
    
              if(blk_proc == dbcsr_env->get_rank())
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
      if (tot_dim > std::numeric_limits<int>::max()){
        size_t off = 0;
        int    idim_act = std::numeric_limits<int>::max();
        while((off+idim_act) < tot_dim){
          idim_act = std::numeric_limits<int>::max();
          if ((off+idim_act) > tot_dim) idim_act = (int)(tot_dim - off);
          MPI_Allreduce(loc.data() + off, ret.data() + off, idim_act,
                        MPI_DOUBLE, MPI_SUM, dbcsr_env->get_comm());
          off += idim_act;
        }
      }else{
        int idim = (int) tot_dim;
        MPI_Allreduce(loc.data(), ret.data(), idim,
                      MPI_DOUBLE, MPI_SUM, dbcsr_env->get_comm());
      }

      return ret;
    };

    /*! \brief Matrix multiplication
     *
     *  this = beta * this + alpha * op(A,mA) \times op(B,mB)
     *
     * @param mA op(A), 'N' or 'T'
     * @param mB op(B), 'N' or 'T'
     * @param A input matrix (lhs)
     * @param B input matrix (rhs)
     * @param alpha scaling-factor for product
     * @param beta  scaling-factor for (initial) output-matrix
     * @param cthr sparse compression threshold
     *
     */
    void mult(const char mA, const char mB, const DistBCSR& A, const DistBCSR& B, const double alpha=1.e0, const double beta=0.e0, const double cthr=-1.e0){
      assert(this->dbcsr_env != nullptr);
      assert(A.dbcsr_env != nullptr);
      assert(B.dbcsr_env != nullptr);
      if (cthr > 0.e0){
        c_dbcsr_multiply_eps_d(mA,mB,alpha,(dm_dbcsr*)&(A.dbcsr_matrix),(dm_dbcsr*)&(B.dbcsr_matrix),beta,&(this->dbcsr_matrix),cthr);
      }else{
        bool retain_sparsity = false; // not sure what that is...
        c_dbcsr_multiply_d(mA,mB,alpha,(dm_dbcsr*)&(A.dbcsr_matrix),(dm_dbcsr*)&(B.dbcsr_matrix),beta,&(this->dbcsr_matrix),&retain_sparsity);
      }
    };

    /*! \brief Symv
     *
     *  y = beta * y + alpha * A \times x
     *
     * @param x vector
     * @param alpha scaling factor
     * @param y vector [output]
     * @param beta scaling factor
     *
     */
    void symv(double const* x, const double alpha, double* y, const double beta) const {
      assert(this->nrow == this->ncol);
      if (fabs(beta) < 1e-13) 
        for(size_t ii=0;ii<nrow;ii++) y[ii] = 0.e0;
      else
        for(size_t ii=0;ii<nrow;ii++) y[ii] *= beta;
      // increment y by A times x
      std::vector<double> Ax_loc(nrow,0.e0);
      std::vector<double> Ax(nrow,0.e0);

      size_t rowoff = 0;
      for(int irow = 0; irow < (int)row_dims.size(); irow++){
        int irowdim = row_dims[irow];
        size_t coloff = 0;
        for(int icol = 0; icol < (int)col_dims.size(); icol++){
          int icoldim = col_dims[icol];
          auto block = this->get_block(irow,icol);
          if (block.size() > 0){
            size_t ctot = coloff;
            for(int col=0;col<icoldim;col++){
              size_t rtot = rowoff;
              double* acol = block.data() + col*irowdim;
              for(int row=0;row<irowdim;row++){
                Ax_loc[rtot] += acol[row] * x[ctot];
                rtot++;
              }
              ctot++;
            }
          }
          coloff += icoldim;
        }
        rowoff += irowdim;
      }
      MPI_Allreduce(Ax_loc.data(),Ax.data(), ((int)nrow), MPI_DOUBLE, MPI_SUM, dbcsr_env->get_comm());

      for(size_t ii=0;ii<nrow;ii++) y[ii] += alpha*Ax[ii];
    };

    /*! \brief Symv
     *
     *  y = beta * y + alpha * A \times x
     *
     * @param x vector
     * @param alpha scaling factor
     * @param y vector [output]
     * @param beta scaling factor
     *
     */
    void symv(const std::vector<double>& x, const double alpha, std::vector<double>& y, const double beta) const {
      this->symv(x.data(),alpha,y.data(),beta);
    }

    /*! \brief Copy matrix
     *
     * Copy from matrix 'src'
     *
     * @param src input matrix
     *
     */
    void copy(const DistBCSR& src){
      assert(this->dbcsr_env != nullptr);
      assert(src.dbcsr_env != nullptr);
      c_dbcsr_copy_d(&(this->dbcsr_matrix),(dm_dbcsr*)&(src.dbcsr_matrix));
    }
    
    /*! \brief Add matrix
     *
     * this += A
     *
     * @param A input matrix
     *
     */
    void add(const DistBCSR& A){
      assert(this->dbcsr_env != nullptr);
      assert(A.dbcsr_env != nullptr);
      double done = 1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), (dm_dbcsr*)&(A.dbcsr_matrix), done, done);
    }

    /*! \brief Subtract matrix
     *
     * this -= A
     *
     * @param A input matrix
     *
     */
    void sub(const DistBCSR& A){
      assert(this->dbcsr_env != nullptr);
      assert(A.dbcsr_env != nullptr);
      double done = 1.e0;
      double mdone = -1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), (dm_dbcsr*)&(A.dbcsr_matrix), done, mdone);
    }

    /*! \brief Add matrices
     *
     * this = A + B
     *
     * @param A input matrix
     * @param B input matrix
     *
     */
    void add(const DistBCSR& A, const DistBCSR& B){
      assert(this->dbcsr_env != nullptr);
      assert(A.dbcsr_env != nullptr);
      assert(B.dbcsr_env != nullptr);
      this->copy(A);
      double done = 1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), (dm_dbcsr*)&(B.dbcsr_matrix), done, done);
    }

    /*! \brief Subtract matrices
     *
     * this = A - B
     *
     * @param A input matrix
     * @param B input matrix
     *
     */
    void sub(const DistBCSR& A, const DistBCSR& B){
      assert(this->dbcsr_env != nullptr);
      assert(A.dbcsr_env != nullptr);
      assert(B.dbcsr_env != nullptr);
      this->copy(A);
      double done = 1.e0;
      double mdone = -1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), (dm_dbcsr*)&(B.dbcsr_matrix), done, mdone);
    }

    /*! \brief Axpy
     *
     * this += fac * A
     *
     * @param A input matrix
     * @param fac scaling factor
     *
     */
    void axpy(const DistBCSR& A, const double fac){
      assert(this->dbcsr_env != nullptr);
      assert(A.dbcsr_env != nullptr);
      double done = 1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), (dm_dbcsr*)&(A.dbcsr_matrix), done, fac);
    }

    /*! \brief Axpy
     *
     * this = A + fac * B
     *
     * @param A input matrix
     * @param B input matrix
     * @param fac scaling factor
     *
     */
    void axpy(const DistBCSR& A, const DistBCSR& B, const double fac){
      this->copy(A);
      double done = 1.e0;
      c_dbcsr_add_d(&(this->dbcsr_matrix), (dm_dbcsr*)&(B.dbcsr_matrix), done, fac);
    }

    /*! \brief Trace
     */
    double trace() const {
      assert(this->dbcsr_env != nullptr);
      assert(nrow == ncol);
      double ret = 0.e0;
      c_dbcsr_trace_a_d((void**)&(this->dbcsr_matrix),ret);
      return ret;
    }

    /*! \brief Set diagonal elements
     *
     * @param val diagonal value
     *
     */
    void set_diag(const double val){
      assert(this->dbcsr_env != nullptr);
      assert(nrow == ncol);
      int dim = (int)this->nrow;
      std::vector<double> dvals(nrow,val);
      c_dbcsr_set_diag_d(&(this->dbcsr_matrix),dvals.data(),dim);
    }

    /*! \brief Set diagonal elements
     *
     * @param dvals diagonal values
     *
     */
    void set_diag(double const* dvals){
      assert(this->dbcsr_env != nullptr);
      assert(this->nrow == this->ncol);
      int dim = (int)this->nrow;
      c_dbcsr_set_diag_d(&(this->dbcsr_matrix),(double*)dvals,dim);
    }

    /*! \brief Set diagonal elements
     *
     * @param dvals diagonal values
     *
     */
    void set_diag(const std::vector<double>& dvals){
      assert(this->nrow == this->ncol);
      assert(this->nrow <= dvals.size());
      this->set_diag(dvals.data());
    }

    /*! \brief Add to diagonal elements
     *
     * @param dvals values to add to diagonal
     *
     */
    void add_diag(double const* dvals){
      assert(this->dbcsr_env != nullptr);
      assert(this->nrow == this->ncol);
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
          c_dbcsr_finalize(dbcsr_matrix);
        }
        ioff += idim;
      }
    }

    /*! \brief Add to diagonal elements
     *
     * @param dvals values to add to diagonal
     *
     */
    void add_diag(const std::vector<double>& dvals){
      assert(this->nrow == this->ncol);
      assert(this->nrow <= dvals.size());
      this->add_diag(dvals.data());
    }

    /*! \brief Add to diagonal elements
     *
     * @param val value to add to diagonal
     *
     */
    void add_diag(const double val){
      assert(nrow == ncol);
      int max_dim = *(std::max_element(row_dims.begin(),row_dims.end()));
      std::vector<double> block1(max_dim*max_dim,0.e0);
    
      for(int i = 0; i < (int)row_dims.size(); i++){
        int idim = row_dims[i];
        bool found1 = false;
        c_dbcsr_get_block_d(&(dbcsr_matrix), i, i, block1.data(), found1, idim, idim);
        if (found1){
          for(int ii=0;ii<idim;ii++) block1[ii+ii*idim] += val;
          c_dbcsr_put_block_d(dbcsr_matrix, i, i, block1.data(), idim*idim);
          c_dbcsr_finalize(dbcsr_matrix);
        }
      }

    }

    /*! \brief Set elements of matrix
     *
     * @param val value
     *
     */
    void set(const double val){
      assert(this->dbcsr_env != nullptr);
      c_dbcsr_set_d(&(this->dbcsr_matrix),val);
    }

    /*! \brief Zero matrix
     */
    void zero(){
      assert(this->dbcsr_env != nullptr);
      // best to remove old matrix and generate new one...
      if (dbcsr_matrix != nullptr){
        c_dbcsr_release(&(this->dbcsr_matrix));
        dbcsr_matrix = nullptr;
      }

      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_env->get_dist(), 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());

      int max_dim = *(std::max_element(row_dims.begin(),row_dims.end()));
      std::vector<double> block(max_dim*max_dim,0.e0);
      for(int i = 0; i < (int)row_dims.size(); i++){
        int blk_proc = -1;
        c_dbcsr_get_stored_coordinates(dbcsr_matrix, i, i, &blk_proc);

        if(blk_proc == dbcsr_env->get_rank()){
          int idim = row_dims[i];
          c_dbcsr_put_block_d(dbcsr_matrix, i, i, block.data(), idim*idim);
        }
      }
      

      c_dbcsr_finalize(dbcsr_matrix);

    }

    /*! \brief Dot-product
     *
     * retval = sum_ij(A_ij * this_ij)
     *
     * @param A input matrix
     *
     */
    double dot(const DistBCSR& A) const{
      assert(this->dbcsr_env != nullptr);
      assert(A.dbcsr_env != nullptr);
      double tr = 0.e0;
      c_dbcsr_trace_ab_d((dm_dbcsr*)&(this->dbcsr_matrix),(dm_dbcsr*)&(A.dbcsr_matrix),tr);
      return tr;
    }

    /*! \brief Filter matrix
     *
     * @param eps sparse compression threshold
     *
     */
    void filter(double eps=-1.e0){
      assert(this->dbcsr_env != nullptr);
      if (eps < 0.e0) eps = dbcsr_thresh;
      c_dbcsr_filter_d(&(this->dbcsr_matrix),eps);
    }

    /*! \brief Filter matrix (block-norm)
     *
     * @param eps sparse compression threshold
     *
     */
    void recompress(double eps=-1.e0){
      assert(this->dbcsr_env != nullptr);
      if (eps < 0.e0) eps = dbcsr_thresh;
      int max_dimr = *(std::max_element(row_dims.begin(),row_dims.end()));
      int max_dimc = *(std::max_element(col_dims.begin(),col_dims.end()));
      std::vector<double> block(max_dimr*max_dimc);
      for(int i = 0; i < (int)row_dims.size(); i++){
        int idim = row_dims[i];
        for(int j = 0; j < (int)col_dims.size(); j++){
          int jdim = col_dims[j];
          bool found = false;
          c_dbcsr_get_block_d(&dbcsr_matrix, i, j, block.data(), found, idim, jdim);
          if(found){
            int bdim = idim*jdim;
            double ddot = 0.e0;
            for(int ii=0;ii<bdim;ii++) ddot += block[ii] * block[ii];
            if (sqrt(ddot/((double)(bdim))) < eps){
              c_dbcsr_remove_blk_d(&dbcsr_matrix, i, j, bdim);
            }
          }
        }
      }
    }

    /*! \brief Scale matrix
     *
     * this *= fac
     *
     * @param fac scaling factor
     *
     */
    void scale(const double fac){
      assert(this->dbcsr_env != nullptr);
      c_dbcsr_scale_d(&(this->dbcsr_matrix),fac);
    }

    /*! \brief Load matrix from disk
     *
     * @param cfname file-name
     *
     */
    void load(const std::string& cfname){
      assert(this->dbcsr_env != nullptr);
      c_dbcsr_read_d(&(this->dbcsr_matrix),(char*)cfname.c_str(),dbcsr_env->get_dist_ptr());
    }

    /*! \brief Write matrix to disk
     *
     * @param cfname file-name
     *
     */
    void write(const std::string& cfname){
      assert(this->dbcsr_env != nullptr);
      c_dbcsr_write_d(&(this->dbcsr_matrix),(char*)cfname.c_str());
    }

    /*! \brief Returns maximal absolute value of matrix
     */
    double maxabs() const {
      assert(this->dbcsr_env != nullptr);
      double amv = 0.e0;
      c_dbcsr_maxabs_d((dm_dbcsr*)&(this->dbcsr_matrix),&amv);
      return amv;
    }

    /*! \brief Hadamard product
     *
     * this_ij *= A_ij
     *
     * @param rhs input matrix
     *
     */
    void hadamard(const DistBCSR& rhs){
      assert(this->dbcsr_env != nullptr);
      assert(rhs.dbcsr_env != nullptr);
      int max_dimr = *(std::max_element(row_dims.begin(),row_dims.end()));
      int max_dimc = *(std::max_element(col_dims.begin(),col_dims.end()));

      std::vector<double> block1(max_dimr*max_dimc,0.e0);
      std::vector<double> block2(max_dimr*max_dimc,0.e0);
      
      size_t rowoff = 0;
      for(int irow = 0; irow < (int)row_dims.size(); irow++){
        int irowdim = row_dims[irow];
        size_t coloff = 0;
        for(int icol = 0; icol < (int)col_dims.size(); icol++){
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
            c_dbcsr_finalize(dbcsr_matrix);

          }
          coloff += icoldim;
        }
        rowoff += irowdim;
      }
    };

    /*! \brief 'Inverse' Hadamard product
     *
     * if (fabs(A_ij) > 1e-13): this_ij /= A_ij
     * else:                    this_ij  = 0.0
     *
     * @param rhs input matrix
     *
     */
    void hadamard_inv(const DistBCSR& rhs){
      assert(this->dbcsr_env != nullptr);
      assert(rhs.dbcsr_env != nullptr);
      int max_dimr = *(std::max_element(row_dims.begin(),row_dims.end()));
      int max_dimc = *(std::max_element(col_dims.begin(),col_dims.end()));

      std::vector<double> block1(max_dimr*max_dimc,0.e0);
      std::vector<double> block2(max_dimr*max_dimc,0.e0);
      
      size_t rowoff = 0;
      for(int irow = 0; irow < (int)row_dims.size(); irow++){
        int irowdim = row_dims[irow];
        size_t coloff = 0;
        for(int icol = 0; icol < (int)col_dims.size(); icol++){
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
            c_dbcsr_finalize(dbcsr_matrix);
          }
          coloff += icoldim;
        }
        rowoff += irowdim;
      }
    };

    /*! \brief Gershgorin estimate
     *
     * Returns estimate of min/max eigenvalues of matrix
     * based on Gershgorin discs
     *
     * @param eps0 estimate of min. eigenvalue [output]
     * @param epsn estimate of max. eigenvalue [output]
     *
     */
    void gershgorin_estimate(double& eps0, double& epsn) const {
      assert(this->dbcsr_env != nullptr);
      assert(nrow == ncol);
      std::vector<double> sums(nrow,0.e0);
      std::vector<double> diags(nrow,0.e0);
      std::vector<double> red_sums(nrow,0.e0);
      std::vector<double> red_diags(nrow,0.e0);
    
      //dbcsr::gershgorin_estimate((dm_dbcsr)this->dbcsr_matrix,row_dims.data(),(int)col_dims.size(),(int)nrow,sums.data(),diags.data());
      c_dbcsr_gershgorin_estimate_d((dm_dbcsr*)&(this->dbcsr_matrix),(int*)row_dims.data(),(int)col_dims.size(),(int)nrow,sums.data(),diags.data());
    
      // gather results
      MPI_Allreduce(sums.data(),red_sums.data(), ((int)nrow), MPI_DOUBLE, MPI_SUM, dbcsr_env->get_comm());
      MPI_Allreduce(diags.data(),red_diags.data(), ((int)nrow), MPI_DOUBLE, MPI_SUM, dbcsr_env->get_comm());
    
      double disc_min[2];
      disc_min[0] = std::numeric_limits<double>::max();
      disc_min[1] = std::numeric_limits<double>::lowest();
      double disc_max[2];
      disc_max[0] = std::numeric_limits<double>::lowest();
      disc_max[1] = std::numeric_limits<double>::lowest();
    
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

    /*! \brief Gershgorin estimate
     *
     * Returns estimate of min/max eigenvalues of matrix
     * based on Gershgorin discs (py)
     *
     */
    std::vector<double>  gershgorin_estimate() const {
      std::vector<double> epsv(2,0.e0);
      this->gershgorin_estimate(epsv[0],epsv[1]);
      return epsv;
    }

    // todo: transpose,...

    /*! \brief Operator: =
     */
    DistBCSR& operator=(const DistBCSR& rhs){
      assert(this->dbcsr_env != nullptr);
      assert(rhs.dbcsr_env != nullptr);
      if (this == &rhs) return *this;
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
      c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_env->get_dist(), 'N',
                            row_dims.data(), (int)row_dims.size(),
                            col_dims.data(), (int)col_dims.size());
      c_dbcsr_finalize(dbcsr_matrix);
      this->copy((DistBCSR&)rhs);
      return *this;
    };

    /*! \brief Operator: = (move)
     */
    DistBCSR& operator=(DistBCSR&& rhs){
      assert(this->dbcsr_env != nullptr);
      assert(rhs.dbcsr_env != nullptr);
      if (this != &rhs){
        this->free();
        mname = rhs.mname;
        dbcsr_env = rhs.dbcsr_env;
        nrow = rhs.nrow;
        ncol = rhs.ncol;
        row_dims = rhs.row_dims;
        col_dims = rhs.col_dims;
  
        dbcsr_matrix = nullptr;
        c_dbcsr_create_new_d(&dbcsr_matrix, mname.data(), dbcsr_env->get_dist(), 'N',
                              row_dims.data(), (int)row_dims.size(),
                              col_dims.data(), (int)col_dims.size());
        c_dbcsr_finalize(dbcsr_matrix);
        this->copy((DistBCSR&)rhs);
        // free rhs...
        rhs.free();
      }

      return *this;
    };

    /*! \brief Operator: +=
     */
    DistBCSR& operator+=(const DistBCSR& rhs){
      this->add((DistBCSR&)rhs);
      return *this;
    };

    /*! \brief Operator: -=
     */
    DistBCSR& operator-=(const DistBCSR& rhs){
      this->sub((DistBCSR&)rhs);
      return *this;
    };

    /*! \brief Operator: *= (scalar)
     */
    DistBCSR& operator*=(const double& fac){
      this->scale(fac);
      return *this;
    };

    /*! \brief Operator: * (scalar)
     */
    DistBCSR operator*(const double& fac){
      DistBCSR ret = *this;
      ret.scale(fac);
      return ret;
    };

    /*! \brief Operator: /= (scalar)
     */
    DistBCSR& operator/=(const double& fac){
      assert(fabs(fac) > 1e-13);
      this->scale(1.e0/fac);
      return *this;
    };

    /*! \brief Operator: +
     */
    DistBCSR operator+(const DistBCSR& rhs) const{
      DistBCSR ret = *this;
      ret.add(rhs);
      return std::move(ret);
    };

    /*! \brief Operator: -
     */
    DistBCSR operator-(const DistBCSR& rhs) const{
      DistBCSR ret = *this;
      ret.sub(rhs);
      return std::move(ret);
    };

    /*! \brief Operator: * (DistBCSR)
     */
    DistBCSR operator*(const DistBCSR& rhs) const{
      DistBCSR ret(*this);
      ret.mult('N','N',*this,rhs);
      return std::move(ret);
    };

    /*! \brief Print matrix
     *
     * @param tit title-string
     *
     */
    void print(const std::string& tit) const {
      assert(this->dbcsr_env != nullptr);
      if (dbcsr_env->get_rank() == 0) printf("  %s\n",tit.c_str());
      MPI_Barrier(dbcsr_env->get_comm());
      //dbcsr::print((dm_dbcsr)this->dbcsr_matrix);
      c_dbcsr_print((dm_dbcsr)this->dbcsr_matrix);
    };

    /*! \brief Set FORTRAN-handle
     *
     * @param din FORTRAN-handle for dbcsr-matrix
     *
     */
    void      set_dbcsr(dm_dbcsr& din){
      assert(this->dbcsr_env != nullptr);
      if (dbcsr_matrix != nullptr){
        c_dbcsr_release(&dbcsr_matrix);
        dbcsr_matrix = nullptr;
      }
      dbcsr_matrix = din;
    };
    
    /*! \brief Return row of matrix
     *
     * @param row row-index
     *
     */
    std::vector<double> get_row(const size_t row) const {
      assert(this->dbcsr_env != nullptr);
      std::vector<double> the_row(ncol);
      std::vector<double> loc_row(ncol);
      std::fill(the_row.begin(),the_row.end(),0);
      std::fill(loc_row.begin(),loc_row.end(),0);
    
      int which_row = 0;
      int rr = 0;
      while(which_row < (int)row_dims.size()){
        int bdim_act = row_dims[which_row];
        if (((int)row) >= rr && ((int)row) < (rr + bdim_act)){
          break;
        }
        which_row++;
        rr += bdim_act;
      }
      if (which_row == (int)row_dims.size()){
        printf("\n  Error: Illegal row-index!\n\n");
        exit(1);
      }
      int roff = ((int)row) - rr;
    
      int max_dimc = *(std::max_element(col_dims.begin(),col_dims.end()));
      int idim1 = row_dims[which_row];
      std::vector<double> block1(max_dimc*idim1,0.e0);

      int joff = 0;
      for(int j = 0; j < (int)col_dims.size(); j++){
        int jdim = col_dims[j];
        bool found1 = false;
        c_dbcsr_get_block_d((void**)&dbcsr_matrix, which_row, j, block1.data(), found1, idim1, jdim);
        if(found1){
          for(int jj=0;jj<jdim;jj++) loc_row[jj+joff] = block1[roff+jj*idim1];
        }
        joff += jdim;
      }
    
      MPI_Allreduce(loc_row.data(),the_row.data(),(int)nrow,MPI_DOUBLE, MPI_SUM, dbcsr_env->get_comm());

      return the_row;

    };

    /*! \brief Return column of matrix
     *
     * @param col column-index
     *
     */
    std::vector<double> get_column(const size_t col) const {
      assert(this->dbcsr_env != nullptr);
      std::vector<double> the_col(nrow);
      std::vector<double> loc_col(nrow);
      std::fill(the_col.begin(),the_col.end(),0);
      std::fill(loc_col.begin(),loc_col.end(),0);
    
      int which_col = 0;
      int cc = 0;
      while(which_col < (int)col_dims.size()){
        int bdim_act = col_dims[which_col];
        if (((int)col) >= cc && ((int)col) < (cc + bdim_act)){
          break;
        }
        which_col++;
        cc += bdim_act;
      }
      if (which_col == (int)col_dims.size()){
        printf("\n  Error: Illegal column-index!\n\n");
        exit(1);
      }
      int coff = ((int)col) - cc;
    
      int max_dimr = *(std::max_element(row_dims.begin(),row_dims.end()));
      int jdim1 = col_dims[which_col];
      std::vector<double> block1(max_dimr*jdim1,0.e0);

      int ioff = 0;
      for(int i = 0; i < (int)row_dims.size(); i++){
        int idim = row_dims[i];
        bool found1 = false;
        c_dbcsr_get_block_d((void**)&dbcsr_matrix, i, which_col, block1.data(), found1, idim, jdim1);
        if(found1){
          for(int ii=0;ii<idim;ii++) loc_col[ii+ioff] = block1[ii+coff*idim];
        }
        ioff += idim;
      }
    
      MPI_Allreduce(loc_col.data(),the_col.data(),(int)nrow,MPI_DOUBLE, MPI_SUM, dbcsr_env->get_comm());

      return the_col;

    };

    /*! \brief Remove block of dbcsr-matrix
     *
     * @param blk_row row block-index
     * @param blk_col column block-index
     *
     */
    void remove_block(const int blk_row, const int blk_col){
      assert(this->dbcsr_env != nullptr);

      int blk_proc = -1;
      c_dbcsr_get_stored_coordinates(dbcsr_matrix, blk_row, blk_col, &blk_proc);
      if (blk_proc == dbcsr_env->get_rank()){
        int bdim = row_dims[blk_row] * col_dims[blk_col];
        c_dbcsr_remove_blk_d(&dbcsr_matrix, blk_row, blk_col, bdim);
      }

    };

    /*! \brief Add/Overwrite block of dbcsr-matrix
     *
     * @param blk_row row block-index
     * @param blk_col column block-index
     * @param data input values for requested block
     *
     */
    void add_block(const int blk_row, const int blk_col, double const* data){
      assert(this->dbcsr_env != nullptr);
      int blk_proc = -1;
      c_dbcsr_get_stored_coordinates(dbcsr_matrix, blk_row, blk_col, &blk_proc);
      if (blk_proc == dbcsr_env->get_rank()){
        int bdim = row_dims[blk_row] * col_dims[blk_col];
        c_dbcsr_put_block_d(dbcsr_matrix, blk_row, blk_col, (double*)data, bdim);
        c_dbcsr_finalize(dbcsr_matrix);
      }
    };

    /*! \brief Add/Overwrite block of dbcsr-matrix
     *
     * @param blk_row row block-index
     * @param blk_col column block-index
     * @param data input values for requested block
     *
     */
    void add_block(int blk_row, int blk_col, const std::vector<double>& data){
      this->add_block(blk_row,blk_col,data.data());
    };

    /*! \brief Get block of dbcsr-matrix
     *
     * If block is not available, the functions returns an empty vector.
     *
     * @param blk_row row block-index
     * @param blk_col column block-index
     * @param reduce flag: return block to all mpi-processes
     *
     */
    std::vector<double> get_block(const int blk_row, const int blk_col, bool reduce=false) const {
      assert(this->dbcsr_env != nullptr);
      std::vector<double> ret;
      int idim = row_dims[blk_row];
      int jdim = col_dims[blk_col];
      std::vector<double> blk(idim*jdim,0.e0);
      bool found = false;
      int blk_proc = -1;
      c_dbcsr_get_stored_coordinates(dbcsr_matrix, blk_row, blk_col, &blk_proc);
      if (blk_proc == dbcsr_env->get_rank())
        c_dbcsr_get_block_d((void**)&dbcsr_matrix, blk_row, blk_col, blk.data(), found, idim, jdim);
      if(found){
        if (!reduce){
          ret = blk;
        }
      }else{
        if (reduce){
          ret = blk;
        }
      }

      if (reduce){
        MPI_Allreduce(blk.data(), ret.data(), idim*jdim,MPI_DOUBLE, MPI_SUM, dbcsr_env->get_comm());
      }
      return ret;
    };

    /*! \brief Returns if block is local
     *
     * @param blk_row row block-index
     * @param blk_col column block-index
     *
     */
    bool local_block(const int blk_row, const int blk_col) const {
      assert(this->dbcsr_env != nullptr);
      int blk_proc = -1;
      c_dbcsr_get_stored_coordinates(dbcsr_matrix, blk_row, blk_col, &blk_proc);
      return (blk_proc == dbcsr_env->get_rank() ? true : false);
    };

    /*! \brief Returns FORTRAN-handle to dbcsr-matrix
     */
    dm_dbcsr& get_dbcsr(){return dbcsr_matrix;};

    /*! \brief Returns DBCSR_Environment object
     */
    std::shared_ptr<const DBCSR_Environment> get_env() const {return dbcsr_env;};

    /*! \brief Returns number of rows in dense matrix
     */
    size_t get_nrow() const {return nrow;};

    /*! \brief Returns number of columns in dense matrix
     */
    size_t get_ncol() const {return ncol;};

    /*! \brief Returns dimensions of row-blocks
     */
    std::vector<int> get_row_dims() const {return row_dims;};

    /*! \brief Returns dimensions of column-blocks
     */
    std::vector<int> get_col_dims() const {return col_dims;};

    /*! \brief Returns sparse-threshold
     */
    double get_thresh() const {return dbcsr_thresh;};

    /*! \brief Sets sparse-threshold
     *
     * @param din sparse compression threshold
     *
     */
    void   set_thresh(const double din){dbcsr_thresh = din;};

};

#endif

