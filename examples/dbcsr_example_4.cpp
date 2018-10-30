
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <random>
#include <cassert>

#include <mpi.h>

#include <dbcsr_c.h>


// Random distribution by using round-robin assignment 
// of blocks to processors
std::vector<int> random_dist(int dist_size, int nbins)
{
    std::vector<int> dist(dist_size);

    for(int i=0; i < dist_size; i++)
        dist[i] = (nbins-i+1) % nbins;

    return std::move(dist);
}

// simple local multiplication for comparison...
std::vector<double> bland_matmult(char mode_a, char mode_b, std::vector<double>& a, int nrow_a, int ncol_a,
                                  std::vector<double>& b, int nrow_b, int ncol_b){

  int nrow_c = (mode_a == 'N' ? nrow_a : ncol_a);
  int ncol_c = (mode_b == 'N' ? ncol_b : nrow_b);

  

  std::vector<double> c(nrow_c*ncol_c,0.e0);

  if (mode_a == 'N'){
    if (mode_b == 'N'){
      assert(ncol_a == nrow_b);
      for(int col=0;col<ncol_c;col++){
        for(int row=0;row<nrow_c;row++){
          double v = 0.e0;
          for(int cont=0;cont<ncol_a;cont++) v += a[row+cont*nrow_a] * b[cont+col*nrow_b];
          c[row+col*nrow_c] = v;
        }
      }
    }else{
      assert(ncol_a == ncol_b);
      for(int col=0;col<ncol_c;col++){
        for(int row=0;row<nrow_c;row++){
          double v = 0.e0;
          for(int cont=0;cont<ncol_a;cont++) v += a[row+cont*nrow_a] * b[col+cont*nrow_b];
          c[row+col*nrow_c] = v;
        }
      }
    }
  }else{
    if (mode_b == 'N'){
      assert(nrow_a == nrow_b);
      for(int col=0;col<ncol_c;col++){
        for(int row=0;row<nrow_c;row++){
          double v = 0.e0;
          for(int cont=0;cont<ncol_a;cont++) v += a[cont+row*nrow_a] * b[cont+col*nrow_b];
          c[row+col*nrow_c] = v;
        }
      }
    }else{
      assert(nrow_a == ncol_b);
      for(int col=0;col<ncol_c;col++){
        for(int row=0;row<nrow_c;row++){
          double v = 0.e0;
          for(int cont=0;cont<ncol_a;cont++) v += a[cont+row*nrow_a] * b[col+cont*nrow_b];
          c[row+col*nrow_c] = v;
        }
      }
    }
  }

  return c;

}

std::vector<double> random_matrix(int nrow, int ncol){

  std::vector<double> matrix(nrow*ncol);

  for(size_t ii=0;ii<matrix.size();ii++)
    matrix[ii] = static_cast<double>(std::rand())/RAND_MAX;

  return matrix;

}

dm_dbcsr distribute(std::vector<double>& loc_matrix, std::vector<int>& row_blk_sizes, std::vector<int>& col_blk_sizes,
                   int nrow_tot, int ncol_tot, void* dist){

  assert(nrow_tot*ncol_tot <= (int)loc_matrix.size());

  dm_dbcsr dmat = nullptr;

  c_dbcsr_create_new_d(&dmat, "this is some matrix", dist, 'N',
        row_blk_sizes.data(), row_blk_sizes.size(),
        col_blk_sizes.data(), col_blk_sizes.size());

  int max_row_size = *std::max_element(row_blk_sizes.begin(),row_blk_sizes.end());
  int max_col_size = *std::max_element(col_blk_sizes.begin(),col_blk_sizes.end());
  int max_nze = max_row_size * max_col_size;

  std::vector<double> block;
  block.reserve(max_nze);

  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int ioff = 0;
  for(int i = 0; i < (int)row_blk_sizes.size(); i++){
    int idim = row_blk_sizes[i];
    int joff = 0;
    for(int j = 0; j < (int)col_blk_sizes.size(); j++){
      int jdim = col_blk_sizes[j];
      int blk_proc = -1;
      c_dbcsr_get_stored_coordinates(dmat, i, j, &blk_proc);
      if(blk_proc == mpi_rank){
        block.resize(idim*jdim);
        for(int cc=0;cc<jdim;cc++){
          for(int rr=0;rr<idim;rr++){
            block[rr+cc*idim] = loc_matrix[rr+ioff+(cc+joff)*nrow_tot];
          }
        }
        c_dbcsr_put_block_d(dmat, i, j, block.data(), block.size());
      }
      joff += jdim;
    }
    ioff += idim;
  }
  c_dbcsr_finalize(dmat);

  return std::move(dmat);

}

std::vector<double> gather(dm_dbcsr& dmat, std::vector<int>& row_blk_sizes, std::vector<int>& col_blk_sizes,
                           int nrow_tot, int ncol_tot, MPI_Comm& dbcsr_group){

  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int totdim = (nrow_tot*ncol_tot);
  std::vector<double> loc_src(totdim,0.e0);
  std::vector<double> loc_matrix(totdim,0.e0);
  std::vector<double> block;
  int ioff = 0;
  for(int i = 0; i < (int)row_blk_sizes.size(); i++)
  {
      int idim = row_blk_sizes[i];
      int joff = 0;
      for(int j = 0; j < (int)col_blk_sizes.size(); j++)
      {
          int jdim = col_blk_sizes[j];
          int blk_proc = -1;
          c_dbcsr_get_stored_coordinates(dmat, i, j, &blk_proc);

          if(blk_proc == mpi_rank)
          {
              block.resize(idim*jdim);
              std::fill(block.begin(), block.end(), 0);
              double* bptr = &block[0];
              bool foundit = false;
              c_dbcsr_get_block_d(&dmat,i,j,bptr,foundit,idim,jdim);
              if (foundit){
                for(int cc=0;cc<jdim;cc++){
                  for(int rr=0;rr<idim;rr++){
                    loc_src[rr+ioff+(cc+joff)*nrow_tot] = block[rr+cc*idim];
                  }
                }
              }
          }
          joff += jdim;
      }
      ioff += idim;
  }

  MPI_Allreduce(loc_src.data(), loc_matrix.data(), totdim, MPI_DOUBLE, MPI_SUM, dbcsr_group);

  return loc_matrix;

}

void gershgorin_loc(std::vector<double>& loc_mat, int ldim, double& eps0, double& epsn){

  double disc_min[2];
  disc_min[0] =  9999.e0;
  disc_min[1] = -9999.e0;
  double disc_max[2];
  disc_max[0] = -9999.e0;
  disc_max[1] = -9999.e0;
  double disc_act[2];

  for(int i=0;i<ldim;i++){
    disc_act[0] = loc_mat[i+i*ldim];
    disc_act[1] = 0.e0;
    for(int j=0;j<i;j++)      disc_act[1] += fabs(loc_mat[i+j*ldim]);
    for(int j=i+1;j<ldim;j++) disc_act[1] += fabs(loc_mat[i+j*ldim]);
    if ((disc_max[0] + disc_max[1]) < (disc_act[0] + disc_act[1])){
      disc_max[0] = disc_act[0];
      disc_max[1] = disc_act[1];
    }
    if ((disc_min[0] - disc_min[1]) > (disc_act[0] - disc_act[1])){
      disc_min[0] = disc_act[0];
      disc_min[1] = disc_act[1];
    }
  }

  eps0 = disc_min[0] - disc_min[1];
  epsn = disc_max[0] + disc_max[1];

}

void gershgorin_dist(dm_dbcsr& dist_mat, int ldim, double& eps0, double& epsn,
                     std::vector<int>& row_blk_sizes){

  std::vector<double> sums(ldim,0.e0);
  std::vector<double> diags(ldim,0.e0);
  std::vector<double> red_sums(ldim,0.e0);
  std::vector<double> red_diags(ldim,0.e0);

  c_dbcsr_gershgorin_estimate_d(&dist_mat,row_blk_sizes.data(),row_blk_sizes.size(),ldim,sums.data(),diags.data());

  // gather results
  MPI_Allreduce(sums.data(),red_sums.data(), ((int)ldim), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(diags.data(),red_diags.data(), ((int)ldim), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  double disc_min[2];
  disc_min[0] =  9999.e0;
  disc_min[1] = -9999.e0;
  double disc_max[2];
  disc_max[0] = -9999.e0;
  disc_max[1] = -9999.e0;

  for(int i=0;i<ldim;++i){
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

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // Make 2D grid
    int dims[2];
    dims[0] = 0;
    dims[1] = 0;
    MPI_Dims_create(mpi_size, 2, dims);
    int periods[2];
    periods[0] = 1;
    periods[1] = 1;
    int reorder = 0;
    MPI_Comm group;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &group);

    int coord[2];
    MPI_Cart_coords(group, mpi_rank, 2, coord);

    std::cout
        << "I'm processor " << mpi_rank
        << " over " << mpi_size << " proc"
        << ", (" << coord[0] << ", " << coord[1] << ") in the 2D grid"
        << std::endl;

    c_dbcsr_init_lib();

    // Total number of blocks
    int nblkrows_total = 4;
    int nblkcols_total = 4;
    int dim_per_block  = 2;

    // only square matrces for these tests
    assert(nblkrows_total == nblkcols_total);

    // Block sizes
    std::vector<int> row_blk_sizes(nblkrows_total, dim_per_block), col_blk_sizes(nblkcols_total, dim_per_block);

    int nrow_tot = nblkrows_total*dim_per_block;
    int ncol_tot = nblkcols_total*dim_per_block;

    std::vector<int> row_dist(nblkrows_total,0);
    std::vector<int> col_dist(nblkcols_total,0);

    for(int i=0; i < nblkrows_total; i++){
      row_dist[i] = (i+1) % dims[0];
      col_dist[i] = (i+1) % dims[1];
    }

    void* dist = nullptr;

    MPI_Fint fcomm = MPI_Comm_c2f(group);
    c_dbcsr_distribution_new_aux(&dist, &fcomm,
                                 row_dist.data(), row_dist.size(),
                                 col_dist.data(), col_dist.size());

    // create and fill matrix a
    auto loc_matrix_a = random_matrix(nrow_tot,ncol_tot);
    auto matrix_a = distribute(loc_matrix_a,row_blk_sizes,col_blk_sizes,nrow_tot,ncol_tot,dist);

    // create and fill matrix b
    auto loc_matrix_b = random_matrix(nrow_tot,ncol_tot);
    auto matrix_b = distribute(loc_matrix_b,row_blk_sizes,col_blk_sizes,nrow_tot,ncol_tot,dist);

    // create matrix c, empty
    std::vector<double> loc_matrix_c(nrow_tot*ncol_tot,0.e0);
    auto matrix_c = distribute(loc_matrix_c,row_blk_sizes,col_blk_sizes,nrow_tot,ncol_tot,dist);

    // check distribute & gather
    auto a_ref = gather(matrix_a,row_blk_sizes,col_blk_sizes,nrow_tot,ncol_tot,group);
    double ddd = 0.e0;
    for(size_t ii=0;ii<loc_matrix_c.size();ii++){
      double dif = a_ref[ii] - loc_matrix_a[ii];
      ddd += dif*dif;
    }
    ddd = sqrt(ddd/((double)loc_matrix_c.size()));
    if (mpi_rank == 0) printf("Distribute & Gather: ||A_dist - A_loc|| = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);

    // multiply the matrices
    c_dbcsr_multiply_d('N', 'N', 1.0, &matrix_a, &matrix_b, 0.0, &matrix_c, nullptr);
    loc_matrix_c = bland_matmult('N','N',loc_matrix_a,nrow_tot,ncol_tot,loc_matrix_b,nrow_tot,ncol_tot);

    auto c_ref = gather(matrix_c,row_blk_sizes,col_blk_sizes,nrow_tot,ncol_tot,group);
    ddd = 0.e0;
    for(size_t ii=0;ii<loc_matrix_c.size();ii++){
      double dif = c_ref[ii] - loc_matrix_c[ii];
      ddd += dif*dif;
    }
    ddd = sqrt(ddd/((double)loc_matrix_c.size()));
    if (mpi_rank == 0) printf("Multiplication:      ||C_dist - C_loc|| = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);

    // copy a to b and subtract (a -= b)
    c_dbcsr_copy_d(&matrix_b,&matrix_a);
    c_dbcsr_add_d(&matrix_a, &matrix_b, 1.e0, -1.e0);

    a_ref = gather(matrix_a,row_blk_sizes,col_blk_sizes,nrow_tot,ncol_tot,group);
    ddd = 0.e0;
    for(size_t ii=0;ii<loc_matrix_a.size();ii++){
      double dif = a_ref[ii];
      ddd += dif*dif;
    }
    ddd = sqrt(ddd/((double)loc_matrix_c.size()));
    if (mpi_rank == 0) printf("Copy & Subtract:     ||Diff||           = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);

    // restore dist-mats
    c_dbcsr_release(&matrix_a);
    c_dbcsr_release(&matrix_b);
    matrix_a = distribute(loc_matrix_a,row_blk_sizes,col_blk_sizes,nrow_tot,ncol_tot,dist);
    matrix_b = distribute(loc_matrix_b,row_blk_sizes,col_blk_sizes,nrow_tot,ncol_tot,dist);

    // trace a
    double dtr = 0.e0;
    c_dbcsr_trace_a_d(&matrix_a,dtr);
    double ltr = 0.e0;
    for(int ii=0;ii<nrow_tot;ii++) ltr += loc_matrix_a[ii+ii*nrow_tot];
    ddd = fabs(dtr-ltr);
    if (mpi_rank == 0) printf("Trace[A]:            ||Dist-Loc||       = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);

    // trace ab
    dtr = 0.e0;
    c_dbcsr_trace_ab_d(&matrix_a,&matrix_b,dtr);
    ltr = 0.e0;
    for(int ii=0;ii<ncol_tot;ii++){
      for(int jj=0;jj<nrow_tot;jj++){
        // actually, this would be the accurate approach
        //ltr += loc_matrix_a[jj+ii*nrow_tot]*loc_matrix_b[ii+jj*nrow_tot];
        ltr += loc_matrix_a[jj+ii*nrow_tot]*loc_matrix_b[jj+ii*nrow_tot];
      }
    }
    ddd = fabs(dtr-ltr);
    if (mpi_rank == 0) printf("Trace[AB]:           ||Dist-Loc||       = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);
    
    // set & set diag
    std::vector<double> diags(nrow_tot,0.e0);
    for(int ii=0;ii<nrow_tot;ii++) diags[ii] = ((double)ii) + 0.5e0;
    c_dbcsr_set_d(&matrix_a,0.e0);
    c_dbcsr_set_diag_d(&matrix_a,diags.data(),nrow_tot);
    auto loc_diags = gather(matrix_a,row_blk_sizes,col_blk_sizes,nrow_tot,ncol_tot,group);
    ddd = 0.e0;
    for(int ii=0;ii<ncol_tot;ii++){
      for(int jj=0;jj<nrow_tot;jj++){
        double dif = 0.e0;
        if (ii == jj)
          dif = loc_diags[ii+ii*nrow_tot] - (((double)ii) + 0.5e0);
        else
          dif = loc_diags[jj+ii*nrow_tot];
        ddd += dif*dif;
      }
    }
    ddd = sqrt(ddd/((double)loc_matrix_c.size()));
    if (mpi_rank == 0) printf("Set & SetDiag:       ||Diff||           = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);
    // scale
    c_dbcsr_scale_d(&matrix_b,2.5e0);
    auto b_ref = gather(matrix_b,row_blk_sizes,col_blk_sizes,nrow_tot,ncol_tot,group);
    ddd = 0.e0;
    for(size_t ii=0;ii<loc_matrix_b.size();ii++){
      double dif = b_ref[ii] - loc_matrix_b[ii]*2.5e0;
      ddd += dif*dif;
    }
    ddd = sqrt(ddd/((double)loc_matrix_c.size()));
    if (mpi_rank == 0) printf("Scale:               ||Diff||           = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);
    // max. absolute value
    double amv = 0.e0;
    c_dbcsr_maxabs_d(&matrix_b,&amv);
    ddd = 0.e0;
    for(size_t ii=0;ii<loc_matrix_b.size();ii++){
      double act = fabs(loc_matrix_b[ii]*2.5e0);
      ddd = (ddd > act ? ddd : act);
    }
    ddd = fabs(ddd-amv);
    if (mpi_rank == 0) printf("abs(max)             ||Dist-Loc||       = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);
    // gershgorin w/ an inappropriate matrix
    c_dbcsr_release(&matrix_c);
    matrix_c = distribute(loc_matrix_c,row_blk_sizes,col_blk_sizes,nrow_tot,ncol_tot,dist);
    double eps0_loc = 0.e0;
    double epsn_loc = 0.e0;
    gershgorin_loc(loc_matrix_c,nrow_tot,eps0_loc,epsn_loc);
    double eps0_dst = 0.e0;
    double epsn_dst = 0.e0;
    gershgorin_dist(matrix_c,nrow_tot,eps0_dst,epsn_dst,row_blk_sizes);
    ddd = fabs(((eps0_loc-eps0_dst)+(epsn_loc-epsn_dst))*0.5e0);
    if (mpi_rank == 0)
      printf("Gershgorin           ||Diff||           = %20.10e --- %s\n                     local = (%e,%e), distr = (%e,%e)",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"),
             eps0_loc,epsn_loc,eps0_dst,epsn_dst);
    fflush(stdout);

    // new stuff...
    //void c_dbcsr_filter_d(dm_dbcsr* mat_a, double eps);
    if (mpi_rank == 0) printf("c_dbcsr_print(matrix_c):\n");
    fflush(stdout);
    c_dbcsr_print(matrix_c);
    fflush(stdout);
    c_ref = gather(matrix_c,row_blk_sizes,col_blk_sizes,nrow_tot,ncol_tot,group);
    if (mpi_rank == 0){
      printf("(loc_matrix_c):\n");
      for(int irow=0;irow<nrow_tot;irow++){
        for(int icol=0;icol<ncol_tot;icol++){
          printf("%12.5e ",c_ref[irow+icol*nrow_tot]);
        }
        printf("\n");
      }
    }

    fflush(stdout);

    c_dbcsr_release(&matrix_a);
    c_dbcsr_release(&matrix_b);
    c_dbcsr_release(&matrix_c);

    c_dbcsr_distribution_release(&dist);

    c_dbcsr_finalize_lib_aux_silent(&fcomm);

    MPI_Comm_free(&group);
    MPI_Finalize();

    return 0;
}
