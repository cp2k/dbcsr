
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <random>
#include <cassert>

#include <mpi.h>

#include <DistBCSR.hpp>


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

void run_test(){

    dbcsr::init_lib();

    double const sthr = 1.e-9;
    // Total number of blocks
    int nblkrows_total = 4;
    int nblkcols_total = 4;
    int dim_per_block  = 2;
    // only square matrces for these tests
    assert(nblkrows_total == nblkcols_total);

    // environment-object...
    std::shared_ptr<const DBCSR_Environment> dbcsr_env = std::make_shared<DBCSR_Environment>(nblkrows_total,nblkcols_total);

    std::cout
        << "I'm processor " << dbcsr_env->mpi_rank
        << " over " << dbcsr_env->mpi_size << " proc"
        << ", (" << dbcsr_env->dbcsr_coords[0] << ", " << dbcsr_env->dbcsr_coords[1] << ") in the 2D grid"
        << std::endl;

    // Block sizes
    std::vector<int> row_blk_sizes(nblkrows_total, dim_per_block), col_blk_sizes(nblkcols_total, dim_per_block);

    int nrow_tot = nblkrows_total*dim_per_block;
    int ncol_tot = nblkcols_total*dim_per_block;
    // create and fill matrix a
    auto loc_matrix_a = random_matrix(nrow_tot,ncol_tot);
    DistBCSR matrix_a(nrow_tot,ncol_tot,row_blk_sizes,col_blk_sizes,dbcsr_env);
    matrix_a.load(loc_matrix_a.data(),sthr);

    // create and fill matrix b
    auto loc_matrix_b = random_matrix(nrow_tot,ncol_tot);
    DistBCSR matrix_b(nrow_tot,ncol_tot,row_blk_sizes,col_blk_sizes,dbcsr_env);
    matrix_b.load(loc_matrix_b.data(),sthr);

    // get 2nd and 3rd row/column
    auto r2 = matrix_a.get_row(2);
    auto r3 = matrix_a.get_row(3);
    auto c2 = matrix_a.get_column(2);
    auto c3 = matrix_a.get_column(3);
    double ddd = 0.e0;
    for(int ii=0;ii<ncol_tot;ii++){
      double d2 = r2[ii] - loc_matrix_a[2+ii*nrow_tot];
      ddd += d2*d2;
      double d3 = r3[ii] - loc_matrix_a[3+ii*nrow_tot];
      ddd += d3*d3;
    }
    for(int ii=0;ii<nrow_tot;ii++){
      double d2 = c2[ii] - loc_matrix_a[ii+2*nrow_tot];
      ddd += d2*d2;
      double d3 = c3[ii] - loc_matrix_a[ii+3*nrow_tot];
      ddd += d3*d3;
    }
    ddd = sqrt(ddd/((double)(2*nrow_tot+2*ncol_tot)));
    if (dbcsr_env->mpi_rank == 0) printf("Get row/column:      ||Diff||           = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);

    // symv
    matrix_b.symv(r2,0.75e0,r3,0.25e0);
    
    auto r3_ref = matrix_a.get_row(3);
    for(size_t ii=0;ii<r3_ref.size();ii++) r3_ref[ii] *= 0.25e0;
    auto Ax = bland_matmult('N','N',loc_matrix_b,nrow_tot,ncol_tot,r2,nrow_tot,1);
    for(size_t ii=0;ii<r3_ref.size();ii++) r3_ref[ii] += 0.75e0 * Ax[ii];
    ddd = 0.e0;
    for(size_t ii=0;ii<r3.size();ii++){
      double dif = r3[ii] - r3_ref[ii];
      ddd += dif*dif;
    }
    ddd = sqrt(ddd/((double)r3_ref.size()));
    if (dbcsr_env->mpi_rank == 0) printf("Symv:                ||Diff||           = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);
    
    // hadamard
    matrix_a.load(loc_matrix_a.data(),sthr);
    matrix_b.load(loc_matrix_b.data(),sthr);
    matrix_a.hadamard(matrix_b);
    auto hada_ref = matrix_a.gather();
    ddd = 0.e0;
    for(size_t ii=0;ii<hada_ref.size();ii++){
      double dif = hada_ref[ii] - (loc_matrix_a[ii] * loc_matrix_b[ii]);
      ddd += dif*dif;
    }
    ddd = sqrt(ddd/((double)hada_ref.size()));
    if (dbcsr_env->mpi_rank == 0) printf("Hadamard:            ||Diff||           = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);

}

int main(int argc, char* argv[])
{

    MPI_Init(&argc, &argv);

    run_test();

    MPI_Finalize();

    return 0;
}
