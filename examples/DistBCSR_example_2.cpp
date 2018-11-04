
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

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);


    dbcsr::init_lib();

    double const sthr = 1.e-9;
    // Total number of blocks
    int nblkrows_total = 4;
    int nblkcols_total = 4;
    int dim_per_block  = 2;
    // only square matrces for these tests
    assert(nblkrows_total == nblkcols_total);

    // environment-object...
    DBCSR_Environment dbcsr_env(nblkrows_total,nblkcols_total);

    std::cout
        << "I'm processor " << dbcsr_env.mpi_rank
        << " over " << dbcsr_env.mpi_size << " proc"
        << ", (" << dbcsr_env.dbcsr_coords[0] << ", " << dbcsr_env.dbcsr_coords[1] << ") in the 2D grid"
        << std::endl;

    // Block sizes
    std::vector<int> row_blk_sizes(nblkrows_total, dim_per_block), col_blk_sizes(nblkcols_total, dim_per_block);

    int nrow_tot = nblkrows_total*dim_per_block;
    int ncol_tot = nblkcols_total*dim_per_block;
    // create and fill matrix a
    auto loc_matrix_a = random_matrix(nrow_tot,ncol_tot);
    DistBCSR matrix_a(nrow_tot,ncol_tot,row_blk_sizes,col_blk_sizes,&dbcsr_env);
    matrix_a.load(loc_matrix_a.data(),sthr);

    // create and fill matrix b
    auto loc_matrix_b = random_matrix(nrow_tot,ncol_tot);
    DistBCSR matrix_b(nrow_tot,ncol_tot,row_blk_sizes,col_blk_sizes,&dbcsr_env);
    matrix_b.load(loc_matrix_b.data(),sthr);

    // multiply the matrices
    auto matrix_c = matrix_a * matrix_b;
    auto loc_matrix_c = bland_matmult('N','N',loc_matrix_a,nrow_tot,ncol_tot,loc_matrix_b,nrow_tot,ncol_tot);

    auto c_ref = matrix_c.gather();
    double ddd = 0.e0;
    for(size_t ii=0;ii<loc_matrix_c.size();ii++){
      double dif = c_ref[ii] - loc_matrix_c[ii];
      ddd += dif*dif;
    }
    ddd = sqrt(ddd/((double)loc_matrix_c.size()));
    if (dbcsr_env.mpi_rank == 0) printf("Operator '*':        ||C_dist - C_loc|| = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);

    // copy a to b and subtract (a -= b)
    matrix_b = matrix_a;
    matrix_a -= matrix_b;

    auto a_ref = matrix_a.gather();
    ddd = 0.e0;
    for(size_t ii=0;ii<loc_matrix_a.size();ii++){
      double dif = a_ref[ii];
      ddd += dif*dif;
    }
    ddd = sqrt(ddd/((double)loc_matrix_a.size()));
    if (dbcsr_env.mpi_rank == 0) printf("Operator '=' & '-=': ||Diff||           = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);

    // restore dist-mat b
    matrix_a.load(loc_matrix_a.data());
    matrix_b.load(loc_matrix_b.data());

    // +=
    matrix_a += matrix_b;
    auto apb_ref = matrix_a.gather();
    ddd = 0.e0;
    for(size_t ii=0;ii<loc_matrix_a.size();ii++){
      double dif = apb_ref[ii] - (loc_matrix_a[ii] + loc_matrix_b[ii]);
      ddd += dif*dif;
    }
    ddd = sqrt(ddd/((double)loc_matrix_b.size()));
    if (dbcsr_env.mpi_rank == 0) printf("Operator '+=':       ||Diff||           = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);

    // scale
    matrix_b *= 2.5e0;
    auto b_ref = matrix_b.gather();
    ddd = 0.e0;
    for(size_t ii=0;ii<loc_matrix_b.size();ii++){
      double dif = b_ref[ii] - loc_matrix_b[ii]*2.5e0;
      ddd += dif*dif;
    }
    ddd = sqrt(ddd/((double)loc_matrix_b.size()));
    if (dbcsr_env.mpi_rank == 0) printf("Operator '*= 2.5':   ||Diff||           = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);

    return 0;
}
