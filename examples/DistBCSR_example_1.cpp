
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
    //std::shared_ptr<const DBCSR_Environment> std::make_shared(dbcsr_env(nblkrows_total,nblkcols_total));
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

    // check distribute & gather
    auto a_ref = matrix_a.gather();
    double ddd = 0.e0;
    for(size_t ii=0;ii<loc_matrix_a.size();ii++){
      double dif = a_ref[ii] - loc_matrix_a[ii];
      ddd += dif*dif;
    }
    ddd = sqrt(ddd/((double)loc_matrix_a.size()));
    if (dbcsr_env->mpi_rank == 0) printf("Distribute & Gather: ||A_dist - A_loc|| = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);

    // multiply the matrices
    DistBCSR matrix_c(matrix_a);
    matrix_c.mult('N','N',matrix_a,matrix_b);
    auto loc_matrix_c = bland_matmult('N','N',loc_matrix_a,nrow_tot,ncol_tot,loc_matrix_b,nrow_tot,ncol_tot);

    auto c_ref = matrix_c.gather();
    ddd = 0.e0;
    for(size_t ii=0;ii<loc_matrix_c.size();ii++){
      double dif = c_ref[ii] - loc_matrix_c[ii];
      ddd += dif*dif;
    }
    ddd = sqrt(ddd/((double)loc_matrix_c.size()));
    if (dbcsr_env->mpi_rank == 0) printf("Multiplication:      ||C_dist - C_loc|| = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);

    // copy a to b and subtract (a -= b)
    matrix_b.copy(matrix_a);
    matrix_a.sub(matrix_b);

    a_ref = matrix_a.gather();
    ddd = 0.e0;
    for(size_t ii=0;ii<loc_matrix_a.size();ii++){
      double dif = a_ref[ii];
      ddd += dif*dif;
    }
    ddd = sqrt(ddd/((double)loc_matrix_c.size()));
    if (dbcsr_env->mpi_rank == 0) printf("Copy & Subtract:     ||Diff||           = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);

    // restore dist-mats
    matrix_a.load(loc_matrix_a.data());
    matrix_b.load(loc_matrix_b.data());

    // trace a
    double dtr = matrix_a.trace();
    double ltr = 0.e0;
    for(int ii=0;ii<nrow_tot;ii++) ltr += loc_matrix_a[ii+ii*nrow_tot];
    ddd = fabs(dtr-ltr);
    if (dbcsr_env->mpi_rank == 0) printf("Trace[A]:            ||Dist-Loc||       = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);

    // trace ab
    dtr = matrix_a.dot(matrix_b);
    ltr = 0.e0;
    for(int ii=0;ii<ncol_tot;ii++){
      for(int jj=0;jj<nrow_tot;jj++){
        // actually, this would be the accurate approach
        //ltr += loc_matrix_a[jj+ii*nrow_tot]*loc_matrix_b[ii+jj*nrow_tot];
        ltr += loc_matrix_a[jj+ii*nrow_tot]*loc_matrix_b[jj+ii*nrow_tot];
      }
    }
    ddd = fabs(dtr-ltr);
    if (dbcsr_env->mpi_rank == 0) printf("Trace[AB]:           ||Dist-Loc||       = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);
    
    // set & set diag
    std::vector<double> diags(nrow_tot,0.e0);
    for(int ii=0;ii<nrow_tot;ii++) diags[ii] = ((double)ii) + 0.5e0;
    matrix_a.set(0.e0);
    matrix_a.set_diag(diags);
    auto loc_diags = matrix_a.gather();
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
    if (dbcsr_env->mpi_rank == 0) printf("Set & SetDiag:       ||Diff||           = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);
    // scale
    matrix_b.scale(2.5e0);
    auto b_ref = matrix_b.gather();
    ddd = 0.e0;
    for(size_t ii=0;ii<loc_matrix_b.size();ii++){
      double dif = b_ref[ii] - loc_matrix_b[ii]*2.5e0;
      ddd += dif*dif;
    }
    ddd = sqrt(ddd/((double)loc_matrix_c.size()));
    if (dbcsr_env->mpi_rank == 0) printf("Scale:               ||Diff||           = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);
    // max. absolute value
    double amv = matrix_b.maxabs();
    ddd = 0.e0;
    for(size_t ii=0;ii<loc_matrix_b.size();ii++){
      double act = fabs(loc_matrix_b[ii]*2.5e0);
      ddd = (ddd > act ? ddd : act);
    }
    ddd = fabs(ddd-amv);
    if (dbcsr_env->mpi_rank == 0) printf("abs(max)             ||Dist-Loc||       = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    fflush(stdout);
    // gershgorin w/ an inappropriate matrix
    matrix_c.load(loc_matrix_c);
    double eps0_loc = 0.e0;
    double epsn_loc = 0.e0;
    gershgorin_loc(loc_matrix_c,nrow_tot,eps0_loc,epsn_loc);
    double eps0_dst = 0.e0;
    double epsn_dst = 0.e0;
    matrix_c.gershgorin_estimate(eps0_dst,epsn_dst);
    ddd = fabs(((eps0_loc-eps0_dst)+(epsn_loc-epsn_dst))*0.5e0);
    if (dbcsr_env->mpi_rank == 0)
      printf("Gershgorin           ||Diff||           = %20.10e --- %s\n                     local = (%e,%e), distr = (%e,%e)\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"),
             eps0_loc,epsn_loc,eps0_dst,epsn_dst);
    fflush(stdout);

    // write/read
    matrix_c.write("matrix_c");
    matrix_b.load("matrix_c");
    matrix_b.sub(matrix_c);
    auto bmc_ref = matrix_b.gather();
    ddd = 0.e0;
    for(size_t ii=0;ii<bmc_ref.size();ii++){
      double act = bmc_ref[ii];
      ddd = act*act;
    }
    ddd = sqrt(ddd/((double)bmc_ref.size()));
    if (dbcsr_env->mpi_rank == 0) printf("Write/Read           ||Diff||           = %20.10e --- %s\n",ddd,(ddd < 1e-13 ? "OK" : "FAILED!"));
    // new stuff...
    //void dbcsr::filter(dm_dbcsr* mat_a, double eps);
    fflush(stdout);
    matrix_c.print("matrix_c.print()\n");
    fflush(stdout);
    c_ref = matrix_c.gather();
    if (dbcsr_env->mpi_rank == 0){
      printf("(loc_matrix_c):\n");
      for(int irow=0;irow<nrow_tot;irow++){
        for(int icol=0;icol<ncol_tot;icol++){
          printf("%12.5e ",c_ref[irow+icol*nrow_tot]);
        }
        printf("\n");
      }
    }

    fflush(stdout);

}

int main(int argc, char* argv[])
{

    MPI_Init(&argc, &argv);

    run_test();

    MPI_Finalize();

    return 0;
}


