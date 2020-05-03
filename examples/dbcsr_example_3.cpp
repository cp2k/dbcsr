/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <random>

#include <mpi.h>

#include <dbcsr.h>


// Random distribution by using round-robin assignment
// of blocks to processors
std::vector<int> random_dist(int dist_size, int nbins)
{
  
    std::vector<int> dist(dist_size);

    for(int i=0; i < dist_size; i++)
        dist[i] = i % nbins;

    return dist;
}


// DBCSR example 3
// This example shows how to multiply two DBCSR matrices
int main(int argc, char* argv[])
{
    // initialize MPI
    MPI_Init(&argc, &argv);

    // setup the mpi environment
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // make 2D grid
    int dims[2] = {0};
    MPI_Dims_create(mpi_size, 2, dims);
    int periods[2] = {1};
    int reorder = 0;
    MPI_Comm group;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &group);

    int coord[2];
    MPI_Cart_coords(group, mpi_rank, 2, coord);

	for (int i = 0; i != mpi_size; ++i) {
		if (mpi_rank == i) {
			std::cout
				<< "I'm processor " << mpi_rank
				<< " over " << mpi_size << " proc"
				<< ", (" << coord[0] << ", " << coord[1] << ") in the 2D grid"
				<< std::endl;
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

    // initialize the DBCSR library
    c_dbcsr_init_lib(MPI_COMM_WORLD, nullptr);

    // Total number of blocks
    int nrows_1 = 4;
    int ncols_1 = 5;
    int nrows_2 = 5;
    int ncols_2 = 4;

    // Block sizes
    std::vector<int> row_blk_sizes_1 = {2, 3, 5, 2};
    std::vector<int> col_blk_sizes_1 = {3, 3, 4, 6, 2};
    std::vector<int> row_blk_sizes_2 = col_blk_sizes_1;
    std::vector<int> col_blk_sizes_2 = {5, 2, 5, 3};

    auto row_dist_1 = random_dist(nrows_1, dims[0]);
    auto col_dist_1 = random_dist(ncols_1, dims[1]);
    auto row_dist_2 = random_dist(nrows_2, dims[0]);
    auto col_dist_2 = random_dist(ncols_2, dims[1]);

    void* dist1 = nullptr;
    void* dist2 = nullptr;
    void* dist3 = nullptr;

	//create distributions
    c_dbcsr_distribution_new(&dist1, group,
        row_dist_1.data(), row_dist_1.size(),
        col_dist_1.data(), col_dist_1.size());
        
    c_dbcsr_distribution_new(&dist2, group,
        row_dist_2.data(), row_dist_2.size(),
        col_dist_2.data(), col_dist_2.size());
        
    c_dbcsr_distribution_new(&dist3, group,
        row_dist_1.data(), row_dist_1.size(),
        col_dist_2.data(), col_dist_2.size());

    // Fill all blocks, i.e. dense matrices
    auto fill_matrix = [&](void* matrix, std::vector<int>& irblks, std::vector<int>& icblks)
    {
        std::vector<double> block;
        std::vector<int> loc_irblks, loc_icblks;
        
        for (int i = 0; i != (int)irblks.size(); ++i) {
			int blk_proc = -1;
			int ix = irblks[i];
			int jx = icblks[i];
            c_dbcsr_get_stored_coordinates(matrix, ix, jx, &blk_proc);
            if (mpi_rank == blk_proc) {
				loc_irblks.push_back(ix);
				loc_icblks.push_back(jx);
			}
		}
		
		c_dbcsr_reserve_blocks(matrix, loc_irblks.data(), loc_icblks.data(), loc_irblks.size());

		void* iter = nullptr;
		c_dbcsr_iterator_start(&iter, matrix, nullptr, nullptr, nullptr, nullptr, nullptr);

        while (c_dbcsr_iterator_blocks_left(iter)) {
                    
                    int i = -1;
                    int j = -1;
                    int nblk = -1;
                    int rsize = -1;
                    int csize = -1;
                    bool tr = false;
                    
                    double* blk = nullptr;
                    c_dbcsr_iterator_next_2d_block_d(iter, &i, &j, &blk, &tr, &nblk, &rsize, &csize, nullptr, nullptr);  
                   
                    std::generate(blk, blk + rsize*csize, [&](){ return static_cast<double>(std::rand())/RAND_MAX; });
                    
        }
        
        c_dbcsr_iterator_stop(&iter);
        
    };
    

    // create the DBCSR matrices, i.e. a double precision non symmetric matrix
    // with nblkrows_total x nblkcols_total blocks and
    // sizes "sum(row_blk_sizes)" x "sum(col_blk_sizes)", distributed as
    // specified by the dist object

    // create, fill and finalize matrix a
    void* matrix_a = nullptr;
    void* matrix_b = nullptr;
    void* matrix_c = nullptr;
        
    c_dbcsr_create_new(&matrix_a, "matrix a", dist1, dbcsr_type_no_symmetry, 
                               row_blk_sizes_1.data(), row_blk_sizes_1.size(), 
                               col_blk_sizes_1.data(), col_blk_sizes_1.size(), 
                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr); 
                               
    c_dbcsr_create_new(&matrix_b, "matrix b", dist2, dbcsr_type_no_symmetry, 
                               row_blk_sizes_2.data(), row_blk_sizes_2.size(), 
                               col_blk_sizes_2.data(), col_blk_sizes_2.size(), 
                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr); 
                               
    c_dbcsr_create_new(&matrix_c, "matrix c", dist3, dbcsr_type_no_symmetry, 
                               row_blk_sizes_1.data(), row_blk_sizes_1.size(), 
                               col_blk_sizes_2.data(), col_blk_sizes_2.size(), 
                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr); 
    
    // indices of non-zero blocks
    std::vector<int> irblks_1 = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
    std::vector<int> icblks_1 = {0, 1, 2, 4, 0, 2, 3, 1, 3, 4, 0, 1, 2};
    
    std::vector<int> irblks_2 = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4};
    std::vector<int> icblks_2 = {0, 2, 3, 0, 1, 2, 3, 0, 2, 3, 1, 2, 3, 0, 1, 2, 3};
    
    std::vector<int> irblks_3 = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3};
    std::vector<int> icblks_3 = {0, 1, 2, 3, 0, 2, 3, 1, 2, 3, 0, 1, 2, 3};

    fill_matrix(matrix_a, irblks_1, icblks_1);
    c_dbcsr_finalize(matrix_a);
    fill_matrix(matrix_b, irblks_2, icblks_2);
    c_dbcsr_finalize(matrix_b);
    fill_matrix(matrix_c, irblks_3, icblks_3);
    c_dbcsr_finalize(matrix_c);
    
    // Compute C = 3.0 * A * B + 2.0 * C
    c_dbcsr_multiply_d('N', 'N', 3.0d, matrix_a, matrix_b, 2.0d, matrix_c, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
                       
    c_dbcsr_print(matrix_c);

    // release the matrices
    c_dbcsr_release(&matrix_a);
    c_dbcsr_release(&matrix_b);
    c_dbcsr_release(&matrix_c);

    c_dbcsr_distribution_release(&dist1);
    c_dbcsr_distribution_release(&dist2);
    c_dbcsr_distribution_release(&dist3);

    MPI_Comm_free(&group);

    // finalize the DBCSR library
    c_dbcsr_finalize_lib();

    // finalize MPI
    MPI_Finalize();

    return 0;
}
