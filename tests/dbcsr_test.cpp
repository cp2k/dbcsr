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


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // Make 2D grid
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
    
    if (mpi_rank == 0) std::cout << "Creating distributions..." << std::endl;

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
                    int roff = -1;
                    int coff = -1;
                    bool tr = false;
                    
                    double* blk = nullptr;
                    c_dbcsr_iterator_next_2d_block_d(iter, &i, &j, &blk, &tr, &nblk, &rsize, &csize, &roff, &coff);  
                   
                    std::generate(blk, blk + rsize*csize, [&](){ return static_cast<double>(std::rand())/RAND_MAX; });
                    
        }
        
        c_dbcsr_iterator_stop(&iter);
        
    };
    

    // create and fill matrix a
    void* matrix_a = nullptr;
    void* matrix_b = nullptr;
    void* matrix_c = nullptr;
    
    if (mpi_rank == 0) std::cout << "Creating matrices..." << std::endl;

        
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
    
    std::vector<int> irblks_1 = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
    std::vector<int> icblks_1 = {0, 1, 2, 4, 0, 2, 3, 1, 3, 4, 0, 1, 2};
    
    std::vector<int> irblks_2 = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4};
    std::vector<int> icblks_2 = {0, 2, 3, 0, 1, 2, 3, 0, 2, 3, 1, 2, 3, 0, 1, 2, 3};
    
    std::vector<int> irblks_3 = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3};
    std::vector<int> icblks_3 = {0, 1, 2, 3, 0, 2, 3, 1, 2, 3, 0, 1, 2, 3};

    if (mpi_rank == 0) std::cout << "Filling matrices..." << std::endl;

    fill_matrix(matrix_a, irblks_1, icblks_1);
    c_dbcsr_finalize(matrix_a);
    fill_matrix(matrix_b, irblks_2, icblks_2);
    c_dbcsr_finalize(matrix_b);
    fill_matrix(matrix_c, irblks_3, icblks_3);
    c_dbcsr_finalize(matrix_c);
    
    if (mpi_rank == 0) std::cout << "Multiplying..." << std::endl;
    
    c_dbcsr_multiply_d('N', 'N', 1.0d, matrix_a, matrix_b, 2.0d, matrix_c, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    
    if (mpi_rank == 0) std::cout << "Testing get_info for matrix_c" << std::endl;
    
    int nblkrowstot(0), nblkcolstot(0),
        nfullrowstot(0), nfullcolstot(0), nblkrowsloc(0), nblkcolsloc(0), 
        nfullrowsloc(0), nfullcolsloc(0), my_prow(0), my_pcol(0);
        
    //int *local_rows, *local_cols, *proc_row_dist, *proc_col_dist, 
    //    *row_blk_size, *col_blk_size, *row_blk_offset, *col_blk_offset;
    
    std::vector<int> local_rows(c_dbcsr_nblkrows_local(matrix_a));
    std::vector<int> local_cols(c_dbcsr_nblkcols_local(matrix_a));
    std::vector<int> proc_row(c_dbcsr_nblkrows_total(matrix_a));
    std::vector<int> proc_col(c_dbcsr_nblkcols_total(matrix_a));
    std::vector<int> row_blk(c_dbcsr_nblkrows_total(matrix_a));
    std::vector<int> col_blk(c_dbcsr_nblkcols_total(matrix_a));
    std::vector<int> row_off(c_dbcsr_nblkrows_total(matrix_a));
    std::vector<int> col_off(c_dbcsr_nblkcols_total(matrix_a));
    
    char* name;
    char matrix_type;
    int data_type;
    
    c_dbcsr_get_info(matrix_c, &nblkrowstot, &nblkcolstot,
                     &nfullrowstot, &nfullcolstot, &nblkrowsloc, &nblkcolsloc, 
                     &nfullrowsloc, &nfullcolsloc, &my_prow, &my_pcol, 
                     local_rows.data(), local_cols.data(), proc_row.data(), proc_col.data(),
                     row_blk.data(), col_blk.data(), nullptr, nullptr, 
                     nullptr, &name, &matrix_type, &data_type, nullptr);
                     
    
    auto printv = [](std::vector<int>& v) {
		for (auto x : v) {
			std::cout << x << " ";
		} std::cout << std::endl;
	}; 
	
	#define print_var(name) \
     std::cout << #name << ": " << name << std::endl;
	
	#define print_vec(name) \
	 std::cout << #name << ": " << std::endl; \
	 printv(name);
	 
	
	   
    if (mpi_rank == 0) {
	
	    std::cout << "Name: " << name << std::endl;
	
		print_var(nblkrowstot) 
		print_var(nblkcolstot) 
		print_var(nfullrowstot) 
		print_var(nfullcolstot) 
		print_var(nblkrowsloc)
		print_var(nblkcolsloc) 
		print_var(nfullrowsloc) 
		print_var(nfullcolsloc)
		
		print_vec(local_rows)
		print_vec(local_cols)
		print_vec(proc_row)
		print_vec(proc_col)
		print_vec(row_blk)
		print_vec(col_blk)
		print_vec(row_off)
		print_vec(col_off)
	} 
	
	c_free_string(&name);
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	if (mpi_rank == 0) std::cout << "Testing distribution_get for dist1" << std::endl;
	
	int* row_dist, *col_dist, *pgrid;
	int nrows, ncols, mynode, numnodes, nprows, 
        npcols, myprow, mypcol, prow_group, pcol_group;
    bool has_threads, subgroups_defined;
    MPI_Comm cgroup;
	
	c_dbcsr_distribution_get(dist1, &row_dist, &col_dist, 
                                  &nrows, &ncols, &has_threads, 
                                  &cgroup, &mynode, &numnodes, &nprows, 
                                  &npcols, &myprow, &mypcol, &pgrid, 
                                  &subgroups_defined, &prow_group, &pcol_group);
                                  
    if (mpi_rank == 0) {
		
		print_var(nrows)
		print_var(ncols) 
		print_var(mynode)
		print_var(numnodes) 
		print_var(nprows) 
        print_var(npcols) 
        print_var(myprow) 
        print_var(mypcol)
        print_var(prow_group) 
        print_var(pcol_group)
        
        if (cgroup == group) 
			std::cout << "Correct MPI communicator." << std::endl;
        
        std::cout << "dist row:" << std::endl;
        for (int i = 0; i != nrows; ++i) {
			std::cout << row_dist[i] << " ";
		} std::cout << std::endl;
		std::cout << "dist col:" << std::endl;
		for (int i = 0; i != ncols; ++i) {
			std::cout << col_dist[i] << " ";
		} std::cout << std::endl;
		
		std::cout << "grid: " << std::endl;
		for (int i = 0; i != nprows; ++i) {
			for (int j = 0; j != npcols; ++j) {
				std::cout << pgrid[i + nprows*j] << " ";
			} std::cout << std::endl;
		}
		
	}
	
	MPI_Barrier(MPI_COMM_WORLD);

    c_dbcsr_release(&matrix_a);
    c_dbcsr_release(&matrix_b);
    c_dbcsr_release(&matrix_c);

    c_dbcsr_distribution_release(&dist1);
    c_dbcsr_distribution_release(&dist2);
    c_dbcsr_distribution_release(&dist3);
    
    if (mpi_rank == 0) std::cout << "Extracting block diagonal..." << std::endl;
    
    std::vector<int> blk_sizes = {3,3,3};
    
    auto rowdist = random_dist(blk_sizes.size(),dims[0]);
    auto coldist = random_dist(blk_sizes.size(),dims[1]);
    
    void* dist4 = nullptr;
    
    c_dbcsr_distribution_new(&dist4, group,
        rowdist.data(), rowdist.size(),
        coldist.data(), coldist.size());
    
    void* matrix_d = nullptr;    
    
    c_dbcsr_create_new(&matrix_d, "matrix d", dist4, dbcsr_type_no_symmetry, 
                               blk_sizes.data(), blk_sizes.size(), 
                               blk_sizes.data(), blk_sizes.size(), 
                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr); 
                               
    c_dbcsr_init_random(matrix_d,nullptr);
    
    c_dbcsr_print(matrix_d);
    
    std::vector<double> alpha(9);
    for (size_t i = 0; i != alpha.size(); ++i) {
		alpha[i] = (double)i;
	}
    c_dbcsr_scale_by_vector_d (matrix_d, alpha.data(), alpha.size(), "right");
    
    c_dbcsr_print(matrix_d);
    
    double* data = nullptr;
    double type = 1.0d;
    int data_size = 0;
    c_dbcsr_get_data_d(matrix_d, &data, &data_size, &type, nullptr, nullptr);
    
    if (mpi_rank == 0) {
		std::cout << "Data on rank 0:" << std::endl;
		for (int i = 0; i != data_size; ++i) {
			std::cout << data[i] << " ";
		} std::cout << std::endl;
	}
		
    void* diag = nullptr;
    c_dbcsr_get_block_diag(matrix_d, &diag);
    
    c_dbcsr_print(diag);
    
    c_dbcsr_release(&matrix_d);
    c_dbcsr_release(&diag);
    c_dbcsr_distribution_release(&dist4);

    c_dbcsr_print_statistics(nullptr, nullptr);

    MPI_Comm_free(&group);

    c_dbcsr_finalize_lib();

    MPI_Finalize();

    return 0;
}
