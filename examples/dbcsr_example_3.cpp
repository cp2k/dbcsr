
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
        dist[i] = (nbins-i+1) % nbins;

    return std::move(dist);
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

    std::cout
        << "I'm processor " << mpi_rank
        << " over " << mpi_size << " proc"
        << ", (" << coord[0] << ", " << coord[1] << ") in the 2D grid"
        << std::endl;

    c_dbcsr_init_lib();

    // Total number of blocks
    int nblkrows_total = 4;
    int nblkcols_total = 4;

    // Block sizes
    std::vector<int> row_blk_sizes(nblkrows_total, 2), col_blk_sizes(nblkcols_total, 2);

    auto row_dist = random_dist(nblkrows_total, dims[0]);
    auto col_dist = random_dist(nblkcols_total, dims[1]);

    void* dist = nullptr;

    c_dbcsr_distribution_new(&dist, group,
        row_dist.data(), row_dist.size(),
        col_dist.data(), col_dist.size());

    // Fill all blocks, i.e. dense matrices
    auto fill_matrix = [&](void*& matrix)
    {
        int max_row_size = *std::max_element(row_blk_sizes.begin(),row_blk_sizes.end());
        int max_col_size = *std::max_element(col_blk_sizes.begin(),col_blk_sizes.end());
        int max_nze = max_row_size * max_col_size;

        std::vector<double> block;
        block.reserve(max_nze);

        for(int i = 0; i < nblkrows_total; i++)
        {
            for(int j = 0; j < nblkcols_total; j++)
            {
                int blk_proc = -1;
                c_dbcsr_get_stored_coordinates(matrix, i, j, &blk_proc);

                if(blk_proc == mpi_rank)
                {
                    block.resize(row_blk_sizes[i] * col_blk_sizes[j]);
                    std::generate(block.begin(), block.end(), [&](){ return static_cast<double>(std::rand())/RAND_MAX; });
                    c_dbcsr_put_block_d(matrix, i, j, block.data(), block.size());
                }
            }
        }
    };

    // create and fill matrix a
    void* matrix_a = nullptr;
    c_dbcsr_create_new_d(&matrix_a, "this is my matrix a", dist, 'N',
        row_blk_sizes.data(), row_blk_sizes.size(),
        col_blk_sizes.data(), col_blk_sizes.size());
    fill_matrix(matrix_a);
    c_dbcsr_finalize(matrix_a);

    // create and fill matrix b
    void* matrix_b = nullptr;
    c_dbcsr_create_new_d(&matrix_b, "this is my matrix b", dist, 'N',
        row_blk_sizes.data(), row_blk_sizes.size(),
        col_blk_sizes.data(), col_blk_sizes.size());
    fill_matrix(matrix_b);
    c_dbcsr_finalize(matrix_b);

    // create matrix c, empty
    void* matrix_c = nullptr;
    c_dbcsr_create_new_d(&matrix_c, "matrix c", dist, 'N',
        row_blk_sizes.data(), row_blk_sizes.size(),
        col_blk_sizes.data(), col_blk_sizes.size());
    c_dbcsr_finalize(matrix_c);

    // multiply the matrices
    c_dbcsr_multiply_d('N', 'N', 1.0, &matrix_a, &matrix_b, 0.0, &matrix_c, nullptr);

    c_dbcsr_print(matrix_a);
    c_dbcsr_print(matrix_b);
    c_dbcsr_print(matrix_c);

    c_dbcsr_release(&matrix_a);
    c_dbcsr_release(&matrix_b);
    c_dbcsr_release(&matrix_c);

    c_dbcsr_distribution_release(&dist);

    c_dbcsr_finalize_lib(group);

    MPI_Comm_free(&group);
    MPI_Finalize();

    return 0;
}
