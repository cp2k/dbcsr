/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <cstdint>
#include <random>
#include <mpi.h>
#include <dbcsr.h>
#include <tensors/dbcsr_tensor.h>
#include <complex.h>

//-------------------------------------------------------------------------------------------------!
// Testing the tensor contraction (13|2)x(54|21)=(3|45)
// and several other functions, to make sure there are not any segmentation faults
//-------------------------------------------------------------------------------------------------!

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-1.0, 1.0);

template <typename T>
T get_rand_real() {

    return dis(gen);

}

template <typename T>
T get_rand_complex() {

    return dis(gen) + dis(gen) * I;

}

std::vector<int> random_dist(int dist_size, int nbins)
{

    std::vector<int> dist(dist_size);

    for(int i=0; i < dist_size; i++)
        dist[i] = i % nbins;

    return dist;
}

template <typename T>
void printvec(T& v) {

    for (auto i : v) {
        std::cout << i << " ";
    }
    std::cout << '\n' << std::endl;

}

template <typename T>
void fill_random(void* tensor, std::vector<std::vector<int>> nzblocks,
    std::function<T()>& rand_func) {

    int myrank, mpi_size;
    int dim = nzblocks.size();

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (myrank == 0) std::cout << "Filling Tensor..." << std::endl;
    if (myrank == 0) std::cout << "Dimension: " << dim << std::endl;

    int nblocks = nzblocks[0].size();
    std::vector<std::vector<int>> mynzblocks(dim);
    std::vector<int> idx(dim);

    for (int i = 0; i != nblocks; ++i) {

        // make index out of nzblocks
        for (int j = 0; j != dim; ++j) idx[j] = nzblocks[j][i];

        int proc = -1;

        c_dbcsr_t_get_stored_coordinates(tensor, idx.data(), &proc);

        if (proc == myrank) {
            for (int j = 0; j != dim; ++j)
                mynzblocks[j].push_back(idx[j]);
        }

    }

    std::vector<int*> dataptr(4, nullptr);

    for (int i = 0; i != dim; ++i) {
        dataptr[i] = mynzblocks[i].size() == 0 ? nullptr : &mynzblocks[i][0];
    }

    if (myrank == 0) std::cout << "Reserving blocks..." << std::endl;

    if (mynzblocks[0].size() != 0)
    c_dbcsr_t_reserve_blocks_index(tensor, mynzblocks[0].size(), dataptr[0], dataptr[1], dataptr[2], dataptr[3]);

    auto fill_rand = [&](std::vector<T>& blk) {
        for (T& e : blk) {
            e = rand_func();
            //std::cout << e << std::endl;
        }
    };

    void* iter = nullptr;

    c_dbcsr_t_iterator_start(&iter, tensor);

    std::vector<int> loc_idx(dim);
    std::vector<int> blk_sizes(dim);
    std::vector<T> block(1);

    int blk = 0;
    int blk_proc = 0;

    while(c_dbcsr_t_iterator_blocks_left(iter)) {

        c_dbcsr_t_iterator_next_block(iter, loc_idx.data(), &blk, &blk_proc, blk_sizes.data(), nullptr);

        int tot = 1;
        for (int i = 0; i != dim; ++i) {
            tot *= blk_sizes[i];
        }

        block.resize(tot);

        fill_rand(block);

        c_dbcsr_t_put_block(tensor, loc_idx.data(), blk_sizes.data(), block.data(), nullptr, nullptr);

    }

    c_dbcsr_t_iterator_stop(&iter);

    c_dbcsr_t_finalize(tensor);

}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    c_dbcsr_init_lib(MPI_COMM_WORLD, nullptr);

    void* pgrid_3d = nullptr;
    void* pgrid_4d = nullptr;

    std::vector<int> dims4(4);
    std::vector<int> dims3(3);

    MPI_Fint fcomm = MPI_Comm_c2f(MPI_COMM_WORLD);

    c_dbcsr_t_pgrid_create(&fcomm, dims3.data(), dims3.size(), &pgrid_3d, nullptr);

    c_dbcsr_t_pgrid_create(&fcomm, dims4.data(), dims4.size(), &pgrid_4d, nullptr);

    if (mpi_rank == 0) {

        std::cout << "pgrid3-dimensions:" << std::endl;
        printvec(dims3);

        std::cout << "pgrid4-dimensions:" << std::endl;
        printvec(dims4);
    }


    // block sizes
    std::vector<int> blk1, blk2, blk3, blk4, blk5;
    // blk indices of non-zero blocks
    std::vector<int> nz11, nz12, nz13, nz21, nz22, nz24, nz25, nz33, nz34, nz35;

    blk1 = {3, 9, 12, 1};
    blk2 = {4, 2, 3, 1, 9, 2, 32, 10, 5, 8, 7};
    blk3 = {7, 3, 8, 7, 9, 5, 10, 23, 2};
    blk4 = {8, 1, 4, 13, 6};
    blk5 = {4, 2, 22};

    nz11 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
            3, 3};
    nz12 = {2, 4, 4, 4, 5, 5, 6, 7, 9,10,10,
            0, 0, 3, 6, 6, 8, 9 ,1, 1, 4, 5,
            7, 7, 8,10,10, 1 ,3, 4, 4, 7};
    nz13 = {6, 2, 4, 8, 5, 7, 1, 7, 2, 1, 2,
            0, 3, 5, 1, 6, 4, 7, 2, 6, 0, 3,
            2, 6, 7, 4, 7, 8, 5, 0, 1, 6};

    nz21 = { 0, 0, 0, 0, 0, 1, 1, 1,  1,  1,
             1, 1, 1, 1, 1, 1, 1, 1,  1,  1,
             2, 2, 2, 2, 2, 2, 2, 2,  2,  2,
             3, 3, 3, 3, 3, 3 };
    nz22 = { 0, 2, 3, 5, 9,  1, 1, 3,  4,  4,
             5, 5, 5, 6,  6,  8, 8, 8, 9, 10,
             0, 2, 2, 3,  4,  5, 7, 8, 10, 10,
             0, 2, 3, 5, 9, 10 };
    nz24 = { 2, 4, 1, 2,  1,  2, 4, 0,  0,  3,
             1, 2, 3, 0,  3,  2, 3, 3,  1,  0,
             2, 0, 0, 2,  3,  2, 3, 1,  1,  2,
             0, 0, 2, 1,  4,  4 };
    nz25 = { 0, 2, 1, 0,  0,  1, 2,  0,  2, 0,
             1, 2, 1, 0,  2,  1, 2,  1,  0, 1,
             2, 0, 1, 2,  1,  1, 1,  2,  0, 1,
             0, 2, 1, 0,  2,  1 };

    nz33 = { 1, 3, 4, 4, 4, 5, 5, 7 };
    nz34 = { 2, 1, 0, 0, 2, 1, 3, 4 };
    nz35 = { 2, 1, 0, 1, 2, 1, 0, 0 };

    // (13|2)x(54|21)=(3|45)
    // distribute blocks
    std::vector<int> dist11 = random_dist(blk1.size(), dims3[0]);
    std::vector<int> dist12 = random_dist(blk2.size(), dims3[1]);
    std::vector<int> dist13 = random_dist(blk3.size(), dims3[2]);

    std::vector<int> dist21 = random_dist(blk1.size(), dims4[0]);
    std::vector<int> dist22 = random_dist(blk2.size(), dims4[1]);
    std::vector<int> dist23 = random_dist(blk4.size(), dims4[2]);
    std::vector<int> dist24 = random_dist(blk5.size(), dims4[3]);

    std::vector<int> dist31 = random_dist(blk3.size(), dims3[0]);
    std::vector<int> dist32 = random_dist(blk4.size(), dims3[1]);
    std::vector<int> dist33 = random_dist(blk5.size(), dims3[2]);


    if (mpi_rank == 0) {

        std::cout << "dist11:" << std::endl;
        printvec(dist11);

        std::cout << "dist12:" << std::endl;
        printvec(dist12);

        std::cout << "dist13:" << std::endl;
        printvec(dist13);

        std::cout << "dist21:" << std::endl;
        printvec(dist21);

        std::cout << "dist22:" << std::endl;
        printvec(dist22);

        std::cout << "dist23:" << std::endl;
        printvec(dist23);

        std::cout << "dist24:" << std::endl;
        printvec(dist24);

        std::cout << "dist31:" << std::endl;
        printvec(dist31);

        std::cout << "dist32:" << std::endl;
        printvec(dist32);

        std::cout << "dist33:" << std::endl;
        printvec(dist33);

    }

    void* dist1 = nullptr;
    void* dist2 = nullptr;
    void* dist3 = nullptr;

    // (13|2)x(54|21)=(3|45)
    std::vector<int> map11, map12, map21, map22, map31, map32;

    map11 = {0, 2};
    map12 = {1};
    map21 = {3, 2};
    map22 = {1, 0};
    map31 = {0};
    map32 = {1, 2};

    if (mpi_rank == 0) std::cout << "Creating dist objects..." << '\n' << std::endl;

    // create distribution objects
    c_dbcsr_t_distribution_new(&dist1, pgrid_3d, dist11.data(), dist11.size(),
        dist12.data(), dist12.size(), dist13.data(), dist13.size(), nullptr, 0);

    c_dbcsr_t_distribution_new(&dist2, pgrid_4d, dist21.data(), dist21.size(),
        dist22.data(), dist22.size(), dist23.data(), dist23.size(),
        dist24.data(), dist24.size());

    c_dbcsr_t_distribution_new(&dist3, pgrid_3d, dist31.data(), dist31.size(),
        dist32.data(), dist32.size(), dist33.data(), dist33.size(), nullptr, 0);

    // create tensors
    // (13|2)x(54|21)=(3|45)

    void* tensor1 = nullptr;
    void* tensor2 = nullptr;
    void* tensor3 = nullptr;

    if (mpi_rank == 0) std::cout << "Creating tensors..." << std::endl;

    c_dbcsr_t_create_new(&tensor1, "(13|2)", dist1, map11.data(), map11.size(), map12.data(), map12.size(), nullptr, blk1.data(),
        blk1.size(), blk2.data(), blk2.size(), blk3.data(), blk3.size(), nullptr, 0);

    c_dbcsr_t_create_new(&tensor2, "(54|21)", dist2, map21.data(), map21.size(), map22.data(), map22.size(), nullptr, blk1.data(),
        blk1.size(), blk2.data(), blk2.size(), blk4.data(), blk4.size(), blk5.data(), blk5.size());

    c_dbcsr_t_create_new(&tensor3, "(3|45)", dist3, map31.data(), map31.size(), map32.data(), map32.size(), nullptr, blk3.data(),
        blk3.size(), blk4.data(), blk4.size(), blk5.data(), blk5.size(), nullptr, 0);

    // fill the tensors

    std::function<double()> drand = get_rand_real<double>;

    if (mpi_rank == 0) std::cout << "Tensor 1" << '\n' << std::endl;
    fill_random<double>(tensor1, {nz11, nz12, nz13}, drand);
    if (mpi_rank == 0) std::cout << "Tensor 2" << '\n' << std::endl;
    fill_random<double>(tensor2, {nz21, nz22, nz24, nz25}, drand);
    if (mpi_rank == 0) std::cout << "Tensor 3" << '\n' << std::endl;
    fill_random<double>(tensor3, {nz33, nz34, nz35}, drand);

    // contracting

    // (13|2)x(54|21)=(3|45)

    if (mpi_rank == 0) std::cout << "Contracting..." << std::endl;

    // cn : indices to be contracted
    // noncn : indices not to be contracted
    // mapn : how nonc indices map to tensor 3
    std::vector<int> c1, nonc1, c2, nonc2, map1, map2;
    c1         = {0,1};
    nonc1     = {2};
    c2         = {0,1};
    nonc2    = {2,3};
    map1    = {0};
    map2    = {1,2};


    int unit_nr = -1;
    if (mpi_rank == 0) unit_nr = 6;
    bool log_verbose = true;

    // tensor_3(map_1, map_2) := 0.2 * tensor_1(notcontract_1, contract_1)
    //                                 * tensor_2(contract_2, notcontract_2)
    //                                 + 0.8 * tensor_3(map_1, map_2)

    c_dbcsr_t_contract_r_dp (0.2, tensor1, tensor2, 0.8, tensor3, c1.data(), c1.size(), nonc1.data(), nonc1.size(),
                                    c2.data(), c2.size(), nonc2.data(), nonc2.size(), map1.data(), map1.size(), map2.data(),
                                    map2.size(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                    nullptr, nullptr, nullptr, &unit_nr, &log_verbose);

    // ====================================================
    // ====== TESTING OTHER FUNCTIONS =============
    // ====================================================

    // ======== GET_INFO ===========

    std::vector<int> cnblkstot(3), nfulltot(3), cnblksloc(3), nfullloc(3), pdims(3), ploc(3);

    char* name = nullptr;
    int data_type = 0;

    std::vector<int> nblksloc(3);
    std::vector<int> nblkstot(3);

    for (int i = 0; i != 3; ++i) {
        nblksloc[i] = c_dbcsr_t_nblks_local(tensor1,i);
        nblkstot[i] = c_dbcsr_t_nblks_total(tensor1,i);
    }

    std::vector<std::vector<int>> c_blks_local(3);
    std::vector<std::vector<int>> c_proc_dist(3);
    std::vector<std::vector<int>> c_blk_size(3);
    std::vector<std::vector<int>> c_blk_offset(3);
    for (int i = 0; i != 3; ++i) {
        c_blks_local[i].resize(nblksloc[i]);
        c_proc_dist[i].resize(nblkstot[i]);
        c_blk_size[i].resize(nblkstot[i]);
        c_blk_offset[i].resize(nblkstot[i]);
    }

    c_dbcsr_t_get_info(tensor1, 3, cnblkstot.data(), nfulltot.data(),cnblksloc.data(),
                               nfullloc.data(), pdims.data(), ploc.data(),
                               nblksloc[0],nblksloc[1],nblksloc[2],0,
                               nblkstot[0],nblkstot[1],nblkstot[2],0,
                               c_blks_local[0].data(), c_blks_local[1].data(), c_blks_local[2].data(), nullptr,
                               c_proc_dist[0].data(), c_proc_dist[1].data(), c_proc_dist[2].data(), nullptr,
                               c_blk_size[0].data(), c_blk_size[1].data(), c_blk_size[2].data(), nullptr,
                               c_blk_offset[0].data(), c_blk_offset[1].data(), c_blk_offset[2].data(), nullptr,
                               nullptr, &name, &data_type);

    std::string tname(name);

    if (mpi_rank == 0) {
        std::cout << "Testing get_info for Tensor 1..." << std::endl;
        std::cout << "Name: " << tname << std::endl;
        std::cout << "Data_type: " << data_type << std::endl;
    }

    for (int rank = 0; rank != mpi_size; ++rank) {
        if (rank == mpi_rank) {
            std::cout << "======= Process: " << rank << " ========" << std::endl;

            std::cout << "Total number of blocks:" << std::endl;
            printvec(cnblkstot);

            std::cout << "Total number of elements:" << std::endl;
            printvec(nfulltot);

            std::cout << "Total number of local blocks:" << std::endl;
            printvec(cnblksloc);

            std::cout << "Total number of local elements:" << std::endl;
            printvec(nfullloc);

            std::cout << "Pgrid dimensions:" << std::endl;
            printvec(pdims);

            std::cout << "Process coordinates:" << std::endl;
            printvec(ploc);

            std::cout << "blks_local:" << std::endl;
            for (int i = 0; i != 3; ++i) {
                printvec(c_blks_local[i]);
            }

            std::cout << "proc_dist:" << std::endl;
            for (int i = 0; i != 3; ++i) {
                printvec(c_proc_dist[i]);
            }

            std::cout << "blk_size:" << std::endl;
            for (int i = 0; i != 3; ++i) {
                printvec(c_blk_size[i]);
            }

            std::cout << "blk_offset:" << std::endl;
            for (int i = 0; i != 3; ++i) {
                printvec(c_blk_offset[i]);
            }

        }

        MPI_Barrier(MPI_COMM_WORLD);

    }

    // ================ GET_MAPPING_INFO ======================

    int ndim_nd = 0, ndim1_2d = 0, ndim2_2d = 0;
    std::vector<long long int> dims_2d_i8(2);
    std::vector<int> dims_2d(2);

    int nd_size = 3;
    int nd_row_size = c_dbcsr_t_ndims_matrix_row(tensor1);
    int nd_col_size = c_dbcsr_t_ndims_matrix_column(tensor1);

    std::vector<int> dims_nd(nd_size), dims1_2d(nd_row_size), dims2_2d(nd_col_size),
        map1_2d(nd_row_size), map2_2d(nd_col_size), map_nd(nd_size);

    int base;
    bool col_major;

    c_dbcsr_t_get_mapping_info(tensor1, 3, nd_row_size, nd_col_size, &ndim_nd, &ndim1_2d, &ndim2_2d,
                        dims_2d_i8.data(), dims_2d.data(), dims_nd.data(),
                        dims1_2d.data(), dims2_2d.data(),
                        map1_2d.data(), map2_2d.data(),
                        map_nd.data(), &base, &col_major);

     if (mpi_rank == 0) {
        std::cout << "Testing get_mapping_info for Tensor 1..." << std::endl;

        std::cout << "ndim_nd = " << ndim_nd << std::endl;
        std::cout << "ndim1_2d = " << ndim1_2d << std::endl;
        std::cout << "ndim2_2d = " << ndim2_2d << std::endl;

        std::cout << "dims_2d_i8: ";
        printvec(dims_2d_i8);

        std::cout << "dims_2d: ";
        printvec(dims_2d);

        std::cout << "dims_nd: " << std::endl;
        printvec(dims_nd);

        std::cout << "dims1_2d: " << std::endl;
        printvec(dims1_2d);

        std::cout << "dims2_2d: " << std::endl;
        printvec(dims2_2d);

        std::cout << "map1_2d: " << std::endl;
        printvec(map1_2d);

        std::cout << "map2_2d: " << std::endl;
        printvec(map2_2d);

        std::cout << "map_nd: " << std::endl;
        printvec(map_nd);

        std::cout << "Base: " << base << std::endl;
        std::cout << "col_major " << col_major << std::endl;

    }

    // ======== TESTING contract_index ================

    long long int rsize = c_dbcsr_t_max_nblks_local(tensor3);
    std::vector<int> result_index(rsize*3);
    int nblks_loc = 0;

    if (mpi_rank == 0) std::cout << "\n" << "Testing c_dbcsr_t_contract_index...\n" << std::endl;
    c_dbcsr_t_contract_index_r_dp (0.2, tensor1, tensor2, 0.8, tensor3, c1.data(), c1.size(), nonc1.data(), nonc1.size(),
                                    c2.data(), c2.size(), nonc2.data(), nonc2.size(), map1.data(), map1.size(), map2.data(),
                                    map2.size(), nullptr, nullptr, nullptr, nullptr, &nblks_loc, result_index.data(), rsize, 3);

    for (int ip = 0; ip != mpi_size; ++ip) {
        if (ip == mpi_rank) {

            std::cout << "Result Indices on Rank " << ip << std::endl;

            for (int i = 0; i != nblks_loc; ++i) {
                for (int n = 0; n != 3; ++n) {
                    std::cout << result_index[i + n*rsize] << " ";
                } std::cout << std::endl;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

    }

    c_dbcsr_t_destroy(&tensor1);
    c_dbcsr_t_destroy(&tensor2);
    c_dbcsr_t_destroy(&tensor3);

    c_dbcsr_t_distribution_destroy(&dist1);
    c_dbcsr_t_distribution_destroy(&dist2);
    c_dbcsr_t_distribution_destroy(&dist3);

    c_dbcsr_t_pgrid_destroy(&pgrid_3d, nullptr);
    c_dbcsr_t_pgrid_destroy(&pgrid_4d, nullptr);

    c_free_string(&name);

    c_dbcsr_finalize_lib();

    MPI_Finalize();

    return 0;
}
