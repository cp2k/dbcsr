#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <random>
#include <mpi.h>
#include <dbcsr.h>
#include <dbcsr_tensor.h>

//-------------------------------------------------------------------------------------------------!
// Example: tensor contraction (13|2)x(54|21)=(3|45)
//                             tensor1 x tensor2 = tensor3
//-------------------------------------------------------------------------------------------------!

std::vector<int> random_dist(int dist_size, int nbins)
{
	
    std::vector<int> dist(dist_size);

    for(int i=0; i < dist_size; i++)
        dist[i] = i % nbins;

    return std::move(dist);
}

void printvec(std::vector<int>& v) {
	
	for (auto i : v) {
		std::cout << i << " ";
	}
	std::cout << '\n' << std::endl;
	
}

void fill_random(void* tensor, std::vector<std::vector<int>> nzblocks) {
	
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
	
	/*
	for (int i = 0; i != mpi_size; ++i) {
		if (i == myrank) {
			std::cout << "Blocks stored on processor " << myrank << " :" << std::endl;
			for (auto idx : mynzblocks) {
				for (auto ele : idx) {
					std::cout << ele << " ";
				}
				std::cout << std::endl;
			}
			std::cout << "Total: " << mynzblocks[0].size() << " " <<  mynzblocks[1].size() << " " <<  mynzblocks[2].size() << std::endl;
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		
	}
	*/
	
	std::vector<int*> dataptr(4, nullptr);
	
	for (int i = 0; i != dim; ++i) {
		dataptr[i] = mynzblocks[i].size() == 0 ? nullptr : &mynzblocks[i][0];
	} 
	
	if (myrank == 0) std::cout << "Reserving blocks..." << std::endl;

    if (mynzblocks[0].size() != 0) 
	c_dbcsr_t_reserve_blocks_index(tensor, mynzblocks[0].size(), dataptr[0], dataptr[1], dataptr[2], dataptr[3]);
	
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    auto fill_rand = [&](std::vector<double>& blk) {
		int val = 0;
		for (double& e : blk) {
			e = dis(gen);
			//std::cout << e << std::endl;
		}
	};
	
	void* iter = nullptr;
    
    c_dbcsr_t_iterator_start(&iter, tensor);
  
    std::vector<int> loc_idx(dim);
    std::vector<int> blk_sizes(dim);
    std::vector<double> block(1);

    int n_b = 0;
    int blk = 0;
    int blk_proc = 0;
    
    // really like this???
    while(c_dbcsr_t_iterator_blocks_left(iter)) {
		
		//std::cout << "Block " << n_b++ << std::endl;
		
		c_dbcsr_t_iterator_next_block(iter, loc_idx.data(), &blk, &blk_proc, blk_sizes.data(), nullptr);
		
		//std::cout << "Blk: " << blk << std::endl;
		//std::cout << "Blk_p: " << blk_proc << std::endl;
		
		//for (auto i : blk_sizes) std::cout << i << " ";
		//std::cout << std::endl;
		
		//std::cout << "Index: " << std::endl;
		//for (auto i : loc_idx) std::cout << i << " ";
		//std::cout << std::endl;
		
		//std::cout << "Generating Block..." << std::endl;
		int tot = 1;
		for (int i = 0; i != dim; ++i) {
			tot *= blk_sizes[i];
		}
				
		block.resize(tot);
				
		fill_rand(block);
					
		c_dbcsr_t_put_block(tensor, loc_idx.data(), blk_sizes.data(), block.data(), nullptr, nullptr);
	
	}
			
	c_dbcsr_t_iterator_stop(&iter);
	
	MPI_Barrier(MPI_COMM_WORLD);	
	
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
		
	MPI_Barrier(MPI_COMM_WORLD);
	
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
	
	 MPI_Barrier(MPI_COMM_WORLD);
	
	// fill the tensors
	
	if (mpi_rank == 0) std::cout << "Tensor 1" << '\n' << std::endl;
	fill_random(tensor1, {nz11, nz12, nz13});
	if (mpi_rank == 0) std::cout << "Tensor 2" << '\n' << std::endl;
	fill_random(tensor2, {nz21, nz22, nz24, nz25});
	if (mpi_rank == 0) std::cout << "Tensor 3" << '\n' << std::endl;
	fill_random(tensor3, {nz33, nz34, nz35});
	
	// contracting
	
	// (13|2)x(54|21)=(3|45)
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	if (mpi_rank == 0) std::cout << "Contracting..." << std::endl;
	
	// cn : indices to be contracted
	// noncn : indices not to be contracted
	// mapn : how nonc indices map to tensor 3
	std::vector<int> c1, nonc1, c2, nonc2, map1, map2;
	c1 		= {0,1};
	nonc1 	= {2};
	c2 		= {0,1};
	nonc2	= {2,3};
	map1	= {0};
	map2	= {1,2};
	
	int unit_nr = 0;
	if (mpi_rank == 0) unit_nr = 6;
	bool log_verbose = true;
	
	// tensor_3(map_1, map_2) := 0.2 * tensor_1(notcontract_1, contract_1)
    //                                 * tensor_2(contract_2, notcontract_2)
    //                                 + 0.8 * tensor_3(map_1, map_2)
    
	c_dbcsr_t_contract_r_dp (0.2, tensor1, tensor2, 0.8, tensor3, c1.data(), c1.size(), nonc1.data(), nonc1.size(),
									c2.data(), c2.size(), nonc2.data(), nonc2.size(), map1.data(), map1.size(), map2.data(),
									map2.size(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
									nullptr, nullptr, &unit_nr, &log_verbose);
                                                         	
	// finalizing
	
	std::vector<int> nblkstot(3), nfulltot(3), nblksloc(3), nfullloc(3), pdims(3), ploc(3);
	
	int *bloc1 = nullptr, *bloc2 = nullptr, *bloc3 = nullptr;
	int bloc1size = 0, bloc2size = 0, bloc3size = 0;
	
	int *proc1 = nullptr, *proc2 = nullptr, *proc3 = nullptr;
	int proc1size = 0, proc2size = 0, proc3size = 0;
	
	int *blk_size1 = nullptr, *blk_size2 = nullptr, *blk_size3 = nullptr;
	int blk_size1size = 0, blk_size2size = 0, blk_size3size = 0;
	
	int *blk_off1 = nullptr, *blk_off2 = nullptr, *blk_off3 = nullptr;
	int blk_off1size = 0, blk_off2size = 0, blk_off3size = 0;
	
	void* dist = nullptr;
	char* name = nullptr;
	int name_size = 0;
	int data_type = 0;
	
	c_dbcsr_t_get_info(tensor1, 3, nblkstot.data(), nfulltot.data(),nblksloc.data(),
							   nfullloc.data(), pdims.data(), ploc.data(),
                               &bloc1, &bloc1size, &bloc2, &bloc2size, &bloc3, &bloc3size, nullptr, 0, 
                               &proc1, &proc1size, &proc2, &proc2size, &proc3, &proc3size, nullptr, 0, 
                               &blk_size1, &blk_size1size, &blk_size2, &blk_size2size, &blk_size3, &blk_size3size, nullptr, 0,
                               &blk_off1, &blk_off1size, &blk_off2, &blk_off2size, &blk_off3, &blk_off3size, nullptr, 0, 
                               &dist, &name, &name_size, &data_type);
                               
    
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
			printvec(nblkstot);
			
			std::cout << "Total number of elements:" << std::endl;
			printvec(nfulltot);
			
			std::cout << "Total number of local blocks:" << std::endl;
			printvec(nblksloc);
			
			std::cout << "Total number of local elements:" << std::endl;
			printvec(nfullloc);
			
			std::cout << "Pgrid dimensions:" << std::endl;
			printvec(pdims);
			
			std::cout << "Process coordinates:" << std::endl;
			printvec(ploc);
			
			std::cout << "blks_local:" << std::endl;
			for (int i = 0; i != bloc1size; ++i) {
				std::cout << bloc1[i] << " ";
			}
			std::cout << std::endl;
			for (int i = 0; i != bloc2size; ++i) {
				std::cout << bloc2[i] << " ";
			}
			std::cout << std::endl;
			for (int i = 0; i != bloc3size; ++i) {
				std::cout << bloc3[i] << " ";
			}
			std::cout << std::endl;
			
			free(bloc1);
			free(bloc2);
			free(bloc3);
			
			std::cout << "proc_dist:" << std::endl;
			for (int i = 0; i != proc1size; ++i) {
				std::cout << proc1[i] << " ";
			}
			std::cout << std::endl;
			for (int i = 0; i != proc2size; ++i) {
				std::cout << proc2[i] << " ";
			}
			std::cout << std::endl;
			for (int i = 0; i != proc3size; ++i) {
				std::cout << proc3[i] << " ";
			}
			std::cout << std::endl;
			
			free(proc1);
			free(proc2);
			free(proc3);
			
			std::cout << "blk_size:" << std::endl;
			for (int i = 0; i != blk_size1size; ++i) {
				std::cout << blk_size1[i] << " ";
			}
			std::cout << std::endl;
			for (int i = 0; i != blk_size2size; ++i) {
				std::cout << blk_size2[i] << " ";
			}
			std::cout << std::endl;
			for (int i = 0; i != blk_size3size; ++i) {
				std::cout << blk_size3[i] << " ";
			}
			std::cout << std::endl;
			
			free(blk_size1);
			free(blk_size2);
			free(blk_size3);
			
			std::cout << "blk_offset:" << std::endl;
			for (int i = 0; i != blk_off1size; ++i) {
				std::cout << blk_off1[i] << " ";
			}
			std::cout << std::endl;
			for (int i = 0; i != blk_off2size; ++i) {
				std::cout << blk_off2[i] << " ";
			}
			std::cout << std::endl;
			for (int i = 0; i != blk_off3size; ++i) {
				std::cout << blk_off3[i] << " ";
			}
			std::cout << std::endl;
			
			free(blk_off1);
			free(blk_off2);
			free(blk_off3);
			
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		
	}

	free(name);
	
	void* ndblk = nullptr;
	
	c_dbcsr_t_get_nd_index_blk(tensor1, &ndblk);
	
	int ndim_nd = 0, ndim1_2d = 0, ndim2_2d = 0;
	std::vector<long long int> dims_2d_i8(2);
	std::vector<int> dims_2d(2);
	
	int dims_nd_size, dims1_2d_size, dims2_2d_size, map1_2d_size, map2_2d_size, map_nd_size;
	int *dims_nd, *dims1_2d, *dims2_2d, *map1_2d, *map2_2d, *map_nd;
	
	int base;
	bool col_major;
	
	c_dbcsr_t_get_mapping_info(ndblk, &ndim_nd, &ndim1_2d, &ndim2_2d, 
                        dims_2d_i8.data(), dims_2d.data(), &dims_nd, &dims_nd_size, 
                        &dims1_2d, &dims1_2d_size, &dims2_2d, &dims2_2d_size, 
                        &map1_2d, &map1_2d_size, &map2_2d, &map2_2d_size, 
                        &map_nd, &map_nd_size, &base, &col_major);
                        
     if (mpi_rank == 0) {
		std::cout << "Testing get_mapping_info for Tensor 1..." << std::endl;
		
		std::cout << "ndim_nd = " << ndim_nd << std::endl;
		std::cout << "ndim1_2d = " << ndim1_2d << std::endl;
		std::cout << "ndim2_2d = " << ndim2_2d << std::endl;
		
		std::cout << "dims_2d_i8: ";
		for (auto i : dims_2d_i8) {
			std::cout << i << " ";
		} std::cout << std::endl;
		
		std::cout << "dims_2d: ";
		for (auto i : dims_2d) {
			std::cout << i << " ";
		} std::cout << std::endl;
		
		auto print_info = [] (int* ptr, int size, std::string name) 
		{
			std::cout << name << ": ";
			for (int i = 0; i != size; ++i) {
				std::cout << ptr[i] << " ";
			} std::cout << std::endl;
		};	
		
		print_info(dims_nd, dims_nd_size, "dims_nd");
		print_info(dims1_2d, dims1_2d_size, "dims1_2d");
		print_info(dims2_2d, dims2_2d_size, "dims1_2d");
		print_info(map1_2d, map1_2d_size, "map1_2d");
		print_info(map2_2d, map2_2d_size, "map2_2d");
		print_info(map_nd, map_nd_size, "map_nd");
		
		std::cout << "Base: " << base << std::endl;
		std::cout << "col_major " << col_major << std::endl;		
		
	}                   
    
                        
    free(dims_nd);
    free(dims1_2d);
    free(dims2_2d);
    free(map1_2d);
    free(map2_2d);
    free(map_nd);

    c_dbcsr_t_destroy(&tensor1);
    c_dbcsr_t_destroy(&tensor2);
    c_dbcsr_t_destroy(&tensor3);
        
    c_dbcsr_t_pgrid_destroy(&pgrid_3d, nullptr);
    c_dbcsr_t_pgrid_destroy(&pgrid_4d, nullptr);
    
    c_dbcsr_t_distribution_destroy(&dist1);
    c_dbcsr_t_distribution_destroy(&dist2);
    c_dbcsr_t_distribution_destroy(&dist3);

    c_dbcsr_finalize_lib();

    MPI_Finalize();


    return 0;
}
