#ifndef DBCSR_TENSOR_H
#define DBCSR_TENSOR_H

#include <mpi.h>
#include <stdbool.h> /* we need bool from C99 */

#:include "tensors/dbcsr_tensor.fypp"
#:set ndims = range(2,maxrank+1)
#:set ddims = range(1,maxrank+1)

#ifdef __cplusplus
extern "C" {
#endif
    void c_dbcsr_t_pgrid_create(MPI_Fint* fcomm, int* c_dims, int dims_size, void** c_pgrid, 
		int* c_map1_2d, int map1_2d_size, int* c_map2_2d, int map2_2d_size, int* nsplit, int* dimsplit);
	
	void c_dbcsr_t_pgrid_destroy(void** c_pgrid, bool* c_keep_comm);
																			  
	void c_dbcsr_t_distribution_new(void** c_dist, void* c_pgrid, int* c_map1_2d, int map1_2d_size,
		int* c_map2_2d, int map2_2d_size, ${extern_varlist_and_size("c_nd_dist")}$, bool* own_comm);
		
    void c_dbcsr_t_distribution_destroy(void** c_dist);
		
	void c_dbcsr_t_create_new(void** c_tensor, const char* c_name, void* c_dist, int* c_map1_2d, int c_map1_2d_size,
       int* c_map2_2d, int c_map2_2d_size, int* data_type, ${extern_varlist_and_size("c_blk_size")}$);
       
    void c_dbcsr_t_create_template(void* c_tensor_in, void** c_tensor, const char* c_name);
    
    void c_dbcsr_t_create_matrix(void* c_matrix_in, void** c_tensor, int* c_order, const char* c_name);
       
    void c_dbcsr_t_destroy(void** c_tensor);
                               
    int c_ndims_tensor(void* tensor);
                               
#:for dsuffix, ctype in c_dtype_float_list
#:for ndim in ndims

	  void c_dbcsr_t_allocate_and_get_${ndim}$d_block_${dsuffix}$ (void* c_tensor, int tensor_dim, int* c_ind, 
		${ctype}$** c_block, bool* c_found);
                    
      void c_dbcsr_t_get_${ndim}$d_block_${dsuffix}$ (void* c_tensor, int tensor_dim, int* c_ind, int* c_sizes, 
		${ctype}$* c_block, bool* c_found);
		
	  void c_dbcsr_t_put_${ndim}$d_block_${dsuffix}$ (void* c_tensor, int tensor_dim, int* c_ind, int*c_sizes, 
        ${ctype}$* c_block, bool* c_summation, ${ctype}$* c_scale);

#:endfor
     
     void c_dbcsr_t_contract_${dsuffix}$ (${ctype}$ c_alpha, void* c_tensor_1, void* c_tensor_2, ${ctype}$ c_beta, void* c_tensor_3, 
                               int* c_contract_1, int c_contract_1_size,
                               int* c_notcontract_1, int c_notcontract_1_size,
                               int* c_contract_2, int c_contract_2_size,
                               int* c_notcontract_2, int c_notcontract_2_size,
                               int* c_map_1, int c_map_1_size,
                               int* c_map_2, int c_map_2_size,
                               int* c_bounds_1, 
                               int* c_bounds_2, 
                               int* c_bounds_3, 
                               bool* c_optimize_dist, 
                               void** c_pgrid_opt_1, 
                               void** c_pgrid_opt_2, 
                               void** c_pgrid_opt_3,
                               double* filter_eps, long long int* flop, 
                               int* move_data, int* unit_nr, bool* log_verbose);
     
     void c_dbcsr_t_filter_${dsuffix}$ (void* c_tensor, ${ctype}$ c_eps, int* c_method, bool* c_use_absolute);
     
     void c_dbcsr_t_set_${dsuffix}$ (void* c_tensor, ${ctype}$ c_alpha);
	 
	 void c_dbcsr_t_scale_${dsuffix}$ (void* c_tensor, ${ctype}$ c_alpha);
	
#:endfor
 

     void c_dbcsr_t_get_stored_coordinates(void* c_tensor, int tensor_dim, int* c_ind_nd, int* c_processor);
   
     void c_dbcsr_t_reserve_blocks_index(void* c_tensor, int nblocks, ${varlist("int* c_blk_ind")}$);
     
     int c_ndims_iterator(void* c_iterator); 
     
	 void c_dbcsr_t_iterator_start(void** c_iterator, void* c_tensor);

     void c_dbcsr_t_iterator_stop(void** c_iterator);
   
     void c_dbcsr_t_iterator_next_block(void* c_iterator, int iterator_size, int* c_ind_nd, 
          int* c_blk, int* c_blk_p, int* c_blk_size, int* c_blk_offset);
          
     bool c_dbcsr_t_iterator_blocks_left(void* c_iterator);
                               
     void c_dbcsr_t_split_blocks(void* c_tensor_in, int tensor_dim, void** c_tensor_out, int* c_block_sizes, bool* c_nodata);
     
     void c_dbcsr_t_copy_matrix_to_tensor(void* c_matrix_in, void* c_tensor_out, bool* c_summation);
     
     void c_dbcsr_t_copy_tensor_to_matrix(void* c_tensor_in, void* c_matrix_out, bool* c_summation);
     
     void c_dbcsr_t_blk_sizes(void* c_tensor, int tensor_dim, int* c_ind, int* c_blk_size);
     
     void c_dbcsr_t_copy(void* c_tensor_in, int tensor_dim, void* c_tensor_out, int* c_order, bool* c_summation, 
      bool* c_move_data, int* c_unit_nr);
      
     void c_dbcsr_t_clear(void* c_tensor);
   
	
#ifdef __cplusplus
}
#endif

/* *************************************************************************************************
 * *                                                                                               *
 * *                                      OVERLOADED C FUNCTIONS                                   *
 * *                                                                                               *
 * *************************************************************************************************/

#:for dsuffix, ctype in c_dtype_float_list
static void c_dbcsr_t_get_block(void* c_tensor, int* c_ind, int* c_sizes, 
		${ctype}$* c_block, bool* c_found) {
	
	int tensor_dim = c_ndims_tensor(c_tensor);
	
	switch(tensor_dim) {
		#:for ndim in ndims
		case ${ndim}$: c_dbcsr_t_get_${ndim}$d_block_${dsuffix}$ (c_tensor, tensor_dim, 
		c_ind, c_sizes, c_block, c_found);
		break;
		#:endfor
	}
	
}
#:endfor

#:for dsuffix, ctype in c_dtype_float_list
static void c_dbcsr_t_get_block(void* c_tensor, int* c_ind, 
		${ctype}$** c_block, bool* c_found) {
	
	int tensor_dim = c_ndims_tensor(c_tensor);
	
	switch(tensor_dim) {
		#:for ndim in ndims
		case ${ndim}$: c_dbcsr_t_allocate_and_get_${ndim}$d_block_${dsuffix}$ (c_tensor, tensor_dim, 
		c_ind, c_block, c_found);
		break;
		#:endfor
	}
	
}
#:endfor

#:for dsuffix, ctype in c_dtype_float_list
static void c_dbcsr_t_put_block(void* c_tensor, int* c_ind, int* c_sizes, 
		${ctype}$* c_block, bool* c_summation, ${ctype}$* c_scale) {
	
	int tensor_dim = c_ndims_tensor(c_tensor);
	
	switch(tensor_dim) {
		#:for ndim in ndims
		case ${ndim}$: c_dbcsr_t_put_${ndim}$d_block_${dsuffix}$ (c_tensor, tensor_dim, 
		c_ind, c_sizes, c_block, c_summation, c_scale);
		break;
		#:endfor
	}
	
}
#:endfor

static void c_dbcsr_t_get_stored_coordinates(void* c_tensor, int* c_ind_nd, int* c_processor) {
	
	int tensor_dim = c_ndims_tensor(c_tensor);
	c_dbcsr_t_get_stored_coordinates(c_tensor, tensor_dim, c_ind_nd, c_processor);
	
}

/*
#:for ndim in range(2, maxrank)
static void c_dbcsr_t_reserve_blocks_index(void* c_tensor, ${extern_varlist_and_size("c_blk_ind", nmax=ndim)}$) {
	
	c_dbcsr_t_reserve_blocks_index(c_tensor, ${c_varlist_and_size("c_blk_ind", nmax=ndim)}$, 
#:for n in range(ndim+1,maxrank+1)
  #:if n == maxrank
    nullptr, 0);
  #:else
    nullptr, 0,
  #:endif
#:endfor
};
#:endfor
*/

static void c_dbcsr_t_iterator_next_block(void* c_iterator, int* c_ind_nd, 
          int* c_blk, int* c_blk_p, int* c_blk_size, int* c_blk_offset) {
			  
	int iterator_size = c_ndims_iterator(c_iterator);
	
	c_dbcsr_t_iterator_next_block(c_iterator, iterator_size, c_ind_nd, 
          c_blk, c_blk_p, c_blk_size, c_blk_offset);
          
}

#:for dsuffix, ctype in c_dtype_float_list
static void c_dbcsr_t_filter(void* c_tensor, ${ctype}$ c_eps, int* c_method, bool* c_use_absolute) {
	
	c_dbcsr_t_filter_${dsuffix}$ (c_tensor, c_eps, c_method, c_use_absolute);
	
}

static void c_dbcsr_t_set(void* c_tensor, ${ctype}$ c_alpha) {
   
   c_dbcsr_t_set_${dsuffix}$ (c_tensor, c_alpha);
   
}

static void c_dbcsr_t_scale(void* c_tensor, ${ctype}$ c_alpha) {
	
   c_dbcsr_t_scale_${dsuffix}$ (c_tensor, c_alpha);
   
}

#:endfor

static void* c_dbcsr_t_blk_sizes(void* c_tensor, int* c_ind, int* c_blk_size) {
	
	int tensor_dim = c_ndims_tensor(c_tensor);
	c_dbcsr_t_blk_sizes(c_tensor, tensor_dim, c_ind, c_blk_size);
	
}


#endif // DBCSR_H
