#:include "tensors/dbcsr_tensor.fypp"
#:set ndims = range(2,maxrank+1)
#:set ddims = range(1,maxrank+1)

MODULE dbcsr_tensor_api_c

   USE, INTRINSIC :: ISO_C_BINDING, ONLY: c_loc, c_ptr, c_double, C_NULL_CHAR, &
                                                                             c_float, c_f_pointer, c_int, c_long_long, &
                                                                             c_char, c_null_ptr, c_bool, &
                                                                             c_float_complex, c_double_complex
   USE dbcsr_api, ONLY: dbcsr_type, dbcsr_scale
   USE dbcsr_tensor, ONLY: &
      dbcsr_t_contract, dbcsr_t_get_block, dbcsr_t_get_stored_coordinates, dbcsr_t_put_block, &
      dbcsr_t_reserve_blocks, dbcsr_t_copy_matrix_to_tensor, dbcsr_t_copy, &
      dbcsr_t_copy_tensor_to_matrix
   USE dbcsr_tensor_block, ONLY: &
      dbcsr_t_iterator_blocks_left, dbcsr_t_iterator_next_block, dbcsr_t_iterator_start, &
      dbcsr_t_iterator_stop, dbcsr_t_iterator_type
   USE dbcsr_tensor_types, ONLY: &
      dbcsr_t_create, dbcsr_t_destroy, dbcsr_t_distribution_destroy, dbcsr_t_distribution_new, &
      dbcsr_t_distribution_type, dbcsr_t_nd_mp_comm, dbcsr_t_nd_mp_free, dbcsr_t_type, &
      dbcsr_t_pgrid_type, dbcsr_t_pgrid_create, dbcsr_t_pgrid_destroy, dbcsr_t_set, dbcsr_t_filter, &
      dbcsr_t_mp_environ_pgrid => mp_environ_pgrid, dbcsr_t_blk_sizes, dbcsr_t_get_info, &
      dbcsr_t_finalize, dbcsr_t_scale, dbcsr_t_get_nze, dbcsr_t_get_nze_total, &
      dbcsr_t_get_num_blocks, dbcsr_t_get_num_blocks_total, dbcsr_t_clear
   USE dbcsr_tensor_split, ONLY: &
      dbcsr_t_split_blocks
   USE dbcsr_kinds, ONLY: &
      ${uselist(dtype_float_prec)}$, default_string_length, int_8
   USE dbcsr_data_types, ONLY: dbcsr_scalar_type
   USE dbcsr_data_methods, ONLY: dbcsr_scalar
   USE dbcsr_machine, ONLY: default_output_unit

   IMPLICIT NONE
   PRIVATE

CONTAINS


   SUBROUTINE c_f_string(c_str, str)
      USE, INTRINSIC :: iso_c_binding, ONLY: c_ptr, c_f_pointer, c_char
      TYPE(c_ptr), INTENT(in) :: c_str
      CHARACTER(kind=c_char), POINTER :: arr(:)
      CHARACTER(:, kind=c_char), ALLOCATABLE, INTENT(out) :: str
      INTEGER(8) :: n, i
      INTERFACE
         ! steal std c library function rather than writing our own.
         FUNCTION strlen(s) bind(c, name='strlen')
            USE, INTRINSIC :: iso_c_binding, ONLY: c_ptr, c_size_t
            IMPLICIT NONE
            !----
            TYPE(c_ptr), INTENT(in), value :: s
            INTEGER(c_size_t) :: strlen
         END FUNCTION strlen
      END INTERFACE
      n = strlen(c_str)
      !****
      CALL c_f_pointer(c_str, arr, [n])
      ALLOCATE (CHARACTER(len=n) :: str)
      DO i = 1, n
         str(i:i) = arr(i)
      ENDDO
   END SUBROUTINE c_f_string


   SUBROUTINE c_dbcsr_t_pgrid_create(fcomm, c_dims, dims_size, c_pgrid, &
                                     c_map1_2d, map1_2d_size, c_map2_2d, map2_2d_size, nsplit, dimsplit) &
      bind(C, name="c_dbcsr_t_pgrid_create")

      INTEGER(kind=c_int), INTENT(in)                    :: fcomm
      INTEGER(kind=c_int), INTENT(in), value             :: dims_size
      TYPE(c_ptr), INTENT(out)                           :: c_pgrid
      INTEGER(kind=c_int), INTENT(inout), TARGET         :: c_dims(dims_size)
      INTEGER(kind=c_int), INTENT(in), VALUE             :: map1_2d_size
      INTEGER(kind=c_int), INTENT(in), TARGET, OPTIONAL  :: c_map1_2d(map1_2d_size)
      INTEGER(kind=c_int), INTENT(in), VALUE             :: map2_2d_size
      INTEGER(kind=c_int), INTENT(in), TARGET, OPTIONAL  :: c_map2_2d(map2_2d_size)
      INTEGER(kind=c_int), INTENT(in), OPTIONAL          :: nsplit, dimsplit

      INTEGER, POINTER                                   :: dims(:)
      INTEGER, ALLOCATABLE                               :: map1_2d(:), map2_2d(:)
      TYPE(dbcsr_t_pgrid_type), POINTER                  :: pgrid

      dims => c_dims

      IF (PRESENT(c_map1_2d) .AND. PRESENT(c_map2_2d)) THEN
         map1_2d = c_map1_2d+1
         map2_2d = c_map2_2d+1
         PRINT *, "PRESENT"
      ENDIF

      ALLOCATE (pgrid)

      CALL dbcsr_t_pgrid_create(fcomm, dims, pgrid, map1_2d, map2_2d, &
                                nsplit, dimsplit)

      c_pgrid = c_loc(pgrid)

      IF (ALLOCATED(map1_2d) .AND. ALLOCATED(map2_2d)) THEN
         DEALLOCATE (map1_2d)
         DEALLOCATE (map2_2d)
      ENDIF

   END SUBROUTINE


   SUBROUTINE c_dbcsr_t_pgrid_destroy(c_pgrid, c_keep_comm) &
      BIND(C, name="c_dbcsr_t_pgrid_destroy")

      TYPE(c_ptr), INTENT(INOUT)                         :: c_pgrid
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_keep_comm
      TYPE(dbcsr_t_pgrid_type), POINTER                  :: pgrid
      LOGICAL, ALLOCATABLE                               :: keep_comm

      CALL c_f_pointer(c_pgrid, pgrid)
      IF (PRESENT(c_keep_comm)) keep_comm = c_keep_comm

      CALL dbcsr_t_pgrid_destroy(pgrid, keep_comm)

      DEALLOCATE (pgrid)

      c_pgrid = c_null_ptr

   END SUBROUTINE


   SUBROUTINE c_dbcsr_t_distribution_new(c_dist, c_pgrid, c_map1_2d, map1_2d_size, &
                                         c_map2_2d, map2_2d_size, ${c_varlist_and_size("c_nd_dist")}$, c_own_comm) &
      bind(C, name="c_dbcsr_t_distribution_new")

      TYPE(c_ptr), INTENT(OUT)                           :: c_dist
      TYPE(c_ptr), INTENT(IN), value                     :: c_pgrid

      INTEGER(kind=c_int), INTENT(in), value             :: map1_2d_size
      INTEGER(kind=c_int), INTENT(in), TARGET            :: c_map1_2d(map1_2d_size)
      INTEGER(kind=c_int), INTENT(in), value             :: map2_2d_size
      INTEGER(kind=c_int), INTENT(in), TARGET            :: c_map2_2d(map2_2d_size)

      INTEGER(kind=c_int), INTENT(in), value             :: ${varlist("c_nd_dist", suffix="_size")}$
      INTEGER(kind=c_int), INTENT(IN), TARGET, OPTIONAL  :: ${c_arrlist_and_size("c_nd_dist")}$
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_own_comm

      INTEGER, POINTER                                   :: ${pointer_list("nd_dist")}$
      INTEGER, POINTER                                   :: map1_2d(:), map2_2d(:)
      TYPE(dbcsr_t_pgrid_type), POINTER                  :: pgrid
      TYPE(dbcsr_t_distribution_type), POINTER           :: dist
      LOGICAL, ALLOCATABLE                               :: own_comm

      map1_2d => c_map1_2d
      map2_2d => c_map2_2d

#:for ddim in ddims
      IF (PRESENT(c_nd_dist_${ddim}$)) THEN
         nd_dist_${ddim}$ => c_nd_dist_${ddim}$
      ELSE
         nd_dist_${ddim}$ => null()
      ENDIF

#:endfor

      IF (PRESENT(c_own_comm)) own_comm = c_own_comm

      ALLOCATE (dist)

      CALL c_f_pointer(c_pgrid, pgrid)

      CALL dbcsr_t_distribution_new(dist, pgrid, map1_2d+1, map2_2d+1, &
                                    ${varlist("nd_dist")}$, own_comm)

      c_dist = c_loc(dist)

   END SUBROUTINE


   SUBROUTINE c_dbcsr_t_distribution_destroy(c_dist) BIND(C, name="c_dbcsr_t_distribution_destroy")
      TYPE(c_ptr), INTENT(INOUT)                         :: c_dist
      TYPE(dbcsr_t_distribution_type), POINTER           :: dist

      CALL c_f_pointer(c_dist, dist); 
      CALL dbcsr_t_distribution_destroy(dist)

      DEALLOCATE (dist)

      c_dist = c_null_ptr

   END SUBROUTINE


   SUBROUTINE c_dbcsr_t_create_new(c_tensor, c_name, c_dist, c_map1_2d, map1_2d_size, &
                                   c_map2_2d, map2_2d_size, c_data_type, ${c_varlist_and_size("c_blk_size")}$) &
      bind(C, name="c_dbcsr_t_create_new")

      TYPE(c_ptr), INTENT(OUT)                           :: c_tensor
      TYPE(c_ptr), INTENT(IN), value                     :: c_dist, c_name

      INTEGER(kind=c_int), INTENT(in), value             :: map1_2d_size
      INTEGER(kind=c_int), INTENT(in), TARGET            :: c_map1_2d(map1_2d_size)
      INTEGER(kind=c_int), INTENT(in), value             :: map2_2d_size
      INTEGER(kind=c_int), INTENT(in), TARGET            :: c_map2_2d(map2_2d_size)

      INTEGER(kind=c_int), INTENT(IN), OPTIONAL          :: c_data_type

      INTEGER(kind=c_int), INTENT(in), value             :: ${varlist("c_blk_size", suffix="_size")}$
      INTEGER(kind=c_int), INTENT(IN), TARGET, OPTIONAL  :: ${c_arrlist_and_size("c_blk_size")}$

      CHARACTER(:, kind=c_char), ALLOCATABLE             :: fname

      INTEGER, POINTER                                   :: map1_2d(:), map2_2d(:)
      INTEGER, POINTER                                   :: ${pointer_list("blk_size")}$
      TYPE(dbcsr_t_type), POINTER                        :: tensor
      TYPE(dbcsr_t_distribution_type), POINTER           :: dist
      INTEGER, ALLOCATABLE :: data_type

      map1_2d => c_map1_2d
      map2_2d => c_map2_2d

#:for ddim in ddims
      IF (PRESENT(c_blk_size_${ddim}$)) THEN
         blk_size_${ddim}$ => c_blk_size_${ddim}$
      ELSE
         blk_size_${ddim}$ => null()
      ENDIF

#:endfor

      IF (PRESENT(c_data_type)) THEN
         data_type = c_data_type
      END IF

      ALLOCATE (tensor)

      CALL c_f_pointer(c_dist, dist)
      CALL c_f_string(c_name, fname)

      CALL dbcsr_t_create(tensor, fname, dist, map1_2d+1, map2_2d+1, data_type, &
                          ${varlist("blk_size")}$)

      c_tensor = c_loc(tensor)

   END SUBROUTINE


   SUBROUTINE c_dbcsr_t_create_template(c_tensor_in, c_tensor, c_name) &
      BIND(C, name="c_dbcsr_t_create_template")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_in
      TYPE(c_ptr), INTENT(OUT)                           :: c_tensor
      TYPE(c_ptr), INTENT(IN), OPTIONAL                  :: c_name

      TYPE(dbcsr_t_type), POINTER                        :: tensor_in
      TYPE(dbcsr_t_type), POINTER                        :: tensor
      CHARACTER(:, kind=c_char), ALLOCATABLE             :: fname

      CALL c_f_pointer(c_tensor_in, tensor_in)
      IF (PRESENT(c_name)) CALL c_f_string(c_name, fname)
      ALLOCATE (tensor)

      CALL dbcsr_t_create(tensor_in, tensor, fname)

      c_tensor = c_loc(tensor)

   END SUBROUTINE


   SUBROUTINE c_dbcsr_t_create_matrix(c_matrix_in, c_tensor, c_order, c_name) &
      BIND(C, name="c_dbcsr_t_create_matrix")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_matrix_in
      TYPE(c_ptr), INTENT(OUT)                           :: c_tensor
      INTEGER(kind=c_int), INTENT(IN), DIMENSION(2), &
        OPTIONAL                                         :: c_order
      TYPE(c_ptr), INTENT(IN), OPTIONAL                  :: c_name

      TYPE(dbcsr_type), POINTER                :: matrix_in
      TYPE(dbcsr_t_type), POINTER            :: tensor
      INTEGER, DIMENSION(2) :: order
      CHARACTER(:, kind=c_char), ALLOCATABLE   :: fname

      CALL c_f_pointer(c_matrix_in, matrix_in)
      IF (PRESENT(c_name)) CALL c_f_string(c_name, fname)

      IF (PRESENT(c_order)) THEN
         order = c_order + 1
      ELSE
         order = [1, 2]
      ENDIF

      ALLOCATE (tensor)

      CALL dbcsr_t_create(matrix_in, tensor, order, fname)

      c_tensor = c_loc(tensor)

   END SUBROUTINE


   SUBROUTINE c_dbcsr_t_destroy(c_tensor) BIND(C, name="c_dbcsr_t_destroy")
      TYPE(c_ptr), INTENT(INOUT)                         :: c_tensor
      TYPE(dbcsr_t_type), POINTER                        :: tensor

      CALL c_f_pointer(c_tensor, tensor)

      CALL dbcsr_t_destroy(tensor)

      DEALLOCATE (tensor)

      c_tensor = c_null_ptr

   END SUBROUTINE

#:for dparam, dtype, dsuffix, dbase, dctype in cf_dtype_float_list

   SUBROUTINE c_dbcsr_t_contract_${dsuffix}$ (c_alpha, c_tensor_1, c_tensor_2, c_beta, c_tensor_3, &
                                              c_contract_1, contract_1_size, &
                                              c_notcontract_1, notcontract_1_size, &
                                              c_contract_2, contract_2_size, &
                                              c_notcontract_2, notcontract_2_size, &
                                              c_map_1, map_1_size, c_map_2, map_2_size, &
                                              c_bounds_1, c_bounds_2, c_bounds_3, &
                                              c_optimize_dist, c_pgrid_opt_1, c_pgrid_opt_2, c_pgrid_opt_3, &
                                              c_filter_eps, c_flop, c_move_data, c_unit_nr, c_log_verbose) &
      BIND(C, name="c_dbcsr_t_contract_${dsuffix}$")

      ${dbase}$ (kind=${dctype}$), INTENT(IN), VALUE     ::  c_alpha
      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_1, c_tensor_2
      ${dbase}$ (kind=${dctype}$), INTENT(IN), VALUE     :: c_beta
      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_3

      INTEGER(kind=c_int), INTENT(IN), VALUE             :: contract_1_size, contract_2_size
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: notcontract_1_size, notcontract_2_size
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: map_1_size, map_2_size

      INTEGER(kind=c_int), INTENT(IN), TARGET            :: c_contract_1(contract_1_size), c_contract_2(contract_2_size)
      INTEGER(kind=c_int), INTENT(IN), TARGET            :: c_map_1(map_1_size), c_map_2(map_2_size)
      INTEGER(kind=c_int), INTENT(IN), TARGET            :: c_notcontract_1(notcontract_1_size), c_notcontract_2(notcontract_2_size)

      INTEGER(kind=c_int), INTENT(IN), DIMENSION(2, contract_1_size), &
        OPTIONAL, TARGET                                 :: c_bounds_1
      INTEGER(kind=c_int), INTENT(IN), DIMENSION(2, notcontract_1_size), &
        OPTIONAL, TARGET                                 :: c_bounds_2
      INTEGER(kind=c_int), INTENT(IN), DIMENSION(2, notcontract_2_size), & 
        OPTIONAL, TARGET                                 :: c_bounds_3

      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_optimize_dist
      TYPE(c_ptr), INTENT(OUT), OPTIONAL                 :: c_pgrid_opt_1, c_pgrid_opt_2, c_pgrid_opt_3

      REAL(kind=c_double), INTENT(IN), OPTIONAL          :: c_filter_eps
      INTEGER(kind=c_long_long), INTENT(IN), OPTIONAL    :: c_flop
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_move_data
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL          :: c_unit_nr
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_log_verbose

      TYPE(dbcsr_t_type), POINTER                        :: tensor_1, tensor_2
      INTEGER, DIMENSION(:), POINTER                     :: contract_1, contract_2, map_1, map_2
      INTEGER, DIMENSION(:), POINTER                     :: notcontract_1, notcontract_2
      TYPE(dbcsr_t_type), POINTER                        :: tensor_3
      INTEGER, DIMENSION(:, :), ALLOCATABLE              :: bounds_1
      INTEGER, DIMENSION(:, :), ALLOCATABLE              :: bounds_2
      INTEGER, DIMENSION(:, :), ALLOCATABLE              :: bounds_3
      LOGICAL, ALLOCATABLE                               :: optimize_dist
      TYPE(dbcsr_t_pgrid_type), POINTER                  :: pgrid_opt_1 => null(), &
                                               pgrid_opt_2 => null(), pgrid_opt_3 => null()

      REAL(KIND=real_8), ALLOCATABLE                     :: filter_eps
      INTEGER(KIND=int_8), ALLOCATABLE                   :: flop
      LOGICAL, ALLOCATABLE                               :: move_data
      INTEGER, ALLOCATABLE                               :: unit_nr
      LOGICAL, ALLOCATABLE                               :: log_verbose

      CALL c_f_pointer(c_tensor_1, tensor_1)
      CALL c_f_pointer(c_tensor_2, tensor_2)
      CALL c_f_pointer(c_tensor_3, tensor_3)

      contract_1 => c_contract_1
      contract_2 => c_contract_2
      map_1 => c_map_1
      map_2 => c_map_2
      notcontract_1 => c_notcontract_1
      notcontract_2 => c_notcontract_2

#:for dim in range(1,4)
      IF (PRESENT(c_bounds_${dim}$)) bounds_${dim}$ = c_bounds_${dim}$+1
      ALLOCATE (pgrid_opt_${dim}$)
#:endfor

#:set list = ['optimize_dist', 'filter_eps', 'flop', 'move_data', 'unit_nr', 'log_verbose']
#:for var in list
      IF (PRESENT(c_${var}$)) ${var}$ = c_${var}$
#:endfor

      CALL dbcsr_t_contract(dbcsr_scalar(c_alpha), tensor_1, tensor_2, dbcsr_scalar(c_beta), &
                            tensor_3, &
                            contract_1 + 1, notcontract_1 + 1, &
                            contract_2 + 1, notcontract_2 + 1, &
                            map_1 + 1, map_2 + 1, &
                            bounds_1, bounds_2, bounds_3, &
                            optimize_dist, pgrid_opt_1, pgrid_opt_2, pgrid_opt_3, &
                            filter_eps, flop, move_data, unit_nr, log_verbose)

#:for dim in range(1,4)
      IF (PRESENT(c_pgrid_opt_${dim}$)) c_pgrid_opt_${dim}$ = c_loc(pgrid_opt_${dim}$)
#:endfor

   END SUBROUTINE

#:endfor

   FUNCTION c_ndims_tensor(c_tensor) BIND(C, name="c_ndims_tensor")
      TYPE(c_ptr), INTENT(IN), VALUE                  :: c_tensor
      TYPE(dbcsr_t_type), POINTER                     :: tensor
      INTEGER(kind=c_int)                             :: c_ndims_tensor

      CALL c_f_pointer(c_tensor, tensor)

      c_ndims_tensor = tensor%nd_index%ndim_nd
   END FUNCTION

#:for dparam, dtype, dsuffix, basetype, cftype in cf_dtype_float_list
#:for ndim in ndims

   SUBROUTINE c_dbcsr_t_allocate_and_get_${ndim}$d_block_${dsuffix}$ (c_tensor, tensor_dim, c_ind, c_block, c_found) &
      bind(C, name="c_dbcsr_t_allocate_and_get_${ndim}$d_block_${dsuffix}$")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
      INTEGER(kind=c_int), DIMENSION(tensor_dim), INTENT(IN), &
         TARGET                                          :: c_ind
      TYPE(c_ptr), INTENT(OUT)                           :: c_block
      TYPE(c_ptr), INTENT(OUT)                           :: c_found

      TYPE(dbcsr_t_type), POINTER                        :: tensor
      INTEGER, DIMENSION(:), POINTER                     :: ind
      ${dtype}$, DIMENSION(${shape_colon(ndim)}$), ALLOCATABLE, &
         TARGET                                          :: block
      LOGICAL, POINTER                                   :: found

      ind => c_ind
      !block => c_block

      CALL c_f_pointer(c_tensor, tensor)
      ALLOCATE (found)

      CALL dbcsr_t_get_block(tensor, ind + 1, block, found)

      c_block = c_loc(block)
      c_found = c_loc(found)

   END SUBROUTINE
#:endfor
#:endfor

#:for dparam, dtype, dsuffix, basetype, cftype in cf_dtype_float_list
#:for ndim in ndims

   SUBROUTINE c_dbcsr_t_get_${ndim}$d_block_${dsuffix}$ (c_tensor, tensor_dim, c_ind, c_sizes, c_block, c_found) &
      BIND(C, name="c_dbcsr_t_get_${ndim}$d_block_${dsuffix}$")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
      INTEGER(kind=c_int), DIMENSION(tensor_dim), &
         INTENT(IN), TARGET                              :: c_ind
      INTEGER(kind=c_int), DIMENSION(tensor_dim), &
         INTENT(IN), TARGET                              :: c_sizes
      ${basetype}$ (kind=${cftype}$), &
         DIMENSION(${arrlist("c_sizes", nmax=ndim)}$), &
         TARGET, INTENT(OUT)                             :: c_block
      TYPE(c_ptr), INTENT(OUT)                           :: c_found

      TYPE(dbcsr_t_type), POINTER                        :: tensor
      INTEGER, DIMENSION(:), POINTER                     :: ind
      INTEGER, DIMENSION(:), POINTER                     :: sizes
      ${dtype}$, DIMENSION(${shape_colon(ndim)}$), &
        POINTER                                          :: block
      LOGICAL, POINTER                                   :: found

      ind => c_ind
      sizes => c_sizes
      block => c_block

      CALL c_f_pointer(c_tensor, tensor)
      ALLOCATE (found)

      CALL dbcsr_t_get_block(tensor, ind + 1, sizes, block, found)

      c_found = c_loc(found)

   END SUBROUTINE
#:endfor
#:endfor

#:for dparam, dtype, dsuffix, basetype, cftype in cf_dtype_float_list
#:for ndim in ndims

   SUBROUTINE c_dbcsr_t_put_${ndim}$d_block_${dsuffix}$ (c_tensor, tensor_dim, c_ind, c_sizes, &
                                            c_block, c_summation, c_scale) &
                                            BIND(C, name="c_dbcsr_t_put_${ndim}$d_block_${dsuffix}$")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
      INTEGER(kind=c_int), DIMENSION(tensor_dim), &
       INTENT(IN), TARGET                                :: c_ind
      INTEGER(kind=c_int), DIMENSION(tensor_dim), &
       INTENT(IN), TARGET                                :: c_sizes
      ${basetype}$ (kind=${cftype}$), INTENT(IN), &
       DIMENSION(${arrlist("c_sizes", nmax=ndim)}$), &
       TARGET                                            :: c_block
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation
      ${basetype}$ (kind=${cftype}$), INTENT(IN), &
       OPTIONAL                                          :: c_scale

      TYPE(dbcsr_t_type), POINTER                        :: tensor
      INTEGER, DIMENSION(:), POINTER                     :: ind
      INTEGER, DIMENSION(:), POINTER                     :: sizes
      ${dtype}$, DIMENSION(${shape_colon(ndim)}$), &
       POINTER                                           :: block
      LOGICAL, ALLOCATABLE                               :: summation
      ${dtype}$, ALLOCATABLE                             :: scale

      ind => c_ind

      IF (PRESENT(c_summation)) summation = c_summation
      IF (PRESENT(c_scale)) scale = c_scale

      CALL c_f_pointer(c_tensor, tensor)
      sizes => c_sizes
      block => c_block

      CALL dbcsr_t_put_block(tensor, ind + 1, sizes, &
                             block, summation, scale)

   END SUBROUTINE
#:endfor
#:endfor


   SUBROUTINE c_dbcsr_t_get_stored_coordinates(c_tensor, tensor_dim, c_ind_nd, c_processor) &
      BIND(C, name="c_dbcsr_t_get_stored_coordinates")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
      INTEGER(kind=c_int), DIMENSION(tensor_dim), &
       INTENT(IN), TARGET                                :: c_ind_nd
      INTEGER(kind=c_int), INTENT(OUT)                   :: c_processor

      TYPE(dbcsr_t_type), POINTER                        :: tensor
      INTEGER, DIMENSION(:), POINTER                     :: ind_nd

      CALL c_f_pointer(c_tensor, tensor)
      ind_nd => c_ind_nd

      CALL dbcsr_t_get_stored_coordinates(tensor, ind_nd + 1, c_processor)

   END SUBROUTINE


   SUBROUTINE c_dbcsr_t_reserve_blocks_index(c_tensor, nblocks, ${varlist("c_blk_ind")}$) &
      BIND(C, name="c_dbcsr_t_reserve_blocks_index")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
      INTEGER(kind=c_int), INTENT(in), value             :: nblocks
      INTEGER(kind=c_int), INTENT(IN), TARGET, OPTIONAL  :: ${varlist("c_blk_ind", suffix ="(nblocks)")}$

      TYPE(dbcsr_t_type), POINTER                        :: tensor
      INTEGER, DIMENSION(:), ALLOCATABLE                 :: ${varlist("blk_ind")}$

#:for ddim in ddims
      IF (PRESENT(c_blk_ind_${ddim}$)) THEN
         blk_ind_${ddim}$ = c_blk_ind_${ddim}$+1
      ENDIF
#:endfor

      CALL c_f_pointer(c_tensor, tensor)

      CALL dbcsr_t_reserve_blocks(tensor, ${varlist("blk_ind")}$)

   END SUBROUTINE


   FUNCTION c_ndims_iterator(c_iterator) BIND(C, name="c_ndims_iterator")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_iterator
      TYPE(dbcsr_t_iterator_type), POINTER               :: iterator
      INTEGER(kind=c_int)                                :: c_ndims_iterator

      CALL c_f_pointer(c_iterator, iterator)

      c_ndims_iterator = iterator%nd_index%ndim_nd

   END FUNCTION


   SUBROUTINE c_dbcsr_t_iterator_start(c_iterator, c_tensor) BIND(C, name="c_dbcsr_t_iterator_start")

      TYPE(c_ptr), INTENT(OUT)                           :: c_iterator
      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor

      TYPE(dbcsr_t_iterator_type), POINTER               :: iterator
      TYPE(dbcsr_t_type), POINTER                        :: tensor

      ALLOCATE (iterator)
      CALL c_f_pointer(c_tensor, tensor)

      CALL dbcsr_t_iterator_start(iterator, tensor)

      c_iterator = c_loc(iterator)

   END SUBROUTINE


   SUBROUTINE c_dbcsr_t_iterator_stop(c_iterator) BIND(C, name="c_dbcsr_t_iterator_stop")

      TYPE(c_ptr), INTENT(INOUT)                         :: c_iterator
      TYPE(dbcsr_t_iterator_type), POINTER               :: iterator

      CALL c_f_pointer(c_iterator, iterator)

      CALL dbcsr_t_iterator_stop(iterator)

      DEALLOCATE (iterator)

      c_iterator = c_null_ptr

   END SUBROUTINE


   SUBROUTINE c_dbcsr_t_iterator_next_block(c_iterator, iterator_size, c_ind_nd, c_blk, c_blk_p, c_blk_size, c_blk_offset) &
      BIND(C, name="c_dbcsr_t_iterator_next_block")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_iterator
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: iterator_size
      INTEGER(kind=c_int), DIMENSION(iterator_size), TARGET, &
         INTENT(OUT)                                     :: c_ind_nd
      INTEGER(kind=c_int), INTENT(OUT)                   :: c_blk
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL         :: c_blk_p
      INTEGER(kind=c_int), DIMENSION(iterator_size), INTENT(OUT), &
         TARGET, OPTIONAL                                :: c_blk_size, c_blk_offset

      TYPE(dbcsr_t_iterator_type), POINTER               :: iterator
      INTEGER, DIMENSION(:), POINTER                     :: ind_nd
      INTEGER, ALLOCATABLE                               :: blk_p
      INTEGER, DIMENSION(:), POINTER                     :: blk_size => null()
      INTEGER, DIMENSION(:), POINTER                     :: blk_offset => null()

      CALL c_f_pointer(c_iterator, iterator)
      ind_nd => c_ind_nd

      IF (PRESENT(c_blk_p)) blk_p = c_blk_p
      IF (PRESENT(c_blk_size)) blk_size => c_blk_size
      IF (PRESENT(c_blk_offset)) blk_offset => c_blk_offset

      CALL dbcsr_t_iterator_next_block(iterator, ind_nd, c_blk, blk_p, blk_size, blk_offset)

      c_ind_nd = c_ind_nd - 1

   END SUBROUTINE


   FUNCTION c_dbcsr_t_iterator_blocks_left(c_iterator) &
      BIND(C, name="c_dbcsr_t_iterator_blocks_left")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_iterator
      TYPE(dbcsr_t_iterator_type), POINTER               :: iterator
      LOGICAL(kind=c_bool)                               :: c_dbcsr_t_iterator_blocks_left

      CALL c_f_pointer(c_iterator, iterator); 
      c_dbcsr_t_iterator_blocks_left = dbcsr_t_iterator_blocks_left(iterator)

   END FUNCTION


   SUBROUTINE c_dbcsr_t_split_blocks(c_tensor_in, tensor_dim, c_tensor_out, c_block_sizes, c_nodata) &
      BIND(C, name="c_dbcsr_t_split_blocks")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_in
      TYPE(c_ptr), INTENT(OUT)                           :: c_tensor_out
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
      INTEGER(kind=c_int), INTENT(IN), DIMENSION(tensor_dim), &
         TARGET                                          :: c_block_sizes
      LOGICAL(kind=c_bool), OPTIONAL                     :: c_nodata

      TYPE(dbcsr_t_type), POINTER                        :: tensor_in
      TYPE(dbcsr_t_type), POINTER                        :: tensor_out
      INTEGER, DIMENSION(:), POINTER                     :: block_sizes
      LOGICAL, ALLOCATABLE                               :: nodata

      CALL c_f_pointer(c_tensor_in, tensor_in)
      ALLOCATE (tensor_out)
      block_sizes => c_block_sizes
      IF (PRESENT(c_nodata)) nodata = c_nodata

      CALL dbcsr_t_split_blocks(tensor_in, tensor_out, block_sizes, nodata)

      c_tensor_out = c_loc(tensor_out)

   END SUBROUTINE

#:for dparam, dtype, dsuffix, dbase, dctype in cf_dtype_float_list

   SUBROUTINE c_dbcsr_t_filter_${dsuffix}$ (c_tensor, c_eps, c_method, c_use_absolute) &
      BIND(C, name="c_dbcsr_t_filter_${dsuffix}$")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
      ${dbase}$ (kind=${dctype}$), INTENT(IN), VALUE     :: c_eps
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL          :: c_method
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_use_absolute

      TYPE(dbcsr_t_type), POINTER                        :: tensor
      INTEGER, ALLOCATABLE                               :: method
      LOGICAL, ALLOCATABLE                               :: use_absolute

      CALL c_f_pointer(c_tensor, tensor)
      IF (PRESENT(c_method)) method = c_method
      IF (PRESENT(c_use_absolute)) use_absolute = c_use_absolute

      CALL dbcsr_t_filter(tensor, c_eps, method, use_absolute)

   END SUBROUTINE
#:endfor

#:for dparam, dtype, dsuffix, dbase, dctype in cf_dtype_float_list

   SUBROUTINE c_dbcsr_t_set_${dsuffix}$ (c_tensor, c_alpha) &
      BIND(C, name="c_dbcsr_t_set_${dsuffix}$")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
      ${dbase}$ (kind=${dctype}$), INTENT(IN), VALUE     :: c_alpha

      TYPE(dbcsr_t_type), POINTER                        :: tensor

      CALL c_f_pointer(c_tensor, tensor)

      CALL dbcsr_t_set(tensor, c_alpha)

   END SUBROUTINE
#:endfor


   SUBROUTINE c_dbcsr_t_copy_matrix_to_tensor(c_matrix_in, c_tensor_out, c_summation) &
      BIND(C, name="c_dbcsr_t_copy_matrix_to_tensor")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_matrix_in
      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_out
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation

      TYPE(dbcsr_type), POINTER                          :: matrix_in
      TYPE(dbcsr_t_type), POINTER                        :: tensor_out
      LOGICAL, ALLOCATABLE                               :: summation

      CALL c_f_pointer(c_matrix_in, matrix_in)
      CALL c_f_pointer(c_tensor_out, tensor_out)

      IF (PRESENT(c_summation)) summation = c_summation

      CALL dbcsr_t_copy_matrix_to_tensor(matrix_in, tensor_out, summation)

   END SUBROUTINE


   SUBROUTINE c_dbcsr_t_copy_tensor_to_matrix(c_tensor_in, c_matrix_out, c_summation) &
      BIND(C, name="c_dbcsr_t_copy_tensor_to_matrix")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_in
      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_matrix_out
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation

      TYPE(dbcsr_t_type), POINTER                        :: tensor_in
      TYPE(dbcsr_type), POINTER                          :: matrix_out
      LOGICAL, ALLOCATABLE                               :: summation

      CALL c_f_pointer(c_tensor_in, tensor_in)
      CALL c_f_pointer(c_matrix_out, matrix_out)

      IF (PRESENT(c_summation)) summation = c_summation

      CALL dbcsr_t_copy_tensor_to_matrix(tensor_in, matrix_out, summation)

   END SUBROUTINE


   SUBROUTINE c_dbcsr_t_blk_sizes(c_tensor, tensor_dim, c_ind, c_blk_size) &
      BIND(C, name="c_dbcsr_t_blk_sizes")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
      INTEGER(kind=c_int), DIMENSION(tensor_dim), INTENT(IN), &
         TARGET                                          :: c_ind
      INTEGER(kind=c_int), DIMENSION(tensor_dim), INTENT(OUT), &
         TARGET                                          :: c_blk_size

      TYPE(dbcsr_t_type), POINTER                        :: tensor
      INTEGER, DIMENSION(:), POINTER                     :: ind
      INTEGER, DIMENSION(:), POINTER                     :: blk_size

      CALL c_f_pointer(c_tensor, tensor)
      ind => c_ind
      blk_size => c_blk_size

      CALL dbcsr_t_blk_sizes(tensor, ind + 1, blk_size)

   END SUBROUTINE


   SUBROUTINE c_dbcsr_t_copy(c_tensor_in, tensor_dim, c_tensor_out, c_order, c_summation, c_move_data, c_unit_nr) &
      BIND(C, name="c_dbcsr_t_copy")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_in
      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_out
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
      INTEGER(kind=c_int), INTENT(IN), DIMENSION(tensor_dim), OPTIONAL, &
         TARGET                                          :: c_order
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation, c_move_data
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL          :: c_unit_nr

      TYPE(dbcsr_t_type), POINTER                        :: tensor_in, tensor_out
      INTEGER, DIMENSION(:), POINTER                     :: order
      LOGICAL, ALLOCATABLE                               :: summation, move_data
      INTEGER, ALLOCATABLE                               :: unit_nr

      CALL c_f_pointer(c_tensor_in, tensor_in)
      CALL c_f_pointer(c_tensor_out, tensor_out)

      order => c_order

      IF (PRESENT(c_summation)) summation = c_summation
      IF (PRESENT(c_move_data)) move_data = c_move_data
      IF (PRESENT(c_unit_nr)) unit_nr = c_unit_nr

      CALL dbcsr_t_copy(tensor_in, tensor_out, order, summation, move_data, unit_nr)

   END SUBROUTINE

#:for dparam, dtype, dsuffix, dbase, dctype in cf_dtype_float_list

   SUBROUTINE c_dbcsr_t_scale_${dsuffix}$ (c_tensor, c_alpha) &
      BIND(C, name="c_dbcsr_t_scale_${dsuffix}$")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
      ${dbase}$ (kind=${dctype}$), INTENT(IN), VALUE     :: c_alpha

      TYPE(dbcsr_t_type), POINTER                        :: tensor

      CALL c_f_pointer(c_tensor, tensor)

      CALL dbcsr_t_scale(tensor, dbcsr_scalar(c_alpha))

   END SUBROUTINE

#:endfor


   SUBROUTINE c_dbcsr_t_clear(c_tensor) BIND(C, name="c_dbcsr_clear")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
      TYPE(dbcsr_t_type), POINTER                        :: tensor

      CALL c_f_pointer(c_tensor, tensor)

      CALL dbcsr_t_clear(tensor)

   END SUBROUTINE

END MODULE
