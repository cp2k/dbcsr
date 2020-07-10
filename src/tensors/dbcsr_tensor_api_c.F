!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

#:include "dbcsr_tensor.fypp"
#:set ndims = range(2,maxrank+1)
#:set ddims = range(1,maxrank+1)

MODULE dbcsr_tensor_api_c

   USE, INTRINSIC :: ISO_C_BINDING, ONLY: c_loc, c_ptr, c_double, c_sizeof, C_NULL_CHAR, &
                                                                             c_float, c_f_pointer, c_int, c_long_long, &
                                                                             c_char, c_null_ptr, c_bool, c_associated, &
                                                                             c_float_complex, c_double_complex
   USE dbcsr_api, ONLY: dbcsr_type, dbcsr_scale
   USE dbcsr_allocate_wrap, ONLY: allocate_any
   USE dbcsr_tensor_api
   USE dbcsr_kinds, ONLY: &
      ${uselist(dtype_float_prec)}$, default_string_length, int_8
   USE dbcsr_data_types, ONLY: dbcsr_scalar_type
   USE dbcsr_data_methods, ONLY: dbcsr_scalar
   USE dbcsr_machine, ONLY: default_output_unit
   USE dbcsr_tensor_index, ONLY: &
      dbcsr_t_get_mapping_info, ndims_mapping, nd_to_2d_mapping
   USE dbcsr_tensor_types, ONLY: dbcsr_t_create, dbcsr_t_type
   USE dbcsr_tas_types, ONLY: dbcsr_tas_type
   USE dbcsr_data_methods, ONLY: dbcsr_get_data_p

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

   SUBROUTINE c_dbcsr_t_finalize(c_tensor) BIND(C, name="c_dbcsr_t_finalize")
      TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor

      TYPE(dbcsr_t_type), POINTER :: tensor

      CALL c_f_pointer(c_tensor, tensor)

      CALL dbcsr_t_finalize(tensor)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_pgrid_create_expert(fcomm, c_dims, dims_size, c_pgrid, &
                                            c_map1_2d, map1_2d_size, c_map2_2d, &
                                            map2_2d_size, c_tensor_dims, nsplit, dimsplit) &
      bind(C, name="c_dbcsr_t_pgrid_create_expert")

      INTEGER(kind=c_int), INTENT(in)                    :: fcomm
      INTEGER(kind=c_int), INTENT(in), value             :: dims_size
      TYPE(c_ptr), INTENT(out)                           :: c_pgrid
      INTEGER(kind=c_int), INTENT(out)                   :: c_dims(dims_size)
      INTEGER(kind=c_int), INTENT(out), OPTIONAL, &
         DIMENSION(dims_size)                           :: c_tensor_dims
      INTEGER(kind=c_int), INTENT(in), VALUE             :: map1_2d_size
      INTEGER(kind=c_int), INTENT(in)                    :: c_map1_2d(map1_2d_size)
      INTEGER(kind=c_int), INTENT(in), VALUE             :: map2_2d_size
      INTEGER(kind=c_int), INTENT(in)                    :: c_map2_2d(map2_2d_size)
      INTEGER(kind=c_int), INTENT(in), OPTIONAL          :: nsplit, dimsplit

      TYPE(dbcsr_t_pgrid_type), POINTER                  :: pgrid

      ALLOCATE (pgrid)

      CALL dbcsr_t_pgrid_create_expert(fcomm, c_dims, pgrid, c_map1_2d+1, c_map2_2d+1, c_tensor_dims, &
                                       nsplit, dimsplit)

      c_pgrid = c_loc(pgrid)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_pgrid_create(fcomm, c_dims, dims_size, c_pgrid, c_tensor_dims) &
      bind(C, name="c_dbcsr_t_pgrid_create")

      INTEGER(kind=c_int), INTENT(in)                    :: fcomm
      INTEGER(kind=c_int), INTENT(in), value             :: dims_size
      TYPE(c_ptr), INTENT(out)                           :: c_pgrid
      INTEGER(kind=c_int), INTENT(out)                   :: c_dims(dims_size)
      INTEGER(kind=c_int), INTENT(in), OPTIONAL          :: c_tensor_dims(dims_size)

      TYPE(dbcsr_t_pgrid_type), POINTER                  :: pgrid

      ALLOCATE (pgrid)

      IF (PRESENT(c_tensor_dims)) THEN
         CALL dbcsr_t_pgrid_create(fcomm, c_dims, pgrid, c_tensor_dims)
      ELSE
         CALL dbcsr_t_pgrid_create(fcomm, c_dims, pgrid)
      ENDIF

      c_pgrid = c_loc(pgrid)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_pgrid_destroy(c_pgrid, c_keep_comm) &
      BIND(C, name="c_dbcsr_t_pgrid_destroy")

      TYPE(c_ptr), INTENT(INOUT)                         :: c_pgrid
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_keep_comm
      TYPE(dbcsr_t_pgrid_type), POINTER                  :: pgrid
      LOGICAL                                            :: keep_comm

      CALL c_f_pointer(c_pgrid, pgrid)

      IF (PRESENT(c_keep_comm)) THEN
         keep_comm = c_keep_comm
         CALL dbcsr_t_pgrid_destroy(pgrid, keep_comm)
      ELSE
         CALL dbcsr_t_pgrid_destroy(pgrid)
      ENDIF

      IF (ASSOCIATED(pgrid)) DEALLOCATE (pgrid)
      c_pgrid = c_null_ptr

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_distribution_new(c_dist, c_pgrid, ${c_varlist_and_size("c_nd_dist")}$) &
      bind(C, name="c_dbcsr_t_distribution_new")

      TYPE(c_ptr), INTENT(OUT)                           :: c_dist
      TYPE(c_ptr), INTENT(IN), value                     :: c_pgrid

      INTEGER(kind=c_int), INTENT(in), value :: ${varlist("c_nd_dist", suffix="_size")}$
      INTEGER(kind=c_int), INTENT(IN), TARGET, OPTIONAL :: ${c_arrlist_and_size("c_nd_dist")}$

      TYPE(dbcsr_t_pgrid_type), POINTER                  :: pgrid
      TYPE(dbcsr_t_distribution_type), POINTER           :: dist

      INTEGER, DIMENSION(:), POINTER                     ::  ${varlist("nd_dist")}$

      ALLOCATE (dist)

      CALL c_f_pointer(c_pgrid, pgrid)

#:for i in range(1, maxrank+1)
      NULLIFY (nd_dist_${i}$)
      IF (PRESENT(c_nd_dist_${i}$)) THEN
         nd_dist_${i}$ => c_nd_dist_${i}$
      ENDIF
#:endfor

      CALL dbcsr_t_distribution_new(dist, pgrid, ${varlist("nd_dist")}$)

      c_dist = c_loc(dist)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_distribution_destroy(c_dist) BIND(C, name="c_dbcsr_t_distribution_destroy")
      TYPE(c_ptr), INTENT(INOUT)                         :: c_dist

      TYPE(dbcsr_t_distribution_type), POINTER           :: dist

      CALL c_f_pointer(c_dist, dist)
      CALL dbcsr_t_distribution_destroy(dist)

      IF (ASSOCIATED(dist)) DEALLOCATE (dist)
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

      TYPE(dbcsr_t_type), POINTER                        :: tensor
      TYPE(dbcsr_t_distribution_type), POINTER           :: dist

      INTEGER, DIMENSION(:), POINTER                     :: ${varlist("blk_size")}$

      ALLOCATE (tensor)

      CALL c_f_pointer(c_dist, dist)
      CALL c_f_string(c_name, fname)

#:for i in range(1, maxrank+1)
      NULLIFY (blk_size_${i}$)
      IF (PRESENT(c_blk_size_${i}$)) THEN
         blk_size_${i}$ => c_blk_size_${i}$
      ENDIF
#:endfor

      CALL dbcsr_t_create(tensor, fname, dist, c_map1_2d+1, c_map2_2d+1, c_data_type, &
                          ${varlist("blk_size")}$)

      c_tensor = c_loc(tensor)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_create_template(c_tensor_in, c_tensor, c_name, c_dist, &
                                        c_map1_2d, map1_2d_size, c_map2_2d, map2_2d_size, data_type) &
      BIND(C, name="c_dbcsr_t_create_template")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_in
      TYPE(c_ptr), INTENT(OUT)                           :: c_tensor
      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_name
      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_dist

      INTEGER(kind=c_int), INTENT(in), value             :: map1_2d_size
      INTEGER(kind=c_int), INTENT(in), TARGET, OPTIONAL  :: c_map1_2d(map1_2d_size)
      INTEGER(kind=c_int), INTENT(in), value             :: map2_2d_size
      INTEGER(kind=c_int), INTENT(in), TARGET, OPTIONAL  :: c_map2_2d(map2_2d_size)

      INTEGER(kind=c_int), INTENT(IN), OPTIONAL          :: data_type

      TYPE(dbcsr_t_type), POINTER                        :: tensor_in
      TYPE(dbcsr_t_type), POINTER                        :: tensor
      CHARACTER(:, kind=c_char), ALLOCATABLE             :: fname
      TYPE(dbcsr_t_distribution_type), POINTER           :: dist
      INTEGER, DIMENSION(:), POINTER                     :: map1_2d, map2_2d

      CALL c_f_pointer(c_tensor_in, tensor_in)

      IF (c_associated(c_name)) CALL c_f_string(c_name, fname)

      ALLOCATE (tensor)

      NULLIFY (dist)
      IF (C_ASSOCIATED(c_dist)) CALL c_f_pointer(c_dist, dist)
      NULLIFY (map1_2d, map2_2d)
      IF (PRESENT(c_map1_2d) .AND. PRESENT(c_map2_2d)) THEN
         ALLOCATE (map1_2d(SIZE(c_map1_2d)))
         ALLOCATE (map2_2d(SIZE(c_map2_2d)))
         map1_2d = c_map1_2d+1
         map2_2d = c_map2_2d+1
      ENDIF

      CALL dbcsr_t_create(tensor_in, tensor, fname, dist, &
                          map1_2d, map2_2d, data_type)

      c_tensor = c_loc(tensor)

      IF (ASSOCIATED(map1_2d)) DEALLOCATE (map1_2d)
      IF (ASSOCIATED(map2_2d)) DEALLOCATE (map2_2d)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_create_matrix(c_matrix_in, c_tensor, c_order, c_name) &
      BIND(C, name="c_dbcsr_t_create_matrix")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_matrix_in
      TYPE(c_ptr), INTENT(OUT)                           :: c_tensor
      INTEGER(kind=c_int), INTENT(IN), DIMENSION(2), &
         OPTIONAL                                         :: c_order
      TYPE(c_ptr), INTENT(IN), OPTIONAL                  :: c_name

      TYPE(dbcsr_type), POINTER                :: matrix_in
      TYPE(dbcsr_t_type), POINTER              :: tensor
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

      IF (ASSOCIATED(tensor)) DEALLOCATE (tensor)
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
                                              c_filter_eps, c_flop, c_move_data, c_retain_sparsity, &
                                              c_unit_nr, c_log_verbose) &
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
         OPTIONAL                                         :: c_bounds_1
      INTEGER(kind=c_int), INTENT(IN), DIMENSION(2, notcontract_1_size), &
         OPTIONAL                                         :: c_bounds_2
      INTEGER(kind=c_int), INTENT(IN), DIMENSION(2, notcontract_2_size), &
         OPTIONAL                                         :: c_bounds_3

      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_optimize_dist
      TYPE(c_ptr), INTENT(OUT), OPTIONAL                 :: c_pgrid_opt_1, c_pgrid_opt_2, c_pgrid_opt_3

      REAL(kind=c_double), INTENT(IN), OPTIONAL          :: c_filter_eps
      INTEGER(kind=c_long_long), INTENT(INOUT), OPTIONAL :: c_flop
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_move_data
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_retain_sparsity
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL          :: c_unit_nr
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_log_verbose

      TYPE(dbcsr_t_type), POINTER                        :: tensor_1
      TYPE(dbcsr_t_type), POINTER                        :: tensor_2
      TYPE(dbcsr_t_type), POINTER                        :: tensor_3
      LOGICAL, POINTER                                   :: optimize_dist
      TYPE(dbcsr_t_pgrid_type), POINTER                  :: pgrid_opt_1, pgrid_opt_2, pgrid_opt_3

      INTEGER, DIMENSION(:, :), ALLOCATABLE              :: bounds_1, bounds_2, bounds_3

      LOGICAL, POINTER                                   :: move_data
      LOGICAL, POINTER                                   :: retain_sparsity
      LOGICAL, POINTER                                   :: log_verbose

      CALL c_f_pointer(c_tensor_1, tensor_1)
      CALL c_f_pointer(c_tensor_2, tensor_2)
      CALL c_f_pointer(c_tensor_3, tensor_3)

#:for dim in range(1,4)
      IF (PRESENT(c_bounds_${dim}$)) THEN
         ALLOCATE (bounds_${dim}$ (2, SIZE(c_bounds_${dim}$, 2)))
         bounds_${dim}$ (:, :) = c_bounds_${dim}$ (:, :) + 1
      ENDIF
#:endfor

#:set list = ['optimize_dist', 'move_data', 'retain_sparsity', 'log_verbose']
#:for var in list
      NULLIFY (${var}$)
      IF (PRESENT(c_${var}$)) THEN
         ALLOCATE (${var}$)
         ${var}$ = c_${var}$
      ENDIF
#:endfor

#:set optvars = [['pgrid_opt_1'],['pgrid_opt_2'],['pgrid_opt_3']]
#:set optgroups = []
      ${gen_vargroups(optvars,optgroups)}$

#:for var in optvars
      NULLIFY (${var[0]}$)
#:endfor

#:for i in range(len(optgroups))
      ${print_groupif(optgroups,optvars,i,'PRESENT','c_')}$

      CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                            tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                            tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                            notcontract_1=c_notcontract_1 + 1, &
                            contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                            map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                            bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                            optimize_dist=optimize_dist &
                            ${print_group(optgroups[i])}$, &
                            filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                            retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                            log_verbose=log_verbose)
#:endfor
      $:"ENDIF"

#:for dim in range(1,4)
      IF (PRESENT(c_pgrid_opt_${dim}$)) c_pgrid_opt_${dim}$ = c_loc(pgrid_opt_${dim}$)
#:endfor

#:set list = ['optimize_dist', 'move_data', 'retain_sparsity', 'log_verbose']
#:for var in list
      IF (ASSOCIATED(${var}$)) DEALLOCATE (${var}$)
#:endfor

#:for dim in range(1,4)
      IF (ALLOCATED(bounds_${dim}$)) DEALLOCATE (bounds_${dim}$)
#:endfor

      IF (PRESENT(c_unit_nr)) THEN
         IF (c_unit_nr .GE. 0) THEN
            flush (c_unit_nr)
         ENDIF
      ENDIF

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_contract_index_${dsuffix}$ (c_alpha, c_tensor_1, &
                                                    c_tensor_2, c_beta, c_tensor_3, &
                                                    c_contract_1, contract_1_size, &
                                                    c_notcontract_1, notcontract_1_size, &
                                                    c_contract_2, contract_2_size, &
                                                    c_notcontract_2, notcontract_2_size, &
                                                    c_map_1, map_1_size, c_map_2, map_2_size, &
                                                    c_bounds_1, c_bounds_2, c_bounds_3, &
                                                    c_filter_eps, c_nblks_local, c_result_index, &
                                                    result_index_size, tensor3_dim) &
      BIND(C, name="c_dbcsr_t_contract_index_${dsuffix}$")

      ${dbase}$ (kind=${dctype}$), INTENT(IN), VALUE     :: c_alpha
      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_1, c_tensor_2
      ${dbase}$ (kind=${dctype}$), INTENT(IN), VALUE     :: c_beta
      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_3

      INTEGER(kind=c_int), INTENT(IN), VALUE             :: contract_1_size, contract_2_size
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: notcontract_1_size, notcontract_2_size
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: map_1_size, map_2_size

      INTEGER(kind=c_int), INTENT(IN) :: c_contract_1(contract_1_size), c_contract_2(contract_2_size)
      INTEGER(kind=c_int), INTENT(IN) :: c_map_1(map_1_size), c_map_2(map_2_size)
      INTEGER(kind=c_int), INTENT(IN) :: c_notcontract_1(notcontract_1_size), c_notcontract_2(notcontract_2_size)

      INTEGER(kind=c_int), INTENT(IN), DIMENSION(2, contract_1_size), &
         OPTIONAL                                 :: c_bounds_1
      INTEGER(kind=c_int), INTENT(IN), DIMENSION(2, notcontract_1_size), &
         OPTIONAL                                 :: c_bounds_2
      INTEGER(kind=c_int), INTENT(IN), DIMENSION(2, notcontract_2_size), &
         OPTIONAL                                  :: c_bounds_3

      REAL(kind=c_double), INTENT(IN), OPTIONAL          :: c_filter_eps
      INTEGER(kind=c_int), INTENT(OUT)                   :: c_nblks_local
      INTEGER(kind=c_long_long), INTENT(IN), VALUE       :: result_index_size
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor3_dim
      INTEGER(kind=c_int), DIMENSION(result_index_size, tensor3_dim), &
         INTENT(OUT)                            :: c_result_index

      TYPE(dbcsr_t_type), POINTER                        :: tensor_1
      TYPE(dbcsr_t_type), POINTER                        :: tensor_2
      TYPE(dbcsr_t_type), POINTER                        :: tensor_3
      INTEGER, DIMENSION(:, :), POINTER                  :: bounds_1, bounds_2, bounds_3
      INTEGER, DIMENSION(result_index_size, tensor3_dim) :: result_index

      CALL c_f_pointer(c_tensor_1, tensor_1)
      CALL c_f_pointer(c_tensor_2, tensor_2)
      CALL c_f_pointer(c_tensor_3, tensor_3)

#:for dim in range(1,4)
      NULLIFY (bounds_${dim}$)
      IF (PRESENT(c_bounds_${dim}$)) THEN
         ALLOCATE (bounds_${dim}$ (2, SIZE(c_bounds_${dim}$, 2)))
         bounds_${dim}$ = c_bounds_${dim}$+1
      ENDIF
#:endfor

      CALL dbcsr_t_contract_index(dbcsr_scalar(c_alpha), tensor_1, &
                                  tensor_2, dbcsr_scalar(c_beta), tensor_3, &
                                  c_contract_1 + 1, c_notcontract_1 + 1, &
                                  c_contract_2 + 1, c_notcontract_2 + 1, &
                                  c_map_1 + 1, c_map_2 + 1, &
                                  bounds_1, bounds_2, bounds_3, &
                                  c_filter_eps, c_nblks_local, result_index)

      c_result_index = result_index - 1

#:for dim in range(1,4)
      IF (ASSOCIATED(bounds_${dim}$)) DEALLOCATE (bounds_${dim}$)
#:endfor

   END SUBROUTINE

#:endfor

#:for dparam, dtype, dsuffix, basetype, cftype in cf_dtype_float_list
#:for ndim in ndims

   SUBROUTINE c_dbcsr_t_get_${ndim}$d_block_${dsuffix}$ (c_tensor, tensor_dim, c_ind, c_sizes, c_block, c_found) &
      BIND(C, name="c_dbcsr_t_get_${ndim}$d_block_${dsuffix}$")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
      INTEGER(kind=c_int), DIMENSION(tensor_dim), &
         INTENT(IN)                                      :: c_ind
      INTEGER(kind=c_int), DIMENSION(tensor_dim), &
         INTENT(IN)                                      :: c_sizes
      ${basetype}$ (kind=${cftype}$), &
         DIMENSION(${arrlist("c_sizes", nmax=ndim)}$), &
         INTENT(OUT)                             :: c_block
      LOGICAL(kind=c_bool), INTENT(OUT)                  :: c_found

      TYPE(dbcsr_t_type), POINTER                        :: tensor
      LOGICAL                                            :: found

      CALL c_f_pointer(c_tensor, tensor)

      CALL dbcsr_t_get_block(tensor, c_ind + 1, c_sizes, c_block, found)
      c_found = LOGICAL(found, kind=c_bool)

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
         INTENT(IN)                               :: c_ind
      INTEGER(kind=c_int), DIMENSION(tensor_dim), &
         INTENT(IN)                                :: c_sizes
      ${basetype}$ (kind=${cftype}$), INTENT(IN), &
         DIMENSION(${arrlist("c_sizes", nmax=ndim)}$) :: c_block
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation
      ${basetype}$ (kind=${cftype}$), INTENT(IN), &
         OPTIONAL                                          :: c_scale

      TYPE(dbcsr_t_type), POINTER                        :: tensor
      LOGICAL                                            :: summation

      CALL c_f_pointer(c_tensor, tensor)

      IF (PRESENT(c_summation)) THEN
         summation = c_summation
         CALL dbcsr_t_put_block(tensor, c_ind + 1, c_sizes, &
                                c_block, summation, c_scale)
      ELSE
         CALL dbcsr_t_put_block(tensor, c_ind + 1, c_sizes, &
                                c_block, scale=c_scale)
      ENDIF

   END SUBROUTINE
#:endfor
#:endfor

   SUBROUTINE c_dbcsr_t_get_stored_coordinates(c_tensor, tensor_dim, c_ind_nd, c_processor) &
      BIND(C, name="c_dbcsr_t_get_stored_coordinates")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
      INTEGER(kind=c_int), DIMENSION(tensor_dim), &
         INTENT(IN)                                :: c_ind_nd
      INTEGER(kind=c_int), INTENT(OUT)                   :: c_processor

      TYPE(dbcsr_t_type), POINTER                        :: tensor

      CALL c_f_pointer(c_tensor, tensor)

      CALL dbcsr_t_get_stored_coordinates(tensor, c_ind_nd + 1, c_processor)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_reserve_blocks_index(c_tensor, nblocks, ${varlist("c_blk_ind")}$) &
      BIND(C, name="c_dbcsr_t_reserve_blocks_index")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
      INTEGER(kind=c_int), INTENT(in), value             :: nblocks
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL          :: ${varlist("c_blk_ind", suffix ="(nblocks)")}$
      INTEGER, DIMENSION(:), POINTER                     :: ${varlist("blk_ind")}$

      TYPE(dbcsr_t_type), POINTER                        :: tensor

#:for ddim in ddims
      NULLIFY (blk_ind_${ddim}$)
      IF (PRESENT(c_blk_ind_${ddim}$)) THEN
         ALLOCATE (blk_ind_${ddim}$ (nblocks))
         blk_ind_${ddim}$ = c_blk_ind_${ddim}$+1
      ENDIF
#:endfor

      CALL c_f_pointer(c_tensor, tensor)

      CALL dbcsr_t_reserve_blocks(tensor, ${varlist("blk_ind")}$)

#:for ddim in ddims
      IF (ASSOCIATED(blk_ind_${ddim}$)) DEALLOCATE (blk_ind_${ddim}$)
#:endfor

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_reserve_blocks_template(c_tensor_in, c_tensor_out) &
      BIND(C, name="c_dbcsr_t_reserve_blocks_template")

      TYPE(c_ptr), INTENT(IN), VALUE              :: c_tensor_in
      TYPE(c_ptr), INTENT(IN), VALUE              :: c_tensor_out

      TYPE(dbcsr_t_type), POINTER :: tensor_in
      TYPE(dbcsr_t_type), POINTER :: tensor_out

      CALL c_f_pointer(c_tensor_in, tensor_in)
      CALL c_f_pointer(c_tensor_out, tensor_out)

      CALL dbcsr_t_reserve_blocks(tensor_in, tensor_out)

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

      IF (ASSOCIATED(iterator)) DEALLOCATE (iterator)
      c_iterator = c_null_ptr

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_iterator_next_block(c_iterator, iterator_size, c_ind_nd, c_blk, c_blk_p, c_blk_size, c_blk_offset) &
      BIND(C, name="c_dbcsr_t_iterator_next_block")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_iterator
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: iterator_size
      INTEGER(kind=c_int), DIMENSION(iterator_size), TARGET, &
         INTENT(OUT)                                   :: c_ind_nd
      INTEGER(kind=c_int), INTENT(OUT)                 :: c_blk
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL       :: c_blk_p
      INTEGER(kind=c_int), DIMENSION(iterator_size), INTENT(OUT), &
         OPTIONAL                                        :: c_blk_size, c_blk_offset

      TYPE(dbcsr_t_iterator_type), POINTER               :: iterator
      INTEGER, DIMENSION(iterator_size)                  :: ind_nd
      INTEGER, DIMENSION(:), POINTER                     :: blk_offset

      CALL c_f_pointer(c_iterator, iterator)

      NULLIFY (blk_offset)
      IF (PRESENT(c_blk_offset)) ALLOCATE (blk_offset(iterator_size))

      CALL dbcsr_t_iterator_next_block(iterator, ind_nd, c_blk, c_blk_p, c_blk_size, blk_offset)

      c_ind_nd = ind_nd - 1
      IF (PRESENT(c_blk_offset)) THEN
         c_blk_offset = blk_offset - 1
         DEALLOCATE (blk_offset)
      ENDIF

   END SUBROUTINE

   FUNCTION c_dbcsr_t_iterator_blocks_left(c_iterator) &
      BIND(C, name="c_dbcsr_t_iterator_blocks_left")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_iterator
      TYPE(dbcsr_t_iterator_type), POINTER               :: iterator
      LOGICAL(kind=c_bool)                               :: c_dbcsr_t_iterator_blocks_left

      CALL c_f_pointer(c_iterator, iterator)
      c_dbcsr_t_iterator_blocks_left = LOGICAL(dbcsr_t_iterator_blocks_left(iterator), kind=c_bool)

   END FUNCTION

   SUBROUTINE c_dbcsr_t_split_blocks(c_tensor_in, tensor_dim, c_tensor_out, c_block_sizes, c_nodata) &
      BIND(C, name="c_dbcsr_t_split_blocks")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_in
      TYPE(c_ptr), INTENT(OUT)                           :: c_tensor_out
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
      INTEGER(kind=c_int), INTENT(IN), DIMENSION(tensor_dim) &
         :: c_block_sizes
      LOGICAL(kind=c_bool), OPTIONAL                     :: c_nodata

      TYPE(dbcsr_t_type), POINTER                        :: tensor_in
      TYPE(dbcsr_t_type), POINTER                        :: tensor_out
      LOGICAL                                            :: nodata

      CALL c_f_pointer(c_tensor_in, tensor_in)
      ALLOCATE (tensor_out)

      IF (PRESENT(c_nodata)) THEN
         nodata = c_nodata
         CALL dbcsr_t_split_blocks(tensor_in, tensor_out, c_block_sizes, nodata)
      ELSE
         CALL dbcsr_t_split_blocks(tensor_in, tensor_out, c_block_sizes)
      ENDIF

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
      LOGICAL                                            :: use_absolute

      CALL c_f_pointer(c_tensor, tensor)

      IF (PRESENT(c_use_absolute)) THEN
         use_absolute = c_use_absolute
         CALL dbcsr_t_filter(tensor, c_eps, c_method, use_absolute)
      ELSE
         CALL dbcsr_t_filter(tensor, c_eps, c_method, use_absolute)
      ENDIF

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
      LOGICAL                                            :: summation

      CALL c_f_pointer(c_matrix_in, matrix_in)
      CALL c_f_pointer(c_tensor_out, tensor_out)

      IF (PRESENT(c_summation)) THEN
         summation = c_summation
         CALL dbcsr_t_copy_matrix_to_tensor(matrix_in, tensor_out, summation)
      ELSE
         CALL dbcsr_t_copy_matrix_to_tensor(matrix_in, tensor_out)
      ENDIF

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_copy_tensor_to_matrix(c_tensor_in, c_matrix_out, c_summation) &
      BIND(C, name="c_dbcsr_t_copy_tensor_to_matrix")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_in
      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_matrix_out
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation

      TYPE(dbcsr_t_type), POINTER                        :: tensor_in
      TYPE(dbcsr_type), POINTER                          :: matrix_out
      LOGICAL                                            :: summation

      CALL c_f_pointer(c_tensor_in, tensor_in)
      CALL c_f_pointer(c_matrix_out, matrix_out)

      IF (PRESENT(c_summation)) THEN
         summation = c_summation
         CALL dbcsr_t_copy_tensor_to_matrix(tensor_in, matrix_out, summation)
      ELSE
         CALL dbcsr_t_copy_tensor_to_matrix(tensor_in, matrix_out)
      ENDIF

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_blk_sizes(c_tensor, tensor_dim, c_ind, c_blk_size) &
      BIND(C, name="c_dbcsr_t_blk_sizes")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
      INTEGER(kind=c_int), DIMENSION(tensor_dim), INTENT(IN) :: c_ind
      INTEGER(kind=c_int), DIMENSION(tensor_dim), INTENT(OUT) :: c_blk_size

      TYPE(dbcsr_t_type), POINTER :: tensor

      CALL c_f_pointer(c_tensor, tensor)

      CALL dbcsr_t_blk_sizes(tensor, c_ind + 1, c_blk_size)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_copy(c_tensor_in, tensor_dim, c_tensor_out, c_order, c_summation, c_bounds, c_move_data, c_unit_nr) &
      BIND(C, name="c_dbcsr_t_copy")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_in
      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_out
      INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
      INTEGER(kind=c_int), INTENT(IN), DIMENSION(tensor_dim), &
         OPTIONAL                                     :: c_order
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation
      INTEGER(kind=c_int), INTENT(IN), DIMENSION(2, tensor_dim), &
         OPTIONAL                                         :: c_bounds
      LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         ::c_move_data
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL          :: c_unit_nr

      TYPE(dbcsr_t_type), POINTER                        :: tensor_in, tensor_out
      INTEGER, DIMENSION(:), POINTER                     :: order
      INTEGER, DIMENSION(:, :), POINTER                  :: bounds
      LOGICAL, POINTER                                   :: summation, move_data

      CALL c_f_pointer(c_tensor_in, tensor_in)
      CALL c_f_pointer(c_tensor_out, tensor_out)

      NULLIFY (order)
      IF (PRESENT(c_order)) THEN
         ALLOCATE (order(tensor_dim))
         order = c_order + 1
      ENDIF

      NULLIFY (bounds)
      IF (PRESENT(c_bounds)) THEN
         ALLOCATE (bounds(2, tensor_dim))
         bounds = c_bounds + 1
      ENDIF

      NULLIFY (summation)
      IF (PRESENT(c_summation)) THEN
         ALLOCATE (summation)
         summation = c_summation
      ENDIF

      NULLIFY (move_data)
      IF (PRESENT(c_move_data)) THEN
         ALLOCATE (move_data)
         move_data = c_move_data
      ENDIF

      CALL dbcsr_t_copy(tensor_in, tensor_out, order, summation, bounds, move_data, c_unit_nr)

#:set list = ['order', 'bounds', 'summation', 'move_data']
#:for var in list
      IF (ASSOCIATED(${var}$)) DEALLOCATE (${var}$)
#:endfor

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

   SUBROUTINE c_dbcsr_t_clear(c_tensor) BIND(C, name="c_dbcsr_t_clear")
      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor

      TYPE(dbcsr_t_type), POINTER                        :: tensor

      CALL c_f_pointer(c_tensor, tensor)

      CALL dbcsr_t_clear(tensor)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_get_info(c_tensor, tensor_dim, c_nblks_total, &
                                 c_nfull_total, &
                                 c_nblks_local, &
                                 c_nfull_local, &
                                 c_pdims, &
                                 c_my_ploc, &
                                 ${varlist("nblks_local")}$, &
                                 ${varlist("nblks_total")}$, &
                                 ${varlist("c_blks_local")}$, &
                                 ${varlist("c_proc_dist")}$, &
                                 ${varlist("c_blk_size")}$, &
                                 ${varlist("c_blk_offset")}$, &
                                 c_distribution, &
                                 c_name, &
                                 c_data_type) &
      BIND(C, name="c_dbcsr_t_get_info")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
      INTEGER(kind=c_int), INTENT(IN), VALUE :: tensor_dim
      INTEGER(kind=c_int), DIMENSION(tensor_dim), INTENT(OUT), &
         OPTIONAL :: c_nblks_total
      INTEGER(kind=c_int), DIMENSION(tensor_dim), INTENT(OUT), &
         OPTIONAL :: c_nfull_total
      INTEGER(kind=c_int), DIMENSION(tensor_dim), INTENT(OUT), &
         OPTIONAL :: c_nblks_local
      INTEGER(kind=c_int), DIMENSION(tensor_dim), INTENT(OUT), &
         OPTIONAL :: c_nfull_local
      INTEGER(kind=c_int), DIMENSION(tensor_dim), INTENT(OUT), &
         OPTIONAL :: c_my_ploc
      INTEGER(kind=c_int), DIMENSION(tensor_dim), INTENT(OUT), &
         OPTIONAL :: c_pdims

#:for idim in range(1, maxrank+1)
      INTEGER(kind=c_int), INTENT(IN), VALUE :: nblks_local_${idim}$
      INTEGER(kind=c_int), INTENT(IN), VALUE :: nblks_total_${idim}$
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
         DIMENSION(nblks_local_${idim}$) :: c_blks_local_${idim}$
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
         DIMENSION(nblks_total_${idim}$) :: c_proc_dist_${idim}$
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
         DIMENSION(nblks_total_${idim}$) :: c_blk_size_${idim}$
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
         DIMENSION(nblks_total_${idim}$) :: c_blk_offset_${idim}$
#:endfor

      TYPE(c_ptr), INTENT(OUT), OPTIONAL :: c_distribution
      TYPE(c_ptr), INTENT(OUT), OPTIONAL :: c_name
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: c_data_type

      TYPE(dbcsr_t_type), POINTER :: tensor

      TYPE(dbcsr_t_distribution_type), POINTER :: distribution
      CHARACTER(kind=c_char, len=:), POINTER :: name

      CALL c_f_pointer(c_tensor, tensor)

      NULLIFY (distribution)
      IF (PRESENT(c_distribution)) ALLOCATE (distribution)
      NULLIFY (name)
      IF (PRESENT(c_name)) ALLOCATE (CHARACTER(len=default_string_length) :: name)

      CALL dbcsr_t_get_info(tensor, c_nblks_total, c_nfull_total, c_nblks_local, &
                            c_nfull_local, c_pdims, c_my_ploc, &
                            ${varlist("c_blks_local")}$, &
                            ${varlist("c_proc_dist")}$, &
                            ${varlist("c_blk_size")}$, &
                            ${varlist("c_blk_offset")}$, &
                            distribution, name, &
                            c_data_type)

#:set list = ['blks_local', 'blk_offset']
#:for idim in range(1, maxrank+1)
 #:for var in list
      IF (PRESENT(c_${var}$_${idim}$)) c_${var}$_${idim}$ = c_${var}$_${idim}$-1
 #:endfor
#:endfor

      IF (PRESENT(c_name)) THEN
         name = TRIM(name)//char(0)
         c_name = c_loc(name)
      ENDIF

      IF (PRESENT(c_distribution)) c_distribution = c_loc(distribution)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_get_mapping_info(c_tensor, nd_size, nd_row_size, &
                                         nd_col_size, ndim_nd, ndim1_2d, ndim2_2d, &
                                         c_dims_2d_i8, c_dims_2d, c_dims_nd, &
                                         c_dims1_2d, c_dims2_2d, &
                                         c_map1_2d, c_map2_2d, &
                                         c_map_nd, base, c_col_major) &
      BIND(C, name="c_dbcsr_t_get_mapping_info")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
      !! nd_size = ndims_mapping(map),
      !! nd_row_size = ndims_mapping_row(map),
      !! nd_col_size = ndims_mapping_column(map)
      INTEGER(kind=c_int), INTENT(IN), VALUE :: nd_size, nd_row_size, nd_col_size
      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: ndim_nd, ndim1_2d, ndim2_2d
      INTEGER(kind=c_long_long), INTENT(OUT), DIMENSION(2), &
         OPTIONAL :: c_dims_2d_i8
      INTEGER(kind=c_int), INTENT(OUT), DIMENSION(2), &
         OPTIONAL :: c_dims_2d

      INTEGER(kind=c_int), INTENT(OUT), DIMENSION(nd_size), &
         OPTIONAL :: c_dims_nd
      INTEGER(kind=c_int), INTENT(OUT), DIMENSION(nd_row_size), &
         OPTIONAL :: c_dims1_2d, c_map1_2d
      INTEGER(kind=c_int), INTENT(OUT), DIMENSION(nd_col_size), &
         OPTIONAL :: c_dims2_2d, c_map2_2d
      INTEGER(kind=c_int), INTENT(OUT), DIMENSION(nd_size), &
         OPTIONAL :: c_map_nd

      INTEGER(kind=c_int), INTENT(OUT), OPTIONAL :: base
      LOGICAL(kind=c_bool), INTENT(OUT), OPTIONAL :: c_col_major

      TYPE(dbcsr_t_type), POINTER :: tensor

      LOGICAL                     :: col_major

      CALL c_f_pointer(c_tensor, tensor)

      IF (PRESENT(c_col_major)) THEN
         CALL dbcsr_t_get_mapping_info(tensor%nd_index_blk, ndim_nd, ndim1_2d, &
                                       ndim2_2d, c_dims_2d_i8, c_dims_2d, c_dims_nd, &
                                       c_dims1_2d, c_dims2_2d, &
                                       c_map1_2d, c_map2_2d, c_map_nd, &
                                       base, col_major)
         c_col_major = LOGICAL(col_major, kind=c_bool)
      ELSE
         CALL dbcsr_t_get_mapping_info(tensor%nd_index_blk, ndim_nd, ndim1_2d, &
                                       ndim2_2d, c_dims_2d_i8, c_dims_2d, c_dims_nd, &
                                       c_dims1_2d, c_dims2_2d, &
                                       c_map1_2d, c_map2_2d, c_map_nd, &
                                       base)
      ENDIF

#:set list = ['dims_nd', 'dims1_2d', 'dims2_2d', 'map1_2d', 'map2_2d', 'map_nd']
#:for var in list
      IF (PRESENT(c_${var}$)) c_${var}$ = c_${var}$-1
#:endfor

   END SUBROUTINE

   FUNCTION c_dbcsr_t_get_num_blocks(c_tensor) RESULT(c_num_blocks) &
      BIND(C, name="c_dbcsr_t_get_num_blocks")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor

      TYPE(dbcsr_t_type), POINTER :: tensor
      INTEGER(kind=c_int) :: c_num_blocks

      CALL c_f_pointer(c_tensor, tensor)
      c_num_blocks = dbcsr_t_get_num_blocks(tensor)

   END FUNCTION

   FUNCTION c_dbcsr_t_get_num_blocks_total(c_tensor) RESULT(c_num_blocks_total) &
      BIND(C, name="c_dbcsr_t_get_num_blocks_total")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor

      TYPE(dbcsr_t_type), POINTER :: tensor
      INTEGER(kind=c_long_long) :: c_num_blocks_total

      CALL c_f_pointer(c_tensor, tensor)
      c_num_blocks_total = dbcsr_t_get_num_blocks_total(tensor)

   END FUNCTION

   FUNCTION c_dbcsr_t_nblks_local(c_tensor, idim) &
      BIND(C, name="c_dbcsr_t_nblks_local")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
      INTEGER(kind=c_int), INTENT(IN), VALUE :: idim
      INTEGER(kind=c_int) :: c_dbcsr_t_nblks_local

      TYPE(dbcsr_t_type), POINTER :: tensor
      CALL c_f_pointer(c_tensor, tensor)

      c_dbcsr_t_nblks_local = dbcsr_t_nblks_local(tensor, idim + 1)

   END FUNCTION

   FUNCTION c_dbcsr_t_nblks_total(c_tensor, idim) &
      BIND(C, name="c_dbcsr_t_nblks_total")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
      INTEGER(kind=c_int), INTENT(IN), VALUE :: idim
      INTEGER(kind=c_int) :: c_dbcsr_t_nblks_total

      TYPE(dbcsr_t_type), POINTER :: tensor
      CALL c_f_pointer(c_tensor, tensor)

      c_dbcsr_t_nblks_total = dbcsr_t_nblks_total(tensor, idim + 1)

   END FUNCTION

   SUBROUTINE c_dbcsr_t_dims(c_tensor, tensor_dim, c_dims) &
      BIND(C, name="c_dbcsr_t_dims")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
      INTEGER(kind=c_int), INTENT(IN), VALUE :: tensor_dim
      INTEGER(kind=c_int), DIMENSION(tensor_dim), &
         INTENT(OUT) :: c_dims

      TYPE(dbcsr_t_type), POINTER :: tensor

      CALL c_f_pointer(c_tensor, tensor)

      CALL dbcsr_t_dims(tensor, c_dims)

   END SUBROUTINE

   FUNCTION c_dbcsr_t_ndims(c_tensor) BIND(C, name="c_dbcsr_t_ndims")
      TYPE(c_ptr), INTENT(IN), VALUE                  :: c_tensor
      TYPE(dbcsr_t_type), POINTER                     :: tensor
      INTEGER(kind=c_int)                             :: c_dbcsr_t_ndims

      CALL c_f_pointer(c_tensor, tensor)

      c_dbcsr_t_ndims = dbcsr_t_ndims(tensor)
   END FUNCTION

   FUNCTION c_dbcsr_t_ndims_matrix_row(c_tensor) &
      BIND(C, name="c_dbcsr_t_ndims_matrix_row")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
      TYPE(dbcsr_t_type), POINTER :: tensor
      INTEGER(kind=c_long_long) :: c_dbcsr_t_ndims_matrix_row

      CALL c_f_pointer(c_tensor, tensor)

      c_dbcsr_t_ndims_matrix_row = &
         dbcsr_t_ndims_matrix_row(tensor)

   END FUNCTION

   FUNCTION c_dbcsr_t_ndims_matrix_column(c_tensor) &
      BIND(C, name="c_dbcsr_t_ndims_matrix_column")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
      TYPE(dbcsr_t_type), POINTER :: tensor
      INTEGER(kind=c_long_long) :: c_dbcsr_t_ndims_matrix_column

      CALL c_f_pointer(c_tensor, tensor)

      c_dbcsr_t_ndims_matrix_column = &
         dbcsr_t_ndims_matrix_column(tensor)

   END FUNCTION

   FUNCTION c_dbcsr_t_get_nze(c_tensor) &
      BIND(C, name="c_dbcsr_t_get_nze")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
      TYPE(dbcsr_t_type), POINTER :: tensor
      INTEGER(kind=c_int) :: c_dbcsr_t_get_nze

      CALL c_f_pointer(c_tensor, tensor)
      c_dbcsr_t_get_nze = dbcsr_t_get_nze(tensor)
   END FUNCTION

   FUNCTION c_dbcsr_t_get_nze_total(c_tensor) &
      BIND(C, name="c_dbcsr_t_get_nze_total")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
      TYPE(dbcsr_t_type), POINTER :: tensor
      INTEGER(kind=c_long_long) :: c_dbcsr_t_get_nze_total

      CALL c_f_pointer(c_tensor, tensor)
      c_dbcsr_t_get_nze_total = dbcsr_t_get_nze_total(tensor)
   END FUNCTION

   FUNCTION c_dbcsr_t_max_nblks_local(c_tensor) RESULT(max_blks) &
      BIND(C, name="c_dbcsr_t_max_nblks_local")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
      INTEGER(kind=c_long_long) :: max_blks
      TYPE(dbcsr_t_type), POINTER :: tensor

      CALL c_f_pointer(c_tensor, tensor)

      max_blks = dbcsr_t_max_nblks_local(tensor)
   END FUNCTION

#:for dparam, dtype, dsuffix, basetype, cftype in cf_dtype_float_list
   SUBROUTINE c_dbcsr_t_get_data_${dsuffix}$ (c_tensor, c_data, c_data_size, &
                                              c_select_data_type, c_lb, c_ub) BIND(C, name="c_dbcsr_t_get_data_${dsuffix}$")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
      TYPE(c_ptr), INTENT(INOUT) :: c_data
      INTEGER(kind=c_long_long), INTENT(INOUT) :: c_data_size
      ${basetype}$ (kind=${cftype}$), INTENT(IN), VALUE :: c_select_data_type
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_lb, c_ub

      TYPE(dbcsr_t_type), POINTER :: tensor
      ${basetype}$ (kind=${cftype}$), DIMENSION(:), POINTER :: DATA
      INTEGER, POINTER :: lb, ub

      CALL c_f_pointer(c_tensor, tensor)

      NULLIFY (lb)
      IF (PRESENT(c_lb)) THEN
         ALLOCATE (lb)
         lb = c_lb + 1
      ENDIF

      NULLIFY (ub)
      IF (PRESENT(c_ub)) THEN
         ALLOCATE (ub)
         ub = ub + 1
      ENDIF

      DATA => dbcsr_get_data_p(tensor%matrix_rep%matrix%data_area, &
                               c_select_data_type, lb, ub)

      c_data = c_loc(DATA)
      c_data_size = SIZE(DATA)

      IF (ASSOCIATED(lb)) DEALLOCATE (lb)
      IF (ASSOCIATED(ub)) DEALLOCATE (ub)

   END SUBROUTINE
#:endfor

   SUBROUTINE c_dbcsr_t_batched_contract_init(c_tensor) &
      BIND(C, name="c_dbcsr_t_batched_contract_init")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
      TYPE(dbcsr_t_type), POINTER :: tensor

      CALL c_f_pointer(c_tensor, tensor)

      CALL dbcsr_t_batched_contract_init(tensor)

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_batched_contract_finalize(c_tensor, c_unit_nr) &
      BIND(C, name="c_dbcsr_t_batched_contract_finalize")

      TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
      INTEGER(kind=c_int), OPTIONAL :: c_unit_nr

      TYPE(dbcsr_t_type), POINTER :: tensor

      CALL c_f_pointer(c_tensor, tensor)
      IF (PRESENT(c_unit_nr)) THEN
         CALL dbcsr_t_batched_contract_finalize(tensor, c_unit_nr)
      ELSE
         CALL dbcsr_t_batched_contract_finalize(tensor)
      ENDIF

   END SUBROUTINE

END MODULE
