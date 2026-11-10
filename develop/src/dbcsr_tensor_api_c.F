# 1 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F" 1
!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

# 1 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp" 1
# 9 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp"

# 241 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp"
# 11 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F" 2
# 12 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 13 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

MODULE dbcsr_tensor_api_c

   USE, INTRINSIC :: ISO_C_BINDING, ONLY: c_loc, c_ptr, c_double, c_sizeof, C_NULL_CHAR, &
                                                                             c_float, c_f_pointer, c_int, c_long_long, &
                                                                             c_char, c_null_ptr, c_bool, c_associated, &
                                                                             c_float_complex, c_double_complex
   USE dbcsr_api, ONLY: dbcsr_type, dbcsr_scale
   USE dbcsr_allocate_wrap, ONLY: allocate_any
   USE dbcsr_tensor_api
   USE dbcsr_kinds, ONLY: &
      real_8, real_4, default_string_length, int_8
# 25 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
   USE dbcsr_data_types, ONLY: dbcsr_scalar_type
   USE dbcsr_data_methods, ONLY: dbcsr_scalar
   USE dbcsr_machine, ONLY: default_output_unit
   USE dbcsr_tensor_index, ONLY: &
      dbcsr_t_get_mapping_info, ndims_mapping, nd_to_2d_mapping
   USE dbcsr_tensor_types, ONLY: dbcsr_t_create, dbcsr_t_type
   USE dbcsr_tas_types, ONLY: dbcsr_tas_type
   USE dbcsr_tas_base, ONLY: dbcsr_tas_get_block_p
   USE dbcsr_tensor_index, ONLY: get_2d_indices_tensor
   USE dbcsr_data_methods, ONLY: dbcsr_get_data_p

   IMPLICIT NONE
   PRIVATE

   PUBLIC :: c_dbcsr_t_finalize, &
             c_dbcsr_t_pgrid_create_expert, &
             c_dbcsr_t_pgrid_create, &
             c_dbcsr_t_pgrid_destroy, &
             c_dbcsr_t_distribution_new, &
             c_dbcsr_t_distribution_destroy, &
             c_dbcsr_t_create_new, &
             c_dbcsr_t_create_template, &
             c_dbcsr_t_create_matrix, &
             c_dbcsr_t_destroy, &
             c_dbcsr_t_get_stored_coordinates, &
             c_dbcsr_t_reserve_blocks_index, &
             c_dbcsr_t_reserve_blocks_template, &
             c_ndims_iterator, &
             c_dbcsr_t_iterator_start, &
             c_dbcsr_t_iterator_stop, &
             c_dbcsr_t_iterator_next_block, &
             c_dbcsr_t_iterator_blocks_left, &
             c_dbcsr_t_split_blocks, &
             c_dbcsr_t_copy_matrix_to_tensor, &
             c_dbcsr_t_copy_tensor_to_matrix, &
             c_dbcsr_t_blk_sizes, &
             c_dbcsr_t_copy, &
             c_dbcsr_t_clear, &
             c_dbcsr_t_get_info, &
             c_dbcsr_t_get_mapping_info, &
             c_dbcsr_t_get_num_blocks, &
             c_dbcsr_t_get_num_blocks_total, &
             c_dbcsr_t_nblks_local, &
             c_dbcsr_t_nblks_total, &
             c_dbcsr_t_dims, &
             c_dbcsr_t_ndims, &
             c_dbcsr_t_ndims_matrix_row, &
             c_dbcsr_t_ndims_matrix_column, &
             c_dbcsr_t_get_nze, &
             c_dbcsr_t_get_nze_total, &
             c_dbcsr_t_max_nblks_local, &
             c_dbcsr_t_batched_contract_init, &
             c_dbcsr_t_batched_contract_finalize

# 80 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
      PUBLIC :: c_dbcsr_t_contract_r_dp, &
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_contract_index_r_dp, &
# 82 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_filter_r_dp, &
# 83 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_set_r_dp, &
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_scale_r_dp, &
# 85 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_get_data_r_dp
# 86 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         PUBLIC :: c_dbcsr_t_get_2d_block_r_dp, &
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_put_2d_block_r_dp, &
# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_get_2d_block_p_r_dp
# 90 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         PUBLIC :: c_dbcsr_t_get_3d_block_r_dp, &
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_put_3d_block_r_dp, &
# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_get_3d_block_p_r_dp
# 90 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         PUBLIC :: c_dbcsr_t_get_4d_block_r_dp, &
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_put_4d_block_r_dp, &
# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_get_4d_block_p_r_dp
# 90 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 91 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 80 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
      PUBLIC :: c_dbcsr_t_contract_r_sp, &
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_contract_index_r_sp, &
# 82 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_filter_r_sp, &
# 83 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_set_r_sp, &
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_scale_r_sp, &
# 85 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_get_data_r_sp
# 86 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         PUBLIC :: c_dbcsr_t_get_2d_block_r_sp, &
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_put_2d_block_r_sp, &
# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_get_2d_block_p_r_sp
# 90 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         PUBLIC :: c_dbcsr_t_get_3d_block_r_sp, &
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_put_3d_block_r_sp, &
# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_get_3d_block_p_r_sp
# 90 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         PUBLIC :: c_dbcsr_t_get_4d_block_r_sp, &
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_put_4d_block_r_sp, &
# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_get_4d_block_p_r_sp
# 90 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 91 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 80 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
      PUBLIC :: c_dbcsr_t_contract_c_dp, &
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_contract_index_c_dp, &
# 82 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_filter_c_dp, &
# 83 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_set_c_dp, &
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_scale_c_dp, &
# 85 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_get_data_c_dp
# 86 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         PUBLIC :: c_dbcsr_t_get_2d_block_c_dp, &
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_put_2d_block_c_dp, &
# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_get_2d_block_p_c_dp
# 90 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         PUBLIC :: c_dbcsr_t_get_3d_block_c_dp, &
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_put_3d_block_c_dp, &
# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_get_3d_block_p_c_dp
# 90 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         PUBLIC :: c_dbcsr_t_get_4d_block_c_dp, &
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_put_4d_block_c_dp, &
# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_get_4d_block_p_c_dp
# 90 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 91 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 80 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
      PUBLIC :: c_dbcsr_t_contract_c_sp, &
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_contract_index_c_sp, &
# 82 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_filter_c_sp, &
# 83 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_set_c_sp, &
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_scale_c_sp, &
# 85 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                c_dbcsr_t_get_data_c_sp
# 86 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         PUBLIC :: c_dbcsr_t_get_2d_block_c_sp, &
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_put_2d_block_c_sp, &
# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_get_2d_block_p_c_sp
# 90 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         PUBLIC :: c_dbcsr_t_get_3d_block_c_sp, &
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_put_3d_block_c_sp, &
# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_get_3d_block_p_c_sp
# 90 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 87 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         PUBLIC :: c_dbcsr_t_get_4d_block_c_sp, &
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_put_4d_block_c_sp, &
# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                   c_dbcsr_t_get_4d_block_p_c_sp
# 90 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 91 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 92 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

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
      END DO
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

      CALL dbcsr_t_pgrid_create_expert(fcomm, c_dims, pgrid, c_map1_2d + 1, c_map2_2d + 1, c_tensor_dims, &
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
      END IF

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
      END IF

      IF (ASSOCIATED(pgrid)) DEALLOCATE (pgrid)
      c_pgrid = c_null_ptr

   END SUBROUTINE

   SUBROUTINE c_dbcsr_t_distribution_new(c_dist, c_pgrid, c_nd_dist_1, c_nd_dist_1_size, c_nd_dist_2, c_nd_dist_2_size,&
# 204 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
       & c_nd_dist_3, c_nd_dist_3_size, c_nd_dist_4, c_nd_dist_4_size) &
# 205 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
      bind(C, name="c_dbcsr_t_distribution_new")

      TYPE(c_ptr), INTENT(OUT)                           :: c_dist
      TYPE(c_ptr), INTENT(IN), value                     :: c_pgrid

      INTEGER(kind=c_int), INTENT(in), value :: c_nd_dist_1_size, c_nd_dist_2_size, c_nd_dist_3_size, c_nd_dist_4_size
# 211 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
      INTEGER(kind=c_int), INTENT(IN), TARGET, OPTIONAL :: c_nd_dist_1(c_nd_dist_1_size), c_nd_dist_2(c_nd_dist_2_size),&
# 211 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
          & c_nd_dist_3(c_nd_dist_3_size), c_nd_dist_4(c_nd_dist_4_size)
# 212 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      TYPE(dbcsr_t_pgrid_type), POINTER                  :: pgrid
      TYPE(dbcsr_t_distribution_type), POINTER           :: dist

      INTEGER, DIMENSION(:), POINTER                     ::  nd_dist_1, nd_dist_2, nd_dist_3, nd_dist_4
# 217 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      ALLOCATE (dist)

      CALL c_f_pointer(c_pgrid, pgrid)

# 223 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         NULLIFY (nd_dist_1)
# 224 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_nd_dist_1)) THEN
# 225 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            nd_dist_1 => c_nd_dist_1
# 226 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         END IF
# 223 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         NULLIFY (nd_dist_2)
# 224 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_nd_dist_2)) THEN
# 225 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            nd_dist_2 => c_nd_dist_2
# 226 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         END IF
# 223 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         NULLIFY (nd_dist_3)
# 224 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_nd_dist_3)) THEN
# 225 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            nd_dist_3 => c_nd_dist_3
# 226 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         END IF
# 223 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         NULLIFY (nd_dist_4)
# 224 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_nd_dist_4)) THEN
# 225 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            nd_dist_4 => c_nd_dist_4
# 226 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         END IF
# 228 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      CALL dbcsr_t_distribution_new(dist, pgrid, nd_dist_1, nd_dist_2, nd_dist_3, nd_dist_4)
# 230 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

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
                                   c_map2_2d, map2_2d_size, c_data_type, c_blk_size_1, c_blk_size_1_size, c_blk_size_2,&
# 249 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                       & c_blk_size_2_size, c_blk_size_3, c_blk_size_3_size, c_blk_size_4, c_blk_size_4_size) &
# 250 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
      bind(C, name="c_dbcsr_t_create_new")

      TYPE(c_ptr), INTENT(OUT)                           :: c_tensor
      TYPE(c_ptr), INTENT(IN), value                     :: c_dist, c_name

      INTEGER(kind=c_int), INTENT(in), value             :: map1_2d_size
      INTEGER(kind=c_int), INTENT(in), TARGET            :: c_map1_2d(map1_2d_size)
      INTEGER(kind=c_int), INTENT(in), value             :: map2_2d_size
      INTEGER(kind=c_int), INTENT(in), TARGET            :: c_map2_2d(map2_2d_size)

      INTEGER(kind=c_int), INTENT(IN), OPTIONAL          :: c_data_type

      INTEGER(kind=c_int), INTENT(in), value             :: c_blk_size_1_size, c_blk_size_2_size, c_blk_size_3_size,&
# 262 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
          & c_blk_size_4_size
# 263 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
      INTEGER(kind=c_int), INTENT(IN), TARGET, OPTIONAL  :: c_blk_size_1(c_blk_size_1_size), c_blk_size_2(c_blk_size_2_size),&
# 263 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
          & c_blk_size_3(c_blk_size_3_size), c_blk_size_4(c_blk_size_4_size)
# 264 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      CHARACTER(:, kind=c_char), ALLOCATABLE             :: fname

      TYPE(dbcsr_t_type), POINTER                        :: tensor
      TYPE(dbcsr_t_distribution_type), POINTER           :: dist

      INTEGER, DIMENSION(:), POINTER                     :: blk_size_1, blk_size_2, blk_size_3, blk_size_4
# 271 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      ALLOCATE (tensor)

      CALL c_f_pointer(c_dist, dist)
      CALL c_f_string(c_name, fname)

# 278 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         NULLIFY (blk_size_1)
# 279 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_blk_size_1)) THEN
# 280 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            blk_size_1 => c_blk_size_1
# 281 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         END IF
# 278 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         NULLIFY (blk_size_2)
# 279 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_blk_size_2)) THEN
# 280 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            blk_size_2 => c_blk_size_2
# 281 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         END IF
# 278 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         NULLIFY (blk_size_3)
# 279 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_blk_size_3)) THEN
# 280 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            blk_size_3 => c_blk_size_3
# 281 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         END IF
# 278 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         NULLIFY (blk_size_4)
# 279 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_blk_size_4)) THEN
# 280 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            blk_size_4 => c_blk_size_4
# 281 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         END IF
# 283 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      CALL dbcsr_t_create(tensor, fname, dist, c_map1_2d + 1, c_map2_2d + 1, c_data_type, &
                          blk_size_1, blk_size_2, blk_size_3, blk_size_4)
# 286 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

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
         map1_2d = c_map1_2d + 1
         map2_2d = c_map2_2d + 1
      END IF

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
      END IF

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

# 384 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      SUBROUTINE c_dbcsr_t_contract_r_dp (c_alpha, c_tensor_1, c_tensor_2, c_beta, c_tensor_3, &
# 386 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                 c_contract_1, contract_1_size, &
                                                 c_notcontract_1, notcontract_1_size, &
                                                 c_contract_2, contract_2_size, &
                                                 c_notcontract_2, notcontract_2_size, &
                                                 c_map_1, map_1_size, c_map_2, map_2_size, &
                                                 c_bounds_1, c_bounds_2, c_bounds_3, &
                                                 c_optimize_dist, c_pgrid_opt_1, c_pgrid_opt_2, c_pgrid_opt_3, &
                                                 c_filter_eps, c_flop, c_move_data, c_retain_sparsity, &
                                                 c_unit_nr, c_log_verbose) &
         BIND(C, name="c_dbcsr_t_contract_r_dp")
# 396 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         REAL (kind=c_double), INTENT(IN), VALUE     ::  c_alpha
# 398 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_1, c_tensor_2
         REAL (kind=c_double), INTENT(IN), VALUE     :: c_beta
# 400 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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

# 444 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_1)) THEN
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_1 (2, SIZE(c_bounds_1, 2)))
# 446 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_1 (:, :) = c_bounds_1 (:, :) + 1
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 444 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_2)) THEN
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_2 (2, SIZE(c_bounds_2, 2)))
# 446 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_2 (:, :) = c_bounds_2 (:, :) + 1
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 444 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_3)) THEN
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_3 (2, SIZE(c_bounds_3, 2)))
# 446 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_3 (:, :) = c_bounds_3 (:, :) + 1
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (optimize_dist)
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_optimize_dist)) THEN
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (optimize_dist)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               optimize_dist = c_optimize_dist
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (move_data)
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_move_data)) THEN
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (move_data)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               move_data = c_move_data
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (retain_sparsity)
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_retain_sparsity)) THEN
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (retain_sparsity)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               retain_sparsity = c_retain_sparsity
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (log_verbose)
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_log_verbose)) THEN
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (log_verbose)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               log_verbose = c_log_verbose
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 460 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 461 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (pgrid_opt_1)
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (pgrid_opt_2)
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (pgrid_opt_3)
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 466 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF(PRESENT(c_pgrid_opt_1) .AND. PRESENT(c_pgrid_opt_2) .AND. PRESENT(c_pgrid_opt_3)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_1 = pgrid_opt_1, pgrid_opt_2 = pgrid_opt_2, pgrid_opt_3 = pgrid_opt_3, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_2) .AND. PRESENT(c_pgrid_opt_3) .AND. .NOT. PRESENT(c_pgrid_opt_1)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_2 = pgrid_opt_2, pgrid_opt_3 = pgrid_opt_3, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_1) .AND. PRESENT(c_pgrid_opt_3) .AND. .NOT. PRESENT(c_pgrid_opt_2)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_1 = pgrid_opt_1, pgrid_opt_3 = pgrid_opt_3, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_3) .AND. .NOT. PRESENT(c_pgrid_opt_1) .AND. .NOT. PRESENT(c_pgrid_opt_2)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_3 = pgrid_opt_3, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_1) .AND. PRESENT(c_pgrid_opt_2) .AND. .NOT. PRESENT(c_pgrid_opt_3)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_1 = pgrid_opt_1, pgrid_opt_2 = pgrid_opt_2, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_2) .AND. .NOT. PRESENT(c_pgrid_opt_1) .AND. .NOT. PRESENT(c_pgrid_opt_3)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_2 = pgrid_opt_2, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_1) .AND. .NOT. PRESENT(c_pgrid_opt_2) .AND. .NOT. PRESENT(c_pgrid_opt_3)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_1 = pgrid_opt_1, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 483 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
ENDIF
# 484 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 486 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_pgrid_opt_1)) c_pgrid_opt_1 = c_loc(pgrid_opt_1)
# 487 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 486 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_pgrid_opt_2)) c_pgrid_opt_2 = c_loc(pgrid_opt_2)
# 487 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 486 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_pgrid_opt_3)) c_pgrid_opt_3 = c_loc(pgrid_opt_3)
# 487 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 488 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 490 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 491 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(optimize_dist)) DEALLOCATE (optimize_dist)
# 492 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 491 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(move_data)) DEALLOCATE (move_data)
# 492 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 491 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(retain_sparsity)) DEALLOCATE (retain_sparsity)
# 492 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 491 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(log_verbose)) DEALLOCATE (log_verbose)
# 492 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 493 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 495 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ALLOCATED(bounds_1)) DEALLOCATE (bounds_1)
# 496 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 495 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ALLOCATED(bounds_2)) DEALLOCATE (bounds_2)
# 496 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 495 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ALLOCATED(bounds_3)) DEALLOCATE (bounds_3)
# 496 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 497 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         IF (PRESENT(c_unit_nr)) THEN
            IF (c_unit_nr .GE. 0) THEN
               flush (c_unit_nr)
            END IF
         END IF

      END SUBROUTINE

      SUBROUTINE c_dbcsr_t_contract_index_r_dp (c_alpha, c_tensor_1, &
# 507 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                       c_tensor_2, c_beta, c_tensor_3, &
                                                       c_contract_1, contract_1_size, &
                                                       c_notcontract_1, notcontract_1_size, &
                                                       c_contract_2, contract_2_size, &
                                                       c_notcontract_2, notcontract_2_size, &
                                                       c_map_1, map_1_size, c_map_2, map_2_size, &
                                                       c_bounds_1, c_bounds_2, c_bounds_3, &
                                                       c_filter_eps, c_nblks_local, c_result_index, &
                                                       result_index_size, tensor3_dim) &
         BIND(C, name="c_dbcsr_t_contract_index_r_dp")
# 517 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         REAL (kind=c_double), INTENT(IN), VALUE     :: c_alpha
# 519 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_1, c_tensor_2
         REAL (kind=c_double), INTENT(IN), VALUE     :: c_beta
# 521 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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

# 556 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (bounds_1)
# 557 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_1)) THEN
# 558 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_1 (2, SIZE(c_bounds_1, 2)))
# 559 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_1 = c_bounds_1+1
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 556 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (bounds_2)
# 557 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_2)) THEN
# 558 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_2 (2, SIZE(c_bounds_2, 2)))
# 559 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_2 = c_bounds_2+1
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 556 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (bounds_3)
# 557 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_3)) THEN
# 558 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_3 (2, SIZE(c_bounds_3, 2)))
# 559 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_3 = c_bounds_3+1
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 562 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         CALL dbcsr_t_contract_index(dbcsr_scalar(c_alpha), tensor_1, &
                                     tensor_2, dbcsr_scalar(c_beta), tensor_3, &
                                     c_contract_1 + 1, c_notcontract_1 + 1, &
                                     c_contract_2 + 1, c_notcontract_2 + 1, &
                                     c_map_1 + 1, c_map_2 + 1, &
                                     bounds_1, bounds_2, bounds_3, &
                                     c_filter_eps, c_nblks_local, result_index)

         c_result_index = result_index - 1

# 574 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(bounds_1)) DEALLOCATE (bounds_1)
# 575 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 574 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(bounds_2)) DEALLOCATE (bounds_2)
# 575 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 574 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(bounds_3)) DEALLOCATE (bounds_3)
# 575 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 576 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      END SUBROUTINE

# 384 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      SUBROUTINE c_dbcsr_t_contract_r_sp (c_alpha, c_tensor_1, c_tensor_2, c_beta, c_tensor_3, &
# 386 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                 c_contract_1, contract_1_size, &
                                                 c_notcontract_1, notcontract_1_size, &
                                                 c_contract_2, contract_2_size, &
                                                 c_notcontract_2, notcontract_2_size, &
                                                 c_map_1, map_1_size, c_map_2, map_2_size, &
                                                 c_bounds_1, c_bounds_2, c_bounds_3, &
                                                 c_optimize_dist, c_pgrid_opt_1, c_pgrid_opt_2, c_pgrid_opt_3, &
                                                 c_filter_eps, c_flop, c_move_data, c_retain_sparsity, &
                                                 c_unit_nr, c_log_verbose) &
         BIND(C, name="c_dbcsr_t_contract_r_sp")
# 396 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         REAL (kind=c_float), INTENT(IN), VALUE     ::  c_alpha
# 398 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_1, c_tensor_2
         REAL (kind=c_float), INTENT(IN), VALUE     :: c_beta
# 400 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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

# 444 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_1)) THEN
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_1 (2, SIZE(c_bounds_1, 2)))
# 446 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_1 (:, :) = c_bounds_1 (:, :) + 1
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 444 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_2)) THEN
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_2 (2, SIZE(c_bounds_2, 2)))
# 446 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_2 (:, :) = c_bounds_2 (:, :) + 1
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 444 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_3)) THEN
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_3 (2, SIZE(c_bounds_3, 2)))
# 446 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_3 (:, :) = c_bounds_3 (:, :) + 1
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (optimize_dist)
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_optimize_dist)) THEN
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (optimize_dist)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               optimize_dist = c_optimize_dist
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (move_data)
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_move_data)) THEN
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (move_data)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               move_data = c_move_data
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (retain_sparsity)
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_retain_sparsity)) THEN
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (retain_sparsity)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               retain_sparsity = c_retain_sparsity
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (log_verbose)
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_log_verbose)) THEN
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (log_verbose)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               log_verbose = c_log_verbose
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 460 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 461 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (pgrid_opt_1)
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (pgrid_opt_2)
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (pgrid_opt_3)
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 466 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF(PRESENT(c_pgrid_opt_1) .AND. PRESENT(c_pgrid_opt_2) .AND. PRESENT(c_pgrid_opt_3)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_1 = pgrid_opt_1, pgrid_opt_2 = pgrid_opt_2, pgrid_opt_3 = pgrid_opt_3, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_2) .AND. PRESENT(c_pgrid_opt_3) .AND. .NOT. PRESENT(c_pgrid_opt_1)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_2 = pgrid_opt_2, pgrid_opt_3 = pgrid_opt_3, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_1) .AND. PRESENT(c_pgrid_opt_3) .AND. .NOT. PRESENT(c_pgrid_opt_2)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_1 = pgrid_opt_1, pgrid_opt_3 = pgrid_opt_3, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_3) .AND. .NOT. PRESENT(c_pgrid_opt_1) .AND. .NOT. PRESENT(c_pgrid_opt_2)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_3 = pgrid_opt_3, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_1) .AND. PRESENT(c_pgrid_opt_2) .AND. .NOT. PRESENT(c_pgrid_opt_3)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_1 = pgrid_opt_1, pgrid_opt_2 = pgrid_opt_2, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_2) .AND. .NOT. PRESENT(c_pgrid_opt_1) .AND. .NOT. PRESENT(c_pgrid_opt_3)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_2 = pgrid_opt_2, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_1) .AND. .NOT. PRESENT(c_pgrid_opt_2) .AND. .NOT. PRESENT(c_pgrid_opt_3)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_1 = pgrid_opt_1, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 483 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
ENDIF
# 484 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 486 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_pgrid_opt_1)) c_pgrid_opt_1 = c_loc(pgrid_opt_1)
# 487 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 486 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_pgrid_opt_2)) c_pgrid_opt_2 = c_loc(pgrid_opt_2)
# 487 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 486 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_pgrid_opt_3)) c_pgrid_opt_3 = c_loc(pgrid_opt_3)
# 487 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 488 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 490 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 491 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(optimize_dist)) DEALLOCATE (optimize_dist)
# 492 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 491 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(move_data)) DEALLOCATE (move_data)
# 492 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 491 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(retain_sparsity)) DEALLOCATE (retain_sparsity)
# 492 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 491 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(log_verbose)) DEALLOCATE (log_verbose)
# 492 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 493 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 495 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ALLOCATED(bounds_1)) DEALLOCATE (bounds_1)
# 496 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 495 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ALLOCATED(bounds_2)) DEALLOCATE (bounds_2)
# 496 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 495 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ALLOCATED(bounds_3)) DEALLOCATE (bounds_3)
# 496 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 497 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         IF (PRESENT(c_unit_nr)) THEN
            IF (c_unit_nr .GE. 0) THEN
               flush (c_unit_nr)
            END IF
         END IF

      END SUBROUTINE

      SUBROUTINE c_dbcsr_t_contract_index_r_sp (c_alpha, c_tensor_1, &
# 507 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                       c_tensor_2, c_beta, c_tensor_3, &
                                                       c_contract_1, contract_1_size, &
                                                       c_notcontract_1, notcontract_1_size, &
                                                       c_contract_2, contract_2_size, &
                                                       c_notcontract_2, notcontract_2_size, &
                                                       c_map_1, map_1_size, c_map_2, map_2_size, &
                                                       c_bounds_1, c_bounds_2, c_bounds_3, &
                                                       c_filter_eps, c_nblks_local, c_result_index, &
                                                       result_index_size, tensor3_dim) &
         BIND(C, name="c_dbcsr_t_contract_index_r_sp")
# 517 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         REAL (kind=c_float), INTENT(IN), VALUE     :: c_alpha
# 519 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_1, c_tensor_2
         REAL (kind=c_float), INTENT(IN), VALUE     :: c_beta
# 521 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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

# 556 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (bounds_1)
# 557 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_1)) THEN
# 558 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_1 (2, SIZE(c_bounds_1, 2)))
# 559 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_1 = c_bounds_1+1
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 556 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (bounds_2)
# 557 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_2)) THEN
# 558 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_2 (2, SIZE(c_bounds_2, 2)))
# 559 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_2 = c_bounds_2+1
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 556 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (bounds_3)
# 557 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_3)) THEN
# 558 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_3 (2, SIZE(c_bounds_3, 2)))
# 559 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_3 = c_bounds_3+1
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 562 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         CALL dbcsr_t_contract_index(dbcsr_scalar(c_alpha), tensor_1, &
                                     tensor_2, dbcsr_scalar(c_beta), tensor_3, &
                                     c_contract_1 + 1, c_notcontract_1 + 1, &
                                     c_contract_2 + 1, c_notcontract_2 + 1, &
                                     c_map_1 + 1, c_map_2 + 1, &
                                     bounds_1, bounds_2, bounds_3, &
                                     c_filter_eps, c_nblks_local, result_index)

         c_result_index = result_index - 1

# 574 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(bounds_1)) DEALLOCATE (bounds_1)
# 575 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 574 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(bounds_2)) DEALLOCATE (bounds_2)
# 575 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 574 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(bounds_3)) DEALLOCATE (bounds_3)
# 575 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 576 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      END SUBROUTINE

# 384 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      SUBROUTINE c_dbcsr_t_contract_c_dp (c_alpha, c_tensor_1, c_tensor_2, c_beta, c_tensor_3, &
# 386 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                 c_contract_1, contract_1_size, &
                                                 c_notcontract_1, notcontract_1_size, &
                                                 c_contract_2, contract_2_size, &
                                                 c_notcontract_2, notcontract_2_size, &
                                                 c_map_1, map_1_size, c_map_2, map_2_size, &
                                                 c_bounds_1, c_bounds_2, c_bounds_3, &
                                                 c_optimize_dist, c_pgrid_opt_1, c_pgrid_opt_2, c_pgrid_opt_3, &
                                                 c_filter_eps, c_flop, c_move_data, c_retain_sparsity, &
                                                 c_unit_nr, c_log_verbose) &
         BIND(C, name="c_dbcsr_t_contract_c_dp")
# 396 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         COMPLEX (kind=c_double_complex), INTENT(IN), VALUE     ::  c_alpha
# 398 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_1, c_tensor_2
         COMPLEX (kind=c_double_complex), INTENT(IN), VALUE     :: c_beta
# 400 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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

# 444 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_1)) THEN
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_1 (2, SIZE(c_bounds_1, 2)))
# 446 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_1 (:, :) = c_bounds_1 (:, :) + 1
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 444 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_2)) THEN
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_2 (2, SIZE(c_bounds_2, 2)))
# 446 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_2 (:, :) = c_bounds_2 (:, :) + 1
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 444 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_3)) THEN
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_3 (2, SIZE(c_bounds_3, 2)))
# 446 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_3 (:, :) = c_bounds_3 (:, :) + 1
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (optimize_dist)
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_optimize_dist)) THEN
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (optimize_dist)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               optimize_dist = c_optimize_dist
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (move_data)
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_move_data)) THEN
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (move_data)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               move_data = c_move_data
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (retain_sparsity)
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_retain_sparsity)) THEN
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (retain_sparsity)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               retain_sparsity = c_retain_sparsity
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (log_verbose)
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_log_verbose)) THEN
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (log_verbose)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               log_verbose = c_log_verbose
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 460 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 461 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (pgrid_opt_1)
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (pgrid_opt_2)
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (pgrid_opt_3)
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 466 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF(PRESENT(c_pgrid_opt_1) .AND. PRESENT(c_pgrid_opt_2) .AND. PRESENT(c_pgrid_opt_3)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_1 = pgrid_opt_1, pgrid_opt_2 = pgrid_opt_2, pgrid_opt_3 = pgrid_opt_3, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_2) .AND. PRESENT(c_pgrid_opt_3) .AND. .NOT. PRESENT(c_pgrid_opt_1)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_2 = pgrid_opt_2, pgrid_opt_3 = pgrid_opt_3, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_1) .AND. PRESENT(c_pgrid_opt_3) .AND. .NOT. PRESENT(c_pgrid_opt_2)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_1 = pgrid_opt_1, pgrid_opt_3 = pgrid_opt_3, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_3) .AND. .NOT. PRESENT(c_pgrid_opt_1) .AND. .NOT. PRESENT(c_pgrid_opt_2)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_3 = pgrid_opt_3, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_1) .AND. PRESENT(c_pgrid_opt_2) .AND. .NOT. PRESENT(c_pgrid_opt_3)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_1 = pgrid_opt_1, pgrid_opt_2 = pgrid_opt_2, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_2) .AND. .NOT. PRESENT(c_pgrid_opt_1) .AND. .NOT. PRESENT(c_pgrid_opt_3)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_2 = pgrid_opt_2, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_1) .AND. .NOT. PRESENT(c_pgrid_opt_2) .AND. .NOT. PRESENT(c_pgrid_opt_3)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_1 = pgrid_opt_1, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 483 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
ENDIF
# 484 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 486 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_pgrid_opt_1)) c_pgrid_opt_1 = c_loc(pgrid_opt_1)
# 487 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 486 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_pgrid_opt_2)) c_pgrid_opt_2 = c_loc(pgrid_opt_2)
# 487 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 486 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_pgrid_opt_3)) c_pgrid_opt_3 = c_loc(pgrid_opt_3)
# 487 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 488 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 490 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 491 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(optimize_dist)) DEALLOCATE (optimize_dist)
# 492 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 491 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(move_data)) DEALLOCATE (move_data)
# 492 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 491 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(retain_sparsity)) DEALLOCATE (retain_sparsity)
# 492 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 491 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(log_verbose)) DEALLOCATE (log_verbose)
# 492 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 493 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 495 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ALLOCATED(bounds_1)) DEALLOCATE (bounds_1)
# 496 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 495 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ALLOCATED(bounds_2)) DEALLOCATE (bounds_2)
# 496 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 495 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ALLOCATED(bounds_3)) DEALLOCATE (bounds_3)
# 496 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 497 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         IF (PRESENT(c_unit_nr)) THEN
            IF (c_unit_nr .GE. 0) THEN
               flush (c_unit_nr)
            END IF
         END IF

      END SUBROUTINE

      SUBROUTINE c_dbcsr_t_contract_index_c_dp (c_alpha, c_tensor_1, &
# 507 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                       c_tensor_2, c_beta, c_tensor_3, &
                                                       c_contract_1, contract_1_size, &
                                                       c_notcontract_1, notcontract_1_size, &
                                                       c_contract_2, contract_2_size, &
                                                       c_notcontract_2, notcontract_2_size, &
                                                       c_map_1, map_1_size, c_map_2, map_2_size, &
                                                       c_bounds_1, c_bounds_2, c_bounds_3, &
                                                       c_filter_eps, c_nblks_local, c_result_index, &
                                                       result_index_size, tensor3_dim) &
         BIND(C, name="c_dbcsr_t_contract_index_c_dp")
# 517 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         COMPLEX (kind=c_double_complex), INTENT(IN), VALUE     :: c_alpha
# 519 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_1, c_tensor_2
         COMPLEX (kind=c_double_complex), INTENT(IN), VALUE     :: c_beta
# 521 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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

# 556 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (bounds_1)
# 557 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_1)) THEN
# 558 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_1 (2, SIZE(c_bounds_1, 2)))
# 559 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_1 = c_bounds_1+1
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 556 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (bounds_2)
# 557 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_2)) THEN
# 558 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_2 (2, SIZE(c_bounds_2, 2)))
# 559 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_2 = c_bounds_2+1
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 556 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (bounds_3)
# 557 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_3)) THEN
# 558 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_3 (2, SIZE(c_bounds_3, 2)))
# 559 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_3 = c_bounds_3+1
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 562 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         CALL dbcsr_t_contract_index(dbcsr_scalar(c_alpha), tensor_1, &
                                     tensor_2, dbcsr_scalar(c_beta), tensor_3, &
                                     c_contract_1 + 1, c_notcontract_1 + 1, &
                                     c_contract_2 + 1, c_notcontract_2 + 1, &
                                     c_map_1 + 1, c_map_2 + 1, &
                                     bounds_1, bounds_2, bounds_3, &
                                     c_filter_eps, c_nblks_local, result_index)

         c_result_index = result_index - 1

# 574 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(bounds_1)) DEALLOCATE (bounds_1)
# 575 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 574 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(bounds_2)) DEALLOCATE (bounds_2)
# 575 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 574 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(bounds_3)) DEALLOCATE (bounds_3)
# 575 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 576 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      END SUBROUTINE

# 384 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      SUBROUTINE c_dbcsr_t_contract_c_sp (c_alpha, c_tensor_1, c_tensor_2, c_beta, c_tensor_3, &
# 386 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                 c_contract_1, contract_1_size, &
                                                 c_notcontract_1, notcontract_1_size, &
                                                 c_contract_2, contract_2_size, &
                                                 c_notcontract_2, notcontract_2_size, &
                                                 c_map_1, map_1_size, c_map_2, map_2_size, &
                                                 c_bounds_1, c_bounds_2, c_bounds_3, &
                                                 c_optimize_dist, c_pgrid_opt_1, c_pgrid_opt_2, c_pgrid_opt_3, &
                                                 c_filter_eps, c_flop, c_move_data, c_retain_sparsity, &
                                                 c_unit_nr, c_log_verbose) &
         BIND(C, name="c_dbcsr_t_contract_c_sp")
# 396 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         COMPLEX (kind=c_float_complex), INTENT(IN), VALUE     ::  c_alpha
# 398 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_1, c_tensor_2
         COMPLEX (kind=c_float_complex), INTENT(IN), VALUE     :: c_beta
# 400 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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

# 444 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_1)) THEN
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_1 (2, SIZE(c_bounds_1, 2)))
# 446 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_1 (:, :) = c_bounds_1 (:, :) + 1
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 444 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_2)) THEN
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_2 (2, SIZE(c_bounds_2, 2)))
# 446 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_2 (:, :) = c_bounds_2 (:, :) + 1
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 444 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_3)) THEN
# 445 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_3 (2, SIZE(c_bounds_3, 2)))
# 446 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_3 (:, :) = c_bounds_3 (:, :) + 1
# 447 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 449 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 451 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (optimize_dist)
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_optimize_dist)) THEN
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (optimize_dist)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               optimize_dist = c_optimize_dist
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (move_data)
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_move_data)) THEN
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (move_data)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               move_data = c_move_data
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (retain_sparsity)
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_retain_sparsity)) THEN
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (retain_sparsity)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               retain_sparsity = c_retain_sparsity
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 452 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (log_verbose)
# 453 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_log_verbose)) THEN
# 454 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (log_verbose)
# 455 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               log_verbose = c_log_verbose
# 456 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 458 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 460 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 461 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         
# 462 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (pgrid_opt_1)
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (pgrid_opt_2)
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 464 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (pgrid_opt_3)
# 465 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 466 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF(PRESENT(c_pgrid_opt_1) .AND. PRESENT(c_pgrid_opt_2) .AND. PRESENT(c_pgrid_opt_3)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_1 = pgrid_opt_1, pgrid_opt_2 = pgrid_opt_2, pgrid_opt_3 = pgrid_opt_3, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_2) .AND. PRESENT(c_pgrid_opt_3) .AND. .NOT. PRESENT(c_pgrid_opt_1)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_2 = pgrid_opt_2, pgrid_opt_3 = pgrid_opt_3, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_1) .AND. PRESENT(c_pgrid_opt_3) .AND. .NOT. PRESENT(c_pgrid_opt_2)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_1 = pgrid_opt_1, pgrid_opt_3 = pgrid_opt_3, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_3) .AND. .NOT. PRESENT(c_pgrid_opt_1) .AND. .NOT. PRESENT(c_pgrid_opt_2)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_3 = pgrid_opt_3, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_1) .AND. PRESENT(c_pgrid_opt_2) .AND. .NOT. PRESENT(c_pgrid_opt_3)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_1 = pgrid_opt_1, pgrid_opt_2 = pgrid_opt_2, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_2) .AND. .NOT. PRESENT(c_pgrid_opt_1) .AND. .NOT. PRESENT(c_pgrid_opt_3)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_2 = pgrid_opt_2, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE IF(PRESENT(c_pgrid_opt_1) .AND. .NOT. PRESENT(c_pgrid_opt_2) .AND. .NOT. PRESENT(c_pgrid_opt_3)) THEN 
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , pgrid_opt_1 = pgrid_opt_1, &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 468 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ELSE
# 469 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL dbcsr_t_contract(alpha=dbcsr_scalar(c_alpha), tensor_1=tensor_1, &
                                  tensor_2=tensor_2, beta=dbcsr_scalar(c_beta), &
                                  tensor_3=tensor_3, contract_1=c_contract_1 + 1, &
                                  notcontract_1=c_notcontract_1 + 1, &
                                  contract_2=c_contract_2 + 1, notcontract_2=c_notcontract_2 + 1, &
                                  map_1=c_map_1 + 1, map_2=c_map_2 + 1, &
                                  bounds_1=bounds_1, bounds_2=bounds_2, bounds_3=bounds_3, &
                                  optimize_dist=optimize_dist &
                                  , &
# 479 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                  filter_eps=c_filter_eps, flop=c_flop, move_data=move_data, &
                                  retain_sparsity=retain_sparsity, unit_nr=c_unit_nr, &
                                  log_verbose=log_verbose)
# 483 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
ENDIF
# 484 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 486 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_pgrid_opt_1)) c_pgrid_opt_1 = c_loc(pgrid_opt_1)
# 487 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 486 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_pgrid_opt_2)) c_pgrid_opt_2 = c_loc(pgrid_opt_2)
# 487 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 486 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_pgrid_opt_3)) c_pgrid_opt_3 = c_loc(pgrid_opt_3)
# 487 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 488 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 490 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 491 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(optimize_dist)) DEALLOCATE (optimize_dist)
# 492 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 491 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(move_data)) DEALLOCATE (move_data)
# 492 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 491 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(retain_sparsity)) DEALLOCATE (retain_sparsity)
# 492 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 491 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(log_verbose)) DEALLOCATE (log_verbose)
# 492 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 493 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 495 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ALLOCATED(bounds_1)) DEALLOCATE (bounds_1)
# 496 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 495 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ALLOCATED(bounds_2)) DEALLOCATE (bounds_2)
# 496 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 495 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ALLOCATED(bounds_3)) DEALLOCATE (bounds_3)
# 496 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 497 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         IF (PRESENT(c_unit_nr)) THEN
            IF (c_unit_nr .GE. 0) THEN
               flush (c_unit_nr)
            END IF
         END IF

      END SUBROUTINE

      SUBROUTINE c_dbcsr_t_contract_index_c_sp (c_alpha, c_tensor_1, &
# 507 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                       c_tensor_2, c_beta, c_tensor_3, &
                                                       c_contract_1, contract_1_size, &
                                                       c_notcontract_1, notcontract_1_size, &
                                                       c_contract_2, contract_2_size, &
                                                       c_notcontract_2, notcontract_2_size, &
                                                       c_map_1, map_1_size, c_map_2, map_2_size, &
                                                       c_bounds_1, c_bounds_2, c_bounds_3, &
                                                       c_filter_eps, c_nblks_local, c_result_index, &
                                                       result_index_size, tensor3_dim) &
         BIND(C, name="c_dbcsr_t_contract_index_c_sp")
# 517 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         COMPLEX (kind=c_float_complex), INTENT(IN), VALUE     :: c_alpha
# 519 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor_1, c_tensor_2
         COMPLEX (kind=c_float_complex), INTENT(IN), VALUE     :: c_beta
# 521 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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

# 556 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (bounds_1)
# 557 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_1)) THEN
# 558 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_1 (2, SIZE(c_bounds_1, 2)))
# 559 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_1 = c_bounds_1+1
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 556 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (bounds_2)
# 557 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_2)) THEN
# 558 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_2 (2, SIZE(c_bounds_2, 2)))
# 559 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_2 = c_bounds_2+1
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 556 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            NULLIFY (bounds_3)
# 557 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_bounds_3)) THEN
# 558 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               ALLOCATE (bounds_3 (2, SIZE(c_bounds_3, 2)))
# 559 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               bounds_3 = c_bounds_3+1
# 560 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            END IF
# 562 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         CALL dbcsr_t_contract_index(dbcsr_scalar(c_alpha), tensor_1, &
                                     tensor_2, dbcsr_scalar(c_beta), tensor_3, &
                                     c_contract_1 + 1, c_notcontract_1 + 1, &
                                     c_contract_2 + 1, c_notcontract_2 + 1, &
                                     c_map_1 + 1, c_map_2 + 1, &
                                     bounds_1, bounds_2, bounds_3, &
                                     c_filter_eps, c_nblks_local, result_index)

         c_result_index = result_index - 1

# 574 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(bounds_1)) DEALLOCATE (bounds_1)
# 575 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 574 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(bounds_2)) DEALLOCATE (bounds_2)
# 575 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 574 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (ASSOCIATED(bounds_3)) DEALLOCATE (bounds_3)
# 575 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 576 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      END SUBROUTINE

# 580 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 582 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 583 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_get_2d_block_r_dp (c_tensor, tensor_dim, c_ind, c_sizes, c_block, c_found) &
# 585 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_2d_block_r_dp")
# 586 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_sizes
            REAL (kind=c_double), &
# 594 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2)), &
# 595 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               INTENT(OUT)                             :: c_block
            LOGICAL(kind=c_bool), INTENT(OUT)                  :: c_found

            TYPE(dbcsr_t_type), POINTER                        :: tensor
            LOGICAL                                            :: found

            CALL c_f_pointer(c_tensor, tensor)

            CALL dbcsr_t_get_block(tensor, c_ind + 1, c_sizes, c_block, found)
            c_found = LOGICAL(found, kind=c_bool)

         END SUBROUTINE
# 583 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_get_3d_block_r_dp (c_tensor, tensor_dim, c_ind, c_sizes, c_block, c_found) &
# 585 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_3d_block_r_dp")
# 586 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_sizes
            REAL (kind=c_double), &
# 594 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2), c_sizes(3)), &
# 595 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               INTENT(OUT)                             :: c_block
            LOGICAL(kind=c_bool), INTENT(OUT)                  :: c_found

            TYPE(dbcsr_t_type), POINTER                        :: tensor
            LOGICAL                                            :: found

            CALL c_f_pointer(c_tensor, tensor)

            CALL dbcsr_t_get_block(tensor, c_ind + 1, c_sizes, c_block, found)
            c_found = LOGICAL(found, kind=c_bool)

         END SUBROUTINE
# 583 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_get_4d_block_r_dp (c_tensor, tensor_dim, c_ind, c_sizes, c_block, c_found) &
# 585 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_4d_block_r_dp")
# 586 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_sizes
            REAL (kind=c_double), &
# 594 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2), c_sizes(3), c_sizes(4)), &
# 595 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               INTENT(OUT)                             :: c_block
            LOGICAL(kind=c_bool), INTENT(OUT)                  :: c_found

            TYPE(dbcsr_t_type), POINTER                        :: tensor
            LOGICAL                                            :: found

            CALL c_f_pointer(c_tensor, tensor)

            CALL dbcsr_t_get_block(tensor, c_ind + 1, c_sizes, c_block, found)
            c_found = LOGICAL(found, kind=c_bool)

         END SUBROUTINE
# 608 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 582 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 583 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_get_2d_block_r_sp (c_tensor, tensor_dim, c_ind, c_sizes, c_block, c_found) &
# 585 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_2d_block_r_sp")
# 586 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_sizes
            REAL (kind=c_float), &
# 594 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2)), &
# 595 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               INTENT(OUT)                             :: c_block
            LOGICAL(kind=c_bool), INTENT(OUT)                  :: c_found

            TYPE(dbcsr_t_type), POINTER                        :: tensor
            LOGICAL                                            :: found

            CALL c_f_pointer(c_tensor, tensor)

            CALL dbcsr_t_get_block(tensor, c_ind + 1, c_sizes, c_block, found)
            c_found = LOGICAL(found, kind=c_bool)

         END SUBROUTINE
# 583 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_get_3d_block_r_sp (c_tensor, tensor_dim, c_ind, c_sizes, c_block, c_found) &
# 585 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_3d_block_r_sp")
# 586 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_sizes
            REAL (kind=c_float), &
# 594 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2), c_sizes(3)), &
# 595 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               INTENT(OUT)                             :: c_block
            LOGICAL(kind=c_bool), INTENT(OUT)                  :: c_found

            TYPE(dbcsr_t_type), POINTER                        :: tensor
            LOGICAL                                            :: found

            CALL c_f_pointer(c_tensor, tensor)

            CALL dbcsr_t_get_block(tensor, c_ind + 1, c_sizes, c_block, found)
            c_found = LOGICAL(found, kind=c_bool)

         END SUBROUTINE
# 583 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_get_4d_block_r_sp (c_tensor, tensor_dim, c_ind, c_sizes, c_block, c_found) &
# 585 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_4d_block_r_sp")
# 586 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_sizes
            REAL (kind=c_float), &
# 594 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2), c_sizes(3), c_sizes(4)), &
# 595 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               INTENT(OUT)                             :: c_block
            LOGICAL(kind=c_bool), INTENT(OUT)                  :: c_found

            TYPE(dbcsr_t_type), POINTER                        :: tensor
            LOGICAL                                            :: found

            CALL c_f_pointer(c_tensor, tensor)

            CALL dbcsr_t_get_block(tensor, c_ind + 1, c_sizes, c_block, found)
            c_found = LOGICAL(found, kind=c_bool)

         END SUBROUTINE
# 608 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 582 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 583 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_get_2d_block_c_dp (c_tensor, tensor_dim, c_ind, c_sizes, c_block, c_found) &
# 585 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_2d_block_c_dp")
# 586 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_sizes
            COMPLEX (kind=c_double_complex), &
# 594 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2)), &
# 595 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               INTENT(OUT)                             :: c_block
            LOGICAL(kind=c_bool), INTENT(OUT)                  :: c_found

            TYPE(dbcsr_t_type), POINTER                        :: tensor
            LOGICAL                                            :: found

            CALL c_f_pointer(c_tensor, tensor)

            CALL dbcsr_t_get_block(tensor, c_ind + 1, c_sizes, c_block, found)
            c_found = LOGICAL(found, kind=c_bool)

         END SUBROUTINE
# 583 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_get_3d_block_c_dp (c_tensor, tensor_dim, c_ind, c_sizes, c_block, c_found) &
# 585 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_3d_block_c_dp")
# 586 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_sizes
            COMPLEX (kind=c_double_complex), &
# 594 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2), c_sizes(3)), &
# 595 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               INTENT(OUT)                             :: c_block
            LOGICAL(kind=c_bool), INTENT(OUT)                  :: c_found

            TYPE(dbcsr_t_type), POINTER                        :: tensor
            LOGICAL                                            :: found

            CALL c_f_pointer(c_tensor, tensor)

            CALL dbcsr_t_get_block(tensor, c_ind + 1, c_sizes, c_block, found)
            c_found = LOGICAL(found, kind=c_bool)

         END SUBROUTINE
# 583 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_get_4d_block_c_dp (c_tensor, tensor_dim, c_ind, c_sizes, c_block, c_found) &
# 585 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_4d_block_c_dp")
# 586 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_sizes
            COMPLEX (kind=c_double_complex), &
# 594 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2), c_sizes(3), c_sizes(4)), &
# 595 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               INTENT(OUT)                             :: c_block
            LOGICAL(kind=c_bool), INTENT(OUT)                  :: c_found

            TYPE(dbcsr_t_type), POINTER                        :: tensor
            LOGICAL                                            :: found

            CALL c_f_pointer(c_tensor, tensor)

            CALL dbcsr_t_get_block(tensor, c_ind + 1, c_sizes, c_block, found)
            c_found = LOGICAL(found, kind=c_bool)

         END SUBROUTINE
# 608 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 582 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 583 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_get_2d_block_c_sp (c_tensor, tensor_dim, c_ind, c_sizes, c_block, c_found) &
# 585 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_2d_block_c_sp")
# 586 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_sizes
            COMPLEX (kind=c_float_complex), &
# 594 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2)), &
# 595 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               INTENT(OUT)                             :: c_block
            LOGICAL(kind=c_bool), INTENT(OUT)                  :: c_found

            TYPE(dbcsr_t_type), POINTER                        :: tensor
            LOGICAL                                            :: found

            CALL c_f_pointer(c_tensor, tensor)

            CALL dbcsr_t_get_block(tensor, c_ind + 1, c_sizes, c_block, found)
            c_found = LOGICAL(found, kind=c_bool)

         END SUBROUTINE
# 583 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_get_3d_block_c_sp (c_tensor, tensor_dim, c_ind, c_sizes, c_block, c_found) &
# 585 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_3d_block_c_sp")
# 586 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_sizes
            COMPLEX (kind=c_float_complex), &
# 594 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2), c_sizes(3)), &
# 595 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               INTENT(OUT)                             :: c_block
            LOGICAL(kind=c_bool), INTENT(OUT)                  :: c_found

            TYPE(dbcsr_t_type), POINTER                        :: tensor
            LOGICAL                                            :: found

            CALL c_f_pointer(c_tensor, tensor)

            CALL dbcsr_t_get_block(tensor, c_ind + 1, c_sizes, c_block, found)
            c_found = LOGICAL(found, kind=c_bool)

         END SUBROUTINE
# 583 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_get_4d_block_c_sp (c_tensor, tensor_dim, c_ind, c_sizes, c_block, c_found) &
# 585 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_4d_block_c_sp")
# 586 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                      :: c_sizes
            COMPLEX (kind=c_float_complex), &
# 594 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2), c_sizes(3), c_sizes(4)), &
# 595 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               INTENT(OUT)                             :: c_block
            LOGICAL(kind=c_bool), INTENT(OUT)                  :: c_found

            TYPE(dbcsr_t_type), POINTER                        :: tensor
            LOGICAL                                            :: found

            CALL c_f_pointer(c_tensor, tensor)

            CALL dbcsr_t_get_block(tensor, c_ind + 1, c_sizes, c_block, found)
            c_found = LOGICAL(found, kind=c_bool)

         END SUBROUTINE
# 608 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 609 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 611 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 612 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_put_2d_block_r_dp (c_tensor, tensor_dim, c_ind, c_sizes, &
# 614 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                               c_block, c_summation, c_scale) &
            BIND(C, name="c_dbcsr_t_put_2d_block_r_dp")
# 616 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                               :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                :: c_sizes
            REAL (kind=c_double), INTENT(IN), &
# 624 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2)) :: c_block
# 625 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation
            REAL (kind=c_double), INTENT(IN), &
# 627 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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
            END IF

         END SUBROUTINE
# 612 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_put_3d_block_r_dp (c_tensor, tensor_dim, c_ind, c_sizes, &
# 614 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                               c_block, c_summation, c_scale) &
            BIND(C, name="c_dbcsr_t_put_3d_block_r_dp")
# 616 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                               :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                :: c_sizes
            REAL (kind=c_double), INTENT(IN), &
# 624 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2), c_sizes(3)) :: c_block
# 625 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation
            REAL (kind=c_double), INTENT(IN), &
# 627 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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
            END IF

         END SUBROUTINE
# 612 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_put_4d_block_r_dp (c_tensor, tensor_dim, c_ind, c_sizes, &
# 614 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                               c_block, c_summation, c_scale) &
            BIND(C, name="c_dbcsr_t_put_4d_block_r_dp")
# 616 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                               :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                :: c_sizes
            REAL (kind=c_double), INTENT(IN), &
# 624 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2), c_sizes(3), c_sizes(4)) :: c_block
# 625 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation
            REAL (kind=c_double), INTENT(IN), &
# 627 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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
            END IF

         END SUBROUTINE
# 645 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 611 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 612 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_put_2d_block_r_sp (c_tensor, tensor_dim, c_ind, c_sizes, &
# 614 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                               c_block, c_summation, c_scale) &
            BIND(C, name="c_dbcsr_t_put_2d_block_r_sp")
# 616 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                               :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                :: c_sizes
            REAL (kind=c_float), INTENT(IN), &
# 624 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2)) :: c_block
# 625 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation
            REAL (kind=c_float), INTENT(IN), &
# 627 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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
            END IF

         END SUBROUTINE
# 612 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_put_3d_block_r_sp (c_tensor, tensor_dim, c_ind, c_sizes, &
# 614 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                               c_block, c_summation, c_scale) &
            BIND(C, name="c_dbcsr_t_put_3d_block_r_sp")
# 616 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                               :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                :: c_sizes
            REAL (kind=c_float), INTENT(IN), &
# 624 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2), c_sizes(3)) :: c_block
# 625 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation
            REAL (kind=c_float), INTENT(IN), &
# 627 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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
            END IF

         END SUBROUTINE
# 612 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_put_4d_block_r_sp (c_tensor, tensor_dim, c_ind, c_sizes, &
# 614 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                               c_block, c_summation, c_scale) &
            BIND(C, name="c_dbcsr_t_put_4d_block_r_sp")
# 616 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                               :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                :: c_sizes
            REAL (kind=c_float), INTENT(IN), &
# 624 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2), c_sizes(3), c_sizes(4)) :: c_block
# 625 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation
            REAL (kind=c_float), INTENT(IN), &
# 627 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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
            END IF

         END SUBROUTINE
# 645 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 611 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 612 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_put_2d_block_c_dp (c_tensor, tensor_dim, c_ind, c_sizes, &
# 614 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                               c_block, c_summation, c_scale) &
            BIND(C, name="c_dbcsr_t_put_2d_block_c_dp")
# 616 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                               :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                :: c_sizes
            COMPLEX (kind=c_double_complex), INTENT(IN), &
# 624 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2)) :: c_block
# 625 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation
            COMPLEX (kind=c_double_complex), INTENT(IN), &
# 627 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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
            END IF

         END SUBROUTINE
# 612 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_put_3d_block_c_dp (c_tensor, tensor_dim, c_ind, c_sizes, &
# 614 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                               c_block, c_summation, c_scale) &
            BIND(C, name="c_dbcsr_t_put_3d_block_c_dp")
# 616 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                               :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                :: c_sizes
            COMPLEX (kind=c_double_complex), INTENT(IN), &
# 624 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2), c_sizes(3)) :: c_block
# 625 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation
            COMPLEX (kind=c_double_complex), INTENT(IN), &
# 627 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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
            END IF

         END SUBROUTINE
# 612 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_put_4d_block_c_dp (c_tensor, tensor_dim, c_ind, c_sizes, &
# 614 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                               c_block, c_summation, c_scale) &
            BIND(C, name="c_dbcsr_t_put_4d_block_c_dp")
# 616 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                               :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                :: c_sizes
            COMPLEX (kind=c_double_complex), INTENT(IN), &
# 624 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2), c_sizes(3), c_sizes(4)) :: c_block
# 625 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation
            COMPLEX (kind=c_double_complex), INTENT(IN), &
# 627 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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
            END IF

         END SUBROUTINE
# 645 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 611 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 612 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_put_2d_block_c_sp (c_tensor, tensor_dim, c_ind, c_sizes, &
# 614 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                               c_block, c_summation, c_scale) &
            BIND(C, name="c_dbcsr_t_put_2d_block_c_sp")
# 616 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                               :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                :: c_sizes
            COMPLEX (kind=c_float_complex), INTENT(IN), &
# 624 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2)) :: c_block
# 625 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation
            COMPLEX (kind=c_float_complex), INTENT(IN), &
# 627 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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
            END IF

         END SUBROUTINE
# 612 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_put_3d_block_c_sp (c_tensor, tensor_dim, c_ind, c_sizes, &
# 614 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                               c_block, c_summation, c_scale) &
            BIND(C, name="c_dbcsr_t_put_3d_block_c_sp")
# 616 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                               :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                :: c_sizes
            COMPLEX (kind=c_float_complex), INTENT(IN), &
# 624 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2), c_sizes(3)) :: c_block
# 625 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation
            COMPLEX (kind=c_float_complex), INTENT(IN), &
# 627 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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
            END IF

         END SUBROUTINE
# 612 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         SUBROUTINE c_dbcsr_t_put_4d_block_c_sp (c_tensor, tensor_dim, c_ind, c_sizes, &
# 614 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                               c_block, c_summation, c_scale) &
            BIND(C, name="c_dbcsr_t_put_4d_block_c_sp")
# 616 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
            INTEGER(kind=c_int), INTENT(IN), VALUE             :: tensor_dim
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                               :: c_ind
            INTEGER(kind=c_int), DIMENSION(tensor_dim), &
               INTENT(IN)                                :: c_sizes
            COMPLEX (kind=c_float_complex), INTENT(IN), &
# 624 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
               DIMENSION(c_sizes(1), c_sizes(2), c_sizes(3), c_sizes(4)) :: c_block
# 625 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_summation
            COMPLEX (kind=c_float_complex), INTENT(IN), &
# 627 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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
            END IF

         END SUBROUTINE
# 645 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 646 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 648 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 649 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         SUBROUTINE c_dbcsr_t_get_2d_block_p_r_dp (c_tensor, c_ind, c_block, c_found) &
# 650 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_2d_block_p_r_dp")
# 651 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
            INTEGER(kind=c_int), INTENT(IN) :: c_ind(2)
# 654 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            TYPE(c_ptr), INTENT(INOUT) :: c_block
            LOGICAL(kind=c_bool), INTENT(INOUT) :: c_found

            TYPE(dbcsr_t_type), POINTER :: tensor
            INTEGER, DIMENSION(2) :: ind
# 659 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL :: found, tr

            INTEGER(KIND=int_8), DIMENSION(2) :: ind_2d
            REAL(kind=real_8), DIMENSION(:, :), POINTER, CONTIGUOUS :: block_2d_ptr
# 663 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL c_f_pointer(c_tensor, tensor)
            ind = c_ind + 1

            NULLIFY (block_2d_ptr)

            ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

            CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)

            c_found = found
            c_block = c_loc(block_2d_ptr)

         END SUBROUTINE
# 649 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         SUBROUTINE c_dbcsr_t_get_3d_block_p_r_dp (c_tensor, c_ind, c_block, c_found) &
# 650 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_3d_block_p_r_dp")
# 651 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
            INTEGER(kind=c_int), INTENT(IN) :: c_ind(3)
# 654 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            TYPE(c_ptr), INTENT(INOUT) :: c_block
            LOGICAL(kind=c_bool), INTENT(INOUT) :: c_found

            TYPE(dbcsr_t_type), POINTER :: tensor
            INTEGER, DIMENSION(3) :: ind
# 659 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL :: found, tr

            INTEGER(KIND=int_8), DIMENSION(2) :: ind_2d
            REAL(kind=real_8), DIMENSION(:, :), POINTER, CONTIGUOUS :: block_2d_ptr
# 663 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL c_f_pointer(c_tensor, tensor)
            ind = c_ind + 1

            NULLIFY (block_2d_ptr)

            ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

            CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)

            c_found = found
            c_block = c_loc(block_2d_ptr)

         END SUBROUTINE
# 649 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         SUBROUTINE c_dbcsr_t_get_4d_block_p_r_dp (c_tensor, c_ind, c_block, c_found) &
# 650 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_4d_block_p_r_dp")
# 651 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
            INTEGER(kind=c_int), INTENT(IN) :: c_ind(4)
# 654 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            TYPE(c_ptr), INTENT(INOUT) :: c_block
            LOGICAL(kind=c_bool), INTENT(INOUT) :: c_found

            TYPE(dbcsr_t_type), POINTER :: tensor
            INTEGER, DIMENSION(4) :: ind
# 659 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL :: found, tr

            INTEGER(KIND=int_8), DIMENSION(2) :: ind_2d
            REAL(kind=real_8), DIMENSION(:, :), POINTER, CONTIGUOUS :: block_2d_ptr
# 663 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL c_f_pointer(c_tensor, tensor)
            ind = c_ind + 1

            NULLIFY (block_2d_ptr)

            ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

            CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)

            c_found = found
            c_block = c_loc(block_2d_ptr)

         END SUBROUTINE
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 648 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 649 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         SUBROUTINE c_dbcsr_t_get_2d_block_p_r_sp (c_tensor, c_ind, c_block, c_found) &
# 650 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_2d_block_p_r_sp")
# 651 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
            INTEGER(kind=c_int), INTENT(IN) :: c_ind(2)
# 654 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            TYPE(c_ptr), INTENT(INOUT) :: c_block
            LOGICAL(kind=c_bool), INTENT(INOUT) :: c_found

            TYPE(dbcsr_t_type), POINTER :: tensor
            INTEGER, DIMENSION(2) :: ind
# 659 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL :: found, tr

            INTEGER(KIND=int_8), DIMENSION(2) :: ind_2d
            REAL(kind=real_4), DIMENSION(:, :), POINTER, CONTIGUOUS :: block_2d_ptr
# 663 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL c_f_pointer(c_tensor, tensor)
            ind = c_ind + 1

            NULLIFY (block_2d_ptr)

            ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

            CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)

            c_found = found
            c_block = c_loc(block_2d_ptr)

         END SUBROUTINE
# 649 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         SUBROUTINE c_dbcsr_t_get_3d_block_p_r_sp (c_tensor, c_ind, c_block, c_found) &
# 650 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_3d_block_p_r_sp")
# 651 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
            INTEGER(kind=c_int), INTENT(IN) :: c_ind(3)
# 654 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            TYPE(c_ptr), INTENT(INOUT) :: c_block
            LOGICAL(kind=c_bool), INTENT(INOUT) :: c_found

            TYPE(dbcsr_t_type), POINTER :: tensor
            INTEGER, DIMENSION(3) :: ind
# 659 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL :: found, tr

            INTEGER(KIND=int_8), DIMENSION(2) :: ind_2d
            REAL(kind=real_4), DIMENSION(:, :), POINTER, CONTIGUOUS :: block_2d_ptr
# 663 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL c_f_pointer(c_tensor, tensor)
            ind = c_ind + 1

            NULLIFY (block_2d_ptr)

            ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

            CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)

            c_found = found
            c_block = c_loc(block_2d_ptr)

         END SUBROUTINE
# 649 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         SUBROUTINE c_dbcsr_t_get_4d_block_p_r_sp (c_tensor, c_ind, c_block, c_found) &
# 650 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_4d_block_p_r_sp")
# 651 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
            INTEGER(kind=c_int), INTENT(IN) :: c_ind(4)
# 654 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            TYPE(c_ptr), INTENT(INOUT) :: c_block
            LOGICAL(kind=c_bool), INTENT(INOUT) :: c_found

            TYPE(dbcsr_t_type), POINTER :: tensor
            INTEGER, DIMENSION(4) :: ind
# 659 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL :: found, tr

            INTEGER(KIND=int_8), DIMENSION(2) :: ind_2d
            REAL(kind=real_4), DIMENSION(:, :), POINTER, CONTIGUOUS :: block_2d_ptr
# 663 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL c_f_pointer(c_tensor, tensor)
            ind = c_ind + 1

            NULLIFY (block_2d_ptr)

            ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

            CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)

            c_found = found
            c_block = c_loc(block_2d_ptr)

         END SUBROUTINE
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 648 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 649 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         SUBROUTINE c_dbcsr_t_get_2d_block_p_c_dp (c_tensor, c_ind, c_block, c_found) &
# 650 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_2d_block_p_c_dp")
# 651 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
            INTEGER(kind=c_int), INTENT(IN) :: c_ind(2)
# 654 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            TYPE(c_ptr), INTENT(INOUT) :: c_block
            LOGICAL(kind=c_bool), INTENT(INOUT) :: c_found

            TYPE(dbcsr_t_type), POINTER :: tensor
            INTEGER, DIMENSION(2) :: ind
# 659 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL :: found, tr

            INTEGER(KIND=int_8), DIMENSION(2) :: ind_2d
            COMPLEX(kind=real_8), DIMENSION(:, :), POINTER, CONTIGUOUS :: block_2d_ptr
# 663 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL c_f_pointer(c_tensor, tensor)
            ind = c_ind + 1

            NULLIFY (block_2d_ptr)

            ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

            CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)

            c_found = found
            c_block = c_loc(block_2d_ptr)

         END SUBROUTINE
# 649 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         SUBROUTINE c_dbcsr_t_get_3d_block_p_c_dp (c_tensor, c_ind, c_block, c_found) &
# 650 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_3d_block_p_c_dp")
# 651 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
            INTEGER(kind=c_int), INTENT(IN) :: c_ind(3)
# 654 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            TYPE(c_ptr), INTENT(INOUT) :: c_block
            LOGICAL(kind=c_bool), INTENT(INOUT) :: c_found

            TYPE(dbcsr_t_type), POINTER :: tensor
            INTEGER, DIMENSION(3) :: ind
# 659 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL :: found, tr

            INTEGER(KIND=int_8), DIMENSION(2) :: ind_2d
            COMPLEX(kind=real_8), DIMENSION(:, :), POINTER, CONTIGUOUS :: block_2d_ptr
# 663 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL c_f_pointer(c_tensor, tensor)
            ind = c_ind + 1

            NULLIFY (block_2d_ptr)

            ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

            CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)

            c_found = found
            c_block = c_loc(block_2d_ptr)

         END SUBROUTINE
# 649 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         SUBROUTINE c_dbcsr_t_get_4d_block_p_c_dp (c_tensor, c_ind, c_block, c_found) &
# 650 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_4d_block_p_c_dp")
# 651 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
            INTEGER(kind=c_int), INTENT(IN) :: c_ind(4)
# 654 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            TYPE(c_ptr), INTENT(INOUT) :: c_block
            LOGICAL(kind=c_bool), INTENT(INOUT) :: c_found

            TYPE(dbcsr_t_type), POINTER :: tensor
            INTEGER, DIMENSION(4) :: ind
# 659 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL :: found, tr

            INTEGER(KIND=int_8), DIMENSION(2) :: ind_2d
            COMPLEX(kind=real_8), DIMENSION(:, :), POINTER, CONTIGUOUS :: block_2d_ptr
# 663 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL c_f_pointer(c_tensor, tensor)
            ind = c_ind + 1

            NULLIFY (block_2d_ptr)

            ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

            CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)

            c_found = found
            c_block = c_loc(block_2d_ptr)

         END SUBROUTINE
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 648 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 649 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         SUBROUTINE c_dbcsr_t_get_2d_block_p_c_sp (c_tensor, c_ind, c_block, c_found) &
# 650 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_2d_block_p_c_sp")
# 651 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
            INTEGER(kind=c_int), INTENT(IN) :: c_ind(2)
# 654 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            TYPE(c_ptr), INTENT(INOUT) :: c_block
            LOGICAL(kind=c_bool), INTENT(INOUT) :: c_found

            TYPE(dbcsr_t_type), POINTER :: tensor
            INTEGER, DIMENSION(2) :: ind
# 659 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL :: found, tr

            INTEGER(KIND=int_8), DIMENSION(2) :: ind_2d
            COMPLEX(kind=real_4), DIMENSION(:, :), POINTER, CONTIGUOUS :: block_2d_ptr
# 663 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL c_f_pointer(c_tensor, tensor)
            ind = c_ind + 1

            NULLIFY (block_2d_ptr)

            ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

            CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)

            c_found = found
            c_block = c_loc(block_2d_ptr)

         END SUBROUTINE
# 649 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         SUBROUTINE c_dbcsr_t_get_3d_block_p_c_sp (c_tensor, c_ind, c_block, c_found) &
# 650 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_3d_block_p_c_sp")
# 651 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
            INTEGER(kind=c_int), INTENT(IN) :: c_ind(3)
# 654 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            TYPE(c_ptr), INTENT(INOUT) :: c_block
            LOGICAL(kind=c_bool), INTENT(INOUT) :: c_found

            TYPE(dbcsr_t_type), POINTER :: tensor
            INTEGER, DIMENSION(3) :: ind
# 659 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL :: found, tr

            INTEGER(KIND=int_8), DIMENSION(2) :: ind_2d
            COMPLEX(kind=real_4), DIMENSION(:, :), POINTER, CONTIGUOUS :: block_2d_ptr
# 663 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL c_f_pointer(c_tensor, tensor)
            ind = c_ind + 1

            NULLIFY (block_2d_ptr)

            ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

            CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)

            c_found = found
            c_block = c_loc(block_2d_ptr)

         END SUBROUTINE
# 649 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         SUBROUTINE c_dbcsr_t_get_4d_block_p_c_sp (c_tensor, c_ind, c_block, c_found) &
# 650 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            BIND(C, name="c_dbcsr_t_get_4d_block_p_c_sp")
# 651 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
            INTEGER(kind=c_int), INTENT(IN) :: c_ind(4)
# 654 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            TYPE(c_ptr), INTENT(INOUT) :: c_block
            LOGICAL(kind=c_bool), INTENT(INOUT) :: c_found

            TYPE(dbcsr_t_type), POINTER :: tensor
            INTEGER, DIMENSION(4) :: ind
# 659 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            LOGICAL :: found, tr

            INTEGER(KIND=int_8), DIMENSION(2) :: ind_2d
            COMPLEX(kind=real_4), DIMENSION(:, :), POINTER, CONTIGUOUS :: block_2d_ptr
# 663 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

            CALL c_f_pointer(c_tensor, tensor)
            ind = c_ind + 1

            NULLIFY (block_2d_ptr)

            ind_2d(:) = get_2d_indices_tensor(tensor%nd_index_blk, ind)

            CALL dbcsr_tas_get_block_p(tensor%matrix_rep, ind_2d(1), ind_2d(2), block_2d_ptr, tr, found)

            c_found = found
            c_block = c_loc(block_2d_ptr)

         END SUBROUTINE
# 678 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 679 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

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

   SUBROUTINE c_dbcsr_t_reserve_blocks_index(c_tensor, nblocks, c_blk_ind_1, c_blk_ind_2, c_blk_ind_3, c_blk_ind_4) &
# 698 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
      BIND(C, name="c_dbcsr_t_reserve_blocks_index")

      TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
      INTEGER(kind=c_int), INTENT(in), value             :: nblocks
      INTEGER(kind=c_int), INTENT(IN), OPTIONAL          :: c_blk_ind_1(nblocks), c_blk_ind_2(nblocks), c_blk_ind_3(nblocks),&
# 702 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
          & c_blk_ind_4(nblocks)
# 703 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
      INTEGER, DIMENSION(:), POINTER                     :: blk_ind_1, blk_ind_2, blk_ind_3, blk_ind_4
# 704 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      TYPE(dbcsr_t_type), POINTER                        :: tensor

# 708 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         NULLIFY (blk_ind_1)
# 709 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_blk_ind_1)) THEN
# 710 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ALLOCATE (blk_ind_1 (nblocks))
# 711 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            blk_ind_1 = c_blk_ind_1+1
# 712 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         END IF
# 708 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         NULLIFY (blk_ind_2)
# 709 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_blk_ind_2)) THEN
# 710 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ALLOCATE (blk_ind_2 (nblocks))
# 711 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            blk_ind_2 = c_blk_ind_2+1
# 712 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         END IF
# 708 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         NULLIFY (blk_ind_3)
# 709 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_blk_ind_3)) THEN
# 710 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ALLOCATE (blk_ind_3 (nblocks))
# 711 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            blk_ind_3 = c_blk_ind_3+1
# 712 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         END IF
# 708 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         NULLIFY (blk_ind_4)
# 709 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_blk_ind_4)) THEN
# 710 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            ALLOCATE (blk_ind_4 (nblocks))
# 711 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            blk_ind_4 = c_blk_ind_4+1
# 712 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         END IF
# 714 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      CALL c_f_pointer(c_tensor, tensor)

      CALL dbcsr_t_reserve_blocks(tensor, blk_ind_1, blk_ind_2, blk_ind_3, blk_ind_4)
# 718 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 720 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (ASSOCIATED(blk_ind_1)) DEALLOCATE (blk_ind_1)
# 721 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 720 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (ASSOCIATED(blk_ind_2)) DEALLOCATE (blk_ind_2)
# 721 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 720 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (ASSOCIATED(blk_ind_3)) DEALLOCATE (blk_ind_3)
# 721 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 720 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (ASSOCIATED(blk_ind_4)) DEALLOCATE (blk_ind_4)
# 721 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 722 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

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
      END IF

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
      END IF

      c_tensor_out = c_loc(tensor_out)

   END SUBROUTINE

# 856 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      SUBROUTINE c_dbcsr_t_filter_r_dp (c_tensor, c_eps, c_method, c_use_absolute) &
# 858 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         BIND(C, name="c_dbcsr_t_filter_r_dp")
# 859 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
         REAL (kind=c_double), INTENT(IN), VALUE     :: c_eps
# 862 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(IN), OPTIONAL          :: c_method
         LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_use_absolute

         TYPE(dbcsr_t_type), POINTER                        :: tensor
         LOGICAL                                            :: use_absolute

         CALL c_f_pointer(c_tensor, tensor)

         IF (PRESENT(c_use_absolute)) THEN
            use_absolute = c_use_absolute
            CALL dbcsr_t_filter(tensor, c_eps, c_method, use_absolute)
         ELSE
            CALL dbcsr_t_filter(tensor, c_eps, c_method)
         END IF

      END SUBROUTINE
# 856 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      SUBROUTINE c_dbcsr_t_filter_r_sp (c_tensor, c_eps, c_method, c_use_absolute) &
# 858 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         BIND(C, name="c_dbcsr_t_filter_r_sp")
# 859 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
         REAL (kind=c_float), INTENT(IN), VALUE     :: c_eps
# 862 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(IN), OPTIONAL          :: c_method
         LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_use_absolute

         TYPE(dbcsr_t_type), POINTER                        :: tensor
         LOGICAL                                            :: use_absolute

         CALL c_f_pointer(c_tensor, tensor)

         IF (PRESENT(c_use_absolute)) THEN
            use_absolute = c_use_absolute
            CALL dbcsr_t_filter(tensor, c_eps, c_method, use_absolute)
         ELSE
            CALL dbcsr_t_filter(tensor, c_eps, c_method)
         END IF

      END SUBROUTINE
# 856 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      SUBROUTINE c_dbcsr_t_filter_c_dp (c_tensor, c_eps, c_method, c_use_absolute) &
# 858 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         BIND(C, name="c_dbcsr_t_filter_c_dp")
# 859 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
         COMPLEX (kind=c_double_complex), INTENT(IN), VALUE     :: c_eps
# 862 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(IN), OPTIONAL          :: c_method
         LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_use_absolute

         TYPE(dbcsr_t_type), POINTER                        :: tensor
         LOGICAL                                            :: use_absolute

         CALL c_f_pointer(c_tensor, tensor)

         IF (PRESENT(c_use_absolute)) THEN
            use_absolute = c_use_absolute
            CALL dbcsr_t_filter(tensor, c_eps, c_method, use_absolute)
         ELSE
            CALL dbcsr_t_filter(tensor, c_eps, c_method)
         END IF

      END SUBROUTINE
# 856 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      SUBROUTINE c_dbcsr_t_filter_c_sp (c_tensor, c_eps, c_method, c_use_absolute) &
# 858 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         BIND(C, name="c_dbcsr_t_filter_c_sp")
# 859 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
         COMPLEX (kind=c_float_complex), INTENT(IN), VALUE     :: c_eps
# 862 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(IN), OPTIONAL          :: c_method
         LOGICAL(kind=c_bool), INTENT(IN), OPTIONAL         :: c_use_absolute

         TYPE(dbcsr_t_type), POINTER                        :: tensor
         LOGICAL                                            :: use_absolute

         CALL c_f_pointer(c_tensor, tensor)

         IF (PRESENT(c_use_absolute)) THEN
            use_absolute = c_use_absolute
            CALL dbcsr_t_filter(tensor, c_eps, c_method, use_absolute)
         ELSE
            CALL dbcsr_t_filter(tensor, c_eps, c_method)
         END IF

      END SUBROUTINE
# 879 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

# 881 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      SUBROUTINE c_dbcsr_t_set_r_dp (c_tensor, c_alpha) &
# 883 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         BIND(C, name="c_dbcsr_t_set_r_dp")
# 884 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
         REAL (kind=c_double), INTENT(IN), VALUE     :: c_alpha
# 887 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(dbcsr_t_type), POINTER                        :: tensor

         CALL c_f_pointer(c_tensor, tensor)

         CALL dbcsr_t_set(tensor, c_alpha)

      END SUBROUTINE
# 881 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      SUBROUTINE c_dbcsr_t_set_r_sp (c_tensor, c_alpha) &
# 883 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         BIND(C, name="c_dbcsr_t_set_r_sp")
# 884 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
         REAL (kind=c_float), INTENT(IN), VALUE     :: c_alpha
# 887 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(dbcsr_t_type), POINTER                        :: tensor

         CALL c_f_pointer(c_tensor, tensor)

         CALL dbcsr_t_set(tensor, c_alpha)

      END SUBROUTINE
# 881 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      SUBROUTINE c_dbcsr_t_set_c_dp (c_tensor, c_alpha) &
# 883 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         BIND(C, name="c_dbcsr_t_set_c_dp")
# 884 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
         COMPLEX (kind=c_double_complex), INTENT(IN), VALUE     :: c_alpha
# 887 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(dbcsr_t_type), POINTER                        :: tensor

         CALL c_f_pointer(c_tensor, tensor)

         CALL dbcsr_t_set(tensor, c_alpha)

      END SUBROUTINE
# 881 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      SUBROUTINE c_dbcsr_t_set_c_sp (c_tensor, c_alpha) &
# 883 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         BIND(C, name="c_dbcsr_t_set_c_sp")
# 884 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
         COMPLEX (kind=c_float_complex), INTENT(IN), VALUE     :: c_alpha
# 887 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(dbcsr_t_type), POINTER                        :: tensor

         CALL c_f_pointer(c_tensor, tensor)

         CALL dbcsr_t_set(tensor, c_alpha)

      END SUBROUTINE
# 896 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

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
      END IF

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
      END IF

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
      END IF

      NULLIFY (bounds)
      IF (PRESENT(c_bounds)) THEN
         ALLOCATE (bounds(2, tensor_dim))
         bounds = c_bounds + 1
      END IF

      NULLIFY (summation)
      IF (PRESENT(c_summation)) THEN
         ALLOCATE (summation)
         summation = c_summation
      END IF

      NULLIFY (move_data)
      IF (PRESENT(c_move_data)) THEN
         ALLOCATE (move_data)
         move_data = c_move_data
      END IF

      CALL dbcsr_t_copy(tensor_in, tensor_out, order, summation, bounds, move_data, c_unit_nr)

# 1008 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1009 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (ASSOCIATED(order)) DEALLOCATE (order)
# 1010 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1009 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (ASSOCIATED(bounds)) DEALLOCATE (bounds)
# 1010 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1009 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (ASSOCIATED(summation)) DEALLOCATE (summation)
# 1010 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1009 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (ASSOCIATED(move_data)) DEALLOCATE (move_data)
# 1010 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1011 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

   END SUBROUTINE

# 1015 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      SUBROUTINE c_dbcsr_t_scale_r_dp (c_tensor, c_alpha) &
# 1017 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         BIND(C, name="c_dbcsr_t_scale_r_dp")
# 1018 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
         REAL (kind=c_double), INTENT(IN), VALUE     :: c_alpha
# 1021 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(dbcsr_t_type), POINTER                        :: tensor

         CALL c_f_pointer(c_tensor, tensor)

         CALL dbcsr_t_scale(tensor, dbcsr_scalar(c_alpha))

      END SUBROUTINE

# 1015 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      SUBROUTINE c_dbcsr_t_scale_r_sp (c_tensor, c_alpha) &
# 1017 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         BIND(C, name="c_dbcsr_t_scale_r_sp")
# 1018 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
         REAL (kind=c_float), INTENT(IN), VALUE     :: c_alpha
# 1021 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(dbcsr_t_type), POINTER                        :: tensor

         CALL c_f_pointer(c_tensor, tensor)

         CALL dbcsr_t_scale(tensor, dbcsr_scalar(c_alpha))

      END SUBROUTINE

# 1015 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      SUBROUTINE c_dbcsr_t_scale_c_dp (c_tensor, c_alpha) &
# 1017 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         BIND(C, name="c_dbcsr_t_scale_c_dp")
# 1018 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
         COMPLEX (kind=c_double_complex), INTENT(IN), VALUE     :: c_alpha
# 1021 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(dbcsr_t_type), POINTER                        :: tensor

         CALL c_f_pointer(c_tensor, tensor)

         CALL dbcsr_t_scale(tensor, dbcsr_scalar(c_alpha))

      END SUBROUTINE

# 1015 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      SUBROUTINE c_dbcsr_t_scale_c_sp (c_tensor, c_alpha) &
# 1017 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         BIND(C, name="c_dbcsr_t_scale_c_sp")
# 1018 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(c_ptr), INTENT(IN), VALUE                     :: c_tensor
         COMPLEX (kind=c_float_complex), INTENT(IN), VALUE     :: c_alpha
# 1021 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(dbcsr_t_type), POINTER                        :: tensor

         CALL c_f_pointer(c_tensor, tensor)

         CALL dbcsr_t_scale(tensor, dbcsr_scalar(c_alpha))

      END SUBROUTINE

# 1031 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

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
                                 nblks_local_1, nblks_local_2, nblks_local_3, nblks_local_4, &
# 1050 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                 nblks_total_1, nblks_total_2, nblks_total_3, nblks_total_4, &
# 1051 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                 c_blks_local_1, c_blks_local_2, c_blks_local_3, c_blks_local_4, &
# 1052 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                 c_proc_dist_1, c_proc_dist_2, c_proc_dist_3, c_proc_dist_4, &
# 1053 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                 c_blk_size_1, c_blk_size_2, c_blk_size_3, c_blk_size_4, &
# 1054 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                 c_blk_offset_1, c_blk_offset_2, c_blk_offset_3, c_blk_offset_4, &
# 1055 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
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

# 1076 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(IN), VALUE :: nblks_local_1
# 1077 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(IN), VALUE :: nblks_total_1
# 1078 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
            DIMENSION(nblks_local_1) :: c_blks_local_1
# 1080 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
            DIMENSION(nblks_total_1) :: c_proc_dist_1
# 1082 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
            DIMENSION(nblks_total_1) :: c_blk_size_1
# 1084 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
            DIMENSION(nblks_total_1) :: c_blk_offset_1
# 1086 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1076 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(IN), VALUE :: nblks_local_2
# 1077 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(IN), VALUE :: nblks_total_2
# 1078 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
            DIMENSION(nblks_local_2) :: c_blks_local_2
# 1080 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
            DIMENSION(nblks_total_2) :: c_proc_dist_2
# 1082 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
            DIMENSION(nblks_total_2) :: c_blk_size_2
# 1084 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
            DIMENSION(nblks_total_2) :: c_blk_offset_2
# 1086 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1076 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(IN), VALUE :: nblks_local_3
# 1077 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(IN), VALUE :: nblks_total_3
# 1078 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
            DIMENSION(nblks_local_3) :: c_blks_local_3
# 1080 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
            DIMENSION(nblks_total_3) :: c_proc_dist_3
# 1082 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
            DIMENSION(nblks_total_3) :: c_blk_size_3
# 1084 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
            DIMENSION(nblks_total_3) :: c_blk_offset_3
# 1086 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1076 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(IN), VALUE :: nblks_local_4
# 1077 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(IN), VALUE :: nblks_total_4
# 1078 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
            DIMENSION(nblks_local_4) :: c_blks_local_4
# 1080 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
            DIMENSION(nblks_total_4) :: c_proc_dist_4
# 1082 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
            DIMENSION(nblks_total_4) :: c_blk_size_4
# 1084 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(OUT), OPTIONAL, &
            DIMENSION(nblks_total_4) :: c_blk_offset_4
# 1086 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1087 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

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
                            c_blks_local_1, c_blks_local_2, c_blks_local_3, c_blks_local_4, &
# 1107 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                            c_proc_dist_1, c_proc_dist_2, c_proc_dist_3, c_proc_dist_4, &
# 1108 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                            c_blk_size_1, c_blk_size_2, c_blk_size_3, c_blk_size_4, &
# 1109 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                            c_blk_offset_1, c_blk_offset_2, c_blk_offset_3, c_blk_offset_4, &
# 1110 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                            distribution, name, &
                            c_data_type)

# 1114 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1115 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1116 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_blks_local_1)) c_blks_local_1 = c_blks_local_1-1
# 1117 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1116 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_blk_offset_1)) c_blk_offset_1 = c_blk_offset_1-1
# 1117 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1118 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1115 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1116 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_blks_local_2)) c_blks_local_2 = c_blks_local_2-1
# 1117 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1116 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_blk_offset_2)) c_blk_offset_2 = c_blk_offset_2-1
# 1117 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1118 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1115 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1116 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_blks_local_3)) c_blks_local_3 = c_blks_local_3-1
# 1117 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1116 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_blk_offset_3)) c_blk_offset_3 = c_blk_offset_3-1
# 1117 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1118 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1115 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1116 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_blks_local_4)) c_blks_local_4 = c_blks_local_4-1
# 1117 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1116 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
            IF (PRESENT(c_blk_offset_4)) c_blk_offset_4 = c_blk_offset_4-1
# 1117 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1118 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1119 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

      IF (PRESENT(c_name)) THEN
         name = TRIM(name)//char(0)
         c_name = c_loc(name)
      END IF

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
      END IF

# 1182 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1183 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_dims_nd)) c_dims_nd = c_dims_nd-1
# 1184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1183 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_dims1_2d)) c_dims1_2d = c_dims1_2d-1
# 1184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1183 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_dims2_2d)) c_dims2_2d = c_dims2_2d-1
# 1184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1183 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_map1_2d)) c_map1_2d = c_map1_2d-1
# 1184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1183 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_map2_2d)) c_map2_2d = c_map2_2d-1
# 1184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1183 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         IF (PRESENT(c_map_nd)) c_map_nd = c_map_nd-1
# 1184 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
# 1185 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

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

# 1331 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
      SUBROUTINE c_dbcsr_t_get_data_r_dp (c_tensor, c_data, c_data_size, &
# 1332 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                 c_select_data_type, c_lb, c_ub) BIND(C, name="c_dbcsr_t_get_data_r_dp")
# 1333 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
         TYPE(c_ptr), INTENT(INOUT) :: c_data
         INTEGER(kind=c_long_long), INTENT(INOUT) :: c_data_size
         REAL (kind=c_double), INTENT(IN), VALUE :: c_select_data_type
# 1338 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_lb, c_ub

         TYPE(dbcsr_t_type), POINTER :: tensor
         REAL (kind=c_double), DIMENSION(:), POINTER :: DATA
# 1342 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER, POINTER :: lb, ub

         CALL c_f_pointer(c_tensor, tensor)

         NULLIFY (lb)
         IF (PRESENT(c_lb)) THEN
            ALLOCATE (lb)
            lb = c_lb + 1
         END IF

         NULLIFY (ub)
         IF (PRESENT(c_ub)) THEN
            ALLOCATE (ub)
            ub = ub + 1
         END IF

         DATA => dbcsr_get_data_p(tensor%matrix_rep%matrix%data_area, &
                                  c_select_data_type, lb, ub)

         c_data = c_loc(DATA)
         c_data_size = SIZE(DATA)

         IF (ASSOCIATED(lb)) DEALLOCATE (lb)
         IF (ASSOCIATED(ub)) DEALLOCATE (ub)

      END SUBROUTINE
# 1331 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
      SUBROUTINE c_dbcsr_t_get_data_r_sp (c_tensor, c_data, c_data_size, &
# 1332 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                 c_select_data_type, c_lb, c_ub) BIND(C, name="c_dbcsr_t_get_data_r_sp")
# 1333 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
         TYPE(c_ptr), INTENT(INOUT) :: c_data
         INTEGER(kind=c_long_long), INTENT(INOUT) :: c_data_size
         REAL (kind=c_float), INTENT(IN), VALUE :: c_select_data_type
# 1338 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_lb, c_ub

         TYPE(dbcsr_t_type), POINTER :: tensor
         REAL (kind=c_float), DIMENSION(:), POINTER :: DATA
# 1342 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER, POINTER :: lb, ub

         CALL c_f_pointer(c_tensor, tensor)

         NULLIFY (lb)
         IF (PRESENT(c_lb)) THEN
            ALLOCATE (lb)
            lb = c_lb + 1
         END IF

         NULLIFY (ub)
         IF (PRESENT(c_ub)) THEN
            ALLOCATE (ub)
            ub = ub + 1
         END IF

         DATA => dbcsr_get_data_p(tensor%matrix_rep%matrix%data_area, &
                                  c_select_data_type, lb, ub)

         c_data = c_loc(DATA)
         c_data_size = SIZE(DATA)

         IF (ASSOCIATED(lb)) DEALLOCATE (lb)
         IF (ASSOCIATED(ub)) DEALLOCATE (ub)

      END SUBROUTINE
# 1331 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
      SUBROUTINE c_dbcsr_t_get_data_c_dp (c_tensor, c_data, c_data_size, &
# 1332 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                 c_select_data_type, c_lb, c_ub) BIND(C, name="c_dbcsr_t_get_data_c_dp")
# 1333 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
         TYPE(c_ptr), INTENT(INOUT) :: c_data
         INTEGER(kind=c_long_long), INTENT(INOUT) :: c_data_size
         COMPLEX (kind=c_double_complex), INTENT(IN), VALUE :: c_select_data_type
# 1338 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_lb, c_ub

         TYPE(dbcsr_t_type), POINTER :: tensor
         COMPLEX (kind=c_double_complex), DIMENSION(:), POINTER :: DATA
# 1342 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER, POINTER :: lb, ub

         CALL c_f_pointer(c_tensor, tensor)

         NULLIFY (lb)
         IF (PRESENT(c_lb)) THEN
            ALLOCATE (lb)
            lb = c_lb + 1
         END IF

         NULLIFY (ub)
         IF (PRESENT(c_ub)) THEN
            ALLOCATE (ub)
            ub = ub + 1
         END IF

         DATA => dbcsr_get_data_p(tensor%matrix_rep%matrix%data_area, &
                                  c_select_data_type, lb, ub)

         c_data = c_loc(DATA)
         c_data_size = SIZE(DATA)

         IF (ASSOCIATED(lb)) DEALLOCATE (lb)
         IF (ASSOCIATED(ub)) DEALLOCATE (ub)

      END SUBROUTINE
# 1331 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
      SUBROUTINE c_dbcsr_t_get_data_c_sp (c_tensor, c_data, c_data_size, &
# 1332 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
                                                 c_select_data_type, c_lb, c_ub) BIND(C, name="c_dbcsr_t_get_data_c_sp")
# 1333 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

         TYPE(c_ptr), INTENT(IN), VALUE :: c_tensor
         TYPE(c_ptr), INTENT(INOUT) :: c_data
         INTEGER(kind=c_long_long), INTENT(INOUT) :: c_data_size
         COMPLEX (kind=c_float_complex), INTENT(IN), VALUE :: c_select_data_type
# 1338 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER(kind=c_int), INTENT(IN), OPTIONAL :: c_lb, c_ub

         TYPE(dbcsr_t_type), POINTER :: tensor
         COMPLEX (kind=c_float_complex), DIMENSION(:), POINTER :: DATA
# 1342 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"
         INTEGER, POINTER :: lb, ub

         CALL c_f_pointer(c_tensor, tensor)

         NULLIFY (lb)
         IF (PRESENT(c_lb)) THEN
            ALLOCATE (lb)
            lb = c_lb + 1
         END IF

         NULLIFY (ub)
         IF (PRESENT(c_ub)) THEN
            ALLOCATE (ub)
            ub = ub + 1
         END IF

         DATA => dbcsr_get_data_p(tensor%matrix_rep%matrix%data_area, &
                                  c_select_data_type, lb, ub)

         c_data = c_loc(DATA)
         c_data_size = SIZE(DATA)

         IF (ASSOCIATED(lb)) DEALLOCATE (lb)
         IF (ASSOCIATED(ub)) DEALLOCATE (ub)

      END SUBROUTINE
# 1369 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_api_c.F"

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
      END IF

   END SUBROUTINE

END MODULE
