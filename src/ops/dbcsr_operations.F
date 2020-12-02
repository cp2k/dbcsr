!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_operations
   !! Higher-level operations on DBCSR matrices.
   USE dbcsr_array_types, ONLY: array_data, &
                                array_equality, &
                                array_get
   USE dbcsr_blas_operations, ONLY: set_larnv_seed
   USE dbcsr_block_access, ONLY: dbcsr_get_block_p, &
                                 dbcsr_put_block, &
                                 dbcsr_remove_block, &
                                 dbcsr_reserve_blocks, &
                                 dbcsr_reserve_block2d
   USE dbcsr_block_operations, ONLY: dbcsr_block_conjg, &
                                     dbcsr_block_partial_copy, &
                                     dbcsr_block_real_neg, &
                                     dbcsr_block_scale, &
                                     dbcsr_data_clear
   USE dbcsr_config, ONLY: default_resize_factor
   USE dbcsr_data_methods, ONLY: &
      dbcsr_data_clear_pointer, dbcsr_data_ensure_size, dbcsr_data_get_size, &
      dbcsr_data_get_size_referenced, dbcsr_data_get_type, dbcsr_data_init, dbcsr_data_new, &
      dbcsr_data_release, dbcsr_data_set_pointer, dbcsr_get_data, dbcsr_scalar, &
      dbcsr_scalar_are_equal, dbcsr_scalar_fill_all, dbcsr_scalar_get_type, dbcsr_scalar_one, &
      dbcsr_scalar_zero, dbcsr_type_1d_to_2d, dbcsr_get_data_p_d, dbcsr_get_data_p_s, dbcsr_scalar_set_type
   USE dbcsr_data_operations, ONLY: dbcsr_data_convert, &
                                    dbcsr_data_copyall, &
                                    dbcsr_switch_data_area
   USE dbcsr_dist_methods, ONLY: dbcsr_distribution_col_dist, &
                                 dbcsr_distribution_local_cols, &
                                 dbcsr_distribution_local_rows, &
                                 dbcsr_distribution_mp, &
                                 dbcsr_distribution_row_dist, &
                                 dbcsr_distribution_get
   USE dbcsr_dist_operations, ONLY: checker_square_proc, &
                                    checker_tr, &
                                    dbcsr_find_column, &
                                    dbcsr_get_stored_coordinates, &
                                    dbcsr_get_stored_block_info
   USE dbcsr_index_operations, ONLY: dbcsr_index_checksum, &
                                     dbcsr_index_compact, &
                                     dbcsr_repoint_index
   USE dbcsr_iterator_operations, ONLY: dbcsr_iterator_blocks_left, &
                                        dbcsr_iterator_next_block, &
                                        dbcsr_iterator_start, &
                                        dbcsr_iterator_stop
   USE dbcsr_methods, ONLY: &
      dbcsr_col_block_offsets, dbcsr_distribution, dbcsr_get_data_size, dbcsr_get_data_type, &
      dbcsr_get_index_memory_type, dbcsr_get_matrix_type, dbcsr_get_num_blocks, &
      dbcsr_get_replication_type, dbcsr_has_symmetry, dbcsr_max_col_size, dbcsr_max_row_size, &
      dbcsr_nblkcols_total, dbcsr_nblkrows_total, dbcsr_nfullcols_total, dbcsr_nfullrows_total, &
      dbcsr_row_block_offsets, dbcsr_valid_index, dbcsr_get_nze, dbcsr_nfullcols_local, &
      dbcsr_nfullrows_local, dbcsr_get_num_blocks, dbcsr_release, dbcsr_wm_use_mutable
   USE dbcsr_mp_methods, ONLY: dbcsr_mp_group, &
                               dbcsr_mp_mynode, &
                               dbcsr_mp_mypcol, &
                               dbcsr_mp_myprow, &
                               dbcsr_mp_numnodes, &
                               dbcsr_mp_pgrid
   USE dbcsr_ptr_util, ONLY: ensure_array_size, &
                             pointer_view
   USE dbcsr_toollib, ONLY: swap
   USE dbcsr_types, ONLY: &
      dbcsr_data_obj, dbcsr_distribution_obj, dbcsr_filter_frobenius, dbcsr_func_artanh, &
      dbcsr_func_asin, dbcsr_func_cos, dbcsr_func_ddsin, dbcsr_func_ddtanh, dbcsr_func_dsin, &
      dbcsr_func_dtanh, dbcsr_func_inverse, dbcsr_func_inverse_special, dbcsr_func_sin, &
      dbcsr_func_spread_from_zero, dbcsr_func_tanh, dbcsr_func_truncate, dbcsr_iterator, &
      dbcsr_mp_obj, dbcsr_norm_column, dbcsr_norm_frobenius, dbcsr_norm_gershgorin, &
      dbcsr_norm_maxabsnorm, dbcsr_repl_full, dbcsr_repl_none, dbcsr_scalar_type, dbcsr_type, &
      dbcsr_type_antihermitian, dbcsr_type_antisymmetric, dbcsr_type_complex_4, &
      dbcsr_type_complex_8, dbcsr_type_hermitian, dbcsr_type_no_symmetry, dbcsr_type_real_4, &
      dbcsr_type_real_8, dbcsr_type_symmetric
   USE dbcsr_dist_util, ONLY: find_block_of_element
   USE dbcsr_work_operations, ONLY: dbcsr_create, &
                                    dbcsr_finalize, &
                                    dbcsr_work_create, &
                                    add_work_coordinate
   USE dbcsr_kinds, ONLY: dp, &
                          int_4, &
                          int_8, &
                          real_4, &
                          real_8, &
                          sp
   USE dbcsr_mpiwrap, ONLY: mp_allgather, &
                            mp_max, &
                            mp_sum
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_operations'

   ! prettify protection
   CHARACTER, PARAMETER :: xa = dbcsr_type_hermitian, xb = dbcsr_type_antihermitian, &
                           xc = dbcsr_type_no_symmetry

   PUBLIC :: dbcsr_trace, dbcsr_dot, dbcsr_add_on_diag, &
             dbcsr_set, dbcsr_scale, dbcsr_add, dbcsr_copy, &
             dbcsr_copy_into_existing, &
             dbcsr_get_diag, dbcsr_set_diag, &
             dbcsr_get_block_diag, dbcsr_hadamard_product, &
             dbcsr_filter, dbcsr_scale_by_vector, &
             dbcsr_function_of_elements, &
             dbcsr_triu, &
             dbcsr_init_random
   PUBLIC :: dbcsr_sum_replicated
   PUBLIC :: dbcsr_norm_scalar, dbcsr_norm_r8_vec, dbcsr_conjg, &
             dbcsr_gershgorin_norm, dbcsr_maxabs, dbcsr_frobenius_norm

   PUBLIC :: dbcsr_crop_matrix
   PUBLIC :: dbcsr_get_info, dbcsr_may_be_dense, dbcsr_get_occupation
   PUBLIC :: dbcsr_clear, dbcsr_add_block_node, dbcsr_conform_scalar
   PUBLIC :: dbcsr_zero

! The interfaces for the generic routines found in the generated
! generic files.

   INTERFACE dbcsr_conform_scalar
      MODULE PROCEDURE make_conformant_scalar_d, make_conformant_scalar_s
      MODULE PROCEDURE make_conformant_scalar_c, make_conformant_scalar_z
   END INTERFACE

   INTERFACE dbcsr_trace
      MODULE PROCEDURE dbcsr_trace_s, dbcsr_trace_sd, &
         dbcsr_trace_c, dbcsr_trace_z
   END INTERFACE

   INTERFACE dbcsr_dot
      MODULE PROCEDURE dbcsr_dot_s, dbcsr_dot_sd, &
         dbcsr_dot_c, dbcsr_dot_z
   END INTERFACE

   INTERFACE dbcsr_scale
      MODULE PROCEDURE dbcsr_scale_anytype
      MODULE PROCEDURE dbcsr_scale_s, dbcsr_scale_d, dbcsr_scale_c, dbcsr_scale_z
   END INTERFACE

   INTERFACE dbcsr_scale_by_vector
      MODULE PROCEDURE dbcsr_scale_by_vector_anytype
      MODULE PROCEDURE dbcsr_scale_by_vector_s, dbcsr_scale_by_vector_d
      MODULE PROCEDURE dbcsr_scale_by_vector_c, dbcsr_scale_by_vector_z
   END INTERFACE

   INTERFACE dbcsr_set
      MODULE PROCEDURE dbcsr_set_s, dbcsr_set_d, dbcsr_set_c, dbcsr_set_z
   END INTERFACE

   INTERFACE dbcsr_add
      MODULE PROCEDURE dbcsr_add_anytype
      MODULE PROCEDURE dbcsr_add_s, dbcsr_add_d, dbcsr_add_c, dbcsr_add_z
   END INTERFACE

   INTERFACE dbcsr_add_on_diag
      MODULE PROCEDURE dbcsr_add_on_diag_s, dbcsr_add_on_diag_ds
      MODULE PROCEDURE dbcsr_add_on_diag_c, dbcsr_add_on_diag_z
   END INTERFACE

   INTERFACE dbcsr_filter
      MODULE PROCEDURE dbcsr_filter_anytype
      MODULE PROCEDURE dbcsr_filter_s, dbcsr_filter_d, &
         dbcsr_filter_c, dbcsr_filter_z
   END INTERFACE

   INTERFACE dbcsr_get_diag
      MODULE PROCEDURE dbcsr_get_diag_s, dbcsr_get_diag_d, dbcsr_get_diag_c, dbcsr_get_diag_z
   END INTERFACE

   INTERFACE dbcsr_set_diag
      MODULE PROCEDURE dbcsr_set_diag_s, dbcsr_set_diag_d, dbcsr_set_diag_c, dbcsr_set_diag_z
   END INTERFACE

   LOGICAL, PARAMETER :: debug_mod = .FALSE.
   LOGICAL, PARAMETER :: careful_mod = .FALSE.

   INTEGER, PARAMETER, PRIVATE :: rpslot_owner = 1
   INTEGER, PARAMETER, PRIVATE :: rpslot_addblks = 2
   INTEGER, PARAMETER, PRIVATE :: rpslot_addoffset = 3
   INTEGER, PARAMETER, PRIVATE :: rpslot_oldblks = 4
   INTEGER, PARAMETER, PRIVATE :: rpslot_oldoffset = 5
   INTEGER, PARAMETER, PRIVATE :: rpslot_totaloffset = 6
   INTEGER, PARAMETER, PRIVATE :: rpnslots = 6

CONTAINS
#:include '../data/dbcsr.fypp'
#:for n, nametype1, base1, prec1, kind1, type1, dkind1  in inst_params_float
   FUNCTION make_conformant_scalar_${nametype1}$ (scalar, matrix) RESULT(encapsulated)
      !! Encapsulates a given scalar value and makes it conform with the
      !! type of the matrix.

      ${type1}$, INTENT(IN)                      :: scalar
      TYPE(dbcsr_type), INTENT(IN)             :: matrix

      TYPE(dbcsr_scalar_type)                  :: encapsulated
      INTEGER                                  :: data_type, scalar_data_type

      encapsulated = dbcsr_scalar(scalar)
      CALL dbcsr_scalar_fill_all(encapsulated)
      data_type = dbcsr_get_data_type(matrix)
      scalar_data_type = dbcsr_scalar_get_type(encapsulated)
      IF (scalar_data_type .EQ. dbcsr_type_complex_4 .OR. &
          scalar_data_type .EQ. dbcsr_type_complex_8) THEN
         IF (data_type .NE. dbcsr_type_complex_4 .AND. data_type .NE. dbcsr_type_complex_8) &
            DBCSR_ABORT("Can not conform a complex to a real number")
      END IF
      CALL dbcsr_scalar_set_type(encapsulated, data_type)
   END FUNCTION make_conformant_scalar_${nametype1}$
#:endfor

   SUBROUTINE dbcsr_add_block_node(matrix, block_row, block_col, block)
      !! Emulation of sparse_matrix_types/add_block_node mapped
      !! to add_real_matrix_block.... should not be used any longer
      !! It adds a block to the dbcsr matrix and returns a rank-2 pointer to the
      !! block. Currently it only and always uses the mutable data.

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
         !! DBCSR matrix
      INTEGER, INTENT(IN)                                :: block_row, block_col
         !! the row
         !! the column
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: block
         !! the block to put

      INTEGER                                            :: c, ithread, mynode, p, r
      LOGICAL                                            :: dbg, existed, is_there, tr
      TYPE(dbcsr_distribution_obj)                      :: dist

!   ---------------------------------------------------------------------------

      dbg = .FALSE.

      ithread = 0
!$    ithread = omp_get_thread_num()
      IF (.NOT. ASSOCIATED(matrix%wms)) THEN
         CALL dbcsr_work_create(matrix, work_mutable=.TRUE.)
         matrix%valid = .FALSE.
      ENDIF
!$    IF (SIZE(matrix%wms) .LT. omp_get_num_threads()) &
!$       DBCSR_ABORT("Too few threads.")
      IF (.NOT. dbcsr_wm_use_mutable(matrix%wms(ithread + 1))) &
         DBCSR_ABORT("Data loss due to no conversion of appendable to mutable data")
      is_there = ASSOCIATED(block)
      !r = row ; c = col ; tr = .FALSE.
      !CALL dbcsr_get_stored_coordinates (matrix, r, c, tr)
      !CALL dbcsr_reserve_block2d (matrix, row, col, block)
      !write(*,*) 'add_block_node: block_row',block_row,' block_col',block_col
      CALL dbcsr_reserve_block2d(matrix, block_row, block_col, block, &
                                 existed=existed)
!
      IF (dbg) THEN
         r = block_row; c = block_col; tr = .FALSE.
         CALL dbcsr_get_stored_coordinates(matrix, r, c, p)
         CALL dbcsr_get_info(matrix, distribution=dist)
         CALL dbcsr_distribution_get(dist, mynode=mynode)
         IF (p .NE. mynode) &
            DBCSR_WARN("Adding non-local element")
      ENDIF
      IF (existed) DBCSR_WARN("You should not add existing blocks according to old API.")
      IF (.NOT. is_there) block(:, :) = 0.0_dp
   END SUBROUTINE dbcsr_add_block_node

   SUBROUTINE dbcsr_conjg(matrix)
      !! Conjugate a DBCSR matrix

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
         !! DBCSR matrix

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_conjg'
      INTEGER                                            :: blk, col, data_type, handle, row
      LOGICAL                                            :: tr
      TYPE(dbcsr_data_obj)                               :: data_any
      TYPE(dbcsr_iterator)                               :: iter

!   ---------------------------------------------------------------------------
!

      CALL timeset(routineN, handle)
      data_type = dbcsr_get_data_type(matrix)
      CALL dbcsr_data_init(data_any)
      CALL dbcsr_data_new(data_any, data_type)
      CALL dbcsr_iterator_start(iter, matrix)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, row, col, data_any, tr, blk)
         SELECT CASE (data_type)
         CASE (dbcsr_type_complex_4)
            data_any%d%c_sp = CONJG(data_any%d%c_sp)
         CASE (dbcsr_type_complex_8)
            data_any%d%c_dp = CONJG(data_any%d%c_dp)
         CASE DEFAULT
            ! needed for g95
         END SELECT
      ENDDO
      CALL dbcsr_iterator_stop(iter)
      CALL dbcsr_data_clear_pointer(data_any)
      CALL dbcsr_data_release(data_any)
      CALL timestop(handle)
   END SUBROUTINE dbcsr_conjg

   SUBROUTINE dbcsr_zero(matrix_a)
      !! fill a dbcsr matrix with zeros
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_a

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_zero'
      INTEGER                                            :: handle

      CALL timeset(routineN, handle)

      SELECT CASE (dbcsr_get_data_type(matrix_a))
      CASE (dbcsr_type_complex_4)
!$OMP       PARALLEL WORKSHARE DEFAULT(NONE), SHARED(matrix_a)
         matrix_a%data_area%d%c_sp = (0.0, 0.0)
!$OMP       END PARALLEL WORKSHARE
      CASE (dbcsr_type_complex_8)
!$OMP       PARALLEL WORKSHARE DEFAULT(NONE), SHARED(matrix_a)
         matrix_a%data_area%d%c_dp = (0.0_dp, 0.0_dp)
!$OMP       END PARALLEL WORKSHARE
      CASE (dbcsr_type_real_4)
!$OMP       PARALLEL WORKSHARE DEFAULT(NONE), SHARED(matrix_a)
         matrix_a%data_area%d%r_sp = 0.0
!$OMP       END PARALLEL WORKSHARE
      CASE (dbcsr_type_real_8)
!$OMP       PARALLEL WORKSHARE DEFAULT(NONE), SHARED(matrix_a)
         matrix_a%data_area%d%r_dp = 0.0_dp
!$OMP       END PARALLEL WORKSHARE
      END SELECT

      CALL timestop(handle)

   END SUBROUTINE dbcsr_zero

   SUBROUTINE dbcsr_scale_anytype(matrix_a, alpha_scalar, limits)
      !! Scales a DBCSR matrix by alpha
      !!
      !! Limits
      !! A 4-tuple describing (first_row, last_row, first_column, last_column). Set
      !! to 0 to avoid limiting.

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_a
         !! DBCSR matrix
      TYPE(dbcsr_scalar_type), INTENT(IN)                :: alpha_scalar
         !! a scalar
      INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL        :: limits
         !! Scale only a subbox

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_scale_anytype'
      INTEGER, PARAMETER                                 :: first_col_i = 3, first_row_i = 1, &
                                                            last_col_i = 4, last_row_i = 2

      INTEGER :: a_col, a_col_size, a_row, a_row_size, col_offset, handle, row_offset, &
                 scale_col_offset, scale_col_size, scale_row_offset, scale_row_size
      INTEGER, DIMENSION(4)                              :: my_limits
      LOGICAL                                            :: do_scale, has_limits, tr
      TYPE(dbcsr_data_obj)                               :: data_any
      TYPE(dbcsr_iterator)                               :: iter
      TYPE(dbcsr_scalar_type)                            :: one

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)

      ! Limits are only honored if the argument is present and any are
      ! non-zero.
      IF (PRESENT(limits)) THEN
         has_limits = ANY(limits(:) .NE. 0)
      ELSE
         has_limits = .FALSE.
      ENDIF
      my_limits(first_row_i) = 1
      my_limits(last_row_i) = dbcsr_nfullrows_total(matrix_a)
      my_limits(first_col_i) = 1
      my_limits(last_col_i) = dbcsr_nfullcols_total(matrix_a)
      IF (has_limits) THEN
         IF (limits(last_col_i) .NE. 0) THEN
            IF (debug_mod .AND. (limits(last_col_i) < 0 .OR. limits(last_col_i) > dbcsr_nfullcols_total(matrix_a))) &
               DBCSR_ABORT("Specified last column is out of bounds.")
            my_limits(last_col_i) = limits(last_col_i)
         ENDIF
         IF (limits(first_col_i) .NE. 0) THEN
            IF (debug_mod .AND. (limits(first_col_i) < 0 .OR. limits(first_col_i) > dbcsr_nfullcols_total(matrix_a))) &
               DBCSR_ABORT("Specified first column is out of bounds.")
            my_limits(first_col_i) = limits(first_col_i)
         ENDIF
         IF (limits(last_row_i) .NE. 0) THEN
            IF (debug_mod .AND. (limits(last_row_i) < 0 .OR. limits(last_row_i) > dbcsr_nfullrows_total(matrix_a))) &
               DBCSR_ABORT("Specified last row is out of bounds.")
            my_limits(last_row_i) = limits(last_row_i)
         ENDIF
         IF (limits(first_row_i) .NE. 0) THEN
            IF (debug_mod .AND. (limits(first_row_i) < 0 .OR. limits(first_row_i) > dbcsr_nfullrows_total(matrix_a))) &
               DBCSR_ABORT("Specified first row is out of bounds.")
            my_limits(first_row_i) = limits(first_row_i)
         ENDIF
      ENDIF
      !
      ! quick return if possible
      one = dbcsr_scalar_one(dbcsr_scalar_get_type(alpha_scalar))
      do_scale = .NOT. dbcsr_scalar_are_equal(alpha_scalar, one)
      !
      ! let's go
      IF (do_scale) THEN
!$OMP        PARALLEL DEFAULT (NONE) &
!$OMP                 PRIVATE (iter, data_any) &
!$OMP                 PRIVATE (a_row, a_col, tr, a_row_size, a_col_size, &
!$OMP                          row_offset, col_offset) &
!$OMP                 PRIVATE (scale_row_size, scale_col_size,&
!$OMP                          scale_row_offset, scale_col_offset) &
!$OMP                 SHARED (matrix_a, my_limits,alpha_scalar)
         CALL dbcsr_data_init(data_any)
         CALL dbcsr_data_new(data_any, dbcsr_type_1d_to_2d(dbcsr_get_data_type(matrix_a)))
         CALL dbcsr_iterator_start(iter, matrix_a, read_only=.FALSE., &
                                   contiguous_pointers=.FALSE., dynamic=.TRUE., &
                                   dynamic_byrows=.TRUE., shared=.TRUE.)
         iterations: DO WHILE (dbcsr_iterator_blocks_left(iter))
            CALL dbcsr_iterator_next_block(iter, a_row, a_col, data_any, tr, &
                                           row_size=a_row_size, col_size=a_col_size, &
                                           row_offset=row_offset, col_offset=col_offset)
            IF (a_row_size .GT. 0 .AND. a_col_size .GT. 0) THEN
               CALL frame_block_limit(a_row_size, row_offset, &
                                      my_limits(first_row_i), my_limits(last_row_i), &
                                      scale_row_size, scale_row_offset)
               CALL frame_block_limit(a_col_size, col_offset, &
                                      my_limits(first_col_i), my_limits(last_col_i), &
                                      scale_col_size, scale_col_offset)
               IF (tr) THEN
                  CALL swap(scale_row_size, scale_col_size)
                  CALL swap(scale_row_offset, scale_col_offset)
               ENDIF
               CALL dbcsr_block_scale(data_any, scale=alpha_scalar, &
                                      row_size=scale_row_size, col_size=scale_col_size, &
                                      lb=scale_row_offset, lb2=scale_col_offset)
            ENDIF
         ENDDO iterations
         CALL dbcsr_iterator_stop(iter)
         CALL dbcsr_data_clear_pointer(data_any)
         CALL dbcsr_data_release(data_any)
!$OMP        END PARALLEL
      ENDIF
      CALL timestop(handle)
   END SUBROUTINE dbcsr_scale_anytype

   ELEMENTAL SUBROUTINE frame_block_limit(block_size, block_offset, &
                                          first_limit, last_limit, &
                                          frame_size, frame_offset)
      !! Determines the effect of limits on a block

      INTEGER, INTENT(IN)                                :: block_size, block_offset, first_limit, &
                                                            last_limit
         !! size of block
         !! global offset of block
         !! lower limit
         !! upper limit
      INTEGER, INTENT(OUT)                               :: frame_size, frame_offset
         !! size of block region within the limits
         !! starting position of the block region that is within the limits

      INTEGER                                            :: f, l

      f = MAX(block_offset, first_limit)
      l = MIN(block_offset + block_size - 1, last_limit)
      frame_size = MAX(l - f + 1, 0)
      frame_offset = MIN(f - block_offset + 1, block_size)
   END SUBROUTINE frame_block_limit

   SUBROUTINE dbcsr_scale_by_vector_anytype(matrix_a, alpha, side)
      !! Scales a DBCSR matrix by alpha

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_a
         !! DBCSR matrix
      TYPE(dbcsr_data_obj), INTENT(IN), OPTIONAL         :: alpha
         !! the scaling vector
      CHARACTER(LEN=*), INTENT(IN)                       :: side
         !! apply the scaling from the side

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_scale_by_vector_anytype'

      INTEGER                                            :: a_blk, a_col, a_col_size, a_nze, a_row, &
                                                            a_row_size, col_offset, data_type, &
                                                            handle, i, icol, irow, row_offset
      LOGICAL                                            :: right, tr
      TYPE(dbcsr_data_obj)                               :: data_any
      TYPE(dbcsr_iterator)                               :: iter

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)

      ! check that alpha and matrix have the same data type
      IF (dbcsr_get_data_type(matrix_a) .NE. alpha%d%data_type) &
         DBCSR_ABORT("wrong data type matrix_a")

      IF (ASSOCIATED(alpha%d%r2_sp) .OR. ASSOCIATED(alpha%d%r2_dp) &
          .OR. ASSOCIATED(alpha%d%c2_sp) .OR. ASSOCIATED(alpha%d%c2_dp)) &
         DBCSR_ABORT("alpha is not a vector")
      !
      ! set vars
      right = .TRUE.
      SELECT CASE (side)
      CASE ('right'); right = .TRUE.
      CASE ('left'); right = .FALSE.
      CASE DEFAULT
         DBCSR_ABORT("wrong side="//side)
      END SELECT

      ! check that alpha and matrix have matching sizes
      IF (right .AND. dbcsr_nfullcols_total(matrix_a) /= dbcsr_data_get_size(alpha)) THEN
         DBCSR_ABORT("vector size does not match matrix row size for RHS scaling")
      ELSE IF ((.NOT. right) .AND. dbcsr_nfullrows_total(matrix_a) /= dbcsr_data_get_size(alpha)) THEN
         DBCSR_ABORT("vector size does not match matrix col size for LHS scaling")
      END IF

      !
      ! let's go
      data_type = dbcsr_get_data_type(matrix_a)
      CALL dbcsr_data_init(data_any)
      CALL dbcsr_data_new(data_any, dbcsr_get_data_type(matrix_a))
      CALL dbcsr_iterator_start(iter, matrix_a)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, a_row, a_col, data_any, tr, &
                                        block_number=a_blk, &
                                        row_size=a_row_size, col_size=a_col_size, &
                                        row_offset=row_offset, col_offset=col_offset)
         a_nze = a_row_size*a_col_size
         IF (a_nze .EQ. 0) CYCLE ! Skip empty blocks
         !
         ! let's scale
         IF (right) THEN
            SELECT CASE (data_type)
            CASE (dbcsr_type_real_4)
               DO i = 1, a_col_size
                  DO icol = (i - 1)*a_row_size + 1, (i - 1)*a_row_size + a_row_size
                     data_any%d%r_sp(icol) = data_any%d%r_sp(icol)*alpha%d%r_sp(col_offset + i - 1)
                  END DO
               END DO
            CASE (dbcsr_type_real_8)
               DO i = 1, a_col_size
                  DO icol = (i - 1)*a_row_size + 1, (i - 1)*a_row_size + a_row_size
                     data_any%d%r_dp(icol) = data_any%d%r_dp(icol)*alpha%d%r_dp(col_offset + i - 1)
                  END DO
               END DO
            CASE (dbcsr_type_complex_4)
               DO i = 1, a_col_size
                  DO icol = (i - 1)*a_row_size + 1, (i - 1)*a_row_size + a_row_size
                     data_any%d%c_sp(icol) = data_any%d%c_sp(icol)*alpha%d%c_sp(col_offset + i - 1)
                  END DO
               END DO
            CASE (dbcsr_type_complex_8)
               DO i = 1, a_col_size
                  DO icol = (i - 1)*a_row_size + 1, (i - 1)*a_row_size + a_row_size
                     data_any%d%c_dp(icol) = data_any%d%c_dp(icol)*alpha%d%c_dp(col_offset + i - 1)
                  END DO
               END DO
            END SELECT
         ELSE
            SELECT CASE (data_type)
            CASE (dbcsr_type_real_4)
               DO i = 1, a_row_size
                  DO irow = i, i + a_col_size*a_row_size - 1, a_row_size
                     data_any%d%r_sp(irow) = data_any%d%r_sp(irow)*alpha%d%r_sp(row_offset + i - 1)
                  END DO
               ENDDO
            CASE (dbcsr_type_real_8)
               DO i = 1, a_row_size
                  DO irow = i, i + a_col_size*a_row_size - 1, a_row_size
                     data_any%d%r_dp(irow) = data_any%d%r_dp(irow)*alpha%d%r_dp(row_offset + i - 1)
                  END DO
               ENDDO
            CASE (dbcsr_type_complex_4)
               DO i = 1, a_row_size
                  DO irow = i, i + a_col_size*a_row_size - 1, a_row_size
                     data_any%d%c_sp(irow) = data_any%d%c_sp(irow)*alpha%d%c_sp(row_offset + i - 1)
                  END DO
               ENDDO
            CASE (dbcsr_type_complex_8)
               DO i = 1, a_row_size
                  DO irow = i, i + a_col_size*a_row_size - 1, a_row_size
                     data_any%d%c_dp(irow) = data_any%d%c_dp(irow)*alpha%d%c_dp(row_offset + i - 1)
                  END DO
               ENDDO
            END SELECT
         ENDIF
      ENDDO
      CALL dbcsr_iterator_stop(iter)
      CALL dbcsr_data_clear_pointer(data_any)
      CALL dbcsr_data_release(data_any)
      CALL timestop(handle)

   END SUBROUTINE dbcsr_scale_by_vector_anytype

   SUBROUTINE dbcsr_add_anytype(matrix_a, matrix_b, alpha_scalar, beta_scalar, flop)
      !! add and scale matrices
      !! A = alpha*A + beta*B or

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_a
         !! DBCSR matrix
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_b
         !! DBCSR matrix
      TYPE(dbcsr_scalar_type), INTENT(IN), OPTIONAL      :: alpha_scalar, beta_scalar
      INTEGER(KIND=int_8), INTENT(INOUT), OPTIONAL       :: flop

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_add_anytype'

      INTEGER                                            :: data_type_a, data_type_b, &
                                                            handle, size_work, iw
      INTEGER(KIND=int_8)                                :: my_flop, local_matrix_size
      LOGICAL                                            :: do_scale
      TYPE(dbcsr_iterator)                               :: iter
      TYPE(dbcsr_scalar_type)                            :: my_beta_scalar

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)
      IF (.NOT. dbcsr_valid_index(matrix_a)) &
         DBCSR_ABORT("Invalid matrix")

      IF ((dbcsr_get_matrix_type(matrix_a) .EQ. dbcsr_type_symmetric .OR. &
           dbcsr_get_matrix_type(matrix_a) .EQ. dbcsr_type_antisymmetric) .NEQV. &
          (dbcsr_get_matrix_type(matrix_b) .EQ. dbcsr_type_symmetric .OR. &
           dbcsr_get_matrix_type(matrix_b) .EQ. dbcsr_type_antisymmetric)) THEN
         DBCSR_ABORT("Summing general with symmetric matrix NYI")
      ENDIF

      data_type_a = dbcsr_get_data_type(matrix_a)
      data_type_b = dbcsr_get_data_type(matrix_b)
      !
      my_beta_scalar = dbcsr_scalar_one(data_type_b)
      IF (PRESENT(beta_scalar)) my_beta_scalar = beta_scalar
      !
      ! let's go
      IF ((dbcsr_nblkrows_total(matrix_a) .NE. dbcsr_nblkrows_total(matrix_b)) .OR. &
          (dbcsr_nblkcols_total(matrix_a) .NE. dbcsr_nblkcols_total(matrix_b)) .OR. &
          (data_type_a .NE. data_type_b)) &
         DBCSR_ABORT("matrices not consistent")

      IF (data_type_a .NE. my_beta_scalar%data_type) &
         DBCSR_ABORT("beta type parameter not consistent with matrices type")

      do_scale = .NOT. dbcsr_scalar_are_equal(my_beta_scalar, dbcsr_scalar_one(data_type_b))

      IF (PRESENT(alpha_scalar)) THEN
         CALL dbcsr_scale(matrix_a, alpha_scalar=alpha_scalar)
      ENDIF

      IF ((.NOT. dbcsr_scalar_are_equal(my_beta_scalar, &
                                        dbcsr_scalar_zero(data_type_b))) .AND. &
          dbcsr_get_num_blocks(matrix_b) .GT. 0) THEN

         ! Pre-size work arrays of matrix_a to avoid continuous reallocation.
         ! Overestimate for symmetric matrix and multiple threads!
         local_matrix_size = INT(dbcsr_nfullrows_local(matrix_a), KIND=int_8)* &
                             dbcsr_nfullcols_local(matrix_a)
         size_work = MAX(0, INT(MIN(local_matrix_size - INT(dbcsr_get_nze(matrix_a), KIND=int_8), &
                                    INT(dbcsr_get_nze(matrix_b), KIND=int_8)), KIND=int_4))
         my_flop = 0
!$OMP        PARALLEL DEFAULT (NONE) &
!$OMP                 PRIVATE (iter, iw) &
!$OMP                 SHARED (matrix_a, matrix_b, data_type_b, size_work) &
!$OMP                 SHARED (do_scale, my_beta_scalar) &
!$OMP                 REDUCTION (+ : my_flop)
         CALL dbcsr_work_create(matrix_a, &
                                nblks_guess=matrix_b%nblks, &
                                sizedata_guess=size_work, &
                                work_mutable=.FALSE.)
!$OMP        BARRIER
         iw = 1
!$       iw = omp_get_thread_num() + 1
         CALL dbcsr_iterator_start(iter, matrix_b, &
                                   shared=.TRUE., read_only=.TRUE., contiguous_pointers=.FALSE., &
                                   dynamic=.TRUE., dynamic_byrows=.TRUE.)
         SELECT CASE (data_type_b)
         CASE (dbcsr_type_real_4)
            CALL dbcsr_add_anytype_s(matrix_a, matrix_b, iter, iw, do_scale, my_beta_scalar, my_flop)
         CASE (dbcsr_type_real_8)
            CALL dbcsr_add_anytype_d(matrix_a, matrix_b, iter, iw, do_scale, my_beta_scalar, my_flop)
         CASE (dbcsr_type_complex_4)
            CALL dbcsr_add_anytype_c(matrix_a, matrix_b, iter, iw, do_scale, my_beta_scalar, my_flop)
         CASE (dbcsr_type_complex_8)
            CALL dbcsr_add_anytype_z(matrix_a, matrix_b, iter, iw, do_scale, my_beta_scalar, my_flop)
         CASE default
            DBCSR_ABORT("Invalid data type")
         END SELECT
         CALL dbcsr_iterator_stop(iter)

         CALL dbcsr_finalize(matrix_a)
!$OMP        END PARALLEL
         IF (PRESENT(flop)) flop = flop + my_flop
      ENDIF
      CALL timestop(handle)
   END SUBROUTINE dbcsr_add_anytype

   SUBROUTINE dbcsr_add_d(matrix_a, matrix_b, alpha_scalar, beta_scalar)
      !! Interface for dbcsr_add
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_a
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_b
      REAL(real_8), INTENT(IN)                           :: alpha_scalar, beta_scalar

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_add_d'
      INTEGER                                            :: handle

      CALL timeset(routineN, handle)
      IF (dbcsr_get_data_type(matrix_a) .EQ. dbcsr_type_real_8 .AND. &
          dbcsr_get_data_type(matrix_b) .EQ. dbcsr_type_real_8) THEN
         CALL dbcsr_add_anytype(matrix_a, matrix_b, &
                                alpha_scalar=dbcsr_scalar(alpha_scalar), &
                                beta_scalar=dbcsr_scalar(beta_scalar))
      ELSEIF (dbcsr_get_data_type(matrix_a) .EQ. dbcsr_type_real_4 .AND. &
              dbcsr_get_data_type(matrix_b) .EQ. dbcsr_type_real_4) THEN
         CALL dbcsr_add_anytype(matrix_a, matrix_b, &
                                alpha_scalar=dbcsr_scalar(REAL(alpha_scalar, real_4)), &
                                beta_scalar=dbcsr_scalar(REAL(beta_scalar, real_4)))
      ELSEIF (dbcsr_get_data_type(matrix_a) .EQ. dbcsr_type_complex_4 .AND. &
              dbcsr_get_data_type(matrix_b) .EQ. dbcsr_type_complex_4) THEN
         CALL dbcsr_add_anytype(matrix_a, matrix_b, &
                                alpha_scalar=dbcsr_scalar(CMPLX(alpha_scalar, 0, real_4)), &
                                beta_scalar=dbcsr_scalar(CMPLX(beta_scalar, 0, real_4)))
      ELSEIF (dbcsr_get_data_type(matrix_a) .EQ. dbcsr_type_complex_8 .AND. &
              dbcsr_get_data_type(matrix_b) .EQ. dbcsr_type_complex_8) THEN
         CALL dbcsr_add_anytype(matrix_a, matrix_b, &
                                alpha_scalar=dbcsr_scalar(CMPLX(alpha_scalar, 0, real_8)), &
                                beta_scalar=dbcsr_scalar(CMPLX(beta_scalar, 0, real_8)))
      ELSE
         DBCSR_ABORT("Invalid combination of data type, NYI")
      ENDIF
      CALL timestop(handle)
   END SUBROUTINE dbcsr_add_d

   SUBROUTINE dbcsr_add_s(matrix_a, matrix_b, alpha_scalar, beta_scalar)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_a
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_b
      REAL(real_4), INTENT(IN)                           :: alpha_scalar, beta_scalar

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_add_s'
      INTEGER                                            :: handle

      CALL timeset(routineN, handle)
      IF (dbcsr_get_data_type(matrix_a) .EQ. dbcsr_type_real_4 .AND. &
          dbcsr_get_data_type(matrix_b) .EQ. dbcsr_type_real_4) THEN
         CALL dbcsr_add_anytype(matrix_a, matrix_b, &
                                alpha_scalar=dbcsr_scalar(alpha_scalar), &
                                beta_scalar=dbcsr_scalar(beta_scalar))
      ELSE
         DBCSR_ABORT("Invalid combination of data type, NYI")
      ENDIF
      CALL timestop(handle)
   END SUBROUTINE dbcsr_add_s

   SUBROUTINE dbcsr_add_z(matrix_a, matrix_b, alpha_scalar, beta_scalar)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_a
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_b
      COMPLEX(real_8), INTENT(IN)                        :: alpha_scalar, beta_scalar

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_add_z'
      INTEGER                                            :: handle

      CALL timeset(routineN, handle)
      IF (dbcsr_get_data_type(matrix_a) .EQ. dbcsr_type_complex_8 .AND. &
          dbcsr_get_data_type(matrix_b) .EQ. dbcsr_type_complex_8) THEN
         CALL dbcsr_add_anytype(matrix_a, matrix_b, &
                                alpha_scalar=dbcsr_scalar(alpha_scalar), &
                                beta_scalar=dbcsr_scalar(beta_scalar))
      ELSEIF (dbcsr_get_data_type(matrix_a) .EQ. dbcsr_type_complex_4 .AND. &
              dbcsr_get_data_type(matrix_b) .EQ. dbcsr_type_complex_4) THEN
         CALL dbcsr_add_anytype(matrix_a, matrix_b, &
                                alpha_scalar=dbcsr_scalar(CMPLX(alpha_scalar, KIND=real_4)), &
                                beta_scalar=dbcsr_scalar(CMPLX(beta_scalar, KIND=real_4)))
      ELSE
         DBCSR_ABORT("Invalid combination of data type, NYI")
      ENDIF
      CALL timestop(handle)
   END SUBROUTINE dbcsr_add_z

   SUBROUTINE dbcsr_add_c(matrix_a, matrix_b, alpha_scalar, beta_scalar)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_a
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_b
      COMPLEX(real_4), INTENT(IN)                        :: alpha_scalar, beta_scalar

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_add_c'
      INTEGER                                            :: handle

      CALL timeset(routineN, handle)
      IF (dbcsr_get_data_type(matrix_a) .EQ. dbcsr_type_complex_4 .AND. &
          dbcsr_get_data_type(matrix_b) .EQ. dbcsr_type_complex_4) THEN
         CALL dbcsr_add_anytype(matrix_a, matrix_b, &
                                alpha_scalar=dbcsr_scalar(alpha_scalar), &
                                beta_scalar=dbcsr_scalar(beta_scalar))
      ELSE
         DBCSR_ABORT("Invalid combination of data type, NYI")
      ENDIF
      CALL timestop(handle)
   END SUBROUTINE dbcsr_add_c

   SUBROUTINE dbcsr_add_on_diag_ds(matrix, alpha)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      REAL(kind=real_8), INTENT(IN)                      :: alpha

      IF (dbcsr_get_data_type(matrix) == dbcsr_type_real_4) THEN
         CALL dbcsr_add_on_diag_s(matrix, REAL(alpha, kind=real_4))
      ELSE
         CALL dbcsr_add_on_diag_d(matrix, alpha)
      ENDIF
   END SUBROUTINE dbcsr_add_on_diag_ds

   SUBROUTINE dbcsr_function_of_elements(matrix_a, func, a0, a1, a2)
      !! Computes various functions (defined by func) of matrix elements
      !! @note  sign(A,B) returns the value of A with the sign of B
      !! dbcsr_func_inverse:   1/(a1*x+a0)
      !! fails if the inversion produces infinite numbers
      !! dbcsr_func_inverse_special: 1/(x+sign(a0,x))
      !! safe inverse: if a0>0 then the denominator is never zero
      !! dbcsr_func_tanh:    tanh(a1*x+a0)
      !! dbcsr_func_dtanh:   d(tanh(a1*x+a0)) / dx
      !! dbcsr_func_ddtanh:  d2(tanh(a1*x+a0)) / dx2
      !! dbcsr_func_artanh:  artanh(a1*x+a0)=ln[(1+(a1*x+a0))/(1-(a1*x+a0))]/2
      !! fails if |a1*x+a0| >= 1
      !! dbcsr_func_sread_from_zero:  if |x|<|a0| then x=sign(a0,x)
      !! dbcsr_func_truncate:  if |x|>|a0| then x=sign(a0,x)
      !! dbcsr_func_sin:     sin(a1*x+a0)
      !! dbcsr_func_cos:     cos(a1*x+a0)
      !! dbcsr_func_dsin:    d(sin(a1*x+a0)) / dx = a1*cos(a1*x+a0)
      !! dbcsr_func_ddsin:   d2(sin(a1*x+a0)) / dx2 = -a1*a1*sin(a1*x+a0)
      !! dbcsr_func_asin:    asin(a1*x+a0)
      !! fails if |a1*x+a0| > 1

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_a
         !! DBCSR matrix
      INTEGER, INTENT(IN)                                :: func
      REAL(kind=dp), INTENT(IN), OPTIONAL                :: a0, a1, a2

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_function_of_elements'

      INTEGER                                            :: blk, col, col_size, data_type, handle, &
                                                            ielem, nze, row, row_size
      LOGICAL                                            :: tr_a
      REAL(kind=dp)                                      :: p0, p1, p2
      TYPE(dbcsr_data_obj)                               :: a_data
      TYPE(dbcsr_iterator)                               :: iter

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)

      IF (PRESENT(a0)) THEN
         p0 = a0
      ELSE
         p0 = 0.0_dp
      ENDIF
      IF (PRESENT(a1)) THEN
         p1 = a1
      ELSE
         p1 = 1.0_dp
      ENDIF
      IF (PRESENT(a2)) THEN
         p2 = a2
      ELSE
         p2 = 0.0_dp
      ENDIF

      data_type = dbcsr_get_data_type(matrix_a)
      CALL dbcsr_data_init(a_data)
      CALL dbcsr_data_new(a_data, data_type)
      CALL dbcsr_iterator_start(iter, matrix_a)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, row, col, a_data, tr_a, blk, &
                                        row_size=row_size, col_size=col_size)
         nze = row_size*col_size
         SELECT CASE (data_type)
            !CASE (dbcsr_type_real_4)
            !   a_data%d%r_sp(1:nze) = 1.0_real_4/a_data%d%r_sp(1:nze)
            !   IF(MAXVAL(ABS(a_data%d%r_sp)).GE.HUGE(0.0_real_4))&
            !      DBCSR_ABORT("Division by zero")
         CASE (dbcsr_type_real_8)
            SELECT CASE (func)
            CASE (dbcsr_func_spread_from_zero)
               ! if |x|<|a0| then x=|a0|*sign(x)
               DO ielem = 1, nze
                  IF (ABS(a_data%d%r_dp(ielem)) .LT. ABS(p0)) THEN
                     a_data%d%r_dp(ielem) = SIGN(p0, a_data%d%r_dp(ielem))
                  ENDIF
               ENDDO
            CASE (dbcsr_func_truncate)
               ! if |x|>|a0| then x=|a0|*sign(x)
               DO ielem = 1, nze
                  IF (ABS(a_data%d%r_dp(ielem)) .GT. ABS(p0)) THEN
                     a_data%d%r_dp(ielem) = SIGN(p0, a_data%d%r_dp(ielem))
                  ENDIF
               ENDDO
            CASE (dbcsr_func_inverse_special)
               !IF (MINVAL(ABS(a_data%d%r_dp)).le.ABS(p2)) THEN
               !   ! there is at least one near-zero element,
               !   ! invert element-by-element
               !   DO ielem=1,nze
               !     IF (a_data%d%r_dp(ielem).le.ABS(p2)) THEN
               !        a_data%d%r_dp(ielem) = 0.0_real_8
               !     ELSE
               !        a_data%d%r_dp(ielem) = &
               !           1.0_real_8/(p1*a_data%d%r_dp(ielem)+p0)
               !     ENDIF
               !   ENDDO
               !ELSE
               !   a_data%d%r_dp(1:nze) = 1.0_real_8/(p1*a_data%d%r_dp(1:nze)+p0)
               !ENDIF
               a_data%d%r_dp(1:nze) = 1.0_real_8/(a_data%d%r_dp(1:nze) + SIGN(p0, a_data%d%r_dp(1:nze)))
            CASE (dbcsr_func_inverse)
               a_data%d%r_dp(1:nze) = 1.0_real_8/(p1*a_data%d%r_dp(1:nze) + p0)
               IF (MAXVAL(ABS(a_data%d%r_dp)) .GE. HUGE(0.0_real_8)) &
                  DBCSR_ABORT("Division by zero")
            CASE (dbcsr_func_tanh)
               a_data%d%r_dp(1:nze) = TANH(p1*a_data%d%r_dp(1:nze) + p0)
            CASE (dbcsr_func_dtanh)
               a_data%d%r_dp(1:nze) = TANH(p1*a_data%d%r_dp(1:nze) + p0)
               a_data%d%r_dp(1:nze) = a_data%d%r_dp(1:nze)**2
               a_data%d%r_dp(1:nze) = p1*(1.0_real_8 - a_data%d%r_dp(1:nze))
            CASE (dbcsr_func_ddtanh)
               a_data%d%r_dp(1:nze) = TANH(p1*a_data%d%r_dp(1:nze) + p0)
               a_data%d%r_dp(1:nze) = a_data%d%r_dp(1:nze)**3 - a_data%d%r_dp(1:nze)
               a_data%d%r_dp(1:nze) = 2.0_real_8*(p1**2)*a_data%d%r_dp(1:nze)
            CASE (dbcsr_func_artanh)
               a_data%d%r_dp(1:nze) = p1*a_data%d%r_dp(1:nze) + p0
               IF (MAXVAL(ABS(a_data%d%r_dp)) .GE. 1.0_real_8) &
                  DBCSR_ABORT("ARTANH is undefined for |x|>=1")
               a_data%d%r_dp(1:nze) = (1.0_real_8 + a_data%d%r_dp(1:nze)) &
                                      /(1.0_real_8 - a_data%d%r_dp(1:nze))
               a_data%d%r_dp(1:nze) = 0.5_real_8*LOG(a_data%d%r_dp(1:nze))
            CASE (dbcsr_func_sin)
               a_data%d%r_dp(1:nze) = SIN(p1*a_data%d%r_dp(1:nze) + p0)
            CASE (dbcsr_func_cos)
               a_data%d%r_dp(1:nze) = COS(p1*a_data%d%r_dp(1:nze) + p0)
            CASE (dbcsr_func_dsin)
               a_data%d%r_dp(1:nze) = p1*COS(p1*a_data%d%r_dp(1:nze) + p0)
            CASE (dbcsr_func_ddsin)
               a_data%d%r_dp(1:nze) = -p1*p1*SIN(p1*a_data%d%r_dp(1:nze) + p0)
            CASE (dbcsr_func_asin)
               a_data%d%r_dp(1:nze) = p1*a_data%d%r_dp(1:nze) + p0
               IF (MAXVAL(ABS(a_data%d%r_dp)) .GT. 1.0_real_8) &
                  DBCSR_ABORT("ASIN is undefined for |x|>1")
               a_data%d%r_dp(1:nze) = ASIN(a_data%d%r_dp(1:nze))
            CASE DEFAULT
               DBCSR_ABORT("Unknown function of matrix elements")
            END SELECT
            !CASE (dbcsr_type_complex_4)
            !CASE (dbcsr_type_complex_8)
         CASE DEFAULT
            DBCSR_ABORT("Operation is implemented only for dp real values")
         END SELECT
      ENDDO
      CALL dbcsr_iterator_stop(iter)
      CALL dbcsr_data_clear_pointer(a_data)
      CALL dbcsr_data_release(a_data)
      CALL timestop(handle)

   END SUBROUTINE dbcsr_function_of_elements

   SUBROUTINE dbcsr_hadamard_product(matrix_a, matrix_b, matrix_c, &
                                     b_assume_value)
      !! Hadamard product
      !! C = A . B (C needs to be different from A and B)

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_a, matrix_b
         !! DBCSR matrix
         !! DBCSR matrix
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_c
         !! DBCSR matrix
      REAL(KIND=dp), INTENT(IN), OPTIONAL                :: b_assume_value

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_hadamard_product'

      INTEGER                                            :: blk, col, col_size, data_type, handle, &
                                                            nze, row, row_size
      LOGICAL                                            :: assume_blocks_in_b, found, tr_a, tr_b
      REAL(KIND=dp)                                      :: assumed_b_value
      TYPE(dbcsr_data_obj)                               :: a_data, b_data, c_data
      TYPE(dbcsr_iterator)                               :: iter

!   ---------------------------------------------------------------------------

      IF (PRESENT(b_assume_value)) THEN
         assume_blocks_in_b = .TRUE.
         assumed_b_value = b_assume_value
      ELSE
         assume_blocks_in_b = .FALSE.
         assumed_b_value = 0.0_dp
      ENDIF

      CALL timeset(routineN, handle)
      IF (dbcsr_get_data_type(matrix_a) .NE. dbcsr_get_data_type(matrix_b) .OR. &
          dbcsr_get_data_type(matrix_a) .NE. dbcsr_get_data_type(matrix_c)) &
         DBCSR_ABORT("data types not consistent, need to fix that")

      IF (dbcsr_nblkrows_total(matrix_a) .NE. dbcsr_nblkrows_total(matrix_b) .OR. &
          dbcsr_nblkrows_total(matrix_c) .NE. dbcsr_nblkrows_total(matrix_a)) &
         DBCSR_ABORT("matrices not consistent")

      data_type = dbcsr_get_data_type(matrix_a)
      CALL dbcsr_data_init(c_data)
      CALL dbcsr_data_new(c_data, data_type, &
                          data_size=dbcsr_max_row_size(matrix_a)*dbcsr_max_col_size(matrix_a))
      CALL dbcsr_zero(matrix_c)
      CALL dbcsr_data_init(a_data)
      CALL dbcsr_data_new(a_data, data_type)
      CALL dbcsr_data_init(b_data)
      CALL dbcsr_data_new(b_data, data_type)
      CALL dbcsr_iterator_start(iter, matrix_a)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         SELECT CASE (dbcsr_get_data_type(matrix_a))
            !CASE (dbcsr_type_real_4)
         CASE (dbcsr_type_real_8)
            CALL dbcsr_iterator_next_block(iter, row, col, a_data, tr_a, blk, &
                                           row_size=row_size, col_size=col_size)
            nze = row_size*col_size
            CALL dbcsr_get_block_p(matrix_b, row, col, b_data, tr_b, found)
            IF (tr_a .NEQV. tr_b) &
               DBCSR_ABORT("tr not consistent, need to fix that")
            IF (found) THEN
               SELECT CASE (data_type)
               CASE (dbcsr_type_real_4)
                  c_data%d%r_sp(1:nze) = a_data%d%r_sp(1:nze)*b_data%d%r_sp(1:nze)
                  CALL dbcsr_put_block(matrix_c, row, col, c_data%d%r_sp(1:nze), transposed=tr_a, &
                                       summation=.FALSE.)
               CASE (dbcsr_type_real_8)
                  c_data%d%r_dp(1:nze) = a_data%d%r_dp(1:nze)*b_data%d%r_dp(1:nze)
                  CALL dbcsr_put_block(matrix_c, row, col, c_data%d%r_dp(1:nze), transposed=tr_a, &
                                       summation=.FALSE.)
               CASE (dbcsr_type_complex_4)
                  c_data%d%c_sp(1:nze) = a_data%d%c_sp(1:nze)*b_data%d%c_sp(1:nze)
                  CALL dbcsr_put_block(matrix_c, row, col, c_data%d%c_sp(1:nze), transposed=tr_a, &
                                       summation=.FALSE.)
               CASE (dbcsr_type_complex_8)
                  c_data%d%c_dp(1:nze) = a_data%d%c_dp(1:nze)*b_data%d%c_dp(1:nze)
                  CALL dbcsr_put_block(matrix_c, row, col, c_data%d%c_dp(1:nze), transposed=tr_a, &
                                       summation=.FALSE.)
               END SELECT
            ELSE
               IF (assume_blocks_in_b) THEN ! this makes not too much sense, to delete ?
                  SELECT CASE (data_type)
                  CASE (dbcsr_type_real_4)
                     c_data%d%r_sp(1:nze) = a_data%d%r_sp(1:nze)*REAL(assumed_b_value, KIND=sp)
                     CALL dbcsr_put_block(matrix_c, row, col, c_data%d%r_sp(1:nze), transposed=tr_a, &
                                          summation=.FALSE.)
                  CASE (dbcsr_type_real_8)
                     c_data%d%r_dp(1:nze) = a_data%d%r_dp(1:nze)*assumed_b_value
                     CALL dbcsr_put_block(matrix_c, row, col, c_data%d%r_dp(1:nze), transposed=tr_a, &
                                          summation=.FALSE.)
                  CASE (dbcsr_type_complex_4)
                     c_data%d%c_sp(1:nze) = a_data%d%c_sp(1:nze)*REAL(assumed_b_value, KIND=sp)
                     CALL dbcsr_put_block(matrix_c, row, col, c_data%d%c_sp(1:nze), transposed=tr_a, &
                                          summation=.FALSE.)
                  CASE (dbcsr_type_complex_8)
                     c_data%d%c_dp(1:nze) = a_data%d%c_dp(1:nze)*assumed_b_value
                     CALL dbcsr_put_block(matrix_c, row, col, c_data%d%c_dp(1:nze), transposed=tr_a, &
                                          summation=.FALSE.)
                  END SELECT
               ENDIF
            ENDIF
            !CASE (dbcsr_type_complex_4)
            !CASE (dbcsr_type_complex_8)
         CASE DEFAULT
            DBCSR_ABORT("Only real double precision")
         END SELECT
      ENDDO
      CALL dbcsr_iterator_stop(iter)
      CALL dbcsr_finalize(matrix_c)
      CALL dbcsr_data_clear_pointer(a_data)
      CALL dbcsr_data_clear_pointer(b_data)
      CALL dbcsr_data_release(c_data)
      CALL dbcsr_data_release(a_data)
      CALL dbcsr_data_release(b_data)
      CALL timestop(handle)
   END SUBROUTINE dbcsr_hadamard_product

   SUBROUTINE dbcsr_init_random(matrix, keep_sparsity)
      !! ... TODO : unify with other version which is generic in the data_type
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      LOGICAL, OPTIONAL                                  :: keep_sparsity

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_init_random'

      INTEGER                                            :: col, col_size, handle, hold, iseed(4), &
                                                            mynode, ncol, nrow, row, row_size, &
                                                            stored_col, stored_row
      INTEGER, DIMENSION(:), POINTER                     :: col_blk_size, row_blk_size
      LOGICAL                                            :: found, my_keep_sparsity, tr
      REAL(real_8), ALLOCATABLE, DIMENSION(:)            :: rnd
      REAL(real_8), DIMENSION(:, :), POINTER             :: buff, data_d

!   ---------------------------------------------------------------------------

      my_keep_sparsity = .FALSE.
      IF (PRESENT(keep_sparsity)) my_keep_sparsity = keep_sparsity

      CALL timeset(routineN, handle)

      row_blk_size => array_data(matrix%row_blk_size)
      col_blk_size => array_data(matrix%col_blk_size)
      mynode = dbcsr_mp_mynode(dbcsr_distribution_mp(dbcsr_distribution(matrix)))
      CALL dbcsr_work_create(matrix, work_mutable=.TRUE.)

      ALLOCATE (rnd(MAXVAL(row_blk_size)*MAXVAL(col_blk_size)))
      nrow = dbcsr_nblkrows_total(matrix)
      ncol = dbcsr_nblkcols_total(matrix)
      DO row = 1, nrow
      DO col = 1, ncol
         row_size = row_blk_size(row)
         col_size = col_blk_size(col)
         tr = .FALSE.
         stored_row = row
         stored_col = col
         CALL dbcsr_get_stored_coordinates(matrix, stored_row, stored_col, hold)
         IF (hold .EQ. mynode) THEN
            CALL dbcsr_get_block_p(matrix, stored_row, stored_col, data_d, tr, found)
            IF (found .OR. (.NOT. my_keep_sparsity)) THEN
               ! set the seed for dlarnv, is here to guarantee same value of the random numbers
               ! for all layouts (and block distributions)
               CALL set_larnv_seed(row, nrow, col, ncol, 1, iseed)
               CALL dlarnv(1, iseed, row_size*col_size, rnd(1))
            ENDIF
            IF (found) THEN
               CALL dcopy(row_size*col_size, rnd, 1, data_d, 1)
            ELSE
               IF (.NOT. my_keep_sparsity) THEN
                  ALLOCATE (buff(row_size, col_size))
                  CALL dcopy(row_size*col_size, rnd, 1, buff, 1)
                  CALL dbcsr_put_block(matrix, stored_row, stored_col, buff)
                  DEALLOCATE (buff)
               ENDIF
            ENDIF
         ENDIF
      ENDDO
      ENDDO
      DEALLOCATE (rnd)

      CALL dbcsr_finalize(matrix)
      CALL timestop(handle)

   END SUBROUTINE dbcsr_init_random

   SUBROUTINE dbcsr_get_block_diag(matrix, diag)
      !! get the diagonal of a dbcsr matrix

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! the matrix
      TYPE(dbcsr_type), INTENT(INOUT)                    :: diag
         !! the diagonal

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_block_diag'

      INTEGER                                            :: blk, col, handle, row
      LOGICAL                                            :: tr
      TYPE(dbcsr_data_obj)                               :: data_a
      TYPE(dbcsr_iterator)                               :: iter

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)
      CALL dbcsr_create(diag, name='diag of '//TRIM(matrix%name), &
                        template=matrix)

      CALL dbcsr_data_init(data_a)
      CALL dbcsr_data_new(data_a, dbcsr_get_data_type(matrix))
      CALL dbcsr_iterator_start(iter, matrix)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, row, col, data_a, tr, blk)
         IF (row .EQ. col) CALL dbcsr_put_block(diag, row, col, data_a, transposed=tr)
      ENDDO
      CALL dbcsr_iterator_stop(iter)
      CALL dbcsr_data_clear_pointer(data_a)
      CALL dbcsr_data_release(data_a)
      CALL dbcsr_finalize(diag)
      CALL timestop(handle)
   END SUBROUTINE dbcsr_get_block_diag

   LOGICAL FUNCTION symmetry_consistent(matrix_type, data_type)
      !! checks if matrix symmetry and data_type are consistent
      !! \brief note: does not check the symmetry of the data itself

      CHARACTER, INTENT(IN)                    :: matrix_type
      INTEGER, INTENT(IN)                      :: data_type

      symmetry_consistent = .FALSE.

      SELECT CASE (data_type)
      CASE (dbcsr_type_real_4, dbcsr_type_real_8)
         SELECT CASE (matrix_type)
         CASE (dbcsr_type_no_symmetry, dbcsr_type_symmetric, dbcsr_type_antisymmetric)
            symmetry_consistent = .TRUE.
         END SELECT
      CASE (dbcsr_type_complex_4, dbcsr_type_complex_8)
         SELECT CASE (matrix_type)
         CASE (dbcsr_type_no_symmetry, dbcsr_type_hermitian, dbcsr_type_antihermitian)
            symmetry_consistent = .TRUE.
         END SELECT
      CASE DEFAULT
         DBCSR_ABORT("Invalid data type.")
      END SELECT

   END FUNCTION symmetry_consistent

   LOGICAL FUNCTION symmetry_compatible(matrix_type_a, matrix_type_b)
      !! checks if symmetries of two matrices are compatible for copying
      !! \brief data from matrix_a(source) to matrix_b(target)

      CHARACTER, INTENT(IN)                    :: matrix_type_a, matrix_type_b

      symmetry_compatible = .FALSE.

      SELECT CASE (matrix_type_a)
      CASE (dbcsr_type_no_symmetry)
         SELECT CASE (matrix_type_b)
         CASE (dbcsr_type_no_symmetry)
            symmetry_compatible = .TRUE.
         END SELECT
      CASE (dbcsr_type_symmetric, dbcsr_type_hermitian)
         SELECT CASE (matrix_type_b)
         CASE (dbcsr_type_symmetric, dbcsr_type_hermitian)
            symmetry_compatible = .TRUE.
         END SELECT
      CASE (dbcsr_type_antisymmetric, dbcsr_type_antihermitian)
         SELECT CASE (matrix_type_b)
         CASE (dbcsr_type_antisymmetric, dbcsr_type_antihermitian)
            symmetry_compatible = .TRUE.
         END SELECT
      CASE DEFAULT
         DBCSR_ABORT("Invalid matrix type.")
      END SELECT

   END FUNCTION symmetry_compatible

   SUBROUTINE dbcsr_copy(matrix_b, matrix_a, name, keep_sparsity, &
                         shallow_data, keep_imaginary, matrix_type)
      !! copy a matrix

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_b
         !! target DBCSR matrix
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_a
         !! source DBCSR matrix
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL             :: name
         !! name of the new matrix
      LOGICAL, INTENT(IN), OPTIONAL                      :: keep_sparsity, shallow_data, &
                                                            keep_imaginary
         !! keep the target matrix sparsity; default is False.
         !! shallow data copy
         !! when copy from complex to real,& the default is to keep only the real part; if this flag is set, the imaginary part is
         !! used
      CHARACTER, INTENT(IN), OPTIONAL                    :: matrix_type
         !! 'N' for normal, 'T' for transposed, 'S' for symmetric, and 'A' for antisymmetric

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_copy'
      CHARACTER                                          :: new_matrix_type, repl_type
      INTEGER                                            :: handle, new_type
      LOGICAL                                            :: keep_sparse, shallow

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)
      IF (.NOT. symmetry_consistent(dbcsr_get_matrix_type(matrix_a), dbcsr_get_data_type(matrix_a))) &
         DBCSR_ABORT("Source matrix symmetry not consistent with its data type.")
      shallow = .FALSE.; IF (PRESENT(shallow_data)) shallow = shallow_data
      keep_sparse = .FALSE.
      IF (PRESENT(keep_sparsity)) keep_sparse = keep_sparsity
      IF (keep_sparse .AND. .NOT. dbcsr_valid_index(matrix_b)) &
         DBCSR_ABORT("Target matrix must be valid to keep its sparsity")
      IF (keep_sparse .AND. shallow) &
         DBCSR_WARN("Shallow copy not compatibly with sparsity retainment")
      IF (keep_sparse) THEN
         IF (PRESENT(name)) matrix_b%name = name
         CALL dbcsr_copy_into_existing(matrix_b, matrix_a)
      ELSE
         IF (dbcsr_valid_index(matrix_b)) THEN
            new_type = dbcsr_get_data_type(matrix_b)
            repl_type = dbcsr_get_replication_type(matrix_b)
         ELSE
            new_type = dbcsr_get_data_type(matrix_a)
            repl_type = dbcsr_get_replication_type(matrix_a)
         ENDIF
         new_matrix_type = dbcsr_get_matrix_type(matrix_a)
         IF (PRESENT(matrix_type)) THEN
            IF (.NOT. symmetry_compatible(dbcsr_get_matrix_type(matrix_a), matrix_type)) &
               CALL dbcsr_abort(__LOCATION__, "Specified target matrix symmetry "//matrix_type// &
                                " not compatible with source matrix type "//dbcsr_get_matrix_type(matrix_a))
            new_matrix_type = matrix_type
         END IF
         IF (.NOT. symmetry_consistent(new_matrix_type, new_type)) &
            CALL dbcsr_abort(__LOCATION__, "Target matrix symmetry "// &
                             new_matrix_type//" not consistent with its data type.")
         IF (PRESENT(name)) THEN
            CALL dbcsr_create(matrix_b, name=TRIM(name), &
                              template=matrix_a, &
                              matrix_type=new_matrix_type, &
                              data_type=new_type)
         ELSE
            CALL dbcsr_create(matrix_b, &
                              data_type=new_type, &
                              matrix_type=new_matrix_type, &
                              template=matrix_a)
         ENDIF
         CALL ensure_array_size(matrix_b%index, ub=SIZE(matrix_a%index), &
                                memory_type=dbcsr_get_index_memory_type(matrix_b))
         !
         ! copy index and data
         matrix_b%index(1:SIZE(matrix_a%index)) = matrix_a%index(:)
         IF (.NOT. shallow) THEN
            IF (matrix_a%nze > dbcsr_get_data_size(matrix_a)) &
               DBCSR_ABORT("Source matrix sizes not consistent!")
            CALL dbcsr_data_ensure_size(matrix_b%data_area, &
                                        dbcsr_data_get_size_referenced(matrix_a%data_area))
            IF (dbcsr_get_data_type(matrix_a) .EQ. dbcsr_get_data_type(matrix_b)) &
               THEN
               CALL dbcsr_data_copyall(matrix_b%data_area, &
                                       matrix_a%data_area)
            ELSE
               CALL dbcsr_data_convert(matrix_b%data_area, &
                                       matrix_a%data_area, drop_real=keep_imaginary)
            ENDIF
         ELSE
            IF (dbcsr_get_data_type(matrix_a) .NE. dbcsr_get_data_type(matrix_b)) &
               DBCSR_ABORT("Shallow copy only possible when retaining data type.")
            CALL dbcsr_switch_data_area(matrix_b, matrix_a%data_area)
         ENDIF
         !
         ! the row_p, col_i and blk_p ...
         CALL dbcsr_repoint_index(matrix_b)
         matrix_b%nze = matrix_a%nze
         matrix_b%nblks = matrix_b%nblks
         matrix_b%valid = .TRUE.

         matrix_b%sparsity_id = matrix_a%sparsity_id
      ENDIF
      CALL timestop(handle)
   END SUBROUTINE dbcsr_copy

   SUBROUTINE dbcsr_copy_into_existing(matrix_b, matrix_a)
      !! copy a matrix, retaining current sparsity

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_b
         !! target DBCSR matrix
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_a
         !! source DBCSR matrix

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_copy_into_existing'

      INTEGER                                            :: col_size, data_type, dst_col, dst_row, &
                                                            handle, rel, row_size, src_col, &
                                                            src_cs, src_row, src_rs
      LOGICAL                                            :: dst_tr, making_symmetric, neg_imag, &
                                                            neg_real, src_tr
      TYPE(dbcsr_data_obj)                               :: dst_data, src_data
      TYPE(dbcsr_iterator)                               :: dst_iter, src_iter

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)
      IF (.NOT. dbcsr_valid_index(matrix_b)) &
         DBCSR_ABORT("Matrix_b is not valid")
      IF (dbcsr_get_data_type(matrix_b) .NE. dbcsr_get_data_type(matrix_a)) &
         DBCSR_ABORT("Matrices have different data types.")
      data_type = dbcsr_get_data_type(matrix_b)
      neg_real = matrix_b%negate_real
      neg_imag = matrix_b%negate_imaginary
      making_symmetric = dbcsr_has_symmetry(matrix_b) &
                         .AND. .NOT. dbcsr_has_symmetry(matrix_a)
      IF (making_symmetric) THEN
         CALL dbcsr_copy_into_existing_sym(matrix_b, matrix_a)
         CALL timestop(handle)
         RETURN
      ENDIF
      CALL dbcsr_data_init(src_data)
      CALL dbcsr_data_init(dst_data)
      CALL dbcsr_data_new(src_data, data_type)
      CALL dbcsr_data_new(dst_data, data_type)
      CALL dbcsr_iterator_start(src_iter, matrix_a)
      CALL dbcsr_iterator_start(dst_iter, matrix_b)
      ! Iterate through the blocks of the source and destination
      ! matrix. There are three possibilities: 1. copy the data for
      ! blocks present in both; 2 skip source blocks not present in the
      ! target; 3 zero blocks not present in the source.
      IF (dbcsr_iterator_blocks_left(src_iter)) THEN
         CALL dbcsr_iterator_next_block(src_iter, src_row, src_col, src_data, &
                                        src_tr)
      ELSE
         src_row = 0; src_col = 0
      ENDIF
      DO WHILE (dbcsr_iterator_blocks_left(dst_iter))
         CALL dbcsr_iterator_next_block(dst_iter, dst_row, dst_col, dst_data, &
                                        dst_tr, row_size=row_size, col_size=col_size)
         ! Now find the source position that is greater or equal to the
         ! target one. I.e, skip blocks that the target doesn't have.
         rel = pos_relation(dst_row, dst_col, src_row, src_col)
         DO WHILE (rel .EQ. 1 .AND. dbcsr_iterator_blocks_left(src_iter))
            CALL dbcsr_iterator_next_block(src_iter, src_row, src_col, &
                                           src_data, src_tr, row_size=src_rs, col_size=src_cs)
            rel = pos_relation(dst_row, dst_col, src_row, src_col)
         ENDDO
         SELECT CASE (rel)
         CASE (-1, 1)
            ! Target lags source or ran out of source
            CALL dbcsr_data_clear(dst_data)
         CASE (0)
            ! Copy the data
            IF (dbcsr_data_get_size(src_data) .NE. dbcsr_data_get_size(dst_data)) &
               DBCSR_ABORT("Block sizes not equal!")
            IF (src_tr .EQV. dst_tr) THEN
               CALL dbcsr_data_copyall(dst_data, src_data)
            ELSE
               CALL dbcsr_block_partial_copy(dst=dst_data, dst_tr=dst_tr, &
                                             dst_rs=row_size, dst_cs=col_size, &
                                             dst_r_lb=1, dst_c_lb=1, &
                                             src=src_data, src_tr=src_tr, &
                                             src_rs=src_rs, src_cs=src_cs, &
                                             src_r_lb=1, src_c_lb=1, &
                                             nrow=row_size, ncol=col_size)
               IF (neg_real) THEN
                  CALL dbcsr_block_real_neg(dst_data, row_size, col_size)
               ENDIF
               IF (neg_imag) THEN
                  CALL dbcsr_block_conjg(dst_data, row_size, col_size)
               ENDIF
            ENDIF
         CASE default
            DBCSR_ABORT("Trouble syncing iterators")
         END SELECT
      END DO
      CALL dbcsr_iterator_stop(src_iter)
      CALL dbcsr_iterator_stop(dst_iter)
      CALL dbcsr_data_clear_pointer(src_data)
      CALL dbcsr_data_clear_pointer(dst_data)
      CALL dbcsr_data_release(src_data)
      CALL dbcsr_data_release(dst_data)
      CALL timestop(handle)

   END SUBROUTINE dbcsr_copy_into_existing

   SUBROUTINE dbcsr_copy_into_existing_sym(matrix_b, matrix_a)
      !! copy a matrix, retaining current sparsity

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_b
         !! target DBCSR matrix
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_a
         !! source DBCSR matrix

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_copy_into_existing_sym'

      INTEGER                                            :: col_size, data_type, dst_col, dst_row, &
                                                            handle, row_size, src_col, src_cs, &
                                                            src_row, src_rs
      LOGICAL                                            :: dst_tr, found, neg_imag, neg_real, src_tr
      TYPE(dbcsr_data_obj)                               :: dst_data, src_data
      TYPE(dbcsr_iterator)                               :: dst_iter

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)
      IF (dbcsr_get_data_type(matrix_b) .NE. dbcsr_get_data_type(matrix_a)) &
         DBCSR_ABORT("Matrices have different data types.")
      data_type = dbcsr_get_data_type(matrix_b)
      IF (.NOT. dbcsr_has_symmetry(matrix_b) .OR. dbcsr_has_symmetry(matrix_a)) &
         DBCSR_ABORT("Must copy from non-symmetric to symmetric matrix.")
      neg_real = matrix_b%negate_real
      neg_imag = matrix_b%negate_imaginary

      CALL dbcsr_data_init(src_data)
      CALL dbcsr_data_init(dst_data)
      CALL dbcsr_data_new(src_data, data_type)
      CALL dbcsr_data_new(dst_data, data_type)
      CALL dbcsr_iterator_start(dst_iter, matrix_b)
      ! Iterate through the blocks of the destination matrix.  For each
      ! one, try to find an appropriate source matrix block and copy it
      ! into the destination matrix.
      DO WHILE (dbcsr_iterator_blocks_left(dst_iter))
         CALL dbcsr_iterator_next_block(dst_iter, dst_row, dst_col, dst_data, &
                                        dst_tr, row_size=row_size, col_size=col_size)
         src_row = dst_row
         src_col = dst_col
         IF (checker_tr(dst_row, dst_col)) &
            CALL swap(src_row, src_col)
         CALL dbcsr_get_block_p(matrix_a, src_row, src_col, src_data, src_tr, &
                                found=found, row_size=src_rs, col_size=src_cs)
         IF (.NOT. found) THEN
            CALL dbcsr_data_clear(dst_data)
         ELSE
            IF (checker_tr(dst_row, dst_col)) THEN
               src_tr = .NOT. src_tr
               CALL swap(src_rs, src_cs)
            ENDIF
            CALL dbcsr_block_partial_copy(dst=dst_data, dst_tr=dst_tr, &
                                          dst_rs=row_size, dst_cs=col_size, &
                                          dst_r_lb=1, dst_c_lb=1, &
                                          src=src_data, src_tr=src_tr, &
                                          src_rs=src_rs, src_cs=src_cs, &
                                          src_r_lb=1, src_c_lb=1, &
                                          nrow=row_size, ncol=col_size)
            IF (neg_real .AND. checker_tr(dst_row, dst_col)) THEN
               CALL dbcsr_block_real_neg(dst_data, row_size, col_size)
            ENDIF
            IF (neg_imag .AND. checker_tr(dst_row, dst_col)) THEN
               CALL dbcsr_block_conjg(dst_data, row_size, col_size)
            ENDIF
         ENDIF
      END DO
      CALL dbcsr_iterator_stop(dst_iter)
      CALL dbcsr_data_clear_pointer(src_data)
      CALL dbcsr_data_clear_pointer(dst_data)
      CALL dbcsr_data_release(src_data)
      CALL dbcsr_data_release(dst_data)
      CALL timestop(handle)

   END SUBROUTINE dbcsr_copy_into_existing_sym

   ELEMENTAL FUNCTION pos_relation(row1, col1, row2, col2) RESULT(relation)
      !! Determines the relation between two matrix positions.

      INTEGER, INTENT(IN)                                :: row1, col1, row2, col2
      INTEGER                                            :: relation
         !! Relation between positions 1 and 2. 0: same -1: pos1 < pos2 1: pos1 > pos2

      IF (row1 .LT. row2) THEN
         relation = -1
      ELSEIF (row1 .GT. row2) THEN
         relation = 1
      ELSE ! rows are equal, check column
         IF (col1 .LT. col2) THEN
            relation = -1
         ELSEIF (col1 .GT. col2) THEN
            relation = 1
         ELSE
            relation = 0
         ENDIF
      ENDIF
   END FUNCTION pos_relation

   SUBROUTINE dbcsr_copy_submatrix(matrix_b, matrix_a, name, &
                                   block_row_bounds, block_column_bounds, &
                                   shallow_data)
      !! Copy a submatrix.

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_b
         !! target DBCSR matrix
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_a
         !! source DBCSR matrix
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL             :: name
         !! name of the new matrix
      INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL        :: block_row_bounds, block_column_bounds
         !! rows to extract (array of size 2 holding the lower and upper inclusive bounds)
         !! columns to extract (array of size 2 holding the lower and upper inclusive bounds)
      LOGICAL, INTENT(IN), OPTIONAL                      :: shallow_data
         !! shallow data copy

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_copy_submatrix'

      INTEGER                                            :: blk_p, col, handle, nblocks, new_blk, &
                                                            old_blk, row
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: blkp_list, col_list, row_list
      LOGICAL                                            :: shallow, tr
      TYPE(dbcsr_data_obj)                               :: data_block
      TYPE(dbcsr_iterator)                               :: iter

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)
      IF (PRESENT(shallow_data)) THEN
         shallow = shallow_data
      ELSE
         shallow = .FALSE.
      ENDIF
      ! Verify assumptions.
      IF (PRESENT(block_row_bounds)) THEN
         IF (SIZE(block_row_bounds) /= 2) &
            DBCSR_ABORT("Size of bounds specifier must be 2")
      ENDIF
      IF (PRESENT(block_column_bounds)) THEN
         IF (SIZE(block_column_bounds) /= 2) &
            DBCSR_ABORT("Size of bounds specifier must be 2")
      ENDIF
      ! Setup target matrix
      CALL dbcsr_create(matrix_b, name=name, template=matrix_a)
      CALL dbcsr_finalize(matrix_b)
      IF (.NOT. shallow) THEN
         ! Non-shallow copy uses the standard iterator on the source and
         ! block put on the target.
!
!$OMP PARALLEL DEFAULT (NONE) &
!$OMP          PRIVATE (data_block, iter, row, col, tr) &
!$OMP          SHARED (matrix_a, matrix_b,&
!$OMP                  block_row_bounds, block_column_bounds)
         CALL dbcsr_work_create(matrix_b, work_mutable=.FALSE.)
         CALL dbcsr_data_init(data_block)
         CALL dbcsr_data_new(data_block, dbcsr_get_data_type(matrix_a))
         CALL dbcsr_iterator_start(iter, matrix_a, dynamic=.TRUE., &
                                   dynamic_byrows=.TRUE.)
         DO WHILE (dbcsr_iterator_blocks_left(iter))
            CALL dbcsr_iterator_next_block(iter, row, col, data_block, tr)
            ! Only keep the block if they are within the specified bounds.
            IF (PRESENT(block_row_bounds)) THEN
               IF (row .LT. block_row_bounds(1)) CYCLE
               IF (row .GT. block_row_bounds(2)) CYCLE
            ENDIF
            IF (PRESENT(block_column_bounds)) THEN
               IF (col .LT. block_column_bounds(1)) CYCLE
               IF (col .GT. block_column_bounds(2)) CYCLE
            ENDIF
            CALL dbcsr_put_block(matrix_b, row, col, data_block, transposed=tr)
         END DO
         CALL dbcsr_iterator_stop(iter)
         CALL dbcsr_data_clear_pointer(data_block)
         CALL dbcsr_data_release(data_block)
         CALL dbcsr_finalize(matrix_b)
!$OMP END PARALLEL
      ELSE
         ! For the shallow copy the source matrix data is referenced.
         CALL dbcsr_switch_data_area(matrix_b, matrix_a%data_area)
         nblocks = dbcsr_get_num_blocks(matrix_a) ! High estimate.
         ! Shallow copy goes through source's data blocks and inserts
         ! the only the ones corresponding to the submatrix specifier
         ! into the target. Block pointers must remain the same as in
         ! the source.
         ALLOCATE (row_list(nblocks), col_list(nblocks), blkp_list(nblocks))
         !
         CALL dbcsr_iterator_start(iter, matrix_a)
         new_blk = 1
         DO WHILE (dbcsr_iterator_blocks_left(iter))
            CALL dbcsr_iterator_next_block(iter, row, col, &
                                           blk=old_blk, blk_p=blk_p)
            ! Only keep the block if they are within the specified bounds.
            IF (PRESENT(block_row_bounds)) THEN
               IF (row .LT. block_row_bounds(1)) CYCLE
               IF (row .GT. block_row_bounds(2)) CYCLE
            ENDIF
            IF (PRESENT(block_column_bounds)) THEN
               IF (col .LT. block_column_bounds(1)) CYCLE
               IF (col .GT. block_column_bounds(2)) CYCLE
            ENDIF
            row_list(new_blk) = row
            col_list(new_blk) = col
            blkp_list(new_blk) = blk_p
            new_blk = new_blk + 1
         ENDDO
         new_blk = new_blk - 1
         CALL dbcsr_iterator_stop(iter)
         CALL dbcsr_reserve_blocks(matrix_b, row_list(1:new_blk), &
                                   col_list(1:new_blk), blkp_list(1:new_blk))
      ENDIF
      !
      CALL timestop(handle)
   END SUBROUTINE dbcsr_copy_submatrix

   SUBROUTINE dbcsr_crop_matrix(matrix_b, matrix_a, &
                                full_row_bounds, full_column_bounds, &
                                shallow_data)
      !! Crop and copies a matrix.

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_b
         !! target DBCSR matrix
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_a
         !! source DBCSR matrix
      INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL        :: full_row_bounds, full_column_bounds
         !! rows to extract (array of size 2 holding the lower and upper inclusive bounds)
         !! columns to extract (array of size 2 holding the lower and upper inclusive bounds)
      LOGICAL, INTENT(IN), OPTIONAL                      :: shallow_data

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_crop_matrix'

      INTEGER                                            :: col, f_col_f, f_row_f, handle, l_col_l, &
                                                            l_row_l, row
      INTEGER, DIMENSION(2)                              :: block_col_bounds, block_row_bounds
      LOGICAL                                            :: part_col, part_f_col, part_f_row, &
                                                            part_l_col, part_l_row, part_row, &
                                                            shallow, tr
      TYPE(dbcsr_data_obj)                               :: data_block
      TYPE(dbcsr_iterator)                               :: iter

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)
      part_l_col = .FALSE.
      part_f_col = .FALSE.
      part_l_row = .FALSE.
      part_f_row = .FALSE.
      IF (PRESENT(shallow_data)) THEN
         shallow = shallow_data
      ELSE
         shallow = .FALSE.
      ENDIF
      block_row_bounds = 0
      block_col_bounds = 0
      part_col = .FALSE.
      part_row = .FALSE.
      !
      ! If row bounds are present, they must be converted to block
      ! addressing.
      IF (PRESENT(full_row_bounds)) THEN
         IF (SIZE(full_row_bounds) /= 2) &
            DBCSR_ABORT("Size of bounds specifier must be 2")
         IF (full_row_bounds(1) < 0) &
            DBCSR_ABORT("Invalid first row bound.")
         IF (full_row_bounds(2) > dbcsr_nfullrows_total(matrix_a)) &
            DBCSR_ABORT("Invalid last row bound.")
         IF (full_row_bounds(1) .EQ. 0) THEN
            block_row_bounds(1) = 1
         ELSE
            CALL find_block_of_element(full_row_bounds(1), block_row_bounds(1), &
                                       dbcsr_nblkrows_total(matrix_a), &
                                       dbcsr_row_block_offsets(matrix_a), &
                                       hint=0)
            part_f_row = array_get(dbcsr_row_block_offsets(matrix_a), block_row_bounds(1)) &
                         .NE. full_row_bounds(1)
         ENDIF
         f_row_f = -7
         IF (part_f_row) THEN
            ! Block offset of last cleared row
            f_row_f = full_row_bounds(1) - &
                      array_get(dbcsr_row_block_offsets(matrix_a), block_row_bounds(1))
         ENDIF
         IF (full_row_bounds(2) .EQ. 0) THEN
            block_row_bounds(2) = dbcsr_nblkrows_total(matrix_a)
         ELSE
            CALL find_block_of_element(full_row_bounds(2), block_row_bounds(2), &
                                       dbcsr_nblkrows_total(matrix_a), &
                                       dbcsr_row_block_offsets(matrix_a), &
                                       hint=0)
            part_l_row = array_get(dbcsr_row_block_offsets(matrix_a), block_row_bounds(2) + 1) - 1 &
                         .NE. full_row_bounds(2)
         ENDIF
         ! Block offset of first cleared row
         l_row_l = -7
         IF (part_l_row) THEN
            l_row_l = 2 + full_row_bounds(2) - &
                      array_get(dbcsr_row_block_offsets(matrix_a), block_row_bounds(2))
         ENDIF
         part_row = part_f_row .OR. part_l_row
      ENDIF
      !
      ! If column bounds are present, they must be converted to block
      ! addressing.
      IF (PRESENT(full_column_bounds)) THEN
         IF (SIZE(full_column_bounds) /= 2) &
            DBCSR_ABORT("Size of bounds specifier must be 2")
         IF (full_column_bounds(1) < 0) &
            DBCSR_ABORT("Invalid first column bound.")
         IF (full_column_bounds(2) > dbcsr_nfullcols_total(matrix_a)) &
            DBCSR_ABORT("Invalid last column bound.")
         IF (full_column_bounds(1) .EQ. 0) THEN
            block_col_bounds(1) = 1
         ELSE
            CALL find_block_of_element(full_column_bounds(1), block_col_bounds(1), &
                                       dbcsr_nblkcols_total(matrix_a), &
                                       dbcsr_col_block_offsets(matrix_a), &
                                       hint=0)
            part_f_col = array_get(dbcsr_col_block_offsets(matrix_a), block_col_bounds(1)) &
                         .NE. full_column_bounds(1)
         ENDIF
         f_col_f = -7
         IF (part_f_col) THEN
            ! Block offset of last cleared column
            f_col_f = full_column_bounds(1) - &
                      array_get(dbcsr_col_block_offsets(matrix_a), block_col_bounds(1))
         ENDIF
         IF (full_column_bounds(2) .EQ. 0) THEN
            block_col_bounds(2) = dbcsr_nblkcols_total(matrix_a)
         ELSE
            CALL find_block_of_element(full_column_bounds(2), block_col_bounds(2), &
                                       dbcsr_nblkcols_total(matrix_a), &
                                       dbcsr_col_block_offsets(matrix_a), &
                                       hint=0)
            part_l_col = array_get(dbcsr_col_block_offsets(matrix_a), block_col_bounds(2) + 1) - 1 &
                         .NE. full_column_bounds(2)
         ENDIF
         l_col_l = -7
         IF (part_l_col) THEN
            ! Block offset of first cleared column
            l_col_l = 2 + full_column_bounds(2) - &
                      array_get(dbcsr_col_block_offsets(matrix_a), block_col_bounds(2))
         ENDIF
         part_col = part_f_col .OR. part_l_col
      ENDIF
      !
      ! First copy the blocks then perform the intra-block zeroing.
      CALL dbcsr_copy_submatrix(matrix_b, matrix_a, &
                                block_row_bounds=block_row_bounds, &
                                block_column_bounds=block_col_bounds, &
                                shallow_data=shallow)
      IF (part_row .OR. part_col) THEN
!$OMP PARALLEL DEFAULT (NONE) &
!$OMP          PRIVATE (data_block, iter, row, col, tr) &
!$OMP          SHARED (matrix_b,&
!$OMP                  part_row, part_f_row, part_l_row, f_row_f, l_row_l, &
!$OMP                  part_col, part_f_col, part_l_col, f_col_f, l_col_l,&
!$OMP                  block_row_bounds, block_col_bounds)
         CALL dbcsr_data_init(data_block)
         CALL dbcsr_data_new(data_block, dbcsr_type_1d_to_2d(dbcsr_get_data_type(matrix_b)))
         CALL dbcsr_iterator_start(iter, matrix_b, &
                                   dynamic=.TRUE., dynamic_byrows=.TRUE.)
         DO WHILE (dbcsr_iterator_blocks_left(iter))
            CALL dbcsr_iterator_next_block(iter, row, col, data_block, tr)
            IF (part_row) THEN
               IF (row .LT. block_row_bounds(1)) CYCLE
               IF (row .GT. block_row_bounds(2)) CYCLE
            ENDIF
            IF (part_col) THEN
               IF (col .LT. block_col_bounds(1)) CYCLE
               IF (col .GT. block_col_bounds(2)) CYCLE
            ENDIF
            IF (part_row) THEN
               IF (part_f_row .AND. row .EQ. block_row_bounds(1)) THEN
                  CALL dbcsr_data_clear(data_block, ub=f_row_f, tr=tr)
               ENDIF
               IF (part_l_row .AND. row .EQ. block_row_bounds(2)) THEN
                  CALL dbcsr_data_clear(data_block, lb=l_row_l, tr=tr)
               ENDIF
            ENDIF
            IF (part_col) THEN
               IF (part_f_col .AND. col .EQ. block_col_bounds(1)) THEN
                  CALL dbcsr_data_clear(data_block, ub2=f_col_f, tr=tr)
               ENDIF
               IF (part_l_col .AND. col .EQ. block_col_bounds(2)) THEN
                  CALL dbcsr_data_clear(data_block, lb2=l_col_l, tr=tr)
               ENDIF
            ENDIF
         END DO
         CALL dbcsr_iterator_stop(iter)
         CALL dbcsr_data_clear_pointer(data_block)
         CALL dbcsr_data_release(data_block)
         CALL dbcsr_finalize(matrix_b)
!$OMP END PARALLEL
      ENDIF
      !
      CALL timestop(handle)
   END SUBROUTINE dbcsr_crop_matrix

   SUBROUTINE dbcsr_triu(matrix_a)
      !! triu of a dbcsr matrix

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_a
         !! the matrix

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_triu'
      INTEGER                                            :: blk, blk_nze, col, col_size, handle, i, &
                                                            j, row, row_size
      LOGICAL                                            :: tr
      REAL(dp), DIMENSION(:, :), POINTER                 :: DATA
      TYPE(dbcsr_iterator)                               :: iter

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)
      CALL dbcsr_iterator_start(iter, matrix_a)

      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, row, col, DATA, tr, &
                                        block_number=blk, row_size=row_size, col_size=col_size)
         blk_nze = row_size*col_size
         IF (row .GT. col) CALL dbcsr_remove_block(matrix_a, row, col, blk_nze, blk)
         IF (row .EQ. col) THEN
            DO j = 1, col_size
            DO i = j + 1, row_size
               DATA(i, j) = 0.0_dp
            ENDDO
            ENDDO
         ENDIF
      ENDDO

      CALL dbcsr_iterator_stop(iter)

      CALL dbcsr_finalize(matrix_a)
      CALL timestop(handle)
   END SUBROUTINE dbcsr_triu

   SUBROUTINE dbcsr_filter_anytype(matrix, eps, method, &
                                   use_absolute, filter_diag)
      !! filter a dbcsr matrix

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
         !! the matrix
      TYPE(dbcsr_scalar_type), INTENT(IN)                :: eps
         !! the threshold
      INTEGER, INTENT(IN), OPTIONAL                      :: method
         !! how the matrix is filtered
      LOGICAL, INTENT(in), OPTIONAL                      :: use_absolute, filter_diag
         !! NYI

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_filter_anytype'

      COMPLEX(KIND=real_4), DIMENSION(:), POINTER        :: data_c
      COMPLEX(KIND=real_8), DIMENSION(:), POINTER        :: data_z
      INTEGER                                            :: blk, blk_nze, col, col_size, handle, &
                                                            my_method, row, row_size, data_type
      LOGICAL                                            :: gt0, my_filter_diag, tr
      REAL(KIND=real_4)                                  :: nrm_s
      REAL(KIND=real_4), DIMENSION(:), POINTER           :: data_s
      REAL(KIND=real_8)                                  :: my_absolute, nrm_d
      REAL(KIND=real_8), DIMENSION(:), POINTER           :: data_d
      TYPE(dbcsr_iterator)                               :: iter

      REAL(KIND=real_8), EXTERNAL                        :: DZNRM2
#if defined (__ACCELERATE)
      REAL(KIND=real_8), EXTERNAL                        :: SCNRM2
#else
      REAL(KIND=real_4), EXTERNAL                        :: SCNRM2
#endif

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)
      my_method = dbcsr_filter_frobenius
      IF (PRESENT(method)) my_method = method
      my_absolute = 1.0_dp
      IF (PRESENT(use_absolute)) my_absolute = dbcsr_maxabs(matrix)
      my_filter_diag = .TRUE.
      IF (PRESENT(filter_diag)) my_filter_diag = filter_diag

      SELECT CASE (eps%data_type)
      CASE (dbcsr_type_real_4)
         gt0 = eps%r_sp .GT. 0.0_real_4
      CASE (dbcsr_type_real_8)
         gt0 = eps%r_dp .GT. 0.0_real_8
      CASE (dbcsr_type_complex_4)
         gt0 = ABS(eps%c_sp) .GT. 0.0_real_4
      CASE (dbcsr_type_complex_8)
         gt0 = ABS(eps%c_dp) .GT. 0.0_real_8
      CASE default
         gt0 = .FALSE.
      END SELECT

      IF (gt0) THEN
         data_type = dbcsr_get_data_type(matrix)

!$OMP        PARALLEL DEFAULT(NONE) PRIVATE(iter,row,col,data_s,data_d,data_c,data_z,tr, &
!$OMP                                       blk,row_size,col_size,blk_nze,nrm_d,nrm_s) &
!$OMP                               SHARED(my_method,my_absolute,eps,matrix,data_type)

         CALL dbcsr_iterator_start(iter, matrix, contiguous_pointers=.TRUE.)
         DO WHILE (dbcsr_iterator_blocks_left(iter))
            SELECT CASE (data_type)

            CASE (dbcsr_type_real_4)
               CALL dbcsr_iterator_next_block(iter, row, col, data_s, tr, blk, &
                                              row_size, col_size)
               blk_nze = row_size*col_size
               IF (blk_nze .EQ. 0) CYCLE ! Skip empty blocks
               SELECT CASE (my_method)
               CASE (dbcsr_filter_frobenius)
                  !
                  ! Frobenius based
                  nrm_s = norm2(data_s)
                  IF (nrm_s .LT. my_absolute*eps%r_sp) &
                     CALL dbcsr_remove_block(matrix, row, col, blk_nze, blk)
               CASE DEFAULT
                  DBCSR_ABORT("Only Frobenius based filtering")
               END SELECT

            CASE (dbcsr_type_real_8)
               CALL dbcsr_iterator_next_block(iter, row, col, data_d, tr, blk, &
                                              row_size, col_size)
               blk_nze = row_size*col_size
               IF (blk_nze .EQ. 0) CYCLE ! Skip empty blocks
               SELECT CASE (my_method)
               CASE (dbcsr_filter_frobenius)
                  !
                  ! Frobenius based
                  nrm_d = norm2(data_d)
                  IF (nrm_d .LT. my_absolute*eps%r_dp) &
                     CALL dbcsr_remove_block(matrix, row, col, blk_nze, blk)
               CASE DEFAULT
                  DBCSR_ABORT("Only Frobenius based filtering")
               END SELECT

            CASE (dbcsr_type_complex_4)
               CALL dbcsr_iterator_next_block(iter, row, col, data_c, tr, blk, &
                                              row_size, col_size)
               blk_nze = row_size*col_size
               IF (blk_nze .EQ. 0) CYCLE ! Skip empty blocks
               SELECT CASE (my_method)
               CASE (dbcsr_filter_frobenius)
                  !
                  ! Frobenius based
                  nrm_d = SCNRM2(SIZE(data_c), data_c(1), 1)
                  IF (nrm_d .LT. my_absolute*eps%r_dp) &
                     CALL dbcsr_remove_block(matrix, row, col, blk_nze, blk)
               CASE DEFAULT
                  DBCSR_ABORT("Only Frobenius based filtering")
               END SELECT

            CASE (dbcsr_type_complex_8)
               CALL dbcsr_iterator_next_block(iter, row, col, data_z, tr, blk, &
                                              row_size, col_size)
               blk_nze = row_size*col_size
               IF (blk_nze .EQ. 0) CYCLE ! Skip empty blocks
               SELECT CASE (my_method)
               CASE (dbcsr_filter_frobenius)
                  !
                  ! Frobenius based
                  nrm_d = DZNRM2(SIZE(data_z), data_z(1), 1)
                  IF (nrm_d .LT. my_absolute*eps%r_dp) &
                     CALL dbcsr_remove_block(matrix, row, col, blk_nze, blk)
               CASE DEFAULT
                  DBCSR_ABORT("Only Frobenius based filtering")
               END SELECT

            CASE DEFAULT
               DBCSR_ABORT("Wrong data type")
            END SELECT
         ENDDO
         CALL dbcsr_iterator_stop(iter)
         CALL dbcsr_finalize(matrix, reshuffle=.TRUE.)
!$OMP        END PARALLEL

         CALL dbcsr_index_compact(matrix)

      ENDIF
      CALL timestop(handle)
   END SUBROUTINE dbcsr_filter_anytype

   SUBROUTINE dbcsr_norm_scalar(matrix, which_norm, norm_scalar)
      !! compute a norm of a dbcsr matrix

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
         !! the matrix
      INTEGER, INTENT(IN)                                :: which_norm
      REAL(KIND=real_8), INTENT(OUT)                     :: norm_scalar

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_norm_scalar'

      INTEGER                                            ::  handle

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)

      SELECT CASE (which_norm)
      CASE (dbcsr_norm_frobenius)
         norm_scalar = dbcsr_frobenius_norm(matrix)
      CASE (dbcsr_norm_maxabsnorm)
         norm_scalar = dbcsr_maxabs(matrix)
      CASE (dbcsr_norm_gershgorin)
         norm_scalar = dbcsr_gershgorin_norm(matrix)
      CASE DEFAULT
         DBCSR_ABORT("this norm is NYI")
      END SELECT
      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE dbcsr_norm_r8_vec(matrix, which_norm, norm_vector)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      INTEGER, INTENT(IN)                                :: which_norm
      REAL(KIND=real_8), DIMENSION(:), INTENT(OUT), &
         TARGET, CONTIGUOUS                              :: norm_vector

      REAL(KIND=real_8), DIMENSION(:), POINTER, CONTIGUOUS :: v_p
      TYPE(dbcsr_data_obj)                               :: norm_vector_a

      CALL dbcsr_data_init(norm_vector_a)
      CALL dbcsr_data_new(norm_vector_a, dbcsr_type_real_8)
      v_p => norm_vector
      CALL dbcsr_data_set_pointer(norm_vector_a, v_p)
      CALL dbcsr_norm_vec(matrix, which_norm, norm_vector_a)
      CALL dbcsr_data_clear_pointer(norm_vector_a)
      CALL dbcsr_data_release(norm_vector_a)

   END SUBROUTINE dbcsr_norm_r8_vec

   SUBROUTINE dbcsr_norm_vec(matrix, which_norm, norm_vector)
      !! compute the column norms of the dbcsr matrix

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
         !! the matrix
      INTEGER, INTENT(IN)                                :: which_norm
      TYPE(dbcsr_data_obj), INTENT(INOUT)                :: norm_vector

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_norm_vec'

      INTEGER                                            :: blk, col, col_offset, i, j, row, &
                                                            row_offset, handle
      LOGICAL                                            :: tr
      TYPE(dbcsr_data_obj)                               :: data_a
      TYPE(dbcsr_iterator)                               :: iter

      CALL timeset(routineN, handle)

      SELECT CASE (which_norm)
      CASE (dbcsr_norm_column)
         IF (dbcsr_data_get_type(norm_vector) /= dbcsr_get_data_type(matrix)) &
            DBCSR_ABORT("Mismatched vector/matrix data types")
         IF (dbcsr_has_symmetry(matrix)) THEN
            IF (dbcsr_data_get_size(norm_vector) < dbcsr_nfullrows_total(matrix)) &
               DBCSR_ABORT("Passed vector too small")
         END IF
         IF (dbcsr_data_get_size(norm_vector) < dbcsr_nfullcols_total(matrix)) &
            DBCSR_ABORT("Passed vector too small")
         CALL dbcsr_data_init(data_a)
         CALL dbcsr_data_new(data_a, dbcsr_type_1d_to_2d(dbcsr_get_data_type(matrix)))
         CALL dbcsr_data_clear(norm_vector)
         CALL dbcsr_iterator_start(iter, matrix)
         DO WHILE (dbcsr_iterator_blocks_left(iter))
            CALL dbcsr_iterator_next_block(iter, row, col, data_a, tr, &
                                           blk, row_offset=row_offset, col_offset=col_offset)
            SELECT CASE (dbcsr_get_data_type(matrix))
            CASE (dbcsr_type_real_4)
               IF (dbcsr_has_symmetry(matrix) .AND. row .NE. col) THEN
                  DO j = 1, SIZE(data_a%d%r2_sp, 2)
                     DO i = 1, SIZE(data_a%d%r2_sp, 1)
                        norm_vector%d%r_sp(col_offset + j - 1) &
                           = norm_vector%d%r_sp(col_offset + j - 1) &
                             + data_a%d%r2_sp(i, j)**2
                        norm_vector%d%r_sp(row_offset + i - 1) &
                           = norm_vector%d%r_sp(row_offset + i - 1) &
                             + data_a%d%r2_sp(i, j)**2
                     ENDDO
                  ENDDO
               ELSE
                  DO j = 1, SIZE(data_a%d%r2_sp, 2)
                     DO i = 1, SIZE(data_a%d%r2_sp, 1)
                        norm_vector%d%r_sp(col_offset + j - 1) &
                           = norm_vector%d%r_sp(col_offset + j - 1) &
                             + data_a%d%r2_sp(i, j)*data_a%d%r2_sp(i, j)
                     ENDDO
                  ENDDO
               ENDIF
            CASE (dbcsr_type_real_8)
               IF (dbcsr_has_symmetry(matrix) .AND. row .NE. col) THEN
                  DO j = 1, SIZE(data_a%d%r2_dp, 2)
                     DO i = 1, SIZE(data_a%d%r2_dp, 1)
                        norm_vector%d%r_dp(col_offset + j - 1) &
                           = norm_vector%d%r_dp(col_offset + j - 1) &
                             + data_a%d%r2_dp(i, j)**2
                        norm_vector%d%r_dp(row_offset + i - 1) &
                           = norm_vector%d%r_dp(row_offset + i - 1) &
                             + data_a%d%r2_dp(i, j)**2
                     ENDDO
                  ENDDO
               ELSE
                  DO j = 1, SIZE(data_a%d%r2_dp, 2)
                     DO i = 1, SIZE(data_a%d%r2_dp, 1)
                        norm_vector%d%r_dp(col_offset + j - 1) &
                           = norm_vector%d%r_dp(col_offset + j - 1) &
                             + data_a%d%r2_dp(i, j)*data_a%d%r2_dp(i, j)
                     ENDDO
                  ENDDO
               ENDIF
            CASE DEFAULT
               DBCSR_ABORT("Only real values")
            END SELECT
         ENDDO
         CALL dbcsr_iterator_stop(iter)
         CALL dbcsr_data_clear_pointer(data_a)
         CALL dbcsr_data_release(data_a)
         SELECT CASE (dbcsr_get_data_type(matrix))
         CASE (dbcsr_type_real_4)
            CALL mp_sum(norm_vector%d%r_sp, &
                        dbcsr_mp_group(dbcsr_distribution_mp(matrix%dist)))
            norm_vector%d%r_sp = SQRT(norm_vector%d%r_sp)
         CASE (dbcsr_type_real_8)
            CALL mp_sum(norm_vector%d%r_dp, &
                        dbcsr_mp_group(dbcsr_distribution_mp(matrix%dist)))
            norm_vector%d%r_dp = SQRT(norm_vector%d%r_dp)
         END SELECT

      CASE DEFAULT
         DBCSR_ABORT("this norm is NYI")
      END SELECT
      CALL timestop(handle)
   END SUBROUTINE dbcsr_norm_vec

   FUNCTION dbcsr_gershgorin_norm(matrix) RESULT(norm)
      !! compute a norm of a dbcsr matrix

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
         !! the matrix
      REAL(KIND=real_8)                                  :: norm

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_gershgorin_norm'

      COMPLEX(KIND=real_4), DIMENSION(:, :), POINTER     :: data_c
      COMPLEX(KIND=real_8), DIMENSION(:, :), POINTER     :: data_z
      INTEGER                                            :: blk, col, col_offset, handle, i, j, nc, &
                                                            nr, row, row_offset
      LOGICAL                                            :: any_sym, tr
      REAL(KIND=real_4), DIMENSION(:, :), POINTER        :: data_r
      REAL(KIND=real_8), DIMENSION(:, :), POINTER        :: data_d
      REAL(real_8), ALLOCATABLE, DIMENSION(:)            :: buff_d
      TYPE(dbcsr_iterator)                               :: iter

      CALL timeset(routineN, handle)

      nr = dbcsr_nfullrows_total(matrix)
      nc = dbcsr_nfullcols_total(matrix)

      any_sym = dbcsr_get_matrix_type(matrix) .EQ. dbcsr_type_symmetric .OR. &
                dbcsr_get_matrix_type(matrix) .EQ. dbcsr_type_antisymmetric

      IF (nr .NE. nc) &
         DBCSR_ABORT("not a square matrix")

      norm = 0.0_dp
      ALLOCATE (buff_d(nr))
      buff_d = 0.0_dp
      CALL dbcsr_iterator_start(iter, matrix)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         SELECT CASE (dbcsr_get_data_type(matrix))
         CASE (dbcsr_type_real_4)
            CALL dbcsr_iterator_next_block(iter, row, col, data_r, tr, blk, &
                                           row_offset=row_offset, col_offset=col_offset)
            DO j = 1, SIZE(data_r, 2)
            DO i = 1, SIZE(data_r, 1)
               buff_d(row_offset + i - 1) = buff_d(row_offset + i - 1) + ABS(data_r(i, j))
               IF (any_sym .AND. row .NE. col) &
                  buff_d(col_offset + j - 1) = buff_d(col_offset + j - 1) + ABS(data_r(i, j))
            ENDDO
            ENDDO
         CASE (dbcsr_type_real_8)
            CALL dbcsr_iterator_next_block(iter, row, col, data_d, tr, blk, &
                                           row_offset=row_offset, col_offset=col_offset)
            DO j = 1, SIZE(data_d, 2)
            DO i = 1, SIZE(data_d, 1)
               buff_d(row_offset + i - 1) = buff_d(row_offset + i - 1) + ABS(data_d(i, j))
               IF (any_sym .AND. row .NE. col) &
                  buff_d(col_offset + j - 1) = buff_d(col_offset + j - 1) + ABS(data_d(i, j))
            ENDDO
            ENDDO
         CASE (dbcsr_type_complex_4)
            CALL dbcsr_iterator_next_block(iter, row, col, data_c, tr, blk, &
                                           row_offset=row_offset, col_offset=col_offset)
            DO j = 1, SIZE(data_c, 2)
            DO i = 1, SIZE(data_c, 1)
               buff_d(row_offset + i - 1) = buff_d(row_offset + i - 1) + ABS(data_c(i, j))
               IF (any_sym .AND. row .NE. col) &
                  DBCSR_ABORT("Only nonsymmetric matrix so far")
               !     buff_d(col_offset+j-1) = buff_d(col_offset+j-1) + ABS(data_c(i,j))
            ENDDO
            ENDDO
         CASE (dbcsr_type_complex_8)
            CALL dbcsr_iterator_next_block(iter, row, col, data_z, tr, blk, &
                                           row_offset=row_offset, col_offset=col_offset)
            DO j = 1, SIZE(data_z, 2)
            DO i = 1, SIZE(data_z, 1)
               buff_d(row_offset + i - 1) = buff_d(row_offset + i - 1) + ABS(data_z(i, j))
               IF (any_sym .AND. row .NE. col) &
                  DBCSR_ABORT("Only nonsymmetric matrix so far")
               !     buff_d(col_offset+j-1) = buff_d(col_offset+j-1) + ABS(data_z(i,j))
            ENDDO
            ENDDO
         CASE DEFAULT
            DBCSR_ABORT("Wrong data type")
         END SELECT
      ENDDO
      CALL dbcsr_iterator_stop(iter)
      CALL mp_sum(buff_d, dbcsr_mp_group(dbcsr_distribution_mp(matrix%dist)))
      norm = MAXVAL(buff_d)
      DEALLOCATE (buff_d)

      CALL timestop(handle)

   END FUNCTION dbcsr_gershgorin_norm

   FUNCTION dbcsr_maxabs(matrix) RESULT(norm)
      !! compute a norm of a dbcsr matrix

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
         !! the matrix
      REAL(real_8)                                       :: norm

      COMPLEX(KIND=real_4), DIMENSION(:, :), POINTER     :: data_c
      COMPLEX(KIND=real_8), DIMENSION(:, :), POINTER     :: data_z
      INTEGER                                            :: blk, col, row
      LOGICAL                                            :: tr
      REAL(KIND=real_4), DIMENSION(:, :), POINTER        :: data_r
      REAL(KIND=real_8), DIMENSION(:, :), POINTER        :: data_d
      TYPE(dbcsr_iterator)                               :: iter

!   ---------------------------------------------------------------------------

      norm = 0.0_dp
      CALL dbcsr_iterator_start(iter, matrix)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         SELECT CASE (dbcsr_get_data_type(matrix))
         CASE (dbcsr_type_real_4)
            CALL dbcsr_iterator_next_block(iter, row, col, data_r, tr, blk)
            norm = MAX(norm, REAL(MAXVAL(ABS(data_r)), dp))
         CASE (dbcsr_type_real_8)
            CALL dbcsr_iterator_next_block(iter, row, col, data_d, tr, blk)
            norm = MAX(norm, MAXVAL(ABS(data_d)))
         CASE (dbcsr_type_complex_4)
            CALL dbcsr_iterator_next_block(iter, row, col, data_c, tr, blk)
            norm = MAX(norm, REAL(MAXVAL(ABS(data_c)), dp))
         CASE (dbcsr_type_complex_8)
            CALL dbcsr_iterator_next_block(iter, row, col, data_z, tr, blk)
            norm = MAX(norm, MAXVAL(ABS(data_z)))
         CASE DEFAULT
            DBCSR_ABORT("Wrong data type")
         END SELECT
      ENDDO
      CALL dbcsr_iterator_stop(iter)

      CALL mp_max(norm, dbcsr_mp_group(dbcsr_distribution_mp(matrix%dist)))

   END FUNCTION dbcsr_maxabs

   FUNCTION dbcsr_frobenius_norm(matrix, local) RESULT(norm)
      !! compute a norm of a dbcsr matrix

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! the matrix
      LOGICAL, INTENT(in), OPTIONAL                      :: local
      REAL(KIND=real_8)                                  :: norm

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_frobenius_norm'

      COMPLEX(KIND=real_4), DIMENSION(:, :), POINTER     :: data_c
      COMPLEX(KIND=real_8), DIMENSION(:, :), POINTER     :: data_z
      INTEGER                                            :: blk, col, handle, row
      LOGICAL                                            :: any_sym, my_local, tr
      REAL(KIND=real_4), DIMENSION(:, :), POINTER        :: data_r
      REAL(KIND=real_8), DIMENSION(:, :), POINTER        :: data_d
      REAL(real_8)                                       :: fac
      TYPE(dbcsr_iterator)                               :: iter

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)

      my_local = .FALSE.
      IF (PRESENT(local)) my_local = local

      any_sym = dbcsr_get_matrix_type(matrix) .EQ. dbcsr_type_symmetric .OR. &
                dbcsr_get_matrix_type(matrix) .EQ. dbcsr_type_antisymmetric

      norm = 0.0_dp
      CALL dbcsr_iterator_start(iter, matrix)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         SELECT CASE (dbcsr_get_data_type(matrix))
         CASE (dbcsr_type_real_4)
            CALL dbcsr_iterator_next_block(iter, row, col, data_r, tr, blk)
            fac = 1.0_dp
            IF (any_sym .AND. row .NE. col) fac = 2.0_dp
            norm = norm + fac*SUM(data_r**2)
         CASE (dbcsr_type_real_8)
            CALL dbcsr_iterator_next_block(iter, row, col, data_d, tr, blk)
            fac = 1.0_dp
            IF (any_sym .AND. row .NE. col) fac = 2.0_dp
            norm = norm + fac*SUM(data_d**2)
         CASE (dbcsr_type_complex_4)
            CALL dbcsr_iterator_next_block(iter, row, col, data_c, tr, blk)
            fac = 1.0_dp
            IF (any_sym .AND. row .NE. col) &
               DBCSR_ABORT("Only nonsymmetric matrix so far")
            norm = norm + fac*REAL(SUM(CONJG(data_c)*data_c), KIND=real_8)
         CASE (dbcsr_type_complex_8)
            CALL dbcsr_iterator_next_block(iter, row, col, data_z, tr, blk)
            fac = 1.0_dp
            IF (any_sym .AND. row .NE. col) &
               DBCSR_ABORT("Only nonsymmetric matrix so far")
            norm = norm + fac*REAL(SUM(CONJG(data_z)*data_z), KIND=real_8)
         CASE DEFAULT
            DBCSR_ABORT("Wrong data type")
         END SELECT
      ENDDO
      CALL dbcsr_iterator_stop(iter)
      IF (.NOT. my_local) CALL mp_sum(norm, dbcsr_mp_group(dbcsr_distribution_mp(matrix%dist)))
      norm = SQRT(norm)

      CALL timestop(handle)

   END FUNCTION dbcsr_frobenius_norm

   SUBROUTINE dbcsr_sum_replicated(matrix)
      !! Sums blocks in a replicated dbcsr matrix, which has the same structure on all ranks.

      TYPE(dbcsr_type), INTENT(inout)                    :: matrix
         !! dbcsr matrix to operate on

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_sum_replicated'

      INTEGER                                            :: comm, handle, index_checksum, mynode, &
                                                            numnodes
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: all_checksums
      TYPE(dbcsr_mp_obj)                                 :: mp

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)
      mp = dbcsr_distribution_mp(dbcsr_distribution(matrix))
      comm = dbcsr_mp_group(mp)
      numnodes = dbcsr_mp_numnodes(mp)
      mynode = dbcsr_mp_mynode(mp)
      !
      ALLOCATE (all_checksums(numnodes))
      CALL dbcsr_index_checksum(matrix, index_checksum)
      CALL mp_allgather(index_checksum, all_checksums, comm)
      !
      IF (.NOT. ALL(all_checksums .EQ. index_checksum)) &
         DBCSR_ABORT("Replicated matrices do not all have the same index structure.")
      !
      SELECT CASE (dbcsr_data_get_type(matrix%data_area))
      CASE (dbcsr_type_real_4)
         CALL mp_sum(matrix%data_area%d%r_sp, comm)
      CASE (dbcsr_type_real_8)
         CALL mp_sum(matrix%data_area%d%r_dp, comm)
      CASE (dbcsr_type_complex_4)
         CALL mp_sum(matrix%data_area%d%c_sp, comm)
      CASE (dbcsr_type_complex_8)
         CALL mp_sum(matrix%data_area%d%c_dp, comm)
      CASE default
         DBCSR_ABORT("Incorrect data type")
      END SELECT
      !
      CALL timestop(handle)
   END SUBROUTINE dbcsr_sum_replicated

   FUNCTION dbcsr_block_in_limits(row, col, block_row_limits, block_column_limits)
      !! check if a block is not in the limits
      INTEGER, INTENT(in)                                :: row, col
      INTEGER, DIMENSION(2), INTENT(in), OPTIONAL        :: block_row_limits, block_column_limits
      LOGICAL                                            :: dbcsr_block_in_limits

      dbcsr_block_in_limits = .TRUE.
      IF (PRESENT(block_row_limits)) THEN
         IF (row .LT. block_row_limits(1)) dbcsr_block_in_limits = .FALSE.
         IF (row .GT. block_row_limits(2)) dbcsr_block_in_limits = .FALSE.
      ENDIF
      IF (PRESENT(block_column_limits)) THEN
         IF (col .LT. block_column_limits(1)) dbcsr_block_in_limits = .FALSE.
         IF (col .GT. block_column_limits(2)) dbcsr_block_in_limits = .FALSE.
      ENDIF
   END FUNCTION dbcsr_block_in_limits

   SUBROUTINE dbcsr_get_info(matrix, nblkrows_total, nblkcols_total, &
                             nfullrows_total, nfullcols_total, &
                             nblkrows_local, nblkcols_local, &
                             nfullrows_local, nfullcols_local, &
                             my_prow, my_pcol, &
                             local_rows, local_cols, proc_row_dist, proc_col_dist, &
                             row_blk_size, col_blk_size, row_blk_offset, col_blk_offset, distribution, name, data_area, &
                             matrix_type, data_type, group)
      !! Gets information about a matrix

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! matrix to query
      INTEGER, INTENT(OUT), OPTIONAL :: nblkrows_total, nblkcols_total, nfullrows_total, &
                                        nfullcols_total, nblkrows_local, nblkcols_local, nfullrows_local, nfullcols_local, &
                                        my_prow, my_pcol
      INTEGER, DIMENSION(:), OPTIONAL, POINTER :: local_rows, local_cols, proc_row_dist, &
                                                  proc_col_dist, row_blk_size, col_blk_size, row_blk_offset, col_blk_offset
      TYPE(dbcsr_distribution_obj), INTENT(OUT), &
         OPTIONAL                                        :: distribution
         !! the data distribution of the matrix
      CHARACTER(len=*), INTENT(OUT), OPTIONAL            :: name
         !! matrix name
      TYPE(dbcsr_data_obj), INTENT(OUT), OPTIONAL        :: data_area
         !! data_area
      CHARACTER, OPTIONAL                                :: matrix_type
         !! matrix type (regular, symmetric, see dbcsr_types.F for values)
      INTEGER, OPTIONAL                                  :: data_type
         !! data type (single/double precision real/complex)
      INTEGER, INTENT(OUT), OPTIONAL                     :: group

!   ---------------------------------------------------------------------------

      !vw avoid massive printing of warnings
      !DBCSR_WARN("Invalid matrix")
      IF (PRESENT(nblkrows_total)) nblkrows_total = matrix%nblkrows_total
      IF (PRESENT(nblkcols_total)) nblkcols_total = matrix%nblkcols_total
      IF (PRESENT(nfullrows_total)) nfullrows_total = matrix%nfullrows_total
      IF (PRESENT(nfullcols_total)) nfullcols_total = matrix%nfullcols_total
      IF (PRESENT(nblkrows_local)) nblkrows_local = matrix%nblkrows_local
      IF (PRESENT(nblkcols_local)) nblkcols_local = matrix%nblkcols_local
      IF (PRESENT(nfullrows_local)) nfullrows_local = matrix%nfullrows_local
      IF (PRESENT(nfullcols_local)) nfullcols_local = matrix%nfullcols_local
      IF (PRESENT(row_blk_size)) row_blk_size => array_data(matrix%row_blk_size)
      IF (PRESENT(col_blk_size)) col_blk_size => array_data(matrix%col_blk_size)
      IF (PRESENT(row_blk_offset)) row_blk_offset => array_data(matrix%row_blk_offset)
      IF (PRESENT(col_blk_offset)) col_blk_offset => array_data(matrix%col_blk_offset)
      IF (PRESENT(distribution)) distribution = matrix%dist
      IF (PRESENT(name)) name = matrix%name
      IF (PRESENT(data_area)) data_area = matrix%data_area
      IF (PRESENT(data_type)) data_type = matrix%data_type
      IF (PRESENT(local_rows)) local_rows => dbcsr_distribution_local_rows(matrix%dist)
      IF (PRESENT(local_cols)) local_cols => dbcsr_distribution_local_cols(matrix%dist)
      IF (PRESENT(proc_row_dist)) proc_row_dist => dbcsr_distribution_row_dist(matrix%dist)
      IF (PRESENT(proc_col_dist)) proc_col_dist => dbcsr_distribution_col_dist(matrix%dist)
      IF (PRESENT(my_prow)) my_prow = dbcsr_mp_myprow(dbcsr_distribution_mp(matrix%dist))
      IF (PRESENT(my_pcol)) my_pcol = dbcsr_mp_mypcol(dbcsr_distribution_mp(matrix%dist))
      IF (PRESENT(matrix_type)) matrix_type = dbcsr_get_matrix_type(matrix)

      IF (PRESENT(group)) group = dbcsr_mp_group(matrix%dist%d%mp_env) ! a shortcut
      !IF (PRESENT(matrix_type)) THEN
      !   matrix_type = dbcsr_get_matrix_type(matrix)
      !   IF (matrix_type .EQ. dbcsr_type_invalid) &
      !      DBCSR_ABORT("Incorrect symmetry")
      !ENDIF
   END SUBROUTINE dbcsr_get_info

   FUNCTION dbcsr_may_be_dense(matrix, occ_thresh) RESULT(may_be_dense)
      !! Returns whether the matrix could be represented in a dense form

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! matrix
      REAL(real_8), INTENT(in)                           :: occ_thresh
      LOGICAL                                            :: may_be_dense
         !! use the mutable and not append-only working structures

      REAL(real_8)                                       :: occ

!   ---------------------------------------------------------------------------

      occ = dbcsr_get_occupation(matrix)
      may_be_dense = .NOT. (occ .LT. occ_thresh)
      ! make sure every proc sees the same
      CALL mp_sum(may_be_dense, dbcsr_mp_group(dbcsr_distribution_mp(matrix%dist)))
   END FUNCTION dbcsr_may_be_dense

   FUNCTION dbcsr_get_occupation(matrix) RESULT(occupation)
      !! Returns the occupation of the matrix

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! matrix from which to get the occupation
      REAL(KIND=real_8)                                  :: occupation

      INTEGER                                            :: nfullcols, nfullrows
      INTEGER(KIND=int_8)                                :: nze_global
      INTEGER, DIMENSION(:), POINTER                     :: row_blk_size

      nze_global = matrix%nze
      CALL mp_sum(nze_global, dbcsr_mp_group(dbcsr_distribution_mp(matrix%dist)))

      nfullrows = dbcsr_nfullrows_total(matrix)
      nfullcols = dbcsr_nfullcols_total(matrix)

      row_blk_size => array_data(matrix%row_blk_size)

      IF (nfullrows .NE. 0 .AND. nfullcols .NE. 0) THEN
         IF (dbcsr_has_symmetry(matrix)) THEN
            IF (2*nze_global .EQ. &
                (INT(nfullrows, KIND=int_8)*INT(nfullrows + 1, KIND=int_8) + SUM(row_blk_size*(row_blk_size - 1)))) THEN
               occupation = 1.0_real_8
            ELSE
               occupation = 2.0_real_8*REAL(nze_global, real_8)/ &
                            (REAL(nfullrows, real_8)*REAL(nfullrows + 1, real_8) + &
                             SUM(REAL(row_blk_size, real_8)*REAL(row_blk_size - 1, real_8)))
            ENDIF
         ELSE
            IF (nze_global .EQ. INT(nfullrows, KIND=int_8)*INT(nfullcols, KIND=int_8)) THEN
               occupation = 1.0_real_8
            ELSE
               occupation = REAL(nze_global, real_8)/(REAL(nfullrows, real_8)*REAL(nfullcols, real_8))
            ENDIF
         ENDIF
      ELSE
         occupation = 0.0_real_8
      ENDIF
   END FUNCTION dbcsr_get_occupation

   SUBROUTINE dbcsr_clear(matrix)
      !! Clear a matrix (remove all blocks)
      TYPE(dbcsr_type), INTENT(INOUT)                       :: matrix
      TYPE(dbcsr_type)                                      :: matrix_tmp

      CALL dbcsr_create(matrix_tmp, matrix)
      CALL dbcsr_release(matrix)
      matrix = matrix_tmp

   END SUBROUTINE

   SUBROUTINE dbcsr_trace_sd(matrix_a, trace)
      !! Trace of DBCSR matrices

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_a
         !! DBCSR matrices
      REAL(kind=real_8), INTENT(INOUT)                   :: trace
         !! the trace of the product of the matrices

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_trace_sd'

      INTEGER                                            :: handle
      REAL(kind=real_4)                                  :: trace_4

      CALL timeset(routineN, handle)
      IF (dbcsr_get_data_type(matrix_a) .EQ. dbcsr_type_real_8) THEN
         CALL dbcsr_trace_d(matrix_a, trace)
      ELSEIF (dbcsr_get_data_type(matrix_a) .EQ. dbcsr_type_real_4) THEN
         trace_4 = 0.0_real_4
         CALL dbcsr_trace_s(matrix_a, trace_4)
         trace = REAL(trace_4, real_8)
      ELSE
         DBCSR_ABORT("Invalid combination of data type, NYI")
      ENDIF
      CALL timestop(handle)
   END SUBROUTINE dbcsr_trace_sd

   SUBROUTINE dbcsr_dot_sd(matrix_a, matrix_b, trace)
      !! Dot product of DBCSR matrices
      !! \result    the dot product of the matrices

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_a, matrix_b
         !! DBCSR matrices
         !! DBCSR matrices
      REAL(kind=real_8), INTENT(INOUT)                   :: trace

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_dot_sd'

      INTEGER                                            :: handle
      REAL(kind=real_4)                                  :: trace_4

      CALL timeset(routineN, handle)
      IF (dbcsr_get_data_type(matrix_a) .EQ. dbcsr_type_real_8 .AND. &
          dbcsr_get_data_type(matrix_b) .EQ. dbcsr_type_real_8) THEN
         CALL dbcsr_dot_d(matrix_a, matrix_b, trace)
      ELSEIF (dbcsr_get_data_type(matrix_a) .EQ. dbcsr_type_real_4 .AND. &
              dbcsr_get_data_type(matrix_b) .EQ. dbcsr_type_real_4) THEN
         trace_4 = 0.0_real_4
         CALL dbcsr_dot_s(matrix_a, matrix_b, trace_4)
         trace = REAL(trace_4, real_8)
      ELSE
         DBCSR_ABORT("Invalid combination of data type, NYI")
      ENDIF
      CALL timestop(handle)
   END SUBROUTINE dbcsr_dot_sd

#:include '../data/dbcsr.fypp'
#:for n, nametype1, base1, prec1, kind1, type1, dkind1 in inst_params_float
   SUBROUTINE dbcsr_trace_${nametype1}$ (matrix_a, trace)
      !! traces a DBCSR matrix

      TYPE(dbcsr_type), INTENT(IN)               :: matrix_a
         !! DBCSR matrix
      ${type1}$, INTENT(INOUT)                   :: trace
         !! the trace of the matrix

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_trace_${nametype1}$'

      INTEGER                                  :: a_blk, a_col, a_col_size, &
                                                  a_nze, a_row, a_row_size, i, &
                                                  mynode, error_handle
      INTEGER, DIMENSION(:), POINTER           :: col_blk_size, row_blk_size, &
                                                  row_dist, col_dist
      ${type1}$, DIMENSION(:), POINTER           :: a_data, data_p
      INTEGER, DIMENSION(:, :), POINTER         :: pgrid
      TYPE(dbcsr_distribution_obj)             :: dist

!   ---------------------------------------------------------------------------
      CALL timeset(routineN, error_handle)

      row_blk_size => array_data(matrix_a%row_blk_size)
      col_blk_size => array_data(matrix_a%col_blk_size)
      IF (dbcsr_get_data_type(matrix_a) /= ${dkind1}$) &
         DBCSR_ABORT("Incompatible data types")
      CALL dbcsr_get_data(matrix_a%data_area, data_p)
      dist = dbcsr_distribution(matrix_a)
      mynode = dbcsr_mp_mynode(dbcsr_distribution_mp(dist))
      pgrid => dbcsr_mp_pgrid(dbcsr_distribution_mp(dist))
      row_dist => dbcsr_distribution_row_dist(dist)
      col_dist => dbcsr_distribution_col_dist(dist)
      !
      ! let's go
      trace = REAL(0.0, ${kind1}$)
      DO a_row = 1, matrix_a%nblkrows_total
         a_row_size = row_blk_size(a_row)
         DO a_blk = matrix_a%row_p(a_row) + 1, matrix_a%row_p(a_row + 1)
            IF (a_blk .EQ. 0) CYCLE
            a_col = matrix_a%col_i(a_blk)
            IF (a_col .ne. a_row) CYCLE
            ! We must skip non-local blocks in a replicated matrix.
            IF (matrix_a%replication_type .NE. dbcsr_repl_full) THEN
               IF (mynode .NE. checker_square_proc(a_row, a_col, pgrid, row_dist, col_dist)) &
                  CYCLE
            ENDIF
            a_col_size = col_blk_size(a_col)
            IF (a_row_size .NE. a_col_size) &
               DBCSR_ABORT("is that a square matrix?")
            a_nze = a_row_size**2
            a_data => pointer_view(data_p, ABS(matrix_a%blk_p(a_blk)), &
                                   ABS(matrix_a%blk_p(a_blk)) + a_nze - 1)
            !data_a => matrix_a%data(ABS(matrix_a%blk_p(a_blk)):ABS(matrix_a%blk_p(a_blk))+a_nze-1)
            !
            ! let's trace the block
            DO i = 1, a_row_size
               trace = trace + a_data((i - 1)*a_row_size + i)
            ENDDO
         ENDDO ! a_col
      ENDDO ! a_row
      !
      ! summe
      CALL mp_sum(trace, dbcsr_mp_group(dbcsr_distribution_mp(matrix_a%dist)))

      CALL timestop(error_handle)
   END SUBROUTINE dbcsr_trace_${nametype1}$

   SUBROUTINE dbcsr_dot_${nametype1}$ (matrix_a, matrix_b, trace)
      !! Dot product of DBCSR matrices

      TYPE(dbcsr_type), INTENT(IN)               :: matrix_a, matrix_b
         !! DBCSR matrices
         !! DBCSR matrices
      ${type1}$, INTENT(INOUT)                   :: trace
         !! the trace of the product of the matrices

      INTEGER :: a_blk, a_col, a_col_size, a_row_size, b_blk, b_col_size, &
                 b_frst_blk, b_last_blk, b_row_size, nze, row, a_beg, a_end, b_beg, b_end
      CHARACTER                                :: matrix_a_type, matrix_b_type
      INTEGER, DIMENSION(:), POINTER           :: a_col_blk_size, &
                                                  a_row_blk_size, &
                                                  b_col_blk_size, b_row_blk_size
      ${type1}$                                  :: sym_fac, fac
      LOGICAL                                  :: found, matrix_a_symm, matrix_b_symm
      ${type1}$, DIMENSION(:), POINTER           :: a_data, b_data

!   ---------------------------------------------------------------------------

      IF (matrix_a%replication_type .NE. dbcsr_repl_none &
          .OR. matrix_b%replication_type .NE. dbcsr_repl_none) &
         DBCSR_ABORT("Trace of product of replicated matrices not yet possible.")

      sym_fac = REAL(1.0, ${kind1}$)
      matrix_a_type = dbcsr_get_matrix_type(matrix_a)
      matrix_b_type = dbcsr_get_matrix_type(matrix_b)
      matrix_a_symm = matrix_a_type == dbcsr_type_symmetric .OR. matrix_a_type == dbcsr_type_antisymmetric
      matrix_b_symm = matrix_b_type == dbcsr_type_symmetric .OR. matrix_b_type == dbcsr_type_antisymmetric

      IF (matrix_a_symm .AND. matrix_b_symm) sym_fac = REAL(2.0, ${kind1}$)

      ! tracing a symmetric with a general matrix is not implemented, as it would require communication of blocks
      IF (matrix_a_symm .NEQV. matrix_b_symm) &
         DBCSR_ABORT("Tracing general with symmetric matrix NYI")

      a_row_blk_size => array_data(matrix_a%row_blk_size)
      a_col_blk_size => array_data(matrix_a%col_blk_size)
      b_row_blk_size => array_data(matrix_b%row_blk_size)
      b_col_blk_size => array_data(matrix_b%col_blk_size)

      CALL dbcsr_get_data(matrix_a%data_area, a_data)
      CALL dbcsr_get_data(matrix_b%data_area, b_data)

      ! let's go
      trace = REAL(0.0, ${kind1}$)
      IF (matrix_a%nblkrows_total .NE. matrix_b%nblkrows_total) &
         DBCSR_ABORT("this combination of transpose is NYI")
      DO row = 1, matrix_a%nblkrows_total
         a_row_size = a_row_blk_size(row)
         b_row_size = b_row_blk_size(row)
         IF (a_row_size .NE. b_row_size) DBCSR_ABORT("matrices not consistent")
         b_blk = matrix_b%row_p(row) + 1
         b_frst_blk = matrix_b%row_p(row) + 1
         b_last_blk = matrix_b%row_p(row + 1)
         DO a_blk = matrix_a%row_p(row) + 1, matrix_a%row_p(row + 1)
            IF (matrix_a%blk_p(a_blk) .EQ. 0) CYCLE ! Deleted block
            a_col = matrix_a%col_i(a_blk)
            a_col_size = a_col_blk_size(a_col)
            !
            ! find the b_blk we assume here that the columns are ordered !
            CALL dbcsr_find_column(a_col, b_frst_blk, b_last_blk, matrix_b%col_i, &
                                   matrix_b%blk_p, b_blk, found)
            IF (found) THEN
               b_col_size = b_col_blk_size(a_col)
               IF (a_col_size .NE. b_col_size) DBCSR_ABORT("matrices not consistent")
               !
               nze = a_row_size*a_col_size
               !
               IF (nze .GT. 0) THEN
                  !
                  ! let's trace the blocks
                  a_beg = ABS(matrix_a%blk_p(a_blk))
                  a_end = a_beg + nze - 1
                  b_beg = ABS(matrix_b%blk_p(b_blk))
                  b_end = b_beg + nze - 1
                  fac = REAL(1.0, ${kind1}$)
                  IF (row .NE. a_col) fac = sym_fac

                  trace = trace + fac*SUM(a_data(a_beg:a_end)*b_data(b_beg:b_end))

               ENDIF
            ENDIF
         ENDDO ! a_col
      ENDDO ! a_row
      !
      ! sum
      CALL mp_sum(trace, dbcsr_mp_group(dbcsr_distribution_mp(matrix_a%dist)))

   END SUBROUTINE dbcsr_dot_${nametype1}$

   SUBROUTINE dbcsr_scale_${nametype1}$ (matrix_a, alpha_scalar, last_column)
      !! Interface for matrix scaling by a scalar
      TYPE(dbcsr_type), INTENT(INOUT)           :: matrix_a
      ${type1}$, INTENT(IN)                      :: alpha_scalar
      INTEGER, INTENT(IN), OPTIONAL            :: last_column

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_scale_${nametype1}$'

      INTEGER                                  :: error_handler
      TYPE(dbcsr_scalar_type)                  :: sc

      sc = dbcsr_scalar(alpha_scalar)
      CALL dbcsr_scalar_fill_all(sc)
      sc%data_type = dbcsr_get_data_type(matrix_a)
      CALL timeset(routineN, error_handler)
      IF (PRESENT(last_column)) THEN
         CALL dbcsr_scale_anytype(matrix_a, &
                                  alpha_scalar=sc, &
                                  limits=(/0, 0, 0, last_column/))
      ELSE
         CALL dbcsr_scale_anytype(matrix_a, alpha_scalar=sc)
      ENDIF
      CALL timestop(error_handler)
   END SUBROUTINE dbcsr_scale_${nametype1}$

   SUBROUTINE dbcsr_scale_by_vector_${nametype1}$ (matrix_a, alpha, side)
      !! Interface for matrix scaling by a vector
      TYPE(dbcsr_type), INTENT(INOUT)            :: matrix_a
      ${type1}$, DIMENSION(:), INTENT(IN), TARGET, CONTIGUOUS :: alpha
      CHARACTER(LEN=*), INTENT(IN)              :: side
      ${type1}$, DIMENSION(:), POINTER, CONTIGUOUS :: tmp_p
      TYPE(dbcsr_data_obj)                      :: enc_alpha_vec

      CALL dbcsr_data_init(enc_alpha_vec)
      CALL dbcsr_data_new(enc_alpha_vec, ${dkind1}$)
      tmp_p => alpha
      CALL dbcsr_data_set_pointer(enc_alpha_vec, tmp_p)
      CALL dbcsr_scale_by_vector_anytype(matrix_a, enc_alpha_vec, side)
      CALL dbcsr_data_clear_pointer(enc_alpha_vec)
      CALL dbcsr_data_release(enc_alpha_vec)
   END SUBROUTINE dbcsr_scale_by_vector_${nametype1}$

   SUBROUTINE dbcsr_set_${nametype1}$ (matrix, alpha)
      !! Interface for dbcsr_set
      TYPE(dbcsr_type), INTENT(INOUT)           :: matrix
      ${type1}$, INTENT(IN)                      :: alpha

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_set'

      INTEGER                                            :: col, handle, row
      TYPE(dbcsr_iterator)                               :: iter
      ${type1}$, DIMENSION(:, :), POINTER                   :: block
      LOGICAL                                            :: tr

      CALL timeset(routineN, handle)

      IF (alpha == ${zero1[n]}$) THEN
         CALL dbcsr_zero(matrix)
      ELSE
         IF (dbcsr_get_data_type(matrix) /= ${dkind1}$) &
            DBCSR_ABORT("Incompatible data types")

         !TODO: could be speedup by direct assignment to data_area, similar to dbcsr_zero()
         CALL dbcsr_iterator_start(iter, matrix)
         DO WHILE (dbcsr_iterator_blocks_left(iter))
            CALL dbcsr_iterator_next_block(iter, row, col, block, tr)
            block(:, :) = alpha
         ENDDO
         CALL dbcsr_iterator_stop(iter)
      ENDIF

      CALL timestop(handle)
   END SUBROUTINE dbcsr_set_${nametype1}$

   SUBROUTINE dbcsr_filter_${nametype1}$ (matrix, eps, method, use_absolute, &
                                          filter_diag)
      TYPE(dbcsr_type), INTENT(INOUT)           :: matrix
      ${type1}$, INTENT(IN)                      :: eps
      INTEGER, INTENT(IN), OPTIONAL            :: method
      LOGICAL, INTENT(in), OPTIONAL            :: use_absolute, filter_diag
      CALL dbcsr_filter_anytype(matrix, dbcsr_scalar(eps), method, &
                                use_absolute, filter_diag)
   END SUBROUTINE dbcsr_filter_${nametype1}$

   SUBROUTINE dbcsr_set_diag_${nametype1}$ (matrix, diag)
      TYPE(dbcsr_type), INTENT(INOUT)            :: matrix
      ${type1}$, DIMENSION(:), INTENT(IN)          :: diag

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_set_diag'

      INTEGER                                            :: icol, irow, row_offset, handle, i
      LOGICAL                                            :: tr
      TYPE(dbcsr_iterator)                               :: iter
      ${type1}$, DIMENSION(:, :), POINTER                   :: block

      CALL timeset(routineN, handle)

      IF (dbcsr_get_data_type(matrix) /= ${dkind1}$) &
         DBCSR_ABORT("Incompatible data types")

      IF (dbcsr_nfullrows_total(matrix) /= SIZE(diag)) &
         DBCSR_ABORT("Diagonal has wrong size")

      IF (.NOT. array_equality(matrix%row_blk_offset, matrix%col_blk_offset)) &
         DBCSR_ABORT("matrix not quadratic")

      CALL dbcsr_iterator_start(iter, matrix)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, irow, icol, block, tr, row_offset=row_offset)
         IF (irow /= icol) CYCLE

         IF (sIZE(block, 1) /= sIZE(block, 2)) &
            DBCSR_ABORT("Diagonal block non-squared")

         DO i = 1, sIZE(block, 1)
            block(i, i) = diag(row_offset + i - 1)
         END DO
      ENDDO
      CALL dbcsr_iterator_stop(iter)

      CALL timestop(handle)
   END SUBROUTINE dbcsr_set_diag_${nametype1}$

   SUBROUTINE dbcsr_get_diag_${nametype1}$ (matrix, diag)
      TYPE(dbcsr_type), INTENT(IN)               :: matrix
      ${type1}$, DIMENSION(:), INTENT(OUT)         :: diag

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_diag'

      INTEGER                                            :: icol, irow, row_offset, handle, i
      LOGICAL                                            :: tr
      TYPE(dbcsr_iterator)                               :: iter
      ${type1}$, DIMENSION(:, :), POINTER                   :: block

      CALL timeset(routineN, handle)

      IF (dbcsr_get_data_type(matrix) /= ${dkind1}$) &
         DBCSR_ABORT("Incompatible data types")

      IF (dbcsr_nfullrows_total(matrix) /= SIZE(diag)) &
         DBCSR_ABORT("Diagonal has wrong size")

      IF (.NOT. array_equality(matrix%row_blk_offset, matrix%col_blk_offset)) &
         DBCSR_ABORT("matrix not quadratic")

      diag(:) = ${zero1[n]}$

      CALL dbcsr_iterator_start(iter, matrix)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, irow, icol, block, tr, row_offset=row_offset)
         IF (irow /= icol) CYCLE

         IF (sIZE(block, 1) /= sIZE(block, 2)) &
            DBCSR_ABORT("Diagonal block non-squared")

         DO i = 1, sIZE(block, 1)
            diag(row_offset + i - 1) = block(i, i)
         END DO
      ENDDO
      CALL dbcsr_iterator_stop(iter)

      CALL timestop(handle)
   END SUBROUTINE dbcsr_get_diag_${nametype1}$

   SUBROUTINE dbcsr_add_on_diag_${nametype1}$ (matrix, alpha)
      !! add a constant to the diagonal of a matrix

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
         !! DBCSR matrix
      ${type1}$, INTENT(IN)                                :: alpha
         !! scalar

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_add_on_diag'

      INTEGER                                            :: handle, mynode, node, irow, i, row_size
      LOGICAL                                            :: found, tr
      ${type1}$, DIMENSION(:, :), POINTER                   :: block

      CALL timeset(routineN, handle)

      IF (dbcsr_get_data_type(matrix) /= ${dkind1}$) &
         DBCSR_ABORT("Incompatible data types")

      IF (.NOT. array_equality(matrix%row_blk_offset, matrix%col_blk_offset)) &
         DBCSR_ABORT("matrix not quadratic")

      mynode = dbcsr_mp_mynode(dbcsr_distribution_mp(dbcsr_distribution(matrix)))

      CALL dbcsr_work_create(matrix, work_mutable=.TRUE.)

      DO irow = 1, dbcsr_nblkrows_total(matrix)
         CALL dbcsr_get_stored_coordinates(matrix, irow, irow, node)
         IF (node /= mynode) CYCLE

         CALL dbcsr_get_block_p(matrix, irow, irow, block, tr, found, row_size=row_size)
         IF (.NOT. found) THEN
            ALLOCATE (block(row_size, row_size))
            block(:, :) = ${zero1[n]}$
         ENDIF

         DO i = 1, row_size
            block(i, i) = block(i, i) + alpha
         END DO

         IF (.NOT. found) THEN
            CALL dbcsr_put_block(matrix, irow, irow, block)
            DEALLOCATE (block)
         ENDIF
      ENDDO

      CALL dbcsr_finalize(matrix)
      CALL timestop(handle)
   END SUBROUTINE dbcsr_add_on_diag_${nametype1}$

   SUBROUTINE dbcsr_update_contiguous_blocks_${nametype1}$ (matrix_a, matrix_b, first_lb_a, first_lb_b, nze, &
                                                            do_scale, my_beta_scalar, found, iw)
      !! Low level function to sum contiguous chunks of blocks of the matrices (matrix_a = matrix_a + beta*matrix_b)

      TYPE(dbcsr_type), INTENT(INOUT)                         :: matrix_a
         !! DBCSR matrix
      TYPE(dbcsr_type), INTENT(IN)                            :: matrix_b
         !! DBCSR matrix
      TYPE(dbcsr_scalar_type), INTENT(IN)                     :: my_beta_scalar
      INTEGER, INTENT(IN)                                     :: first_lb_a, first_lb_b, nze, iw
      LOGICAL, INTENT(IN)                                     :: found, do_scale

      INTEGER                                                 :: ub_a, ub_b

      ub_a = first_lb_a + nze - 1
      ub_b = first_lb_b + nze - 1

      IF (found) THEN
         IF (do_scale) THEN
            CALL ${nametype1}$axpy(nze, my_beta_scalar%${base1}$_${prec1}$, &
                                   matrix_b%data_area%d%${base1}$_${prec1}$ (first_lb_b:ub_b), 1, &
                                   matrix_a%data_area%d%${base1}$_${prec1}$ (first_lb_a:ub_a), 1)
         ELSE
            matrix_a%data_area%d%${base1}$_${prec1}$ (first_lb_a:ub_a) = &
               matrix_a%data_area%d%${base1}$_${prec1}$ (first_lb_a:ub_a) + &
               matrix_b%data_area%d%${base1}$_${prec1}$ (first_lb_b:ub_b)
         ENDIF
      ELSE
         IF (do_scale) THEN
            matrix_a%wms(iw)%data_area%d%${base1}$_${prec1}$ (first_lb_a:ub_a) = &
               my_beta_scalar%${base1}$_${prec1}$* &
               matrix_b%data_area%d%${base1}$_${prec1}$ (first_lb_b:ub_b)
         ELSE
            matrix_a%wms(iw)%data_area%d%${base1}$_${prec1}$ (first_lb_a:ub_a) = &
               matrix_b%data_area%d%${base1}$_${prec1}$ (first_lb_b:ub_b)
         ENDIF
      ENDIF
   END SUBROUTINE dbcsr_update_contiguous_blocks_${nametype1}$

   SUBROUTINE dbcsr_add_anytype_${nametype1}$ (matrix_a, matrix_b, iter, iw, do_scale, &
                                               my_beta_scalar, my_flop)
      !! Low level function to sum two matrices (matrix_a = matrix_a + beta*matrix_b

      TYPE(dbcsr_type), INTENT(INOUT)                         :: matrix_a
         !! DBCSR matrix
      TYPE(dbcsr_type), INTENT(IN)                            :: matrix_b
         !! DBCSR matrix
      TYPE(dbcsr_iterator), INTENT(INOUT)                     :: iter
      INTEGER, INTENT(IN)                                     :: iw
      LOGICAL, INTENT(IN)                                     :: do_scale
      TYPE(dbcsr_scalar_type), INTENT(IN)                     :: my_beta_scalar
      INTEGER(KIND=int_8), INTENT(INOUT)                      :: my_flop

      INTEGER                                                 :: row, col, row_size, col_size, &
                                                                 nze, tot_nze, blk, &
                                                                 lb_a, first_lb_a, lb_a_val, &
                                                                 lb_b, first_lb_b
      INTEGER, DIMENSION(2)                                   :: lb_row_blk
      LOGICAL                                                 :: was_found, found, tr

      ! some start values
      lb_row_blk(:) = 0
      first_lb_a = matrix_a%wms(iw)%datasize + 1
      first_lb_b = 0
      tot_nze = 0
      !
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, row, col, blk, tr, lb_b, row_size, col_size)
         nze = row_size*col_size
         IF (nze .LE. 0) CYCLE
         IF (lb_row_blk(1) .LT. row) THEN
            lb_row_blk(1) = row
            lb_row_blk(2) = matrix_a%row_p(row) + 1
         ENDIF
         ! get b-block index
         lb_b = ABS(lb_b)
         CALL dbcsr_find_column(col, lb_row_blk(2), matrix_a%row_p(row + 1), matrix_a%col_i, matrix_a%blk_p, blk, found)
         lb_row_blk(2) = blk + 1
         ! get index of a-block lb_a whether found (from matrix_a) or not (from workspace array)
         IF (found) THEN
            my_flop = my_flop + nze*2
            lb_a = ABS(matrix_a%blk_p(blk))
         ELSE
            lb_a = matrix_a%wms(iw)%datasize + 1
            lb_a_val = lb_a
            IF (tr) lb_a_val = -lb_a
            matrix_a%wms(iw)%lastblk = matrix_a%wms(iw)%lastblk + 1
            matrix_a%wms(iw)%row_i(matrix_a%wms(iw)%lastblk) = row
            matrix_a%wms(iw)%col_i(matrix_a%wms(iw)%lastblk) = col
            matrix_a%wms(iw)%blk_p(matrix_a%wms(iw)%lastblk) = lb_a_val
            matrix_a%wms(iw)%datasize = matrix_a%wms(iw)%datasize + nze
         ENDIF
         ! at the first iteration we skip this and go directly to initialization after
         IF (first_lb_b .NE. 0) THEN
            ! if found status is the same as before then probably we are in contiguous blocks
            IF ((found .EQV. was_found) .AND. &
                (first_lb_b + tot_nze .EQ. lb_b) .AND. &
                (first_lb_a + tot_nze) .EQ. lb_a) THEN
               tot_nze = tot_nze + nze
               CYCLE
            ENDIF
            ! save block chunk
            CALL dbcsr_update_contiguous_blocks_${nametype1}$ (matrix_a, matrix_b, first_lb_a, first_lb_b, tot_nze, &
                                                               do_scale, my_beta_scalar, was_found, iw)
         ENDIF
         !
         first_lb_a = lb_a
         first_lb_b = lb_b
         tot_nze = nze
         was_found = found
      ENDDO

      ! save the last block or chunk of blocks
      IF (first_lb_b .NE. 0) THEN
         call dbcsr_update_contiguous_blocks_${nametype1}$ (matrix_a, matrix_b, first_lb_a, first_lb_b, tot_nze, &
                                                            do_scale, my_beta_scalar, was_found, iw)
      ENDIF

   END SUBROUTINE dbcsr_add_anytype_${nametype1}$
#:endfor

END MODULE dbcsr_operations
