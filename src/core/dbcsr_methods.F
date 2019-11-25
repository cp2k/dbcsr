!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_methods
   !! Base methods on DBCSR data structures
   USE dbcsr_array_types, ONLY: array_data, &
                                array_release
   USE dbcsr_btree, ONLY: btree_delete, &
                          btree_new
   USE dbcsr_data_methods, ONLY: dbcsr_data_get_size, &
                                 dbcsr_data_release
   USE dbcsr_kinds, ONLY: default_string_length
   USE dbcsr_mpiwrap, ONLY: mp_comm_free
   USE dbcsr_ptr_util, ONLY: memory_deallocate
   USE dbcsr_types, ONLY: &
      dbcsr_1d_array_type, dbcsr_2d_array_type, dbcsr_data_obj, dbcsr_distribution_obj, &
      dbcsr_imagedistribution_obj, dbcsr_imagedistribution_type, dbcsr_memtype_type, &
      dbcsr_mp_obj, dbcsr_mutable_obj, dbcsr_type, dbcsr_type_antihermitian, &
      dbcsr_type_antisymmetric, dbcsr_type_complex_4, dbcsr_type_complex_8, &
      dbcsr_type_hermitian, dbcsr_type_invalid, dbcsr_type_no_symmetry, dbcsr_type_real_4, &
      dbcsr_type_real_8, dbcsr_type_symmetric, dbcsr_work_type
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads
   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_methods'

   INTEGER, PUBLIC, SAVE :: dbcsr_matrix_counter = 111111

   PUBLIC :: dbcsr_release
   PUBLIC :: dbcsr_valid_index
   PUBLIC :: dbcsr_release_locals
   PUBLIC :: dbcsr_distribution, &
             dbcsr_get_matrix_type, dbcsr_get_data_type, dbcsr_get_replication_type, &
             dbcsr_row_block_sizes, dbcsr_col_block_sizes, &
             dbcsr_nblkrows_total, dbcsr_nblkcols_total, dbcsr_nfullrows_total, &
             dbcsr_nfullcols_total, dbcsr_nblkcols_local, dbcsr_nblkrows_local, &
             dbcsr_max_row_size, dbcsr_max_col_size, &
             dbcsr_get_index_memory_type, dbcsr_get_data_memory_type, &
             dbcsr_name, dbcsr_setname, dbcsr_get_data_size, &
             dbcsr_use_mutable, dbcsr_wm_use_mutable, dbcsr_has_symmetry, &
             dbcsr_get_nze, dbcsr_nfullrows_local, dbcsr_nfullcols_local
   PUBLIC :: dbcsr_get_data_size_used
   PUBLIC :: dbcsr_col_block_offsets, dbcsr_row_block_offsets
   PUBLIC :: dbcsr_data_area
   PUBLIC :: dbcsr_get_num_blocks

   PUBLIC :: dbcsr_blk_row_size, dbcsr_blk_column_size, &
             dbcsr_blk_row_offset, dbcsr_blk_col_offset

   PUBLIC :: dbcsr_destroy_array
   PUBLIC :: dbcsr_image_dist_init, dbcsr_image_dist_hold, dbcsr_image_dist_release

   PUBLIC :: dbcsr_mutable_init, dbcsr_mutable_new, dbcsr_mutable_destroy, &
             dbcsr_mutable_release, &
             dbcsr_mutable_instantiated

   PUBLIC :: dbcsr_distribution_release
   PUBLIC :: dbcsr_mp_release, dbcsr_mp_grid_remove

   ! For the 1-D and 2-D arrays

   INTERFACE dbcsr_destroy_array
      MODULE PROCEDURE dbcsr_destroy_1d_array, dbcsr_destroy_2d_array
   END INTERFACE

CONTAINS

   PURE FUNCTION dbcsr_valid_index(matrix) RESULT(valid_index)
      !! Returns whether the index structure of the matrix is valid.

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! verify index validity of this matrix
      LOGICAL                                            :: valid_index
         !! index validity

      valid_index = matrix%valid
   END FUNCTION dbcsr_valid_index

   RECURSIVE SUBROUTINE dbcsr_release(matrix)
      !! Releases a reference for a DBCSR matrix
      !! If there are no references left, the matrix is destroyed.

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
         !! DBCSR matrix

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_release', routineP = moduleN//':'//routineN

!   ---------------------------------------------------------------------------

      matrix%refcount = matrix%refcount - 1
      IF (matrix%refcount .EQ. 0) THEN
         CALL dbcsr_destroy(matrix)
      ENDIF
   END SUBROUTINE dbcsr_release

   RECURSIVE SUBROUTINE dbcsr_destroy(matrix, force)
      !! Deallocates and destroys a matrix.

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
         !! matrix
      LOGICAL, INTENT(IN), OPTIONAL                      :: force
         !! force deallocation

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_destroy', routineP = moduleN//':'//routineN

      INTEGER                                            :: error_handle
      LOGICAL                                            :: force_all

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, error_handle)
      force_all = .FALSE.
      IF (PRESENT(force)) force_all = force
      IF (.NOT. force_all .AND. matrix%refcount .NE. 0) &
         DBCSR_WARN("You should not destroy referenced matrix.")
      IF (force_all .AND. matrix%refcount .GT. 1) &
         DBCSR_WARN("You should not destroy referenced matrix.")
      IF (force_all .OR. matrix%refcount .EQ. 0) THEN
         IF (ASSOCIATED(matrix%wms)) &
            DBCSR_WARN("Destroying unfinalized matrix")
         IF (ASSOCIATED(matrix%index)) THEN
            CALL memory_deallocate(matrix%index, matrix%index_memory_type)
         ENDIF
         CALL dbcsr_data_release(matrix%data_area)
         CALL array_release(matrix%row_blk_size)
         CALL array_release(matrix%col_blk_size)
         CALL array_release(matrix%row_blk_offset)
         CALL array_release(matrix%col_blk_offset)
         CALL dbcsr_distribution_release(matrix%dist)
         CALL dbcsr_release_locals(matrix)
         matrix%valid = .FALSE.
         matrix%refcount = 0
      ENDIF
      CALL timestop(error_handle)
   END SUBROUTINE dbcsr_destroy

   SUBROUTINE dbcsr_distribution_release(dist)
      !! Releases and potentially destroys a distribution
      TYPE(dbcsr_distribution_obj), INTENT(INOUT)        :: dist

!   ---------------------------------------------------------------------------

      IF (ASSOCIATED(dist%d)) THEN
         dist%d%refcount = dist%d%refcount - 1
         IF (dist%d%refcount .EQ. 0) THEN
            CALL array_release(dist%d%row_dist_block)
            CALL array_release(dist%d%col_dist_block)
            CALL array_release(dist%d%local_rows)
            CALL array_release(dist%d%local_cols)
            CALL dbcsr_mp_release(dist%d%mp_env)
            IF (dist%d%has_thread_dist) &
               CALL array_release(dist%d%thread_dist)
            CALL array_release(dist%d%row_map)
            CALL array_release(dist%d%col_map)
            CALL dbcsr_dist_release_locals(dist)
            DEALLOCATE (dist%d)
         ENDIF
      ENDIF
   END SUBROUTINE dbcsr_distribution_release

   SUBROUTINE dbcsr_dist_release_locals(dist)
      TYPE(dbcsr_distribution_obj), INTENT(INOUT)        :: dist

      INTEGER                                            :: i

      IF (dist%d%has_other_l_rows) THEN
         DO i = LBOUND(dist%d%other_l_rows, 1), UBOUND(dist%d%other_l_rows, 1)
            CALL array_release(dist%d%other_l_rows(i))
         END DO
         DEALLOCATE (dist%d%other_l_rows)
      ENDIF
      IF (dist%d%has_other_l_cols) THEN
         DO i = LBOUND(dist%d%other_l_cols, 1), UBOUND(dist%d%other_l_cols, 1)
            CALL array_release(dist%d%other_l_cols(i))
         END DO
         DEALLOCATE (dist%d%other_l_cols)
      ENDIF
      IF (dist%d%has_global_row_map) THEN
         CALL array_release(dist%d%global_row_map)
      ENDIF
      IF (dist%d%has_global_col_map) THEN
         CALL array_release(dist%d%global_col_map)
      ENDIF
      dist%d%has_other_l_rows = .FALSE.
      dist%d%has_other_l_cols = .FALSE.
      dist%d%has_global_row_map = .FALSE.
      dist%d%has_global_col_map = .FALSE.
   END SUBROUTINE dbcsr_dist_release_locals

   SUBROUTINE dbcsr_mp_release(mp_env)
      !! Releases and potentially destroys an mp_env

      TYPE(dbcsr_mp_obj), INTENT(INOUT)                  :: mp_env
         !! multiprocessor environment

!   ---------------------------------------------------------------------------

      IF (ASSOCIATED(mp_env%mp)) THEN
         mp_env%mp%refcount = mp_env%mp%refcount - 1
         IF (mp_env%mp%refcount .LE. 0) THEN
            CALL dbcsr_mp_grid_remove(mp_env)
            DEALLOCATE (mp_env%mp%pgrid)
            DEALLOCATE (mp_env%mp)
         ENDIF
      ENDIF
   END SUBROUTINE dbcsr_mp_release

   SUBROUTINE dbcsr_mp_grid_remove(mp_env)
      !! Removes an MPI cartesian process grid

      TYPE(dbcsr_mp_obj), INTENT(INOUT)                  :: mp_env
         !! multiprocessor environment

      IF (mp_env%mp%subgroups_defined) THEN
         CALL mp_comm_free(mp_env%mp%prow_group)
         CALL mp_comm_free(mp_env%mp%pcol_group)
      ENDIF
   END SUBROUTINE dbcsr_mp_grid_remove

   SUBROUTINE dbcsr_release_locals(matrix)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix

      IF (matrix%has_local_rows) &
         CALL array_release(matrix%local_rows)
      IF (matrix%has_global_rows) &
         CALL array_release(matrix%global_rows)
      IF (matrix%has_local_cols) &
         CALL array_release(matrix%local_cols)
      IF (matrix%has_global_cols) &
         CALL array_release(matrix%global_cols)
      matrix%has_local_rows = .FALSE.
      matrix%has_global_rows = .FALSE.
      matrix%has_local_cols = .FALSE.
      matrix%has_global_cols = .FALSE.
   END SUBROUTINE dbcsr_release_locals

!  SUBROUTINE dbcsr_release_vlocals (matrix)
!    TYPE(dbcsr_type), INTENT(INOUT)           :: matrix
!
!    IF (matrix%has_local_vrows) &
!         CALL array_release (matrix%local_vrows)
!    IF (matrix%has_global_vrows) &
!         CALL array_release (matrix%global_vrows)
!    IF (matrix%has_local_vcols) &
!         CALL array_release (matrix%local_vcols)
!    IF (matrix%has_global_vcols) &
!         CALL array_release (matrix%global_vcols)
!    matrix%has_local_vrows  = .FALSE.
!    matrix%has_global_vrows = .FALSE.
!    matrix%has_local_vcols  = .FALSE.
!    matrix%has_global_vcols = .FALSE.
!  END SUBROUTINE dbcsr_release_vlocals
!

! Pertaining to the dbcsr matrix.

   FUNCTION dbcsr_nblkrows_total(matrix) RESULT(nblkrows_total)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: nblkrows_total

      nblkrows_total = matrix%nblkrows_total
   END FUNCTION dbcsr_nblkrows_total

   FUNCTION dbcsr_nblkcols_total(matrix) RESULT(nblkcols_total)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: nblkcols_total

      nblkcols_total = matrix%nblkcols_total
   END FUNCTION dbcsr_nblkcols_total
   FUNCTION dbcsr_nfullrows_total(matrix) RESULT(nfullrows_total)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: nfullrows_total

      nfullrows_total = matrix%nfullrows_total
   END FUNCTION dbcsr_nfullrows_total
   FUNCTION dbcsr_nfullcols_total(matrix) RESULT(nfullcols_total)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: nfullcols_total

      nfullcols_total = matrix%nfullcols_total
   END FUNCTION dbcsr_nfullcols_total
   FUNCTION dbcsr_nblkrows_local(matrix) RESULT(nblkrows_local)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: nblkrows_local

      nblkrows_local = matrix%nblkrows_local
   END FUNCTION dbcsr_nblkrows_local
   FUNCTION dbcsr_nblkcols_local(matrix) RESULT(nblkcols_local)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: nblkcols_local

      nblkcols_local = matrix%nblkcols_local
   END FUNCTION dbcsr_nblkcols_local
   FUNCTION dbcsr_nfullrows_local(matrix) RESULT(nfullrows_local)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: nfullrows_local

      nfullrows_local = matrix%nfullrows_local
   END FUNCTION dbcsr_nfullrows_local
   FUNCTION dbcsr_nfullcols_local(matrix) RESULT(nfullcols_local)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: nfullcols_local

      nfullcols_local = matrix%nfullcols_local
   END FUNCTION dbcsr_nfullcols_local
   FUNCTION dbcsr_max_row_size(matrix) RESULT(max_row_size)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: max_row_size

      max_row_size = matrix%max_rbs
   END FUNCTION dbcsr_max_row_size
   FUNCTION dbcsr_max_col_size(matrix) RESULT(max_col_size)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER                                            :: max_col_size

      max_col_size = matrix%max_cbs
   END FUNCTION dbcsr_max_col_size

   FUNCTION dbcsr_distribution(matrix) RESULT(distribution)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      TYPE(dbcsr_distribution_obj)                       :: distribution

      distribution = matrix%dist
   END FUNCTION dbcsr_distribution

   FUNCTION dbcsr_name(matrix) RESULT(name)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      CHARACTER(len=default_string_length)               :: name

      name = matrix%name
   END FUNCTION dbcsr_name

   SUBROUTINE dbcsr_setname(matrix, newname)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
      CHARACTER(len=*), INTENT(IN)                       :: newname

      matrix%name = newname
   END SUBROUTINE dbcsr_setname

   PURE FUNCTION dbcsr_wm_use_mutable(wm) RESULT(use_mutable)
      !! Returns whether this work matrix uses the mutable type

      TYPE(dbcsr_work_type), INTENT(IN)                  :: wm
         !! work matrix
      LOGICAL                                            :: use_mutable
         !! use the mutable and not append-only working structures

!   ---------------------------------------------------------------------------

      use_mutable = dbcsr_mutable_instantiated(wm%mutable)
   END FUNCTION dbcsr_wm_use_mutable

   PURE FUNCTION dbcsr_use_mutable(matrix) RESULT(use_mutable)
      !! Returns whether work matrices should use the mutable data type

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! matrix
      LOGICAL                                            :: use_mutable
         !! use the mutable and not append-only working structures

!   ---------------------------------------------------------------------------

      use_mutable = matrix%work_mutable
   END FUNCTION dbcsr_use_mutable

   FUNCTION dbcsr_row_block_sizes(matrix) RESULT(row_blk_sizes)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER, DIMENSION(:), POINTER                     :: row_blk_sizes

      row_blk_sizes => array_data(matrix%row_blk_size)
   END FUNCTION dbcsr_row_block_sizes

   FUNCTION dbcsr_col_block_sizes(matrix) RESULT(col_blk_sizes)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER, DIMENSION(:), POINTER                     :: col_blk_sizes

      col_blk_sizes => array_data(matrix%col_blk_size)
   END FUNCTION dbcsr_col_block_sizes

   FUNCTION dbcsr_col_block_offsets(matrix) RESULT(col_blk_offsets)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER, DIMENSION(:), POINTER                     :: col_blk_offsets

      col_blk_offsets => array_data(matrix%col_blk_offset)
   END FUNCTION dbcsr_col_block_offsets

   FUNCTION dbcsr_row_block_offsets(matrix) RESULT(row_blk_offsets)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER, DIMENSION(:), POINTER                     :: row_blk_offsets

      row_blk_offsets => array_data(matrix%row_blk_offset)
   END FUNCTION dbcsr_row_block_offsets

   PURE FUNCTION dbcsr_blk_row_size(matrix, row) RESULT(row_size)
      !! Returns the blocked row size of a row
      !! This routine is optimized for speed and no checks are performed.

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! DBCSR matrix
      INTEGER, INTENT(IN)                                :: row
         !! row number
      INTEGER                                            :: row_size
         !! blocked row size

      row_size = matrix%row_blk_size%low%data(row)
   END FUNCTION dbcsr_blk_row_size

   PURE FUNCTION dbcsr_blk_row_offset(matrix, row) RESULT(row_offset)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER, INTENT(IN)                                :: row
      INTEGER                                            :: row_offset

      row_offset = matrix%row_blk_offset%low%data(row)
   END FUNCTION dbcsr_blk_row_offset

   PURE FUNCTION dbcsr_blk_column_size(matrix, column) RESULT(column_size)
      !! Returns the blocked column size of a column
      !! This routine is optimized for speed and no checks are performed.

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! DBCSR matrix
      INTEGER, INTENT(IN)                                :: column
         !! column number
      INTEGER                                            :: column_size
         !! blocked row size

      column_size = matrix%col_blk_size%low%data(column)
   END FUNCTION dbcsr_blk_column_size

   PURE FUNCTION dbcsr_blk_col_offset(matrix, col) RESULT(col_offset)
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
      INTEGER, INTENT(IN)                                :: col
      INTEGER                                            :: col_offset

      col_offset = matrix%col_blk_offset%low%data(col)
   END FUNCTION dbcsr_blk_col_offset

   FUNCTION dbcsr_data_area(matrix) RESULT(data_area)
      !! Returns the data area

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! matrix from which to get data
      TYPE(dbcsr_data_obj)                               :: data_area
         !! data area

      data_area = matrix%data_area
   END FUNCTION dbcsr_data_area

   PURE FUNCTION dbcsr_get_matrix_type(matrix) RESULT(matrix_type)
      !! Returns the matrix type

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! query this matrix
      CHARACTER                                          :: matrix_type
         !! matrix_type (see dbcsr_types.F for possible values)

      matrix_type = dbcsr_type_invalid
      IF (matrix%symmetry) THEN
         IF ((.NOT. matrix%negate_real) .AND. matrix%negate_imaginary) THEN
            matrix_type = dbcsr_type_hermitian
         ELSEIF (matrix%negate_real .AND. (.NOT. matrix%negate_imaginary)) THEN
            matrix_type = dbcsr_type_antihermitian
         ELSEIF (matrix%negate_real .AND. matrix%negate_imaginary) THEN
            matrix_type = dbcsr_type_antisymmetric
         ELSEIF ((.NOT. matrix%negate_real) .AND. (.NOT. matrix%negate_imaginary)) THEN
            matrix_type = dbcsr_type_symmetric
         ENDIF
      ELSE
         matrix_type = dbcsr_type_no_symmetry
      ENDIF
   END FUNCTION dbcsr_get_matrix_type

   PURE FUNCTION dbcsr_has_symmetry(matrix) RESULT(has_symmetry)
      !! Whether matrix has symmetry

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! query this matrix
      LOGICAL                                            :: has_symmetry
         !! matrix has symmetry

      has_symmetry = matrix%symmetry
   END FUNCTION dbcsr_has_symmetry

   PURE FUNCTION dbcsr_get_replication_type(matrix) RESULT(repl_type)
      !! Returns the data type stored in the matrix

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! query this matrix
      CHARACTER                                          :: repl_type
         !! repl_type (see dbcsr_types.F for possible values)

      repl_type = matrix%replication_type
   END FUNCTION dbcsr_get_replication_type

   PURE FUNCTION dbcsr_get_data_type(matrix) RESULT(data_type)
      !! Returns the data type stored in the matrix

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! query this matrix
      INTEGER                                            :: data_type
         !! data_type (see dbcsr_types.F for possible values)

      data_type = matrix%data_type
   END FUNCTION dbcsr_get_data_type

   FUNCTION dbcsr_get_data_memory_type(matrix) &
      RESULT(memory_type)
      !! Returns the type of memory used for data in the matrix
      !! @note It returns the declared data type, not the actually used type

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! query this matrix
      TYPE(dbcsr_memtype_type)                           :: memory_type
         !! memory type used for data

      memory_type = matrix%data_memory_type
   END FUNCTION dbcsr_get_data_memory_type

   FUNCTION dbcsr_get_index_memory_type(matrix) RESULT(memory_type)
      !! Returns the type of memory used for the index in the matrix

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! query this matrix
      TYPE(dbcsr_memtype_type)                           :: memory_type
         !! memory type used for the index

      memory_type = matrix%index_memory_type
   END FUNCTION dbcsr_get_index_memory_type

!  PURE FUNCTION uses_special_memory_matrix (matrix) RESULT (uses_special)
!   !! Returns whether the matrix uses specially-allocated memory
!    TYPE(dbcsr_type), INTENT(IN)              :: matrix
!       !! query this matrix
!    LOGICAL                                  :: uses_special
!       !! whether the matrix uses specially allocated memory
!
!    uses_special = matrix%data_memory_type .NE. dbcsr_memory_default
!  END FUNCTION uses_special_memory_matrix
!
!
!  PURE FUNCTION uses_special_memory_area (area) RESULT (uses_special)
!    !! Returns whether the data area uses special-allocated memory
!    TYPE(dbcsr_data_obj), INTENT(IN)         :: area
!       !! query this data area
!    LOGICAL                                  :: uses_special
!       !! whether the data area uses specially allocated memory
!
!    IF (ASSOCIATED (area%d)) THEN
!       uses_special = area%d%memory_type .NE. dbcsr_memory_default
!    ELSE
!       uses_special = .FALSE.
!    ENDIF
!  END FUNCTION uses_special_memory_area
!

   FUNCTION dbcsr_get_data_size(matrix) RESULT(data_size)
      !! Returns the allocated data size of a DBCSR matrix

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! matrix
      INTEGER                                            :: data_size
         !! size of data

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_data_size', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: data_type

      data_size = 0
      data_type = dbcsr_get_data_type(matrix)
      IF (data_type .NE. dbcsr_type_real_8 .AND. &
          data_type .NE. dbcsr_type_real_4 .AND. &
          data_type .NE. dbcsr_type_complex_8 .AND. &
          data_type .NE. dbcsr_type_complex_4) DBCSR_ABORT("Incorrect data type")
      data_size = dbcsr_data_get_size(matrix%data_area)
   END FUNCTION dbcsr_get_data_size

   FUNCTION dbcsr_get_data_size_used(matrix) RESULT(data_size)
      !! Count actual data storage used for matrix data.

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! Count data of this matrix
      INTEGER                                            :: data_size
         !! Data size used by matrix

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_get_data_size_used', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: blk, col, error_handle, nze, row
      INTEGER, DIMENSION(:), POINTER                     :: col_blk_sizes, row_blk_sizes

!type(dbcsr_iterator_type) :: iter
!   ---------------------------------------------------------------------------

      CALL timeset(routineN, error_handle)
      row_blk_sizes => dbcsr_row_block_sizes(matrix)
      col_blk_sizes => dbcsr_col_block_sizes(matrix)
      data_size = 0
!$OMP     DO
      DO row = 1, matrix%nblkrows_total
         DO blk = matrix%row_p(row) + 1, matrix%row_p(row + 1)
            col = matrix%col_i(blk)
            IF (matrix%blk_p(blk) .NE. 0) THEN
               nze = row_blk_sizes(row)*col_blk_sizes(col)
               data_size = data_size + nze
            ENDIF
         ENDDO
      ENDDO
!$OMP     END DO
      CALL timestop(error_handle)
   END FUNCTION dbcsr_get_data_size_used

   PURE FUNCTION dbcsr_get_num_blocks(matrix) RESULT(num_blocks)
      !! Returns the number of blocks in the matrix

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! matrix from which to get data
      INTEGER                                            :: num_blocks

      num_blocks = matrix%nblks
   END FUNCTION dbcsr_get_num_blocks

   PURE FUNCTION dbcsr_get_nze(matrix) RESULT(num_nze)
      !! Returns the number of non-zero elements in the matrix

      TYPE(dbcsr_type), INTENT(IN)                       :: matrix
         !! matrix from which to get data
      INTEGER                                            :: num_nze

      num_nze = matrix%nze
   END FUNCTION dbcsr_get_nze

! **************************************************************************************************
! Arrays
! **************************************************************************************************

   SUBROUTINE dbcsr_destroy_1d_array(marray)
      !! Releases all matrices in a 1-d array.

      TYPE(dbcsr_1d_array_type), INTENT(INOUT)           :: marray
         !! matrix array

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_destroy_1d_array', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: i

!   ---------------------------------------------------------------------------

      DO i = LBOUND(marray%mats, 1), UBOUND(marray%mats, 1)
         CALL dbcsr_destroy(marray%mats(i), force=.TRUE.)
      ENDDO
      CALL dbcsr_image_dist_release(marray%image_dist)
      DEALLOCATE (marray%mats)
   END SUBROUTINE dbcsr_destroy_1d_array

   SUBROUTINE dbcsr_destroy_2d_array(marray)
      !! Releases all matrices in 2-d array.

      TYPE(dbcsr_2d_array_type), INTENT(INOUT)           :: marray
         !! matrix array

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_destroy_2d_array', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: col, row

!   ---------------------------------------------------------------------------

      DO row = LBOUND(marray%mats, 1), UBOUND(marray%mats, 1)
         DO col = LBOUND(marray%mats, 2), UBOUND(marray%mats, 2)
            CALL dbcsr_destroy(marray%mats(row, col), force=.TRUE.)
         ENDDO
      ENDDO
      CALL dbcsr_image_dist_release(marray%image_dist)
      DEALLOCATE (marray%mats)
   END SUBROUTINE dbcsr_destroy_2d_array

   SUBROUTINE dbcsr_image_dist_release(imgdist)
      !! Releases a reference to and possible deallocates an image
      !! distribution

      TYPE(dbcsr_imagedistribution_obj), INTENT(INOUT)   :: imgdist

      IF (ASSOCIATED(imgdist%i)) THEN
         imgdist%i%refcount = imgdist%i%refcount - 1
         IF (imgdist%i%refcount .EQ. 0) THEN
            CALL dbcsr_destroy_image_dist(imgdist%i)
            DEALLOCATE (imgdist%i)
         ENDIF
      ENDIF
   END SUBROUTINE dbcsr_image_dist_release

   SUBROUTINE dbcsr_image_dist_hold(imgdist)
      !! Retains a reference to an image distribution
      TYPE(dbcsr_imagedistribution_obj), INTENT(INOUT)   :: imgdist

      imgdist%i%refcount = imgdist%i%refcount + 1
   END SUBROUTINE dbcsr_image_dist_hold

   SUBROUTINE dbcsr_image_dist_init(imgdist)
      !! Initialized an image distribution
      !!
      !! Akin to nullify.

      TYPE(dbcsr_imagedistribution_obj), INTENT(OUT)     :: imgdist

      NULLIFY (imgdist%i)
   END SUBROUTINE dbcsr_image_dist_init

   SUBROUTINE dbcsr_destroy_image_dist(imgdist)
      !! Destroys a DBCSR distribution for a matrix multiplication based on
      !! the right matrix

      TYPE(dbcsr_imagedistribution_type), INTENT(INOUT)  :: imgdist
         !! product distribution repetition

      INTEGER                                            :: i

!   ---------------------------------------------------------------------------

      CALL array_release(imgdist%row_image)
      CALL array_release(imgdist%col_image)
      CALL dbcsr_distribution_release(imgdist%main)
      !
      CALL array_release(imgdist%vrow_dist)
      CALL array_release(imgdist%vcol_dist)
      !
      IF (imgdist%has_other_vl_rows) THEN
         DO i = LBOUND(imgdist%other_vl_rows, 1), UBOUND(imgdist%other_vl_rows, 1)
            CALL array_release(imgdist%other_vl_rows(i))
         END DO
         DEALLOCATE (imgdist%other_vl_rows)
         imgdist%has_other_vl_rows = .FALSE.
      ENDIF
      !
      IF (imgdist%has_other_vl_cols) THEN
         DO i = LBOUND(imgdist%other_vl_cols, 1), UBOUND(imgdist%other_vl_cols, 1)
            CALL array_release(imgdist%other_vl_cols(i))
         END DO
         DEALLOCATE (imgdist%other_vl_cols)
         imgdist%has_other_vl_cols = .FALSE.
      ENDIF
      !
      IF (imgdist%has_global_vrow_map) THEN
         CALL array_release(imgdist%global_vrow_map)
      ENDIF
      IF (imgdist%has_global_vcol_map) THEN
         CALL array_release(imgdist%global_vcol_map)
      ENDIF
   END SUBROUTINE dbcsr_destroy_image_dist

! **************************************************************************************************
! Mutable data
! **************************************************************************************************

   SUBROUTINE dbcsr_mutable_init(mutable)
      !! Initializes a mutable data type

      TYPE(dbcsr_mutable_obj), INTENT(OUT)               :: mutable
         !! mutable data

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_mutable_init', &
                                     routineP = moduleN//':'//routineN

!   ---------------------------------------------------------------------------

      NULLIFY (mutable%m)
   END SUBROUTINE dbcsr_mutable_init

   SUBROUTINE dbcsr_mutable_destroy(mutable)
      !! Destroys a mutable data type

      TYPE(dbcsr_mutable_obj), INTENT(INOUT)             :: mutable
         !! mutable data

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_mutable_destroy', &
                                     routineP = moduleN//':'//routineN

!   ---------------------------------------------------------------------------

      IF (ASSOCIATED(mutable%m)) THEN
         CALL btree_delete(mutable%m%btree_s)
         CALL btree_delete(mutable%m%btree_d)
         CALL btree_delete(mutable%m%btree_c)
         CALL btree_delete(mutable%m%btree_z)
         DEALLOCATE (mutable%m)
      ENDIF
      NULLIFY (mutable%m)
   END SUBROUTINE dbcsr_mutable_destroy

   SUBROUTINE dbcsr_mutable_release(mutable)
      !! Unregisters a reference to the mutable data type
      !! The object is destroy when there is no reference to it left.

      TYPE(dbcsr_mutable_obj), INTENT(INOUT)             :: mutable
         !! mutable data

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_mutable_release', &
                                     routineP = moduleN//':'//routineN

!   ---------------------------------------------------------------------------

      IF (.NOT. ASSOCIATED(mutable%m)) &
         DBCSR_ABORT("Mutable data area not instantiated")
      mutable%m%refcount = mutable%m%refcount - 1
      IF (mutable%m%refcount .EQ. 0) THEN
         CALL dbcsr_mutable_destroy(mutable)
      ENDIF
   END SUBROUTINE dbcsr_mutable_release

   SUBROUTINE dbcsr_mutable_new(mutable, data_type)
      !! Creates a new mutable instance.

      TYPE(dbcsr_mutable_obj), INTENT(INOUT)             :: mutable
         !! mutable data
      INTEGER, INTENT(IN)                                :: data_type
         !! data type to be stored here (see dbcsr_types for possibilities)

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_mutable_new', &
                                     routineP = moduleN//':'//routineN

!   ---------------------------------------------------------------------------

      IF (ASSOCIATED(mutable%m)) &
         DBCSR_ABORT("Mutable data area already instantiated")
      IF (data_type .NE. dbcsr_type_real_4 &
          .AND. data_type .NE. dbcsr_type_real_8 &
          .AND. data_type .NE. dbcsr_type_complex_4 &
          .AND. data_type .NE. dbcsr_type_complex_8) &
         DBCSR_ABORT("Invalid data type")
      ALLOCATE (mutable%m)
      mutable%m%refcount = 1
      mutable%m%data_type = data_type
      CALL btree_new(mutable%m%btree_s)
      CALL btree_new(mutable%m%btree_d)
      CALL btree_new(mutable%m%btree_c)
      CALL btree_new(mutable%m%btree_z)
   END SUBROUTINE dbcsr_mutable_new

   PURE FUNCTION dbcsr_mutable_instantiated(mutable) RESULT(instantiated)
      !! Unregisters a reference to the mutable data type
      !! The object is destroy when there is no reference to it left.

      TYPE(dbcsr_mutable_obj), INTENT(IN)                :: mutable
         !! mutable data
      LOGICAL                                            :: instantiated
         !! whether the object is instantiated

!   ---------------------------------------------------------------------------

      instantiated = ASSOCIATED(mutable%m)
   END FUNCTION dbcsr_mutable_instantiated

END MODULE dbcsr_methods
