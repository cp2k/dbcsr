!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tas_reshape_ops
   !! communication routines to reshape / replicate / merge tall-and-skinny matrices.
#:include "dbcsr_tas.fypp"

   USE dbcsr_block_access, ONLY: &
      dbcsr_put_block, dbcsr_reserve_blocks
   USE dbcsr_data_methods, ONLY: &
      dbcsr_data_clear_pointer, dbcsr_data_init, dbcsr_data_new, dbcsr_data_release, &
      dbcsr_type_1d_to_2d, dbcsr_data_get_sizes, dbcsr_data_set_pointer
   USE dbcsr_data_methods_low, ONLY: &
      internal_data_allocate, internal_data_deallocate, dbcsr_get_data_p_2d_d, dbcsr_get_data_p_2d_s, &
      dbcsr_get_data_p_2d_z, dbcsr_get_data_p_2d_c
   USE dbcsr_data_types, ONLY: &
      dbcsr_data_obj, dbcsr_type_real_8, dbcsr_type_real_4, dbcsr_type_complex_8, dbcsr_type_complex_4
   USE dbcsr_dist_methods, ONLY: &
      dbcsr_distribution_col_dist, dbcsr_distribution_row_dist
   USE dbcsr_dist_operations, ONLY: &
      dbcsr_get_stored_coordinates
   USE dbcsr_iterator_operations, ONLY: &
      dbcsr_iterator_blocks_left, dbcsr_iterator_next_block, dbcsr_iterator_start, dbcsr_iterator_stop
   USE dbcsr_methods, ONLY: &
      dbcsr_blk_column_size, dbcsr_blk_row_size
   USE dbcsr_operations, ONLY: &
      dbcsr_get_info, dbcsr_clear
   USE dbcsr_tas_base, ONLY: &
      dbcsr_tas_blk_sizes, dbcsr_tas_create, dbcsr_tas_distribution_new, dbcsr_tas_finalize, &
      dbcsr_tas_get_data_type, dbcsr_tas_get_stored_coordinates, dbcsr_tas_info, &
      dbcsr_tas_iterator_blocks_left, dbcsr_tas_iterator_next_block, dbcsr_tas_iterator_start, &
      dbcsr_tas_iterator_stop, dbcsr_tas_put_block, dbcsr_tas_reserve_blocks, &
      dbcsr_repl_get_stored_coordinates, dbcsr_tas_clear
   USE dbcsr_tas_types, ONLY: &
      dbcsr_tas_distribution_type, dbcsr_tas_iterator, dbcsr_tas_split_info, dbcsr_tas_type
   USE dbcsr_tas_global, ONLY: &
      dbcsr_tas_blk_size_arb, dbcsr_tas_blk_size_repl, dbcsr_tas_dist_arb, dbcsr_tas_dist_repl, &
      dbcsr_tas_distribution, dbcsr_tas_rowcol_data
   USE dbcsr_tas_split, ONLY: &
      colsplit, dbcsr_tas_get_split_info, rowsplit
   USE dbcsr_types, ONLY: &
      dbcsr_distribution_obj, dbcsr_iterator, dbcsr_type
   USE dbcsr_work_operations, ONLY: &
      dbcsr_finalize
   USE dbcsr_kinds, ONLY: &
      int_8, real_8, real_4
   USE dbcsr_mpiwrap, ONLY: &
      mp_alltoall, mp_environ, mp_isend, mp_irecv, mp_waitall
   USE dbcsr_tas_util, ONLY: &
      swap, index_unique
#include "../base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tas_reshape_ops'

   PUBLIC :: &
      dbcsr_tas_merge, &
      dbcsr_tas_replicate, &
      dbcsr_tas_reshape

   TYPE block_buffer_type
      INTEGER :: nblock = -1
      INTEGER(KIND=int_8), DIMENSION(:, :), ALLOCATABLE :: indx
#:for dparam, dtype, dsuffix in dtype_float_list
      ${dtype}$, DIMENSION(:), ALLOCATABLE :: msg_${dsuffix}$
#:endfor
      INTEGER :: data_type = -1
      INTEGER :: endpos = -1
   END TYPE

   INTERFACE block_buffer_get_next_block
      MODULE PROCEDURE block_buffer_get_next_area_block
#!for dparam, dtype, dsuffix in dtype_float_list
      !MODULE PROCEDURE block_buffer_get_next_block_${dsuffix}$ ! issue: ambiguous interface
#!endfor
   END INTERFACE

   INTERFACE block_buffer_add_block
      MODULE PROCEDURE block_buffer_add_area_block
#:for dparam, dtype, dsuffix in dtype_float_list
      MODULE PROCEDURE block_buffer_add_block_${dsuffix}$
#:endfor
   END INTERFACE

CONTAINS

   RECURSIVE SUBROUTINE dbcsr_tas_reshape(matrix_in, matrix_out, summation, transposed, move_data)
      !! copy data (involves reshape)

      TYPE(dbcsr_tas_type), INTENT(INOUT)                  :: matrix_in, matrix_out
      LOGICAL, INTENT(IN), OPTIONAL                      :: summation
         !! whether matrix_out = matrix_out + matrix_in
      LOGICAL, INTENT(IN), OPTIONAL                      :: transposed
      LOGICAL, INTENT(IN), OPTIONAL                      :: move_data
         !! memory optimization: move data to matrix_out such that matrix_in is empty on return

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_reshape', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: handle, handle2, iproc, mp_comm, mynode, nblock, &
                                                            ndata, numnodes, bcount, nblk
      INTEGER(KIND=int_8), ALLOCATABLE, DIMENSION(:, :)  :: index_recv, blks_to_allocate
      INTEGER(KIND=int_8), DIMENSION(2)                  :: blk_index
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: num_blocks_recv, num_blocks_send, &
                                                            num_entries_recv, num_entries_send, &
                                                            num_rec, num_send
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: req_array
      INTEGER, DIMENSION(2)                              :: blk_size
      LOGICAL                                            :: tr, tr_in, move_prv
      TYPE(block_buffer_type), ALLOCATABLE, DIMENSION(:) :: buffer_recv, buffer_send
      TYPE(dbcsr_data_obj)                               :: block
      TYPE(dbcsr_tas_iterator)                           :: iter
      TYPE(dbcsr_tas_split_info)                         :: info

      CALL timeset(routineN, handle)

      IF (PRESENT(summation)) THEN
         IF (.NOT. summation) CALL dbcsr_clear(matrix_out%matrix)
      ELSE
         CALL dbcsr_clear(matrix_out%matrix)
      ENDIF

      IF (PRESENT(move_data)) THEN
         move_prv = move_data
      ELSE
         move_prv = .FALSE.
      ENDIF

      IF (PRESENT(transposed)) THEN
         tr_in = transposed
      ELSE
         tr_in = .FALSE.
      ENDIF

      IF (.NOT. matrix_out%valid) THEN
         DBCSR_ABORT("can not reshape into invalid matrix")
      ENDIF

      info = dbcsr_tas_info(matrix_in)
      mp_comm = info%mp_comm
      CALL mp_environ(numnodes, mynode, mp_comm)
      ALLOCATE (buffer_send(0:numnodes - 1))
      ALLOCATE (buffer_recv(0:numnodes - 1))
      ALLOCATE (num_blocks_recv(0:numnodes - 1))
      ALLOCATE (num_blocks_send(0:numnodes - 1))
      ALLOCATE (num_entries_recv(0:numnodes - 1))
      ALLOCATE (num_entries_send(0:numnodes - 1))
      ALLOCATE (num_rec(0:2*numnodes - 1))
      ALLOCATE (num_send(0:2*numnodes - 1))
      num_send(:) = 0
      ALLOCATE (req_array(1:numnodes, 4))
      CALL dbcsr_tas_iterator_start(iter, matrix_in)

      CALL timeset(routineN//"_get_coord", handle2)
      DO WHILE (dbcsr_tas_iterator_blocks_left(iter))
         CALL dbcsr_tas_iterator_next_block(iter, blk_index(1), blk_index(2), nblock, transposed=tr, &
                                            row_size=blk_size(1), col_size=blk_size(2))

         IF (tr_in) THEN
            CALL dbcsr_tas_get_stored_coordinates(matrix_out, blk_index(2), blk_index(1), iproc)
         ELSE
            CALL dbcsr_tas_get_stored_coordinates(matrix_out, blk_index(1), blk_index(2), iproc)
         ENDIF

         num_send(2*iproc) = num_send(2*iproc) + PRODUCT(blk_size)
         num_send(2*iproc + 1) = num_send(2*iproc + 1) + 1
      ENDDO
      CALL dbcsr_tas_iterator_stop(iter)
      CALL timestop(handle2)

      CALL timeset(routineN//"_alltoall", handle2)
      CALL mp_alltoall(num_send, num_rec, 2, mp_comm)
      CALL timestop(handle2)

      CALL timeset(routineN//"_buffer_fill", handle2)
      DO iproc = 0, numnodes - 1
         num_entries_recv(iproc) = num_rec(2*iproc)
         num_blocks_recv(iproc) = num_rec(2*iproc + 1)
         num_entries_send(iproc) = num_send(2*iproc)
         num_blocks_send(iproc) = num_send(2*iproc + 1)

         CALL block_buffer_create(buffer_send(iproc), num_blocks_send(iproc), num_entries_send(iproc), &
                                  dbcsr_tas_get_data_type(matrix_in))

         CALL block_buffer_create(buffer_recv(iproc), num_blocks_recv(iproc), num_entries_recv(iproc), &
                                  dbcsr_tas_get_data_type(matrix_in))

      ENDDO

      CALL dbcsr_data_init(block)
      CALL dbcsr_data_new(block, dbcsr_type_1d_to_2d(dbcsr_tas_get_data_type(matrix_in))) ! need to convert to 2d data type
      CALL dbcsr_tas_iterator_start(iter, matrix_in)
      DO WHILE (dbcsr_tas_iterator_blocks_left(iter))
         CALL dbcsr_tas_iterator_next_block(iter, blk_index(1), blk_index(2), block, tr, &
                                            row_size=blk_size(1), col_size=blk_size(2))

         DBCSR_ASSERT(tr .EQV. .FALSE.)

         IF (tr_in) THEN
            CALL dbcsr_tas_get_stored_coordinates(matrix_out, blk_index(2), blk_index(1), iproc)
         ELSE
            CALL dbcsr_tas_get_stored_coordinates(matrix_out, blk_index(1), blk_index(2), iproc)
         ENDIF
         CALL block_buffer_add_block(buffer_send(iproc), blk_index, block, transposed=tr_in)
      ENDDO
      CALL dbcsr_tas_iterator_stop(iter)
      CALL dbcsr_data_clear_pointer(block)

      IF (move_prv) CALL dbcsr_tas_clear(matrix_in)

      CALL timestop(handle2)

      CALL timeset(routineN//"_communicate_buffer", handle2)
      CALL dbcsr_tas_communicate_buffer(mp_comm, buffer_recv, buffer_send, req_array)

      DO iproc = 0, numnodes - 1
         CALL block_buffer_destroy(buffer_send(iproc))
      ENDDO

      CALL timestop(handle2)

      CALL timeset(routineN//"_buffer_obtain", handle2)

      nblk = SUM(num_blocks_recv)
      ALLOCATE (blks_to_allocate(nblk, 2))

      bcount = 0
      DO iproc = 0, numnodes - 1
         CALL block_buffer_get_index(buffer_recv(iproc), index_recv)
         blks_to_allocate(bcount + 1:bcount + SIZE(index_recv, 1), :) = index_recv(:, :)
         bcount = bcount + SIZE(index_recv, 1)
         DEALLOCATE (index_recv)
      ENDDO

      CALL dbcsr_tas_reserve_blocks(matrix_out, blks_to_allocate(:, 1), blks_to_allocate(:, 2))
      DEALLOCATE (blks_to_allocate)

      DO iproc = 0, numnodes - 1
         ! First, we need to get the index to create block
         DO WHILE (block_buffer_blocks_left(buffer_recv(iproc)))
            CALL block_buffer_get_next_block(buffer_recv(iproc), ndata, blk_index)
            CALL dbcsr_tas_blk_sizes(matrix_out, blk_index(1), blk_index(2), blk_size(1), blk_size(2))
            CALL internal_data_allocate(block%d, blk_size)
            CALL block_buffer_get_next_block(buffer_recv(iproc), ndata, blk_index, block)
            CALL dbcsr_tas_put_block(matrix_out, blk_index(1), blk_index(2), block, summation=summation)
            CALL internal_data_deallocate(block%d)
         ENDDO
         CALL block_buffer_destroy(buffer_recv(iproc))
      ENDDO
      CALL dbcsr_data_clear_pointer(block)
      CALL dbcsr_data_release(block)

      CALL timestop(handle2)

      CALL dbcsr_tas_finalize(matrix_out)

      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_replicate(matrix_in, info, matrix_out, nodata, move_data)
      !! Replicate matrix_in such that each submatrix of matrix_out is an exact copy of matrix_in

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_in
      TYPE(dbcsr_tas_split_info), INTENT(IN)             :: info
      TYPE(dbcsr_tas_type), INTENT(OUT)                  :: matrix_out
      LOGICAL, INTENT(IN), OPTIONAL                      :: nodata
         !! Don't copy data but create matrix_out
      LOGICAL, INTENT(IN), OPTIONAL                      :: move_data
         !! memory optimization: move data to matrix_out such that matrix_in is empty on return

      INTEGER                                            :: data_type, nblkcols, nblkrows
      INTEGER, DIMENSION(2)                              :: pcoord, pdims
      INTEGER, DIMENSION(:), POINTER                     :: col_blk_size => NULL(), col_dist => NULL(), &
                                                            row_blk_size => NULL(), row_dist => NULL()
      TYPE(dbcsr_distribution_obj)                       :: dbcsr_dist
      TYPE(dbcsr_tas_dist_arb), TARGET                     :: dir_dist
      TYPE(dbcsr_tas_dist_repl), TARGET                    :: repl_dist

      CLASS(dbcsr_tas_distribution), ALLOCATABLE :: col_dist_obj, row_dist_obj
      CLASS(dbcsr_tas_rowcol_data), ALLOCATABLE :: row_bsize_obj, col_bsize_obj
      TYPE(dbcsr_tas_blk_size_repl), TARGET :: repl_blksize
      TYPE(dbcsr_tas_blk_size_arb), TARGET :: dir_blksize
      TYPE(dbcsr_tas_distribution_type) :: dist
      INTEGER :: mp_comm, numnodes, mynode
      TYPE(block_buffer_type), ALLOCATABLE, DIMENSION(:) :: buffer_recv, buffer_send
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: num_blocks_recv, num_blocks_send, &
                                                            num_entries_recv, num_entries_send, &
                                                            num_rec, num_send
      INTEGER, ALLOCATABLE, DIMENSION(:, :) :: req_array, blks_to_allocate
      INTEGER, DIMENSION(2) :: blk_size
      INTEGER, DIMENSION(2) :: blk_index
      INTEGER(KIND=int_8), DIMENSION(2) :: blk_index_i8
      TYPE(dbcsr_iterator) :: iter
      INTEGER :: nblock, i, iproc, bcount, nblk
      INTEGER, DIMENSION(:), ALLOCATABLE :: iprocs
      LOGICAL :: tr, nodata_prv, move_prv
      INTEGER(KIND=int_8), ALLOCATABLE, DIMENSION(:, :) :: index_recv
      INTEGER :: ndata

      TYPE(dbcsr_data_obj) :: block

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_replicate', &
                                     routineP = moduleN//':'//routineN

      INTEGER :: handle, handle2

      CALL timeset(routineN, handle)

      IF (PRESENT(nodata)) THEN
         nodata_prv = nodata
      ELSE
         nodata_prv = .FALSE.
      ENDIF

      IF (PRESENT(move_data)) THEN
         move_prv = move_data
      ELSE
         move_prv = .FALSE.
      ENDIF

      CALL dbcsr_get_info(matrix_in, distribution=dbcsr_dist, data_type=data_type, &
                          nblkrows_total=nblkrows, nblkcols_total=nblkcols, &
                          row_blk_size=row_blk_size, col_blk_size=col_blk_size)
      row_dist => dbcsr_distribution_row_dist(dbcsr_dist)
      col_dist => dbcsr_distribution_col_dist(dbcsr_dist)

      mp_comm = info%mp_comm

      CALL mp_environ(numnodes, mynode, mp_comm)
      CALL mp_environ(numnodes, pdims, pcoord, mp_comm)

      SELECT CASE (info%split_rowcol)
      CASE (rowsplit)
         repl_dist = dbcsr_tas_dist_repl(row_dist, pdims(1), nblkrows, info%ngroup, info%pgrid_split_size)
         dir_dist = dbcsr_tas_dist_arb(col_dist, pdims(2), INT(nblkcols, KIND=int_8))
         repl_blksize = dbcsr_tas_blk_size_repl(row_blk_size, info%ngroup)
         dir_blksize = dbcsr_tas_blk_size_arb(col_blk_size)
         ALLOCATE (row_dist_obj, source=repl_dist)
         ALLOCATE (col_dist_obj, source=dir_dist)
         ALLOCATE (row_bsize_obj, source=repl_blksize)
         ALLOCATE (col_bsize_obj, source=dir_blksize)
      CASE (colsplit)
         dir_dist = dbcsr_tas_dist_arb(row_dist, pdims(1), INT(nblkrows, KIND=int_8))
         repl_dist = dbcsr_tas_dist_repl(col_dist, pdims(2), nblkcols, info%ngroup, info%pgrid_split_size)
         dir_blksize = dbcsr_tas_blk_size_arb(row_blk_size)
         repl_blksize = dbcsr_tas_blk_size_repl(col_blk_size, info%ngroup)
         ALLOCATE (row_dist_obj, source=dir_dist)
         ALLOCATE (col_dist_obj, source=repl_dist)
         ALLOCATE (row_bsize_obj, source=dir_blksize)
         ALLOCATE (col_bsize_obj, source=repl_blksize)
      END SELECT

      CALL dbcsr_tas_distribution_new(dist, mp_comm, row_dist_obj, col_dist_obj, split_info=info)
      CALL dbcsr_tas_create(matrix_out, TRIM(matrix_in%name)//" replicated", &
                            dist, data_type, row_bsize_obj, col_bsize_obj, own_dist=.TRUE.)

      IF (nodata_prv) THEN
         CALL dbcsr_tas_finalize(matrix_out)
         CALL timestop(handle)
         RETURN
      ENDIF

      ALLOCATE (buffer_send(0:numnodes - 1))
      ALLOCATE (buffer_recv(0:numnodes - 1))
      ALLOCATE (num_blocks_recv(0:numnodes - 1))
      ALLOCATE (num_blocks_send(0:numnodes - 1))
      ALLOCATE (num_entries_recv(0:numnodes - 1))
      ALLOCATE (num_entries_send(0:numnodes - 1))
      ALLOCATE (num_rec(0:2*numnodes - 1))
      ALLOCATE (num_send(0:2*numnodes - 1))
      num_send(:) = 0
      ALLOCATE (req_array(1:numnodes, 4))

      ALLOCATE (iprocs(info%ngroup))
      CALL dbcsr_iterator_start(iter, matrix_in)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, blk_index(1), blk_index(2), nblock, transposed=tr, &
                                        row_size=blk_size(1), col_size=blk_size(2))
         CALL dbcsr_repl_get_stored_coordinates(matrix_out, blk_index(1), blk_index(2), iprocs)
         DO i = 1, SIZE(iprocs)
            num_send(2*iprocs(i)) = num_send(2*iprocs(i)) + PRODUCT(blk_size)
            num_send(2*iprocs(i) + 1) = num_send(2*iprocs(i) + 1) + 1
         ENDDO
      ENDDO
      CALL dbcsr_iterator_stop(iter)

      CALL timeset(routineN//"_alltoall", handle2)
      CALL mp_alltoall(num_send, num_rec, 2, mp_comm)
      CALL timestop(handle2)

      DO iproc = 0, numnodes - 1
         num_entries_recv(iproc) = num_rec(2*iproc)
         num_blocks_recv(iproc) = num_rec(2*iproc + 1)
         num_entries_send(iproc) = num_send(2*iproc)
         num_blocks_send(iproc) = num_send(2*iproc + 1)

         CALL block_buffer_create(buffer_send(iproc), num_blocks_send(iproc), num_entries_send(iproc), &
                                  data_type)

         CALL block_buffer_create(buffer_recv(iproc), num_blocks_recv(iproc), num_entries_recv(iproc), &
                                  data_type)

      ENDDO

      CALL dbcsr_data_init(block)
      CALL dbcsr_data_new(block, dbcsr_type_1d_to_2d(data_type))
      CALL dbcsr_iterator_start(iter, matrix_in)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, blk_index(1), blk_index(2), block, tr, &
                                        row_size=blk_size(1), col_size=blk_size(2))
         CALL dbcsr_repl_get_stored_coordinates(matrix_out, blk_index(1), blk_index(2), iprocs)
         DO i = 1, SIZE(iprocs)
            CALL block_buffer_add_block(buffer_send(iprocs(i)), INT(blk_index, KIND=int_8), block)
         ENDDO
      ENDDO
      CALL dbcsr_iterator_stop(iter)
      CALL dbcsr_data_clear_pointer(block)

      IF (move_prv) CALL dbcsr_clear(matrix_in)

      CALL timeset(routineN//"_communicate_buffer", handle2)
      CALL dbcsr_tas_communicate_buffer(mp_comm, buffer_recv, buffer_send, req_array)

      DO iproc = 0, numnodes - 1
         CALL block_buffer_destroy(buffer_send(iproc))
      ENDDO

      CALL timestop(handle2)

      nblk = SUM(num_blocks_recv)
      ALLOCATE (blks_to_allocate(nblk, 2))

      bcount = 0
      DO iproc = 0, numnodes - 1
         CALL block_buffer_get_index(buffer_recv(iproc), index_recv)
         blks_to_allocate(bcount + 1:bcount + SIZE(index_recv, 1), :) = INT(index_recv(:, :))
         bcount = bcount + SIZE(index_recv, 1)
         DEALLOCATE (index_recv)
      ENDDO

      CALL dbcsr_reserve_blocks(matrix_out%matrix, blks_to_allocate(:, 1), blks_to_allocate(:, 2))
      DEALLOCATE (blks_to_allocate)

      DO iproc = 0, numnodes - 1
         ! First, we need to get the index to create block
         DO WHILE (block_buffer_blocks_left(buffer_recv(iproc)))
            CALL block_buffer_get_next_block(buffer_recv(iproc), ndata, blk_index_i8)
            CALL dbcsr_tas_blk_sizes(matrix_out, blk_index_i8(1), blk_index_i8(2), blk_size(1), blk_size(2))
            CALL internal_data_allocate(block%d, blk_size)
            CALL block_buffer_get_next_block(buffer_recv(iproc), ndata, blk_index_i8, block)
            CALL dbcsr_put_block(matrix_out%matrix, INT(blk_index_i8(1)), INT(blk_index_i8(2)), block)
            CALL internal_data_deallocate(block%d)
         ENDDO

         CALL block_buffer_destroy(buffer_recv(iproc))
      ENDDO
      CALL dbcsr_data_clear_pointer(block)
      CALL dbcsr_data_release(block)

      CALL dbcsr_tas_finalize(matrix_out)

      CALL timestop(handle)

   END SUBROUTINE

   SUBROUTINE dbcsr_tas_merge(matrix_out, matrix_in, move_data)
      !! Merge submatrices of matrix_in to matrix_out by sum

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_out
      TYPE(dbcsr_tas_type), INTENT(INOUT)                  :: matrix_in
      LOGICAL, INTENT(IN), OPTIONAL                      :: move_data
         !! memory optimization: move data to matrix_out such that matrix_in is empty on return

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_merge', routineP = moduleN//':'//routineN

      INTEGER                                            :: data_type, handle, handle2, iproc, mp_comm, &
                                                            mynode, nblock, ndata, numnodes, nblk, bcount
      INTEGER(KIND=int_8), ALLOCATABLE, DIMENSION(:, :)  :: index_recv
      INTEGER(KIND=int_8), DIMENSION(2)                  :: blk_index_i8
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: iprocs, num_blocks_recv, &
                                                            num_blocks_send, num_entries_recv, &
                                                            num_entries_send, num_rec, num_send
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: req_array, blks_to_allocate, blks_to_allocate_u
      INTEGER, DIMENSION(2)                              :: blk_index, blk_size
      LOGICAL                                            :: tr, move_prv
      TYPE(block_buffer_type), ALLOCATABLE, DIMENSION(:) :: buffer_recv, buffer_send
      TYPE(dbcsr_data_obj)                               :: block
      TYPE(dbcsr_iterator)                               :: iter
      TYPE(dbcsr_tas_split_info)                         :: info

      CALL timeset(routineN, handle)

      CALL dbcsr_clear(matrix_out)

      IF (PRESENT(move_data)) THEN
         move_prv = move_data
      ELSE
         move_prv = .FALSE.
      ENDIF

      data_type = dbcsr_tas_get_data_type(matrix_in)

      info = dbcsr_tas_info(matrix_in)
      CALL dbcsr_tas_get_split_info(info, mp_comm=mp_comm)
      CALL mp_environ(numnodes, mynode, mp_comm)

      ALLOCATE (buffer_send(0:numnodes - 1))
      ALLOCATE (buffer_recv(0:numnodes - 1))
      ALLOCATE (num_blocks_recv(0:numnodes - 1))
      ALLOCATE (num_blocks_send(0:numnodes - 1))
      ALLOCATE (num_entries_recv(0:numnodes - 1))
      ALLOCATE (num_entries_send(0:numnodes - 1))
      ALLOCATE (num_rec(0:2*numnodes - 1))
      ALLOCATE (num_send(0:2*numnodes - 1))
      num_send(:) = 0
      ALLOCATE (req_array(1:numnodes, 4))

      ALLOCATE (iprocs(info%ngroup))

      CALL dbcsr_iterator_start(iter, matrix_in%matrix)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, blk_index(1), blk_index(2), nblock, transposed=tr, &
                                        row_size=blk_size(1), col_size=blk_size(2))
         CALL dbcsr_get_stored_coordinates(matrix_out, blk_index(1), blk_index(2), iproc)
         num_send(2*iproc) = num_send(2*iproc) + PRODUCT(blk_size)
         num_send(2*iproc + 1) = num_send(2*iproc + 1) + 1
      ENDDO
      CALL dbcsr_iterator_stop(iter)

      CALL timeset(routineN//"_alltoall", handle2)
      CALL mp_alltoall(num_send, num_rec, 2, mp_comm)
      CALL timestop(handle2)

      DO iproc = 0, numnodes - 1
         num_entries_recv(iproc) = num_rec(2*iproc)
         num_blocks_recv(iproc) = num_rec(2*iproc + 1)
         num_entries_send(iproc) = num_send(2*iproc)
         num_blocks_send(iproc) = num_send(2*iproc + 1)

         CALL block_buffer_create(buffer_send(iproc), num_blocks_send(iproc), num_entries_send(iproc), &
                                  data_type)

         CALL block_buffer_create(buffer_recv(iproc), num_blocks_recv(iproc), num_entries_recv(iproc), &
                                  data_type)

      ENDDO

      CALL dbcsr_data_init(block)
      CALL dbcsr_data_new(block, dbcsr_type_1d_to_2d(data_type))
      CALL dbcsr_iterator_start(iter, matrix_in%matrix)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, blk_index(1), blk_index(2), block, tr, &
                                        row_size=blk_size(1), col_size=blk_size(2))
         CALL dbcsr_get_stored_coordinates(matrix_out, blk_index(1), blk_index(2), iproc)
         CALL block_buffer_add_block(buffer_send(iproc), INT(blk_index, KIND=int_8), block)
      ENDDO

      CALL dbcsr_iterator_stop(iter)

      IF (move_prv) CALL dbcsr_tas_clear(matrix_in)

      CALL timeset(routineN//"_communicate_buffer", handle2)
      CALL dbcsr_tas_communicate_buffer(mp_comm, buffer_recv, buffer_send, req_array)

      DO iproc = 0, numnodes - 1
         CALL block_buffer_destroy(buffer_send(iproc))
      ENDDO

      CALL timestop(handle2)

      nblk = SUM(num_blocks_recv)
      ALLOCATE (blks_to_allocate(nblk, 2))

      bcount = 0
      DO iproc = 0, numnodes - 1
         CALL block_buffer_get_index(buffer_recv(iproc), index_recv)
         blks_to_allocate(bcount + 1:bcount + SIZE(index_recv, 1), :) = INT(index_recv(:, :))
         bcount = bcount + SIZE(index_recv, 1)
         DEALLOCATE (index_recv)
      ENDDO

      CALL index_unique(blks_to_allocate, blks_to_allocate_u)

      CALL dbcsr_reserve_blocks(matrix_out, blks_to_allocate_u(:, 1), blks_to_allocate_u(:, 2))
      DEALLOCATE (blks_to_allocate, blks_to_allocate_u)

      DO iproc = 0, numnodes - 1
         ! First, we need to get the index to create block
         DO WHILE (block_buffer_blocks_left(buffer_recv(iproc)))
            CALL block_buffer_get_next_block(buffer_recv(iproc), ndata, blk_index_i8)
            blk_size(1) = dbcsr_blk_row_size(matrix_out, INT(blk_index_i8(1)))
            blk_size(2) = dbcsr_blk_column_size(matrix_out, INT(blk_index_i8(2)))
            CALL internal_data_allocate(block%d, blk_size)
            CALL block_buffer_get_next_block(buffer_recv(iproc), ndata, blk_index_i8, block)
            CALL dbcsr_put_block(matrix_out, INT(blk_index_i8(1)), INT(blk_index_i8(2)), block, summation=.TRUE.)
            CALL internal_data_deallocate(block%d)
         ENDDO
         CALL block_buffer_destroy(buffer_recv(iproc))
      ENDDO
      CALL dbcsr_data_clear_pointer(block)
      CALL dbcsr_data_release(block)

      CALL dbcsr_finalize(matrix_out)

      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE block_buffer_get_index(buffer, index)
      !! get all indices from buffer
      TYPE(block_buffer_type), INTENT(IN)                :: buffer
      INTEGER(KIND=int_8), ALLOCATABLE, &
         DIMENSION(:, :), INTENT(OUT)                    :: index

      INTEGER, DIMENSION(2)                              :: indx_shape
      CHARACTER(LEN=*), PARAMETER :: routineN = 'block_buffer_get_index', &
                                     routineP = moduleN//':'//routineN
      INTEGER :: handle

      CALL timeset(routineN, handle)

      indx_shape = SHAPE(buffer%indx) - [0, 1]
      ALLOCATE (INDEX(indx_shape(1), indx_shape(2)))
      INDEX(:, :) = buffer%indx(1:indx_shape(1), 1:indx_shape(2))
      CALL timestop(handle)
   END SUBROUTINE

   SUBROUTINE block_buffer_iterator_reset(buffer)
      !! Reset buffer block iterator
      TYPE(block_buffer_type), INTENT(INOUT)             :: buffer

      buffer%endpos = 0
   END SUBROUTINE

   PURE FUNCTION block_buffer_blocks_left(buffer)
      !! how many blocks left in iterator
      TYPE(block_buffer_type), INTENT(IN)                :: buffer
      LOGICAL                                            :: block_buffer_blocks_left

      block_buffer_blocks_left = buffer%endpos .LT. buffer%nblock
   END FUNCTION

   SUBROUTINE block_buffer_create(buffer, nblock, ndata, data_type)
      !! Create block buffer for MPI communication.

      TYPE(block_buffer_type), INTENT(OUT) :: buffer
         !! block buffer
      INTEGER, INTENT(IN) :: nblock, ndata, data_type
         !! number of blocks
         !! total number of block entries

      buffer%nblock = nblock
      buffer%data_type = data_type
      buffer%endpos = 0
      SELECT CASE (data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
      CASE (${dparam}$)
         ALLOCATE (buffer%msg_${dsuffix}$ (ndata))
#:endfor
      END SELECT
      ALLOCATE (buffer%indx(nblock, 3))
   END SUBROUTINE

   SUBROUTINE block_buffer_destroy(buffer)
      TYPE(block_buffer_type), INTENT(INOUT) :: buffer

      SELECT CASE (buffer%data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
      CASE (${dparam}$)
         DEALLOCATE (buffer%msg_${dsuffix}$)
#:endfor
      END SELECT
      DEALLOCATE (buffer%indx)
      buffer%nblock = -1
      buffer%data_type = -1
      buffer%endpos = -1
   END SUBROUTINE block_buffer_destroy

   SUBROUTINE block_buffer_add_area_block(buffer, index, block, transposed)
      TYPE(block_buffer_type), INTENT(INOUT)      :: buffer
      INTEGER(KIND=int_8), DIMENSION(2), &
         INTENT(IN)                               :: index
      TYPE(dbcsr_data_obj), INTENT(IN)         :: block
      LOGICAL, INTENT(IN), OPTIONAL :: transposed

#:for dparam, dtype, dsuffix in dtype_float_list
      ${dtype}$, DIMENSION(:, :), POINTER :: block_${dsuffix}$ => NULL()
#:endfor

      SELECT CASE (buffer%data_type)
#:for dparam, dtype, dsuffix, dsuffix_dbcsr in dtype_float_list_dbcsr
      CASE (${dparam}$)
         block_${dsuffix}$ => dbcsr_get_data_p_2d_${dsuffix_dbcsr}$ (block)
         CALL block_buffer_add_block_${dsuffix}$ (buffer, index, block_${dsuffix}$, transposed)
#:endfor
      END SELECT
   END SUBROUTINE

   SUBROUTINE block_buffer_get_next_area_block(buffer, ndata, index, block, advance_iter)
      TYPE(block_buffer_type), INTENT(INOUT)      :: buffer
      INTEGER, INTENT(OUT)                        :: ndata
      INTEGER(KIND=int_8), DIMENSION(2), &
         INTENT(OUT)                              :: index
      TYPE(dbcsr_data_obj), INTENT(INOUT), OPTIONAL           :: block
      LOGICAL, INTENT(IN), OPTIONAL               :: advance_iter
      LOGICAL :: valid
      INTEGER, DIMENSION(2) :: sizes
#:for dparam, dtype, dsuffix in dtype_float_list
      ${dtype}$, DIMENSION(:, :), POINTER :: data_${dsuffix}$ => NULL()
#:endfor
      INTEGER :: handle

      CHARACTER(LEN=*), PARAMETER :: routineN = 'block_buffer_get_next_area_block', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)
      IF (PRESENT(block)) THEN
         CALL dbcsr_data_get_sizes(block, sizes, valid)
         DBCSR_ASSERT(valid)
         CALL internal_data_deallocate(block%d)
      ENDIF

      SELECT CASE (buffer%data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
      CASE (${dparam}$)
         IF (PRESENT(block)) THEN
            ALLOCATE (data_${dsuffix}$ (sizes(1), sizes(2)))
            CALL block_buffer_get_next_block_${dsuffix}$ (buffer, ndata, index, data_${dsuffix}$, advance_iter=advance_iter)
            CALL dbcsr_data_set_pointer(block, data_${dsuffix}$)
         ELSE
            CALL block_buffer_get_next_block_${dsuffix}$ (buffer, ndata, index, advance_iter=advance_iter)
         ENDIF
#:endfor
      END SELECT
      CALL timestop(handle)
   END SUBROUTINE

#:for dparam, dtype, dsuffix in dtype_float_list
   SUBROUTINE block_buffer_add_block_${dsuffix}$ (buffer, index, block, transposed)
      !! insert a block into block buffer (at current iterator position)

      TYPE(block_buffer_type), INTENT(INOUT)      :: buffer
      INTEGER(KIND=int_8), DIMENSION(2), &
         INTENT(IN)                               :: index
         !! index of block
      ${dtype}$, DIMENSION(:, :), INTENT(IN)                  :: block
         !! block
      LOGICAL, INTENT(IN), OPTIONAL :: transposed
      INTEGER(KIND=int_8), DIMENSION(2)   :: index_prv

      LOGICAL :: tr
      INTEGER :: p
      INTEGER :: ndata
      INTEGER :: p_data

      IF (PRESENT(transposed)) THEN
         tr = transposed
      ELSE
         tr = .FALSE.
      ENDIF

      index_prv(:) = index(:)
      IF (tr) THEN
         CALL swap(index_prv)
      ENDIF
      ndata = PRODUCT(SHAPE(block))

      DBCSR_ASSERT(buffer%data_type .EQ. ${dparam}$)
      p = buffer%endpos
      IF (p .EQ. 0) THEN
         p_data = 0
      ELSE
         p_data = INT(buffer%indx(p, 3))
      ENDIF

      IF (tr) THEN
         buffer%msg_${dsuffix}$ (p_data + 1:p_data + ndata) = RESHAPE(TRANSPOSE(block), [ndata])
      ELSE
         buffer%msg_${dsuffix}$ (p_data + 1:p_data + ndata) = RESHAPE(block, [ndata])
      ENDIF

      buffer%indx(p + 1, 1:2) = index_prv(:)
      IF (p > 0) THEN
         buffer%indx(p + 1, 3) = buffer%indx(p, 3) + INT(ndata, KIND=int_8)
      ELSE
         buffer%indx(p + 1, 3) = INT(ndata, KIND=int_8)
      ENDIF
      buffer%endpos = buffer%endpos + 1
   END SUBROUTINE
#:endfor

#:for dparam, dtype, dsuffix in dtype_float_list
   SUBROUTINE block_buffer_get_next_block_${dsuffix}$ (buffer, ndata, index, block, advance_iter)
      !! get next block from buffer. Iterator is advanced only if block is retrieved or advance_iter.
      TYPE(block_buffer_type), INTENT(INOUT)      :: buffer
      INTEGER, INTENT(OUT)                        :: ndata
      INTEGER(KIND=int_8), DIMENSION(2), &
         INTENT(OUT)                              :: index
      ${dtype}$, DIMENSION(:, :), INTENT(OUT), OPTIONAL     :: block
      LOGICAL, INTENT(IN), OPTIONAL               :: advance_iter
      INTEGER :: p, p_data
      LOGICAL :: do_advance

      do_advance = .FALSE.
      IF (PRESENT(advance_iter)) THEN
         do_advance = advance_iter
      ELSE IF (PRESENT(block)) THEN
         do_advance = .TRUE.
      ENDIF

      DBCSR_ASSERT(buffer%data_type .EQ. ${dparam}$)

      p = buffer%endpos
      IF (p .EQ. 0) THEN
         p_data = 0
      ELSE
         p_data = INT(buffer%indx(p, 3))
      ENDIF

      IF (p > 0) THEN
         ndata = INT(buffer%indx(p + 1, 3) - buffer%indx(p, 3))
      ELSE
         ndata = INT(buffer%indx(p + 1, 3))
      ENDIF
      index(:) = buffer%indx(p + 1, 1:2)

      IF (PRESENT(block)) THEN
         block(:, :) = RESHAPE(buffer%msg_${dsuffix}$ (p_data + 1:p_data + ndata), SHAPE(block))
      ENDIF

      IF (do_advance) buffer%endpos = buffer%endpos + 1
   END SUBROUTINE
#:endfor

   SUBROUTINE dbcsr_tas_communicate_buffer(mp_comm, buffer_recv, buffer_send, req_array)
      !! communicate buffer
      INTEGER, INTENT(IN)                    :: mp_comm
      TYPE(block_buffer_type), DIMENSION(0:), INTENT(INOUT) :: buffer_recv, buffer_send
      INTEGER, DIMENSION(:, :), INTENT(OUT)               :: req_array

      INTEGER                                :: iproc, mynode, numnodes, rec_counter, &
                                                send_counter
      INTEGER                                   :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_communicate_buffer', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)
      CALL mp_environ(numnodes, mynode, mp_comm)

      IF (numnodes > 1) THEN

         send_counter = 0
         rec_counter = 0

         DO iproc = 0, numnodes - 1
            IF (buffer_recv(iproc)%nblock > 0) THEN
               rec_counter = rec_counter + 1
               CALL mp_irecv(buffer_recv(iproc)%indx, iproc, mp_comm, req_array(rec_counter, 3), tag=4)
               SELECT CASE (buffer_recv(iproc)%data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
               CASE (${dparam}$)
                  CALL mp_irecv(buffer_recv(iproc)%msg_${dsuffix}$, iproc, mp_comm, req_array(rec_counter, 4), tag=7)
#:endfor
               END SELECT
            END IF
         END DO

         DO iproc = 0, numnodes - 1
            IF (buffer_send(iproc)%nblock > 0) THEN
               send_counter = send_counter + 1
               CALL mp_isend(buffer_send(iproc)%indx, iproc, mp_comm, req_array(send_counter, 1), tag=4)
               SELECT CASE (buffer_recv(iproc)%data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
               CASE (${dparam}$)
                  CALL mp_isend(buffer_send(iproc)%msg_${dsuffix}$, iproc, mp_comm, req_array(send_counter, 2), tag=7)
#:endfor
               END SELECT
            END IF
         END DO

         IF (send_counter > 0) THEN
            CALL mp_waitall(req_array(1:send_counter, 1:2))
         ENDIF
         IF (rec_counter > 0) THEN
            CALL mp_waitall(req_array(1:rec_counter, 3:4))
         ENDIF

      ELSE
         IF (buffer_recv(0)%nblock > 0) THEN
            buffer_recv(0)%indx(:, :) = buffer_send(0)%indx(:, :)
            SELECT CASE (buffer_recv(0)%data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
            CASE (${dparam}$)
               buffer_recv(0)%msg_${dsuffix}$ (:) = buffer_send(0)%msg_${dsuffix}$ (:)
#:endfor
            END SELECT
         ENDIF
      ENDIF
      CALL timestop(handle)
   END SUBROUTINE

END MODULE
