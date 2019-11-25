!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tensor_reshape
   !! Routines to reshape / redistribute tensors

#:include "dbcsr_tensor.fypp"
#:set maxdim = maxrank
#:set ndims = range(2,maxdim+1)

   USE dbcsr_allocate_wrap, ONLY: allocate_any
   USE dbcsr_tas_base, ONLY: dbcsr_tas_copy, dbcsr_tas_get_info, dbcsr_tas_info
   USE dbcsr_tensor_block, ONLY: &
      block_nd, create_block, destroy_block, dbcsr_t_iterator_type, dbcsr_t_iterator_next_block, &
      dbcsr_t_iterator_blocks_left, dbcsr_t_iterator_start, dbcsr_t_iterator_stop, dbcsr_t_get_block, &
      dbcsr_t_reserve_blocks, dbcsr_t_put_block, reshape_2d_to_nd_block
   USE dbcsr_tensor_types, ONLY: dbcsr_t_blk_sizes, &
                                 dbcsr_t_create, &
                                 dbcsr_t_get_data_type, &
                                 dbcsr_t_type, &
                                 ndims_tensor, &
                                 dbcsr_t_get_stored_coordinates, &
                                 dbcsr_t_clear
   USE dbcsr_kinds, ONLY: default_string_length
   USE dbcsr_kinds, ONLY: ${uselist(dtype_float_prec)}$
   USE dbcsr_api, ONLY: ${uselist(dtype_float_param)}$
   USE dbcsr_mpiwrap, ONLY: mp_alltoall, &
                            mp_environ, &
                            mp_irecv, &
                            mp_isend, &
                            mp_waitall

#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tensor_reshape'

   PUBLIC :: &
      dbcsr_t_reshape

   TYPE block_buffer_type
      INTEGER                                    :: ndim = -1
      INTEGER                                    :: nblock = -1
      INTEGER, DIMENSION(:, :), ALLOCATABLE      :: indx
#:for dparam, dtype, dsuffix in dtype_float_list
      ${dtype}$, DIMENSION(:), ALLOCATABLE       :: msg_${dsuffix}$
#:endfor
      INTEGER                                    :: data_type = -1
      INTEGER                                    :: endpos = -1
   END TYPE

   INTERFACE block_buffer_add_block
#:for dparam, dtype, dsuffix in dtype_float_list
      MODULE PROCEDURE block_buffer_add_block_${dsuffix}$
#:endfor
   END INTERFACE

CONTAINS

   SUBROUTINE dbcsr_t_reshape(tensor_in, tensor_out, summation, move_data)
      !! copy data (involves reshape)

      TYPE(dbcsr_t_type), INTENT(INOUT)               :: tensor_in, tensor_out
      LOGICAL, INTENT(IN), OPTIONAL                    :: summation
         !! tensor_out = tensor_out + tensor_in move_data memory optimization: transfer data from tensor_in to tensor_out s.t.
         !! tensor_in is empty on return
      LOGICAL, INTENT(IN), OPTIONAL                    :: move_data

      INTEGER                                            :: blk, iproc, mp_comm, mynode, ndata, &
                                                            numnodes, bcount, nblk
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: num_blocks_recv, num_blocks_send, &
                                                            num_entries_recv, num_entries_send, &
                                                            num_rec, num_send
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: req_array, index_recv, blks_to_allocate
      TYPE(dbcsr_t_iterator_type)                        :: iter
      TYPE(block_nd)                                     :: blk_data
      TYPE(block_buffer_type), ALLOCATABLE, DIMENSION(:) :: buffer_recv, buffer_send
      INTEGER, DIMENSION(ndims_tensor(tensor_in))       :: blk_size, ind_nd, index
      LOGICAL :: found, summation_prv, move_prv

      IF (PRESENT(summation)) THEN
         summation_prv = summation
      ELSE
         summation_prv = .FALSE.
      ENDIF

      IF (PRESENT(move_data)) THEN
         move_prv = move_data
      ELSE
         move_prv = .FALSE.
      ENDIF

      DBCSR_ASSERT(tensor_out%valid)

      IF (.NOT. summation_prv) CALL dbcsr_t_clear(tensor_out)

      mp_comm = tensor_in%pgrid%mp_comm_2d
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
      CALL dbcsr_t_iterator_start(iter, tensor_in)
      DO WHILE (dbcsr_t_iterator_blocks_left(iter))
         CALL dbcsr_t_iterator_next_block(iter, ind_nd, blk, blk_size=blk_size)
         CALL dbcsr_t_get_stored_coordinates(tensor_out, ind_nd, iproc)
         num_send(2*iproc) = num_send(2*iproc) + PRODUCT(blk_size)
         num_send(2*iproc + 1) = num_send(2*iproc + 1) + 1
      ENDDO
      CALL dbcsr_t_iterator_stop(iter)
      CALL mp_alltoall(num_send, num_rec, 2, mp_comm)
      DO iproc = 0, numnodes - 1
         num_entries_recv(iproc) = num_rec(2*iproc)
         num_blocks_recv(iproc) = num_rec(2*iproc + 1)
         num_entries_send(iproc) = num_send(2*iproc)
         num_blocks_send(iproc) = num_send(2*iproc + 1)

         CALL block_buffer_create(buffer_send(iproc), num_blocks_send(iproc), num_entries_send(iproc), &
                                  dbcsr_t_get_data_type(tensor_in), ndims_tensor(tensor_in))
         CALL block_buffer_create(buffer_recv(iproc), num_blocks_recv(iproc), num_entries_recv(iproc), &
                                  dbcsr_t_get_data_type(tensor_in), ndims_tensor(tensor_in))
      ENDDO
      CALL dbcsr_t_iterator_start(iter, tensor_in)
      DO WHILE (dbcsr_t_iterator_blocks_left(iter))
         CALL dbcsr_t_iterator_next_block(iter, ind_nd, blk, blk_size=blk_size)
         CALL dbcsr_t_get_block(tensor_in, ind_nd, blk_data, found)
         DBCSR_ASSERT(found)
         CALL dbcsr_t_get_stored_coordinates(tensor_out, ind_nd, iproc)
         CALL block_buffer_add_anyd_block(buffer_send(iproc), ind_nd, blk_data)
         CALL destroy_block(blk_data)
      ENDDO
      CALL dbcsr_t_iterator_stop(iter)

      IF (move_prv) CALL dbcsr_t_clear(tensor_in)

      CALL dbcsr_t_communicate_buffer(mp_comm, buffer_recv, buffer_send, req_array)
      DO iproc = 0, numnodes - 1
         CALL block_buffer_destroy(buffer_send(iproc))
      ENDDO

      nblk = SUM(num_blocks_recv)
      ALLOCATE (blks_to_allocate(nblk, ndims_tensor(tensor_in)))

      bcount = 0
      DO iproc = 0, numnodes - 1
         CALL block_buffer_get_index(buffer_recv(iproc), index_recv)
         blks_to_allocate(bcount + 1:bcount + SIZE(index_recv, 1), :) = index_recv(:, :)
         bcount = bcount + SIZE(index_recv, 1)
         DEALLOCATE (index_recv)
      ENDDO

      CALL dbcsr_t_reserve_blocks(tensor_out, blks_to_allocate)
      DEALLOCATE (blks_to_allocate)

      DO iproc = 0, numnodes - 1
         DO WHILE (block_buffer_blocks_left(buffer_recv(iproc)))
            CALL block_buffer_get_next_anyd_block(buffer_recv(iproc), ndata, index)
            CALL dbcsr_t_blk_sizes(tensor_in, index, blk_size)
            ! create block
            CALL create_block(blk_data, blk_size, dbcsr_t_get_data_type(tensor_in))
            ! get actual block data
            CALL block_buffer_get_next_anyd_block(buffer_recv(iproc), ndata, index, blk_data)
            CALL dbcsr_t_put_block(tensor_out, index, blk_data, summation=summation)
            CALL destroy_block(blk_data)
         ENDDO
         CALL block_buffer_destroy(buffer_recv(iproc))
      ENDDO
   END SUBROUTINE

   SUBROUTINE block_buffer_create(buffer, nblock, ndata, data_type, ndim)
      !! Create block buffer for MPI communication.

      TYPE(block_buffer_type), INTENT(OUT) :: buffer
         !! block buffer
      INTEGER, INTENT(IN)                  :: nblock, ndata, data_type, ndim
         !! number of blocks
         !! total number of block entries
         !! number of dimensions

      buffer%nblock = nblock
      buffer%data_type = data_type
      buffer%endpos = 0
      buffer%ndim = ndim
      SELECT CASE (data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
      CASE (${dparam}$)
         ALLOCATE (buffer%msg_${dsuffix}$ (ndata))
#:endfor
      END SELECT
      ALLOCATE (buffer%indx(nblock, ndim + 1))
   END SUBROUTINE block_buffer_create

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
      buffer%ndim = -1
      buffer%endpos = -1
   END SUBROUTINE block_buffer_destroy

   PURE FUNCTION ndims_buffer(buffer)
      TYPE(block_buffer_type), INTENT(IN) :: buffer
      INTEGER                             :: ndims_buffer

      ndims_buffer = buffer%ndim
   END FUNCTION

   SUBROUTINE block_buffer_add_anyd_block(buffer, index, block)
      !! insert a block into block buffer (at current iterator position)

      TYPE(block_buffer_type), INTENT(INOUT)      :: buffer
      INTEGER, DIMENSION(ndims_buffer(buffer)), &
         INTENT(IN)                               :: index
         !! index of block
      TYPE(block_nd), INTENT(IN)                  :: block
         !! block

      SELECT CASE (block%data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
      CASE (${dparam}$)
         CALL block_buffer_add_block_${dsuffix}$ (buffer, SIZE(block%${dsuffix}$%blk), index, block%${dsuffix}$%blk)
#:endfor
      END SELECT
   END SUBROUTINE

   SUBROUTINE block_buffer_get_next_anyd_block(buffer, ndata, index, block, advance_iter)
      !! get next block from buffer. Iterator is advanced only if block is retrieved or advance_iter.
      TYPE(block_buffer_type), INTENT(INOUT)      :: buffer
      INTEGER, INTENT(OUT)                        :: ndata
      INTEGER, DIMENSION(ndims_buffer(buffer)), &
         INTENT(OUT)                              :: index
      TYPE(block_nd), INTENT(INOUT), OPTIONAL     :: block
      LOGICAL, INTENT(IN), OPTIONAL               :: advance_iter

      SELECT CASE (buffer%data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
      CASE (${dparam}$)
         IF (PRESENT(block)) THEN
            CALL block_buffer_get_next_block_${dsuffix}$ (buffer, ndata, index, block%${dsuffix}$%blk, advance_iter=advance_iter)
         ELSE
            CALL block_buffer_get_next_block_${dsuffix}$ (buffer, ndata, index, advance_iter=advance_iter)
         ENDIF
#:endfor
      END SELECT
   END SUBROUTINE

   SUBROUTINE block_buffer_get_index(buffer, index)
      !! Get all indices from buffer
      TYPE(block_buffer_type), INTENT(IN)               :: buffer
      INTEGER, INTENT(OUT), DIMENSION(:, :), ALLOCATABLE :: index
      INTEGER, DIMENSION(2)                             :: indx_shape

      indx_shape = SHAPE(buffer%indx) - [0, 1]
      CALL allocate_any(index, source=buffer%indx(1:indx_shape(1), 1:indx_shape(2)))
   END SUBROUTINE

   SUBROUTINE block_buffer_iterator_reset(buffer)
      !! Reset buffer block iterator
      TYPE(block_buffer_type), INTENT(INOUT) :: buffer

      buffer%endpos = 0
   END SUBROUTINE

   PURE FUNCTION block_buffer_blocks_left(buffer)
      !! how many blocks left in iterator
      TYPE(block_buffer_type), INTENT(IN) :: buffer
      LOGICAL                             :: block_buffer_blocks_left

      block_buffer_blocks_left = buffer%endpos .LT. buffer%nblock
   END FUNCTION

   SUBROUTINE dbcsr_t_communicate_buffer(mp_comm, buffer_recv, buffer_send, req_array)
      !! communicate buffer
      INTEGER, INTENT(IN)                    :: mp_comm
      TYPE(block_buffer_type), DIMENSION(0:), INTENT(INOUT) :: buffer_recv, buffer_send
      INTEGER, DIMENSION(:, :), INTENT(OUT)               :: req_array

      INTEGER                                :: iproc, mynode, numnodes, rec_counter, &
                                                send_counter
      INTEGER                                   :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_t_communicate_buffer', &
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

#:for dparam, dtype, dsuffix in dtype_float_list
   SUBROUTINE block_buffer_add_block_${dsuffix}$ (buffer, ndata, index, block)
      !! add block to buffer.
      TYPE(block_buffer_type), INTENT(INOUT)               :: buffer
      INTEGER, INTENT(IN)                                  :: ndata
      ${dtype}$, DIMENSION(ndata), INTENT(IN)              :: block
      INTEGER, DIMENSION(ndims_buffer(buffer)), INTENT(IN) :: index
      INTEGER                                              :: p, ndims, p_data
      DBCSR_ASSERT(buffer%data_type .EQ. ${dparam}$)
      ndims = ndims_buffer(buffer)
      p = buffer%endpos
      IF (p .EQ. 0) THEN
         p_data = 0
      ELSE
         p_data = buffer%indx(p, ndims + 1)
      ENDIF

      buffer%msg_${dsuffix}$ (p_data + 1:p_data + ndata) = block(:)
      buffer%indx(p + 1, 1:ndims) = index(:)
      IF (p > 0) THEN
         buffer%indx(p + 1, ndims + 1) = buffer%indx(p, ndims + 1) + ndata
      ELSE
         buffer%indx(p + 1, ndims + 1) = ndata
      ENDIF
      buffer%endpos = buffer%endpos + 1
   END SUBROUTINE
#:endfor

#:for dparam, dtype, dsuffix in dtype_float_list
   SUBROUTINE block_buffer_get_next_block_${dsuffix}$ (buffer, ndata, index, block, advance_iter)
      !! get next block from buffer. Iterator is advanced only if block is retrieved or advance_iter.

      TYPE(block_buffer_type), INTENT(INOUT)                      :: buffer
      INTEGER, INTENT(OUT)                                        :: ndata
      ${dtype}$, DIMENSION(:), ALLOCATABLE, OPTIONAL, INTENT(OUT) :: block
      INTEGER, DIMENSION(ndims_buffer(buffer)), INTENT(OUT)       :: index
      INTEGER                                                     :: p, ndims, p_data
      LOGICAL, INTENT(IN), OPTIONAL                               :: advance_iter
      LOGICAL                                                     :: do_advance

      do_advance = .FALSE.
      IF (PRESENT(advance_iter)) THEN
         do_advance = advance_iter
      ELSE IF (PRESENT(block)) THEN
         do_advance = .TRUE.
      ENDIF

      DBCSR_ASSERT(buffer%data_type .EQ. ${dparam}$)
      ndims = ndims_buffer(buffer)
      p = buffer%endpos
      IF (p .EQ. 0) THEN
         p_data = 0
      ELSE
         p_data = buffer%indx(p, ndims + 1)
      ENDIF
      IF (p > 0) THEN
         ndata = buffer%indx(p + 1, ndims + 1) - buffer%indx(p, ndims + 1)
      ELSE
         ndata = buffer%indx(p + 1, ndims + 1)
      ENDIF
      index(:) = buffer%indx(p + 1, 1:ndims)
      IF (PRESENT(block)) THEN
         CALL allocate_any(block, source=buffer%msg_${dsuffix}$ (p_data + 1:p_data + ndata))
      ENDIF

      IF (do_advance) buffer%endpos = buffer%endpos + 1
   END SUBROUTINE
#:endfor

END MODULE dbcsr_tensor_reshape
