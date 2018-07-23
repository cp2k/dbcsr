!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright (C) 2000 - 2018  CP2K developers group                                               !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Routines to reshape / redistribute tensors
!> \author Patrick Seewald
! **************************************************************************************************
MODULE dbcsr_reshape

#:include "../data/dbcsr.fypp"


   !USE allocate_wrap,                   ONLY: allocate_any
   use dbcsr_operations,                only: dbcsr_copy, dbcsr_get_info

   USE dbcsr_iterator_operations,       ONLY: dbcsr_iterator_blocks_left,&
                                              dbcsr_iterator_next_block,&
                                              dbcsr_iterator_start,&
                                              dbcsr_iterator_stop
   
   USE dbcsr_block_access,              ONLY: dbcsr_get_block_p, &
                                              dbcsr_reserve_blocks, dbcsr_put_block
  
   USE dbcsr_types,                     only: dbcsr_iterator,&
                                              dbcsr_type
                                              
   use dbcsr_work_operations            only: dbcsr_create                        
   use dbcsr_dist_operations,           only: dbcsr_get_stored_coordinates
   USE dbcsr_methods,                   ONLY: dbcsr_get_data_type
                                                  
!   USE dbcsr_tensor_types,              ONLY: &
!                                              dbcsr_t_create,&
!                                              dbcsr_t_get_data_type,&
!                                              dbcsr_t_type,&
!                                              ndims_tensor,&
!                                              dbcsr_t_get_stored_coordinates
                                              
   USE dbcsr_kinds,                           ONLY: default_string_length
   USE dbcsr_kinds,                           ONLY: ${uselist(kind1)}$
   USE dbcsr_kinds,                           ONLY: ${uselist(dkind1)}$
   USE dbcsr_message_passing,                 ONLY: mp_alltoall,&
                                                    mp_environ,&
                                                    mp_irecv,&
                                                    mp_isend,&
                                                    mp_waitall


#include "../base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_reshape'

   PUBLIC :: &
      dbcsr_reshape

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

! **************************************************************************************************
!> \brief copy data (involves reshape)
!> \param matrix_in ...
!> \param matrix_out ...
! **************************************************************************************************
   SUBROUTINE dbcsr_reshape(matrix_in, matrix_out)

      TYPE(dbcsr_type), INTENT(INOUT)               :: matrix_in, matrix_out

      INTEGER                                            :: blk, iproc, mp_comm, mynode, ndata, &
                                                            numnodes
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: num_blocks_recv, num_blocks_send, &
                                                            num_entries_recv, num_entries_send, &
                                                            num_rec, num_send
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: req_array, index_recv
      TYPE(dbcsr_iterator)                               :: iter
      TYPE(dbcsr_data_obj)                               :: blk_data
      TYPE(block_buffer_type), ALLOCATABLE, DIMENSION(:) :: buffer_recv, buffer_send
      
      ! temp hack
      INTEGER                                            :: ndims_tensor = 2
      !INTEGER, DIMENSION(ndims_tensor(matrix_in))       :: blk_size, ind_nd, index
      INTEGER, DIMENSION(ndims_tensor)       :: blk_size, ind_nd, index
      
      logical :: tr
      integer :: lb_b
      real(8), dimension(:), pointer :: p_blk
  
         
      IF (matrix_out%valid) THEN

         CALL dbcsr_get_info(matrix_in%matrix_rep, group=mp_comm)
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
         
         CALL dbcsr_iterator_start(iter, matrix_in)
         
         DO WHILE (dbcsr_iterator_blocks_left(iter))
            CALL dbcsr_iterator_next_block(iter, ind_nd(1), ind_nd(2), blk, tr, lb_b, blk_size(1), blk_size(2))
            CALL dbcsr_get_stored_coordinates(matrix_out, ind_nd(1), ind_nd(2), iproc)
            num_send(2*iproc) = num_send(2*iproc) + PRODUCT(blk_size)
            num_send(2*iproc + 1) = num_send(2*iproc + 1) + 1
         ENDDO
         
         CALL dbcsr_iterator_stop(iter)
         
         CALL mp_alltoall(num_send, num_rec, 2, mp_comm)
         
         DO iproc = 0, numnodes - 1
            num_entries_recv(iproc) = num_rec(2*iproc)
            num_blocks_recv(iproc) = num_rec(2*iproc + 1)
            num_entries_send(iproc) = num_send(2*iproc)
            num_blocks_send(iproc) = num_send(2*iproc + 1)

            CALL block_buffer_create(buffer_send(iproc), num_blocks_send(iproc), num_entries_send(iproc), &
                                     dbcsr_get_data_type(matrix_in), ndims_tensor)
            CALL block_buffer_create(buffer_recv(iproc), num_blocks_recv(iproc), num_entries_recv(iproc), &
                                     dbcsr_get_data_type(matrix_in), ndims_tensor)
         ENDDO
         
         CALL dbcsr_iterator_start(iter, matrix_in)
         
         DO WHILE (dbcsr_iterator_blocks_left(iter))
            CALL dbcsr_iterator_next_block(iter, ind_nd(1), ind_nd(2), blk, tr, lb_b, blk_size(1), blk_size(2))
            
            ! temp, TODO for fypp
            p_blk => matrix_in % data_area % d % r_dp (lb_b : PRODUCT(blk_size) + lb_b)
            call dbcsr_data_set_pointer(blk_data, p_blk)
            
            CALL dbcsr_get_stored_coordinates(matrix_out, ind_nd(1), ind_nd(2), iproc)
            
            CALL block_buffer_add_anyd_block(buffer_send(iproc), ind_nd, blk_data)
         ENDDO
         
         CALL dbcsr_iterator_stop(iter)

         CALL communicate_buffer(mp_comm, buffer_recv, buffer_send, req_array)

         DO iproc = 0, numnodes - 1
            ! First, we need to get the index to create block
            !TODO
            !CALL block_buffer_get_index(buffer_recv(iproc), index_recv)
            
            CALL dbcsr_reserve_blocks(matrix_out, buffer_recv(iproc) % indx(:,1), buffer_recv(iproc) % indx(:,2))
            
            DO WHILE (block_buffer_blocks_left(buffer_recv(iproc)))
               ! get actual block data
               call block_buffer_get_next_anyd_block(buffer_recv(iproc), ndata, index, blk_data)
               CALL dbcsr_put_block(matrix_out, index, blk_data)
            ENDDO
            CALL block_buffer_destroy(buffer_recv(iproc))
            CALL block_buffer_destroy(buffer_send(iproc))
         ENDDO
      ELSE
         !call dbcsr_finalize(matrix_out)
         !ALL dbcsr_t_create(matrix_in, matrix_out)
         !CALL dbcsr_t_reserve_blocks(matrix_in, matrix_out)
         !CALL dbcsr_copy(matrix_out%matrix_rep, matrix_in%matrix_rep, shallow_data=.TRUE.)
      ENDIF

   END SUBROUTINE

! **************************************************************************************************
!> \brief Create block buffer for MPI communication.
!> \param buffer block buffer
!> \param nblock number of blocks
!> \param ndata total number of block entries
!> \param data_type ...
!> \param ndim number of dimensions
! **************************************************************************************************
   SUBROUTINE block_buffer_create(buffer, nblock, ndata, data_type, ndim)
      TYPE(block_buffer_type), INTENT(OUT) :: buffer
      INTEGER, INTENT(IN)                  :: nblock, ndata, data_type, ndim

      buffer%nblock = nblock
      buffer%data_type = data_type
      buffer%endpos = 0
      buffer%ndim = ndim
      SELECT CASE (data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
      CASE (${dparam}$)
         ALLOCATE (buffer%msg_${dsuffix}$(ndata))
#:endfor
      END SELECT
      ALLOCATE (buffer%indx(nblock, ndim+1))
   END SUBROUTINE block_buffer_create

! **************************************************************************************************
!> \brief ...
!> \param buffer ...
! **************************************************************************************************
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

! **************************************************************************************************
!> \brief ...
!> \param buffer ...
!> \return ...
! **************************************************************************************************
   PURE FUNCTION ndims_buffer(buffer)
      TYPE(block_buffer_type), INTENT(IN) :: buffer
      INTEGER                             :: ndims_buffer

      ndims_buffer = buffer%ndim
   END FUNCTION

! **************************************************************************************************
!> \brief insert a block into block buffer (at current iterator position)
!> \param buffer ...
!> \param index index of block
!> \param block block
! **************************************************************************************************
   SUBROUTINE block_buffer_add_anyd_block(buffer, index, block)
      TYPE(block_buffer_type), INTENT(INOUT)      :: buffer
      INTEGER, DIMENSION(ndims_buffer(buffer)), &
         INTENT(IN)                               :: index
      TYPE(dbcsr_data_obj), INTENT(IN)                  :: block

      SELECT CASE (buffer%data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
      CASE (${dparam}$)
         CALL block_buffer_add_block_${dsuffix}$(buffer, SIZE(block % d % ${dsuffix}$), index, block % d % ${dsuffix}$)
#:endfor
      END SELECT
   END SUBROUTINE

! **************************************************************************************************
!> \brief get next block from buffer. Iterator is advanced only if block is retrieved or advance_iter.
!> \param buffer ...
!> \param ndata ...
!> \param index ...
!> \param block ...
!> \param advance_iter ...
! **************************************************************************************************
   SUBROUTINE block_buffer_get_next_anyd_block(buffer, ndata, index, block, advance_iter)
      TYPE(block_buffer_type), INTENT(INOUT)      :: buffer
      INTEGER, INTENT(OUT)                        :: ndata
      INTEGER, DIMENSION(ndims_buffer(buffer)), &
         INTENT(OUT)                              :: index
      TYPE(dbcsr_data_obj), INTENT(INOUT), OPTIONAL     :: block
      LOGICAL, INTENT(IN), OPTIONAL               :: advance_iter

      SELECT CASE (buffer%data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
      CASE (${dparam}$)
         IF (PRESENT(block)) THEN
            CALL block_buffer_get_next_block_${dsuffix}$(buffer, ndata, index, block % d % ${dsuffix}$, advance_iter=advance_iter)
         ELSE
            CALL block_buffer_get_next_block_${dsuffix}$(buffer, ndata, index, advance_iter=advance_iter)
         ENDIF
#:endfor
      END SELECT
   END SUBROUTINE

! **************************************************************************************************
!> \brief Get all indices from buffer
! **************************************************************************************************
!   SUBROUTINE block_buffer_get_index(buffer, index)
!      TYPE(block_buffer_type), INTENT(IN)               :: buffer
!      INTEGER, INTENT(OUT), DIMENSION(:,:), ALLOCATABLE :: index
!      INTEGER, DIMENSION(2)                             :: indx_shape
!
!      indx_shape = SHAPE(buffer%indx) - [0,1]
!      CALL allocate_any(index, source=buffer%indx(1:indx_shape(1), 1:indx_shape(2)))
!   END SUBROUTINE

! **************************************************************************************************
!> \brief Reset buffer block iterator
!> \param buffer ...
! **************************************************************************************************
   SUBROUTINE block_buffer_iterator_reset(buffer)
      TYPE(block_buffer_type), INTENT(INOUT) :: buffer

      buffer%endpos = 0
   END SUBROUTINE

! **************************************************************************************************
!> \brief how many blocks left in iterator
!> \param buffer ...
!> \return ...
! **************************************************************************************************
   PURE FUNCTION block_buffer_blocks_left(buffer)
      TYPE(block_buffer_type), INTENT(IN) :: buffer
      LOGICAL                             :: block_buffer_blocks_left

      block_buffer_blocks_left = buffer%endpos .LT. buffer%nblock
   END FUNCTION

! **************************************************************************************************
!> \brief communicate buffer
!> \param mp_comm ...
!> \param buffer_recv ...
!> \param buffer_send ...
!> \param req_array ...
! **************************************************************************************************
   SUBROUTINE communicate_buffer(mp_comm, buffer_recv, buffer_send, req_array)
      INTEGER, INTENT(IN)                    :: mp_comm
      TYPE(block_buffer_type), DIMENSION(0:), INTENT(INOUT) :: buffer_recv, buffer_send
      INTEGER, DIMENSION(:, :), INTENT(OUT)               :: req_array

      INTEGER                                :: iproc, mynode, numnodes, rec_counter, &
                                                send_counter
      INTEGER                                   :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'communicate_buffer', &
         routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)
      CALL mp_environ(numnodes, mynode, mp_comm)

      IF (numnodes > 1) THEN

         send_counter = 0
         rec_counter = 0

         DO iproc = 0, numnodes-1
            IF (buffer_recv(iproc)%nblock > 0) THEN
               rec_counter = rec_counter+1
               CALL mp_irecv(buffer_recv(iproc)%indx, iproc, mp_comm, req_array(rec_counter, 3), tag=4)
               SELECT CASE (buffer_recv (iproc)%data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
               CASE (${dparam}$)
                  CALL mp_irecv(buffer_recv(iproc)%msg_${dsuffix}$, iproc, mp_comm, req_array(rec_counter, 4), tag=7)
#:endfor
               END SELECT
            END IF
         END DO

         DO iproc = 0, numnodes-1
            IF (buffer_send(iproc)%nblock > 0) THEN
               send_counter = send_counter+1
               CALL mp_isend(buffer_send(iproc)%indx, iproc, mp_comm, req_array(send_counter, 1), tag=4)
               SELECT CASE (buffer_recv (iproc)%data_type)
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
            SELECT CASE (buffer_recv (0)%data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
            CASE (${dparam}$)
               buffer_recv(0)%msg_${dsuffix}$(:) = buffer_send(0)%msg_${dsuffix}$(:)
#:endfor
            END SELECT
         ENDIF
      ENDIF
      CALL timestop(handle)

   END SUBROUTINE

#:for dparam, dtype, dsuffix in dtype_float_list
! **************************************************************************************************
!> \brief add block to buffer.
!> \param buffer ...
!> \param ndata ...
!> \param index ...
!> \param block ...
! **************************************************************************************************
   SUBROUTINE block_buffer_add_block_${dsuffix}$(buffer, ndata, index, block)
      TYPE(block_buffer_type), INTENT(INOUT)               :: buffer
      INTEGER, INTENT(IN)                                  :: ndata
      ${dtype}$, DIMENSION(ndata), INTENT(IN)              :: block
      INTEGER, DIMENSION(ndims_buffer(buffer)), INTENT(IN) :: index
      INTEGER                                              :: p, ndims, p_data
      CPASSERT(buffer%data_type .EQ. ${dparam}$)
      ndims = ndims_buffer(buffer)
      p = buffer%endpos
      IF (p .EQ. 0) THEN
         p_data = 0
      ELSE
         p_data = buffer%indx(p, ndims+1)
      ENDIF

      buffer%msg_${dsuffix}$(p_data+1:p_data+ndata) = block(:)
      buffer%indx(p+1, 1:ndims) = index(:)
      IF (p > 0) THEN
         buffer%indx(p+1,ndims+1) = buffer%indx(p,ndims+1)+ndata
      ELSE
         buffer%indx(p+1, ndims+1) = ndata
      ENDIF
      buffer%endpos = buffer%endpos+1
   END SUBROUTINE
#:endfor

#:for dparam, dtype, dsuffix in dtype_float_list
! **************************************************************************************************
!> \brief get next block from buffer. Iterator is advanced only if block is retrieved or advance_iter.
!> \param buffer ...
!> \param ndata ...
!> \param index ...
!> \param block ...
!> \param advance_iter
! **************************************************************************************************
   SUBROUTINE block_buffer_get_next_block_${dsuffix}$(buffer, ndata, index, block, advance_iter)
      TYPE(block_buffer_type), INTENT(INOUT)                      :: buffer
      INTEGER, INTENT(OUT)                                        :: ndata
      ${dtype}$, DIMENSION(:), pointer, OPTIONAL, INTENT(INOUT)   :: block
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

      CPASSERT(buffer%data_type .EQ. ${dparam}$)
      ndims = ndims_buffer(buffer)
      p = buffer%endpos
      IF (p .EQ. 0) THEN
         p_data = 0
      ELSE
         p_data = buffer%indx(p, ndims+1)
      ENDIF
      IF (p > 0) THEN
         ndata = buffer%indx(p+1, ndims+1)-buffer%indx(p, ndims+1)
      ELSE
         ndata = buffer%indx(p+1, ndims+1)
      ENDIF
      index(:) = buffer%indx(p+1,1:ndims)
      IF (PRESENT(block)) THEN
         block => buffer % msg_${dsuffix}$(p_data+1 : p_data+ndata)
         !CALL allocate_any(block, source=buffer%msg_${dsuffix}$(p_data+1:p_data+ndata))
      ENDIF

      IF(do_advance) buffer%endpos = buffer%endpos+1
   END SUBROUTINE
#:endfor

END MODULE dbcsr_reshape
