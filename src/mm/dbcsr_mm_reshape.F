!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright (C) 2000 - 2018  CP2K developers group                                               !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Routines to reshape / redistribute tensors
!> \author Patrick Seewald, Ilia Sivkov
! **************************************************************************************************
MODULE dbcsr_mm_reshape

#:include "../data/dbcsr.fypp"

   USE dbcsr_array_types,               ONLY: array_i1d_obj
   use dbcsr_operations,                only: dbcsr_get_info, &
                                              dbcsr_add_iter_s, &
                                              dbcsr_add_iter_d, &
                                              dbcsr_add_iter_c, &
                                              dbcsr_add_iter_z
   USE dbcsr_iterator_operations,       ONLY: dbcsr_iterator_blocks_left,&
                                              dbcsr_iterator_next_block,&
                                              dbcsr_iterator_start,&
                                              dbcsr_iterator_stop
   USE dbcsr_block_access,              ONLY: dbcsr_get_block_p, &
                                              dbcsr_reserve_blocks, dbcsr_put_block
   USE dbcsr_types,                     only: dbcsr_iterator,&
                                              dbcsr_type,&
                                              dbcsr_data_obj,&
                                              dbcsr_scalar_type,&
                                              dbcsr_mp_obj,&
                                              ${uselist(dkind1)}$
   use dbcsr_work_operations,           only: dbcsr_create, dbcsr_finalize, &
                                              dbcsr_work_create
   use dbcsr_dist_operations,           only: dbcsr_get_stored_coordinates
   USE dbcsr_methods,                   ONLY: dbcsr_get_data_type, dbcsr_release, &
                                              dbcsr_distribution,&
                                              dbcsr_get_num_blocks,&
                                              dbcsr_get_nze,&
                                              dbcsr_nfullcols_local,&
                                              dbcsr_nfullrows_local
   USE dbcsr_kinds,                     ONLY: default_string_length,&
                                              int_4,&
                                              int_8
   USE dbcsr_kinds,                     ONLY: ${uselist(kind1)}$
   USE dbcsr_mpiwrap,                   ONLY: mp_alltoall,&
                                              mp_environ,&
                                              mp_irecv,&
                                              mp_isend,&
                                              mp_waitall
   USE dbcsr_dist_methods,              ONLY: dbcsr_distribution_mp
   USE dbcsr_mp_methods,                ONLY: dbcsr_mp_group, dbcsr_mp_mynode, dbcsr_mp_numnodes
   USE dbcsr_data_methods,              only: dbcsr_data_set_pointer, dbcsr_data_init, &
                                              dbcsr_data_new, dbcsr_data_release, &
                                              dbcsr_scalar_one, dbcsr_data_clear_pointer
   USE dbcsr_index_operations,          ONLY: dbcsr_repoint_index
   USE dbcsr_mm_dist_operations,        ONLY: dbcsr_get_global_row_map

#include "base/dbcsr_base_uses.f90"
   
!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE
   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_mm_reshape'

   PUBLIC :: dbcsr_reshape

   TYPE block_buffer_type
      integer                                    :: datasize = 0
      INTEGER                                    :: nblock = 0
      INTEGER, DIMENSION(:, :), pointer          :: indx
      type(dbcsr_data_obj)                       :: data
      INTEGER                                    :: data_type = -1
      INTEGER                                    :: endpos = 0
   END TYPE

   INTERFACE block_buffer_add_block
#:for dparam, dtype, dsuffix in dtype_float_list
      MODULE PROCEDURE block_buffer_add_block_${dsuffix}$
#:endfor
   END INTERFACE

CONTAINS

! **************************************************************************************************
!> \brief Summation matrix_out = matrix_out + reshaped( matrix_in ), 
!>        where matrix_in is localy distributed (involves reshape)
!> \param matrix_in ...
!> \param matrix_out ...
! **************************************************************************************************
   SUBROUTINE dbcsr_reshape(matrix_in, matrix_out)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_in, matrix_out

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_reshape', &
         routineP = moduleN//':'//routineN

      TYPE(dbcsr_iterator)                               :: iter
      INTEGER                                            :: col, row, blk, row_size, col_size, &
                                                            handle, mynode, iproc, numnodes, &
                                                            data_type, blk_p, mp_comm, &
                                                            size_work, nblks_guess
      TYPE(dbcsr_mp_obj)                                 :: mp_obj
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: num_send, num_recv
      INTEGER, DIMENSION(2)                              :: total_num
      TYPE(block_buffer_type), ALLOCATABLE, DIMENSION(:) :: buffer_recv, buffer_send
      TYPE(dbcsr_data_obj)                               :: blk_data
      TYPE(dbcsr_type), ALLOCATABLE, DIMENSION(:)        :: partial_matrices

      INTEGER, PARAMETER                                 :: imeta = 1, idata = 2
      INTEGER(KIND=int_8)                                :: local_matrix_size
      
      CALL timeset(routineN, handle)
      !
      mp_obj = dbcsr_distribution_mp(dbcsr_distribution(matrix_out))
      mynode = dbcsr_mp_mynode(mp_obj)
      numnodes = dbcsr_mp_numnodes(mp_obj)
      mp_comm = dbcsr_mp_group(mp_obj)
      !
      data_type = dbcsr_get_data_type(matrix_out)
      !
      ALLOCATE (num_send(2, 0:numnodes - 1))
      num_send(:, :) = 0
      !
      ! Loop over all blocks to get the sizes
      CALL dbcsr_iterator_start(iter, matrix_in)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, row, col, blk, &
              row_size=row_size, col_size=col_size)
         CALL dbcsr_get_stored_coordinates(matrix_out, row, col, iproc)
         num_send(imeta, iproc) = num_send(imeta, iproc) + 1
         num_send(idata, iproc) = num_send(idata, iproc) + row_size * col_size
      ENDDO
      CALL dbcsr_iterator_stop(iter)
      !
      ALLOCATE (num_recv(2, 0:numnodes - 1))
      num_recv(:, :) = 0
      CALL mp_alltoall(num_send, num_recv, 2, mp_comm)
      !
      ! Create the buffers
      ALLOCATE (buffer_send(0:numnodes - 1))
      ALLOCATE (buffer_recv(0:numnodes - 1))
      DO iproc = 0, numnodes - 1
         CALL block_buffer_create(buffer_send(iproc), num_send(imeta, iproc), &
                                  num_send(idata, iproc), data_type)
         CALL block_buffer_create(buffer_recv(iproc), num_recv(imeta, iproc), &
                                  num_recv(idata, iproc), data_type)
      ENDDO
      !
      ! Fill the buffers
      CALL dbcsr_data_init(blk_data)
      CALL dbcsr_data_new(blk_data, data_type)
      CALL dbcsr_iterator_start(iter, matrix_in)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, row, col, blk, blk_p=blk_p, row_size=row_size, col_size=col_size)
         call set_wrap_blk_pointer(matrix_in%data_area, blk_p, row_size*col_size, blk_data)
         CALL dbcsr_get_stored_coordinates(matrix_out, row, col, iproc)
         CALL block_buffer_add_anyd_block(buffer_send(iproc), row, col, blk_data)
      ENDDO
      CALL dbcsr_iterator_stop(iter)
      !
      CALL communicate_buffer(mp_comm, buffer_recv, buffer_send)
      !
      ALLOCATE(partial_matrices(0: numnodes-1))
      DO iproc = 0, numnodes - 1
         CALL dbcsr_create(matrix = partial_matrices(iproc), template = matrix_out, &
              data_buffer = buffer_recv(iproc) % data, nze = num_recv(idata, iproc))
         CALL dbcsr_reserve_blocks(partial_matrices(iproc), buffer_recv(iproc) % indx(:,1), &
                                   buffer_recv(iproc) % indx(:,2), buffer_recv(iproc) % indx(:,3))
         call dbcsr_repoint_index(partial_matrices(iproc))
         partial_matrices(iproc) % valid=.true.
         total_num(imeta) = total_num(imeta) + partial_matrices(iproc) % nblks
         total_num(idata) = total_num(idata) + partial_matrices(iproc) % nze
      ENDDO
      !
      ! Accumulate
      !
      ! Pre-size work arrays of matrix_a to avoid continuous reallocation.
      local_matrix_size = INT(dbcsr_nfullrows_local(matrix_out), KIND=int_8)* &
           dbcsr_nfullcols_local(matrix_out)
      size_work = MAX(0, INT(MIN(local_matrix_size-INT(dbcsr_get_nze(matrix_out), KIND=int_8), &
           INT(total_num(idata), KIND=int_8)), KIND=int_4))
      nblks_guess = MIN(matrix_in % nblkrows_total * matrix_in % nblkcols_total, total_num(idata))
!$OMP PARALLEL DEFAULT (NONE) &
!$OMP          SHARED (matrix_out, size_work, nblks_guess) 
      CALL dbcsr_work_create(matrix_out, &
           nblks_guess = nblks_guess, &
           sizedata_guess = size_work, &
           work_mutable=.FALSE.,create_hashes_par = .true.)

!$OMP        END PARALLEL
      DO iproc = 0, numnodes - 1
         CALL dbcsr_accumulation_anytype(matrix_out, partial_matrices(iproc))         
         CALL dbcsr_release(partial_matrices(iproc))
         CALL block_buffer_destroy(buffer_send(iproc))
         CALL block_buffer_destroy(buffer_recv(iproc))
      ENDDO
      CALL dbcsr_finalize(matrix_out)      
      CALL dbcsr_data_clear_pointer(blk_data)
      CALL dbcsr_data_release(blk_data)

      CALL timestop(handle)
   END SUBROUTINE dbcsr_reshape

! **************************************************************************************************
!> \brief Create block buffer for MPI communication.
!> \param buffer block buffer
!> \param nblock number of blocks
!> \param ndata total number of block entries
!> \param data_type ...
!> \param ndim number of dimensions
! **************************************************************************************************
   SUBROUTINE block_buffer_create(buffer, nblock, ndata, data_type)
      TYPE(block_buffer_type), INTENT(OUT) :: buffer
      INTEGER, INTENT(IN)                  :: nblock, ndata, data_type

      buffer%nblock = nblock
      buffer%data_type = data_type
      buffer%endpos = 0

      call dbcsr_data_init( buffer % data )
      call dbcsr_data_new( buffer % data, data_type, ndata)
      ALLOCATE (buffer%indx(nblock, 3))
   END SUBROUTINE block_buffer_create

! **************************************************************************************************
!> \brief ...
!> \param buffer ...
! **************************************************************************************************
   SUBROUTINE block_buffer_destroy(buffer)
      TYPE(block_buffer_type), INTENT(INOUT) :: buffer

      call dbcsr_data_release(buffer % data)

      DEALLOCATE (buffer%indx)
      buffer%nblock = -1
      buffer%data_type = -1
      buffer%endpos = -1
   END SUBROUTINE block_buffer_destroy

! **************************************************************************************************
!> \brief insert a block into block buffer (at current iterator position)
!> \param buffer ...
!> \param row row index of block
!> \param col col index of block
!> \param block block
! **************************************************************************************************
   SUBROUTINE block_buffer_add_anyd_block(buffer, row, col, block)
      TYPE(block_buffer_type), INTENT(INOUT)      :: buffer
      INTEGER, INTENT(IN)                         :: row, col
      TYPE(dbcsr_data_obj), INTENT(IN)            :: block

      SELECT CASE (buffer%data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
      CASE (${dparam}$)
         CALL block_buffer_add_block_${dsuffix}$(buffer, SIZE(block % d % ${dsuffix}$), row, col, block % d % ${dsuffix}$)
#:endfor
      END SELECT
   END SUBROUTINE

   subroutine set_wrap_blk_pointer(data, lb_b, ndata, block_data)
      TYPE(dbcsr_data_obj), INTENT(inout) :: data
      integer, intent(in)              :: lb_b, ndata  
      TYPE(dbcsr_data_obj), INTENT(inout) :: block_data

      SELECT CASE (data % d % data_type)
#:for dparam, dtype, dsuffix in dtype_float_list
      CASE (${dparam}$)
         block_data % d % ${dsuffix}$ => data % d % ${dsuffix}$ (lb_b : ndata + lb_b - 1)
#:endfor
      END SELECT
   end subroutine set_wrap_blk_pointer

! **************************************************************************************************
!> \brief communicate buffer
!> \param mp_comm ...
!> \param buffer_recv ...
!> \param buffer_send ...
! **************************************************************************************************
   SUBROUTINE communicate_buffer(mp_comm, buffer_recv, buffer_send)
      INTEGER, INTENT(IN)                    :: mp_comm
      TYPE(block_buffer_type), DIMENSION(0:), INTENT(INOUT) :: buffer_recv, buffer_send

      INTEGER                                :: iproc, mynode, numnodes, rec_counter, &
                                                send_counter
      INTEGER, ALLOCATABLE, DIMENSION(:, :)  :: req_array
      INTEGER                                   :: handle
      CHARACTER(LEN=*), PARAMETER :: routineN = 'communicate_buffer', &
         routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)
      CALL mp_environ(numnodes, mynode, mp_comm)

      ALLOCATE (req_array(1:numnodes, 4))

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
                  CALL mp_irecv(buffer_recv(iproc) % data % d % ${dsuffix}$, iproc, mp_comm, req_array(rec_counter, 4), tag=7)
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
                  CALL mp_isend(buffer_send(iproc) % data % d % ${dsuffix}$, iproc, mp_comm, req_array(send_counter, 2), tag=7)
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
               buffer_recv(0) % data % d % ${dsuffix}$ (:) = buffer_send(0) % data % d % ${dsuffix}$ (:)
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
   SUBROUTINE block_buffer_add_block_${dsuffix}$(buffer, ndata, row, col, block)
      TYPE(block_buffer_type), INTENT(INOUT)               :: buffer
      INTEGER, INTENT(IN)                                  :: ndata
      ${dtype}$, DIMENSION(ndata), INTENT(IN)              :: block
      INTEGER, INTENT(IN)                                  :: row, col
      INTEGER                                              :: p, p_data
       
      p = buffer%endpos
      
      p_data = buffer%datasize
      
      buffer % data % d % ${dsuffix}$ (p_data+1:p_data+ndata) = block(:)
      buffer%indx(p+1, 1) = row
      buffer%indx(p+1, 2) = col
      buffer%indx(p+1, 3) = buffer%datasize + 1
      buffer%datasize = buffer%datasize + ndata
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
   SUBROUTINE block_buffer_get_next_block_${dsuffix}$(buffer, ndata, row, col, block, advance_iter)
      TYPE(block_buffer_type), INTENT(INOUT)                      :: buffer
      INTEGER, INTENT(OUT)                                        :: ndata
      ${dtype}$, DIMENSION(:), pointer, OPTIONAL, INTENT(INOUT)   :: block
      INTEGER, INTENT(OUT)                                        :: row, col
      INTEGER                                                     :: p, p_data
      LOGICAL, INTENT(IN), OPTIONAL                               :: advance_iter
      LOGICAL                                                     :: do_advance

      do_advance = .FALSE.
      IF (PRESENT(advance_iter)) THEN
         do_advance = advance_iter
      ELSE IF (PRESENT(block)) THEN
         do_advance = .TRUE.
      ENDIF

      p = buffer%endpos
      IF (p .EQ. 0) THEN
         p_data = 0
      ELSE
         p_data = buffer%indx(p, 3)
      ENDIF
      IF (p > 0) THEN
         ndata = buffer%indx(p+1, 3)-buffer%indx(p, 3)
      ELSE
         ndata = buffer%indx(p+1, 3)
      ENDIF
      row = buffer%indx(p+1, 1)
      col = buffer%indx(p+1, 2)
      IF (PRESENT(block)) THEN
         block => buffer % data % d % ${dsuffix}$ (p_data+1 : p_data+ndata)
      ENDIF

      IF(do_advance) buffer%endpos = buffer%endpos+1
   END SUBROUTINE
#:endfor

! **************************************************************************************************
!> \brief Accumulation A += B 
!> \param[in,out] matrix_a   DBCSR matrix
!> \param[in] matrix_b       DBCSR matrix
!> \param[in] alpha_scalar   (optional) ...
!> \param[in] beta_scalar    (optional) ...
!>
!> \param flop ...
! **************************************************************************************************
   SUBROUTINE dbcsr_accumulation_anytype(matrix_a, matrix_b, flop)
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_a
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_b
      INTEGER(KIND=int_8), INTENT(INOUT), OPTIONAL       :: flop

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_batched_add_anytype', &
         routineP = moduleN//':'//routineN

      INTEGER                                            :: handle, iw
      INTEGER(KIND=int_8)                                :: my_flop
      LOGICAL                                            :: do_scale
      TYPE(dbcsr_iterator)                               :: iter
      TYPE(dbcsr_scalar_type)                            :: my_beta_scalar
      type(array_i1d_obj)                                :: map_row_g2l
      !----------------------------------------------------------------------------

      CALL timeset(routineN, handle)
      my_beta_scalar = dbcsr_scalar_one(dbcsr_get_data_type(matrix_b))
      do_scale = .FALSE.
      IF (dbcsr_get_num_blocks(matrix_b) .GT. 0) THEN
         ! just to initialize if it was not initialized, map_row_g2l is not used
         call dbcsr_get_global_row_map(matrix_a % dist, map_row_g2l) 
!$OMP        PARALLEL DEFAULT (NONE) &
!$OMP                 PRIVATE (iter, iw) &
!$OMP                 SHARED (matrix_a, matrix_b) &
!$OMP                 SHARED (do_scale, my_beta_scalar) &
!$OMP                 REDUCTION (+ : my_flop)
         iw = 1
!$       iw = omp_get_thread_num () + 1
         CALL dbcsr_iterator_start(iter, matrix_b, &
                                   shared=.TRUE., read_only=.TRUE., contiguous_pointers=.FALSE.)
         SELECT CASE (dbcsr_get_data_type(matrix_b))
         CASE (dbcsr_type_real_4)
            CALL dbcsr_add_iter_s(matrix_a, matrix_b, iter, iw, do_scale, my_beta_scalar, my_flop, .true.)
         CASE (dbcsr_type_real_8)
            CALL dbcsr_add_iter_d(matrix_a, matrix_b, iter, iw, do_scale, my_beta_scalar, my_flop, .true.)
         CASE (dbcsr_type_complex_4)
            CALL dbcsr_add_iter_c(matrix_a, matrix_b, iter, iw, do_scale, my_beta_scalar, my_flop, .true.)
         CASE (dbcsr_type_complex_8)
            CALL dbcsr_add_iter_z(matrix_a, matrix_b, iter, iw, do_scale, my_beta_scalar, my_flop, .true.)
         CASE default
            DBCSR_ABORT("Invalid data type")
         END SELECT
         CALL dbcsr_iterator_stop(iter)
!$OMP        END PARALLEL
         IF (PRESENT(flop)) flop = flop+my_flop
      ENDIF
      CALL timestop(handle)
   END SUBROUTINE dbcsr_accumulation_anytype

END MODULE dbcsr_mm_reshape
