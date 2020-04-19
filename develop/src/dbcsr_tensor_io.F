# 1 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F" 1
!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tensor_io
   !! DBCSR tensor Input / Output

# 1 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp" 1
# 9 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp"

# 241 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp"
# 14 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F" 2
# 15 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
# 16 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"

   USE dbcsr_tensor_types, ONLY: &
      dbcsr_t_get_info, dbcsr_t_type, ndims_tensor, dbcsr_t_get_num_blocks, dbcsr_t_get_num_blocks_total, &
      blk_dims_tensor, dbcsr_t_get_stored_coordinates, dbcsr_t_get_nze, dbcsr_t_get_nze_total, &
      dbcsr_t_pgrid_type, dbcsr_t_nblks_total
   USE dbcsr_kinds, ONLY: default_string_length, int_8, real_8
   USE dbcsr_mpiwrap, ONLY: mp_environ, mp_sum, mp_max
   USE dbcsr_tensor_block, ONLY: &
      dbcsr_t_iterator_type, dbcsr_t_iterator_next_block, dbcsr_t_iterator_start, &
      dbcsr_t_iterator_blocks_left, dbcsr_t_iterator_stop, dbcsr_t_get_block
   USE dbcsr_tas_io, ONLY: dbcsr_tas_write_split_info

#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tensor_types'

   PUBLIC :: &
      dbcsr_t_write_tensor_info, &
      dbcsr_t_write_tensor_dist, &
      dbcsr_t_write_blocks, &
      dbcsr_t_write_block, &
      dbcsr_t_write_block_indices, &
      dbcsr_t_write_split_info, &
      prep_output_unit

CONTAINS

   SUBROUTINE dbcsr_t_write_tensor_info(tensor, unit_nr, full_info)
      !! Write tensor global info: block dimensions, full dimensions and process grid dimensions

      TYPE(dbcsr_t_type), INTENT(IN) :: tensor
      INTEGER, INTENT(IN)            :: unit_nr
      LOGICAL, OPTIONAL, INTENT(IN)  :: full_info
         !! Whether to print distribution and block size vectors
      INTEGER, DIMENSION(ndims_tensor(tensor)) :: nblks_total, nfull_total, pdims, my_ploc, nblks_local, nfull_local

# 55 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
      INTEGER, DIMENSION(dbcsr_t_nblks_total(tensor, 1)) :: proc_dist_1
      INTEGER, DIMENSION(dbcsr_t_nblks_total(tensor, 1)) :: blk_size_1
      INTEGER, DIMENSION(dbcsr_t_nblks_total(tensor, 1)) :: blks_local_1
# 55 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
      INTEGER, DIMENSION(dbcsr_t_nblks_total(tensor, 2)) :: proc_dist_2
      INTEGER, DIMENSION(dbcsr_t_nblks_total(tensor, 2)) :: blk_size_2
      INTEGER, DIMENSION(dbcsr_t_nblks_total(tensor, 2)) :: blks_local_2
# 55 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
      INTEGER, DIMENSION(dbcsr_t_nblks_total(tensor, 3)) :: proc_dist_3
      INTEGER, DIMENSION(dbcsr_t_nblks_total(tensor, 3)) :: blk_size_3
      INTEGER, DIMENSION(dbcsr_t_nblks_total(tensor, 3)) :: blks_local_3
# 55 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
      INTEGER, DIMENSION(dbcsr_t_nblks_total(tensor, 4)) :: proc_dist_4
      INTEGER, DIMENSION(dbcsr_t_nblks_total(tensor, 4)) :: blk_size_4
      INTEGER, DIMENSION(dbcsr_t_nblks_total(tensor, 4)) :: blks_local_4
# 59 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
      CHARACTER(len=default_string_length)                     :: name
      INTEGER                                                  :: idim
      INTEGER                                                  :: iblk
      INTEGER                                                  :: unit_nr_prv

      unit_nr_prv = prep_output_unit(unit_nr)
      IF (unit_nr_prv == 0) RETURN

      CALL dbcsr_t_get_info(tensor, nblks_total, nfull_total, nblks_local, nfull_local, pdims, my_ploc, &
                            blks_local_1, blks_local_2, blks_local_3, blks_local_4, proc_dist_1, proc_dist_2, proc_dist_3,&
# 68 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
                                & proc_dist_4, blk_size_1, blk_size_2, blk_size_3, blk_size_4, &
                            name=name)

      IF (unit_nr_prv > 0) THEN
         WRITE (unit_nr_prv, "(T2,A)") &
            "GLOBAL INFO OF "//TRIM(name)
         WRITE (unit_nr_prv, "(T4,A,1X)", advance="no") "block dimensions:"
         DO idim = 1, ndims_tensor(tensor)
            WRITE (unit_nr_prv, "(I6)", advance="no") nblks_total(idim)
         ENDDO
         WRITE (unit_nr_prv, "(/T4,A,1X)", advance="no") "full dimensions:"
         DO idim = 1, ndims_tensor(tensor)
            WRITE (unit_nr_prv, "(I8)", advance="no") nfull_total(idim)
         ENDDO
         WRITE (unit_nr_prv, "(/T4,A,1X)", advance="no") "process grid dimensions:"
         DO idim = 1, ndims_tensor(tensor)
            WRITE (unit_nr_prv, "(I6)", advance="no") pdims(idim)
         ENDDO
         WRITE (unit_nr_prv, *)

         IF (PRESENT(full_info)) THEN
            IF (full_info) THEN
               WRITE (unit_nr_prv, '(T4,A)', advance='no') "Block sizes:"
# 92 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
               IF (ndims_tensor(tensor) >= 1) THEN
                  WRITE (unit_nr_prv, '(/T8,A,1X,I1,A,1X)', advance='no') 'Dim', 1, ':'
                  DO iblk = 1, SIZE(blk_size_1)
                     WRITE (unit_nr_prv, '(I2,1X)', advance='no') blk_size_1 (iblk)
                  ENDDO
               ENDIF
# 92 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
               IF (ndims_tensor(tensor) >= 2) THEN
                  WRITE (unit_nr_prv, '(/T8,A,1X,I1,A,1X)', advance='no') 'Dim', 2, ':'
                  DO iblk = 1, SIZE(blk_size_2)
                     WRITE (unit_nr_prv, '(I2,1X)', advance='no') blk_size_2 (iblk)
                  ENDDO
               ENDIF
# 92 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
               IF (ndims_tensor(tensor) >= 3) THEN
                  WRITE (unit_nr_prv, '(/T8,A,1X,I1,A,1X)', advance='no') 'Dim', 3, ':'
                  DO iblk = 1, SIZE(blk_size_3)
                     WRITE (unit_nr_prv, '(I2,1X)', advance='no') blk_size_3 (iblk)
                  ENDDO
               ENDIF
# 92 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
               IF (ndims_tensor(tensor) >= 4) THEN
                  WRITE (unit_nr_prv, '(/T8,A,1X,I1,A,1X)', advance='no') 'Dim', 4, ':'
                  DO iblk = 1, SIZE(blk_size_4)
                     WRITE (unit_nr_prv, '(I2,1X)', advance='no') blk_size_4 (iblk)
                  ENDDO
               ENDIF
# 99 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
               WRITE (unit_nr_prv, '(/T4,A)', advance='no') "Block distribution:"
# 101 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
               IF (ndims_tensor(tensor) >= 1) THEN
                  WRITE (unit_nr_prv, '(/T8,A,1X,I1,A,1X)', advance='no') 'Dim', 1, ':'
                  DO iblk = 1, SIZE(proc_dist_1)
                     WRITE (unit_nr_prv, '(I3,1X)', advance='no') proc_dist_1 (iblk)
                  ENDDO
               ENDIF
# 101 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
               IF (ndims_tensor(tensor) >= 2) THEN
                  WRITE (unit_nr_prv, '(/T8,A,1X,I1,A,1X)', advance='no') 'Dim', 2, ':'
                  DO iblk = 1, SIZE(proc_dist_2)
                     WRITE (unit_nr_prv, '(I3,1X)', advance='no') proc_dist_2 (iblk)
                  ENDDO
               ENDIF
# 101 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
               IF (ndims_tensor(tensor) >= 3) THEN
                  WRITE (unit_nr_prv, '(/T8,A,1X,I1,A,1X)', advance='no') 'Dim', 3, ':'
                  DO iblk = 1, SIZE(proc_dist_3)
                     WRITE (unit_nr_prv, '(I3,1X)', advance='no') proc_dist_3 (iblk)
                  ENDDO
               ENDIF
# 101 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
               IF (ndims_tensor(tensor) >= 4) THEN
                  WRITE (unit_nr_prv, '(/T8,A,1X,I1,A,1X)', advance='no') 'Dim', 4, ':'
                  DO iblk = 1, SIZE(proc_dist_4)
                     WRITE (unit_nr_prv, '(I3,1X)', advance='no') proc_dist_4 (iblk)
                  ENDDO
               ENDIF
# 108 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            ENDIF
            WRITE (unit_nr_prv, *)
         ENDIF
      ENDIF

   END SUBROUTINE

   SUBROUTINE dbcsr_t_write_tensor_dist(tensor, unit_nr)
      !! Write info on tensor distribution & load balance
      TYPE(dbcsr_t_type), INTENT(IN) :: tensor
      INTEGER, INTENT(IN)            :: unit_nr
      INTEGER                        :: nproc, myproc, nblock_max, nelement_max
      INTEGER(KIND=int_8)            :: nblock_sum, nelement_sum, nblock_tot
      INTEGER                        :: nblock, nelement, unit_nr_prv
      INTEGER                        :: mp_comm
      INTEGER, DIMENSION(2)          :: tmp
      INTEGER, DIMENSION(ndims_tensor(tensor)) :: bdims
      REAL(KIND=real_8)              :: occupation

      mp_comm = tensor%pgrid%mp_comm_2d
      unit_nr_prv = prep_output_unit(unit_nr)
      IF (unit_nr_prv == 0) RETURN

      CALL mp_environ(nproc, myproc, mp_comm)

      nblock = dbcsr_t_get_num_blocks(tensor)
      nelement = dbcsr_t_get_nze(tensor)

      nblock_sum = dbcsr_t_get_num_blocks_total(tensor)
      nelement_sum = dbcsr_t_get_nze_total(tensor)

      tmp = (/nblock, nelement/)
      CALL mp_max(tmp, mp_comm)
      nblock_max = tmp(1); nelement_max = tmp(2)

      CALL blk_dims_tensor(tensor, bdims)
      nblock_tot = PRODUCT(INT(bdims, KIND=int_8))

      occupation = -1.0_real_8
      IF (nblock_tot .NE. 0) occupation = 100.0_real_8*REAL(nblock_sum, real_8)/REAL(nblock_tot, real_8)

      IF (unit_nr_prv > 0) THEN
         WRITE (unit_nr_prv, "(T2,A)") &
            "DISTRIBUTION OF "//TRIM(tensor%name)
         WRITE (unit_nr_prv, "(T15,A,T68,I13)") "Number of non-zero blocks:", nblock_sum
         WRITE (unit_nr_prv, "(T15,A,T75,F6.2)") "Percentage of non-zero blocks:", occupation
         WRITE (unit_nr_prv, "(T15,A,T68,I13)") "Average number of blocks per CPU:", (nblock_sum + nproc - 1)/nproc
         WRITE (unit_nr_prv, "(T15,A,T68,I13)") "Maximum number of blocks per CPU:", nblock_max
         WRITE (unit_nr_prv, "(T15,A,T68,I13)") "Average number of matrix elements per CPU:", (nelement_sum + nproc - 1)/nproc
         WRITE (unit_nr_prv, "(T15,A,T68,I13)") "Maximum number of matrix elements per CPU:", nelement_max
      ENDIF

   END SUBROUTINE

   SUBROUTINE dbcsr_t_write_blocks(tensor, io_unit_master, io_unit_all, write_int)
      !! Write all tensor blocks

      TYPE(dbcsr_t_type), INTENT(INOUT)                  :: tensor
      INTEGER, INTENT(IN)                                :: io_unit_master, io_unit_all
         !! for global output
         !! for local output
      LOGICAL, INTENT(IN), OPTIONAL                      :: write_int
         !! convert to integers (useful for testing with integer tensors)
      INTEGER                                            :: blk
      INTEGER, DIMENSION(ndims_tensor(tensor))          :: blk_index, blk_size
# 174 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
      REAL(KIND=real_8), ALLOCATABLE, &
         DIMENSION(:,:)                :: blk_values_2
# 174 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
      REAL(KIND=real_8), ALLOCATABLE, &
         DIMENSION(:,:,:)                :: blk_values_3
# 174 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
      REAL(KIND=real_8), ALLOCATABLE, &
         DIMENSION(:,:,:,:)                :: blk_values_4
# 177 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
      TYPE(dbcsr_t_iterator_type)                        :: iterator
      INTEGER                                            :: proc, mynode, numnodes
      LOGICAL                                            :: found

      IF (io_unit_master > 0) THEN
         WRITE (io_unit_master, '(T7,A)') "(block index) @ process: (array index) value"
      ENDIF
      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, blk_index, blk, blk_size=blk_size)
         CALL dbcsr_t_get_stored_coordinates(tensor, blk_index, proc)
         CALL mp_environ(numnodes, mynode, tensor%pgrid%mp_comm_2d)
         DBCSR_ASSERT(proc .EQ. mynode)
# 191 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
         IF (ndims_tensor(tensor) == 2) THEN
            CALL dbcsr_t_get_block(tensor, blk_index, blk_values_2, found)
            DBCSR_ASSERT(found)
            CALL dbcsr_t_write_block(tensor%name, blk_size, blk_index, proc, io_unit_all, &
                                     blk_values_2=blk_values_2, write_int=write_int)
            DEALLOCATE (blk_values_2)
         ENDIF
# 191 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
         IF (ndims_tensor(tensor) == 3) THEN
            CALL dbcsr_t_get_block(tensor, blk_index, blk_values_3, found)
            DBCSR_ASSERT(found)
            CALL dbcsr_t_write_block(tensor%name, blk_size, blk_index, proc, io_unit_all, &
                                     blk_values_3=blk_values_3, write_int=write_int)
            DEALLOCATE (blk_values_3)
         ENDIF
# 191 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
         IF (ndims_tensor(tensor) == 4) THEN
            CALL dbcsr_t_get_block(tensor, blk_index, blk_values_4, found)
            DBCSR_ASSERT(found)
            CALL dbcsr_t_write_block(tensor%name, blk_size, blk_index, proc, io_unit_all, &
                                     blk_values_4=blk_values_4, write_int=write_int)
            DEALLOCATE (blk_values_4)
         ENDIF
# 199 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)
   END SUBROUTINE

   SUBROUTINE dbcsr_t_write_block(name, blk_size, blk_index, proc, unit_nr, &
                                  blk_values_2, blk_values_3, blk_values_4, write_int)
      !! Write a tensor block
      CHARACTER(LEN=*), INTENT(IN)                       :: name
         !! tensor name
      INTEGER, DIMENSION(:), INTENT(IN)                  :: blk_size
         !! block size
      INTEGER, DIMENSION(:), INTENT(IN)                  :: blk_index
         !! block index
# 213 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
      REAL(KIND=real_8), &
         DIMENSION(blk_size(1), blk_size(2)), &
         INTENT(IN), OPTIONAL                            :: blk_values_2
         !! block values for 2 dimensions
# 213 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
      REAL(KIND=real_8), &
         DIMENSION(blk_size(1), blk_size(2), blk_size(3)), &
         INTENT(IN), OPTIONAL                            :: blk_values_3
         !! block values for 2 dimensions
# 213 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
      REAL(KIND=real_8), &
         DIMENSION(blk_size(1), blk_size(2), blk_size(3), blk_size(4)), &
         INTENT(IN), OPTIONAL                            :: blk_values_4
         !! block values for 2 dimensions
# 218 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
      LOGICAL, INTENT(IN), OPTIONAL                      :: write_int
         !! write_int convert values to integers
      LOGICAL                                            :: write_int_prv
      INTEGER, INTENT(IN)                                :: unit_nr
         !! unit number
      INTEGER, INTENT(IN)                                :: proc
         !! which process am I
      INTEGER                                            :: i_1, i_2, i_3, i_4
      INTEGER                                            :: ndim

      IF (PRESENT(write_int)) THEN
         write_int_prv = write_int
      ELSE
         write_int_prv = .FALSE.
      ENDIF

      ndim = SIZE(blk_size)

      IF (unit_nr > 0) THEN
# 238 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
         IF (ndim == 2) THEN
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            DO i_2 = 1, blk_size(2)
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            DO i_1 = 1, blk_size(1)
# 242 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
               IF (write_int_prv) THEN
                  WRITE (unit_nr, '(T7,A,T16,A,2I3,1X,A,1X,I3,A,1X,A,2I3,1X,A,1X,I20)') &
                     TRIM(name), "(", blk_index, ") @", proc, ':', &
                     "(", i_1, i_2, ")", &
                     INT(blk_values_2 (i_1, i_2), KIND=int_8)
               ELSE
                  WRITE (unit_nr, '(T7,A,T16,A,2I3,1X,A,1X,I3,A,1X,A,2I3,1X,A,1X,F10.5)') &
                     TRIM(name), "(", blk_index, ") @", proc, ':', &
                     "(", i_1, i_2, ")", &
                     blk_values_2 (i_1, i_2)
               ENDIF
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            ENDDO
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            ENDDO
# 256 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
         ENDIF
# 238 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
         IF (ndim == 3) THEN
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            DO i_3 = 1, blk_size(3)
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            DO i_2 = 1, blk_size(2)
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            DO i_1 = 1, blk_size(1)
# 242 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
               IF (write_int_prv) THEN
                  WRITE (unit_nr, '(T7,A,T16,A,3I3,1X,A,1X,I3,A,1X,A,3I3,1X,A,1X,I20)') &
                     TRIM(name), "(", blk_index, ") @", proc, ':', &
                     "(", i_1, i_2, i_3, ")", &
                     INT(blk_values_3 (i_1, i_2, i_3), KIND=int_8)
               ELSE
                  WRITE (unit_nr, '(T7,A,T16,A,3I3,1X,A,1X,I3,A,1X,A,3I3,1X,A,1X,F10.5)') &
                     TRIM(name), "(", blk_index, ") @", proc, ':', &
                     "(", i_1, i_2, i_3, ")", &
                     blk_values_3 (i_1, i_2, i_3)
               ENDIF
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            ENDDO
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            ENDDO
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            ENDDO
# 256 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
         ENDIF
# 238 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
         IF (ndim == 4) THEN
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            DO i_4 = 1, blk_size(4)
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            DO i_3 = 1, blk_size(3)
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            DO i_2 = 1, blk_size(2)
# 240 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            DO i_1 = 1, blk_size(1)
# 242 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
               IF (write_int_prv) THEN
                  WRITE (unit_nr, '(T7,A,T16,A,4I3,1X,A,1X,I3,A,1X,A,4I3,1X,A,1X,I20)') &
                     TRIM(name), "(", blk_index, ") @", proc, ':', &
                     "(", i_1, i_2, i_3, i_4, ")", &
                     INT(blk_values_4 (i_1, i_2, i_3, i_4), KIND=int_8)
               ELSE
                  WRITE (unit_nr, '(T7,A,T16,A,4I3,1X,A,1X,I3,A,1X,A,4I3,1X,A,1X,F10.5)') &
                     TRIM(name), "(", blk_index, ") @", proc, ':', &
                     "(", i_1, i_2, i_3, i_4, ")", &
                     blk_values_4 (i_1, i_2, i_3, i_4)
               ENDIF
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            ENDDO
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            ENDDO
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            ENDDO
# 254 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
            ENDDO
# 256 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
         ENDIF
# 258 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
      ENDIF
   END SUBROUTINE

   SUBROUTINE dbcsr_t_write_block_indices(tensor, io_unit_master, io_unit_all)
      TYPE(dbcsr_t_type), INTENT(INOUT)                  :: tensor
      INTEGER, INTENT(IN)                                :: io_unit_master, io_unit_all
      TYPE(dbcsr_t_iterator_type)                        :: iterator
      INTEGER, DIMENSION(ndims_tensor(tensor))          :: blk_index, blk_size
      INTEGER                                            :: blk, mynode, numnodes, proc

      IF (io_unit_master > 0) THEN
         WRITE (io_unit_master, '(T7,A)') "(block index) @ process: size"
      ENDIF

      CALL dbcsr_t_iterator_start(iterator, tensor)
      DO WHILE (dbcsr_t_iterator_blocks_left(iterator))
         CALL dbcsr_t_iterator_next_block(iterator, blk_index, blk, blk_size=blk_size)
         CALL dbcsr_t_get_stored_coordinates(tensor, blk_index, proc)
         CALL mp_environ(numnodes, mynode, tensor%pgrid%mp_comm_2d)
         DBCSR_ASSERT(proc .EQ. mynode)
# 279 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
         IF (ndims_tensor(tensor) == 2) THEN
            WRITE (io_unit_all, '(T7,A,T16,A,2I3,1X,A,1X,I3,A2,2I3)') &
               TRIM(tensor%name), "blk index (", blk_index, ") @", proc, ":", blk_size
         ENDIF
# 279 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
         IF (ndims_tensor(tensor) == 3) THEN
            WRITE (io_unit_all, '(T7,A,T16,A,3I3,1X,A,1X,I3,A2,3I3)') &
               TRIM(tensor%name), "blk index (", blk_index, ") @", proc, ":", blk_size
         ENDIF
# 279 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
         IF (ndims_tensor(tensor) == 4) THEN
            WRITE (io_unit_all, '(T7,A,T16,A,4I3,1X,A,1X,I3,A2,4I3)') &
               TRIM(tensor%name), "blk index (", blk_index, ") @", proc, ":", blk_size
         ENDIF
# 284 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor_io.F"
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)
   END SUBROUTINE

   SUBROUTINE dbcsr_t_write_split_info(pgrid, unit_nr)
      TYPE(dbcsr_t_pgrid_type), INTENT(IN) :: pgrid
      INTEGER, INTENT(IN) :: unit_nr

      IF (ALLOCATED(pgrid%tas_split_info)) THEN
         CALL dbcsr_tas_write_split_info(pgrid%tas_split_info, unit_nr)
      ENDIF
   END SUBROUTINE

   FUNCTION prep_output_unit(unit_nr) RESULT(unit_nr_out)
      INTEGER, INTENT(IN), OPTIONAL :: unit_nr
      INTEGER                       :: unit_nr_out

      IF (PRESENT(unit_nr)) THEN
         unit_nr_out = unit_nr
      ELSE
         unit_nr_out = 0
      ENDIF

   END FUNCTION
END MODULE
