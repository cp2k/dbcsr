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

#:include "dbcsr_tensor.fypp"
#:set maxdim = maxrank
#:set ndims = range(2,maxdim+1)

   USE dbcsr_tensor_types, ONLY: &
      dbcsr_t_get_info, dbcsr_t_type, ndims_tensor, dbcsr_t_get_num_blocks, dbcsr_t_get_num_blocks_total, &
      blk_dims_tensor, dbcsr_t_get_stored_coordinates, dbcsr_t_get_nze, dbcsr_t_get_nze_total, &
      dbcsr_t_pgrid_type
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
      dbcsr_t_write_split_info

CONTAINS

   SUBROUTINE dbcsr_t_write_tensor_info(tensor, output_unit, full_info)
      !! Write tensor global info: block dimensions, full dimensions and process grid dimensions

      TYPE(dbcsr_t_type), INTENT(IN) :: tensor
      INTEGER, INTENT(IN)            :: output_unit
      LOGICAL, OPTIONAL, INTENT(IN)  :: full_info
         !! Whether to print distribution and block size vectors
      INTEGER, DIMENSION(ndims_tensor(tensor)) :: nblks_total, nfull_total, pdims, my_ploc, nblks_local, nfull_local
      INTEGER, DIMENSION(:), ALLOCATABLE :: ${varlist("blks_local")}$
      INTEGER, DIMENSION(:), ALLOCATABLE :: ${varlist("proc_dist")}$
      INTEGER, DIMENSION(:), ALLOCATABLE :: ${varlist("blk_size")}$
      INTEGER, DIMENSION(:), ALLOCATABLE :: ${varlist("blk_offset")}$
      CHARACTER(len=default_string_length)                   :: name
      INTEGER                            :: idim
      INTEGER                            :: iblk

      CALL dbcsr_t_get_info(tensor, nblks_total, nfull_total, nblks_local, nfull_local, pdims, my_ploc, &
                            ${varlist("blks_local")}$, ${varlist("proc_dist")}$, ${varlist("blk_size")}$, &
                            ${varlist("blk_offset")}$, &
                            name=name)

      IF (output_unit > 0) THEN
         WRITE (output_unit, "(T2,A)") &
            "GLOBAL INFO OF "//TRIM(name)
         WRITE (output_unit, "(T4,A,1X)", advance="no") "block dimensions:"
         DO idim = 1, ndims_tensor(tensor)
            WRITE (output_unit, "(I6)", advance="no") nblks_total(idim)
         ENDDO
         WRITE (output_unit, "(/T4,A,1X)", advance="no") "full dimensions:"
         DO idim = 1, ndims_tensor(tensor)
            WRITE (output_unit, "(I8)", advance="no") nfull_total(idim)
         ENDDO
         WRITE (output_unit, "(/T4,A,1X)", advance="no") "process grid dimensions:"
         DO idim = 1, ndims_tensor(tensor)
            WRITE (output_unit, "(I6)", advance="no") pdims(idim)
         ENDDO
         WRITE (output_unit, *)

         IF (PRESENT(full_info)) THEN
            IF (full_info) THEN
               WRITE (output_unit, '(T4,A)', advance='no') "Block sizes:"
#:for dim in range(1, maxdim+1)
               IF (ndims_tensor(tensor) >= ${dim}$) THEN
                  WRITE (output_unit, '(/T8,A,1X,I1,A,1X)', advance='no') 'Dim', ${dim}$, ':'
                  DO iblk = 1, SIZE(blk_size_${dim}$)
                     WRITE (output_unit, '(I2,1X)', advance='no') blk_size_${dim}$ (iblk)
                  ENDDO
               ENDIF
#:endfor
               WRITE (output_unit, '(/T4,A)', advance='no') "Block distribution:"
#:for dim in range(1, maxdim+1)
               IF (ndims_tensor(tensor) >= ${dim}$) THEN
                  WRITE (output_unit, '(/T8,A,1X,I1,A,1X)', advance='no') 'Dim', ${dim}$, ':'
                  DO iblk = 1, SIZE(proc_dist_${dim}$)
                     WRITE (output_unit, '(I3,1X)', advance='no') proc_dist_${dim}$ (iblk)
                  ENDDO
               ENDIF
#:endfor
            ENDIF
            WRITE (output_unit, *)
         ENDIF
      ENDIF

   END SUBROUTINE

   SUBROUTINE dbcsr_t_write_tensor_dist(tensor, output_unit)
      !! Write info on tensor distribution & load balance
      TYPE(dbcsr_t_type), INTENT(IN) :: tensor
      INTEGER, INTENT(IN)            :: output_unit
      INTEGER                        :: nproc, myproc, nblock_max, nelement_max
      INTEGER(KIND=int_8)            :: nblock_sum, nelement_sum, nblock_tot
      INTEGER                        :: nblock, nelement
      INTEGER                        :: mp_comm
      INTEGER, DIMENSION(2)          :: tmp
      INTEGER, DIMENSION(ndims_tensor(tensor)) :: bdims
      REAL(KIND=real_8)              :: occupation

      mp_comm = tensor%pgrid%mp_comm_2d
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

      IF (output_unit > 0) THEN
         WRITE (output_unit, "(T2,A)") &
            "DISTRIBUTION OF "//TRIM(tensor%name)
         WRITE (output_unit, "(T15,A,T68,I13)") "Number of non-zero blocks:", nblock_sum
         WRITE (output_unit, "(T15,A,T75,F6.2)") "Percentage of non-zero blocks:", occupation
         WRITE (output_unit, "(T15,A,T68,I13)") "Average number of blocks per CPU:", (nblock_sum + nproc - 1)/nproc
         WRITE (output_unit, "(T15,A,T68,I13)") "Maximum number of blocks per CPU:", nblock_max
         WRITE (output_unit, "(T15,A,T68,I13)") "Average number of matrix elements per CPU:", (nelement_sum + nproc - 1)/nproc
         WRITE (output_unit, "(T15,A,T68,I13)") "Maximum number of matrix elements per CPU:", nelement_max
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
#:for ndim in ndims
      REAL(KIND=real_8), ALLOCATABLE, &
         DIMENSION(${shape_colon(ndim)}$)                :: blk_values_${ndim}$
#:endfor
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
#:for ndim in ndims
         IF (ndims_tensor(tensor) == ${ndim}$) THEN
            CALL dbcsr_t_get_block(tensor, blk_index, blk_values_${ndim}$, found)
            DBCSR_ASSERT(found)
            CALL dbcsr_t_write_block(tensor%name, blk_size, blk_index, proc, io_unit_all, &
                                     blk_values_${ndim}$=blk_values_${ndim}$, write_int=write_int)
            DEALLOCATE (blk_values_${ndim}$)
         ENDIF
#:endfor
      ENDDO
      CALL dbcsr_t_iterator_stop(iterator)
   END SUBROUTINE

   SUBROUTINE dbcsr_t_write_block(name, blk_size, blk_index, proc, unit_nr, &
                                  ${varlist("blk_values",nmin=2)}$, write_int)
      !! Write a tensor block
      CHARACTER(LEN=*), INTENT(IN)                       :: name
         !! tensor name
      INTEGER, DIMENSION(:), INTENT(IN)                  :: blk_size
         !! block size
      INTEGER, DIMENSION(:), INTENT(IN)                  :: blk_index
         !! block index
#:for ndim in ndims
      REAL(KIND=real_8), &
         DIMENSION(${arrlist("blk_size", nmax=ndim)}$), &
         INTENT(IN), OPTIONAL                            :: blk_values_${ndim}$
         !! block values for 2 dimensions
#:endfor
      LOGICAL, INTENT(IN), OPTIONAL                      :: write_int
         !! write_int convert values to integers
      LOGICAL                                            :: write_int_prv
      INTEGER, INTENT(IN)                                :: unit_nr
         !! unit number
      INTEGER, INTENT(IN)                                :: proc
         !! which process am I
      INTEGER                                            :: ${varlist("i")}$
      INTEGER                                            :: ndim

      IF (PRESENT(write_int)) THEN
         write_int_prv = write_int
      ELSE
         write_int_prv = .FALSE.
      ENDIF

      ndim = SIZE(blk_size)

      IF (unit_nr > 0) THEN
#:for ndim in ndims
         IF (ndim == ${ndim}$) THEN
#:for idim in range(ndim,0,-1)
            DO i_${idim}$ = 1, blk_size(${idim}$)
#:endfor
               IF (write_int_prv) THEN
                  WRITE (unit_nr, '(T7,A,T16,A,${ndim}$I3,1X,A,1X,I3,A,1X,A,${ndim}$I3,1X,A,1X,I20)') &
                     TRIM(name), "(", blk_index, ") @", proc, ':', &
                     "(", ${varlist("i", nmax=ndim)}$, ")", &
                     INT(blk_values_${ndim}$ (${varlist("i", nmax=ndim)}$), KIND=int_8)
               ELSE
                  WRITE (unit_nr, '(T7,A,T16,A,${ndim}$I3,1X,A,1X,I3,A,1X,A,${ndim}$I3,1X,A,1X,F10.5)') &
                     TRIM(name), "(", blk_index, ") @", proc, ':', &
                     "(", ${varlist("i", nmax=ndim)}$, ")", &
                     blk_values_${ndim}$ (${varlist("i", nmax=ndim)}$)
               ENDIF
#:for idim in range(ndim,0,-1)
            ENDDO
#:endfor
         ENDIF
#:endfor
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
#:for ndim in ndims
         IF (ndims_tensor(tensor) == ${ndim}$) THEN
            WRITE (io_unit_all, '(T7,A,T16,A,${ndim}$I3,1X,A,1X,I3,A2,${ndim}$I3)') &
               TRIM(tensor%name), "blk index (", blk_index, ") @", proc, ":", blk_size
         ENDIF
#:endfor
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
END MODULE
