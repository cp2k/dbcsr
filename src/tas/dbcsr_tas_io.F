!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tas_io

   !! tall-and-skinny matrices: Input / Output
   USE dbcsr_tas_types, ONLY: &
      dbcsr_tas_type, dbcsr_tas_split_info
   USE dbcsr_tas_global, ONLY: &
      dbcsr_tas_rowcol_data, dbcsr_tas_distribution
   USE dbcsr_kinds, ONLY: &
      int_8, real_8, default_string_length
   USE dbcsr_tas_base, ONLY: &
      dbcsr_tas_get_info, dbcsr_tas_get_num_blocks, dbcsr_tas_get_num_blocks_total, dbcsr_tas_get_nze_total, &
      dbcsr_tas_get_nze, dbcsr_tas_nblkrows_total, dbcsr_tas_nblkcols_total, dbcsr_tas_info
   USE dbcsr_tas_split, ONLY: &
      dbcsr_tas_get_split_info, rowsplit, colsplit
   USE dbcsr_mpiwrap, ONLY: &
      mp_environ, mp_sum, mp_max, mp_comm_type
   USE dbcsr_dist_methods, ONLY: &
      dbcsr_distribution_row_dist, dbcsr_distribution_col_dist

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tas_io'

   PUBLIC :: &
      dbcsr_tas_write_dist, &
      dbcsr_tas_write_matrix_info, &
      dbcsr_tas_write_split_info, &
      prep_output_unit

CONTAINS

   SUBROUTINE dbcsr_tas_write_matrix_info(matrix, unit_nr, full_info)
      !! Write basic infos of tall-and-skinny matrix: block dimensions, full dimensions, process grid dimensions

      TYPE(dbcsr_tas_type), INTENT(IN) :: matrix
      INTEGER, INTENT(IN)            :: unit_nr
      LOGICAL, OPTIONAL, INTENT(IN)  :: full_info
         !! Whether to print distribution and block size vectors

      INTEGER(KIND=int_8)                      :: nblkrows_total, nblkcols_total, nfullrows_total, &
                                                  nfullcols_total
      INTEGER                                  :: nprow, npcol, unit_nr_prv
      CLASS(dbcsr_tas_distribution), ALLOCATABLE :: proc_row_dist, proc_col_dist
      CLASS(dbcsr_tas_rowcol_data), ALLOCATABLE  :: row_blk_size, col_blk_size
      INTEGER(KIND=int_8)                      :: iblk
      CHARACTER(default_string_length)         :: name

      unit_nr_prv = prep_output_unit(unit_nr)
      IF (unit_nr_prv == 0) RETURN

      CALL dbcsr_tas_get_info(matrix, nblkrows_total=nblkrows_total, nblkcols_total=nblkcols_total, &
                              nfullrows_total=nfullrows_total, nfullcols_total=nfullcols_total, &
                              nprow=nprow, npcol=npcol, proc_row_dist=proc_row_dist, proc_col_dist=proc_col_dist, &
                              row_blk_size=row_blk_size, col_blk_size=col_blk_size, name=name)

      IF (unit_nr_prv > 0) THEN
         WRITE (unit_nr_prv, "(T2,A)") &
            "GLOBAL INFO OF "//TRIM(name)
         WRITE (unit_nr_prv, "(T4,A,1X)", advance="no") "block dimensions:"
         WRITE (unit_nr_prv, "(I12,I12)", advance="no") nblkrows_total, nblkcols_total
         WRITE (unit_nr_prv, "(/T4,A,1X)", advance="no") "full dimensions:"
         WRITE (unit_nr_prv, "(I14,I14)", advance="no") nfullrows_total, nfullcols_total
         WRITE (unit_nr_prv, "(/T4,A,1X)", advance="no") "process grid dimensions:"
         WRITE (unit_nr_prv, "(I10,I10)", advance="no") nprow, npcol
         IF (PRESENT(full_info)) THEN
            IF (full_info) THEN
               WRITE (unit_nr_prv, '(/T4,A)', advance='no') "Block sizes:"
               WRITE (unit_nr_prv, '(/T8,A)', advance='no') 'Row:'
               DO iblk = 1, row_blk_size%nmrowcol
                  WRITE (unit_nr_prv, '(I4,1X)', advance='no') row_blk_size%data(iblk)
               END DO
               WRITE (unit_nr_prv, '(/T8,A)', advance='no') 'Column:'
               DO iblk = 1, col_blk_size%nmrowcol
                  WRITE (unit_nr_prv, '(I4,1X)', advance='no') col_blk_size%data(iblk)
               END DO
               WRITE (unit_nr_prv, '(/T4,A)', advance='no') "Block distribution:"
               WRITE (unit_nr_prv, '(/T8,A)', advance='no') 'Row:'
               DO iblk = 1, proc_row_dist%nmrowcol
                  WRITE (unit_nr_prv, '(I4,1X)', advance='no') proc_row_dist%dist(iblk)
               END DO
               WRITE (unit_nr_prv, '(/T8,A)', advance='no') 'Column:'
               DO iblk = 1, proc_col_dist%nmrowcol
                  WRITE (unit_nr_prv, '(I4,1X)', advance='no') proc_col_dist%dist(iblk)
               END DO

            END IF
         END IF
         WRITE (unit_nr_prv, *)
      END IF

   END SUBROUTINE

   SUBROUTINE dbcsr_tas_write_dist(matrix, unit_nr, full_info)
      !! Write info on tall-and-skinny matrix distribution & load balance

      TYPE(dbcsr_tas_type), INTENT(IN) :: matrix
      INTEGER, INTENT(IN)              :: unit_nr
      LOGICAL, INTENT(IN), OPTIONAL    :: full_info
         !! Whether to print subgroup DBCSR distribution

      CHARACTER(default_string_length) :: name
      INTEGER                          :: ngroup, igroup, nproc, iproc, &
                                          nblock_p_max, nelement_p_max, &
                                          nblock, nelement
      INTEGER(KIND=int_8), DIMENSION(2) :: tmp_i8
      INTEGER, DIMENSION(2)            :: tmp
      INTEGER(KIND=int_8)              :: nblock_tot, nblock_p_sum, nelement_p_sum, nelement_s_max, &
                                          nblock_s, nelement_s, nblock_s_max
      REAL(KIND=real_8)                :: occupation
      INTEGER, DIMENSION(:), POINTER   :: rowdist, coldist
      INTEGER                          :: split_rowcol, icol, irow, unit_nr_prv
      TYPE(mp_comm_type)               :: mp_comm, mp_comm_group

      unit_nr_prv = prep_output_unit(unit_nr)
      IF (unit_nr_prv == 0) RETURN

      CALL dbcsr_tas_get_split_info(matrix%dist%info, mp_comm, ngroup, igroup, mp_comm_group, split_rowcol)
      CALL dbcsr_tas_get_info(matrix, name=name)
      CALL mp_environ(nproc, iproc, mp_comm)

      nblock = dbcsr_tas_get_num_blocks(matrix)
      nelement = dbcsr_tas_get_nze(matrix)

      nblock_p_sum = dbcsr_tas_get_num_blocks_total(matrix)
      nelement_p_sum = dbcsr_tas_get_nze_total(matrix)

      tmp = (/nblock, nelement/)
      CALL mp_max(tmp, mp_comm)
      nblock_p_max = tmp(1); nelement_p_max = tmp(2)

      nblock_s = nblock
      nelement_s = nelement

      CALL mp_sum(nblock_s, mp_comm_group)
      CALL mp_sum(nelement_s, mp_comm_group)

      tmp_i8 = (/nblock_s, nelement_s/)
      CALL mp_max(tmp_i8, mp_comm)
      nblock_s_max = tmp_i8(1); nelement_s_max = tmp_i8(2)

      nblock_tot = dbcsr_tas_nblkrows_total(matrix)*dbcsr_tas_nblkcols_total(matrix)
      occupation = -1.0_real_8
      IF (nblock_tot .NE. 0) occupation = 100.0_real_8*REAL(nblock_p_sum, real_8)/REAL(nblock_tot, real_8)

      rowdist => dbcsr_distribution_row_dist(matrix%matrix%dist)
      coldist => dbcsr_distribution_col_dist(matrix%matrix%dist)

      IF (unit_nr_prv > 0) THEN
         WRITE (unit_nr_prv, "(T2,A)") &
            "DISTRIBUTION OF "//TRIM(name)
         WRITE (unit_nr_prv, "(T15,A,T68,I13)") "Number of non-zero blocks:", nblock_p_sum
         WRITE (unit_nr_prv, "(T15,A,T75,F6.2)") "Percentage of non-zero blocks:", occupation
         WRITE (unit_nr_prv, "(T15,A,T68,I13)") "Average number of blocks per group:", (nblock_p_sum + ngroup - 1)/ngroup
         WRITE (unit_nr_prv, "(T15,A,T68,I13)") "Maximum number of blocks per group:", nblock_s_max
         WRITE (unit_nr_prv, "(T15,A,T68,I13)") "Average number of matrix elements per group:", (nelement_p_sum + ngroup - 1)/ngroup
         WRITE (unit_nr_prv, "(T15,A,T68,I13)") "Maximum number of matrix elements per group:", nelement_s_max
         WRITE (unit_nr_prv, "(T15,A,T68,I13)") "Average number of blocks per CPU:", (nblock_p_sum + nproc - 1)/nproc
         WRITE (unit_nr_prv, "(T15,A,T68,I13)") "Maximum number of blocks per CPU:", nblock_p_max
         WRITE (unit_nr_prv, "(T15,A,T68,I13)") "Average number of matrix elements per CPU:", (nelement_p_sum + nproc - 1)/nproc
         WRITE (unit_nr_prv, "(T15,A,T68,I13)") "Maximum number of matrix elements per CPU:", nelement_p_max
         IF (PRESENT(full_info)) THEN
            IF (full_info) THEN
               WRITE (unit_nr_prv, "(T15,A)") "Row distribution on subgroup:"
               WRITE (unit_nr_prv, '(T15)', advance='no')
               DO irow = 1, SIZE(rowdist)
                  WRITE (unit_nr_prv, '(I3, 1X)', advance='no') rowdist(irow)
               END DO
               WRITE (unit_nr_prv, "(/T15,A)") "Column distribution on subgroup:"
               WRITE (unit_nr_prv, '(T15)', advance='no')
               DO icol = 1, SIZE(coldist)
                  WRITE (unit_nr_prv, '(I3, 1X)', advance='no') coldist(icol)
               END DO
               WRITE (unit_nr_prv, *)
            END IF
         END IF
      END IF
   END SUBROUTINE

   SUBROUTINE dbcsr_tas_write_split_info(info, unit_nr, name)
      !! Print info on how matrix is split
      TYPE(dbcsr_tas_split_info), INTENT(IN)             :: info
      INTEGER, INTENT(IN) :: unit_nr
      CHARACTER(len=*), INTENT(IN), OPTIONAL             :: name
      INTEGER                                            :: groupsize, igroup, &
                                                            mynode, nsplit, &
                                                            numnodes, split_rowcol, unit_nr_prv
      INTEGER, DIMENSION(2)                              :: coord, dims, groupcoord, groupdims, &
                                                            pgrid_offset
      CHARACTER(len=:), ALLOCATABLE                      :: name_prv
      TYPE(mp_comm_type)                                 :: mp_comm, mp_comm_group

      unit_nr_prv = prep_output_unit(unit_nr)
      IF (unit_nr_prv == 0) RETURN

      IF (PRESENT(name)) THEN
         ALLOCATE (name_prv, SOURCE=TRIM(name))
      ELSE
         ALLOCATE (name_prv, SOURCE="")
      END IF

      CALL dbcsr_tas_get_split_info(info, mp_comm, nsplit, igroup, mp_comm_group, split_rowcol, pgrid_offset)

      CALL mp_environ(numnodes, mynode, mp_comm)
      CALL mp_environ(numnodes, dims, coord, mp_comm)
      CALL mp_environ(groupsize, groupdims, groupcoord, mp_comm_group)

      IF (unit_nr_prv > 0) THEN
         SELECT CASE (split_rowcol)
         CASE (rowsplit)
            WRITE (unit_nr_prv, "(T4,A,I4,1X,A,I4)") name_prv//"splitting rows by factor", nsplit
         CASE (colsplit)
            WRITE (unit_nr_prv, "(T4,A,I4,1X,A,I4)") name_prv//"splitting columns by factor", nsplit
         END SELECT
         WRITE (unit_nr_prv, "(T4,A,I4,A1,I4)") name_prv//"global grid sizes:", dims(1), "x", dims(2)
      END IF

      IF (unit_nr_prv > 0) THEN
         WRITE (unit_nr_prv, "(T4,A,I4,A1,I4)") &
            name_prv//"grid sizes on subgroups:", &
            groupdims(1), "x", groupdims(2)
      END IF

   END SUBROUTINE

   FUNCTION prep_output_unit(unit_nr) RESULT(unit_nr_out)
      INTEGER, INTENT(IN), OPTIONAL :: unit_nr
      INTEGER                       :: unit_nr_out

      IF (PRESENT(unit_nr)) THEN
         unit_nr_out = unit_nr
      ELSE
         unit_nr_out = 0
      END IF

   END FUNCTION

END MODULE

