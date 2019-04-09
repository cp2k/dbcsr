!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Rectangular matrices: Input / Output
!>
!> \author Patrick Seewald
! **************************************************************************************************

MODULE dbcsr_rectangular_io
   USE dbcsr_rectangular_types, ONLY: &
      dbcsr_r_type, dbcsr_r_split_info
   USE dbcsr_rectangular_global, ONLY: &
      dbcsr_r_rowcol_data, dbcsr_r_distribution
   USE dbcsr_kinds, ONLY: &
      int_8, real_8, default_string_length
   USE dbcsr_rectangular_base, ONLY: &
      dbcsr_r_get_info, dbcsr_r_get_num_blocks, dbcsr_r_get_num_blocks_total, dbcsr_r_get_nze_total, &
      dbcsr_r_get_nze, dbcsr_r_nblkrows_total, dbcsr_r_nblkcols_total
   USE dbcsr_rectangular_split, ONLY: &
      dbcsr_r_get_split_info, rowsplit, colsplit
   USE dbcsr_mpiwrap, ONLY: &
      mp_environ, mp_sum, mp_max
   USE dbcsr_dist_methods, ONLY: &
      dbcsr_distribution_row_dist, dbcsr_distribution_col_dist

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_rectangular_io'

   PUBLIC :: &
      dbcsr_r_write_dist, &
      dbcsr_r_write_matrix_info, &
      dbcsr_r_write_split_info

CONTAINS

! **************************************************************************************************
!> \brief Write basic infos of rectangular matrix: block dimensions, full dimensions, process grid dimensions
!> \param matrix ...
!> \param output_unit ...
!> \param full_info Whether to print distribution and block size vectors
! **************************************************************************************************
   SUBROUTINE dbcsr_r_write_matrix_info(matrix, output_unit, full_info)
      TYPE(dbcsr_r_type), INTENT(IN) :: matrix
      INTEGER, INTENT(IN)            :: output_unit
      LOGICAL, OPTIONAL, INTENT(IN)  :: full_info

      INTEGER(KIND=int_8)                      :: nblkrows_total, nblkcols_total, nfullrows_total, &
                                                  nfullcols_total
      INTEGER                                  :: nprow, npcol
      CLASS(dbcsr_r_distribution), ALLOCATABLE :: proc_row_dist, proc_col_dist
      CLASS(dbcsr_r_rowcol_data), ALLOCATABLE  :: row_blk_size, col_blk_size
      INTEGER(KIND=int_8)                      :: iblk
      CHARACTER(default_string_length)         :: name

      CALL dbcsr_r_get_info(matrix, nblkrows_total=nblkrows_total, nblkcols_total=nblkcols_total, &
                            nfullrows_total=nfullrows_total, nfullcols_total=nfullcols_total, &
                            nprow=nprow, npcol=npcol, proc_row_dist=proc_row_dist, proc_col_dist=proc_col_dist, &
                            row_blk_size=row_blk_size, col_blk_size=col_blk_size, name=name)

      IF (output_unit > 0) THEN
         WRITE (output_unit, "(T2,A)") &
            "GLOBAL INFO OF "//TRIM(name)
         WRITE (output_unit, "(T4,A,1X)", advance="no") "block dimensions:"
         WRITE (output_unit, "(I12,I12)", advance="no") nblkrows_total, nblkcols_total
         WRITE (output_unit, "(/T4,A,1X)", advance="no") "full dimensions:"
         WRITE (output_unit, "(I14,I14)", advance="no") nfullrows_total, nfullcols_total
         WRITE (output_unit, "(/T4,A,1X)", advance="no") "process grid dimensions:"
         WRITE (output_unit, "(I10,I10)", advance="no") nprow, npcol
         IF (PRESENT(full_info)) THEN
            IF (full_info) THEN
               WRITE (output_unit, '(/T4,A)', advance='no') "Block sizes:"
               WRITE (output_unit, '(/T8,A)', advance='no') 'Row:'
               DO iblk = 1, row_blk_size%nmrowcol
                  WRITE (output_unit, '(I4,1X)', advance='no') row_blk_size%data(iblk)
               ENDDO
               WRITE (output_unit, '(/T8,A)', advance='no') 'Column:'
               DO iblk = 1, col_blk_size%nmrowcol
                  WRITE (output_unit, '(I4,1X)', advance='no') col_blk_size%data(iblk)
               ENDDO
               WRITE (output_unit, '(/T4,A)', advance='no') "Block distribution:"
               WRITE (output_unit, '(/T8,A)', advance='no') 'Row:'
               DO iblk = 1, proc_row_dist%nmrowcol
                  WRITE (output_unit, '(I4,1X)', advance='no') proc_row_dist%dist(iblk)
               ENDDO
               WRITE (output_unit, '(/T8,A)', advance='no') 'Column:'
               DO iblk = 1, proc_col_dist%nmrowcol
                  WRITE (output_unit, '(I4,1X)', advance='no') proc_col_dist%dist(iblk)
               ENDDO

            ENDIF
         ENDIF
         WRITE (output_unit, *)
      ENDIF

   END SUBROUTINE

! **************************************************************************************************
!> \brief Write info on rectangular matrix distribution & load balance
!> \param matrix ...
!> \param output_unit ...
!> \param full_info Whether to print subgroup DBCSR distribution
! **************************************************************************************************
   SUBROUTINE dbcsr_r_write_dist(matrix, output_unit, full_info)
      TYPE(dbcsr_r_type), INTENT(IN)   :: matrix
      INTEGER, INTENT(IN)              :: output_unit
      LOGICAL, INTENT(IN), OPTIONAL    :: full_info

      CHARACTER(default_string_length) :: name
      INTEGER                          :: mp_comm, ngroup, igroup, mp_comm_group, nproc, iproc, &
                                          nblock_p_max, nelement_p_max, nblock_s, nelement_s, &
                                          nblock_s_max, nelement_s_max, nblock, nelement
      INTEGER, DIMENSION(2)            :: tmp
      INTEGER(KIND=int_8)              :: nblock_tot, nblock_p_sum, nelement_p_sum
      REAL(KIND=real_8)                :: occupation
      INTEGER, DIMENSION(:), POINTER   :: rowdist, coldist
      INTEGER                          :: split_rowcol, icol, irow

      CALL dbcsr_r_get_info(matrix, name=name)
      CALL dbcsr_r_get_split_info(matrix%dist%info, mp_comm, ngroup, igroup, mp_comm_group, split_rowcol)
      CALL mp_environ(nproc, iproc, mp_comm)

      nblock = dbcsr_r_get_num_blocks(matrix)
      nelement = dbcsr_r_get_nze(matrix)

      nblock_p_sum = dbcsr_r_get_num_blocks_total(matrix)
      nelement_p_sum = dbcsr_r_get_nze_total(matrix)

      tmp = (/nblock, nelement/)
      CALL mp_max(tmp, mp_comm)
      nblock_p_max = tmp(1); nelement_p_max = tmp(2)

      nblock_s = nblock
      nelement_s = nelement

      CALL mp_sum(nblock_s, mp_comm_group)
      CALL mp_sum(nelement_s, mp_comm_group)

      tmp = (/nblock_s, nelement_s/)
      CALL mp_max(tmp, mp_comm)
      nblock_s_max = tmp(1); nelement_s_max = tmp(2)

      nblock_tot = dbcsr_r_nblkrows_total(matrix)*dbcsr_r_nblkcols_total(matrix)
      occupation = -1.0_real_8
      IF (nblock_tot .NE. 0) occupation = 100.0_real_8*REAL(nblock_p_sum, real_8)/REAL(nblock_tot, real_8)

      rowdist => dbcsr_distribution_row_dist(matrix%matrix%dist)
      coldist => dbcsr_distribution_col_dist(matrix%matrix%dist)

      IF (output_unit > 0) THEN
         WRITE (output_unit, "(T2,A)") &
            "DISTRIBUTION OF "//TRIM(name)
         WRITE (output_unit, "(T15,A,T68,I13)") "Number of non-zero blocks:", nblock_p_sum
         WRITE (output_unit, "(T15,A,T75,F6.2)") "Percentage of non-zero blocks:", occupation
         WRITE (output_unit, "(T15,A,T68,I13)") "Average number of blocks per group:", (nblock_p_sum + ngroup - 1)/ngroup
         WRITE (output_unit, "(T15,A,T68,I13)") "Maximum number of blocks per group:", nblock_s_max
         WRITE (output_unit, "(T15,A,T68,I13)") "Average number of matrix elements per group:", (nelement_p_sum + ngroup - 1)/ngroup
         WRITE (output_unit, "(T15,A,T68,I13)") "Maximum number of matrix elements per group:", nelement_s_max
         WRITE (output_unit, "(T15,A,T68,I13)") "Average number of blocks per CPU:", (nblock_p_sum + nproc - 1)/nproc
         WRITE (output_unit, "(T15,A,T68,I13)") "Maximum number of blocks per CPU:", nblock_p_max
         WRITE (output_unit, "(T15,A,T68,I13)") "Average number of matrix elements per CPU:", (nelement_p_sum + nproc - 1)/nproc
         WRITE (output_unit, "(T15,A,T68,I13)") "Maximum number of matrix elements per CPU:", nelement_p_max
         IF (PRESENT(full_info)) THEN
            IF (full_info) THEN
               WRITE (output_unit, "(T15,A)") "Row distribution on subgroup:"
               WRITE (output_unit, '(T15)', advance='no')
               DO irow = 1, SIZE(rowdist)
                  WRITE (output_unit, '(I3, 1X)', advance='no') rowdist(irow)
               ENDDO
               WRITE (output_unit, "(/T15,A)") "Column distribution on subgroup:"
               WRITE (output_unit, '(T15)', advance='no')
               DO icol = 1, SIZE(coldist)
                  WRITE (output_unit, '(I3, 1X)', advance='no') coldist(icol)
               ENDDO
               WRITE (output_unit, *)
            ENDIF
         ENDIF
      ENDIF
   END SUBROUTINE

! **************************************************************************************************
!> \brief Print info on how matrix is split
!> \param info ...
!> \param name ...
!> \param io_unit ...
! **************************************************************************************************
   SUBROUTINE dbcsr_r_write_split_info(info, name, io_unit)
      TYPE(dbcsr_r_split_info), INTENT(IN)               :: info
      CHARACTER(len=*), INTENT(IN)                       :: name
      INTEGER, INTENT(IN) :: io_unit
      INTEGER                                            :: groupsize, igroup, mp_comm, &
                                                            mp_comm_group, mynode, nsplit, &
                                                            numnodes, split_rowcol
      INTEGER, DIMENSION(2)                              :: coord, dims, groupcoord, groupdims, &
                                                            pgrid_offset

      CALL dbcsr_r_get_split_info(info, mp_comm, nsplit, igroup, mp_comm_group, split_rowcol, pgrid_offset)
      CALL mp_environ(numnodes, mynode, mp_comm)
      CALL mp_environ(numnodes, dims, coord, mp_comm)
      CALL mp_environ(groupsize, groupdims, groupcoord, mp_comm_group)

      IF (io_unit > 0) THEN
         SELECT CASE (split_rowcol)
         CASE (rowsplit)
            WRITE (io_unit, "(T4,A,I4,1X,A,I4)") TRIM(name)//" splitting rows by factor", nsplit
         CASE (colsplit)
            WRITE (io_unit, "(T4,A,I4,1X,A,I4)") TRIM(name)//" splitting columns by factor", nsplit
         END SELECT
         WRITE (io_unit, "(T4,A, I4,A1,I4)") TRIM(name)//" global grid sizes:", dims(1), "x", dims(2)
      ENDIF

      IF (io_unit > 0) THEN
         WRITE (io_unit, "(T4,A,I4,A1,I4)") &
            TRIM(name)//" grid sizes on subgroups:", &
            groupdims(1), "x", groupdims(2)
      ENDIF

   END SUBROUTINE

END MODULE

