!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief methods to split tall-and-skinny matrices along longest dimension.
!>
!> Basically, we are splitting process grid and each subgrid holds its own DBCSR matrix.
!> \author Patrick Seewald
! **************************************************************************************************
MODULE dbcsr_tas_split

   USE dbcsr_tas_types, ONLY: dbcsr_tas_distribution_type, &
                              dbcsr_tas_create_split_info
   USE dbcsr_tas_global, ONLY: dbcsr_tas_distribution
   USE dbcsr_tas_util, ONLY: swap
   USE dbcsr_toollib, ONLY: sort
   USE dbcsr_kinds, ONLY: int_8
   USE dbcsr_mpiwrap, ONLY: mp_bcast, &
                            mp_cart_create, &
                            mp_comm_dup, &
                            mp_comm_free, &
                            mp_comm_split_direct, &
                            mp_dims_create, &
                            mp_environ
#include "../base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   PUBLIC :: &
      block_index_global_to_local, &
      block_index_local_to_global, &
      colsplit, &
      dbcsr_tas_get_split_info, &
      dbcsr_tas_info_hold, &
      dbcsr_tas_mp_comm, &
      dbcsr_tas_release_info, &
      dbcsr_tas_create_split, &
      dbcsr_tas_create_split_rows_or_cols, &
      group_to_mrowcol, &
      group_to_world_proc_map, &
      mrowcol_to_group, &
      rowsplit, &
      world_to_group_proc_map

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tas_split'

   INTEGER, PARAMETER :: rowsplit = 1, colsplit = 2

CONTAINS

! **************************************************************************************************
!> \brief split mpi grid by rows or columns
!> \param split_info ...
!> \param mp_comm global mpi communicator with a 2d cartesian grid
!> \param ngroup number of groups. Note: this number may be diminished on out in order to have a
!>        distribution where all groups have the same number of processes
!> \param igroup my group ID
!> \param split_rowcol split rows or columns
! **************************************************************************************************
   SUBROUTINE dbcsr_tas_create_split_rows_or_cols(split_info, mp_comm, ngroup, igroup, split_rowcol)
      TYPE(dbcsr_tas_create_split_info), INTENT(OUT)              :: split_info
      INTEGER, INTENT(IN)                                :: mp_comm
      INTEGER, INTENT(INOUT)                             :: ngroup
      INTEGER, INTENT(IN)                                :: igroup, split_rowcol

      INTEGER :: &
         igroup_check, iproc, iproc_group, iproc_group_check, mp_comm_group, numproc, &
         numproc_group, numproc_group_check, handle
      INTEGER, DIMENSION(2)                              :: pcoord, pcoord_group, pdims, pdims_group
      LOGICAL                                            :: to_assert
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_create_split_rows_or_cols', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)

      CALL mp_comm_dup(mp_comm, split_info%mp_comm)
      split_info%igroup = igroup
      split_info%split_rowcol = split_rowcol

      CALL mp_comm_split_direct(mp_comm, mp_comm_group, igroup)

      CALL mp_environ(numproc, iproc, mp_comm)
      CALL mp_environ(numproc, pdims, pcoord, mp_comm)

      CALL mp_environ(numproc_group, iproc_group, mp_comm_group)

      IF (iproc == 0) THEN
         to_assert = MOD(numproc_group, pdims(MOD(split_rowcol, 2) + 1)) == 0
         DBCSR_ASSERT(to_assert)
         split_info%pgrid_split_size = numproc_group/pdims(MOD(split_rowcol, 2) + 1)
      ENDIF
      CALL mp_bcast(split_info%pgrid_split_size, 0, split_info%mp_comm)

      ngroup = (pdims(split_rowcol) + split_info%pgrid_split_size - 1)/split_info%pgrid_split_size
      split_info%ngroup = ngroup
      split_info%group_size = split_info%pgrid_split_size*pdims(MOD(split_rowcol, 2) + 1)

      CALL world_to_group_proc_map(iproc, pdims, split_rowcol, split_info%pgrid_split_size, igroup_check, pdims_group, iproc_group)

      IF (igroup_check .NE. split_info%igroup) THEN
         DBCSR_ABORT('inconsistent subgroups')
      ENDIF

      CALL mp_cart_create(mp_comm_group, 2, pdims_group, pcoord_group, split_info%mp_comm_group)

      CALL mp_environ(numproc_group_check, iproc_group_check, split_info%mp_comm_group)

      DBCSR_ASSERT(iproc_group_check .EQ. iproc_group)

      CALL mp_comm_free(mp_comm_group)

      ALLOCATE (split_info%refcount)
      split_info%refcount = 1

      CALL timestop(handle)

   END SUBROUTINE

! **************************************************************************************************
!> \brief Create default cartesian grid that is consistent with default split heuristic of dbcsr_tas_create_split
!>        This is experimental and should be used for testing purposes only!
!> \param mp_comm ...
!> \param nblkrows_total ...
!> \param nblkcols_total ...
!> \return new communicator
! **************************************************************************************************
   FUNCTION dbcsr_tas_mp_comm(mp_comm, nblkrows_total, nblkcols_total)
      INTEGER, INTENT(IN)                                :: mp_comm
      INTEGER(KIND=int_8), INTENT(IN)                    :: nblkrows_total, nblkcols_total
      INTEGER                                            :: dbcsr_tas_mp_comm

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_mp_comm', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: group_size, handle, iproc, numproc, s, &
                                                            split_rowcol
      INTEGER(KIND=int_8)                                :: d_max, d_min
      INTEGER, DIMENSION(2)                              :: group_dims, mdims_group, myploc, npdims, &
                                                            sort_indices

      CALL timeset(routineN, handle)

      d_max = MAX(nblkrows_total, nblkcols_total)
      d_min = MIN(nblkrows_total, nblkcols_total)

      s = INT((d_max - 1)/d_min + 1)

      CALL mp_environ(numproc, iproc, mp_comm)

      DO WHILE (MOD(numproc, s) .NE. 0)
         s = s - 1
      ENDDO

      group_size = numproc/s
      group_dims(:) = 0

      CALL mp_dims_create(group_size, group_dims)
      CALL sort(group_dims, 2, sort_indices)

      IF (d_max == nblkrows_total) THEN
         split_rowcol = rowsplit
      ELSE
         split_rowcol = colsplit
      ENDIF

      SELECT CASE (split_rowcol)
      CASE (rowsplit)
         mdims_group = [INT(nblkrows_total/s), INT(nblkcols_total)]
      CASE (colsplit)
         mdims_group = [INT(nblkrows_total), INT(nblkcols_total/s)]
      END SELECT

      IF (mdims_group(1) .GT. mdims_group(2)) THEN
         CALL swap(group_dims)
      ENDIF

      SELECT CASE (split_rowcol)
      CASE (rowsplit)
         npdims(:) = [group_dims(1)*s, group_dims(2)]
      CASE (colsplit)
         npdims(:) = [group_dims(1), group_dims(2)*s]
      END SELECT

      CALL mp_cart_create(mp_comm, 2, npdims, myploc, dbcsr_tas_mp_comm)

      CALL timestop(handle)
   END FUNCTION

! **************************************************************************************************
!> \brief Split Cartesian process grid using a default split heuristic. The split factor may not be
!>        ideal.
!> \param split_info object storing all data corresponding to split, submatrices and parallelization
!> \param mp_comm ...
!> \param nblkrows_total ...
!> \param nblkcols_total ...
!> \param nosplit Matrix should not be split
! **************************************************************************************************
   SUBROUTINE dbcsr_tas_create_split(split_info, mp_comm, nblkrows_total, nblkcols_total, nosplit)
      TYPE(dbcsr_tas_create_split_info), INTENT(OUT)              :: split_info
      INTEGER, INTENT(IN)                                :: mp_comm
      INTEGER(KIND=int_8), INTENT(IN)                    :: nblkrows_total, nblkcols_total
      LOGICAL, INTENT(IN), OPTIONAL                      :: nosplit

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_create_split', routineP = moduleN//':'//routineN

      INTEGER                                            :: handle, igroup, iproc, numproc, s, &
                                                            split_rowcol
      INTEGER(KIND=int_8)                                :: d_max, d_min
      INTEGER, DIMENSION(2)                              :: pcoord, pdims, pdims_group
      LOGICAL                                            :: nosplit_prv

      CALL timeset(routineN, handle)

      IF (PRESENT(nosplit)) THEN
         nosplit_prv = nosplit
      ELSE
         nosplit_prv = .FALSE.
      ENDIF

      IF (nosplit_prv) THEN
         s = 1
         igroup = 0
         split_rowcol = rowsplit

      ELSE

         d_max = MAX(nblkrows_total, nblkcols_total)
         d_min = MIN(nblkrows_total, nblkcols_total)

         s = INT((d_max - 1)/d_min + 1)

         CALL mp_environ(numproc, iproc, mp_comm)
         CALL mp_environ(numproc, pdims, pcoord, mp_comm)

         IF (d_max == nblkrows_total) THEN
            split_rowcol = rowsplit
         ELSE
            split_rowcol = colsplit
         ENDIF

         DO WHILE (MOD(pdims(split_rowcol), s) .NE. 0)
            s = s - 1
         ENDDO

         SELECT CASE (split_rowcol)
         CASE (rowsplit)
            pdims_group = [pdims(1)/s, pdims(2)]
         CASE (colsplit)
            pdims_group = [pdims(1), pdims(2)/s]
         END SELECT

         igroup = pcoord(split_rowcol)/pdims_group(split_rowcol)

      ENDIF
      CALL dbcsr_tas_create_split_rows_or_cols(split_info, mp_comm, s, igroup, split_rowcol)

      CALL timestop(handle)

   END SUBROUTINE

! **************************************************************************************************
!> \brief Get info on split
!> \param info ...
!> \param mp_comm communicator (global process grid)
!> \param nsplit split factor
!> \param igroup which group do I belong to
!> \param mp_comm_group subgroup communicator (group-local process grid)
!> \param split_rowcol split rows or columns
!> \param pgrid_offset group-local offset in process grid
! **************************************************************************************************
   SUBROUTINE dbcsr_tas_get_split_info(info, mp_comm, nsplit, igroup, mp_comm_group, split_rowcol, pgrid_offset)
      TYPE(dbcsr_tas_create_split_info), INTENT(IN)               :: info
      INTEGER, INTENT(OUT), OPTIONAL                     :: mp_comm, nsplit, igroup, mp_comm_group, &
                                                            split_rowcol
      INTEGER, DIMENSION(2), INTENT(OUT), OPTIONAL       :: pgrid_offset

      IF (PRESENT(mp_comm)) mp_comm = info%mp_comm
      IF (PRESENT(mp_comm_group)) mp_comm_group = info%mp_comm_group
      IF (PRESENT(split_rowcol)) split_rowcol = info%split_rowcol
      IF (PRESENT(igroup)) igroup = info%igroup
      IF (PRESENT(nsplit)) nsplit = info%ngroup

      IF (PRESENT(pgrid_offset)) THEN
         SELECT CASE (info%split_rowcol)
         CASE (rowsplit)
            pgrid_offset(:) = [info%igroup*info%pgrid_split_size, 0]
         CASE (colsplit)
            pgrid_offset(:) = [0, info%igroup*info%pgrid_split_size]
         END SELECT
      ENDIF

   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param split_info ...
! **************************************************************************************************
   SUBROUTINE dbcsr_tas_release_info(split_info)
      TYPE(dbcsr_tas_create_split_info), INTENT(INOUT)            :: split_info
      LOGICAL                                            :: abort

      abort = .FALSE.

      IF (.NOT. ASSOCIATED(split_info%refcount)) THEN
         abort = .TRUE.
      ELSEIF (split_info%refcount < 1) THEN
         abort = .TRUE.
      ENDIF

      IF (abort) THEN
         DBCSR_ABORT("can not destroy non-existing split_info")
      ENDIF

      split_info%refcount = split_info%refcount - 1

      IF (split_info%refcount == 0) THEN
         CALL mp_comm_free(split_info%mp_comm_group)
         CALL mp_comm_free(split_info%mp_comm)
         DEALLOCATE (split_info%refcount)
      ENDIF
   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param split_info ...
! **************************************************************************************************
   SUBROUTINE dbcsr_tas_info_hold(split_info)
      TYPE(dbcsr_tas_create_split_info), INTENT(IN)               :: split_info

      INTEGER, POINTER                                   :: ref

      IF (split_info%refcount < 1) THEN
         DBCSR_ABORT("can not hold non-existing split_info")
      ENDIF
      ref => split_info%refcount
      ref = ref + 1
   END SUBROUTINE

! **************************************************************************************************
!> \brief map global process info to group
!> \param iproc global process ID
!> \param pdims global process dimensions
!> \param split_rowcol split rows or column
!> \param pgrid_split_size how many process rows/cols per group
!> \param igroup group ID
!> \param pdims_group local process grid dimensions
!> \param iproc_group group local process ID
! **************************************************************************************************
   SUBROUTINE world_to_group_proc_map(iproc, pdims, split_rowcol, pgrid_split_size, igroup, &
                                      pdims_group, iproc_group)
      INTEGER, INTENT(IN)                                :: iproc
      INTEGER, DIMENSION(2), INTENT(IN)                  :: pdims
      INTEGER, INTENT(IN)                                :: split_rowcol, pgrid_split_size
      INTEGER, INTENT(OUT)                               :: igroup
      INTEGER, DIMENSION(2), INTENT(OUT), OPTIONAL       :: pdims_group
      INTEGER, INTENT(OUT), OPTIONAL                     :: iproc_group

      INTEGER, DIMENSION(2)                              :: pcoord, pcoord_group

      IF (PRESENT(iproc_group)) THEN
         DBCSR_ASSERT(PRESENT(pdims_group))
      ENDIF

      pcoord = [iproc/pdims(2), MOD(iproc, pdims(2))]

      igroup = pcoord(split_rowcol)/pgrid_split_size

      SELECT CASE (split_rowcol)
      CASE (rowsplit)
         IF (PRESENT(pdims_group)) pdims_group = [pgrid_split_size, pdims(2)]
         IF (PRESENT(iproc_group)) pcoord_group = [MOD(pcoord(1), pgrid_split_size), pcoord(2)]
      CASE (colsplit)
         IF (PRESENT(pdims_group)) pdims_group = [pdims(1), pgrid_split_size]
         IF (PRESENT(iproc_group)) pcoord_group = [pcoord(1), MOD(pcoord(2), pgrid_split_size)]
      END SELECT
      IF (PRESENT(iproc_group)) iproc_group = pcoord_group(1)*pdims_group(2) + pcoord_group(2)
   END SUBROUTINE

! **************************************************************************************************
!> \brief map local process info to global info
!> \param iproc global process id
!> \param pdims global process grid dimensions
!> \param split_rowcol split rows or column
!> \param pgrid_split_size how many process rows/cols per group
!> \param igroup group ID
!> \param iproc_group local process ID
! **************************************************************************************************
   SUBROUTINE group_to_world_proc_map(iproc, pdims, split_rowcol, pgrid_split_size, &
                                      igroup, iproc_group)
      INTEGER, INTENT(OUT)                               :: iproc
      INTEGER, DIMENSION(2), INTENT(IN)                  :: pdims
      INTEGER, INTENT(IN)                                :: split_rowcol, pgrid_split_size, &
                                                            igroup, iproc_group

      INTEGER, DIMENSION(2)                              :: pcoord, pcoord_group, pdims_group

      SELECT CASE (split_rowcol)
      CASE (rowsplit)
         pdims_group = [pgrid_split_size, pdims(2)]
      CASE (colsplit)
         pdims_group = [pdims(1), pgrid_split_size]
      END SELECT

      pcoord_group = [iproc_group/pdims_group(2), MOD(iproc_group, pdims_group(2))]

      SELECT CASE (split_rowcol)
      CASE (rowsplit)
         pcoord = [igroup*pgrid_split_size + pcoord_group(1), pcoord_group(2)]
      CASE (colsplit)
         pcoord = [pcoord_group(1), igroup*pgrid_split_size + pcoord_group(2)]
      END SELECT
      iproc = pcoord(1)*pdims(2) + pcoord(2)
   END SUBROUTINE

! **************************************************************************************************
!> \brief map group local block index to global matrix index
!> \param info ...
!> \param dist ...
!> \param row_group group local row block index
!> \param column_group group local column block index
!> \param row global block row
!> \param column global block column
! **************************************************************************************************
   SUBROUTINE block_index_local_to_global(info, dist, row_group, column_group, &
                                          row, column)
      TYPE(dbcsr_tas_create_split_info), INTENT(IN)               :: info
      TYPE(dbcsr_tas_distribution_type), INTENT(IN)        :: dist
      INTEGER, INTENT(IN), OPTIONAL                      :: row_group, column_group
      INTEGER(KIND=int_8), INTENT(OUT), OPTIONAL         :: row, column

      IF (PRESENT(row)) THEN
         DBCSR_ASSERT(PRESENT(row_group))
      ENDIF
      IF (PRESENT(column)) THEN
         DBCSR_ASSERT(PRESENT(column_group))
      ENDIF

      SELECT CASE (info%split_rowcol)
      CASE (rowsplit)
         ASSOCIATE (rows=>dist%local_rowcols)
            IF (PRESENT(row)) row = rows(row_group)
            IF (PRESENT(column)) column = column_group
         END ASSOCIATE
      CASE (colsplit)
         ASSOCIATE (cols=>dist%local_rowcols)
            IF (PRESENT(row)) row = row_group
            IF (PRESENT(column)) column = cols(column_group)
         END ASSOCIATE
      END SELECT
   END SUBROUTINE

! **************************************************************************************************
!> \brief map global block index to group local index
!> \param info ...
!> \param dist ...
!> \param row ...
!> \param column ...
!> \param row_group ...
!> \param column_group ...
! **************************************************************************************************
   SUBROUTINE block_index_global_to_local(info, dist, row, column, row_group, column_group)
      TYPE(dbcsr_tas_create_split_info), INTENT(IN)               :: info
      TYPE(dbcsr_tas_distribution_type), INTENT(IN)        :: dist
      INTEGER(KIND=int_8), INTENT(IN), OPTIONAL          :: row, column
      INTEGER, INTENT(OUT), OPTIONAL                     :: row_group, column_group
      CHARACTER(LEN=*), PARAMETER :: routineN = 'block_index_global_to_local', &
                                     routineP = moduleN//':'//routineN

      IF (PRESENT(row_group)) THEN
         DBCSR_ASSERT(PRESENT(row))
      ENDIF
      IF (PRESENT(column_group)) THEN
         DBCSR_ASSERT(PRESENT(column))
      ENDIF

      SELECT CASE (info%split_rowcol)
      CASE (rowsplit)
         IF (PRESENT(row_group)) CALL mrowcol_to_group(dist, row, row_group)
         IF (PRESENT(column_group)) column_group = INT(column)
      CASE (colsplit)
         IF (PRESENT(row_group)) row_group = INT(row)
         IF (PRESENT(column_group)) CALL mrowcol_to_group(dist, column, column_group)
      END SELECT

   END SUBROUTINE

! **************************************************************************************************
!> \brief map matrix rows/column to group local rows/column
!> \param dist ...
!> \param rowcol global row/column to map
!> \param rowcol_group group local row/column
! **************************************************************************************************
   SUBROUTINE mrowcol_to_group(dist, rowcol, rowcol_group)

      CLASS(dbcsr_tas_distribution_type), INTENT(IN) :: dist
      INTEGER(KIND=int_8), INTENT(IN)              :: rowcol
      INTEGER, INTENT(OUT)                         :: rowcol_group
      CHARACTER(LEN=*), PARAMETER                  :: routineN = 'mrowcol_to_group', &
                                                      routineP = moduleN//':'//routineN

      ASSOCIATE (rowcols=>dist%local_rowcols)
         rowcol_group = i8_bsearch(rowcols, rowcol)
      END ASSOCIATE
   END SUBROUTINE

! **************************************************************************************************
!> \brief binary search for 8-byte integers
!> \param array ...
!> \param el ...
!> \param l_index ...
!> \param u_index ...
!> \return ...
! **************************************************************************************************
   FUNCTION i8_bsearch(array, el, l_index, u_index) result(res)
      INTEGER(KIND=int_8), intent(in) :: array(:)
      INTEGER(KIND=int_8), intent(in) :: el
      INTEGER, INTENT(in), OPTIONAL   :: l_index, u_index
      INTEGER                         :: res, lindex, uindex, aindex

      character(len=*), parameter :: routineN = 'i8_bsearch', &
                                     routineP = moduleN//':'//routineN

      lindex = 1
      uindex = size(array)
      if (present(l_index)) lindex = l_index
      if (present(u_index)) uindex = u_index
      DO WHILE (lindex <= uindex)
         aindex = (lindex + uindex)/2
         IF (array(aindex) < el) THEN
            lindex = aindex + 1
         ELSE
            uindex = aindex - 1
         END IF
      END DO
      res = lindex
   END FUNCTION

! **************************************************************************************************
!> \brief maps a process subgroup to matrix rows/columns
!> \param info ...
!> \param rowcol_dist ...
!> \param igroup group ID
!> \param rowcols rows/ columns on this group
! **************************************************************************************************
   SUBROUTINE group_to_mrowcol(info, rowcol_dist, igroup, rowcols)
      TYPE(dbcsr_tas_create_split_info), INTENT(IN)                        :: info
      CLASS(dbcsr_tas_distribution), INTENT(IN)                     :: rowcol_dist
      INTEGER, INTENT(IN)                                         :: igroup
      INTEGER(KIND=int_8), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: rowcols
      INTEGER, DIMENSION(0:info%pgrid_split_size - 1)             :: nrowcols_group
      INTEGER                                                     :: pcoord, nrowcols, count, pcoord_group
      INTEGER, DIMENSION(:), ALLOCATABLE                          :: sort_indices

      nrowcols_group(:) = 0
      DO pcoord = igroup*info%pgrid_split_size, (igroup + 1)*info%pgrid_split_size - 1
         pcoord_group = pcoord - igroup*info%pgrid_split_size
         nrowcols_group(pcoord_group) = SIZE(rowcol_dist%rowcols(pcoord))
      ENDDO
      nrowcols = SUM(nrowcols_group)

      ALLOCATE (rowcols(nrowcols))

      count = 0
      DO pcoord = igroup*info%pgrid_split_size, (igroup + 1)*info%pgrid_split_size - 1
         pcoord_group = pcoord - igroup*info%pgrid_split_size
         rowcols(count + 1:count + nrowcols_group(pcoord_group)) = rowcol_dist%rowcols(pcoord)
         count = count + nrowcols_group(pcoord_group)
      ENDDO

      ALLOCATE (sort_indices(nrowcols))
      CALL sort(rowcols, nrowcols, sort_indices)
   END SUBROUTINE

END MODULE
