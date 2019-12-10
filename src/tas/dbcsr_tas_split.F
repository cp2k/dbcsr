!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tas_split
   !! methods to split tall-and-skinny matrices along longest dimension.
   !! Basically, we are splitting process grid and each subgrid holds its own DBCSR matrix.

   USE dbcsr_tas_types, ONLY: &
      dbcsr_tas_distribution_type, dbcsr_tas_split_info
   USE dbcsr_tas_global, ONLY: &
      dbcsr_tas_distribution
   USE dbcsr_tas_util, ONLY: &
      swap
   USE dbcsr_toollib, ONLY: &
      sort
   USE dbcsr_kinds, ONLY: &
      int_8
   USE dbcsr_mpiwrap, ONLY: &
      mp_bcast, mp_cart_create, mp_comm_dup, mp_comm_free, mp_comm_split_direct, mp_dims_create, mp_environ
   USE dbcsr_kinds, ONLY: &
      real_8
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
      dbcsr_tas_mp_dims, &
      dbcsr_tas_release_info, &
      dbcsr_tas_create_split, &
      dbcsr_tas_create_split_rows_or_cols, &
      group_to_mrowcol, &
      group_to_world_proc_map, &
      mrowcol_to_group, &
      rowsplit, &
      world_to_group_proc_map, &
      accept_pgrid_dims, &
      default_nsplit_accept_ratio_1

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tas_split'

   INTEGER, PARAMETER :: rowsplit = 1, colsplit = 2
   REAL(real_8), PARAMETER :: default_pdims_accept_ratio = 1.2_real_8
   REAL(real_8), PARAMETER :: default_nsplit_accept_ratio_1 = 1.5_real_8
   REAL(real_8), PARAMETER :: default_nsplit_accept_ratio_2 = 2.5_real_8

   INTERFACE dbcsr_tas_mp_comm
      MODULE PROCEDURE dbcsr_tas_mp_comm
      MODULE PROCEDURE dbcsr_tas_mp_comm_from_matrix_sizes
   END INTERFACE

CONTAINS

   SUBROUTINE dbcsr_tas_create_split_rows_or_cols(split_info, mp_comm, ngroup, igroup, split_rowcol, own_comm)
      !! split mpi grid by rows or columns

      TYPE(dbcsr_tas_split_info), INTENT(OUT)            :: split_info
      INTEGER, INTENT(IN)                                :: mp_comm
         !! global mpi communicator with a 2d cartesian grid
      INTEGER, INTENT(INOUT)                             :: ngroup
         !! number of groups
      INTEGER, INTENT(IN)                                :: igroup, split_rowcol
         !! my group ID
         !! split rows or columns
      LOGICAL, INTENT(IN), OPTIONAL                      :: own_comm
         !! Whether split_info should own communicator

      INTEGER :: &
         igroup_check, iproc, iproc_group, iproc_group_check, mp_comm_group, numproc, &
         numproc_group, numproc_group_check, handle
      INTEGER, DIMENSION(2)                              :: pcoord, pcoord_group, pdims, pdims_group
      LOGICAL                                            :: to_assert, own_comm_prv
      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_create_split_rows_or_cols', &
                                     routineP = moduleN//':'//routineN

      CALL timeset(routineN, handle)

      IF (PRESENT(own_comm)) THEN
         own_comm_prv = own_comm
      ELSE
         own_comm_prv = .FALSE.
      ENDIF

      IF (own_comm_prv) THEN
         split_info%mp_comm = mp_comm
      ELSE
         CALL mp_comm_dup(mp_comm, split_info%mp_comm)
      ENDIF

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

   FUNCTION dbcsr_tas_mp_comm(mp_comm, split_rowcol, nsplit)
      !! Create default cartesian process grid that is consistent with default split heuristic
      !! of dbcsr_tas_create_split
      !! \return new communicator

      INTEGER, INTENT(IN)                                :: mp_comm
      INTEGER, INTENT(IN)                                :: split_rowcol
      INTEGER, INTENT(INOUT)                             :: nsplit
      INTEGER                                            :: dbcsr_tas_mp_comm

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_tas_mp_comm', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: handle, iproc, numproc
      INTEGER, DIMENSION(2)                              :: myploc, npdims

      CALL timeset(routineN, handle)

      CALL mp_environ(numproc, iproc, mp_comm)

      npdims = dbcsr_tas_mp_dims(numproc, split_rowcol, nsplit)

      CALL mp_cart_create(mp_comm, 2, npdims, myploc, dbcsr_tas_mp_comm)

      CALL timestop(handle)
   END FUNCTION

   FUNCTION dbcsr_tas_mp_dims(numproc, split_rowcol, nsplit)
      !! Get optimal process grid dimensions consistent with dbcsr_tas_create_split
      INTEGER, INTENT(IN)                                :: numproc
      INTEGER, INTENT(IN)                                :: split_rowcol
      INTEGER, INTENT(IN)                                :: nsplit
      INTEGER, DIMENSION(2)                              :: dbcsr_tas_mp_dims

      INTEGER                                            :: dbcsr_tas_mp_comm, nsplit_opt
      INTEGER                                            :: group_size
      INTEGER, DIMENSION(2)                              :: group_dims

      nsplit_opt = get_opt_nsplit(numproc, nsplit, split_pgrid=.FALSE.)

      group_size = numproc/nsplit_opt
      group_dims(:) = 0

      CALL mp_dims_create(group_size, group_dims)

      ! here we choose order of group dims s.t. a split factor < nsplit_opt is favoured w.r.t.
      ! optimal subgrid dimensions
      SELECT CASE (split_rowcol)
      CASE (rowsplit)
         group_dims = [MINVAL(group_dims), MAXVAL(group_dims)]
      CASE (colsplit)
         group_dims = [MAXVAL(group_dims), MINVAL(group_dims)]
      END SELECT

      SELECT CASE (split_rowcol)
      CASE (rowsplit)
         dbcsr_tas_mp_dims(:) = [group_dims(1)*nsplit_opt, group_dims(2)]
      CASE (colsplit)
         dbcsr_tas_mp_dims(:) = [group_dims(1), group_dims(2)*nsplit_opt]
      END SELECT

   END FUNCTION

   FUNCTION get_opt_nsplit(numproc, nsplit, split_pgrid, pdim_nonsplit)
      !! Heuristic to get good split factor for a given process grid OR a given number of processes
      !! \return split factor consistent with process grid or number of processes

      INTEGER, INTENT(IN)                        :: numproc, nsplit
         !! total number of processes or (if split_pgrid) process grid dimension to split
         !! Desired split factor
      LOGICAL, INTENT(IN)                        :: split_pgrid
         !! whether to split process grid
      INTEGER, INTENT(IN), OPTIONAL              :: pdim_nonsplit
         !! if split_pgrid: other process grid dimension
      INTEGER, ALLOCATABLE, DIMENSION(:)         :: nsplit_list, nsplit_list_square, nsplit_list_accept
      INTEGER                                    :: lb, ub, count, count_square, count_accept, split, &
                                                    minpos, get_opt_nsplit
      INTEGER, DIMENSION(2)                      :: dims_sub
      REAL(KIND=real_8)                          :: nsplit_accept_ratio

      DBCSR_ASSERT(nsplit > 0)

      IF (split_pgrid) THEN
         DBCSR_ASSERT(PRESENT(pdim_nonsplit))
      ENDIF

      IF (split_pgrid) THEN
         ! if process grid dimensions are fixed, we need more flexibility in split factor in order
         ! to obtain balanced grid dimensions --> larger accept ratio
         nsplit_accept_ratio = default_nsplit_accept_ratio_2
      ELSE
         ! if process grid dimensions are flexible, less flexibility in split factor sufficient in
         ! order to obtain balanced grid dimensions --> smaller accept ratio
         nsplit_accept_ratio = default_nsplit_accept_ratio_1
      ENDIF

      lb = CEILING(REAL(nsplit, real_8)/nsplit_accept_ratio)
      ub = FLOOR(REAL(nsplit, real_8)*nsplit_accept_ratio)

      IF (ub < lb) ub = lb

      ALLOCATE (nsplit_list(1:ub - lb + 1), nsplit_list_square(1:ub - lb + 1), nsplit_list_accept(1:ub - lb + 1))
      count = 0
      count_square = 0
      count_accept = 0
      DO split = lb, ub
         IF (MOD(numproc, split) == 0) THEN
            count = count + 1
            nsplit_list(count) = split

            dims_sub = 0
            IF (.NOT. split_pgrid) THEN
               CALL mp_dims_create(numproc/split, dims_sub)
            ELSE
               dims_sub = [numproc/split, pdim_nonsplit]
            ENDIF

            IF (dims_sub(1) == dims_sub(2)) THEN
               count_square = count_square + 1
               nsplit_list_square(count_square) = split
               count_accept = count_accept + 1
               nsplit_list_accept(count_accept) = split
            ELSEIF (accept_pgrid_dims(dims_sub, relative=.FALSE.)) THEN
               count_accept = count_accept + 1
               nsplit_list_accept(count_accept) = split
            ENDIF

         ENDIF
      ENDDO

      IF (count_square > 0) THEN
         minpos = MINLOC(ABS(nsplit_list_square(1:count_square) - nsplit), DIM=1)
         get_opt_nsplit = nsplit_list_square(minpos)
      ELSEIF (count_accept > 0) THEN
         minpos = MINLOC(ABS(nsplit_list_accept(1:count_accept) - nsplit), DIM=1)
         get_opt_nsplit = nsplit_list_accept(minpos)
      ELSEIF (count > 0) THEN
         minpos = MINLOC(ABS(nsplit_list(1:count) - nsplit), DIM=1)
         get_opt_nsplit = nsplit_list(minpos)
      ELSE
         get_opt_nsplit = nsplit
         DO WHILE (MOD(numproc, get_opt_nsplit) .NE. 0)
            get_opt_nsplit = get_opt_nsplit - 1
         ENDDO
      ENDIF

   END FUNCTION

   FUNCTION dbcsr_tas_mp_comm_from_matrix_sizes(mp_comm, nblkrows, nblkcols) RESULT(mp_comm_new)
      !! Derive optimal cartesian process grid from matrix sizes. This ensures optimality for
      !! dense matrices only

      INTEGER, INTENT(IN)                                :: mp_comm
      INTEGER(KIND=int_8), INTENT(IN)                    :: nblkrows, nblkcols
         !! total number of block rows
         !! total number of block columns
      INTEGER                                            :: nsplit, split_rowcol
      INTEGER                                            :: mp_comm_new
         !! MPI communicator

      IF (nblkrows >= nblkcols) THEN
         split_rowcol = rowsplit
         nsplit = INT((nblkrows - 1)/nblkcols + 1)
      ELSE
         split_rowcol = colsplit
         nsplit = INT((nblkcols - 1)/nblkrows + 1)
      ENDIF

      mp_comm_new = dbcsr_tas_mp_comm(mp_comm, split_rowcol, nsplit)
   END FUNCTION

   SUBROUTINE dbcsr_tas_create_split(split_info, mp_comm, split_rowcol, nsplit, own_comm, opt_nsplit)
      !! Split Cartesian process grid using a default split heuristic.

      TYPE(dbcsr_tas_split_info), INTENT(OUT)            :: split_info
         !! object storing all data corresponding to split, submatrices and parallelization
      INTEGER, INTENT(IN)                                :: mp_comm
         !! MPI communicator with associated cartesian grid
      INTEGER, INTENT(IN)                                :: split_rowcol
         !! split rows or columns
      INTEGER, INTENT(IN)                                :: nsplit
         !! desired split factor, set to 0 if split factor of exactly 1 is required
      LOGICAL, INTENT(IN), OPTIONAL                      :: own_comm
         !! whether split_info should own communicator
      LOGICAL, INTENT(IN), OPTIONAL                      :: opt_nsplit

      CHARACTER(LEN=*), PARAMETER                        :: routineN = 'dbcsr_tas_create_split', &
                                                            routineP = moduleN//':'//routineN

      INTEGER                                            :: &
         handle, iproc, numproc, igroup, nsplit_opt, pdim_split, pdim_nonsplit
      INTEGER, DIMENSION(2)                              :: pcoord, pdims, pdims_group
      LOGICAL                                            :: opt_nsplit_prv

      CALL timeset(routineN, handle)

      IF (PRESENT(opt_nsplit)) THEN
         opt_nsplit_prv = opt_nsplit
      ELSE
         opt_nsplit_prv = .TRUE.
      ENDIF

      DBCSR_ASSERT(nsplit > 0)

      CALL mp_environ(numproc, iproc, mp_comm)
      CALL mp_environ(numproc, pdims, pcoord, mp_comm)

      SELECT CASE (split_rowcol)
      CASE (rowsplit)
         pdim_split = pdims(1)
         pdim_nonsplit = pdims(2)
      CASE (colsplit)
         pdim_split = pdims(2)
         pdim_nonsplit = pdims(1)
      END SELECT

      IF (opt_nsplit_prv) THEN
         nsplit_opt = get_opt_nsplit(pdim_split, nsplit, split_pgrid=.TRUE., pdim_nonsplit=pdim_nonsplit)
      ELSE
         IF (MOD(pdims(split_rowcol), nsplit) .NE. 0) THEN
            DBCSR_ABORT("Split factor does not divide process grid dimension")
         ENDIF
         nsplit_opt = nsplit
      ENDIF

      pdims_group = pdims
      pdims_group(split_rowcol) = pdims_group(split_rowcol)/nsplit_opt

      igroup = pcoord(split_rowcol)/pdims_group(split_rowcol)

      CALL dbcsr_tas_create_split_rows_or_cols(split_info, mp_comm, nsplit_opt, igroup, split_rowcol, own_comm=own_comm)
      IF (.NOT. opt_nsplit_prv) split_info%strict_split = .TRUE.

      CALL timestop(handle)

   END SUBROUTINE

   FUNCTION accept_pgrid_dims(dims, relative)
      !! Whether to accept proposed process grid dimensions (based on ratio of dimensions)
      INTEGER, DIMENSION(2), INTENT(IN)          :: dims
      LOGICAL, INTENT(IN)                        :: relative
      INTEGER, DIMENSION(2)                      :: dims_opt
      LOGICAL                                    :: accept_pgrid_dims

      IF (relative) THEN
         dims_opt = 0
         CALL mp_dims_create(PRODUCT(dims), dims_opt)
         accept_pgrid_dims = (MAXVAL(REAL(dims, real_8))/MAXVAL(dims_opt) .LT. default_pdims_accept_ratio)
      ELSE
         accept_pgrid_dims = (MAXVAL(REAL(dims, real_8))/MINVAL(dims) .LT. default_pdims_accept_ratio**2)
      ENDIF
   END FUNCTION

   SUBROUTINE dbcsr_tas_get_split_info(info, mp_comm, nsplit, igroup, mp_comm_group, split_rowcol, pgrid_offset)
      !! Get info on split

      TYPE(dbcsr_tas_split_info), INTENT(IN)             :: info
      INTEGER, INTENT(OUT), OPTIONAL                     :: mp_comm, nsplit, igroup, mp_comm_group, &
                                                            split_rowcol
         !! communicator (global process grid)
         !! split factor
         !! which group do I belong to
         !! subgroup communicator (group-local process grid)
         !! split rows or columns
      INTEGER, DIMENSION(2), INTENT(OUT), OPTIONAL       :: pgrid_offset
         !! group-local offset in process grid

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

   SUBROUTINE dbcsr_tas_release_info(split_info)
      TYPE(dbcsr_tas_split_info), INTENT(INOUT)          :: split_info
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

   SUBROUTINE dbcsr_tas_info_hold(split_info)
      TYPE(dbcsr_tas_split_info), INTENT(IN)             :: split_info

      INTEGER, POINTER                                   :: ref => NULL()

      IF (split_info%refcount < 1) THEN
         DBCSR_ABORT("can not hold non-existing split_info")
      ENDIF
      ref => split_info%refcount
      ref = ref + 1
   END SUBROUTINE

   SUBROUTINE world_to_group_proc_map(iproc, pdims, split_rowcol, pgrid_split_size, igroup, &
                                      pdims_group, iproc_group)
      !! map global process info to group

      INTEGER, INTENT(IN)                                :: iproc
         !! global process ID
      INTEGER, DIMENSION(2), INTENT(IN)                  :: pdims
         !! global process dimensions
      INTEGER, INTENT(IN)                                :: split_rowcol, pgrid_split_size
         !! split rows or column
         !! how many process rows/cols per group
      INTEGER, INTENT(OUT)                               :: igroup
         !! group ID
      INTEGER, DIMENSION(2), INTENT(OUT), OPTIONAL       :: pdims_group
         !! local process grid dimensions
      INTEGER, INTENT(OUT), OPTIONAL                     :: iproc_group
         !! group local process ID

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

   SUBROUTINE group_to_world_proc_map(iproc, pdims, split_rowcol, pgrid_split_size, &
                                      igroup, iproc_group)
      !! map local process info to global info

      INTEGER, INTENT(OUT)                               :: iproc
         !! global process id
      INTEGER, DIMENSION(2), INTENT(IN)                  :: pdims
         !! global process grid dimensions
      INTEGER, INTENT(IN)                                :: split_rowcol, pgrid_split_size, &
                                                            igroup, iproc_group
         !! split rows or column
         !! how many process rows/cols per group
         !! group ID
         !! local process ID

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

   SUBROUTINE block_index_local_to_global(info, dist, row_group, column_group, &
                                          row, column)
      !! map group local block index to global matrix index

      TYPE(dbcsr_tas_split_info), INTENT(IN)             :: info
      TYPE(dbcsr_tas_distribution_type), INTENT(IN)      :: dist
      INTEGER, INTENT(IN), OPTIONAL                      :: row_group, column_group
         !! group local row block index
         !! group local column block index
      INTEGER(KIND=int_8), INTENT(OUT), OPTIONAL         :: row, column
         !! global block row
         !! global block column

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

   SUBROUTINE block_index_global_to_local(info, dist, row, column, row_group, column_group)
      !! map global block index to group local index
      TYPE(dbcsr_tas_split_info), INTENT(IN)               :: info
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

   SUBROUTINE mrowcol_to_group(dist, rowcol, rowcol_group)
      !! map matrix rows/column to group local rows/column

      CLASS(dbcsr_tas_distribution_type), INTENT(IN) :: dist
      INTEGER(KIND=int_8), INTENT(IN)              :: rowcol
         !! global row/column to map
      INTEGER, INTENT(OUT)                         :: rowcol_group
         !! group local row/column
      CHARACTER(LEN=*), PARAMETER                  :: routineN = 'mrowcol_to_group', &
                                                      routineP = moduleN//':'//routineN

      ASSOCIATE (rowcols=>dist%local_rowcols)
         rowcol_group = i8_bsearch(rowcols, rowcol)
      END ASSOCIATE
   END SUBROUTINE

   FUNCTION i8_bsearch(array, el, l_index, u_index) result(res)
      !! binary search for 8-byte integers
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

   SUBROUTINE group_to_mrowcol(info, rowcol_dist, igroup, rowcols)
      !! maps a process subgroup to matrix rows/columns

      TYPE(dbcsr_tas_split_info), INTENT(IN)                      :: info
      CLASS(dbcsr_tas_distribution), INTENT(IN)                   :: rowcol_dist
      INTEGER, INTENT(IN)                                         :: igroup
         !! group ID
      INTEGER(KIND=int_8), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: rowcols
         !! rows/ columns on this group
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
