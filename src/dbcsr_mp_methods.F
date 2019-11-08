!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_mp_methods
   USE dbcsr_methods, ONLY: dbcsr_mp_grid_remove, &
                            dbcsr_mp_release
   USE dbcsr_mpiwrap, ONLY: mp_cart_create, &
                            mp_cart_sub, &
                            mp_comm_free,&
                            mp_environ,&
                            mp_cart_rank,&
                            mp_dims_create,&
                            mp_comm_null
   USE dbcsr_types, ONLY: dbcsr_mp_obj

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_mp_methods'

   PUBLIC :: dbcsr_mp_new, dbcsr_mp_hold, dbcsr_mp_release, &
             dbcsr_mp_pgrid, dbcsr_mp_numnodes, dbcsr_mp_mynode, dbcsr_mp_group, &
             dbcsr_mp_new_transposed, dbcsr_mp_nprows, dbcsr_mp_npcols, &
             dbcsr_mp_myprow, dbcsr_mp_mypcol, &
             dbcsr_mp_my_row_group, dbcsr_mp_my_col_group, &
             dbcsr_mp_has_subgroups, dbcsr_mp_get_process, &
             dbcsr_mp_grid_setup, dbcsr_mp_grid_remove, &
             dbcsr_mp_init, dbcsr_mp_active, dbcsr_mp_make_env

   INTERFACE dbcsr_mp_new
      MODULE PROCEDURE dbcsr_mp_new_grid
      MODULE PROCEDURE dbcsr_mp_new_group
   END INTERFACE dbcsr_mp_new

CONTAINS

   SUBROUTINE dbcsr_mp_init(mp_env)
      !! Initializes a new process grid
      TYPE(dbcsr_mp_obj), INTENT(OUT)                    :: mp_env

      NULLIFY (mp_env%mp)
   END SUBROUTINE dbcsr_mp_init

   FUNCTION dbcsr_mp_active(mp_env) RESULT(active)
      !! Checks whether this process is part of the message passing environment
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      LOGICAL                                            :: active

      active = ASSOCIATED(mp_env%mp)
   END FUNCTION dbcsr_mp_active

   SUBROUTINE dbcsr_mp_new_grid(mp_env, mp_group, pgrid, mynode, &
                                numnodes, myprow, mypcol, source)
      !! Creates new process grid

      TYPE(dbcsr_mp_obj), INTENT(OUT)                    :: mp_env
         !! multiprocessor environment
      INTEGER, INTENT(IN)                                :: mp_group, mynode
         !! my processor number
      INTEGER, DIMENSION(0:, 0:), INTENT(IN)             :: pgrid
         !! process grid
      INTEGER, INTENT(IN), OPTIONAL                      :: numnodes, myprow, mypcol, source
         !! total number of processors (processes)

      INTEGER                                            :: pcol, prow

!   ---------------------------------------------------------------------------

      ALLOCATE (mp_env%mp)
      mp_env%mp%refcount = 1
      ALLOCATE (mp_env%mp%pgrid(0:SIZE(pgrid, 1) - 1, 0:SIZE(pgrid, 2) - 1))
      mp_env%mp%pgrid(:, :) = pgrid(:, :)
      mp_env%mp%mynode = mynode
      mp_env%mp%mp_group = mp_group
      mp_env%mp%source = 0
      IF (PRESENT(source)) mp_env%mp%source = source
      IF (PRESENT(numnodes)) THEN
         mp_env%mp%numnodes = numnodes
      ELSE
         mp_env%mp%numnodes = SIZE(pgrid)
      ENDIF
      IF (PRESENT(myprow) .AND. PRESENT(mypcol)) THEN
         mp_env%mp%myprow = myprow
         mp_env%mp%mypcol = mypcol
      ELSE
         mp_env%mp%myprow = -33777
         mp_env%mp%mypcol = -33777
         column_loop: DO pcol = LBOUND(pgrid, 2), UBOUND(pgrid, 2)
            row_loop: DO prow = LBOUND(pgrid, 1), UBOUND(pgrid, 1)
               test_position: IF (pgrid(prow, pcol) .EQ. mynode) THEN
                  mp_env%mp%myprow = prow
                  mp_env%mp%mypcol = pcol
                  EXIT column_loop
               ENDIF test_position
            ENDDO row_loop
         ENDDO column_loop
      ENDIF
      mp_env%mp%subgroups_defined = .FALSE.
      !call dbcsr_mp_grid_setup(mp_env)
   END SUBROUTINE dbcsr_mp_new_grid

   SUBROUTINE dbcsr_mp_new_group(mp_env, mp_group, pgrid)
      !! Creates a new dbcsr_mp_obj based on a input template

      TYPE(dbcsr_mp_obj), INTENT(OUT)                    :: mp_env
      INTEGER, INTENT(IN)                                :: mp_group
      INTEGER, DIMENSION(:, :), OPTIONAL, POINTER        :: pgrid
         !! Optional, if not provided group is assumed to be a 2D cartesian communicator

      INTEGER                                            :: mynode, mypcol, myprow, numnodes, pcol, &
                                                            prow
      INTEGER, DIMENSION(2)                              :: coord, mycoord, pdims
      INTEGER, DIMENSION(:, :), POINTER                  :: mypgrid
      LOGICAL, DIMENSION(2)                              :: periods

      CALL mp_environ(numnodes, mynode, mp_group)

      IF (PRESENT(pgrid)) THEN
         mypgrid => pgrid
         DBCSR_ASSERT(LBOUND(pgrid, 1) == 0 .AND. LBOUND(pgrid, 2) == 0)
         pdims(1) = SIZE(pgrid, 1)
         pdims(2) = SIZE(pgrid, 2)
         myprow = -1; mypcol = -1
         outer: &
            DO prow = 0, pdims(1) - 1
            DO pcol = 0, pdims(2) - 1
               IF (pgrid(prow, pcol) == mynode) THEN
                  myprow = prow
                  mypcol = pcol
                  EXIT outer
               ENDIF
            ENDDO
         ENDDO outer

      ELSE
         CALL mp_environ(mp_group, 2, pdims, mycoord, periods)
         DBCSR_ASSERT(pdims(1)*pdims(2) == numnodes)
         myprow = mycoord(1)
         mypcol = mycoord(2)
         ALLOCATE (mypgrid(0:pdims(1) - 1, 0:pdims(2) - 1))
         DO prow = 0, pdims(1) - 1
            DO pcol = 0, pdims(2) - 1
               coord = (/prow, pcol/)
               CALL mp_cart_rank(mp_group, coord, mypgrid(prow, pcol))
            ENDDO
         ENDDO
      ENDIF

      DBCSR_ASSERT(mynode == mypgrid(myprow, mypcol))

      ! create the new mp environment
      CALL dbcsr_mp_new(mp_env, mp_group, mypgrid, &
                        mynode=mynode, numnodes=numnodes, myprow=myprow, mypcol=mypcol)

      IF (.NOT. PRESENT(pgrid)) DEALLOCATE (mypgrid)

   END SUBROUTINE dbcsr_mp_new_group

   SUBROUTINE dbcsr_mp_grid_setup(mp_env)
      !! Sets up MPI cartesian process grid

      TYPE(dbcsr_mp_obj), INTENT(INOUT)                  :: mp_env
         !! multiprocessor environment

      INTEGER                                            :: ndims, tmp_group
      INTEGER, DIMENSION(2)                              :: dims, my_pos
      LOGICAL, DIMENSION(2)                              :: remain

!   ---------------------------------------------------------------------------

      IF (.NOT. mp_env%mp%subgroups_defined) THEN
         ! KG workaround.
         ! This will be deleted (replaced by code in mp_new).
         ndims = 2
         dims(1:2) = (/SIZE(mp_env%mp%pgrid, 1), SIZE(mp_env%mp%pgrid, 2)/)
         CALL mp_cart_create(mp_env%mp%mp_group, ndims, &
                             dims, my_pos, &
                             tmp_group)
         IF (my_pos(1) .NE. mp_env%mp%myprow) &
            DBCSR_ABORT("Got different MPI process grid")
         IF (my_pos(2) .NE. mp_env%mp%mypcol) &
            DBCSR_ABORT("Got different MPI process grid")
         !
         remain = (/.FALSE., .TRUE./)
         CALL mp_cart_sub(tmp_group, remain, mp_env%mp%prow_group)
         remain = (/.TRUE., .FALSE./)
         CALL mp_cart_sub(tmp_group, remain, mp_env%mp%pcol_group)
         CALL mp_comm_free(tmp_group)
         mp_env%mp%subgroups_defined = .TRUE.
      ENDIF
   END SUBROUTINE dbcsr_mp_grid_setup

   SUBROUTINE dbcsr_mp_make_env(mp_env, cart_group, mp_group, &
                                nprocs, pgrid_dims)
      !! Creates a sane mp_obj from the given MPI comm that is not a cartesian one (hack)

      TYPE(dbcsr_mp_obj), INTENT(OUT)                    :: mp_env
         !! Message-passing environment object to create
      INTEGER, INTENT(OUT)                               :: cart_group
         !! the created cartesian group (to be freed by the user)
      INTEGER, INTENT(IN)                                :: mp_group
         !! MPI group
      INTEGER, INTENT(IN), OPTIONAL                      :: nprocs
         !! Number of processes
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL        :: pgrid_dims
         !! Dimensions of MPI group

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_mp_make_env', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: error_handle, mynode, numnodes, pcol, &
                                                            prow
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: pgrid
      INTEGER, DIMENSION(2)                              :: coord, myploc, npdims
      LOGICAL                                            :: alive

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, error_handle)
      CALL mp_environ(numnodes, mynode, mp_group)
      IF (PRESENT(nprocs)) THEN
         IF (nprocs > numnodes) &
            DBCSR_ABORT("Can not grow processes.")
         numnodes = nprocs
      ENDIF
      !
      IF (PRESENT(pgrid_dims)) THEN
         npdims(:) = pgrid_dims
      ELSE
         npdims(:) = 0
         CALL mp_dims_create(numnodes, npdims)
      ENDIF
      CALL mp_cart_create(mp_group, 2, npdims, myploc, cart_group)
      alive = cart_group .NE. mp_comm_null
      IF (alive) THEN
         CALL mp_environ(numnodes, mynode, cart_group)
         ALLOCATE (pgrid(0:npdims(1) - 1, 0:npdims(2) - 1))
         DO prow = 0, npdims(1) - 1
            DO pcol = 0, npdims(2) - 1
               coord = (/prow, pcol/)
               CALL mp_cart_rank(cart_group, coord, pgrid(prow, pcol))
            ENDDO
         ENDDO
         CALL dbcsr_mp_new(mp_env, cart_group, pgrid, &
                           mynode, numnodes, &
                           myprow=myploc(1), mypcol=myploc(2))
      ELSE
         CALL dbcsr_mp_init(mp_env)
      ENDIF
      CALL timestop(error_handle)
   END SUBROUTINE dbcsr_mp_make_env

   PURE SUBROUTINE dbcsr_mp_hold(mp_env)
      !! Marks another use of the mp_env

      TYPE(dbcsr_mp_obj), INTENT(INOUT)                  :: mp_env
         !! multiprocessor environment

!   ---------------------------------------------------------------------------

      mp_env%mp%refcount = mp_env%mp%refcount + 1
   END SUBROUTINE dbcsr_mp_hold

   PURE FUNCTION dbcsr_mp_get_process(mp_env, prow, pcol) RESULT(process)
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      INTEGER, INTENT(IN)                                :: prow, pcol
      INTEGER                                            :: process

      process = mp_env%mp%pgrid(prow, pcol)
   END FUNCTION dbcsr_mp_get_process

   FUNCTION dbcsr_mp_pgrid(mp_env) RESULT(pgrid)
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      INTEGER, DIMENSION(:, :), POINTER                  :: pgrid

      pgrid => mp_env%mp%pgrid
   END FUNCTION dbcsr_mp_pgrid
   PURE FUNCTION dbcsr_mp_numnodes(mp_env) RESULT(numnodes)
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      INTEGER                                            :: numnodes

      numnodes = mp_env%mp%numnodes
   END FUNCTION dbcsr_mp_numnodes
   PURE FUNCTION dbcsr_mp_mynode(mp_env) RESULT(mynode)
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      INTEGER                                            :: mynode

      mynode = mp_env%mp%mynode
   END FUNCTION dbcsr_mp_mynode
   PURE FUNCTION dbcsr_mp_group(mp_env) RESULT(mp_group)
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      INTEGER                                            :: mp_group

      mp_group = mp_env%mp%mp_group
   END FUNCTION dbcsr_mp_group
   PURE FUNCTION dbcsr_mp_nprows(mp_env) RESULT(nprows)
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      INTEGER                                            :: nprows

      nprows = SIZE(mp_env%mp%pgrid, 1)
   END FUNCTION dbcsr_mp_nprows
   PURE FUNCTION dbcsr_mp_npcols(mp_env) RESULT(npcols)
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      INTEGER                                            :: npcols

      npcols = SIZE(mp_env%mp%pgrid, 2)
   END FUNCTION dbcsr_mp_npcols
   PURE FUNCTION dbcsr_mp_myprow(mp_env) RESULT(myprow)
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      INTEGER                                            :: myprow

      myprow = mp_env%mp%myprow
   END FUNCTION dbcsr_mp_myprow
   PURE FUNCTION dbcsr_mp_mypcol(mp_env) RESULT(mypcol)
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      INTEGER                                            :: mypcol

      mypcol = mp_env%mp%mypcol
   END FUNCTION dbcsr_mp_mypcol
   PURE FUNCTION dbcsr_mp_has_subgroups(mp_env) RESULT(has_subgroups)
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      LOGICAL                                            :: has_subgroups

      has_subgroups = mp_env%mp%subgroups_defined
   END FUNCTION dbcsr_mp_has_subgroups
   PURE FUNCTION dbcsr_mp_my_row_group(mp_env) RESULT(row_group)
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      INTEGER                                            :: row_group

      row_group = mp_env%mp%prow_group
   END FUNCTION dbcsr_mp_my_row_group
   PURE FUNCTION dbcsr_mp_my_col_group(mp_env) RESULT(col_group)
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      INTEGER                                            :: col_group

      col_group = mp_env%mp%pcol_group
   END FUNCTION dbcsr_mp_my_col_group

   SUBROUTINE dbcsr_mp_new_transposed(mp_t, mp)
      !! Transposes a multiprocessor environment

      TYPE(dbcsr_mp_obj), INTENT(OUT)                    :: mp_t
         !! transposed multiprocessor environment
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp
         !! original multiprocessor environment

!   ---------------------------------------------------------------------------

      CALL dbcsr_mp_new(mp_t, dbcsr_mp_group(mp), &
                        TRANSPOSE(dbcsr_mp_pgrid(mp)), &
                        dbcsr_mp_mynode(mp), dbcsr_mp_numnodes(mp), &
                        dbcsr_mp_mypcol(mp), dbcsr_mp_myprow(mp))
   END SUBROUTINE dbcsr_mp_new_transposed

END MODULE dbcsr_mp_methods
