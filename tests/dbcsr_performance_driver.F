!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

PROGRAM dbcsr_performance_driver
   !! Performance tester for DBCSR operations
   USE dbcsr_config, ONLY: dbcsr_set_config, dbcsr_print_config
   USE dbcsr_files, ONLY: open_file
   USE dbcsr_kinds, ONLY: default_string_length
   USE dbcsr_lib, ONLY: dbcsr_finalize_lib, &
                        dbcsr_init_lib,&
                        dbcsr_print_statistics
   USE dbcsr_machine, ONLY: default_output_unit, &
                            m_getarg, &
                            m_iargc
   USE dbcsr_mp_methods, ONLY: dbcsr_mp_new, &
                               dbcsr_mp_release
   USE dbcsr_mpiwrap, ONLY: &
      mp_bcast, mp_cart_create, mp_cart_rank, mp_comm_free, mp_environ, &
      mp_world_finalize, mp_world_init
   USE dbcsr_performance_multiply, ONLY: dbcsr_perf_multiply
   USE dbcsr_toollib, ONLY: atoi, atol
   USE dbcsr_types, ONLY: dbcsr_mp_obj
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE

   INTEGER                                  :: mp_comm, group, numnodes, mynode, &
                                               prow, pcol, io_unit, narg, handle
   INTEGER, DIMENSION(2)                    :: npdims, myploc
   INTEGER, DIMENSION(:, :), POINTER        :: pgrid
   TYPE(dbcsr_mp_obj)                       :: mp_env
   CHARACTER(len=default_string_length)     :: args(100)

   CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_performance_driver'

   !***************************************************************************************

   !
   ! initialize mpi
   CALL mp_world_init(mp_comm)

   !
   ! Number of nodes and rankid
   CALL mp_environ(numnodes, mynode, mp_comm)

   !
   ! read and distribute input args
   IF (mynode .EQ. 0) CALL dbcsr_test_read_args(narg, args)
   CALL mp_bcast(narg, 0, mp_comm)
   CALL mp_bcast(args, 0, mp_comm)
   IF (narg .LT. 1) &
      DBCSR_ABORT("nargs not correct")

   !
   ! setup the mp environment
   IF (atoi(args(1)) .LE. 0) THEN
      npdims(:) = 0
   ELSE
      npdims(2) = atoi(args(1))
      IF (MOD(numnodes, npdims(2)) .NE. 0) THEN
         DBCSR_ABORT("numnodes is not multiple of npcols")
      ENDIF
      npdims(1) = numnodes/npdims(2)
   ENDIF
   CALL mp_cart_create(mp_comm, 2, npdims, myploc, group)
   ALLOCATE (pgrid(0:npdims(1) - 1, 0:npdims(2) - 1))
   DO prow = 0, npdims(1) - 1
      DO pcol = 0, npdims(2) - 1
         CALL mp_cart_rank(group, (/prow, pcol/), pgrid(prow, pcol))
      ENDDO
   ENDDO
   CALL dbcsr_mp_new(mp_env, group, pgrid, mynode, numnodes, &
                     myprow=myploc(1), mypcol=myploc(2))
   DEALLOCATE (pgrid)

   !
   ! set standard output parameters
   io_unit = 0
   IF (mynode .EQ. mp_env%mp%source) io_unit = default_output_unit

   !
   ! initialize libdbcsr
   CALL dbcsr_init_lib(mp_comm,io_unit)

   ! initialize libdbcsr errors
   CALL timeset(routineN, handle)

   ! Check for MPI-RMA algorithm
   CALL dbcsr_set_config(use_mpi_rma=atol(args(2)))

   !
   ! print DBCSR configuration
   CALL dbcsr_print_config(io_unit)

   !
   ! select the operation
   SELECT CASE (args(3))
   CASE ('dbcsr_multiply')
      CALL dbcsr_perf_multiply(group, mp_env, npdims, io_unit, narg, 3, args)
   CASE DEFAULT
      DBCSR_ABORT("operation not found")
   END SELECT

   !
   ! finalize libdbcsr errors
   CALL timestop(handle)

   !
   ! clean mp environment
   CALL dbcsr_mp_release(mp_env)

   !
   ! free comm
   CALL mp_comm_free(group)

   !
   ! print statistics
   CALL dbcsr_print_statistics(.true., "test.callgraph")

   !
   ! finalize DBCSR
   CALL dbcsr_finalize_lib()
   !
   !
   ! finalize mpi
   CALL mp_world_finalize()

CONTAINS

   SUBROUTINE dbcsr_test_read_args(narg, args)
      INTEGER, INTENT(out)                               :: narg
      CHARACTER(len=*), DIMENSION(:), INTENT(out)        :: args

      CHARACTER(len=1000)                                :: line
      INTEGER                                            :: istat, unit

      ! Read for standard input
      unit = 5
      !
      ! Read from a file
      IF (m_iargc() .GT. 0) THEN
         CALL m_getarg(1, line)
         CALL open_file(TRIM(line), unit_number=unit)
      ENDIF

      narg = 0
      DO
         READ (unit, *, IOSTAT=istat) line
         IF (istat .NE. 0) EXIT
         IF (line(1:1) .EQ. '#') CYCLE
         narg = narg + 1
         args(narg) = line
      ENDDO

   END SUBROUTINE dbcsr_test_read_args

END PROGRAM dbcsr_performance_driver
