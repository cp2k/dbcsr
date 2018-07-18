!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright (C) 2000 - 2018  CP2K developers group                                               !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief   Performance tester for DBCSR operations
!> \author  VW
!> \date    2010
!> \version 1.0
!>
!> <b>Modification history:</b>
!> - Created 2010
! **************************************************************************************************
PROGRAM dbcsr_performance_driver
   USE dbcsr_error_handling,            ONLY: dbcsr_error_handling_setup
   USE dbcsr_files,                     ONLY: open_file
   USE dbcsr_kinds,                     ONLY: default_string_length,&
                                              dp
   USE dbcsr_lib,                       ONLY: dbcsr_finalize_lib,&
                                              dbcsr_init_lib
   USE dbcsr_log_handling,              ONLY: dbcsr_add_default_logger,&
                                              dbcsr_logger_create,&
                                              dbcsr_logger_release,&
                                              dbcsr_logger_type,&
                                              dbcsr_rm_default_logger
   USE dbcsr_machine,                   ONLY: default_output_unit,&
                                              m_getarg,&
                                              m_iargc
   USE dbcsr_mp_methods,                ONLY: dbcsr_mp_new,&
                                              dbcsr_mp_release
   USE dbcsr_mpiwrap,                   ONLY: &
        add_mp_perf_env, mp_bcast, mp_cart_create, mp_cart_rank, mp_comm_free, mp_environ, &
        mp_world_finalize, mp_world_init, rm_mp_perf_env
   USE dbcsr_performance_multiply,      ONLY: dbcsr_perf_multiply
   USE dbcsr_timings,                   ONLY: add_timer_env,&
                                              rm_timer_env,&
                                              timings_register_hooks
   USE dbcsr_timings_report,            ONLY: cost_type_time,&
                                              timings_report_callgraph,&
                                              timings_report_print
   USE dbcsr_toollib,                   ONLY: atoi
   USE dbcsr_types,                     ONLY: dbcsr_mp_obj
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE

   INTEGER                                  :: mp_comm, group, numnodes, mynode, &
                                               prow, pcol, io_unit, narg, handle
   INTEGER, DIMENSION(2)                    :: npdims, myploc
   INTEGER, DIMENSION(:, :), POINTER        :: pgrid
   TYPE(dbcsr_mp_obj)                       :: mp_env
   CHARACTER(len=default_string_length)     :: args(100)

   CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_check_multiply'

   TYPE(dbcsr_logger_type), POINTER            :: logger

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
   ALLOCATE (pgrid(0:npdims(1)-1, 0:npdims(2)-1))
   DO prow = 0, npdims(1)-1
      DO pcol = 0, npdims(2)-1
         CALL mp_cart_rank(group, (/prow, pcol/), pgrid(prow, pcol))
      ENDDO
   ENDDO
   CALL dbcsr_mp_new(mp_env, pgrid, group, mynode, numnodes, &
                     myprow=myploc(1), mypcol=myploc(2))
   DEALLOCATE (pgrid)

   !
   ! set standard output parameters
   io_unit = 0
   IF (mynode .EQ. mp_env%mp%source) io_unit = default_output_unit

   !
   ! Timers
   NULLIFY (logger)
   CALL dbcsr_logger_create(logger, mp_env=mp_env, &
                            default_global_unit_nr=io_unit, &
                            close_global_unit_on_dealloc=.FALSE.)
   CALL dbcsr_add_default_logger(logger)
   CALL dbcsr_logger_release(logger)
   CALL dbcsr_error_handling_setup()
   CALL timings_register_hooks()
   CALL add_mp_perf_env()
   CALL add_timer_env()

   !
   ! initialize libdbcsr errors
   CALL timeset(routineN, handle)

   !
   ! initialize libdbcsr
   CALL dbcsr_init_lib()

   !
   ! select the operation
   SELECT CASE (args (2))
   CASE ('dbcsr_multiply')
      CALL dbcsr_perf_multiply(group, mp_env, npdims, io_unit, narg, args)
   CASE DEFAULT
      DBCSR_ABORT("operation not found")
   END SELECT

   !
   ! finalize libdbcsr
   CALL dbcsr_finalize_lib(mp_comm, io_unit)

   !
   ! finalize libdbcsr errors
   CALL timestop(handle)

   ! Print timers
   CALL timings_report_print(io_unit, 0.0_dp, .FALSE., cost_type_time, .TRUE., mp_env)
   ! Dump callgraph
   CALL timings_report_callgraph("test.callgraph")

   !
   ! clean mp enviroment
   CALL dbcsr_mp_release(mp_env)

   !
   ! Remove logger
   CALL dbcsr_rm_default_logger()

   !
   ! free comm
   CALL mp_comm_free(group)

   !
   ! Remove timers
   CALL rm_mp_perf_env()
   CALL rm_timer_env()

   !
   ! finalize mpi
   CALL mp_world_finalize()

CONTAINS

   ! **************************************************************************************************
   !> \brief ...
   !> \param narg ...
   !> \param args ...
   ! **************************************************************************************************
! **************************************************************************************************
!> \brief ...
!> \param narg ...
!> \param args ...
! **************************************************************************************************
   SUBROUTINE dbcsr_test_read_args(narg, args)
      INTEGER, INTENT(out)                               :: narg
      CHARACTER(len=*), DIMENSION(:), INTENT(out)        :: args

      CHARACTER(len=default_string_length)               :: line
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
         narg = narg+1
         args(narg) = line
      ENDDO

   END SUBROUTINE dbcsr_test_read_args

END PROGRAM dbcsr_performance_driver
