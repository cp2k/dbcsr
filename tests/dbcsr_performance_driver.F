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
   USE dbcsr_kinds,                     ONLY: default_string_length,&
                                              dp
   USE dbcsr_lib,                       ONLY: dbcsr_finalize_lib,&
                                              dbcsr_init_lib
   USE dbcsr_machine,                   ONLY: default_output_unit
   USE dbcsr_message_passing,           ONLY: &
        add_mp_perf_env, mp_bcast, mp_cart_create, mp_cart_rank, mp_comm_free, mp_environ, &
        mp_world_finalize, mp_world_init, rm_mp_perf_env
   USE dbcsr_mp_methods,                ONLY: dbcsr_mp_new,&
                                              dbcsr_mp_release
   USE dbcsr_para_env,                  ONLY: dbcsr_para_env_create
   USE dbcsr_para_types,                ONLY: dbcsr_para_env_type
   USE dbcsr_performance_multiply,      ONLY: dbcsr_perf_multiply
   USE dbcsr_string_table,              ONLY: string_table_allocate,&
                                              string_table_deallocate
   USE dbcsr_test_methods,              ONLY: dbcsr_test_read_args
   USE dbcsr_timings,                   ONLY: add_timer_env,&
                                              rm_timer_env,&
                                              timings_register_hooks
   USE dbcsr_timings_report,            ONLY: cost_type_time,&
                                              timings_report_callgraph,&
                                              timings_report_print
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

   TYPE(dbcsr_para_env_type), POINTER          :: para_env

   !***************************************************************************************

   !
   ! initialize mpi
   CALL mp_world_init(mp_comm)

   ! DBCSR timers
   NULLIFY (para_env)
   CALL dbcsr_para_env_create(para_env, group=mp_comm, owns_group=.FALSE.)
   CALL dbcsr_error_handling_setup()
   CALL timings_register_hooks()
   CALL string_table_allocate()
   CALL add_mp_perf_env()
   CALL add_timer_env()

   !
   ! initialize libdbcsr errors
   CALL timeset(routineN, handle)

   !
   ! setup the mp environment
   npdims(:) = 0
   CALL mp_cart_create(mp_comm, 2, npdims, myploc, group)
   CALL mp_environ(numnodes, mynode, group)
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
   IF (mynode .EQ. 0) io_unit = default_output_unit

   !
   ! read and distribute input args
   IF (mynode .EQ. 0) CALL dbcsr_test_read_args(narg, args)
   CALL mp_bcast(narg, 0, group)
   CALL mp_bcast(args, 0, group)
   IF (narg .LT. 1) &
      DBCSR_ABORT("nargs not correct")

   !
   ! initialize libdbcsr
   CALL dbcsr_init_lib()

   !
   ! select the operation
   SELECT CASE (args (1))
   CASE ('dbcsr_multiply')
      CALL dbcsr_perf_multiply(group, mp_env, npdims, io_unit, narg, args)
   CASE DEFAULT
      DBCSR_ABORT("operation not found")
   END SELECT

   !
   ! finalize libdbcsr
   CALL dbcsr_finalize_lib(mp_comm, io_unit)

   !
   ! clean mp enviroment
   CALL dbcsr_mp_release(mp_env)

   !
   ! free comm
   CALL mp_comm_free(group)

   !
   ! finalize libdbcsr errors
   CALL timestop(handle)

   ! Print timers
   CALL timings_report_print(io_unit, 0.0_dp, .FALSE., cost_type_time, .TRUE., para_env)

   CALL timings_report_callgraph("test.callgraph")

   ! Remove timers
   CALL rm_timer_env()
   CALL rm_mp_perf_env()
   CALL string_table_deallocate(0)

   !
   ! finalize mpi
   CALL mp_world_finalize()

END PROGRAM dbcsr_performance_driver
