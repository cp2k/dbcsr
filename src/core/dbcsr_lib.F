!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief   Routines that affect the DBCSR library as a whole
!> \author  Ole Schuett
!> \date    10-2014
! **************************************************************************************************

MODULE dbcsr_lib
   USE dbcsr_config, ONLY: cublas_handles
   USE dbcsr_cublas, ONLY: cublas_handle_destroy, &
                           cublas_handle_init
   USE dbcsr_kinds, ONLY: int_1_size, &
                          int_2_size, &
                          int_4_size, &
                          int_8_size, dp
   USE dbcsr_machine, ONLY: default_output_unit
   USE dbcsr_mpiwrap, ONLY: add_mp_perf_env, &
                            describe_mp_perf_env, &
                            has_mp_perf_env, &
                            mp_environ, mp_cart_rank,&
                            rm_mp_perf_env,&
                            mp_comm_free
   USE dbcsr_multiply_api, ONLY: dbcsr_multiply_clear_mempools, &
                                 dbcsr_multiply_lib_finalize, &
                                 dbcsr_multiply_lib_init
   USE dbcsr_timings, ONLY: add_timer_env, &
                            rm_timer_env, &
                            timings_register_hooks
   USE dbcsr_timings_report, ONLY: cost_type_time, &
                                   timings_report_callgraph, &
                                   timings_report_print
   USE dbcsr_log_handling, ONLY: dbcsr_add_default_logger, &
                                 dbcsr_logger_create, &
                                 dbcsr_logger_release, &
                                 dbcsr_logger_type, &
                                 dbcsr_rm_default_logger
   USE dbcsr_base_hooks, ONLY: timeset_hook, &
                               timestop_hook,&
                               dbcsr_abort_hook,&
                               dbcsr_warn_hook,&
                               dbcsr_abort_interface, dbcsr_warn_interface, &
                               timeset_interface, timestop_interface
   use dbcsr_types, only: dbcsr_mp_obj
   use dbcsr_mp_methods, only: dbcsr_mp_new, dbcsr_mp_release, &
                               dbcsr_mp_make_env
   use dbcsr_error_handling, only: dbcsr_error_handling_setup

#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_lib'

   PUBLIC :: dbcsr_init_lib, dbcsr_finalize_lib, dbcsr_clear_mempools

   LOGICAL, PRIVATE, SAVE :: is_initialized = .FALSE.

   TYPE(dbcsr_logger_type), POINTER :: logger => Null()
   TYPE(dbcsr_mp_obj), SAVE         :: mp_env
   INTEGER, SAVE                    :: default_group, ext_io_unit

   INTERFACE dbcsr_init_lib
      MODULE PROCEDURE dbcsr_init_lib_def
      MODULE PROCEDURE dbcsr_init_lib_hooks
   END INTERFACE

CONTAINS

! **************************************************************************************************
!> \brief Initialize the DBCSR library using internal loggers and timer callbacks
!> \param mp_comm ...
!> \param io_unit ...
! **************************************************************************************************
   SUBROUTINE dbcsr_init_lib_def(mp_comm, io_unit)
      INTEGER, INTENT(IN)  :: mp_comm
      INTEGER, INTENT(IN), OPTIONAL :: io_unit

      CALL dbcsr_init_lib_pre(mp_comm, io_unit)
      !
      ! Declare loggers
      CALL dbcsr_logger_create(logger, mp_env=mp_env, &
                               default_global_unit_nr=ext_io_unit, &
                               close_global_unit_on_dealloc=.FALSE.)
      CALL dbcsr_add_default_logger(logger)
      CALL dbcsr_logger_release(logger)
      ! abort/warn hooks
      CALL dbcsr_error_handling_setup()
      ! timeset/timestop hooks
      CALL timings_register_hooks()
      ! timer environment
      CALL add_mp_perf_env()
      CALL add_timer_env()
      !
      CALL dbcsr_init_lib_low()
   END SUBROUTINE dbcsr_init_lib_def

! **************************************************************************************************
!> \brief Initialize the DBCSR library using external loggers and timer callbacks
!> \param mp_comm ...
!> \param in_timeset_hook ...
!> \param in_timestop_hook ...
!> \param in_abort_hook ...
!> \param in_warn_hook ...
!> \param io_unit ...
! **************************************************************************************************
   SUBROUTINE dbcsr_init_lib_hooks(mp_comm, &
                                   in_timeset_hook, in_timestop_hook, &
                                   in_abort_hook, in_warn_hook, io_unit)
      INTEGER, INTENT(IN)  :: mp_comm
      PROCEDURE(timeset_interface), INTENT(IN), POINTER :: in_timeset_hook
      PROCEDURE(timestop_interface), INTENT(IN), POINTER :: in_timestop_hook
      PROCEDURE(dbcsr_abort_interface), INTENT(IN), POINTER :: in_abort_hook
      PROCEDURE(dbcsr_warn_interface), INTENT(IN), POINTER :: in_warn_hook
      INTEGER, INTENT(IN), OPTIONAL :: io_unit

      CALL dbcsr_init_lib_pre(mp_comm, io_unit)
      ! abort/warn hooks
      dbcsr_abort_hook => in_abort_hook
      dbcsr_warn_hook => in_warn_hook
      ! timeset/timestop hooks
      timeset_hook => in_timeset_hook
      timestop_hook => in_timestop_hook
      ! timer environment is assumed
      !
      CALL dbcsr_init_lib_low()
   END SUBROUTINE dbcsr_init_lib_hooks

! **************************************************************************************************
!> \brief Initialize the DBCSR library
!>
!> Prepares the DBCSR library for use.
!> \param mp_comm ...
!> \param io_unit ...
! **************************************************************************************************
   SUBROUTINE dbcsr_init_lib_pre(mp_comm, io_unit)
#if defined(__LIBXSMM)
      USE libxsmm, ONLY: libxsmm_init
#endif
      INTEGER, INTENT(IN)  :: mp_comm
      INTEGER, INTENT(IN), OPTIONAL :: io_unit

      INTEGER :: numnodes, mynode

      IF (PRESENT(io_unit)) THEN
         ext_io_unit = io_unit
      ELSE
         ext_io_unit = 0
         CALL mp_environ(numnodes, mynode, mp_comm)
         IF (mynode .EQ. 0) ext_io_unit = default_output_unit
      ENDIF
      CALL dbcsr_mp_make_env(mp_env, default_group, mp_comm)
#if defined(__LIBXSMM)
      CALL libxsmm_init()
#endif
   END SUBROUTINE dbcsr_init_lib_pre

! **************************************************************************************************
!> \brief Initialize the DBCSR library
!>
!> Prepares the DBCSR library for use.
! **************************************************************************************************
   SUBROUTINE dbcsr_init_lib_low()

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_init_low', routineP = moduleN//':'//routineN

      INTEGER                                            :: error_handle, ithread, nthreads

      IF (is_initialized) RETURN
      CALL timeset(routineN, error_handle)
      !
      IF (int_1_size /= 1) &
         DBCSR_ABORT("Incorrect assumption of an 8-bit integer size!")
      IF (int_2_size /= 2) &
         DBCSR_ABORT("Incorrect assumption of a 16-bit integer size!")
      IF (int_4_size /= 4) &
         DBCSR_ABORT("Incorrect assumption of a 32-bit integer size!")
      IF (int_8_size /= 8) &
         DBCSR_ABORT("Incorrect assumption of a 64-bit integer size!")

      IF (.NOT. has_mp_perf_env()) THEN
         CALL add_mp_perf_env()
      ENDIF

!$OMP     PARALLEL DEFAULT(NONE) PRIVATE(ithread) SHARED(nthreads, cublas_handles)
      CALL dbcsr_multiply_lib_init()
!$OMP MASTER
      nthreads = 1
!$    nthreads = omp_get_num_threads()
      ALLOCATE (cublas_handles(nthreads))
!$OMP END MASTER
!$OMP BARRIER
      ithread = 0
!$    ithread = omp_get_thread_num()
      ! init cublas. Internal check for cublas existing
      CALL cublas_handle_init(cublas_handles(ithread + 1))
!$OMP     END PARALLEL

      is_initialized = .TRUE.

      CALL timestop(error_handle)
   END SUBROUTINE dbcsr_init_lib_low

! **************************************************************************************************
!> \brief Finalize the DBCSR library
!>
!> Cleans up after the DBCSR library.  Used to deallocate persistent objects.
! **************************************************************************************************
   SUBROUTINE dbcsr_finalize_lib()
#if defined(__LIBXSMM)
      USE libxsmm, ONLY: libxsmm_finalize
#endif
      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_finalize_lib', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: error_handle, ithread

      IF (.NOT. is_initialized) RETURN
      CALL timeset(routineN, error_handle)

      IF (ext_io_unit > 0) THEN
         WRITE (UNIT=ext_io_unit, FMT="(/,T2,A)") REPEAT("-", 79)
         WRITE (UNIT=ext_io_unit, FMT="(T2,A,T80,A)") "-", "-"
         WRITE (UNIT=ext_io_unit, FMT="(T2,A,T35,A,T80,A)") "-", "DBCSR STATISTICS", "-"
         WRITE (UNIT=ext_io_unit, FMT="(T2,A,T80,A)") "-", "-"
         WRITE (UNIT=ext_io_unit, FMT="(T2,A)") REPEAT("-", 79)
      END IF

!$OMP     PARALLEL DEFAULT(NONE) PRIVATE(ithread) SHARED(ext_io_unit, default_group, cublas_handles)
      CALL dbcsr_multiply_lib_finalize(default_group, ext_io_unit)
      ithread = 0
!$    ithread = omp_get_thread_num()
      CALL cublas_handle_destroy(cublas_handles(ithread + 1))
!$OMP BARRIER
!$OMP MASTER
      DEALLOCATE (cublas_handles)
!$OMP END MASTER
!$OMP     END PARALLEL
      IF (ext_io_unit > 0) WRITE (UNIT=ext_io_unit, FMT="(T2,A)") REPEAT("-", 79)

      is_initialized = .FALSE.

      IF (ASSOCIATED(logger)) THEN
         CALL dbcsr_print_timers()
         CALL timestop(error_handle)
         CALL dbcsr_rm_default_logger()
         CALL rm_mp_perf_env()
         CALL rm_timer_env()
         NULLIFY(logger)
      ELSE
         CALL timestop(error_handle)
      ENDIF
      CALL dbcsr_mp_release(mp_env)
      CALL mp_comm_free(default_group)
#if defined(__LIBXSMM)
      CALL libxsmm_finalize()
#endif
      NULLIFY(timeset_hook)
      NULLIFY(timestop_hook)
      NULLIFY(dbcsr_abort_hook)
      NULLIFY(dbcsr_warn_hook)
   END SUBROUTINE dbcsr_finalize_lib

! **************************************************************************************************
!> \brief Print timers
!>
!> Prepares the DBCSR library for use.
! **************************************************************************************************
   SUBROUTINE dbcsr_print_timers()
      CALL describe_mp_perf_env(ext_io_unit)
      CALL timings_report_print(ext_io_unit, 0.0_dp, .FALSE., cost_type_time, .TRUE., mp_env)
      ! Dump callgraph
      ! CALL timings_report_callgraph("test.callgraph")
   END SUBROUTINE dbcsr_print_timers

! **************************************************************************************************
!> \brief  Deallocate memory contained in mempools
! **************************************************************************************************
   SUBROUTINE dbcsr_clear_mempools()

!$OMP     PARALLEL DEFAULT(NONE)
      CALL dbcsr_multiply_clear_mempools()
!$OMP     END PARALLEL
   END SUBROUTINE dbcsr_clear_mempools

END MODULE dbcsr_lib
