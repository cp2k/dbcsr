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
                            rm_mp_perf_env
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
   
   use dbcsr_types, only: dbcsr_mp_obj

   use dbcsr_mp_methods, only: dbcsr_mp_new,dbcsr_mp_release

   use dbcsr_error_handling, only: dbcsr_error_handling_setup

#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_lib'

   PUBLIC :: dbcsr_init_lib, dbcsr_finalize_lib, dbcsr_clear_mempools
   PUBLIC :: dbcsr_init_timer,dbcsr_kill_timer, dbcsr_print_timer


   LOGICAL, PRIVATE, SAVE :: is_initialized = .FALSE.
   
   TYPE(dbcsr_logger_type), POINTER         :: logger
   TYPE(dbcsr_mp_obj)   , save                    :: mp_env


CONTAINS

! **************************************************************************************************
!> \brief Initialize timers
!>
!> Prepares the DBCSR library for use.
! **************************************************************************************************
   subroutine dbcsr_init_timer(mp_env)
      TYPE(dbcsr_mp_obj) , intent(in) :: mp_env
      integer :: io_unit

      io_unit = 0
      IF (mp_env % mp % mynode .EQ. mp_env % mp % source) io_unit = default_output_unit
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
   end subroutine

   
! **************************************************************************************************
!> \brief Kill timers
!>
!> Prepares the DBCSR library for use.
! **************************************************************************************************
   subroutine dbcsr_kill_timer()
      ! clean mp environment
      CALL dbcsr_mp_release(mp_env)

      !
      ! Remove logger
      CALL dbcsr_rm_default_logger()

      !
      !
      ! Remove timers
      CALL rm_mp_perf_env()
      CALL rm_timer_env()
   end subroutine

   
! **************************************************************************************************
!> \brief Print timers
!>
!> Prepares the DBCSR library for use.
! **************************************************************************************************
   subroutine dbcsr_print_timer()
      integer :: io_unit

      io_unit = 0
      IF (mp_env % mp % mynode .EQ. mp_env % mp % source) io_unit = default_output_unit

      CALL timings_report_print(io_unit, 0.0_dp, .FALSE., cost_type_time, .TRUE., mp_env)
   end subroutine

   
! **************************************************************************************************
!> \brief Initialize the DBCSR library
!>
!> Prepares the DBCSR library for use.
! **************************************************************************************************
   SUBROUTINE dbcsr_init_lib(mp_comm)
      INTEGER, INTENT(IN)  :: mp_comm

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_init_lib', routineP = moduleN//':'//routineN

      INTEGER                                            :: error_handle, ithread, nthreads
      integer :: comm
      
      integer, dimension(2),             :: npdims
      integer, dimension(2)              :: myploc
      INTEGER, DIMENSION(:, :), POINTER  :: pgrid      
      integer :: io_unit, group, numnodes, pcol, prow, mynode

      if (present(mp_comm)) then
          comm = mp_comm
      else
          comm = MPI_COMM_WORLD
      endif
      
      npdims(1) = numnodes
      npdims(2) = 1
      
      CALL mp_environ(numnodes, mynode, mp_comm)
      CALL mp_cart_create(mp_comm, 2, npdims, myploc, group)

      ALLOCATE (pgrid(0:npdims(1) - 1, 0:npdims(2) - 1))
      DO prow = 0, npdims(1) - 1
         DO pcol = 0, npdims(2) - 1
            CALL mp_cart_rank(group, (/prow, pcol/), pgrid(prow, pcol))
         ENDDO
      ENDDO

      CALL dbcsr_mp_new(mp_env, pgrid, group, mynode, numnodes, &
                        myploc(1), myploc(2))

      DEALLOCATE (pgrid)

      call  dbcsr_init_timer(mp_env)

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
   END SUBROUTINE dbcsr_init_lib

! **************************************************************************************************
!> \brief Finalize the DBCSR library
!>
!> Cleans up after the DBCSR library.  Used to deallocate persistent objects.
!> \param group ...
!> \param output_unit ...
! **************************************************************************************************
   SUBROUTINE dbcsr_finalize_lib(group, output_unit)
      INTEGER, INTENT(IN)                                :: group
      INTEGER, INTENT(IN), OPTIONAL                      :: output_unit

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_finalize_lib', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: error_handle, io_unit, ithread, mynode

      IF (.NOT. is_initialized) RETURN
      CALL timeset(routineN, error_handle)

      io_unit = 0
      IF (PRESENT(output_unit)) THEN
         io_unit = output_unit
      ELSE
         CALL mp_environ(taskid=mynode, groupid=group)
         IF (mynode .EQ. 0) io_unit = default_output_unit
      ENDIF

      IF (io_unit > 0) THEN
         WRITE (UNIT=io_unit, FMT="(/,T2,A)") REPEAT("-", 79)
         WRITE (UNIT=io_unit, FMT="(T2,A,T80,A)") "-", "-"
         WRITE (UNIT=io_unit, FMT="(T2,A,T35,A,T80,A)") "-", "DBCSR STATISTICS", "-"
         WRITE (UNIT=io_unit, FMT="(T2,A,T80,A)") "-", "-"
         WRITE (UNIT=io_unit, FMT="(T2,A)") REPEAT("-", 79)
      END IF

!$OMP     PARALLEL DEFAULT(NONE) PRIVATE(ithread) SHARED(io_unit, group, cublas_handles)
      CALL dbcsr_multiply_lib_finalize(group, io_unit)
      ithread = 0
!$    ithread = omp_get_thread_num()
      CALL cublas_handle_destroy(cublas_handles(ithread + 1))
!$OMP BARRIER
!$OMP MASTER
      DEALLOCATE (cublas_handles)
!$OMP END MASTER
!$OMP     END PARALLEL
      IF (io_unit > 0) WRITE (UNIT=io_unit, FMT="(T2,A)") REPEAT("-", 79)

      CALL describe_mp_perf_env(io_unit)

      is_initialized = .FALSE.

      CALL timestop(error_handle)

      call dbcsr_kill_timer()
   END SUBROUTINE dbcsr_finalize_lib

! **************************************************************************************************
!> \brief  Deallocate memory contained in mempools
! **************************************************************************************************
   SUBROUTINE dbcsr_clear_mempools()

!$OMP     PARALLEL DEFAULT(NONE)
      CALL dbcsr_multiply_clear_mempools()
!$OMP     END PARALLEL
   END SUBROUTINE dbcsr_clear_mempools

END MODULE dbcsr_lib
