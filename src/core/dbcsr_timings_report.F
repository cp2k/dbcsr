!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_timings_report
   !! Timing routines for accounting
   USE dbcsr_dict, ONLY: dict_get, &
                         dict_haskey, &
                         dict_i4tuple_callstat_item_type, &
                         dict_items
   USE dbcsr_files, ONLY: close_file, &
                          open_file
   USE dbcsr_kinds, ONLY: default_string_length, &
                          dp, &
                          int_8
   USE dbcsr_list, ONLY: list_destroy, &
                         list_get, &
                         list_init, &
                         list_isready, &
                         list_pop, &
                         list_push, &
                         list_size
   USE dbcsr_list_routinereport, ONLY: list_routinereport_type
   USE dbcsr_mpiwrap, ONLY: mp_bcast, &
                            mp_max, &
                            mp_maxloc, &
                            mp_sum
   USE dbcsr_timings, ONLY: get_timer_env
   USE dbcsr_timings_base_type, ONLY: call_stat_type, &
                                      routine_report_type, &
                                      routine_stat_type
   USE dbcsr_timings_types, ONLY: timer_env_type
   USE dbcsr_toollib, ONLY: sort
   USE dbcsr_types, ONLY: dbcsr_mp_obj
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   INTEGER, PUBLIC, PARAMETER :: cost_type_time = 17, cost_type_energy = 18

   PUBLIC :: timings_report_print, timings_report_callgraph

CONTAINS

   SUBROUTINE timings_report_print(iw, r_timings, sort_by_self_time, cost_type, report_maxloc, mp_env)
      !! Print accumulated information on timers

      INTEGER, INTENT(IN)                                :: iw
      REAL(KIND=dp), INTENT(IN)                          :: r_timings
      LOGICAL, INTENT(IN)                                :: sort_by_self_time
      INTEGER, INTENT(IN)                                :: cost_type
      LOGICAL, INTENT(IN)                                :: report_maxloc
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
         !! is needed to collect statistics from other nodes.

      TYPE(list_routinereport_type)                      :: reports
      TYPE(routine_report_type), POINTER                 :: r_report

      CALL list_init(reports)
      CALL collect_reports_from_ranks(reports, cost_type, mp_env)

      IF (list_size(reports) > 0 .AND. iw > 0) &
         CALL print_reports(reports, iw, r_timings, sort_by_self_time, cost_type, report_maxloc, mp_env)

      ! deallocate reports
      DO WHILE (list_size(reports) > 0)
         r_report => list_pop(reports)
         DEALLOCATE (r_report)
      END DO
      CALL list_destroy(reports)

   END SUBROUTINE timings_report_print

   SUBROUTINE collect_reports_from_ranks(reports, cost_type, mp_env)
      !! Collects the timing or energy reports from all MPI ranks.
      TYPE(list_routinereport_type), INTENT(INOUT)       :: reports
      INTEGER, INTENT(IN)                                :: cost_type
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env

      CHARACTER(LEN=default_string_length)               :: routineN
      INTEGER                                            :: local_routine_id, sending_rank
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: collected
      REAL(KIND=dp)                                      :: foobar
      REAL(KIND=dp), DIMENSION(2)                        :: dbuf
      TYPE(routine_report_type), POINTER                 :: r_report
      TYPE(routine_stat_type), POINTER                   :: r_stat
      TYPE(timer_env_type), POINTER                      :: timer_env

      NULLIFY (r_stat, r_report, timer_env)
      IF (.NOT. list_isready(reports)) &
         DBCSR_ABORT("BUG")

      timer_env => get_timer_env()

      ! make sure all functions have been called so that list_size(timer_env%routine_stats)
      ! and the actual dictionary are consistent in the loop below, preventing out of bounds.
      ! this hack makes sure they are called before
      routineN = ""
      CALL mp_bcast(routineN, 0, mp_env%mp%mp_group)
      sending_rank = 0
      CALL mp_max(sending_rank, mp_env%mp%mp_group)
      CALL mp_sum(sending_rank, mp_env%mp%mp_group)
      foobar = 0.0_dp
      CALL mp_max(foobar, mp_env%mp%mp_group)
      dbuf = 0.0_dp
      CALL mp_maxloc(dbuf, mp_env%mp%mp_group)
      CALL mp_sum(foobar, mp_env%mp%mp_group)
      ! end hack

      ! Array collected is used as a bit field.
      ! It's of type integer in order to use the convenient MINLOC routine.
      ALLOCATE (collected(list_size(timer_env%routine_stats)))
      collected(:) = 0

      DO
         ! does any rank have uncollected stats?
         sending_rank = -1
         IF (.NOT. ALL(collected == 1)) sending_rank = mp_env%mp%mynode
         CALL mp_max(sending_rank, mp_env%mp%mp_group)
         IF (sending_rank < 0) EXIT ! every rank got all routines collected
         IF (sending_rank == mp_env%mp%mynode) THEN
            local_routine_id = MINLOC(collected, dim=1)
            r_stat => list_get(timer_env%routine_stats, local_routine_id)
            routineN = r_stat%routineN
         ENDIF
         CALL mp_bcast(routineN, sending_rank, mp_env%mp%mp_group)

         ! Create new report for routineN
         ALLOCATE (r_report)
         CALL list_push(reports, r_report)
         r_report%routineN = routineN

         ! If routineN was called on local node, add local stats
         IF (dict_haskey(timer_env%routine_names, routineN)) THEN
            local_routine_id = dict_get(timer_env%routine_names, routineN)
            collected(local_routine_id) = 1
            r_stat => list_get(timer_env%routine_stats, local_routine_id)
            r_report%max_total_calls = r_stat%total_calls
            r_report%sum_total_calls = r_stat%total_calls
            r_report%sum_stackdepth = r_stat%stackdepth_accu
            SELECT CASE (cost_type)
            CASE (cost_type_energy)
               r_report%max_icost = r_stat%incl_energy_accu
               r_report%sum_icost = r_stat%incl_energy_accu
               r_report%max_ecost = r_stat%excl_energy_accu
               r_report%sum_ecost = r_stat%excl_energy_accu
            CASE (cost_type_time)
               r_report%max_icost = r_stat%incl_walltime_accu
               r_report%sum_icost = r_stat%incl_walltime_accu
               r_report%max_ecost = r_stat%excl_walltime_accu
               r_report%sum_ecost = r_stat%excl_walltime_accu
            CASE DEFAULT
               DBCSR_ABORT("BUG")
            END SELECT
         END IF

         ! collect stats of routineN via MPI
         CALL mp_max(r_report%max_total_calls, mp_env%mp%mp_group)
         CALL mp_sum(r_report%sum_total_calls, mp_env%mp%mp_group)
         CALL mp_sum(r_report%sum_stackdepth, mp_env%mp%mp_group)

         ! get value and rank of the maximum inclusive cost
         dbuf = (/r_report%max_icost, REAL(mp_env%mp%mynode, KIND=dp)/)
         CALL mp_maxloc(dbuf, mp_env%mp%mp_group)
         r_report%max_icost = dbuf(1)
         r_report%max_irank = INT(dbuf(2))

         CALL mp_sum(r_report%sum_icost, mp_env%mp%mp_group)

         ! get value and rank of the maximum exclusive cost
         dbuf = (/r_report%max_ecost, REAL(mp_env%mp%mynode, KIND=dp)/)
         CALL mp_maxloc(dbuf, mp_env%mp%mp_group)
         r_report%max_ecost = dbuf(1)
         r_report%max_erank = INT(dbuf(2))

         CALL mp_sum(r_report%sum_ecost, mp_env%mp%mp_group)
      ENDDO

   END SUBROUTINE collect_reports_from_ranks

   SUBROUTINE print_reports(reports, iw, threshold, sort_by_exclusiv_cost, cost_type, report_maxloc, mp_env)
      !! Print the collected reports
      TYPE(list_routinereport_type), INTENT(IN)          :: reports
      INTEGER, INTENT(IN)                                :: iw
      REAL(KIND=dp), INTENT(IN)                          :: threshold
      LOGICAL, INTENT(IN)                                :: sort_by_exclusiv_cost
      INTEGER, INTENT(IN)                                :: cost_type
      LOGICAL, INTENT(IN)                                :: report_maxloc
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env

      CHARACTER(LEN=4)                                   :: label
      CHARACTER(LEN=default_string_length)               :: fmt, title
      INTEGER                                            :: decimals, i, j, num_routines
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: indices
      REAL(KIND=dp)                                      :: asd, maxcost, mincost
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:)           :: max_costs
      TYPE(routine_report_type), POINTER                 :: r_report_i, r_report_j

      NULLIFY (r_report_i, r_report_j)
      IF (.NOT. list_isready(reports)) &
         DBCSR_ABORT("BUG")

      ! are we printing timing or energy ?
      SELECT CASE (cost_type)
      CASE (cost_type_energy)
         title = "E N E R G Y"
         label = "ENER"
      CASE (cost_type_time)
         title = "T I M I N G"
         label = "TIME"
      CASE DEFAULT
         DBCSR_ABORT("BUG")
      END SELECT

      ! write banner
      WRITE (UNIT=iw, FMT="(/,T2,A)") REPEAT("-", 79)
      WRITE (UNIT=iw, FMT="(T2,A,T80,A)") "-", "-"
      WRITE (UNIT=iw, FMT="(T2,A,T35,A,T80,A)") "-", TRIM(title), "-"
      WRITE (UNIT=iw, FMT="(T2,A,T80,A)") "-", "-"
      WRITE (UNIT=iw, FMT="(T2,A)") REPEAT("-", 79)
      IF (report_maxloc) THEN
         WRITE (UNIT=iw, FMT="(T2,A,T35,A,T41,A,T45,2A18,A8)") &
            "SUBROUTINE", "CALLS", " ASD", "SELF "//label, "TOTAL "//label, "MAXRANK"
      ELSE
         WRITE (UNIT=iw, FMT="(T2,A,T35,A,T41,A,T45,2A18)") &
            "SUBROUTINE", "CALLS", " ASD", "SELF "//label, "TOTAL "//label
      ENDIF

      WRITE (UNIT=iw, FMT="(T33,A)") &
         "MAXIMUM       AVERAGE  MAXIMUM  AVERAGE  MAXIMUM"

      ! sort statistics
      num_routines = list_size(reports)
      ALLOCATE (max_costs(num_routines))
      DO i = 1, num_routines
         r_report_i => list_get(reports, i)
         IF (sort_by_exclusiv_cost) THEN
            max_costs(i) = r_report_i%max_ecost
         ELSE
            max_costs(i) = r_report_i%max_icost
         END IF
      ENDDO
      ALLOCATE (indices(num_routines))
      CALL sort(max_costs, num_routines, indices)

      maxcost = MAXVAL(max_costs)
      mincost = maxcost*threshold

      ! adjust fmt dynamically based on the max walltime.
      ! few clocks have more than 3 digits resolution, so stop there
      decimals = 3
      IF (maxcost >= 10000) decimals = 2
      IF (maxcost >= 100000) decimals = 1
      IF (maxcost >= 1000000) decimals = 0
      IF (report_maxloc) THEN
         WRITE (UNIT=fmt, FMT="(A,I0,A)") &
            "(T2,A30,1X,I7,1X,F4.1,4(1X,F8.", decimals, "),I8)"
      ELSE
         WRITE (UNIT=fmt, FMT="(A,I0,A)") &
            "(T2,A30,1X,I7,1X,F4.1,4(1X,F8.", decimals, "))"
      ENDIF

      !write output
      DO i = num_routines, 1, -1
         IF (max_costs(i) >= mincost) THEN
            j = indices(i)
            r_report_j => list_get(reports, j)
            ! average stack depth
            asd = REAL(r_report_j%sum_stackdepth, KIND=dp)/ &
                  REAL(MAX(1_int_8, r_report_j%sum_total_calls), KIND=dp)
            IF (report_maxloc) THEN
               WRITE (UNIT=iw, FMT=fmt) &
                  ADJUSTL(r_report_j%routineN(1:31)), &
                  r_report_j%max_total_calls, &
                  asd, &
                  r_report_j%sum_ecost/mp_env%mp%numnodes, &
                  r_report_j%max_ecost, &
                  r_report_j%sum_icost/mp_env%mp%numnodes, &
                  r_report_j%max_icost, &
                  r_report_j%max_erank
            ELSE
               WRITE (UNIT=iw, FMT=fmt) &
                  ADJUSTL(r_report_j%routineN(1:31)), &
                  r_report_j%max_total_calls, &
                  asd, &
                  r_report_j%sum_ecost/mp_env%mp%numnodes, &
                  r_report_j%max_ecost, &
                  r_report_j%sum_icost/mp_env%mp%numnodes, &
                  r_report_j%max_icost
            ENDIF
         END IF
      END DO
      WRITE (UNIT=iw, FMT="(T2,A,/)") REPEAT("-", 79)

   END SUBROUTINE print_reports

   SUBROUTINE timings_report_callgraph(filename)
      !! Write accumulated callgraph information as cachegrind-file.
      !! http://kcachegrind.sourceforge.net/cgi-bin/show.cgi/KcacheGrindCalltreeFormat

      CHARACTER(len=*), INTENT(in)                       :: filename

      INTEGER, PARAMETER                                 :: E = 1000, T = 100000

      INTEGER                                            :: i, unit
      TYPE(call_stat_type), POINTER                      :: c_stat
      TYPE(dict_i4tuple_callstat_item_type), &
         DIMENSION(:), POINTER                           :: ct_items
      TYPE(routine_stat_type), POINTER                   :: r_stat
      TYPE(timer_env_type), POINTER                      :: timer_env

      CALL open_file(file_name=filename, file_status="REPLACE", file_action="WRITE", &
                     file_form="FORMATTED", unit_number=unit)
      timer_env => get_timer_env()

      ! use outermost routine as total runtime
      r_stat => list_get(timer_env%routine_stats, 1)
      WRITE (UNIT=unit, FMT="(A)") "events: Walltime Energy"
      WRITE (UNIT=unit, FMT="(A,I0,1X,I0)") "summary: ", &
         INT(T*r_stat%incl_walltime_accu, KIND=int_8), &
         INT(E*r_stat%incl_energy_accu, KIND=int_8)

      DO i = 1, list_size(timer_env%routine_stats)
         r_stat => list_get(timer_env%routine_stats, i)
         WRITE (UNIT=unit, FMT="(A,I0,A,A)") "fn=(", r_stat%routine_id, ") ", r_stat%routineN
         WRITE (UNIT=unit, FMT="(A,I0,1X,I0)") "1 ", &
            INT(T*r_stat%excl_walltime_accu, KIND=int_8), &
            INT(E*r_stat%excl_energy_accu, KIND=int_8)
      END DO

      ct_items => dict_items(timer_env%callgraph)
      DO i = 1, SIZE(ct_items)
         c_stat => ct_items(i)%value
         WRITE (UNIT=unit, FMT="(A,I0,A)") "fn=(", ct_items(i)%key(1), ")"
         WRITE (UNIT=unit, FMT="(A,I0,A)") "cfn=(", ct_items(i)%key(2), ")"
         WRITE (UNIT=unit, FMT="(A,I0,A)") "calls=", c_stat%total_calls, " 1"
         WRITE (UNIT=unit, FMT="(A,I0,1X,I0)") "1 ", &
            INT(T*c_stat%incl_walltime_accu, KIND=int_8), &
            INT(E*c_stat%incl_energy_accu, KIND=int_8)
      END DO
      DEALLOCATE (ct_items)

      CALL close_file(unit_number=unit, file_status="KEEP")

   END SUBROUTINE timings_report_callgraph
END MODULE dbcsr_timings_report

