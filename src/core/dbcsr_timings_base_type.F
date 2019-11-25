!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_timings_base_type
   !! Types used by timings.F and timings_report.F
   !! The types in this module are used within dict or list, which are
   !! in turn used in timer_env_type from timings_type.F
   !! Due to the fortran restriction on circular module-dependencies these
   !! "inner-types" had to go into a separate module.

   USE dbcsr_kinds, ONLY: default_string_length, &
                          dp, &
                          int_8

   IMPLICIT NONE
   PRIVATE

   TYPE routine_stat_type
      INTEGER       :: routine_id
      CHARACTER(len=default_string_length) :: routineN
      REAL(kind=dp) :: excl_walltime_accu
      REAL(kind=dp) :: incl_walltime_accu
      REAL(kind=dp) :: excl_energy_accu
      REAL(kind=dp) :: incl_energy_accu
      INTEGER       :: active_calls
      INTEGER       :: total_calls
      INTEGER       :: stackdepth_accu
      LOGICAL       :: trace
   END TYPE routine_stat_type

   TYPE call_stat_type
      INTEGER       :: total_calls
      REAL(kind=dp) :: incl_walltime_accu
      REAL(kind=dp) :: incl_energy_accu
   END TYPE call_stat_type

   TYPE callstack_entry_type
      INTEGER       :: routine_id
      REAL(kind=dp) :: walltime_start
      REAL(kind=dp) :: energy_start
   END TYPE callstack_entry_type

   TYPE routine_report_type
      CHARACTER(LEN=default_string_length) :: routineN
      REAL(KIND=dp)                        :: max_icost = 0.0_dp
      REAL(KIND=dp)                        :: sum_icost = 0.0_dp
      REAL(KIND=dp)                        :: max_ecost = 0.0_dp
      REAL(KIND=dp)                        :: sum_ecost = 0.0_dp
      INTEGER                              :: max_irank = 0
      INTEGER                              :: max_erank = 0
      INTEGER(kind=int_8)                  :: max_total_calls = 0
      INTEGER(kind=int_8)                  :: sum_total_calls = 0
      INTEGER(kind=int_8)                  :: sum_stackdepth = 0
   END TYPE routine_report_type

   PUBLIC :: routine_stat_type, call_stat_type, callstack_entry_type, routine_report_type

END MODULE dbcsr_timings_base_type

! **************************************************************************************************

