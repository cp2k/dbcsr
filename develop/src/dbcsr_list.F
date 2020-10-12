# 1 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F" 1
!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

# 1 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.fypp" 1
# 9 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.fypp"

# 327 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.fypp"
# 11 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F" 2
MODULE dbcsr_list
   !! An array-based list which grows on demand.
   !! When the internal array is full, a new array of twice the size will be
   !! allocated and the items are copied over.
   !! This list can also be used as a stack.
   !! Have look at list_push(), list_pop() and list_peek().
   !! @note it's not possible to put all templates in a single module because this would lead to circular
   !! dependencies (timer_env_type contains list_routinestat_type and list_callstackentry_type, and
   !! list_timerenv_type contains timer_env_type)

   USE dbcsr_timings_base_type, ONLY: routine_stat_type, callstack_entry_type, routine_report_type
   USE dbcsr_list_callstackentry, ONLY: list_callstackentry_type, list_callstackentry_init, list_callstackentry_push,&
# 22 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
       & list_callstackentry_pop, list_callstackentry_peek, list_callstackentry_insert, list_callstackentry_set,&
# 22 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
       & list_callstackentry_get, list_callstackentry_del, list_callstackentry_clear, list_callstackentry_size,&
# 22 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
       & list_callstackentry_destroy, list_callstackentry_isready
   USE dbcsr_list_routinereport, ONLY: list_routinereport_type, list_routinereport_init, list_routinereport_push,&
# 23 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
       & list_routinereport_pop, list_routinereport_peek, list_routinereport_insert, list_routinereport_set,&
# 23 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
       & list_routinereport_get, list_routinereport_del, list_routinereport_clear, list_routinereport_size,&
# 23 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
       & list_routinereport_destroy, list_routinereport_isready
   USE dbcsr_list_routinestat, ONLY: list_routinestat_type, list_routinestat_init, list_routinestat_push, list_routinestat_pop,&
# 24 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
       & list_routinestat_peek, list_routinestat_insert, list_routinestat_set, list_routinestat_get, list_routinestat_del,&
# 24 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
       & list_routinestat_clear, list_routinestat_size, list_routinestat_destroy, list_routinestat_isready
   USE dbcsr_list_timerenv, ONLY: list_timerenv_type, list_timerenv_init, list_timerenv_push, list_timerenv_pop,&
# 25 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
       & list_timerenv_peek, list_timerenv_insert, list_timerenv_set, list_timerenv_get, list_timerenv_del, list_timerenv_clear,&
# 25 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
       & list_timerenv_size, list_timerenv_destroy, list_timerenv_isready

#include "base/dbcsr_base_uses.f90"
   IMPLICIT NONE

   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_list'

   PUBLIC :: list_init, list_isready, list_push, list_pop, list_peek, &
             list_get, list_size, list_destroy

# 37 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
   PUBLIC :: list_timerenv_type
# 37 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
   PUBLIC :: list_routinestat_type
# 37 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
   PUBLIC :: list_routinereport_type
# 37 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
   PUBLIC :: list_callstackentry_type
# 39 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"

   INTERFACE list_init
# 42 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_timerenv_init
# 42 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinestat_init
# 42 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinereport_init
# 42 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_callstackentry_init
# 44 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
   END INTERFACE

   INTERFACE list_isready
# 48 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_timerenv_isready
# 48 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinestat_isready
# 48 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinereport_isready
# 48 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_callstackentry_isready
# 50 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
   END INTERFACE

   INTERFACE list_push
# 54 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_timerenv_push
# 54 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinestat_push
# 54 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinereport_push
# 54 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_callstackentry_push
# 56 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
   END INTERFACE

   INTERFACE list_pop
# 60 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_timerenv_pop
# 60 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinestat_pop
# 60 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinereport_pop
# 60 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_callstackentry_pop
# 62 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
   END INTERFACE

   INTERFACE list_peek
# 66 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_timerenv_peek
# 66 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinestat_peek
# 66 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinereport_peek
# 66 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_callstackentry_peek
# 68 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
   END INTERFACE

   INTERFACE list_insert
# 72 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_timerenv_insert
# 72 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinestat_insert
# 72 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinereport_insert
# 72 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_callstackentry_insert
# 74 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
   END INTERFACE

   INTERFACE list_set
# 78 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_timerenv_set
# 78 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinestat_set
# 78 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinereport_set
# 78 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_callstackentry_set
# 80 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
   END INTERFACE

   INTERFACE list_get
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_timerenv_get
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinestat_get
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinereport_get
# 84 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_callstackentry_get
# 86 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
   END INTERFACE

   INTERFACE list_del
# 90 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_timerenv_del
# 90 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinestat_del
# 90 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinereport_del
# 90 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_callstackentry_del
# 92 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
   END INTERFACE

   INTERFACE list_clear
# 96 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_timerenv_clear
# 96 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinestat_clear
# 96 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinereport_clear
# 96 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_callstackentry_clear
# 98 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
   END INTERFACE

   INTERFACE list_size
# 102 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_timerenv_size
# 102 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinestat_size
# 102 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinereport_size
# 102 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_callstackentry_size
# 104 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
   END INTERFACE

   INTERFACE list_destroy
# 108 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_timerenv_destroy
# 108 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinestat_destroy
# 108 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_routinereport_destroy
# 108 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
      MODULE PROCEDURE list_callstackentry_destroy
# 110 "/__w/dbcsr/dbcsr/src/core/dbcsr_list.F"
   END INTERFACE

END MODULE dbcsr_list
