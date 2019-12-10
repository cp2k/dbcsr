!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!


MODULE dbcsr_list
   !! An array-based list which grows on demand.
   !! When the internal array is full, a new array of twice the size will be
   !! allocated and the items are copied over.
   !! This list can also be used as a stack.
   !! Have look at list_push(), list_pop() and list_peek().
   !! @note it's not possible to put all templates in a single module because this would lead to circular
   !! dependencies (timer_env_type contains list_routinestat_type and list_callstackentry_type, and
   !! list_timerenv_type contains timer_env_type)

   USE dbcsr_timings_base_type, ONLY: routine_stat_type, routine_report_type, callstack_entry_type
   USE dbcsr_list_callstackentry, ONLY: list_callstackentry_type, list_callstackentry_init, list_callstackentry_push,&
       & list_callstackentry_pop, list_callstackentry_peek, list_callstackentry_insert, list_callstackentry_set,&
       & list_callstackentry_get, list_callstackentry_del, list_callstackentry_clear, list_callstackentry_size,&
       & list_callstackentry_destroy, list_callstackentry_isready
   USE dbcsr_list_routinereport, ONLY: list_routinereport_type, list_routinereport_init, list_routinereport_push,&
       & list_routinereport_pop, list_routinereport_peek, list_routinereport_insert, list_routinereport_set,&
       & list_routinereport_get, list_routinereport_del, list_routinereport_clear, list_routinereport_size,&
       & list_routinereport_destroy, list_routinereport_isready
   USE dbcsr_list_routinestat, ONLY: list_routinestat_type, list_routinestat_init, list_routinestat_push, list_routinestat_pop,&
       & list_routinestat_peek, list_routinestat_insert, list_routinestat_set, list_routinestat_get, list_routinestat_del,&
       & list_routinestat_clear, list_routinestat_size, list_routinestat_destroy, list_routinestat_isready
   USE dbcsr_list_timerenv, ONLY: list_timerenv_type, list_timerenv_init, list_timerenv_push, list_timerenv_pop,&
       & list_timerenv_peek, list_timerenv_insert, list_timerenv_set, list_timerenv_get, list_timerenv_del, list_timerenv_clear,&
       & list_timerenv_size, list_timerenv_destroy, list_timerenv_isready

#include "base/dbcsr_base_uses.f90"
   IMPLICIT NONE

   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_list'

   PUBLIC :: list_init, list_isready, list_push, list_pop, list_peek, &
             list_get, list_size, list_destroy

   PUBLIC :: list_timerenv_type
   PUBLIC :: list_routinestat_type
   PUBLIC :: list_routinereport_type
   PUBLIC :: list_callstackentry_type

   INTERFACE list_init
      MODULE PROCEDURE list_timerenv_init
      MODULE PROCEDURE list_routinestat_init
      MODULE PROCEDURE list_routinereport_init
      MODULE PROCEDURE list_callstackentry_init
   END INTERFACE

   INTERFACE list_isready
      MODULE PROCEDURE list_timerenv_isready
      MODULE PROCEDURE list_routinestat_isready
      MODULE PROCEDURE list_routinereport_isready
      MODULE PROCEDURE list_callstackentry_isready
   END INTERFACE

   INTERFACE list_push
      MODULE PROCEDURE list_timerenv_push
      MODULE PROCEDURE list_routinestat_push
      MODULE PROCEDURE list_routinereport_push
      MODULE PROCEDURE list_callstackentry_push
   END INTERFACE

   INTERFACE list_pop
      MODULE PROCEDURE list_timerenv_pop
      MODULE PROCEDURE list_routinestat_pop
      MODULE PROCEDURE list_routinereport_pop
      MODULE PROCEDURE list_callstackentry_pop
   END INTERFACE

   INTERFACE list_peek
      MODULE PROCEDURE list_timerenv_peek
      MODULE PROCEDURE list_routinestat_peek
      MODULE PROCEDURE list_routinereport_peek
      MODULE PROCEDURE list_callstackentry_peek
   END INTERFACE

   INTERFACE list_insert
      MODULE PROCEDURE list_timerenv_insert
      MODULE PROCEDURE list_routinestat_insert
      MODULE PROCEDURE list_routinereport_insert
      MODULE PROCEDURE list_callstackentry_insert
   END INTERFACE

   INTERFACE list_set
      MODULE PROCEDURE list_timerenv_set
      MODULE PROCEDURE list_routinestat_set
      MODULE PROCEDURE list_routinereport_set
      MODULE PROCEDURE list_callstackentry_set
   END INTERFACE

   INTERFACE list_get
      MODULE PROCEDURE list_timerenv_get
      MODULE PROCEDURE list_routinestat_get
      MODULE PROCEDURE list_routinereport_get
      MODULE PROCEDURE list_callstackentry_get
   END INTERFACE

   INTERFACE list_del
      MODULE PROCEDURE list_timerenv_del
      MODULE PROCEDURE list_routinestat_del
      MODULE PROCEDURE list_routinereport_del
      MODULE PROCEDURE list_callstackentry_del
   END INTERFACE

   INTERFACE list_clear
      MODULE PROCEDURE list_timerenv_clear
      MODULE PROCEDURE list_routinestat_clear
      MODULE PROCEDURE list_routinereport_clear
      MODULE PROCEDURE list_callstackentry_clear
   END INTERFACE

   INTERFACE list_size
      MODULE PROCEDURE list_timerenv_size
      MODULE PROCEDURE list_routinestat_size
      MODULE PROCEDURE list_routinereport_size
      MODULE PROCEDURE list_callstackentry_size
   END INTERFACE

   INTERFACE list_destroy
      MODULE PROCEDURE list_timerenv_destroy
      MODULE PROCEDURE list_routinestat_destroy
      MODULE PROCEDURE list_routinereport_destroy
      MODULE PROCEDURE list_callstackentry_destroy
   END INTERFACE

END MODULE dbcsr_list
