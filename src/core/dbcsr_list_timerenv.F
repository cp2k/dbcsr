!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

#:include 'dbcsr_list.fypp'
#:set n_inst = 0
   !! An array-based list which grows on demand.
   !! When the internal array is full, a new array of twice the size will be
   !! allocated and the items are copied over.
   !! This list can also be used as a stack.
   !! Have look at list_push(), list_pop() and list_peek().

MODULE dbcsr_list_timerenv
   USE dbcsr_timings_types, ONLY: ${uselist(usetimings)}$

$:list_body(valuetype[n_inst], valuetype_in[n_inst], valuetype_out[n_inst], value_assign[n_inst])

END MODULE dbcsr_list_timerenv
