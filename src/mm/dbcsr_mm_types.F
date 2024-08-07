!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_mm_types
   !! Datatypes used by the dbcsr matrix-matrix multiplication machinery.
   !! <b>Modification history:</b>
   !! - 2013-01    reorganized code (Ole Schuett)

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_mm_types'

   PUBLIC :: stack_descriptor_type

   ! Definitions for the members of a parameter stack.
   INTEGER, PARAMETER, PUBLIC :: dbcsr_ps_width = 7
   INTEGER, PARAMETER, PUBLIC :: dbcsr_ps_acc_width = 3
   INTEGER, PARAMETER, PUBLIC :: p_m = 1
   INTEGER, PARAMETER, PUBLIC :: p_n = 2
   INTEGER, PARAMETER, PUBLIC :: p_k = 3
   INTEGER, PARAMETER, PUBLIC :: p_a_first = 4
   INTEGER, PARAMETER, PUBLIC :: p_b_first = 5
   INTEGER, PARAMETER, PUBLIC :: p_c_first = 6
   INTEGER, PARAMETER, PUBLIC :: p_c_blk = 7

   TYPE stack_descriptor_type
      INTEGER :: m = -1, n = -1, k = -1, max_m = -1, max_n = -1, max_k = -1
      LOGICAL :: defined_mnk = .FALSE.
   END TYPE stack_descriptor_type

END MODULE dbcsr_mm_types
