!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

  TYPE ele_type
     !! Types needed for the hashtable.

     INTEGER :: c = 0
     INTEGER :: p = 0
  END TYPE ele_type

  TYPE hash_table_type
     TYPE(ele_type), DIMENSION(:), POINTER :: table
     INTEGER :: nele = 0
     INTEGER :: nmax = 0
     INTEGER :: prime = 0
  END TYPE hash_table_type
