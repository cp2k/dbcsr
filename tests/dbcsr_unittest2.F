!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief   Tests for DBCSR operations
!> \author  VW
!> \date    2010
!> \version 1.0
!>
!> <b>Modification history:</b>
!> - Created 2010
!> - 2015 Rewrote into standalone unittest (Ole Schuett)
! **************************************************************************************************
PROGRAM dbcsr_unittest

   USE dbcsr_acc_device,                ONLY: acc_get_ndevices,&
                                              acc_set_active_device
   USE dbcsr_kinds,                     ONLY: dp
   USE dbcsr_lib,                       ONLY: dbcsr_finalize_lib,&
                                              dbcsr_init_lib
   USE dbcsr_machine,                   ONLY: default_output_unit
   USE dbcsr_mp_methods,                ONLY: dbcsr_mp_new,&
                                              dbcsr_mp_release
   USE dbcsr_mpiwrap,                   ONLY: mp_cart_create,&
                                              mp_cart_rank,&
                                              mp_comm_free,&
                                              mp_environ,&
                                              mp_world_finalize,&
                                              mp_world_init
   USE dbcsr_test_methods,              ONLY: dbcsr_reset_randmat_seed
   USE dbcsr_test_multiply,             ONLY: dbcsr_test_multiplies
   USE dbcsr_types,                     ONLY: dbcsr_mp_obj
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE

   INTEGER                                  :: mp_comm, group, numnodes, mynode, &
                                               prow, pcol, io_unit, handle
   INTEGER, DIMENSION(2)                    :: npdims, myploc
   INTEGER, DIMENSION(:, :), POINTER         :: pgrid
   TYPE(dbcsr_mp_obj)                       :: mp_env

   CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_check_multiply'

   !***************************************************************************************

   !
   ! initialize libdbcsr errors
   CALL timeset(routineN, handle)

   !
   ! initialize mpi
   CALL mp_world_init(mp_comm)

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
   ! initialize libdbcsr
   IF (acc_get_ndevices() > 0) &
      CALL acc_set_active_device(MOD(mynode, acc_get_ndevices()))
   CALL dbcsr_init_lib()

   CALL dbcsr_reset_randmat_seed()

   ! run tests

   ! multiply ------------------------------------------------------------------

   ! Large Blocks
   CALL dbcsr_test_multiplies("large_blocks_1", &
                              group, mp_env, npdims, io_unit, matrix_sizes=(/500, 500, 500/), &
                              sparsities=(/0.5_dp, 0.5_dp, 0.5_dp/), retain_sparsity=.FALSE., &
                              alpha=CMPLX(1.0_dp, 0.0_dp, dp), beta=CMPLX(0.0_dp, 0.0_dp, dp), &
                              bs_m=(/1, 100/), bs_n=(/1, 100/), bs_k=(/1, 100/), &
                              limits=(/1, 500, 1, 500, 1, 500/))

   CALL dbcsr_test_multiplies("large_blocks_2", &
                              group, mp_env, npdims, io_unit, matrix_sizes=(/500, 50, 50/), &
                              sparsities=(/0.5_dp, 0.5_dp, 0.5_dp/), retain_sparsity=.FALSE., &
                              alpha=CMPLX(1.0_dp, 0.0_dp, dp), beta=CMPLX(0.0_dp, 0.0_dp, dp), &
                              bs_m=(/1, 100/), bs_n=(/1, 10/), bs_k=(/1, 10/), &
                              limits=(/1, 500, 1, 50, 1, 50/))

   ! Rectangular matrices
   CALL dbcsr_test_multiplies("rectangular_matrix_M", &
                              group, mp_env, npdims, io_unit, matrix_sizes=(/500, 50, 50/), &
                              sparsities=(/0.5_dp, 0.5_dp, 0.5_dp/), retain_sparsity=.FALSE., &
                              alpha=CMPLX(1.0_dp, 0.0_dp, dp), beta=CMPLX(0.0_dp, 0.0_dp, dp), &
                              bs_m=(/1, 5/), bs_n=(/1, 5/), bs_k=(/1, 5/), &
                              limits=(/1, 500, 1, 50, 1, 50/))

   CALL dbcsr_test_multiplies("rectangular_matrix_K", &
                              group, mp_env, npdims, io_unit, matrix_sizes=(/50, 50, 500/), &
                              sparsities=(/0.5_dp, 0.5_dp, 0.5_dp/), retain_sparsity=.FALSE., &
                              alpha=CMPLX(1.0_dp, 0.0_dp, dp), beta=CMPLX(0.0_dp, 0.0_dp, dp), &
                              bs_m=(/1, 5/), bs_n=(/1, 5/), bs_k=(/1, 5/), &
                              limits=(/1, 50, 1, 50, 1, 500/))

   ! end of test cases ---------------------------------------------------------

   ! finalize libdbcsr
   CALL dbcsr_finalize_lib(mp_comm, io_unit)

   !
   ! clean mp enviroment
   CALL dbcsr_mp_release(mp_env)

   !
   ! finalize mpi
   CALL mp_comm_free(group)
   CALL mp_world_finalize()

   !
   ! finalize libdbcsr errors
   CALL timestop(handle)

END PROGRAM dbcsr_unittest
