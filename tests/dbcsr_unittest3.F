!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

PROGRAM dbcsr_unittest
   !! Tests for DBCSR's libsmm_acc backend

   USE dbcsr_acc_device, ONLY: dbcsr_acc_get_ndevices, &
                               dbcsr_acc_set_active_device
   USE dbcsr_kinds, ONLY: dp
   USE dbcsr_lib, ONLY: dbcsr_finalize_lib, &
                        dbcsr_init_lib, &
                        dbcsr_print_statistics
   USE dbcsr_machine, ONLY: default_output_unit
   USE dbcsr_mp_methods, ONLY: dbcsr_mp_new, &
                               dbcsr_mp_release
   USE dbcsr_mpiwrap, ONLY: mp_cart_create, &
                            mp_cart_rank, &
                            mp_comm_free, &
                            mp_environ, &
                            mp_world_finalize, &
                            mp_world_init
   USE dbcsr_test_methods, ONLY: dbcsr_reset_randmat_seed
   USE dbcsr_test_multiply, ONLY: dbcsr_test_multiplies
   USE dbcsr_types, ONLY: dbcsr_mp_obj
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   INTEGER                                  :: mp_comm, group, numnodes, mynode, &
                                               prow, pcol, io_unit, handle
   INTEGER, DIMENSION(2)                    :: npdims, myploc
   INTEGER, DIMENSION(:, :), POINTER         :: pgrid
   TYPE(dbcsr_mp_obj)                       :: mp_env

   CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_unittest'

   !***************************************************************************************

   !
   ! initialize mpi
   CALL mp_world_init(mp_comm)

   ! setup the mp environment
   npdims(:) = 0
   CALL mp_cart_create(mp_comm, 2, npdims, myploc, group)
   CALL mp_environ(numnodes, mynode, group)
   ALLOCATE (pgrid(0:npdims(1) - 1, 0:npdims(2) - 1))
   DO prow = 0, npdims(1) - 1
      DO pcol = 0, npdims(2) - 1
         CALL mp_cart_rank(group, (/prow, pcol/), pgrid(prow, pcol))
      ENDDO
   ENDDO
   CALL dbcsr_mp_new(mp_env, group, pgrid, mynode, numnodes, &
                     myprow=myploc(1), mypcol=myploc(2))
   DEALLOCATE (pgrid)
   !
   ! set standard output parameters
   io_unit = 0
   IF (mynode .EQ. 0) io_unit = default_output_unit

   !
   ! initialize libdbcsr
   IF (dbcsr_acc_get_ndevices() > 0) &
      CALL dbcsr_acc_set_active_device(MOD(mynode, dbcsr_acc_get_ndevices()))

   CALL dbcsr_init_lib(mp_comm, io_unit)
   !
   ! initialize libdbcsr errors
   CALL timeset(routineN, handle)

   CALL dbcsr_reset_randmat_seed()

   ! run tests

   ! multiply ------------------------------------------------------------------

   ! ...
   CALL dbcsr_test_multiplies("blocks_4_5_7", &
                              group, mp_env, npdims, io_unit, matrix_sizes=(/496, 48, 48/), &
                              sparsities=(/0.5_dp, 0.5_dp, 0.5_dp/), retain_sparsity=.FALSE., &
                              alpha=CMPLX(1.0_dp, 0.0_dp, dp), beta=CMPLX(0.0_dp, 0.0_dp, dp), &
                              bs_m=(/1, 4, 1, 5, 1, 7/), bs_n=(/1, 4, 1, 5, 1, 7/), bs_k=(/1, 4, 1, 5, 1, 7/), &
                              limits=(/1, 496, 1, 48, 1, 48/))
   CALL dbcsr_test_multiplies("blocks_5_8_9", &
                              group, mp_env, npdims, io_unit, matrix_sizes=(/506, 44, 44/), &
                              sparsities=(/0.5_dp, 0.5_dp, 0.5_dp/), retain_sparsity=.FALSE., &
                              alpha=CMPLX(1.0_dp, 0.0_dp, dp), beta=CMPLX(0.0_dp, 0.0_dp, dp), &
                              bs_m=(/1, 5, 1, 8, 1, 9/), bs_n=(/1, 5, 1, 8, 1, 9/), bs_k=(/1, 5, 1, 8, 1, 9/), &
                              limits=(/1, 506, 1, 44, 1, 44/))
   CALL dbcsr_test_multiplies("blocks_4_13_25", &
                              group, mp_env, npdims, io_unit, matrix_sizes=(/504, 42, 42/), &
                              sparsities=(/0.5_dp, 0.5_dp, 0.5_dp/), retain_sparsity=.FALSE., &
                              alpha=CMPLX(1.0_dp, 0.0_dp, dp), beta=CMPLX(0.0_dp, 0.0_dp, dp), &
                              bs_m=(/1, 4, 1, 13, 1, 25/), bs_n=(/1, 4, 1, 13, 1, 25/), bs_k=(/1, 4, 1, 13, 1, 25/), &
                              limits=(/1, 504, 1, 42, 1, 42/))
   CALL dbcsr_test_multiplies("blocks_14_29_32", &
                              group, mp_env, npdims, io_unit, matrix_sizes=(/525, 75, 75/), &
                              sparsities=(/0.5_dp, 0.5_dp, 0.5_dp/), retain_sparsity=.FALSE., &
                              alpha=CMPLX(1.0_dp, 0.0_dp, dp), beta=CMPLX(0.0_dp, 0.0_dp, dp), &
                              bs_m=(/1, 14, 1, 29, 1, 32/), bs_n=(/1, 14, 1, 29, 1, 32/), bs_k=(/1, 14, 1, 29, 1, 32/), &
                              limits=(/1, 525, 1, 75, 1, 75/))
   CALL dbcsr_test_multiplies("blocks_H2O", &
                              group, mp_env, npdims, io_unit, matrix_sizes=(/552, 46, 46/), &
                              sparsities=(/0.5_dp, 0.5_dp, 0.5_dp/), retain_sparsity=.FALSE., &
                              alpha=CMPLX(1.0_dp, 0.0_dp, dp), beta=CMPLX(0.0_dp, 0.0_dp, dp), &
                              bs_m=(/1, 23/), bs_n=(/1, 23/), bs_k=(/1, 23/), &
                              limits=(/1, 552, 1, 46, 1, 46/))

   ! end of test cases ---------------------------------------------------------

   !
   ! finalize libdbcsr errors
   CALL timestop(handle)

   !
   ! clean mp environment
   CALL dbcsr_mp_release(mp_env)

   !
   ! finalize mpi
   CALL mp_comm_free(group)

   call dbcsr_print_statistics(.true.)
   ! finalize libdbcsr
   CALL dbcsr_finalize_lib()

   CALL mp_world_finalize()

END PROGRAM dbcsr_unittest
