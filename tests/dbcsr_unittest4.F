!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

PROGRAM dbcsr_unittest
   !! Tests for DBCSR operations

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
                            mp_world_init, mp_comm_type
   USE dbcsr_test_add, ONLY: dbcsr_test_adds
   USE dbcsr_test_methods, ONLY: dbcsr_reset_randmat_seed
   USE dbcsr_test_scale_by_vector, ONLY: dbcsr_test_scale_by_vectors
   USE dbcsr_types, ONLY: dbcsr_mp_obj
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   INTEGER                                  :: numnodes, mynode, &
                                               prow, pcol, io_unit, handle
   INTEGER, DIMENSION(2)                    :: npdims, myploc
   INTEGER, DIMENSION(:, :), POINTER        :: pgrid
   TYPE(dbcsr_mp_obj)                       :: mp_env
   LOGICAL                                  :: success
   TYPE(mp_comm_type)                       :: mp_comm, group

   CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_unittest'

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
      END DO
   END DO
   CALL dbcsr_mp_new(mp_env, group, pgrid, mynode, numnodes, &
                     myprow=myploc(1), mypcol=myploc(2))
   DEALLOCATE (pgrid)

   ! set standard output parameters
   io_unit = 0
   IF (mynode .EQ. 0) io_unit = default_output_unit

   ! initialize DBCSR
   CALL dbcsr_init_lib(mp_comm%get_handle(), io_unit)

   ! start measuring the complete test
   CALL timeset(routineN, handle)

   CALL dbcsr_reset_randmat_seed()

   ! run tests
   success = .TRUE.

   success = dbcsr_test_scale_by_vectors("scale_by_vector_symmetric", &
                                         group, mp_env, npdims, io_unit, matrix_size=[20, 20], &
                                         sparsity=0.5_dp, bs_m=[1, 4], bs_n=[1, 4], do_exact_comparison=.FALSE.) &
             .AND. success

   ! specific reproducers of https://github.com/cp2k/dbcsr/issues/362
   ! the first one gives wrong results when scaling
   success = dbcsr_test_scale_by_vectors("scale_by_vector_asymm_exact1", &
                                         group, mp_env, npdims, io_unit, matrix_size=[30, 20], &
                                         sparsity=0.0_dp, bs_m=[1, 4], bs_n=[1, 4], do_exact_comparison=.TRUE.) &
             .AND. success
   ! the second one triggers segfaults without the fix
   success = dbcsr_test_scale_by_vectors("scale_by_vector_asymm_exact2", &
                                         group, mp_env, npdims, io_unit, matrix_size=[20, 30], &
                                         sparsity=0.0_dp, bs_m=[1, 4], bs_n=[1, 4], do_exact_comparison=.TRUE.) &
             .AND. success
   CALL timestop(handle)

   ! clean mp environment
   CALL dbcsr_mp_release(mp_env)

   ! finalize mpi
   CALL mp_comm_free(group)

   call dbcsr_print_statistics(.true.)
   ! finalize libdbcsr
   CALL dbcsr_finalize_lib()
   CALL mp_world_finalize()

   ! finalize libdbcsr errors
   IF (.NOT. success) &
      ERROR STOP "one or more tests failed"
END PROGRAM dbcsr_unittest
