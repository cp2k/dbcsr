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
   USE dbcsr_methods, ONLY: dbcsr_release
   USE dbcsr_mp_methods, ONLY: dbcsr_mp_new, &
                               dbcsr_mp_release
   USE dbcsr_mpiwrap, ONLY: mp_cart_create, &
                            mp_cart_rank, &
                            mp_comm_free, &
                            mp_environ, &
                            mp_world_finalize, &
                            mp_world_init, mp_comm_type
   USE dbcsr_block_operations, ONLY: dbcsr_data_clear
   USE dbcsr_dist_methods, ONLY: dbcsr_distribution_new, dbcsr_distribution_release
   USE dbcsr_dist_util, ONLY: dbcsr_checksum, dbcsr_verify_matrix
   USE dbcsr_test_methods, ONLY: dbcsr_make_random_matrix, dbcsr_reset_randmat_seed
   USE dbcsr_transformations, ONLY: dbcsr_redistribute
   USE dbcsr_test_scale_by_vector, ONLY: dbcsr_test_scale_by_vectors
   USE dbcsr_types, ONLY: dbcsr_distribution_obj, dbcsr_mp_obj, dbcsr_type, &
                          dbcsr_type_real_8, dbcsr_type_complex_8, &
                          dbcsr_type_no_symmetry, dbcsr_type_hermitian
   USE dbcsr_work_operations, ONLY: dbcsr_create
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

   CALL test_local_redistribute()

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

CONTAINS

   SUBROUTINE test_local_redistribute()
      !! Test the local copy, nonempty-target fallback and independent data ownership.
      INTEGER, PARAMETER :: types(2) = [dbcsr_type_real_8, dbcsr_type_complex_8]
      CHARACTER, PARAMETER :: syms(2) = [dbcsr_type_no_symmetry, dbcsr_type_hermitian]
      INTEGER, POINTER, CONTIGUOUS :: sizes(:), row_dist(:), col_dist(:)
      TYPE(dbcsr_distribution_obj) :: dist
      TYPE(dbcsr_type) :: source, target
      INTEGER :: i, itype
      REAL(dp) :: reference

      ALLOCATE (sizes(4), row_dist(4), col_dist(4))
      sizes = [1, 3, 2, 4]
      row_dist = [(MOD(i - 1, npdims(1)), i=1, 4)]
      col_dist = [(MOD(i - 1, npdims(2)), i=1, 4)]
      CALL dbcsr_distribution_new(dist, mp_env, row_dist, col_dist)

      DO itype = 1, SIZE(types)
         CALL dbcsr_make_random_matrix(source, sizes, sizes, "source", 0.3_dp, group, &
                                       data_type=types(itype), symmetry=syms(itype))
         CALL dbcsr_create(target, "target", dist, syms(itype), &
                           row_blk_size=sizes, col_blk_size=sizes, data_type=types(itype))
         reference = dbcsr_checksum(source, pos=.TRUE.)

         CALL dbcsr_redistribute(source, target)
         CALL check_copy(target, dist, reference)
         ! Populated targets follow the existing, general redistribution path.
         CALL dbcsr_redistribute(source, target)
         CALL check_copy(target, dist, reference)
         CALL dbcsr_data_clear(source%data_area)
         CALL dbcsr_release(source)
         CALL check_copy(target, dist, reference)

         CALL dbcsr_release(target)
      END DO
      CALL dbcsr_distribution_release(dist)
      DEALLOCATE (sizes, row_dist, col_dist)

   END SUBROUTINE test_local_redistribute

   SUBROUTINE check_copy(target, dist, reference)
      TYPE(dbcsr_type), INTENT(IN) :: target
      TYPE(dbcsr_distribution_obj), INTENT(IN) :: dist
      REAL(dp), INTENT(IN) :: reference

      CALL dbcsr_verify_matrix(target)
      IF (.NOT. ASSOCIATED(target%dist%d, dist%d)) DBCSR_ABORT("Target distribution changed")
      IF (TRIM(target%name) /= "target") DBCSR_ABORT("Target name changed")
      IF (ABS(dbcsr_checksum(target, pos=.TRUE.) - reference) > 1.0E-10_dp) &
         DBCSR_ABORT("Redistribution changed matrix data")
   END SUBROUTINE check_copy

END PROGRAM dbcsr_unittest
