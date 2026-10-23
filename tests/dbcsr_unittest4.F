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
   USE dbcsr_block_access, ONLY: dbcsr_reserve_blocks
   USE dbcsr_data_methods, ONLY: dbcsr_data_init, dbcsr_data_new, dbcsr_data_release, &
                                 dbcsr_data_set_size_referenced
   USE dbcsr_data_operations, ONLY: dbcsr_data_convert
   USE dbcsr_dist_methods, ONLY: dbcsr_distribution_new, dbcsr_distribution_release
   USE dbcsr_dist_util, ONLY: dbcsr_checksum, dbcsr_verify_matrix
   USE dbcsr_operations, ONLY: dbcsr_clear, dbcsr_copy, dbcsr_copy_into_existing
   USE dbcsr_test_methods, ONLY: dbcsr_make_random_matrix, dbcsr_reset_randmat_seed, dbcsr_to_dense_local
   USE dbcsr_transformations, ONLY: dbcsr_redistribute, dbcsr_complete_redistribute
   USE dbcsr_test_scale_by_vector, ONLY: dbcsr_test_scale_by_vectors
   USE dbcsr_types, ONLY: dbcsr_data_obj, dbcsr_distribution_obj, dbcsr_mp_obj, dbcsr_type, &
                          dbcsr_type_real_4, dbcsr_type_real_8, dbcsr_type_complex_4, dbcsr_type_complex_8, &
                          dbcsr_type_no_symmetry, dbcsr_type_hermitian, dbcsr_type_complex_8_2d
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
   INTEGER, PARAMETER :: data_types(4) = [dbcsr_type_real_4, dbcsr_type_real_8, &
                                          dbcsr_type_complex_4, dbcsr_type_complex_8]

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

   CALL test_complete_redistribute()
   CALL test_copy_into_existing()
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

   SUBROUTINE test_complete_redistribute()
      INTEGER, POINTER, CONTIGUOUS :: rbs(:), cbs(:), rd(:), cd(:)
      INTEGER, POINTER, CONTIGUOUS :: new_rbs(:), new_cbs(:), new_rd(:), new_cd(:)
      INTEGER :: itype, icase, r, c, i, j, k, n, p, ro, co, offset
      INTEGER :: rows(16), cols(16)
      INTEGER, ALLOCATABLE :: index_before(:)
      TYPE(dbcsr_distribution_obj) :: dist, new_dist
      TYPE(dbcsr_type) :: source, target, converted
      TYPE(dbcsr_data_obj) :: values, actual
      COMPLEX(dp) :: reference(10, 10), expected(10, 10)
      LOGICAL :: keep

      ALLOCATE (rbs(4), cbs(4), rd(4), cd(4), new_rbs(3), new_cbs(3), new_rd(3), new_cd(3))
      rbs = [2, 3, 1, 4]; cbs = [3, 1, 2, 4]
      new_rbs = [3, 4, 3]; new_cbs = [2, 5, 3]
      rd = [(MOD(i - 1, npdims(1)), i=1, 4)]
      cd = [(MOD(i - 1, npdims(2)), i=1, 4)]
      new_rd = [(MOD(i, npdims(1)), i=1, 3)]
      new_cd = [(MOD(i, npdims(2)), i=1, 3)]
      CALL dbcsr_distribution_new(dist, mp_env, rd, cd)
      CALL dbcsr_distribution_new(new_dist, mp_env, new_rd, new_cd)
      CALL dbcsr_data_init(values)
      CALL dbcsr_data_init(actual)
      CALL dbcsr_data_new(values, dbcsr_type_complex_8, data_size=100)
      CALL dbcsr_data_new(actual, dbcsr_type_complex_8_2d, 10, 10)
      DO itype = 1, SIZE(data_types)
         reference = CMPLX(0.0_dp, 0.0_dp, dp)
         CALL dbcsr_create(source, "reblock source", dist, dbcsr_type_no_symmetry, &
                           row_blk_size=rbs, col_blk_size=cbs, data_type=data_types(itype))
         n = 0
         DO r = 1, SIZE(rbs)
            ro = SUM(rbs(:r - 1))
            DO c = 1, SIZE(cbs)
               IF (MOD(r + c, 3) == 0) CYCLE
               co = SUM(cbs(:c - 1))
               DO j = 1, cbs(c)
                  DO i = 1, rbs(r)
                     reference(ro + i, co + j) = CMPLX(100*(ro + i) + co + j, 0, dp)
                  END DO
               END DO
               CALL mp_cart_rank(group, [rd(r), cd(c)], p)
               IF (p /= mynode) CYCLE
               n = n + 1; rows(n) = r; cols(n) = c
            END DO
         END DO
         IF (itype > 2) reference = reference*CMPLX(1.0_dp, -1.0_dp, dp)
         CALL dbcsr_reserve_blocks(source, rows(:n), cols(:n))
         DO k = 1, n
            r = rows(k); c = cols(k)
            ro = SUM(rbs(:r - 1)); co = SUM(cbs(:c - 1))
            offset = source%blk_p(k)
            values%d%c_dp(offset:offset + rbs(r)*cbs(c) - 1) = &
               RESHAPE(reference(ro + 1:ro + rbs(r), co + 1:co + cbs(c)), [rbs(r)*cbs(c)])
         END DO
         CALL dbcsr_data_set_size_referenced(values, source%nze)
         CALL dbcsr_data_convert(source%data_area, values)
         DO icase = 1, 4
            ! Empty target, populated target, retained sparsity, and empty source.
            keep = icase == 3
            CALL dbcsr_create(target, "reblock target", new_dist, dbcsr_type_no_symmetry, &
                              row_blk_size=new_rbs, col_blk_size=new_cbs, data_type=data_types(itype))
            IF (icase > 1) THEN
               n = 0
               DO r = 1, SIZE(new_rbs)
                  DO c = 1, SIZE(new_cbs)
                     IF (MOD(r + c, 2) == 0) CYCLE
                     CALL mp_cart_rank(group, [new_rd(r), new_cd(c)], p)
                     IF (p /= mynode) CYCLE
                     n = n + 1; rows(n) = r; cols(n) = c
                  END DO
               END DO
               CALL dbcsr_reserve_blocks(target, rows(:n), cols(:n))
               values%d%c_dp(:) = CMPLX(-7.0_dp, 0.0_dp, dp)
               CALL dbcsr_data_set_size_referenced(values, target%nze)
               CALL dbcsr_data_convert(target%data_area, values)
            END IF
            IF (icase == 4) THEN
               CALL dbcsr_clear(source)
               reference = CMPLX(0.0_dp, 0.0_dp, dp)
            END IF
            ALLOCATE (index_before, SOURCE=target%index)
            CALL dbcsr_complete_redistribute(source, target, keep_sparsity=keep)
            CALL dbcsr_verify_matrix(target)
            IF (.NOT. ASSOCIATED(target%dist%d, new_dist%d)) &
               DBCSR_ABORT("Reblocking changed distribution")
            IF (keep .OR. icase == 4) THEN
               IF (ANY(target%index /= index_before)) &
                  DBCSR_ABORT("Reblocking rebuilt an unchanged index")
            END IF
            DEALLOCATE (index_before)
            IF (icase == 1) THEN
               ! CSR conversion requires newly built matrices to have packed row-wise data.
               offset = 1
               DO r = 1, SIZE(new_rbs)
                  DO k = target%row_p(r) + 1, target%row_p(r + 1)
                     IF (target%blk_p(k) /= offset) &
                        DBCSR_ABORT("Reblocking changed data order")
                     offset = offset + new_rbs(r)*new_cbs(target%col_i(k))
                  END DO
               END DO
            END IF
            CALL dbcsr_create(converted, template=target, data_type=dbcsr_type_complex_8)
            CALL dbcsr_copy(converted, target)
            CALL dbcsr_to_dense_local(converted, actual)
            CALL dbcsr_release(converted)
            expected = reference
            ! Compare locally, including zeros outside this rank's ownership.
            DO r = 1, SIZE(new_rbs)
               ro = SUM(new_rbs(:r - 1))
               DO c = 1, SIZE(new_cbs)
                  co = SUM(new_cbs(:c - 1))
                  CALL mp_cart_rank(group, [new_rd(r), new_cd(c)], p)
                  IF (p /= mynode .OR. (keep .AND. MOD(r + c, 2) == 0)) &
                     expected(ro + 1:ro + new_rbs(r), co + 1:co + new_cbs(c)) = 0.0_dp
               END DO
            END DO
            IF (ANY(ABS(actual%d%c2_dp - expected) > 1.0E-6_dp)) &
               DBCSR_ABORT("Reblocking changed matrix values")
            CALL dbcsr_release(target)
         END DO
         CALL dbcsr_release(source)
      END DO
      CALL dbcsr_data_release(values)
      CALL dbcsr_data_release(actual)
      CALL dbcsr_distribution_release(dist)
      CALL dbcsr_distribution_release(new_dist)
      DEALLOCATE (rbs, cbs, rd, cd, new_rbs, new_cbs, new_rd, new_cd)
   END SUBROUTINE test_complete_redistribute

   SUBROUTINE test_copy_into_existing()
      INTEGER, POINTER, CONTIGUOUS :: sizes(:), row_dist(:), col_dist(:)
      INTEGER, ALLOCATABLE :: index_before(:)
      INTEGER :: itype, icase, i
      TYPE(dbcsr_distribution_obj) :: dist
      TYPE(dbcsr_type) :: source, target
      TYPE(dbcsr_data_obj) :: values, actual
      COMPLEX(dp) :: reference(25)

      ALLOCATE (sizes(4), row_dist(4), col_dist(4))
      sizes = [2, 3, 2, 3]
      row_dist = 0; col_dist = 0
      CALL dbcsr_distribution_new(dist, mp_env, row_dist, col_dist)
      CALL dbcsr_data_init(values)
      CALL dbcsr_data_init(actual)
      CALL dbcsr_data_new(values, dbcsr_type_complex_8, data_size=25)
      CALL dbcsr_data_new(actual, dbcsr_type_complex_8, data_size=25)
      DO itype = 1, SIZE(data_types)
         values%d%c_dp(:) = CMPLX([(REAL(i, dp), i=1, 25)], 0.0_dp, dp)
         IF (itype > 2) values%d%c_dp(:) = values%d%c_dp(:)*CMPLX(1.0_dp, -1.0_dp, dp)
         CALL dbcsr_create(source, "source", dist, dbcsr_type_no_symmetry, &
                           row_blk_size=sizes, col_blk_size=sizes, data_type=data_types(itype))
         IF (mynode == 0) THEN
            CALL dbcsr_reserve_blocks(source, [1, 1, 2, 3], [1, 2, 2, 4])
            CALL dbcsr_data_convert(source%data_area, values)
         END IF
         CALL dbcsr_copy(target, source)
         DO icase = 1, 4
            ! Equal layouts, different sparsity with equal sizes, transposition, empty source.
            IF (icase == 2) THEN
               CALL dbcsr_release(target)
               CALL dbcsr_create(target, template=source)
               IF (mynode == 0) CALL dbcsr_reserve_blocks(target, [1, 2, 3, 3], [2, 2, 3, 4])
            END IF
            IF (icase == 3 .AND. mynode == 0) source%blk_p(4) = -source%blk_p(4)
            IF (icase == 4) CALL dbcsr_clear(source)
            ALLOCATE (index_before, SOURCE=target%index)
            CALL dbcsr_copy_into_existing(target, source)
            IF (ANY(target%index /= index_before)) DBCSR_ABORT("Copy changed target sparsity")
            DEALLOCATE (index_before)
            IF (mynode == 0) THEN
               reference = values%d%c_dp(:)
               IF (icase > 1) THEN
                  reference(1:15) = values%d%c_dp(5:19)
                  reference(16:19) = 0.0_dp
               END IF
               IF (icase == 3) reference(20:25) = values%d%c_dp([20, 23, 21, 24, 22, 25])
               IF (icase == 4) reference = 0.0_dp
               CALL dbcsr_data_convert(actual, target%data_area)
               IF (ANY(ABS(actual%d%c_dp(:) - reference) > 1.0E-6_dp)) &
                  DBCSR_ABORT("Copy changed matrix values")
            END IF
         END DO
         CALL dbcsr_release(target)
         CALL dbcsr_release(source)
      END DO
      CALL dbcsr_data_release(values)
      CALL dbcsr_data_release(actual)
      CALL dbcsr_distribution_release(dist)
      DEALLOCATE (sizes, row_dist, col_dist)
   END SUBROUTINE test_copy_into_existing

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
