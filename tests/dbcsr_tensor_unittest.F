!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

PROGRAM dbcsr_tensor_unittest
   !! DBCSR tensor unit test.
   USE dbcsr_api, ONLY: dbcsr_finalize_lib, &
                        dbcsr_init_lib, &
                        dbcsr_type_real_8, &
                        dbcsr_print_statistics
   USE dbcsr_machine, ONLY: default_output_unit
   USE dbcsr_mpiwrap, ONLY: mp_cart_create, &
                            mp_comm_free, &
                            mp_environ, &
                            mp_world_finalize, &
                            mp_world_init
   USE dbcsr_tensor_test, ONLY: dbcsr_t_contract_test, &
                                dbcsr_t_random_dist, &
                                dbcsr_t_setup_test_tensor, &
                                dbcsr_t_test_formats
   USE dbcsr_tensor_types, ONLY: dbcsr_t_create, &
                                 dbcsr_t_destroy, &
                                 dbcsr_t_distribution_destroy, &
                                 dbcsr_t_distribution_new, &
                                 dbcsr_t_distribution_type, &
                                 dbcsr_t_nd_mp_comm, &
                                 dbcsr_t_type, &
                                 dbcsr_t_pgrid_type, &
                                 dbcsr_t_pgrid_create, &
                                 dbcsr_t_get_info, &
                                 dbcsr_t_pgrid_destroy
   USE dbcsr_data_methods, ONLY: dbcsr_scalar
   USE dbcsr_tensor,       ONLY: dbcsr_t_ndims
   USE dbcsr_kinds,        ONLY: real_8
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   INTEGER                            :: mp_comm, numnodes, mynode, io_unit
   INTEGER                            :: ndims, nblks_alloc, nblks_1, nblks_2, nblks_3, nblks_4, nblks_5, &
                                         nblks_alloc_1, nblks_alloc_2, nblks_alloc_3, nblks_alloc_4, nblks_alloc_5
   INTEGER, DIMENSION(:), ALLOCATABLE :: size_1, size_2, size_3, size_4, size_5, dist1_1, dist1_2, dist1_3, &
                                         dist2_1, dist2_2, dist3_1, dist3_2, dist3_3, dist4_1, dist4_2, &
                                         dist4_3, dist4_4, dist5_1, dist5_2, dist5_3
   INTEGER, DIMENSION(:), ALLOCATABLE :: blk_ind_1, blk_ind_2, blk_ind_3, blk_ind_4, blk_ind_1_1, blk_ind_2_1, &
                                         blk_ind_3_1, blk_ind_3_2, blk_ind_4_2, blk_ind_1_3, blk_ind_2_3, &
                                         blk_ind_4_3, blk_ind_1_4, blk_ind_2_4, blk_ind_4_4, blk_ind_5_4, &
                                         blk_ind_3_5, blk_ind_4_5, blk_ind_5_5
   INTEGER, DIMENSION(:), ALLOCATABLE :: map11, map31, map12, map32, map21, map22

   LOGICAL, PARAMETER                 :: verbose = .FALSE.
   TYPE(dbcsr_t_distribution_type)         :: dist1, dist2, dist3
   TYPE(dbcsr_t_type)            :: tensor_A, tensor_B, tensor_C

   LOGICAL, PARAMETER                 :: test_format = .TRUE.
   LOGICAL, PARAMETER                 :: test_contraction = .TRUE.
   INTEGER, DIMENSION(4)              :: pdims_4d
   INTEGER, DIMENSION(3)              :: pdims_3d
   INTEGER, DIMENSION(2)              :: pdims_2d
   TYPE(dbcsr_t_pgrid_type)           :: pgrid_2d, pgrid_3d, pgrid_4d
   INTEGER, DIMENSION(:), ALLOCATABLE :: bounds_t
   INTEGER, DIMENSION(:, :), ALLOCATABLE :: bounds, bounds_1, bounds_2

   CALL mp_world_init(mp_comm)
   CALL mp_environ(numnodes, mynode, mp_comm)

   ! set standard output parameters
   io_unit = 0
   IF (mynode .EQ. 0) io_unit = default_output_unit

   ! initialize libdbcsr
   CALL dbcsr_init_lib(mp_comm, io_unit)

   ! Process grid

   IF (test_format) THEN
!--------------------------------------------------------------------------------------------------!
! Test 1: Testing matrix representations of tensor rank 2                                                  !
!--------------------------------------------------------------------------------------------------!
      ndims = 2

      ! Number of blocks in each dimension
      nblks_1 = 14
      nblks_2 = 21

      ! Block sizes in each dimension
      ALLOCATE (size_1(nblks_1), size_2(nblks_2))

      size_1(:) = [3, 5, 1, 23, 2, 3, 1, 6, 3, 8, 2, 3, 5, 1]
      size_2(:) = [4, 2, 5, 3, 1, 5, 13, 5, 2, 4, 5, 6, 7, 2, 3, 1, 2, 6, 9, 12, 21]

      ! Number of non-zero blocks
      nblks_alloc = 12
      ALLOCATE (blk_ind_1(nblks_alloc), blk_ind_2(nblks_alloc))

      ! Indices of non-zero blocks (s.t. index of ith block is [blk_ind_1(i), blk_ind_2(i), ...])
      blk_ind_1(:) = [1, 1,  1,  2, 4,  4,  7,  10, 10, 10, 10, 13] !&
      blk_ind_2(:) = [1, 3, 11, 15, 4, 17, 21,   6,  9, 13, 19,  7] !&

      ! Test tensor formats
      CALL dbcsr_t_test_formats(ndims, mp_comm, default_output_unit, verbose, &
                                blk_size_1=size_1, blk_size_2=size_2, &
                                blk_ind_1=blk_ind_1, blk_ind_2=blk_ind_2)

      DEALLOCATE (size_1, size_2)
      DEALLOCATE (blk_ind_1, blk_ind_2)

!--------------------------------------------------------------------------------------------------!
! Test 2: Testing matrix representations of tensor rank 3                                          !
!--------------------------------------------------------------------------------------------------!
      ndims = 3

      ! Number of blocks in each dimension
      nblks_1 = 4
      nblks_2 = 6
      nblks_3 = 3

      ! Block sizes in each dimension
      ALLOCATE (size_1(nblks_1), size_2(nblks_2), size_3(nblks_3))

      size_1(:) = [3, 1, 5, 2]
      size_2(:) = [1, 2, 5, 3, 2, 4]
      size_3(:) = [4, 2, 10]

      ! Number of non-zero blocks
      nblks_alloc = 6
      ALLOCATE (blk_ind_1(nblks_alloc), blk_ind_2(nblks_alloc), blk_ind_3(nblks_alloc))

      ! Indices of non-zero blocks (s.t. index of ith block is [blk_ind_1(i), blk_ind_2(i), ...])
      blk_ind_1(:) = [1, 1, 1, 2, 2, 2] !&
      blk_ind_2(:) = [2, 2, 4, 1, 1, 2] !&
      blk_ind_3(:) = [1, 3, 3, 2, 3, 2] !&

      ! Test tensor formats
      CALL dbcsr_t_test_formats(ndims, mp_comm, default_output_unit, verbose, &
                                blk_size_1=size_1, blk_size_2=size_2, blk_size_3=size_3, &
                                blk_ind_1=blk_ind_1, blk_ind_2=blk_ind_2, blk_ind_3=blk_ind_3)

      DEALLOCATE (size_1, size_2, size_3)
      DEALLOCATE (blk_ind_1, blk_ind_2, blk_ind_3)

!--------------------------------------------------------------------------------------------------!
! Test 3: Testing matrix representations of tensor rank 4                                          !
!--------------------------------------------------------------------------------------------------!
      ndims = 4

      ! Number of blocks in each dimension
      nblks_1 = 2
      nblks_2 = 13
      nblks_3 = 7
      nblks_4 = 3

      ! Block sizes in each dimension
      ALLOCATE (size_1(nblks_1), size_2(nblks_2), size_3(nblks_3), size_4(nblks_4))

      size_1(:) = [5, 9]
      size_2(:) = [6, 2, 5, 12, 3, 1, 7, 2, 5, 17, 9, 3, 4]
      size_3(:) = [2, 7, 3, 8, 5, 15, 1]
      size_4(:) = [12, 5, 3]

      ! Number of non-zero blocks
      nblks_alloc = 19
      ALLOCATE (blk_ind_1(nblks_alloc), blk_ind_2(nblks_alloc), blk_ind_3(nblks_alloc), blk_ind_4(nblks_alloc))

      ! Indices of non-zero blocks (s.t. index of ith block is [blk_ind_1(i), blk_ind_2(i), ...])
      blk_ind_1(:) = [1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1, 2, 2, 2, 2, 2, 2, 2,  2] !&
      blk_ind_2(:) = [2, 2, 3, 4, 7, 7, 10, 11, 11, 12, 12, 1, 1, 3, 5, 6, 6, 9, 12] !&
      blk_ind_3(:) = [1, 4, 6, 3, 1, 4,  2,  5,  7,  3,  3, 1, 4, 7, 6, 4, 5, 2,  3] !&
      blk_ind_4(:) = [3, 2, 3, 1, 1, 2,  1,  3,  2,  2,  3, 1, 3, 2, 1, 1, 3, 2,  2] !&

      ! Test tensor formats
      CALL dbcsr_t_test_formats(ndims, mp_comm, default_output_unit, verbose, &
                                blk_size_1=size_1, blk_size_2=size_2, blk_size_3=size_3, blk_size_4=size_4, &
                                blk_ind_1=blk_ind_1, blk_ind_2=blk_ind_2, blk_ind_3=blk_ind_3, blk_ind_4=blk_ind_4)

      DEALLOCATE (size_1, size_2, size_3, size_4)
      DEALLOCATE (blk_ind_1, blk_ind_2, blk_ind_3, blk_ind_4)

   ENDIF
   IF (test_contraction) THEN

!--------------------------------------------------------------------------------------------------!
! Preparations for tensor contraction tests                                                        !
!--------------------------------------------------------------------------------------------------!

      nblks_1 = 4
      nblks_2 = 11
      nblks_3 = 9
      nblks_4 = 5
      nblks_5 = 3

      ! Block sizes in each dimension
      ALLOCATE (size_1(nblks_1), size_2(nblks_2), size_3(nblks_3), size_4(nblks_4), size_5(nblks_5))

      size_1(:) = [3, 9, 12, 1]
      size_2(:) = [4, 2, 3, 1, 9, 2, 32, 10, 5, 8, 7]
      size_3(:) = [7, 3, 8, 7, 9, 5, 10, 23, 2]
      size_4(:) = [8, 1, 4, 13, 6]
      size_5(:) = [4, 2, 22]

      nblks_alloc_1 = 32
      ALLOCATE (blk_ind_1_1(nblks_alloc_1), blk_ind_2_1(nblks_alloc_1), blk_ind_3_1(nblks_alloc_1))

      blk_ind_1_1(:) = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, & !&
                        1, 2, 2, 2, 2, 2, 2, 2, 3, 3, & !&
                        3, 3, 3, 3, 3, 3, 3, 4, 4, 4, & !&
                        4, 4] !&

      blk_ind_2_1(:) = [ 3, 5, 5, 5, 6,  6,  7,  8, 10, 11, & !&
                        11, 1, 1, 4, 7,  7,  9, 10 , 2,  2, & !&
                         5, 6, 8, 8, 9, 11, 11,  2 , 4,  5, & !&
                         5, 8] !&

      blk_ind_3_1(:) = [7, 3, 5, 9, 6, 8, 2, 8, 3, 2, & !&
                        3, 1, 4, 6, 2, 7, 5, 8, 3, 7, & !&
                        1, 4, 3, 7, 8, 5, 8, 9, 6, 1, & !&
                        2, 7] !&

      nblks_alloc_2 = 12
      ALLOCATE (blk_ind_3_2(nblks_alloc_2), blk_ind_4_2(nblks_alloc_2))

      blk_ind_3_2(:) = [1, 1, 2, 2, 2, 4, 4, 5, 5, 6, & !&
                        8, 8] !&
      blk_ind_4_2(:) = [2, 3, 2, 4, 5, 3, 5, 1, 3, 3, & !&
                        1, 4] !&

      nblks_alloc_3 = 5
      ALLOCATE (blk_ind_1_3(nblks_alloc_3), blk_ind_2_3(nblks_alloc_3), blk_ind_4_3(nblks_alloc_3))

      blk_ind_1_3(:) = [1, 1, 2, 4, 4]
      blk_ind_2_3(:) = [2, 6, 6, 7, 9]
      blk_ind_4_3(:) = [1, 3, 4, 4, 5]

      nblks_alloc_4 = 36
      ALLOCATE (blk_ind_1_4(nblks_alloc_4))
      ALLOCATE (blk_ind_2_4(nblks_alloc_4))
      ALLOCATE (blk_ind_4_4(nblks_alloc_4))
      ALLOCATE (blk_ind_5_4(nblks_alloc_4))

      blk_ind_1_4(:) = [ 1, 1, 1, 1, 1, 2, 2, 2,  2,  2, & !&
                         2, 2, 2, 2, 2, 2, 2, 2,  2,  2, & !&
                         3, 3, 3, 3, 3, 3, 3, 3,  3,  3, & !&
                         4, 4, 4, 4, 4, 4] !&

      blk_ind_2_4(:) = [ 1, 3, 4, 6, 10,  2, 2, 4,  5,  5, & !&
                         6, 6, 6, 7,  7,  9, 9, 9, 10, 11, & !&
                         1, 3, 3, 4,  5,  6, 8, 9, 11, 11, & !&
                         1, 3, 4, 6, 10, 11] !&

      blk_ind_4_4(:) = [ 3, 5, 2, 3,  2,  3, 5, 1,  1,  4, & !&
                         2, 3, 4, 1,  4,  3, 4, 4,  2,  1, & !&
                         3, 1, 1, 3,  4,  3, 4, 2,  2,  3, & !&
                         1, 1, 3, 2,  5,  5] !&

      blk_ind_5_4(:) = [ 1, 3, 2, 1,  1,  2, 3,  1,  3, 1, & !&
                         2, 3, 2, 1,  3,  2, 3,  2,  1, 2, & !&
                         3, 1, 2, 3,  2,  2, 2,  3,  1, 2, & !&
                         1, 3, 2, 1,  3,  2] !&

      nblks_alloc_5 = 8

      ALLOCATE (blk_ind_3_5(nblks_alloc_5), blk_ind_4_5(nblks_alloc_5), blk_ind_5_5(nblks_alloc_5))

      blk_ind_3_5(:) = [2, 4, 5, 5, 5, 6, 6, 8]
      blk_ind_4_5(:) = [3, 2, 1, 1, 3, 2, 4, 5]
      blk_ind_5_5(:) = [3, 2, 1, 2, 3, 2, 1, 1]

      pdims_4d(:) = 0; pdims_3d(:) = 0; pdims_2d(:) = 0
      CALL dbcsr_t_pgrid_create(mp_comm, pdims_4d, pgrid_4d)
      CALL dbcsr_t_pgrid_create(mp_comm, pdims_3d, pgrid_3d)
      CALL dbcsr_t_pgrid_create(mp_comm, pdims_2d, pgrid_2d)

      CALL dbcsr_t_random_dist(dist1_1, nblks_1, pdims_3d(1), mp_comm)
      CALL dbcsr_t_random_dist(dist1_2, nblks_2, pdims_3d(2), mp_comm)
      CALL dbcsr_t_random_dist(dist1_3, nblks_3, pdims_3d(3), mp_comm)

      CALL dbcsr_t_random_dist(dist2_1, nblks_3, pdims_2d(1), mp_comm)
      CALL dbcsr_t_random_dist(dist2_2, nblks_4, pdims_2d(2), mp_comm)

      CALL dbcsr_t_random_dist(dist3_1, nblks_1, pdims_3d(1), mp_comm)
      CALL dbcsr_t_random_dist(dist3_2, nblks_2, pdims_3d(2), mp_comm)
      CALL dbcsr_t_random_dist(dist3_3, nblks_4, pdims_3d(3), mp_comm)

      CALL dbcsr_t_random_dist(dist4_1, nblks_1, pdims_4d(1), mp_comm)
      CALL dbcsr_t_random_dist(dist4_2, nblks_2, pdims_4d(2), mp_comm)
      CALL dbcsr_t_random_dist(dist4_3, nblks_4, pdims_4d(3), mp_comm)
      CALL dbcsr_t_random_dist(dist4_4, nblks_5, pdims_4d(4), mp_comm)

      CALL dbcsr_t_random_dist(dist5_1, nblks_3, pdims_3d(1), mp_comm)
      CALL dbcsr_t_random_dist(dist5_2, nblks_4, pdims_3d(2), mp_comm)
      CALL dbcsr_t_random_dist(dist5_3, nblks_5, pdims_3d(3), mp_comm)

!--------------------------------------------------------------------------------------------------!
! Test 4: Testing tensor contraction (12|3)x(3|4)=(12|4)                                           !
!--------------------------------------------------------------------------------------------------!

      ALLOCATE (map11(2), map12(1), map21(1), map22(1), map31(2), map32(1))
      map11(:) = [1, 2]
      map12(:) = [3]
      map21(:) = [1]
      map22(:) = [2]
      map31(:) = [1, 2]
      map32(:) = [3]

      CALL dbcsr_t_distribution_new(dist1, pgrid_3d, map11, map12, dist1_1, dist1_2, dist1_3)
      CALL dbcsr_t_distribution_new(dist2, pgrid_2d, map21, map22, dist2_1, dist2_2)
      CALL dbcsr_t_distribution_new(dist3, pgrid_3d, map31, map32, dist3_1, dist3_2, dist3_3)

      CALL dbcsr_t_create(tensor_A, "(12|3)", dist1, map11, map12, dbcsr_type_real_8, &
                          size_1, size_2, size_3)
      CALL dbcsr_t_create(tensor_B, "(3|4)", dist2, map21, map22, dbcsr_type_real_8, &
                          size_3, size_4)
      CALL dbcsr_t_create(tensor_C, "(12|4)", dist3, map31, map32, dbcsr_type_real_8, &
                          size_1, size_2, size_4)

      CALL dbcsr_t_setup_test_tensor(tensor_A, mp_comm, .FALSE., blk_ind_1_1, blk_ind_2_1, blk_ind_3_1)
      CALL dbcsr_t_setup_test_tensor(tensor_B, mp_comm, .FALSE., blk_ind_3_2, blk_ind_4_2)
      CALL dbcsr_t_setup_test_tensor(tensor_C, mp_comm, .FALSE., blk_ind_1_3, blk_ind_2_3, blk_ind_4_3)

      CALL dbcsr_t_contract_test(dbcsr_scalar(0.9_real_8), tensor_A, tensor_B, dbcsr_scalar(0.1_real_8), tensor_C, &
                                 [3], [2, 1], &
                                 [1], [2], &
                                 [2, 1], [3], &
                                 default_output_unit, &
                                 log_verbose=verbose, &
                                 write_int=.TRUE.)

      DEALLOCATE (map11, map12, map21, map22, map31, map32)

      CALL dbcsr_t_destroy(tensor_A)
      CALL dbcsr_t_destroy(tensor_B)
      CALL dbcsr_t_destroy(tensor_C)
      CALL dbcsr_t_distribution_destroy(dist1)
      CALL dbcsr_t_distribution_destroy(dist2)
      CALL dbcsr_t_distribution_destroy(dist3)

!--------------------------------------------------------------------------------------------------!
! Test 5: Testing tensor contraction (2|31)x(4|3)=(24|1)                                           !
!--------------------------------------------------------------------------------------------------!

      ALLOCATE (map11(1), map12(2), map21(1), map22(1), map31(2), map32(1))
      map11(:) = [2]
      map12(:) = [3, 1]
      map21(:) = [2]
      map22(:) = [1]
      map31(:) = [2, 3]
      map32(:) = [1]

      CALL dbcsr_t_distribution_new(dist1, pgrid_3d, map11, map12, dist1_1, dist1_2, dist1_3)
      CALL dbcsr_t_distribution_new(dist2, pgrid_2d, map21, map22, dist2_1, dist2_2)
      CALL dbcsr_t_distribution_new(dist3, pgrid_3d, map31, map32, dist3_1, dist3_2, dist3_3)

      CALL dbcsr_t_create(tensor_A, "(2|31)", dist1, map11, map12, dbcsr_type_real_8, &
                          size_1, size_2, size_3)
      CALL dbcsr_t_create(tensor_B, "(4|3)", dist2, map21, map22, dbcsr_type_real_8, &
                          size_3, size_4)
      CALL dbcsr_t_create(tensor_C, "(24|1)", dist3, map31, map32, dbcsr_type_real_8, &
                          size_1, size_2, size_4)

      CALL dbcsr_t_setup_test_tensor(tensor_A, mp_comm, .FALSE., blk_ind_1_1, blk_ind_2_1, blk_ind_3_1)
      CALL dbcsr_t_setup_test_tensor(tensor_B, mp_comm, .FALSE., blk_ind_3_2, blk_ind_4_2)
      CALL dbcsr_t_setup_test_tensor(tensor_C, mp_comm, .FALSE., blk_ind_1_3, blk_ind_2_3, blk_ind_4_3)

      CALL dbcsr_t_contract_test(dbcsr_scalar(0.9_real_8), tensor_A, tensor_B, dbcsr_scalar(0.1_real_8), tensor_C, &
                                 [3], [1, 2], &
                                 [1], [2], &
                                 [1, 2], [3], &
                                 default_output_unit, &
                                 log_verbose=verbose, &
                                 write_int=.TRUE.)

      DEALLOCATE (map11, map12, map21, map22, map31, map32)

      CALL dbcsr_t_destroy(tensor_A)
      CALL dbcsr_t_destroy(tensor_B)
      CALL dbcsr_t_destroy(tensor_C)
      CALL dbcsr_t_distribution_destroy(dist1)
      CALL dbcsr_t_distribution_destroy(dist2)
      CALL dbcsr_t_distribution_destroy(dist3)

!-------------------------------------------------------------------------------------------------!
! Test 6: Testing tensor contraction (4|3)x(1|32)=(24|1)                                           !
!-------------------------------------------------------------------------------------------------!

      ALLOCATE (map11(1), map12(2), map21(1), map22(1), map31(2), map32(1))
      map11(:) = [1]
      map12(:) = [3, 2]
      map21(:) = [2]
      map22(:) = [1]
      map31(:) = [2, 3]
      map32(:) = [1]

      CALL dbcsr_t_distribution_new(dist1, pgrid_3d, map11, map12, dist1_1, dist1_2, dist1_3)
      CALL dbcsr_t_distribution_new(dist2, pgrid_2d, map21, map22, dist2_1, dist2_2)
      CALL dbcsr_t_distribution_new(dist3, pgrid_3d, map31, map32, dist3_1, dist3_2, dist3_3)

      CALL dbcsr_t_create(tensor_A, "(1|32)", dist1, map11, map12, dbcsr_type_real_8, &
                          size_1, size_2, size_3)
      CALL dbcsr_t_create(tensor_B, "(4|3)", dist2, map21, map22, dbcsr_type_real_8, &
                          size_3, size_4)
      CALL dbcsr_t_create(tensor_C, "(24|1)", dist3, map31, map32, dbcsr_type_real_8, &
                          size_1, size_2, size_4)

      CALL dbcsr_t_setup_test_tensor(tensor_A, mp_comm, .FALSE., blk_ind_1_1, blk_ind_2_1, blk_ind_3_1)
      CALL dbcsr_t_setup_test_tensor(tensor_B, mp_comm, .FALSE., blk_ind_3_2, blk_ind_4_2)
      CALL dbcsr_t_setup_test_tensor(tensor_C, mp_comm, .FALSE., blk_ind_1_3, blk_ind_2_3, blk_ind_4_3)

      ALLOCATE(bounds_t(dbcsr_t_ndims(tensor_B)))
      CALL dbcsr_t_get_info(tensor_B, nfull_total=bounds_t)

      ALLOCATE(bounds(2,1))
      bounds(1, 1) = 1
      bounds(2, 1) = bounds_t(1) - 21

      CALL dbcsr_t_contract_test(dbcsr_scalar(0.9_real_8), tensor_B, tensor_A, dbcsr_scalar(0.1_real_8), tensor_C, &
                                 [1], [2], &
                                 [3], [1, 2], &
                                 [3], [1, 2], &
                                 default_output_unit, &
                                 bounds_1=bounds, &
                                 log_verbose=verbose, &
                                 write_int=.TRUE.)

      DEALLOCATE (map11, map12, map21, map22, map31, map32, bounds_t, bounds)

      CALL dbcsr_t_destroy(tensor_A)
      CALL dbcsr_t_destroy(tensor_B)
      CALL dbcsr_t_destroy(tensor_C)
      CALL dbcsr_t_distribution_destroy(dist1)
      CALL dbcsr_t_distribution_destroy(dist2)
      CALL dbcsr_t_distribution_destroy(dist3)

!-------------------------------------------------------------------------------------------------!
! Test 7: Testing tensor contraction (1|24)x(3|4)=(21|3)                                          !
!-------------------------------------------------------------------------------------------------!

      ALLOCATE (map11(2), map12(1), map21(1), map22(1), map31(1), map32(2))
      map11(:) = [2, 1]
      map12(:) = [3]
      map21(:) = [1]
      map22(:) = [2]
      map31(:) = [1]
      map32(:) = [2, 3]

      CALL dbcsr_t_distribution_new(dist1, pgrid_3d, map11, map12, dist1_1, dist1_2, dist1_3)
      CALL dbcsr_t_distribution_new(dist2, pgrid_2d, map21, map22, dist2_1, dist2_2)
      CALL dbcsr_t_distribution_new(dist3, pgrid_3d, map31, map32, dist3_1, dist3_2, dist3_3)

      CALL dbcsr_t_create(tensor_A, "(21|3)", dist1, map11, map12, dbcsr_type_real_8, &
                          size_1, size_2, size_3)
      CALL dbcsr_t_create(tensor_B, "(3|4)", dist2, map21, map22, dbcsr_type_real_8, &
                          size_3, size_4)
      CALL dbcsr_t_create(tensor_C, "(1|24)", dist3, map31, map32, dbcsr_type_real_8, &
                          size_1, size_2, size_4)

      CALL dbcsr_t_setup_test_tensor(tensor_A, mp_comm, .FALSE., blk_ind_1_1, blk_ind_2_1, blk_ind_3_1)
      CALL dbcsr_t_setup_test_tensor(tensor_B, mp_comm, .FALSE., blk_ind_3_2, blk_ind_4_2)
      CALL dbcsr_t_setup_test_tensor(tensor_C, mp_comm, .FALSE., blk_ind_1_3, blk_ind_2_3, blk_ind_4_3)

      ALLOCATE(bounds_t(dbcsr_t_ndims(tensor_C)))
      CALL dbcsr_t_get_info(tensor_C, nfull_total=bounds_t)

      ALLOCATE(bounds(2,2))
      bounds(1, 1) = 4
      bounds(2, 1) = bounds_t(1)
      bounds(1, 2) = 13
      bounds(2, 2) = bounds_t(2) - 10
      DEALLOCATE(bounds_t)

      CALL dbcsr_t_contract_test(dbcsr_scalar(0.2_real_8), tensor_C, tensor_B, dbcsr_scalar(0.8_real_8), tensor_A, &
                                 [3], [1, 2], &
                                 [2], [1], &
                                 [1, 2], [3], &
                                 default_output_unit, &
                                 bounds_2=bounds, &
                                 log_verbose=verbose, &
                                 write_int=.TRUE.)

      DEALLOCATE (map11, map12, map21, map22, map31, map32, bounds)

      CALL dbcsr_t_destroy(tensor_A)
      CALL dbcsr_t_destroy(tensor_B)
      CALL dbcsr_t_destroy(tensor_C)
      CALL dbcsr_t_distribution_destroy(dist1)
      CALL dbcsr_t_distribution_destroy(dist2)
      CALL dbcsr_t_distribution_destroy(dist3)

!-------------------------------------------------------------------------------------------------!
! Test 8: Testing tensor contraction (12|3)x(12|45)=(3|45)
!-------------------------------------------------------------------------------------------------!

      ALLOCATE (map11(2), map12(1), map21(2), map22(2), map31(1), map32(2))
      map11(:) = [1, 2]
      map12(:) = [3]
      map21(:) = [1, 2]
      map22(:) = [3, 4]
      map31(:) = [1]
      map32(:) = [2, 3]

      CALL dbcsr_t_distribution_new(dist1, pgrid_3d, map11, map12, dist1_1, dist1_2, dist1_3)
      CALL dbcsr_t_distribution_new(dist2, pgrid_4d, map21, map22, dist4_1, dist4_2, dist4_3, dist4_4)
      CALL dbcsr_t_distribution_new(dist3, pgrid_3d, map31, map32, dist5_1, dist5_2, dist5_3)

      CALL dbcsr_t_create(tensor_A, "(12|3)", dist1, map11, map12, dbcsr_type_real_8, &
                          size_1, size_2, size_3)
      CALL dbcsr_t_create(tensor_B, "(12|45)", dist2, map21, map22, dbcsr_type_real_8, &
                          size_1, size_2, size_4, size_5)
      CALL dbcsr_t_create(tensor_C, "(3|45)", dist3, map31, map32, dbcsr_type_real_8, &
                          size_3, size_4, size_5)

      CALL dbcsr_t_setup_test_tensor(tensor_A, mp_comm, .FALSE., blk_ind_1_1, blk_ind_2_1, blk_ind_3_1)
      CALL dbcsr_t_setup_test_tensor(tensor_B, mp_comm, .FALSE., blk_ind_1_4, blk_ind_2_4, blk_ind_4_4, blk_ind_5_4)
      CALL dbcsr_t_setup_test_tensor(tensor_C, mp_comm, .FALSE., blk_ind_3_5, blk_ind_4_5, blk_ind_5_5)

      ALLOCATE(bounds_t(dbcsr_t_ndims(tensor_A)))
      CALL dbcsr_t_get_info(tensor_A, nfull_total=bounds_t)
      ALLOCATE(bounds_1(2, 2))
      bounds_1(1, 1) = 7
      bounds_1(2, 1) = bounds_t(2) - 17
      bounds_1(1, 2) = 8
      bounds_1(2, 2) = bounds_t(1)
      DEALLOCATE(bounds_t)

      ALLOCATE(bounds_t(dbcsr_t_ndims(tensor_B)))
      CALL dbcsr_t_get_info(tensor_B, nfull_total=bounds_t)
      ALLOCATE(bounds_2(2, 2))
      bounds_2(1, 1) = 1
      bounds_2(2, 1) = bounds_t(3)
      bounds_2(1, 2) = 1
      bounds_2(2, 2) = bounds_t(4) - 18
      DEALLOCATE(bounds_t)

      CALL dbcsr_t_contract_test(dbcsr_scalar(0.2_real_8), tensor_A, tensor_B, dbcsr_scalar(0.8_real_8), tensor_C, &
                                 [2, 1], [3], &
                                 [2, 1], [3, 4], &
                                 [1], [2, 3], &
                                 default_output_unit, &
                                 bounds_1=bounds_1,&
                                 bounds_3=bounds_2, &
                                 log_verbose=verbose, &
                                 write_int=.TRUE.)

      DEALLOCATE (map11, map12, map21, map22, map31, map32, bounds_1, bounds_2)

      CALL dbcsr_t_destroy(tensor_A)
      CALL dbcsr_t_destroy(tensor_B)
      CALL dbcsr_t_destroy(tensor_C)
      CALL dbcsr_t_distribution_destroy(dist1)
      CALL dbcsr_t_distribution_destroy(dist2)
      CALL dbcsr_t_distribution_destroy(dist3)

!-------------------------------------------------------------------------------------------------!
! Test 9: Testing tensor contraction (3|21)x(12|45)=(3|45)
!-------------------------------------------------------------------------------------------------!

      ALLOCATE (map11(1), map12(2), map21(2), map22(2), map31(1), map32(2))
      map11(:) = [3]
      map12(:) = [2, 1]
      map21(:) = [1, 2]
      map22(:) = [3, 4]
      map31(:) = [1]
      map32(:) = [2, 3]

      CALL dbcsr_t_distribution_new(dist1, pgrid_3d, map11, map12, dist1_1, dist1_2, dist1_3)
      CALL dbcsr_t_distribution_new(dist2, pgrid_4d, map21, map22, dist4_1, dist4_2, dist4_3, dist4_4)
      CALL dbcsr_t_distribution_new(dist3, pgrid_3d, map31, map32, dist5_1, dist5_2, dist5_3)

      CALL dbcsr_t_create(tensor_A, "(3|21)", dist1, map11, map12, dbcsr_type_real_8, &
                          size_1, size_2, size_3)
      CALL dbcsr_t_create(tensor_B, "(12|45)", dist2, map21, map22, dbcsr_type_real_8, &
                          size_1, size_2, size_4, size_5)
      CALL dbcsr_t_create(tensor_C, "(3|45)", dist3, map31, map32, dbcsr_type_real_8, &
                          size_3, size_4, size_5)

      CALL dbcsr_t_setup_test_tensor(tensor_A, mp_comm, .FALSE., blk_ind_1_1, blk_ind_2_1, blk_ind_3_1)
      CALL dbcsr_t_setup_test_tensor(tensor_B, mp_comm, .FALSE., blk_ind_1_4, blk_ind_2_4, blk_ind_4_4, blk_ind_5_4)
      CALL dbcsr_t_setup_test_tensor(tensor_C, mp_comm, .FALSE., blk_ind_3_5, blk_ind_4_5, blk_ind_5_5)

      CALL dbcsr_t_contract_test(dbcsr_scalar(0.2_real_8), tensor_A, tensor_B, dbcsr_scalar(0.8_real_8), tensor_C, &
                                 [2, 1], [3], &
                                 [2, 1], [3, 4], &
                                 [1], [2, 3], &
                                 default_output_unit, &
                                 log_verbose=verbose, &
                                 write_int=.TRUE.)

      DEALLOCATE (map11, map12, map21, map22, map31, map32)

      CALL dbcsr_t_destroy(tensor_A)
      CALL dbcsr_t_destroy(tensor_B)
      CALL dbcsr_t_destroy(tensor_C)
      CALL dbcsr_t_distribution_destroy(dist1)
      CALL dbcsr_t_distribution_destroy(dist2)
      CALL dbcsr_t_distribution_destroy(dist3)

!-------------------------------------------------------------------------------------------------!
! Test 10: Testing tensor contraction (13|2)x(54|21)=(3|45)
!-------------------------------------------------------------------------------------------------!

      ALLOCATE (map11(2), map12(1), map21(2), map22(2), map31(1), map32(2))
      map11(:) = [1, 3]
      map12(:) = [2]
      map21(:) = [4, 3]
      map22(:) = [2, 1]
      map31(:) = [1]
      map32(:) = [2, 3]

      CALL dbcsr_t_distribution_new(dist1, pgrid_3d, map11, map12, dist1_1, dist1_2, dist1_3)
      CALL dbcsr_t_distribution_new(dist2, pgrid_4d, map21, map22, dist4_1, dist4_2, dist4_3, dist4_4)
      CALL dbcsr_t_distribution_new(dist3, pgrid_3d, map31, map32, dist5_1, dist5_2, dist5_3)

      CALL dbcsr_t_create(tensor_A, "(13|2)", dist1, map11, map12, dbcsr_type_real_8, &
                          size_1, size_2, size_3)
      CALL dbcsr_t_create(tensor_B, "(54|21)", dist2, map21, map22, dbcsr_type_real_8, &
                          size_1, size_2, size_4, size_5)
      CALL dbcsr_t_create(tensor_C, "(3|45)", dist3, map31, map32, dbcsr_type_real_8, &
                          size_3, size_4, size_5)

      CALL dbcsr_t_setup_test_tensor(tensor_A, mp_comm, .FALSE., blk_ind_1_1, blk_ind_2_1, blk_ind_3_1)
      CALL dbcsr_t_setup_test_tensor(tensor_B, mp_comm, .FALSE., blk_ind_1_4, blk_ind_2_4, blk_ind_4_4, blk_ind_5_4)
      CALL dbcsr_t_setup_test_tensor(tensor_C, mp_comm, .FALSE., blk_ind_3_5, blk_ind_4_5, blk_ind_5_5)

      CALL dbcsr_t_contract_test(dbcsr_scalar(0.2_real_8), tensor_A, tensor_B, dbcsr_scalar(0.8_real_8), tensor_C, &
                                 [1, 2], [3], &
                                 [1, 2], [3, 4], &
                                 [1], [2, 3], &
                                 default_output_unit, &
                                 log_verbose=verbose, &
                                 write_int=.TRUE.)

      DEALLOCATE (map11, map12, map21, map22, map31, map32)

      CALL dbcsr_t_destroy(tensor_A)
      CALL dbcsr_t_destroy(tensor_B)
      CALL dbcsr_t_destroy(tensor_C)
      CALL dbcsr_t_distribution_destroy(dist1)
      CALL dbcsr_t_distribution_destroy(dist2)
      CALL dbcsr_t_distribution_destroy(dist3)

!-------------------------------------------------------------------------------------------------!
! Test 10: Testing tensor contraction (54|21)x(2|31)=(43|5)
!-------------------------------------------------------------------------------------------------!

      ALLOCATE (map11(1), map12(2), map21(2), map22(2), map31(2), map32(1))
      map11(:) = [2]
      map12(:) = [3, 1]
      map21(:) = [4, 3]
      map22(:) = [2, 1]
      map31(:) = [2, 1]
      map32(:) = [3]

      CALL dbcsr_t_distribution_new(dist1, pgrid_3d, map11, map12, dist1_1, dist1_2, dist1_3)
      CALL dbcsr_t_distribution_new(dist2, pgrid_4d, map21, map22, dist4_1, dist4_2, dist4_3, dist4_4)
      CALL dbcsr_t_distribution_new(dist3, pgrid_3d, map31, map32, dist5_1, dist5_2, dist5_3)

      CALL dbcsr_t_create(tensor_A, "(2|31)", dist1, map11, map12, dbcsr_type_real_8, &
                          size_1, size_2, size_3)
      CALL dbcsr_t_create(tensor_B, "(54|21)", dist2, map21, map22, dbcsr_type_real_8, &
                          size_1, size_2, size_4, size_5)
      CALL dbcsr_t_create(tensor_C, "(43|5)", dist3, map31, map32, dbcsr_type_real_8, &
                          size_3, size_4, size_5)

      CALL dbcsr_t_setup_test_tensor(tensor_A, mp_comm, .FALSE., blk_ind_1_1, blk_ind_2_1, blk_ind_3_1)
      CALL dbcsr_t_setup_test_tensor(tensor_B, mp_comm, .FALSE., blk_ind_1_4, blk_ind_2_4, blk_ind_4_4, blk_ind_5_4)
      CALL dbcsr_t_setup_test_tensor(tensor_C, mp_comm, .FALSE., blk_ind_3_5, blk_ind_4_5, blk_ind_5_5)

      CALL dbcsr_t_contract_test(dbcsr_scalar(0.2_real_8), tensor_B, tensor_A, dbcsr_scalar(0.8_real_8), tensor_C, &
                                 [2, 1], [4, 3], &
                                 [2, 1], [3], &
                                 [3, 2], [1], &
                                 default_output_unit, &
                                 log_verbose=verbose, &
                                 write_int=.TRUE.)

      DEALLOCATE (map11, map12, map21, map22, map31, map32)

      CALL dbcsr_t_destroy(tensor_A)
      CALL dbcsr_t_destroy(tensor_B)
      CALL dbcsr_t_destroy(tensor_C)
      CALL dbcsr_t_distribution_destroy(dist1)
      CALL dbcsr_t_distribution_destroy(dist2)
      CALL dbcsr_t_distribution_destroy(dist3)

!-------------------------------------------------------------------------------------------------!
! Test 11: Testing tensor contraction (241|5)x(31|2)=(5|43)
!-------------------------------------------------------------------------------------------------!

      ALLOCATE (map11(2), map12(1), map21(3), map22(1), map31(1), map32(2))
      map11(:) = [3, 1]
      map12(:) = [2]
      map21(:) = [2, 3, 1]
      map22(:) = [4]
      map31(:) = [3]
      map32(:) = [2, 1]

      CALL dbcsr_t_distribution_new(dist1, pgrid_3d, map11, map12, dist1_1, dist1_2, dist1_3)
      CALL dbcsr_t_distribution_new(dist2, pgrid_4d, map21, map22, dist4_1, dist4_2, dist4_3, dist4_4)
      CALL dbcsr_t_distribution_new(dist3, pgrid_3d, map31, map32, dist5_1, dist5_2, dist5_3)

      CALL dbcsr_t_create(tensor_A, "(31|2)", dist1, map11, map12, dbcsr_type_real_8, &
                          size_1, size_2, size_3)
      CALL dbcsr_t_create(tensor_B, "(241|5)", dist2, map21, map22, dbcsr_type_real_8, &
                          size_1, size_2, size_4, size_5)
      CALL dbcsr_t_create(tensor_C, "(5|43)", dist3, map31, map32, dbcsr_type_real_8, &
                          size_3, size_4, size_5)

      CALL dbcsr_t_setup_test_tensor(tensor_A, mp_comm, .FALSE., blk_ind_1_1, blk_ind_2_1, blk_ind_3_1)
      CALL dbcsr_t_setup_test_tensor(tensor_B, mp_comm, .FALSE., blk_ind_1_4, blk_ind_2_4, blk_ind_4_4, blk_ind_5_4)
      CALL dbcsr_t_setup_test_tensor(tensor_C, mp_comm, .FALSE., blk_ind_3_5, blk_ind_4_5, blk_ind_5_5)

      CALL dbcsr_t_contract_test(dbcsr_scalar(0.6_real_8), tensor_B, tensor_A, dbcsr_scalar(0.4_real_8), tensor_C, &
                                 [2, 1], [3, 4], &
                                 [2, 1], [3], &
                                 [2, 3], [1], &
                                 default_output_unit, &
                                 log_verbose=verbose, &
                                 write_int=.TRUE.)

      DEALLOCATE (map11, map12, map21, map22, map31, map32)

      CALL dbcsr_t_destroy(tensor_A)
      CALL dbcsr_t_destroy(tensor_B)
      CALL dbcsr_t_destroy(tensor_C)
      CALL dbcsr_t_distribution_destroy(dist1)
      CALL dbcsr_t_distribution_destroy(dist2)
      CALL dbcsr_t_distribution_destroy(dist3)

!-------------------------------------------------------------------------------------------------!
! Test 12: Testing tensor contraction (34|5)x(12|3)=(14|25)
!-------------------------------------------------------------------------------------------------!

      ALLOCATE (map11(2), map12(1), map21(2), map22(2), map31(2), map32(1))
      map11(:) = [1, 2]
      map12(:) = [3]
      map21(:) = [1, 3]
      map22(:) = [2, 4]
      map31(:) = [1, 2]
      map32(:) = [3]

      CALL dbcsr_t_distribution_new(dist1, pgrid_3d, map11, map12, dist1_1, dist1_2, dist1_3)
      CALL dbcsr_t_distribution_new(dist2, pgrid_4d, map21, map22, dist4_1, dist4_2, dist4_3, dist4_4)
      CALL dbcsr_t_distribution_new(dist3, pgrid_3d, map31, map32, dist5_1, dist5_2, dist5_3)

      CALL dbcsr_t_create(tensor_A, "(12|3)", dist1, map11, map12, dbcsr_type_real_8, &
                          size_1, size_2, size_3)
      CALL dbcsr_t_create(tensor_B, "(14|25)", dist2, map21, map22, dbcsr_type_real_8, &
                          size_1, size_2, size_4, size_5)
      CALL dbcsr_t_create(tensor_C, "(34|5)", dist3, map31, map32, dbcsr_type_real_8, &
                          size_3, size_4, size_5)

      CALL dbcsr_t_setup_test_tensor(tensor_A, mp_comm, .FALSE., blk_ind_1_1, blk_ind_2_1, blk_ind_3_1)
      CALL dbcsr_t_setup_test_tensor(tensor_B, mp_comm, .FALSE., blk_ind_1_4, blk_ind_2_4, blk_ind_4_4, blk_ind_5_4)
      CALL dbcsr_t_setup_test_tensor(tensor_C, mp_comm, .FALSE., blk_ind_3_5, blk_ind_4_5, blk_ind_5_5)

      CALL dbcsr_t_contract_test(dbcsr_scalar(0.2_real_8), tensor_C, tensor_A, dbcsr_scalar(0.8_real_8), tensor_B, &
                                 [1], [2, 3], &
                                 [3], [1, 2], &
                                 [3, 4], [1, 2], &
                                 default_output_unit, &
                                 log_verbose=verbose, &
                                 write_int=.TRUE.)

      DEALLOCATE (map11, map12, map21, map22, map31, map32)

      CALL dbcsr_t_destroy(tensor_A)
      CALL dbcsr_t_destroy(tensor_B)
      CALL dbcsr_t_destroy(tensor_C)
      CALL dbcsr_t_distribution_destroy(dist1)
      CALL dbcsr_t_distribution_destroy(dist2)
      CALL dbcsr_t_distribution_destroy(dist3)

!--------------------------------------------------------------------------------------------------!
! Cleanup for tensor contraction tests                                                             !
!--------------------------------------------------------------------------------------------------!

      DEALLOCATE (size_1, size_2, size_3, size_4, size_5)
      DEALLOCATE (blk_ind_1_1, blk_ind_2_1, blk_ind_3_1)
      DEALLOCATE (blk_ind_3_2, blk_ind_4_2)
      DEALLOCATE (blk_ind_1_3, blk_ind_2_3, blk_ind_4_3)
      DEALLOCATE (blk_ind_1_4, blk_ind_2_4, blk_ind_4_4, blk_ind_5_4)
      DEALLOCATE (blk_ind_3_5, blk_ind_4_5, blk_ind_5_5)
      DEALLOCATE (dist1_1, dist1_2, dist1_3, dist2_1, dist2_2, dist3_1, dist3_2, dist3_3)
      CALL dbcsr_t_pgrid_destroy(pgrid_3d)
      CALL dbcsr_t_pgrid_destroy(pgrid_2d)
      CALL dbcsr_t_pgrid_destroy(pgrid_4d)

   ENDIF

!--------------------------------------------------------------------------------------------------!
! End tests                                                                                        !
!--------------------------------------------------------------------------------------------------!

   call dbcsr_print_statistics(.true.)

   ! finalize libdbcsr
   CALL dbcsr_finalize_lib()

   !
   ! finalize mpi
   CALL mp_world_finalize()

END PROGRAM
