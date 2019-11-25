!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_performance_multiply
   !! Performance for DBCSR multiply
   USE dbcsr_data_methods, ONLY: dbcsr_scalar, &
                                 dbcsr_scalar_negative, &
                                 dbcsr_scalar_one
   USE dbcsr_dist_methods, ONLY: dbcsr_distribution_col_dist, &
                                 dbcsr_distribution_new, &
                                 dbcsr_distribution_release, &
                                 dbcsr_distribution_row_dist
   USE dbcsr_dist_operations, ONLY: dbcsr_dist_bin
   USE dbcsr_dist_util, ONLY: dbcsr_checksum
   USE dbcsr_io, ONLY: dbcsr_print
   USE dbcsr_kinds, ONLY: int_8, &
                          real_4, &
                          real_8
   USE dbcsr_machine, ONLY: m_walltime
   USE dbcsr_methods, ONLY: &
      dbcsr_col_block_offsets, dbcsr_col_block_sizes, dbcsr_distribution, dbcsr_get_data_type, &
      dbcsr_get_matrix_type, dbcsr_name, dbcsr_nfullcols_total, dbcsr_nfullrows_total, &
      dbcsr_release, dbcsr_row_block_offsets, dbcsr_row_block_sizes
   USE dbcsr_mp_methods, ONLY: dbcsr_mp_npcols, &
                               dbcsr_mp_nprows
   USE dbcsr_mpiwrap, ONLY: mp_environ, &
                            mp_sum, &
                            mp_sync
   USE dbcsr_multiply_api, ONLY: dbcsr_multiply
   USE dbcsr_operations, ONLY: dbcsr_copy, &
                               dbcsr_scale
   USE dbcsr_test_methods, ONLY: dbcsr_make_random_block_sizes, &
                                 dbcsr_make_random_matrix, &
                                 dbcsr_reset_randmat_seed
   USE dbcsr_toollib, ONLY: atoi, &
                            atol, &
                            ator
   USE dbcsr_transformations, ONLY: dbcsr_redistribute
   USE dbcsr_types, ONLY: &
      dbcsr_conjugate_transpose, dbcsr_distribution_obj, dbcsr_mp_obj, dbcsr_no_transpose, &
      dbcsr_scalar_type, dbcsr_transpose, dbcsr_type, dbcsr_type_antisymmetric, &
      dbcsr_type_complex_4, dbcsr_type_complex_8, dbcsr_type_no_symmetry, dbcsr_type_real_4, &
      dbcsr_type_real_8
   USE dbcsr_work_operations, ONLY: dbcsr_create, &
                                    dbcsr_finalize
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: dbcsr_perf_multiply

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_performance_multiply'

CONTAINS

   SUBROUTINE dbcsr_perf_multiply(group, mp_env, npdims, io_unit, narg, args_shift, args)

      INTEGER                                            :: group
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      INTEGER, DIMENSION(2), INTENT(in)                  :: npdims
      INTEGER                                            :: io_unit, narg, args_shift
      CHARACTER(len=*), DIMENSION(:), INTENT(IN)         :: args

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_perf_multiply', &
                                     routineP = moduleN//':'//routineN

      CHARACTER                                          :: symmetries(3), trans(2)
      INTEGER                                            :: i, iblk, kblk_to_read, limits(6), &
                                                            matrix_sizes(3), mblk_to_read, &
                                                            nblk_to_read, nrep, TYPE
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: bs_k, bs_m, bs_n
      LOGICAL                                            :: chksum_check, retain_sparsity
      REAL(real_8)                                       :: alpha(2), beta(2), chksum_ref, &
                                                            chksum_ref_pos, chksum_threshold, &
                                                            sparsities(3)

!
! parsing

      IF (narg .LT. args_shift+31) THEN
         WRITE (io_unit, *) "Input file format:"
         WRITE (io_unit, *) "      npcols for MPI grid \\"
         WRITE (io_unit, *) "      use MPI-RMA algorithm \\"
         WRITE (io_unit, *) "      dbcsr_multiply \\"
         WRITE (io_unit, *) "      M N K \\"
         WRITE (io_unit, *) "      SpA SpB SpC \\"
         WRITE (io_unit, *) "      TrA TrB \\"
         WRITE (io_unit, *) "      SymA SymB SymC \\"
         WRITE (io_unit, *) "      data_type \\"
         WRITE (io_unit, *) "      Re(alpha) Im(alpha) Re(beta) Im(beta) \\"
         WRITE (io_unit, *) "      limRowL limRowU limColL limColU limKL limKU \\"
         WRITE (io_unit, *) "      retain_sparsity nrep \\"
         WRITE (io_unit, *) "      nmblksizes nnblksizes nkblksizes \\"
         WRITE (io_unit, *) "      [mblksizes] [nblksizes] [kblksizes] \\"
         WRITE (io_unit, *) "      checksum_check checksum_threshold checksum_ref checksum_ref_pos"
         DBCSR_ABORT("narg not correct")
      ENDIF

      matrix_sizes(1) = atoi(args(args_shift+1))
      matrix_sizes(2) = atoi(args(args_shift+2))
      matrix_sizes(3) = atoi(args(args_shift+3))
      sparsities(1) = ator(args(args_shift+4))
      sparsities(2) = ator(args(args_shift+5))
      sparsities(3) = ator(args(args_shift+6))
      trans(1) = args(args_shift+7)
      trans(2) = args(args_shift+8)
      symmetries(1) = args(args_shift+9)
      symmetries(2) = args(args_shift+10)
      symmetries(3) = args(args_shift+11)
      TYPE = atoi(args(args_shift+12))
      alpha(1) = ator(args(args_shift+13))
      alpha(2) = ator(args(args_shift+14))
      beta(1) = ator(args(args_shift+15))
      beta(2) = ator(args(args_shift+16))
      limits(1) = atoi(args(args_shift+17))
      limits(2) = atoi(args(args_shift+18))
      limits(3) = atoi(args(args_shift+19))
      limits(4) = atoi(args(args_shift+20))
      limits(5) = atoi(args(args_shift+21))
      limits(6) = atoi(args(args_shift+22))
      retain_sparsity = atol(args(args_shift+23))
      nrep = atoi(args(args_shift+24))
      mblk_to_read = atoi(args(args_shift+25))
      nblk_to_read = atoi(args(args_shift+26))
      kblk_to_read = atoi(args(args_shift+27))

      IF (narg < 34 + 2*(mblk_to_read + nblk_to_read + kblk_to_read)) &
         DBCSR_ABORT("narg not correct")

      ALLOCATE (bs_m(2*mblk_to_read), bs_n(2*nblk_to_read), bs_k(2*kblk_to_read))

      i = args_shift+27
      DO iblk = 1, mblk_to_read
         i = i + 1
         bs_m(2*(iblk - 1) + 1) = atoi(args(i))
         i = i + 1
         bs_m(2*(iblk - 1) + 2) = atoi(args(i))
      ENDDO
      DO iblk = 1, nblk_to_read
         i = i + 1
         bs_n(2*(iblk - 1) + 1) = atoi(args(i))
         i = i + 1
         bs_n(2*(iblk - 1) + 2) = atoi(args(i))
      ENDDO
      DO iblk = 1, kblk_to_read
         i = i + 1
         bs_k(2*(iblk - 1) + 1) = atoi(args(i))
         i = i + 1
         bs_k(2*(iblk - 1) + 2) = atoi(args(i))
      ENDDO

      chksum_check = atol(args(i + 1))
      chksum_threshold = ator(args(i + 2))
      IF (chksum_check .AND. chksum_threshold .LE. 0.0) DBCSR_ABORT("Checksum threshold must be positive!")
      chksum_ref = ator(args(i + 3))
      chksum_ref_pos = ator(args(i + 4))

      !
      ! do checks here

      !
      ! if the limits are not specified (i.e 0), we set them here
      IF (limits(1) .EQ. 0) limits(1) = 1
      IF (limits(2) .EQ. 0) limits(2) = matrix_sizes(1)
      IF (limits(3) .EQ. 0) limits(3) = 1
      IF (limits(4) .EQ. 0) limits(4) = matrix_sizes(2)
      IF (limits(5) .EQ. 0) limits(5) = 1
      IF (limits(6) .EQ. 0) limits(6) = matrix_sizes(3)

      !
      ! lets go !
      CALL dbcsr_perf_multiply_low(group, mp_env, npdims, io_unit, matrix_sizes, &
                                   bs_m, bs_n, bs_k, sparsities, trans, symmetries, TYPE, &
                                   alpha, beta, limits, retain_sparsity, nrep, &
                                   chksum_check, chksum_threshold, chksum_ref, &
                                   chksum_ref_pos)

      DEALLOCATE (bs_m, bs_n, bs_k)

   END SUBROUTINE dbcsr_perf_multiply

   SUBROUTINE dbcsr_perf_multiply_low(mp_group, mp_env, npdims, io_unit, &
                                      matrix_sizes, bs_m, bs_n, bs_k, sparsities, trans, symmetries, data_type, &
                                      alpha_in, beta_in, limits, retain_sparsity, nrep, &
                                      chksum_check, chksum_threshold, chksum_ref, chksum_ref_pos)
      !! Performs a variety of matrix multiplies of same matrices on different
      !! processor grids

      INTEGER, INTENT(IN)                                :: mp_group
         !! MPI communicator
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      INTEGER, DIMENSION(2), INTENT(in)                  :: npdims
      INTEGER, INTENT(IN)                                :: io_unit
         !! which unit to write to, if not negative
      INTEGER, DIMENSION(:), INTENT(in)                  :: matrix_sizes, bs_m, bs_n, bs_k
         !! size of matrices to test
         !! block sizes of the 3 dimensions
         !! block sizes of the 3 dimensions
         !! block sizes of the 3 dimensions
      REAL(real_8), DIMENSION(3), INTENT(in)             :: sparsities
         !! sparsities of matrices to create
      CHARACTER, DIMENSION(2), INTENT(in)                :: trans
         !! transposes of the two matrices
      CHARACTER, DIMENSION(3), INTENT(in)                :: symmetries
      INTEGER, INTENT(IN)                                :: data_type
         !! types of matrices to create
      REAL(real_8), DIMENSION(2), INTENT(in)             :: alpha_in, beta_in
         !! alpha value to use in multiply
         !! beta value to use in multiply
      INTEGER, DIMENSION(6), INTENT(in)                  :: limits
      LOGICAL, INTENT(in)                                :: retain_sparsity
      INTEGER, INTENT(IN)                                :: nrep
      LOGICAL, INTENT(in)                                :: chksum_check
      REAL(real_8), INTENT(in)                           :: chksum_threshold, chksum_ref, &
                                                            chksum_ref_pos

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_perf_multiply_low', &
                                     routineP = moduleN//':'//routineN

      CHARACTER                                          :: a_symm, b_symm, c_symm, transa, transb
      INTEGER                                            :: handle, mynode, numnodes, numthreads
      INTEGER, DIMENSION(:), POINTER :: col_dist_a, col_dist_b, col_dist_c, my_sizes_k, &
                                        my_sizes_m, my_sizes_n, row_dist_a, row_dist_b, row_dist_c, sizes_k, sizes_m, sizes_n
      LOGICAL                                            :: do_complex
      LOGICAL, DIMENSION(2)                              :: trs
      TYPE(dbcsr_distribution_obj)                       :: dist_a, dist_b, dist_c
      TYPE(dbcsr_scalar_type)                            :: alpha, beta
      TYPE(dbcsr_type)                                   :: matrix_a, matrix_b, matrix_c

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)
      NULLIFY (my_sizes_k, my_sizes_m, my_sizes_n, &
               sizes_k, sizes_m, sizes_n)

      !
      ! print
      CALL mp_environ(numnodes, mynode, mp_group)
      IF (io_unit .GT. 0) THEN
         numthreads = 1
!$OMP PARALLEL
!$OMP MASTER
!$       numthreads = omp_get_num_threads()
!$OMP END MASTER
!$OMP END PARALLEL
         WRITE (io_unit, *) 'numthreads', numthreads
         WRITE (io_unit, *) 'numnodes', numnodes
         WRITE (io_unit, *) 'matrix_sizes', matrix_sizes
         WRITE (io_unit, *) 'sparsities', sparsities
         WRITE (io_unit, *) 'trans ', trans
         WRITE (io_unit, *) 'symmetries ', symmetries
         WRITE (io_unit, *) 'type ', data_type
         WRITE (io_unit, *) 'alpha_in', alpha_in
         WRITE (io_unit, *) 'beta_in', beta_in
         WRITE (io_unit, *) 'limits', limits
         WRITE (io_unit, *) 'retain_sparsity', retain_sparsity
         WRITE (io_unit, *) 'nrep', nrep
         WRITE (io_unit, *) 'bs_m', bs_m
         WRITE (io_unit, *) 'bs_n', bs_n
         WRITE (io_unit, *) 'bs_k', bs_k
      ENDIF
      !
      CALL dbcsr_reset_randmat_seed()
      !
      a_symm = symmetries(1)
      b_symm = symmetries(2)
      c_symm = symmetries(3)

      IF (a_symm .NE. dbcsr_type_no_symmetry .AND. matrix_sizes(1) .NE. matrix_sizes(3)) &
         DBCSR_ABORT("")

      IF (b_symm .NE. dbcsr_type_no_symmetry .AND. matrix_sizes(2) .NE. matrix_sizes(3)) &
         DBCSR_ABORT("")

      IF (c_symm .NE. dbcsr_type_no_symmetry .AND. matrix_sizes(1) .NE. matrix_sizes(2)) &
         DBCSR_ABORT("")

      do_complex = data_type .EQ. dbcsr_type_complex_4 .OR. data_type .EQ. dbcsr_type_complex_8

      SELECT CASE (data_type)
      CASE (dbcsr_type_real_4)
         alpha = dbcsr_scalar(REAL(alpha_in(1), real_4))
         beta = dbcsr_scalar(REAL(beta_in(1), real_4))
      CASE (dbcsr_type_real_8)
         alpha = dbcsr_scalar(REAL(alpha_in(1), real_8))
         beta = dbcsr_scalar(REAL(beta_in(1), real_8))
      CASE (dbcsr_type_complex_4)
         alpha = dbcsr_scalar(CMPLX(alpha_in(1), alpha_in(2), real_4))
         beta = dbcsr_scalar(CMPLX(beta_in(1), beta_in(2), real_4))
      CASE (dbcsr_type_complex_8)
         alpha = dbcsr_scalar(CMPLX(alpha_in(1), alpha_in(2), real_8))
         beta = dbcsr_scalar(CMPLX(beta_in(1), beta_in(2), real_8))
      END SELECT

      transa = trans(1)
      transb = trans(2)

      !
      ! if C has a symmetry, we need special transpositions
      IF (c_symm .NE. dbcsr_type_no_symmetry) THEN
         IF (.NOT. (transa .EQ. dbcsr_no_transpose .AND. &
                    transb .EQ. dbcsr_transpose .OR. &
                    transa .EQ. dbcsr_transpose .AND. &
                    transb .EQ. dbcsr_no_transpose .OR. &
                    transa .EQ. dbcsr_no_transpose .AND. &
                    transb .EQ. dbcsr_conjugate_transpose .AND. &
                    .NOT. do_complex .OR. &
                    transa .EQ. dbcsr_conjugate_transpose .AND. &
                    transb .EQ. dbcsr_no_transpose .AND. &
                    .NOT. do_complex)) THEN
            DBCSR_ABORT("")
         ENDIF
      ENDIF
      !
      ! if C has symmetry and special limits
      IF (c_symm .NE. dbcsr_type_no_symmetry) THEN
         IF (limits(1) .NE. 1 .OR. limits(2) .NE. matrix_sizes(1) .OR. &
             limits(3) .NE. 1 .OR. limits(4) .NE. matrix_sizes(2)) THEN
            DBCSR_ABORT("")
         ENDIF
      ENDIF

      !
      ! Create the row/column block sizes.
      CALL dbcsr_make_random_block_sizes(sizes_m, matrix_sizes(1), bs_m)
      CALL dbcsr_make_random_block_sizes(sizes_n, matrix_sizes(2), bs_n)
      CALL dbcsr_make_random_block_sizes(sizes_k, matrix_sizes(3), bs_k)

      !
      ! if we have symmetry the row and column block sizes have to match
      IF (c_symm .NE. dbcsr_type_no_symmetry .AND. a_symm .NE. dbcsr_type_no_symmetry .AND. &
          b_symm .NE. dbcsr_type_no_symmetry) THEN
         my_sizes_m => sizes_m; my_sizes_n => sizes_m; my_sizes_k => sizes_m
      ELSEIF ((c_symm .EQ. dbcsr_type_no_symmetry .AND. a_symm .NE. dbcsr_type_no_symmetry .AND. &
               b_symm .NE. dbcsr_type_no_symmetry) .OR. &
              (c_symm .NE. dbcsr_type_no_symmetry .AND. a_symm .EQ. dbcsr_type_no_symmetry .AND. &
               b_symm .NE. dbcsr_type_no_symmetry) .OR. &
              (c_symm .NE. dbcsr_type_no_symmetry .AND. a_symm .NE. dbcsr_type_no_symmetry .AND. &
               b_symm .EQ. dbcsr_type_no_symmetry)) THEN
         my_sizes_m => sizes_m; my_sizes_n => sizes_m; my_sizes_k => sizes_m
      ELSEIF (c_symm .EQ. dbcsr_type_no_symmetry .AND. a_symm .EQ. dbcsr_type_no_symmetry .AND. &
              b_symm .NE. dbcsr_type_no_symmetry) THEN
         my_sizes_m => sizes_m; my_sizes_n => sizes_n; my_sizes_k => sizes_n
      ELSEIF (c_symm .EQ. dbcsr_type_no_symmetry .AND. a_symm .NE. dbcsr_type_no_symmetry .AND. &
              b_symm .EQ. dbcsr_type_no_symmetry) THEN
         my_sizes_m => sizes_m; my_sizes_n => sizes_n; my_sizes_k => sizes_m
      ELSEIF (c_symm .NE. dbcsr_type_no_symmetry .AND. a_symm .EQ. dbcsr_type_no_symmetry .AND. &
              b_symm .EQ. dbcsr_type_no_symmetry) THEN
         my_sizes_m => sizes_m; my_sizes_n => sizes_m; my_sizes_k => sizes_k
      ELSEIF (c_symm .EQ. dbcsr_type_no_symmetry .AND. a_symm .EQ. dbcsr_type_no_symmetry .AND. &
              b_symm .EQ. dbcsr_type_no_symmetry) THEN
         my_sizes_m => sizes_m; my_sizes_n => sizes_n; my_sizes_k => sizes_k
      ELSE
         DBCSR_ABORT("something wrong here...")
      ENDIF

      ! Create the random matrices.
      trs(1) = transa .NE. dbcsr_no_transpose
      trs(2) = transb .NE. dbcsr_no_transpose
      CALL dbcsr_dist_bin(row_dist_c, SIZE(sizes_m), npdims(1), &
                          sizes_m)
      CALL dbcsr_dist_bin(col_dist_c, SIZE(sizes_n), npdims(2), &
                          sizes_n)
      CALL dbcsr_distribution_new(dist_c, mp_env, row_dist_c, col_dist_c)
      CALL dbcsr_make_random_matrix(matrix_c, sizes_m, sizes_n, "Matrix C", &
                                    REAL(sparsities(3), real_8), &
                                    mp_group, data_type=data_type, dist=dist_c)
      CALL dbcsr_distribution_release(dist_c)
      IF (trs(1)) THEN
         CALL dbcsr_dist_bin(row_dist_a, SIZE(sizes_k), npdims(1), &
                             sizes_k)
         CALL dbcsr_dist_bin(col_dist_a, SIZE(sizes_m), npdims(2), &
                             sizes_m)
         CALL dbcsr_distribution_new(dist_a, mp_env, row_dist_a, col_dist_a)
         CALL dbcsr_make_random_matrix(matrix_a, sizes_k, sizes_m, "Matrix A", &
                                       REAL(sparsities(1), real_8), &
                                       mp_group, data_type=data_type, dist=dist_a)
         DEALLOCATE (row_dist_a, col_dist_a)
      ELSE
         CALL dbcsr_dist_bin(col_dist_a, SIZE(sizes_k), npdims(2), &
                             sizes_k)
         CALL dbcsr_distribution_new(dist_a, mp_env, row_dist_c, col_dist_a)
         CALL dbcsr_make_random_matrix(matrix_a, sizes_m, sizes_k, "Matrix A", &
                                       REAL(sparsities(1), real_8), &
                                       mp_group, data_type=data_type, dist=dist_a)
         DEALLOCATE (col_dist_a)
      ENDIF
      CALL dbcsr_distribution_release(dist_a)
      IF (trs(2)) THEN
         CALL dbcsr_dist_bin(row_dist_b, SIZE(sizes_n), npdims(1), &
                             sizes_n)
         CALL dbcsr_dist_bin(col_dist_b, SIZE(sizes_k), npdims(2), &
                             sizes_k)
         CALL dbcsr_distribution_new(dist_b, mp_env, row_dist_b, col_dist_b)
         CALL dbcsr_make_random_matrix(matrix_b, sizes_n, sizes_k, "Matrix B", &
                                       REAL(sparsities(2), real_8), &
                                       mp_group, data_type=data_type, dist=dist_b)
         DEALLOCATE (row_dist_b, col_dist_b)
      ELSE
         CALL dbcsr_dist_bin(row_dist_b, SIZE(sizes_k), npdims(1), &
                             sizes_k)
         CALL dbcsr_distribution_new(dist_b, mp_env, row_dist_b, col_dist_c)
         CALL dbcsr_make_random_matrix(matrix_b, sizes_k, sizes_n, "Matrix B", &
                                       REAL(sparsities(2), real_8), &
                                       mp_group, data_type=data_type, dist=dist_b)
         DEALLOCATE (row_dist_b)
      ENDIF
      CALL dbcsr_distribution_release(dist_b)
      DEALLOCATE (row_dist_c, col_dist_c, sizes_m, sizes_n, sizes_k)

      !
      ! if C has a symmetry, we build it accordingly, i.e. C=A*A and C=A*(-A)
      IF (c_symm .NE. dbcsr_type_no_symmetry) THEN
         CALL dbcsr_copy(matrix_b, matrix_a)
         !print*, a_symm,b_symm,dbcsr_get_matrix_type(matrix_a),dbcsr_get_matrix_type(matrix_b)
         IF (c_symm .EQ. dbcsr_type_antisymmetric) THEN
            CALL dbcsr_scale(matrix_b, &
                             alpha_scalar=dbcsr_scalar_negative( &
                             dbcsr_scalar_one(data_type)))
         ENDIF
      ENDIF

      !
      ! Prepare test parameters
      CALL perf_multiply(mp_group, mp_env, io_unit, &
                         matrix_a, matrix_b, matrix_c, &
                         transa, transb, &
                         alpha, beta, &
                         limits, retain_sparsity, &
                         nrep, &
                         chksum_check, chksum_threshold, chksum_ref, &
                         chksum_ref_pos)

      !
      ! cleanup
      CALL dbcsr_release(matrix_a)
      CALL dbcsr_release(matrix_b)
      CALL dbcsr_release(matrix_c)

      CALL timestop(handle)

   END SUBROUTINE dbcsr_perf_multiply_low

   SUBROUTINE perf_multiply(mp_group, mp_env, io_unit, &
                            matrix_a, matrix_b, matrix_c, &
                            transa, transb, alpha, beta, limits, retain_sparsity, &
                            nrep, chksum_check, chksum_threshold, chksum_ref, chksum_ref_pos)
      !! Performs a variety of matrix multiplies of same matrices on different
      !! processor grids

      INTEGER, INTENT(IN)                                :: mp_group
         !! MPI communicator
      TYPE(dbcsr_mp_obj), INTENT(IN)                     :: mp_env
      INTEGER, INTENT(IN)                                :: io_unit
         !! which unit to write to, if not negative
      TYPE(dbcsr_type), INTENT(in)                       :: matrix_a, matrix_b, matrix_c
         !! matrices to multiply
         !! matrices to multiply
         !! matrices to multiply
      CHARACTER, INTENT(in)                              :: transa, transb
      TYPE(dbcsr_scalar_type), INTENT(in)                :: alpha, beta
      INTEGER, DIMENSION(6), INTENT(in)                  :: limits
      LOGICAL, INTENT(in)                                :: retain_sparsity
      INTEGER, INTENT(IN)                                :: nrep
      LOGICAL, INTENT(in)                                :: chksum_check
      REAL(real_8), INTENT(in)                           :: chksum_threshold, chksum_ref, &
                                                            chksum_ref_pos

      CHARACTER(len=*), PARAMETER :: routineN = 'perf_multiply', routineP = moduleN//':'//routineN

      INTEGER                                            :: c_a, c_b, c_c, handle, irep, mynode, &
                                                            nthreads, numnodes, r_a, r_b, r_c
      INTEGER(int_8)                                     :: flop
      INTEGER(int_8), ALLOCATABLE, DIMENSION(:)          :: flop_sum
      INTEGER(int_8), ALLOCATABLE, DIMENSION(:, :)       :: flops
      INTEGER, DIMENSION(:), POINTER                     :: blk_offsets, col_dist_a, col_dist_b, &
                                                            col_dist_c, row_dist_a, row_dist_b, &
                                                            row_dist_c
      LOGICAL                                            :: chksum_err
      REAL(real_8)                                       :: chksum_a, chksum_b, chksum_c_in, &
                                                            chksum_c_out, chksum_c_out_pos, &
                                                            rel_diff, std_all, std_t, t1, t2
      REAL(real_8), ALLOCATABLE, DIMENSION(:)            :: flops_all, flops_node, flops_thread, &
                                                            load_imb, t, t_max, t_min
      REAL(real_8), ALLOCATABLE, DIMENSION(:, :)         :: times
      TYPE(dbcsr_distribution_obj)                       :: dist_a, dist_b, dist_c
      TYPE(dbcsr_type)                                   :: m_a, m_b, m_c, m_c_orig

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)

      CALL mp_environ(numnodes, mynode, mp_group)

      nthreads = 1
!$    nthreads = OMP_GET_MAX_THREADS()

      ALLOCATE (times(0:numnodes - 1, nrep), flops(0:numnodes - 1, nrep), t_max(nrep), &
                flops_node(nrep), flops_thread(nrep), flops_all(nrep), flop_sum(nrep), &
                t_min(nrep), t(nrep), load_imb(nrep))
      times(:, :) = 0.0_real_8
      t_max(:) = 0.0_real_8
      t_min(:) = 0.0_real_8
      t(:) = 0.0_real_8
      flops_node(:) = 0.0_real_8
      flops_thread(:) = 0.0_real_8
      flops_all(:) = 0.0_real_8
      flops(:, :) = 0
      flop_sum(:) = 0
      load_imb(:) = 0.0_real_8

      ! Row & column distributions
      row_dist_c => dbcsr_distribution_row_dist(dbcsr_distribution(matrix_c))
      col_dist_c => dbcsr_distribution_col_dist(dbcsr_distribution(matrix_c))
      row_dist_a => dbcsr_distribution_row_dist(dbcsr_distribution(matrix_a))
      col_dist_a => dbcsr_distribution_col_dist(dbcsr_distribution(matrix_a))
      row_dist_b => dbcsr_distribution_row_dist(dbcsr_distribution(matrix_b))
      col_dist_b => dbcsr_distribution_col_dist(dbcsr_distribution(matrix_b))

      CALL dbcsr_distribution_new(dist_a, mp_env, row_dist_a, col_dist_a)
      CALL dbcsr_distribution_new(dist_b, mp_env, row_dist_b, col_dist_b)
      CALL dbcsr_distribution_new(dist_c, mp_env, row_dist_c, col_dist_c)
      ! Redistribute the matrices
      ! A
      CALL dbcsr_create(m_a, "Test for "//TRIM(dbcsr_name(matrix_a)), &
                        dist_a, dbcsr_get_matrix_type(matrix_a), &
                        dbcsr_row_block_sizes(matrix_a), &
                        dbcsr_col_block_sizes(matrix_a), &
                        data_type=dbcsr_get_data_type(matrix_a))
      CALL dbcsr_distribution_release(dist_a)
      CALL dbcsr_redistribute(matrix_a, m_a)
      ! B
      CALL dbcsr_create(m_b, "Test for "//TRIM(dbcsr_name(matrix_b)), &
                        dist_b, dbcsr_get_matrix_type(matrix_b), &
                        dbcsr_row_block_sizes(matrix_b), &
                        dbcsr_col_block_sizes(matrix_b), &
                        data_type=dbcsr_get_data_type(matrix_b))
      CALL dbcsr_distribution_release(dist_b)
      CALL dbcsr_redistribute(matrix_b, m_b)
      ! C orig
      CALL dbcsr_create(m_c_orig, "Test for "//TRIM(dbcsr_name(matrix_c)), &
                        dist_c, dbcsr_get_matrix_type(matrix_c), &
                        dbcsr_row_block_sizes(matrix_c), &
                        dbcsr_col_block_sizes(matrix_c), &
                        data_type=dbcsr_get_data_type(matrix_c))
      CALL dbcsr_distribution_release(dist_c)
      CALL dbcsr_redistribute(matrix_c, m_c_orig)
      ! C
      CALL dbcsr_create(m_c, "Test for "//TRIM(dbcsr_name(matrix_c)), &
                        dist_c, dbcsr_get_matrix_type(matrix_c), &
                        dbcsr_row_block_sizes(matrix_c), &
                        dbcsr_col_block_sizes(matrix_c), &
                        data_type=dbcsr_get_data_type(matrix_c))
      CALL dbcsr_finalize(m_c)

      IF (.FALSE.) THEN
         blk_offsets => dbcsr_row_block_offsets(matrix_c)
         WRITE (io_unit, *) 'row_block_offsets(matrix_c)', blk_offsets
         blk_offsets => dbcsr_col_block_offsets(matrix_c)
         WRITE (io_unit, *) 'col_block_offsets(matrix_c)', blk_offsets
      ENDIF

      IF (.FALSE.) THEN
         CALL dbcsr_print(m_c, matlab_format=.FALSE., variable_name='c_in_')
         CALL dbcsr_print(m_a, matlab_format=.FALSE., variable_name='a_')
         CALL dbcsr_print(m_b, matlab_format=.FALSE., variable_name='b_')
         CALL dbcsr_print(m_c, matlab_format=.FALSE., variable_name='c_out_')
      ENDIF

      r_a = dbcsr_nfullrows_total(m_a)
      c_a = dbcsr_nfullcols_total(m_a)
      r_b = dbcsr_nfullrows_total(m_b)
      c_b = dbcsr_nfullcols_total(m_b)
      r_c = dbcsr_nfullrows_total(m_c_orig)
      c_c = dbcsr_nfullcols_total(m_c_orig)

      chksum_a = dbcsr_checksum(m_a)
      chksum_b = dbcsr_checksum(m_b)
      chksum_c_in = dbcsr_checksum(m_c_orig)

      !
      !
      DO irep = 1, nrep

         !
         ! set the C matrix
         CALL dbcsr_copy(m_c, m_c_orig)

         !
         ! Perform multiply
         CALL mp_sync(mp_group)
         t1 = m_walltime()
         flop = 0
         CALL dbcsr_multiply(transa, transb, alpha, &
                             m_a, m_b, beta, m_c, &
                             first_row=limits(1), &
                             last_row=limits(2), &
                             first_column=limits(3), &
                             last_column=limits(4), &
                             first_k=limits(5), &
                             last_k=limits(6), &
                             retain_sparsity=retain_sparsity, &
                             flop=flop)
         t2 = m_walltime()
         times(mynode, irep) = t2 - t1
         flops(mynode, irep) = flop
      ENDDO

      chksum_c_out = dbcsr_checksum(m_c)
      chksum_c_out_pos = dbcsr_checksum(m_c, pos=.TRUE.)

      CALL mp_sum(times, 0, mp_group)
      CALL mp_sum(flops, 0, mp_group)

      !
      !
      t_max(:) = MAXVAL(times, DIM=1)
      t_min(:) = MINVAL(times, DIM=1)
      t(:) = SUM(times, DIM=1)/REAL(numnodes, real_8)
      flop_sum(:) = SUM(flops, DIM=1)
      t_max(:) = MAX(t_max(:), 0.001_real_8)
      flops_all(:) = REAL(flop_sum(:), KIND=real_8)/t_max(:) !* 1.0e-9_real_8
      flops_node(:) = flops_all(:)/REAL(numnodes, real_8)
      flops_thread(:) = flops_node(:)/REAL(nthreads, real_8)
      load_imb(:) = t_max(:) - t(:)/REAL(numnodes, real_8)

      std_t = 1.0_real_8
      std_all = 1.0_real_8

      IF (io_unit .GT. 0) THEN
         WRITE (io_unit, *) REPEAT("*", 80)
         WRITE (io_unit, *) " -- PERF dbcsr_multiply (", transa, ", ", transb, &
            ", ", dbcsr_get_data_type(m_a), &
            ", ", dbcsr_get_matrix_type(m_a), &
            ", ", dbcsr_get_matrix_type(m_b), &
            ", ", dbcsr_get_matrix_type(m_c), &
            ")"
         WRITE (io_unit, '(T4,3(A,I9,A,I9),A)') &
            "matrix sizes A(", r_a, " x", c_a, "), B(", r_b, " x", c_b, ") and C(", r_c, " x", c_c, ")"
         WRITE (io_unit, '(T4,A,I5,A,I3,A,I3,A)') 'numnodes (nprows X npcols) = ', numnodes, &
            "(", dbcsr_mp_nprows(mp_env), " X ", &
            dbcsr_mp_npcols(mp_env), ")"
         WRITE (io_unit, '(T4,A,I5)') 'nthreads        = ', nthreads
         WRITE (io_unit, '(T4,A,E26.15)') 'checksum(A)     = ', chksum_a
         WRITE (io_unit, '(T4,A,E26.15)') 'checksum(B)     = ', chksum_b
         WRITE (io_unit, '(T4,A,E26.15)') 'checksum(C_in)  = ', chksum_c_in
         WRITE (io_unit, '(T4,A,E26.15)') 'checksum(C_out) = ', chksum_c_out
         WRITE (io_unit, '(T4,A,E26.15)') 'checksum(C_out) POS = ', chksum_c_out_pos
         IF (chksum_check) THEN
            chksum_err = .FALSE.
            rel_diff = ABS(chksum_c_out/MAX(chksum_ref, chksum_threshold) - 1.)
            IF (rel_diff .GT. chksum_threshold) THEN
               WRITE (io_unit, '(T4,A,E26.15,A,E26.15,A,E26.15)') "Wrong checksum(C_out), ref = ", &
                  chksum_ref, &
                  "  rel_diff = ", rel_diff, &
                  "  threshold = ", chksum_threshold
               chksum_err = .TRUE.
            ENDIF
            rel_diff = ABS(chksum_c_out_pos/MAX(chksum_ref_pos, chksum_threshold) - 1.)
            IF (rel_diff .GT. chksum_threshold) THEN
               WRITE (io_unit, '(T4,A,E26.15,A,E26.15,A,E26.15)') "Wrong checksum(C_out) POS, ref = ", &
                  chksum_ref_pos, &
                  "  rel_diff = ", rel_diff, &
                  "  threshold = ", chksum_threshold
               chksum_err = .TRUE.
            ENDIF
            IF (chksum_err) DBCSR_ABORT("Wrong Checksums. Test failed!")
         ENDIF
         WRITE (io_unit, *)
         WRITE (io_unit, *)
         WRITE (io_unit, '(T4,A)') "                       mean        std         minmin      maxmax"
         WRITE (io_unit, '(T4,A,4EN12.2,A)') "time            = ", mean(t), std(t), &
            MINVAL(times), MAXVAL(times), ' seconds'
         WRITE (io_unit, '(T4,A,4EN12.2,A)') "perf total      = ", mean(flops_all), std(flops_all), &
            MINVAL(flops_all), MAXVAL(flops_all), ' FLOPS'
         WRITE (io_unit, '(T4,A,4EN12.2,A)') "perf per node   = ", mean(flops_node), std(flops_node), &
            MINVAL(flops_node), MAXVAL(flops_node), ' FLOPS'
         WRITE (io_unit, '(T4,A,4EN12.2,A)') "perf per thread = ", mean(flops_thread), std(flops_thread), &
            MINVAL(flops_thread), MAXVAL(flops_thread), ' FLOPS'
         WRITE (io_unit, '(T4,A,4E12.2,A)') "load imbalance  = ", mean(load_imb), std(load_imb), &
            MINVAL(load_imb), MAXVAL(load_imb), ''
         WRITE (io_unit, '(T4,A,4E12.2,A)') "rel load imbal  = ", mean(load_imb/t_max), std(load_imb/t_max), &
            MINVAL(load_imb/t_max), MAXVAL(load_imb/t_max), ''

         WRITE (io_unit, *) REPEAT("*", 80)
      ENDIF

      CALL dbcsr_release(m_a)
      CALL dbcsr_release(m_b)
      CALL dbcsr_release(m_c)
      CALL dbcsr_release(m_c_orig)

      DEALLOCATE (times, flops, t_max, flops_node, flops_thread, flops_all, &
                  flop_sum, t_min, t, load_imb)

      CALL timestop(handle)

   END SUBROUTINE perf_multiply

   FUNCTION mean(v)
      REAL(real_8), DIMENSION(:)                         :: v
      REAL(real_8)                                       :: mean

      INTEGER                                            :: i, n

      mean = 0.0_real_8
      n = SIZE(v, 1)
      DO i = 1, n
         mean = mean + v(i)
      ENDDO
      mean = mean/REAL(n, real_8)
   END FUNCTION mean
   FUNCTION std(v)
      REAL(real_8), DIMENSION(:)                         :: v
      REAL(real_8)                                       :: std

      INTEGER                                            :: i, n
      REAL(real_8)                                       :: mn

      mn = mean(v)
      std = 0.0_real_8
      n = SIZE(v, 1)
      DO i = 1, n
         std = std + (v(i) - mn)**2
      ENDDO
      std = SQRT(std)/REAL(n, real_8)
   END FUNCTION std

END MODULE dbcsr_performance_multiply
