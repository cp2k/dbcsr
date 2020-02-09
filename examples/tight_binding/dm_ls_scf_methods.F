!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!
MODULE dm_ls_scf_methods
   USE dbcsr_log_handling,                 ONLY: dbcsr_get_default_logger,&
                                               dbcsr_logger_get_default_unit_nr,&
                                               dbcsr_logger_type, dbcsr_logger_get_default_io_unit
   USE dbcsr_api,                       ONLY: &
        dbcsr_add, dbcsr_add_on_diag, dbcsr_copy, dbcsr_create, dbcsr_desymmetrize, dbcsr_filter, &
        dbcsr_finalize, dbcsr_frobenius_norm, dbcsr_get_data_type, dbcsr_get_occupation, &
        dbcsr_iterator_blocks_left, dbcsr_iterator_next_block, dbcsr_iterator_start, &
        dbcsr_iterator_stop, dbcsr_iterator_type, dbcsr_multiply, dbcsr_put_block, dbcsr_release, &
        dbcsr_scale, dbcsr_set, dbcsr_trace, dbcsr_type, dbcsr_type_no_symmetry, dbcsr_type_real_4
   USE dbcsr_arnoldi_api,               ONLY: arnoldi_extremal
   USE dbcsr_iterate_matrix,            ONLY: invert_Hotelling,&
                                              matrix_sign_Newton_Schulz,&
                                              matrix_sqrt_Newton_Schulz
   USE dbcsr_kinds,                           ONLY: dp,&
                                              int_8,&
                                              sp
   USE dbcsr_machine,                         ONLY: m_flush,&
                                              m_walltime, default_output_unit

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dm_ls_scf_methods'

   PUBLIC :: density_matrix_sign, density_matrix_sign_fixed_mu

CONTAINS


! **************************************************************************************************
! **************************************************************************************************
   SUBROUTINE density_matrix_sign(matrix_p, mu, fixed_mu, matrix_ks, matrix_s, matrix_s_inv, nelectron, threshold)

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_p
      REAL(KIND=dp), INTENT(INOUT)                       :: mu
      LOGICAL                                            :: fixed_mu
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_ks, matrix_s, matrix_s_inv
      INTEGER, INTENT(IN)                                :: nelectron
      REAL(KIND=dp), INTENT(IN)                          :: threshold

      CHARACTER(LEN=*), PARAMETER :: routineN = 'density_matrix_sign', &
         routineP = moduleN//':'//routineN
      REAL(KIND=dp), PARAMETER                           :: initial_increment = 0.01_dp

      INTEGER                                            :: handle, iter, unit_nr
      LOGICAL                                            :: has_mu_high, has_mu_low
      REAL(KIND=dp)                                      :: increment, mu_high, mu_low, trace

      unit_nr = dbcsr_logger_get_default_io_unit()

      increment = initial_increment

      has_mu_low = .FALSE.
      has_mu_high = .FALSE.

      ! bisect if both bounds are known, otherwise find the bounds with a linear search
      DO iter = 1, 30
         IF (has_mu_low .AND. has_mu_high) THEN
            mu = (mu_low+mu_high)/2
            IF (ABS(mu_high-mu_low) < threshold) EXIT
         ENDIF

        CALL density_matrix_sign_fixed_mu(matrix_p, trace, mu, matrix_ks, matrix_s, matrix_s_inv, threshold)
         IF (unit_nr > 0) WRITE (unit_nr, '(T2,A,I2,1X,F13.9,1X,F15.9)') &
            "Density matrix:  iter, mu, trace error: ", iter, mu, trace-nelectron

         ! OK, we can skip early if we are as close as possible to the exact result
         ! smaller differences should be considered 'noise'
         IF (ABS(trace-nelectron) < 0.5_dp .OR. fixed_mu) EXIT

         IF (trace < nelectron) THEN
            mu_low = mu
            mu = mu+increment
            has_mu_low = .TRUE.
            increment = increment*2
         ELSE
            mu_high = mu
            mu = mu-increment
            has_mu_high = .TRUE.
            increment = increment*2
         ENDIF
      ENDDO


   END SUBROUTINE density_matrix_sign

   SUBROUTINE density_matrix_sign_fixed_mu(matrix_p, trace, mu, matrix_ks, matrix_s, matrix_s_inv, threshold)
      !! for a fixed mu, compute the corresponding density matrix and its trace

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_p
      REAL(KIND=dp), INTENT(OUT)                         :: trace
      REAL(KIND=dp), INTENT(INOUT)                       :: mu
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_ks, matrix_s, matrix_s_inv
      REAL(KIND=dp), INTENT(IN)                          :: threshold

      CHARACTER(LEN=*), PARAMETER :: routineN = 'density_matrix_sign_fixed_mu', &
         routineP = moduleN//':'//routineN

      INTEGER                                            :: handle, unit_nr
      REAL(KIND=dp)                                      :: frob_matrix
      TYPE(dbcsr_type)                                   :: matrix_p_ud, matrix_sign, &
                                                            matrix_sinv_ks, matrix_tmp


      unit_nr = dbcsr_logger_get_default_io_unit()

      ! get inv(S)*H-I*mu
      CALL dbcsr_create(matrix_sinv_ks, template=matrix_s, matrix_type=dbcsr_type_no_symmetry)
      CALL dbcsr_multiply("N", "N", 1.0_dp, matrix_s_inv, matrix_ks, &
                          0.0_dp, matrix_sinv_ks, filter_eps=threshold)
      CALL dbcsr_add_on_diag(matrix_sinv_ks, -mu)

      ! compute sign(inv(S)*H-I*mu)
      CALL dbcsr_create(matrix_sign, template=matrix_s, matrix_type=dbcsr_type_no_symmetry)
      CALL matrix_sign_Newton_Schulz(matrix_sign, matrix_sinv_ks, threshold)
      CALL dbcsr_release(matrix_sinv_ks)

      ! now construct the density matrix PS=0.5*(I-sign(inv(S)H-I*mu))
      CALL dbcsr_create(matrix_p_ud, template=matrix_s, matrix_type=dbcsr_type_no_symmetry)
      CALL dbcsr_copy(matrix_p_ud, matrix_sign)
      CALL dbcsr_scale(matrix_p_ud, -0.5_dp)
      CALL dbcsr_add_on_diag(matrix_p_ud, 0.5_dp)
      CALL dbcsr_release(matrix_sign)

      ! we now have PS, lets get its trace
      CALL dbcsr_trace(matrix_p_ud, trace)

      ! we can also check it is idempotent PS*PS=PS
      CALL dbcsr_create(matrix_tmp, template=matrix_s, matrix_type=dbcsr_type_no_symmetry)
      CALL dbcsr_multiply("N", "N", 1.0_dp, matrix_p_ud, matrix_p_ud, &
                          0.0_dp, matrix_tmp, filter_eps=threshold)
      CALL dbcsr_add(matrix_tmp, matrix_p_ud, 1.0_dp, -1.0_dp)
      frob_matrix = dbcsr_frobenius_norm(matrix_tmp)
      CALL dbcsr_release(matrix_tmp)
      IF (unit_nr > 0) WRITE (unit_nr, '(T2,A,F20.12)') "Deviation from idempotency: ", frob_matrix

      ! get P=PS*inv(S)
      CALL dbcsr_multiply("N", "N", 1.0_dp, matrix_p_ud, matrix_s_inv, &
                          0.0_dp, matrix_p, filter_eps=threshold)
      CALL dbcsr_release(matrix_p_ud)


   END SUBROUTINE density_matrix_sign_fixed_mu

END MODULE dm_ls_scf_methods

