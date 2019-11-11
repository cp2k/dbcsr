!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_multiply_api
   USE dbcsr_data_methods, ONLY: dbcsr_scalar
   USE dbcsr_kinds, ONLY: int_8, &
                          real_4, &
                          real_8
   USE dbcsr_methods, ONLY: dbcsr_get_data_type
   USE dbcsr_mm, ONLY: dbcsr_multiply_generic
   USE dbcsr_types, ONLY: dbcsr_type, &
                          dbcsr_type_real_4, &
                          dbcsr_type_real_8

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_multiply_api'

   PUBLIC :: dbcsr_multiply

   INTERFACE dbcsr_multiply
      MODULE PROCEDURE dbcsr_multiply_generic
      MODULE PROCEDURE dbcsr_multiply_s, dbcsr_multiply_d, &
         dbcsr_multiply_c, dbcsr_multiply_z
   END INTERFACE

CONTAINS

   SUBROUTINE dbcsr_multiply_s(transa, transb, &
                               alpha, matrix_a, matrix_b, beta, matrix_c, &
                               first_row, last_row, first_column, last_column, first_k, last_k, &
                               retain_sparsity, filter_eps, &
                               flop)
      CHARACTER(LEN=1), INTENT(IN)                       :: transa, transb
      REAL(KIND=real_4), INTENT(IN)                      :: alpha
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_a, matrix_b
      REAL(KIND=real_4), INTENT(IN)                      :: beta
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_c
      INTEGER, INTENT(IN), OPTIONAL                      :: first_row, last_row, first_column, &
                                                            last_column, first_k, last_k
      LOGICAL, INTENT(IN), OPTIONAL                      :: retain_sparsity
      REAL(KIND=real_8), INTENT(IN), OPTIONAL            :: filter_eps
      INTEGER(KIND=int_8), INTENT(OUT), OPTIONAL         :: flop

      CALL dbcsr_multiply_generic(transa, transb, &
                                  dbcsr_scalar(alpha), matrix_a, matrix_b, dbcsr_scalar(beta), matrix_c, &
                                  first_row, last_row, first_column, last_column, first_k, last_k, &
                                  retain_sparsity, &
                                  filter_eps=filter_eps, &
                                  flop=flop)
   END SUBROUTINE dbcsr_multiply_s

   SUBROUTINE dbcsr_multiply_d(transa, transb, &
                               alpha, matrix_a, matrix_b, beta, matrix_c, &
                               first_row, last_row, first_column, last_column, first_k, last_k, &
                               retain_sparsity, filter_eps, &
                               flop)
      CHARACTER(LEN=1), INTENT(IN)                       :: transa, transb
      REAL(KIND=real_8), INTENT(IN)                      :: alpha
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_a, matrix_b
      REAL(KIND=real_8), INTENT(IN)                      :: beta
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_c
      INTEGER, INTENT(IN), OPTIONAL                      :: first_row, last_row, first_column, &
                                                            last_column, first_k, last_k
      LOGICAL, INTENT(IN), OPTIONAL                      :: retain_sparsity
      REAL(KIND=real_8), INTENT(IN), OPTIONAL            :: filter_eps
      INTEGER(KIND=int_8), INTENT(OUT), OPTIONAL         :: flop

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_multiply_d', &
                                     routineP = moduleN//':'//routineN

      IF (dbcsr_get_data_type(matrix_a) .EQ. dbcsr_type_real_4 .AND. &
          dbcsr_get_data_type(matrix_b) .EQ. dbcsr_type_real_4 .AND. &
          dbcsr_get_data_type(matrix_c) .EQ. dbcsr_type_real_4) THEN
         CALL dbcsr_multiply_generic(transa, transb, &
                                     dbcsr_scalar(REAL(alpha, real_4)), matrix_a, matrix_b, &
                                     dbcsr_scalar(REAL(beta, real_4)), matrix_c, &
                                     first_row, last_row, first_column, last_column, first_k, last_k, &
                                     retain_sparsity, &
                                     filter_eps=filter_eps, &
                                     flop=flop)
      ELSEIF (dbcsr_get_data_type(matrix_a) .EQ. dbcsr_type_real_8 .AND. &
              dbcsr_get_data_type(matrix_b) .EQ. dbcsr_type_real_8 .AND. &
              dbcsr_get_data_type(matrix_c) .EQ. dbcsr_type_real_8) THEN
         CALL dbcsr_multiply_generic(transa, transb, &
                                     dbcsr_scalar(alpha), matrix_a, matrix_b, dbcsr_scalar(beta), matrix_c, &
                                     first_row, last_row, first_column, last_column, first_k, last_k, &
                                     retain_sparsity, &
                                     filter_eps=filter_eps, &
                                     flop=flop)
      ELSE
         DBCSR_ABORT("This combination of data types NYI")
      ENDIF
   END SUBROUTINE dbcsr_multiply_d

   SUBROUTINE dbcsr_multiply_c(transa, transb, &
                               alpha, matrix_a, matrix_b, beta, matrix_c, &
                               first_row, last_row, first_column, last_column, first_k, last_k, &
                               retain_sparsity, filter_eps, &
                               flop)
      CHARACTER(LEN=1), INTENT(IN)                       :: transa, transb
      COMPLEX(KIND=real_4), INTENT(IN)                   :: alpha
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_a, matrix_b
      COMPLEX(KIND=real_4), INTENT(IN)                   :: beta
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_c
      INTEGER, INTENT(IN), OPTIONAL                      :: first_row, last_row, first_column, &
                                                            last_column, first_k, last_k
      LOGICAL, INTENT(IN), OPTIONAL                      :: retain_sparsity
      REAL(KIND=real_8), INTENT(IN), OPTIONAL            :: filter_eps
      INTEGER(KIND=int_8), INTENT(OUT), OPTIONAL         :: flop

      CALL dbcsr_multiply_generic(transa, transb, &
                                  dbcsr_scalar(alpha), matrix_a, matrix_b, dbcsr_scalar(beta), matrix_c, &
                                  first_row, last_row, first_column, last_column, first_k, last_k, &
                                  retain_sparsity, &
                                  filter_eps=filter_eps, &
                                  flop=flop)
   END SUBROUTINE dbcsr_multiply_c

   SUBROUTINE dbcsr_multiply_z(transa, transb, &
                               alpha, matrix_a, matrix_b, beta, matrix_c, &
                               first_row, last_row, first_column, last_column, first_k, last_k, &
                               retain_sparsity, filter_eps, &
                               flop)
      CHARACTER(LEN=1), INTENT(IN)                       :: transa, transb
      COMPLEX(KIND=real_8), INTENT(IN)                   :: alpha
      TYPE(dbcsr_type), INTENT(IN)                       :: matrix_a, matrix_b
      COMPLEX(KIND=real_8), INTENT(IN)                   :: beta
      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix_c
      INTEGER, INTENT(IN), OPTIONAL                      :: first_row, last_row, first_column, &
                                                            last_column, first_k, last_k
      LOGICAL, INTENT(IN), OPTIONAL                      :: retain_sparsity
      REAL(KIND=real_8), INTENT(IN), OPTIONAL            :: filter_eps
      INTEGER(KIND=int_8), INTENT(OUT), OPTIONAL         :: flop

      CALL dbcsr_multiply_generic(transa, transb, &
                                  dbcsr_scalar(alpha), matrix_a, matrix_b, dbcsr_scalar(beta), matrix_c, &
                                  first_row, last_row, first_column, last_column, first_k, last_k, &
                                  retain_sparsity, &
                                  filter_eps=filter_eps, &
                                  flop=flop)
   END SUBROUTINE dbcsr_multiply_z

END MODULE dbcsr_multiply_api
