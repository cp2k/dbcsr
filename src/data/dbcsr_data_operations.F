!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_data_operations
   !! DBCSR data operations

   USE dbcsr_block_operations, ONLY: dbcsr_block_transpose, &
                                     dbcsr_data_copy, &
                                     dbcsr_data_set
   USE dbcsr_data_methods, ONLY: dbcsr_data_get_size, &
                                 dbcsr_data_get_size_referenced, &
                                 dbcsr_data_hold, &
                                 dbcsr_data_release, &
                                 dbcsr_data_set_size_referenced, &
                                 dbcsr_get_data
   USE dbcsr_dist_util, ONLY: sgn
   USE dbcsr_kinds, ONLY: real_4, &
                          real_8
   USE dbcsr_types, ONLY: dbcsr_data_obj, &
                          dbcsr_type, &
                          dbcsr_type_complex_4, &
                          dbcsr_type_complex_8, &
                          dbcsr_type_real_4, &
                          dbcsr_type_real_8
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_data_operations'

   PUBLIC :: dbcsr_data_copyall, dbcsr_data_convert, &
             dbcsr_copy_sort_data, &
             dbcsr_sort_data
   PUBLIC :: dbcsr_switch_data_area

CONTAINS

   SUBROUTINE dbcsr_switch_data_area(matrix, data_area, previous_data_area)
      !! Sets the data area of a matrix

      TYPE(dbcsr_type), INTENT(INOUT)                    :: matrix
         !! matrix for which to set the data area
      TYPE(dbcsr_data_obj), INTENT(IN)                   :: data_area
         !! data area to set
      TYPE(dbcsr_data_obj), INTENT(OUT), OPTIONAL        :: previous_data_area
         !! previous data area

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_switch_data_area', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: handle

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)
      IF (PRESENT(previous_data_area)) THEN
         previous_data_area = matrix%data_area
      ELSE
         CALL dbcsr_data_release(matrix%data_area)
      ENDIF
      matrix%data_area = data_area
      CALL dbcsr_data_hold(matrix%data_area)
      CALL timestop(handle)
   END SUBROUTINE dbcsr_switch_data_area

   SUBROUTINE dbcsr_data_copyall(target_area, source_area, shallow)
      !! Copies a data area, deep by default.

      TYPE(dbcsr_data_obj), INTENT(INOUT)                :: target_area
         !! target data area
      TYPE(dbcsr_data_obj), INTENT(IN)                   :: source_area
         !! source data area
      LOGICAL, INTENT(IN), OPTIONAL                      :: shallow
         !! shallow copy (default is deep)

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_data_copyall', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: i, n
      LOGICAL                                            :: shallow_copy

!   ---------------------------------------------------------------------------

!      CALL timeset(routineN, error_handle)
      IF (.NOT. ASSOCIATED(source_area%d)) &
         DBCSR_ABORT("Attempt to copy unassigned data")
      IF (source_area%d%refcount .LE. 0) &
         DBCSR_WARN("Attempt to copy unheld data")
      shallow_copy = .FALSE.
      IF (PRESENT(shallow)) shallow_copy = shallow
      IF (shallow_copy) THEN
         target_area = source_area
         CALL dbcsr_data_hold(target_area)
      ELSE
         IF (.NOT. ASSOCIATED(target_area%d)) &
            DBCSR_ABORT("Target area does not exist.")
         CALL dbcsr_data_set_size_referenced(target_area, &
                                             dbcsr_data_get_size_referenced(source_area))
         n = dbcsr_data_get_size_referenced(source_area)
         SELECT CASE (target_area%d%data_type)
         CASE (dbcsr_type_real_4)
!$OMP           PARALLEL DO DEFAULT(NONE) PRIVATE(I) SHARED(target_area,source_area,n)
            DO i = 1, n
               target_area%d%r_sp(i) = source_area%d%r_sp(i)
            ENDDO
         CASE (dbcsr_type_real_8)
!$OMP           PARALLEL DO DEFAULT(NONE) PRIVATE(I) SHARED(target_area,source_area,n)
            DO i = 1, n
               target_area%d%r_dp(i) = source_area%d%r_dp(i)
            ENDDO
         CASE (dbcsr_type_complex_4)
!$OMP           PARALLEL DO DEFAULT(NONE) PRIVATE(I) SHARED(target_area,source_area,n)
            DO i = 1, n
               target_area%d%c_sp(i) = source_area%d%c_sp(i)
            ENDDO
         CASE (dbcsr_type_complex_8)
!$OMP           PARALLEL DO DEFAULT(NONE) PRIVATE(I) SHARED(target_area,source_area,n)
            DO i = 1, n
               target_area%d%c_dp(i) = source_area%d%c_dp(i)
            ENDDO
         CASE default
            DBCSR_ABORT("Invalid data type")
         END SELECT
      ENDIF
!      CALL timestop(error_handle)
   END SUBROUTINE dbcsr_data_copyall

   SUBROUTINE dbcsr_data_convert(target_area, source_area, drop_real, &
                                 multiply_by_i)
      !! Copies a data area, converting data type

      TYPE(dbcsr_data_obj), INTENT(INOUT)                :: target_area
         !! target data area
      TYPE(dbcsr_data_obj), INTENT(IN)                   :: source_area
         !! source data area
      LOGICAL, INTENT(IN), OPTIONAL                      :: drop_real, multiply_by_i
         !! drops real part of complex numbers instead of the imaginary part; default is false
         !! converts real to complex by placing into imaginary instead of real part

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_data_convert', &
                                     routineP = moduleN//':'//routineN

      COMPLEX(KIND=real_4), DIMENSION(:), POINTER        :: s_data_c, t_data_c
      COMPLEX(KIND=real_8), DIMENSION(:), POINTER        :: s_data_z, t_data_z
      INTEGER                                            :: n, ns, nt
      LOGICAL                                            :: keep_real, noimult
      REAL(KIND=real_4), DIMENSION(:), POINTER           :: s_data_r, t_data_r
      REAL(KIND=real_8), DIMENSION(:), POINTER           :: s_data_d, t_data_d

!   ---------------------------------------------------------------------------

      IF (.NOT. ASSOCIATED(source_area%d)) &
         DBCSR_WARN("Attempt to copy unassigned data")
      IF (source_area%d%refcount .LE. 0) &
         DBCSR_WARN("Attempt to copy unheld data")
      IF (.NOT. ASSOCIATED(source_area%d)) THEN
         RETURN
      ENDIF
      keep_real = .TRUE.
      IF (PRESENT(drop_real)) keep_real = .NOT. drop_real
      noimult = .TRUE.
      IF (PRESENT(multiply_by_i)) noimult = .NOT. multiply_by_i
      ns = dbcsr_data_get_size_referenced(source_area)
      nt = dbcsr_data_get_size_referenced(target_area)
      n = MIN(ns, nt)
      IF (n .GT. 0) THEN
         SELECT CASE (source_area%d%data_type)
         CASE (dbcsr_type_real_8)
            CALL dbcsr_get_data(source_area, s_data_d)
            SELECT CASE (target_area%d%data_type)
            CASE (dbcsr_type_real_8)
               CALL dbcsr_get_data(target_area, t_data_d)
               t_data_d(1:n) = s_data_d(1:n)
            CASE (dbcsr_type_real_4)
               CALL dbcsr_get_data(target_area, t_data_r)
               t_data_r(1:n) = REAL(s_data_d(1:n), KIND=real_4)
            CASE (dbcsr_type_complex_8)
               CALL dbcsr_get_data(target_area, t_data_z)
               IF (noimult) THEN
                  t_data_z(1:n) = CMPLX(s_data_d(1:n), KIND=real_8)
               ELSE
                  t_data_z(1:n) = CMPLX(0.0, s_data_d(1:n), KIND=real_8)
               ENDIF
            CASE (dbcsr_type_complex_4)
               CALL dbcsr_get_data(target_area, t_data_c)
               IF (noimult) THEN
                  t_data_c(1:n) = CMPLX(s_data_d(1:n), KIND=real_4)
               ELSE
                  t_data_c(1:n) = CMPLX(0.0, s_data_d(1:n), KIND=real_4)
               ENDIF
            CASE default
               DBCSR_ABORT("Invalid data type")
            END SELECT
         CASE (dbcsr_type_real_4)
            CALL dbcsr_get_data(source_area, s_data_r)
            SELECT CASE (target_area%d%data_type)
            CASE (dbcsr_type_real_8)
               CALL dbcsr_get_data(target_area, t_data_d)
               t_data_d(1:n) = REAL(s_data_r(1:n), KIND=real_8)
            CASE (dbcsr_type_real_4)
               CALL dbcsr_get_data(target_area, t_data_r)
               t_data_r(1:n) = s_data_r(1:n)
            CASE (dbcsr_type_complex_8)
               CALL dbcsr_get_data(target_area, t_data_z)
               IF (noimult) THEN
                  t_data_z(1:n) = CMPLX(s_data_r(1:n), KIND=real_8)
               ELSE
                  t_data_z(1:n) = CMPLX(0.0, s_data_r(1:n), KIND=real_8)
               ENDIF
            CASE (dbcsr_type_complex_4)
               CALL dbcsr_get_data(target_area, t_data_c)
               IF (noimult) THEN
                  t_data_c(1:n) = CMPLX(s_data_r(1:n), KIND=real_4)
               ELSE
                  t_data_c(1:n) = CMPLX(0.0, s_data_r(1:n), KIND=real_4)
               ENDIF
            CASE default
               DBCSR_ABORT("Invalid data type")
            END SELECT
         CASE (dbcsr_type_complex_8)
            CALL dbcsr_get_data(source_area, s_data_z)
            SELECT CASE (target_area%d%data_type)
            CASE (dbcsr_type_real_8)
               CALL dbcsr_get_data(target_area, t_data_d)
               IF (keep_real) THEN
                  t_data_d(1:n) = REAL(s_data_z(1:n), KIND=real_8)
               ELSE
                  t_data_d(1:n) = AIMAG(s_data_z(1:n))
               ENDIF
            CASE (dbcsr_type_real_4)
               CALL dbcsr_get_data(target_area, t_data_r)
               IF (keep_real) THEN
                  t_data_r(1:n) = REAL(s_data_z(1:n), KIND=real_4)
               ELSE
                  t_data_r(1:n) = REAL(AIMAG(s_data_z(1:n)), KIND=real_4)
               ENDIF
            CASE (dbcsr_type_complex_8)
               CALL dbcsr_get_data(target_area, t_data_z)
               t_data_z(1:n) = s_data_z(1:n)
            CASE (dbcsr_type_complex_4)
               CALL dbcsr_get_data(target_area, t_data_c)
               t_data_c(1:n) = CMPLX(s_data_z(1:n), KIND=real_4)
            CASE default
               DBCSR_ABORT("Invalid data type")
            END SELECT
         CASE (dbcsr_type_complex_4)
            CALL dbcsr_get_data(source_area, s_data_c)
            SELECT CASE (target_area%d%data_type)
            CASE (dbcsr_type_real_8)
               CALL dbcsr_get_data(target_area, t_data_d)
               IF (keep_real) THEN
                  t_data_d(1:n) = REAL(s_data_c(1:n), KIND=real_8)
               ELSE
                  t_data_d(1:n) = REAL(AIMAG(s_data_c(1:n)), KIND=real_8)
               ENDIF
            CASE (dbcsr_type_real_4)
               CALL dbcsr_get_data(target_area, t_data_r)
               IF (keep_real) THEN
                  t_data_r(1:n) = REAL(s_data_c(1:n), KIND=real_4)
               ELSE
                  t_data_r(1:n) = AIMAG(s_data_c(1:n))
               ENDIF
            CASE (dbcsr_type_complex_8)
               CALL dbcsr_get_data(target_area, t_data_z)
               t_data_z(1:n) = CMPLX(s_data_c(1:n), KIND=real_8)
            CASE (dbcsr_type_complex_4)
               CALL dbcsr_get_data(target_area, t_data_c)
               t_data_c(1:n) = s_data_c(1:n)
            CASE default
               DBCSR_ABORT("Invalid data type")
            END SELECT
         CASE default
            DBCSR_ABORT("Invalid data type")
         END SELECT
      ENDIF
   END SUBROUTINE dbcsr_data_convert

   SUBROUTINE dbcsr_copy_sort_data(blk_p, old_blk_p, row_p, col_i, rbs, cbs, &
                                   dst, src, mark_transposed, transpose_blocks)
      !! Sorts the data in a matrix so that the data blocks follow
      !! sequentially and does various transposing options.
      !! As opposed to dbcsr_sort_data, this routine calculates block sizes

      INTEGER, DIMENSION(:), INTENT(INOUT)               :: blk_p
         !! re-arranged block pointers reflecting the new data order
      INTEGER, DIMENSION(:), INTENT(IN)                  :: old_blk_p, row_p, col_i, rbs, cbs
         !! current block pointers
         !! index
         !! index
         !! sizes of the blocked rows
         !! sizes of the blocked columns
      TYPE(dbcsr_data_obj), INTENT(INOUT)                :: dst
         !! sorted data
      TYPE(dbcsr_data_obj), INTENT(IN)                   :: src
         !! existing unordered data
      LOGICAL, INTENT(IN), OPTIONAL                      :: mark_transposed, transpose_blocks
         !! mark data as transposed by negating the blk_p index entries
         !! transpose data blocks

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_copy_sort_data', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: blk, col_size, nblks, nrows, nze, &
                                                            nze_prev, row, row_size
      LOGICAL                                            :: mark, trb

!   ---------------------------------------------------------------------------
! Analyze parameters

      mark = .FALSE.
      IF (PRESENT(mark_transposed)) mark = mark_transposed
      trb = .FALSE.
      IF (PRESENT(transpose_blocks)) trb = transpose_blocks
      !
      nblks = SIZE(old_blk_p)
      nrows = SIZE(row_p) - 1
      IF (SIZE(blk_p) < nblks) &
         DBCSR_ABORT('Destination blk_p too small.')
      IF (nblks .GE. 1) &
         blk_p(1) = SGN(1, old_blk_p(1), mark)
      nze_prev = 0
      DO row = 1, nrows
         row_size = rbs(row)
         DO blk = row_p(row) + 1, row_p(row + 1)
            IF (old_blk_p(blk) .NE. 0) THEN
               col_size = cbs(col_i(blk))
               nze = row_size*col_size
               IF (blk .GT. 1) THEN
                  blk_p(blk) = SGN(ABS(blk_p(blk - 1)) + nze_prev, old_blk_p(blk), &
                                   mark)
               ENDIF
               IF (ABS(blk_p(blk)) + nze - 1 > dbcsr_data_get_size(dst)) &
                  DBCSR_ABORT('Destination data space is too small.')
               IF (.NOT. trb) THEN
                  CALL dbcsr_data_copy(dst=dst, dst_lb=(/ABS(blk_p(blk))/), &
                                       dst_sizes=(/nze/), &
                                       src=src, src_lb=(/ABS(old_blk_p(blk))/), &
                                       src_sizes=(/nze/))
                  !CALL dbcsr_data_set (dst, ABS(blk_p(blk)), nze,&
                  !     src, source_lb=ABS(old_blk_p(blk)))
               ELSE
                  CALL dbcsr_block_transpose(dst, src, &
                                             col_size, row_size, &
                                             lb=ABS(blk_p(blk)), source_lb=ABS(old_blk_p(blk)))
               ENDIF
               nze_prev = nze
            ENDIF ! blk exists
         ENDDO ! blk
      ENDDO ! row
   END SUBROUTINE dbcsr_copy_sort_data

   SUBROUTINE dbcsr_sort_data(blk_p, old_blk_p, sizes, dsts, src, &
                              srcs, old_blk_d)
      !! Sorts the data in a matrix so that the data blocks follow
      !! sequentially.

      INTEGER, DIMENSION(:), INTENT(INOUT)               :: blk_p
         !! re-arranged block pointers reflecting the new data order
      INTEGER, DIMENSION(:), INTENT(IN)                  :: old_blk_p, sizes
         !! current block pointers
         !! sizes of the data blocks
      TYPE(dbcsr_data_obj), INTENT(INOUT)                :: dsts
         !! sorted data
      TYPE(dbcsr_data_obj), INTENT(IN)                   :: src
         !! existing unordered data
      TYPE(dbcsr_data_obj), DIMENSION(:), INTENT(IN), &
         OPTIONAL                                        :: srcs
         !! multiple source areas
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL        :: old_blk_d

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_sort_data', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: handle, i, nblks
      LOGICAL                                            :: multidata

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)
      multidata = PRESENT(srcs) .AND. PRESENT(old_blk_d)
      nblks = SIZE(old_blk_p)
      IF (nblks .GT. 0) THEN
!$OMP        BARRIER
!$OMP        MASTER
         blk_p(1) = SIGN(1, old_blk_p(1))
         DO i = 2, nblks
            blk_p(i) = SIGN(ABS(blk_p(i - 1)) + sizes(i - 1), old_blk_p(i))
         ENDDO
         CALL dbcsr_data_set_size_referenced(dsts, &
                                             ABS(blk_p(nblks)) + sizes(nblks) - 1)
!$OMP        END MASTER
!$OMP        BARRIER
!$OMP        DO
         DO i = 1, nblks
            IF (old_blk_p(i) .NE. 0) THEN
               IF (.NOT. multidata) THEN
                  CALL dbcsr_data_set(dsts, &
                                      ABS(blk_p(i)), sizes(i), &
                                      src, source_lb=ABS(old_blk_p(i)))
                  !dst(ABS(blk_p(i)):ABS(blk_p(i))+sizes(i)-1) =&
                  !     src(ABS(old_blk_p(i)):ABS(old_blk_p(i))+sizes(i)-1)
               ELSE
                  CALL dbcsr_data_set(dsts, &
                                      ABS(blk_p(i)), sizes(i), &
                                      srcs(old_blk_d(i)), source_lb=ABS(old_blk_p(i)))
                  !dst(ABS(blk_p(i)):ABS(blk_p(i))+sizes(i)-1) =&
                  !     srcs(old_blk_d(i))%d&
                  !     %r_dp(ABS(old_blk_p(i)):ABS(old_blk_p(i))+sizes(i)-1)
               ENDIF
            ENDIF
         ENDDO
!$OMP        END DO NOWAIT
      ENDIF
      CALL timestop(handle)
   END SUBROUTINE dbcsr_sort_data

END MODULE dbcsr_data_operations
