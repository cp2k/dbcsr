!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

#:include '../data/dbcsr.fypp'
#:for n, nametype1, base1, prec1, kind1, type1, dkind1 in inst_params_float
  PURE_BLOCKOPS SUBROUTINE block_partial_copy_${nametype1}$ (dst, dst_rs, dst_cs, dst_tr, &
                                                             src, src_rs, src_cs, src_tr, &
                                                             dst_r_lb, dst_c_lb, src_r_lb, src_c_lb, nrow, ncol, &
                                                             dst_offset, src_offset)
     !! Copies a block subset
     !! @note see block_partial_copy_a

#if defined(__LIBXSMM_BLOCKOPS)
     USE libxsmm, ONLY: libxsmm_matcopy, libxsmm_otrans, libxsmm_ptr0
#endif
     ${type1}$, DIMENSION(:), &
        INTENT(INOUT)                         :: dst
     INTEGER, INTENT(IN)                      :: dst_rs, dst_cs
     INTEGER, INTENT(IN)                      :: src_offset, dst_offset
     LOGICAL, INTENT(IN)                      :: dst_tr
     ${type1}$, DIMENSION(:), &
        INTENT(IN)                            :: src
     INTEGER, INTENT(IN)                      :: src_rs, src_cs
     LOGICAL, INTENT(IN)                      :: src_tr
     INTEGER, INTENT(IN)                      :: dst_r_lb, dst_c_lb, src_r_lb, &
                                                 src_c_lb, nrow, ncol

     CHARACTER(len=*), PARAMETER :: routineN = 'block_partial_copy_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER                                  :: col, row
!    ---------------------------------------------------------------------------
!    Factors out the 4 combinations to remove branches from the inner loop.
!    rs is the logical row size so it always remains the leading dimension.
     IF (.NOT. dst_tr .AND. .NOT. src_tr) THEN
#if defined(__LIBXSMM_BLOCKOPS)
        IF ((0 .LT. ncol) .AND. (0 .LT. nrow)) THEN
           CALL libxsmm_matcopy(libxsmm_ptr0(dst(dst_offset + dst_r_lb + (dst_c_lb - 1)*dst_rs)), &
                                libxsmm_ptr0(src(src_offset + src_r_lb + (src_c_lb - 1)*src_rs)), &
                                ${typesize1[n]}$, nrow, ncol, src_rs, dst_rs)
        ENDIF
#else
        DO col = 0, ncol - 1
           DO row = 0, nrow - 1
              dst(dst_offset + dst_r_lb + row + (dst_c_lb + col - 1)*dst_rs) &
                 = src(src_offset + src_r_lb + row + (src_c_lb + col - 1)*src_rs)
           END DO
        END DO
#endif
     ELSEIF (dst_tr .AND. .NOT. src_tr) THEN
        DO col = 0, ncol - 1
           DO row = 0, nrow - 1
              dst(dst_offset + dst_c_lb + col + (dst_r_lb + row - 1)*dst_cs) &
                 = src(src_offset + src_r_lb + row + (src_c_lb + col - 1)*src_rs)
           END DO
        END DO
     ELSEIF (.NOT. dst_tr .AND. src_tr) THEN
        DO col = 0, ncol - 1
           DO row = 0, nrow - 1
              dst(dst_offset + dst_r_lb + row + (dst_c_lb + col - 1)*dst_rs) &
                 = src(src_offset + src_c_lb + col + (src_r_lb + row - 1)*src_cs)
           END DO
        END DO
     ELSE
        DBCSR_ASSERT(dst_tr .AND. src_tr)
#if defined(__LIBXSMM_BLOCKOPS)
        IF ((0 .LT. ncol) .AND. (0 .LT. nrow)) THEN
           CALL libxsmm_matcopy(libxsmm_ptr0(dst(dst_offset + dst_c_lb + (dst_r_lb - 1)*dst_cs)), &
                                libxsmm_ptr0(src(src_offset + src_c_lb + (src_r_lb - 1)*src_cs)), &
                                ${typesize1[n]}$, nrow, ncol, src_cs, dst_cs)
        END IF
#else
        DO col = 0, ncol - 1
           DO row = 0, nrow - 1
              dst(dst_offset + dst_c_lb + col + (dst_r_lb + row - 1)*dst_cs) &
                 = src(src_offset + src_c_lb + col + (src_r_lb + row - 1)*src_cs)
           END DO
        END DO
#endif
     ENDIF
  END SUBROUTINE block_partial_copy_${nametype1}$

  PURE_BLOCKOPS SUBROUTINE block_partial_copy_1d2d_${nametype1}$ (dst, dst_rs, dst_cs, dst_tr, &
                                                                  src, src_tr, &
                                                                  dst_r_lb, dst_c_lb, src_r_lb, src_c_lb, nrow, ncol, &
                                                                  dst_offset)
     !! Copies a block subset
     !! @note see block_partial_copy_a

#if defined(__LIBXSMM_BLOCKOPS)
     USE libxsmm, ONLY: libxsmm_matcopy, libxsmm_otrans, libxsmm_ptr0
#else
     INTEGER                                  :: col, row
#endif
     ${type1}$, DIMENSION(:), &
        INTENT(INOUT)                         :: dst
     INTEGER, INTENT(IN)                      :: dst_rs, dst_cs
     INTEGER, INTENT(IN)                      :: dst_offset
     LOGICAL, INTENT(IN)                      :: dst_tr
     ${type1}$, DIMENSION(:, :), &
        INTENT(IN)                            :: src
     LOGICAL, INTENT(IN)                      :: src_tr
     INTEGER, INTENT(IN)                      :: dst_r_lb, dst_c_lb, src_r_lb, &
                                                 src_c_lb, nrow, ncol

     CHARACTER(len=*), PARAMETER :: routineN = 'block_partial_copy_1d2d_${nametype1}$', &
                                    routineP = moduleN//':'//routineN
!    ---------------------------------------------------------------------------
!    Factors out the 4 combinations to remove branches from the inner loop.
!    rs is the logical row size so it always remains the leading dimension.
     IF (.NOT. dst_tr .AND. .NOT. src_tr) THEN
#if defined(__LIBXSMM_BLOCKOPS)
        IF ((0 .LT. ncol) .AND. (0 .LT. nrow)) THEN
           CALL libxsmm_matcopy(libxsmm_ptr0(dst(dst_offset + dst_r_lb + (dst_c_lb - 1)*dst_rs)), &
                                libxsmm_ptr0(src(src_r_lb, src_c_lb)), &
                                ${typesize1[n]}$, nrow, ncol, SIZE(src, 1), dst_rs)
        ENDIF
#else
        DO col = 0, ncol - 1
           DO row = 0, nrow - 1
              dst(dst_offset + dst_r_lb + row + (dst_c_lb + col - 1)*dst_rs) &
                 = src(src_r_lb + row, src_c_lb + col)
           END DO
        END DO
#endif
     ELSEIF (dst_tr .AND. .NOT. src_tr) THEN
#if defined(__LIBXSMM_BLOCKOPS)
        IF ((0 .LT. ncol) .AND. (0 .LT. nrow)) THEN
           CALL libxsmm_otrans(libxsmm_ptr0(dst(dst_offset + dst_c_lb + (dst_r_lb - 1)*dst_cs)), &
                               libxsmm_ptr0(src(src_r_lb, src_c_lb)), &
                               ${typesize1[n]}$, nrow, ncol, SIZE(src, 1), dst_cs)
        ENDIF
#else
        DO col = 0, ncol - 1
           DO row = 0, nrow - 1
              dst(dst_offset + dst_c_lb + col + (dst_r_lb + row - 1)*dst_cs) &
                 = src(src_r_lb + row, src_c_lb + col)
           END DO
        END DO
#endif
     ELSEIF (.NOT. dst_tr .AND. src_tr) THEN
#if defined(__LIBXSMM_BLOCKOPS)
        IF ((0 .LT. ncol) .AND. (0 .LT. nrow)) THEN
           CALL libxsmm_otrans(libxsmm_ptr0(dst(dst_offset + dst_r_lb + (dst_c_lb - 1)*dst_rs)), &
                               libxsmm_ptr0(src(src_c_lb, src_r_lb)), &
                               ${typesize1[n]}$, nrow, ncol, SIZE(src, 2), dst_rs)
        ENDIF
#else
        DO col = 0, ncol - 1
           DO row = 0, nrow - 1
              dst(dst_offset + dst_r_lb + row + (dst_c_lb + col - 1)*dst_rs) &
                 = src(src_c_lb + col, src_r_lb + row)
           END DO
        END DO
#endif
     ELSE
        DBCSR_ASSERT(dst_tr .AND. src_tr)
#if defined(__LIBXSMM_BLOCKOPS)
        IF ((0 .LT. ncol) .AND. (0 .LT. nrow)) THEN
           CALL libxsmm_matcopy(libxsmm_ptr0(dst(dst_offset + dst_c_lb + (dst_r_lb - 1)*dst_cs)), &
                                libxsmm_ptr0(src(src_c_lb, src_r_lb)), &
                                ${typesize1[n]}$, nrow, ncol, SIZE(src, 2), dst_cs)
        ENDIF
#else
        DO col = 0, ncol - 1
           DO row = 0, nrow - 1
              dst(dst_offset + dst_c_lb + col + (dst_r_lb + row - 1)*dst_cs) &
                 = src(src_c_lb + col, src_r_lb + row)
           END DO
        END DO
#endif
     ENDIF
  END SUBROUTINE block_partial_copy_1d2d_${nametype1}$

  PURE_BLOCKOPS SUBROUTINE block_partial_copy_2d1d_${nametype1}$ (dst, dst_tr, &
                                                                  src, src_rs, src_cs, src_tr, &
                                                                  dst_r_lb, dst_c_lb, src_r_lb, src_c_lb, nrow, ncol, &
                                                                  src_offset)
     !! Copies a block subset
     !! @note see block_partial_copy_a

#if defined(__LIBXSMM_BLOCKOPS)
     USE libxsmm, ONLY: libxsmm_matcopy, libxsmm_otrans, libxsmm_ptr0
#endif
     ${type1}$, DIMENSION(:, :), &
        INTENT(INOUT)                         :: dst
     INTEGER, INTENT(IN)                      :: src_offset
     LOGICAL, INTENT(IN)                      :: dst_tr
     ${type1}$, DIMENSION(:), &
        INTENT(IN)                            :: src
     INTEGER, INTENT(IN)                      :: src_rs, src_cs
     LOGICAL, INTENT(IN)                      :: src_tr
     INTEGER, INTENT(IN)                      :: dst_r_lb, dst_c_lb, src_r_lb, &
                                                 src_c_lb, nrow, ncol

     CHARACTER(len=*), PARAMETER :: routineN = 'block_partial_copy_2d1d_${nametype1}$', &
                                    routineP = moduleN//':'//routineN

     INTEGER                                  :: col, row
!    ---------------------------------------------------------------------------
!    Factors out the 4 combinations to remove branches from the inner loop.
!    rs is the logical row size so it always remains the leading dimension.
     IF (.NOT. dst_tr .AND. .NOT. src_tr) THEN
        DO col = 0, ncol - 1
           DO row = 0, nrow - 1
              dst(dst_r_lb + row, dst_c_lb + col) &
                 = src(src_offset + src_r_lb + row + (src_c_lb + col - 1)*src_rs)
           END DO
        END DO
     ELSEIF (dst_tr .AND. .NOT. src_tr) THEN
        DO col = 0, ncol - 1
           DO row = 0, nrow - 1
              dst(dst_c_lb + col, dst_r_lb + row) &
                 = src(src_offset + src_r_lb + row + (src_c_lb + col - 1)*src_rs)
           END DO
        END DO
     ELSEIF (.NOT. dst_tr .AND. src_tr) THEN
        DO col = 0, ncol - 1
           DO row = 0, nrow - 1
              dst(dst_r_lb + row, dst_c_lb + col) &
                 = src(src_offset + src_c_lb + col + (src_r_lb + row - 1)*src_cs)
           END DO
        END DO
     ELSE
        DBCSR_ASSERT(dst_tr .AND. src_tr)
        DO col = 0, ncol - 1
           DO row = 0, nrow - 1
              dst(dst_c_lb + col, dst_r_lb + row) &
                 = src(src_offset + src_c_lb + col + (src_r_lb + row - 1)*src_cs)
           END DO
        END DO
     ENDIF
  END SUBROUTINE block_partial_copy_2d1d_${nametype1}$

  PURE_BLOCKOPS SUBROUTINE block_partial_copy_2d2d_${nametype1}$ (dst, dst_tr, &
                                                                  src, src_tr, &
                                                                  dst_r_lb, dst_c_lb, src_r_lb, src_c_lb, nrow, ncol)
     !! Copies a block subset
     !! @note see block_partial_copy_a

#if defined(__LIBXSMM_BLOCKOPS)
     USE libxsmm, ONLY: libxsmm_matcopy, libxsmm_otrans, libxsmm_ptr0
#else
     INTEGER                                  :: col, row
#endif
     ${type1}$, DIMENSION(:, :), &
        INTENT(INOUT)                         :: dst
     LOGICAL, INTENT(IN)                      :: dst_tr
     ${type1}$, DIMENSION(:, :), &
        INTENT(IN)                            :: src
     LOGICAL, INTENT(IN)                      :: src_tr
     INTEGER, INTENT(IN)                      :: dst_r_lb, dst_c_lb, src_r_lb, &
                                                 src_c_lb, nrow, ncol

     CHARACTER(len=*), PARAMETER :: routineN = 'block_partial_copy_2d2d_${nametype1}$', &
                                    routineP = moduleN//':'//routineN
!    ---------------------------------------------------------------------------
!    Factors out the 4 combinations to remove branches from the inner loop.
!    rs is the logical row size so it always remains the leading dimension.
     IF (.NOT. dst_tr .AND. .NOT. src_tr) THEN
#if defined(__LIBXSMM_BLOCKOPS)
        IF ((0 .LT. ncol) .AND. (0 .LT. nrow)) THEN
           CALL libxsmm_matcopy(libxsmm_ptr0(dst(dst_r_lb, dst_c_lb)), &
                                libxsmm_ptr0(src(src_r_lb, src_c_lb)), &
                                ${typesize1[n]}$, nrow, ncol, &
                                SIZE(src, 1), SIZE(dst, 1))
        ENDIF
#else
        DO col = 0, ncol - 1
           DO row = 0, nrow - 1
              dst(dst_r_lb + row, dst_c_lb + col) &
                 = src(src_r_lb + row, src_c_lb + col)
           END DO
        END DO
#endif
     ELSEIF (dst_tr .AND. .NOT. src_tr) THEN
#if defined(__LIBXSMM_BLOCKOPS)
        IF ((0 .LT. ncol) .AND. (0 .LT. nrow)) THEN
           CALL libxsmm_otrans(libxsmm_ptr0(dst(dst_c_lb, dst_r_lb)), &
                               libxsmm_ptr0(src(src_r_lb, src_c_lb)), &
                               ${typesize1[n]}$, nrow, ncol, &
                               SIZE(src, 1), SIZE(dst, 2))
        ENDIF
#else
        DO col = 0, ncol - 1
           DO row = 0, nrow - 1
              dst(dst_c_lb + col, dst_r_lb + row) &
                 = src(src_r_lb + row, src_c_lb + col)
           END DO
        END DO
#endif
     ELSEIF (.NOT. dst_tr .AND. src_tr) THEN
#if defined(__LIBXSMM_BLOCKOPS)
        IF ((0 .LT. ncol) .AND. (0 .LT. nrow)) THEN
           CALL libxsmm_otrans(libxsmm_ptr0(dst(dst_r_lb, dst_c_lb)), &
                               libxsmm_ptr0(src(src_c_lb, src_r_lb)), &
                               ${typesize1[n]}$, nrow, ncol, &
                               SIZE(src, 2), SIZE(dst, 1))
        ENDIF
#else
        DO col = 0, ncol - 1
           DO row = 0, nrow - 1
              dst(dst_r_lb + row, dst_c_lb + col) &
                 = src(src_c_lb + col, src_r_lb + row)
           END DO
        END DO
#endif
     ELSE
        DBCSR_ASSERT(dst_tr .AND. src_tr)
#if defined(__LIBXSMM_BLOCKOPS)
        IF ((0 .LT. ncol) .AND. (0 .LT. nrow)) THEN
           CALL libxsmm_matcopy(libxsmm_ptr0(dst(dst_c_lb, dst_r_lb)), &
                                libxsmm_ptr0(src(src_c_lb, src_r_lb)), &
                                ${typesize1[n]}$, nrow, ncol, &
                                SIZE(src, 2), SIZE(dst, 2))
        ENDIF
#else
        DO col = 0, ncol - 1
           DO row = 0, nrow - 1
              dst(dst_c_lb + col, dst_r_lb + row) &
                 = src(src_c_lb + col, src_r_lb + row)
           END DO
        END DO
#endif
     ENDIF
  END SUBROUTINE block_partial_copy_2d2d_${nametype1}$

  PURE SUBROUTINE block_copy_${nametype1}$ (extent_out, extent_in, n, out_fe, in_fe)
     !! Copy a block

     INTEGER, INTENT(IN) :: n, out_fe, in_fe
        !! number of elements to copy
        !! first element of output
        !! first element of input
     ${type1}$, DIMENSION(*), INTENT(OUT) :: extent_out
        !! output data
     ${type1}$, DIMENSION(*), INTENT(IN)  :: extent_in
        !! input data

     CHARACTER(len=*), PARAMETER :: routineN = 'block_copy_${nametype1}$', &
                                    routineP = moduleN//':'//routineN
!    ---------------------------------------------------------------------------
     extent_out(out_fe:out_fe + n - 1) = extent_in(in_fe:in_fe + n - 1)
  END SUBROUTINE block_copy_${nametype1}$

  PURE_BLOCKOPS SUBROUTINE block_transpose_copy_${nametype1}$ (extent_out, extent_in, &
                                                               rows, columns)
     !! Copy and transpose block.

#if defined(__LIBXSMM_TRANS)
     USE libxsmm, ONLY: libxsmm_otrans, libxsmm_ptr1
#endif
     ${type1}$, DIMENSION(:), INTENT(OUT), TARGET :: extent_out
        !! output matrix in the form of a 1-d array
     ${type1}$, DIMENSION(:), INTENT(IN)          :: extent_in
        !! input matrix in the form of a 1-d array
     INTEGER, INTENT(IN) :: rows, columns
        !! input matrix size
        !! input matrix size

     CHARACTER(len=*), PARAMETER :: routineN = 'block_transpose_copy_${nametype1}$', &
                                    routineP = moduleN//':'//routineN
!    ---------------------------------------------------------------------------
#if defined(__LIBXSMM_TRANS)
     CALL libxsmm_otrans(libxsmm_ptr1(extent_out), libxsmm_ptr1(extent_in), &
                         ${typesize1[n]}$, rows, columns, rows, columns)
#elif defined(__MKL)
     CALL mkl_${nametype1}$omatcopy('C', 'T', rows, columns, ${one1[n]}$, extent_in, rows, extent_out, columns)
#else
     extent_out(1:rows*columns) = RESHAPE(TRANSPOSE( &
                                          RESHAPE(extent_in(1:rows*columns), (/rows, columns/))), (/rows*columns/))
#endif
  END SUBROUTINE block_transpose_copy_${nametype1}$

  PURE SUBROUTINE block_copy_2d1d_${nametype1}$ (extent_out, extent_in, &
                                                 rows, columns)
     !! Copy a block

     INTEGER, INTENT(IN) :: rows, columns
        !! input matrix size
        !! input matrix size
     ${type1}$, DIMENSION(rows, columns), INTENT(OUT) :: extent_out
        !! output matrix in the form of a 2-d array
     ${type1}$, DIMENSION(:), INTENT(IN)             :: extent_in
        !! input matrix in the form of a 1-d array

     CHARACTER(len=*), PARAMETER :: routineN = 'block_copy_2d1d_${nametype1}$', &
                                    routineP = moduleN//':'//routineN
!    ---------------------------------------------------------------------------
     extent_out = RESHAPE(extent_in, (/rows, columns/))
  END SUBROUTINE block_copy_2d1d_${nametype1}$

  PURE SUBROUTINE block_copy_1d1d_${nametype1}$ (extent_out, extent_in, &
                                                 rows, columns)
     !! Copy a block

     INTEGER, INTENT(IN) :: rows, columns
        !! input matrix size
        !! input matrix size
     ${type1}$, DIMENSION(rows*columns), INTENT(OUT) :: extent_out
        !! output matrix in the form of a 1-d array
     ${type1}$, DIMENSION(rows*columns), INTENT(IN)  :: extent_in
        !! input matrix in the form of a 1-d array

     CHARACTER(len=*), PARAMETER :: routineN = 'block_copy_1d1d_${nametype1}$', &
                                    routineP = moduleN//':'//routineN
!    ---------------------------------------------------------------------------
     extent_out(:) = extent_in(:)
  END SUBROUTINE block_copy_1d1d_${nametype1}$

  PURE SUBROUTINE block_copy_2d2d_${nametype1}$ (extent_out, extent_in, &
                                                 rows, columns)
     !! Copy a block

     INTEGER, INTENT(IN) :: rows, columns
        !! input matrix size
        !! input matrix size
     ${type1}$, DIMENSION(rows, columns), INTENT(OUT) :: extent_out
        !! output matrix in the form of a 2-d array
     ${type1}$, DIMENSION(rows, columns), INTENT(IN)  :: extent_in
        !! input matrix in the form of a 2-d array

     CHARACTER(len=*), PARAMETER :: routineN = 'block_copy_2d2d_${nametype1}$', &
                                    routineP = moduleN//':'//routineN
!    ---------------------------------------------------------------------------
     extent_out(:, :) = extent_in(:, :)
  END SUBROUTINE block_copy_2d2d_${nametype1}$

  PURE_BLOCKOPS SUBROUTINE block_transpose_copy_2d1d_${nametype1}$ (extent_out, extent_in, &
                                                                    rows, columns)
     !! Copy and transpose block.

#if defined(__LIBXSMM_TRANS)
     USE libxsmm, ONLY: libxsmm_otrans, libxsmm_ptr1, libxsmm_ptr2
#endif
     INTEGER, INTENT(IN) :: rows, columns
        !! input matrix size
        !! input matrix size
     ${type1}$, DIMENSION(columns, rows), INTENT(OUT), TARGET :: extent_out
        !! output matrix in the form of a 2-d array
     ${type1}$, DIMENSION(:), INTENT(IN)                      :: extent_in
        !! input matrix in the form of a 1-d array

     CHARACTER(len=*), PARAMETER :: routineN = 'block_transpose_copy_2d1d_${nametype1}$', &
                                    routineP = moduleN//':'//routineN
!    ---------------------------------------------------------------------------
#if defined(__LIBXSMM_TRANS)
     CALL libxsmm_otrans(libxsmm_ptr2(extent_out), libxsmm_ptr1(extent_in), &
                         ${typesize1[n]}$, rows, columns, rows, columns)
#elif defined(__MKL)
     CALL mkl_${nametype1}$omatcopy('C', 'T', rows, columns, ${one1[n]}$, extent_in, rows, extent_out, columns)
#else
     extent_out = TRANSPOSE(RESHAPE(extent_in, (/rows, columns/)))
#endif
  END SUBROUTINE block_transpose_copy_2d1d_${nametype1}$

  PURE SUBROUTINE block_copy_1d2d_${nametype1}$ (extent_out, extent_in, &
                                                 rows, columns)
     !! Copy and transpose block.

     INTEGER, INTENT(IN) :: rows, columns
        !! input matrix size
        !! input matrix size
     ${type1}$, DIMENSION(:), INTENT(OUT)            :: extent_out
        !! output matrix in the form of a 1-d array
     ${type1}$, DIMENSION(rows, columns), INTENT(IN) :: extent_in
        !! input matrix in the form of a 2-d array

     CHARACTER(len=*), PARAMETER :: routineN = 'block_copy_1d2d_${nametype1}$', &
                                    routineP = moduleN//':'//routineN
!    ---------------------------------------------------------------------------
     extent_out = RESHAPE(extent_in, (/rows*columns/))
  END SUBROUTINE block_copy_1d2d_${nametype1}$

  PURE_BLOCKOPS SUBROUTINE block_transpose_copy_1d2d_${nametype1}$ (extent_out, extent_in, &
                                                                    rows, columns)
     !! Copy and transpose block.

#if defined(__LIBXSMM_TRANS)
     USE libxsmm, ONLY: libxsmm_otrans, libxsmm_ptr1, libxsmm_ptr2
#endif
     INTEGER, INTENT(IN) :: rows, columns
        !! input matrix size
        !! input matrix size
     ${type1}$, DIMENSION(:), INTENT(OUT), TARGET    :: extent_out
        !! output matrix in the form of a 1-d array
     ${type1}$, DIMENSION(rows, columns), INTENT(IN) :: extent_in
        !! input matrix in the form of a 2-d array

     CHARACTER(len=*), PARAMETER :: routineN = 'block_transpose_copy_1d2d_${nametype1}$', &
                                    routineP = moduleN//':'//routineN
!    ---------------------------------------------------------------------------
#if defined(__LIBXSMM_TRANS)
     CALL libxsmm_otrans(libxsmm_ptr1(extent_out), libxsmm_ptr2(extent_in), &
                         ${typesize1[n]}$, rows, columns, rows, columns)
#elif defined(__MKL)
     CALL mkl_${nametype1}$omatcopy('C', 'T', rows, columns, ${one1[n]}$, extent_in, rows, extent_out, columns)
#else
     extent_out = RESHAPE(TRANSPOSE(extent_in), (/rows*columns/))
#endif
  END SUBROUTINE block_transpose_copy_1d2d_${nametype1}$

  PURE_BLOCKOPS SUBROUTINE block_transpose_inplace_${nametype1}$ (extent, rows, columns)
     !! In-place block transpose.

#if defined(__LIBXSMM_TRANS) && 0
     USE libxsmm, ONLY: libxsmm_itrans, libxsmm_ptr1
#endif
     INTEGER, INTENT(IN) :: rows, columns
        !! input matrix size
        !! input matrix size
     ${type1}$, DIMENSION(rows*columns), INTENT(INOUT) :: extent
        !! Matrix in the form of a 1-d array

     CHARACTER(len=*), PARAMETER :: routineN = 'block_transpose_inplace_${nametype1}$', &
                                    routineP = moduleN//':'//routineN
!    ---------------------------------------------------------------------------
#if defined(__LIBXSMM_TRANS) && 0
     CALL libxsmm_itrans(libxsmm_ptr1(extent), ${typesize1[n]}$, rows, columns, rows)
#elif defined(__MKL)
     CALL mkl_${nametype1}$imatcopy('C', 'T', rows, columns, ${one1[n]}$, extent, rows, columns)
#else
     ${type1}$, DIMENSION(rows*columns) :: extent_tr
     INTEGER :: r, c
     DO r = 1, columns
        DO c = 1, rows
           extent_tr(r + (c - 1)*columns) = extent(c + (r - 1)*rows)
        END DO
     END DO
     DO r = 1, columns
        DO c = 1, rows
           extent(r + (c - 1)*columns) = extent_tr(r + (c - 1)*columns)
        END DO
     END DO
#endif
  END SUBROUTINE block_transpose_inplace_${nametype1}$

  SUBROUTINE dbcsr_data_set_a${nametype1}$ (dst, lb, data_size, src, source_lb)
     !! Copy data from a double real array to a data area
     !! There are no checks done for correctness!

     TYPE(dbcsr_data_obj), INTENT(INOUT)      :: dst
        !! destination data area
     INTEGER, INTENT(IN)                      :: lb, data_size
        !! lower bound for destination (and source if not given explicitly)
        !! number of elements to copy
     ${type1}$, DIMENSION(:), INTENT(IN)      :: src
        !! source data array
     INTEGER, INTENT(IN), OPTIONAL            :: source_lb
        !! lower bound of source
     CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_data_set_a${nametype1}$', &
                                    routineP = moduleN//':'//routineN
     INTEGER                                  :: lb_s, ub, ub_s
!    ---------------------------------------------------------------------------
     IF (debug_mod) THEN
        IF (.NOT. ASSOCIATED(dst%d)) &
           DBCSR_ABORT("Target data area must be setup.")
        IF (SIZE(src) .LT. data_size) &
           DBCSR_ABORT("Not enough source data.")
        IF (dst%d%data_type .NE. ${dkind1}$) &
           DBCSR_ABORT("Data type mismatch.")
     ENDIF
     ub = lb + data_size - 1
     IF (PRESENT(source_lb)) THEN
        lb_s = source_lb
        ub_s = source_lb + data_size - 1
     ELSE
        lb_s = lb
        ub_s = ub
     ENDIF
     CALL memory_copy(dst%d%${base1}$_${prec1}$ (lb:ub), src(lb_s:ub_s), data_size)
  END SUBROUTINE dbcsr_data_set_a${nametype1}$

  PURE SUBROUTINE block_add_${nametype1}$ (block_a, block_b, len)
     INTEGER, INTENT(IN) :: len
     ${type1}$, DIMENSION(len), INTENT(INOUT) :: block_a
     ${type1}$, DIMENSION(len), INTENT(IN)    :: block_b
     block_a(1:len) = block_a(1:len) + block_b(1:len)
  END SUBROUTINE block_add_${nametype1}$
#:endfor
