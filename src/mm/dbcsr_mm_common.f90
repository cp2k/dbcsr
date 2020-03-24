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
  SUBROUTINE calc_norms_${nametype1}$ (norms, nblks, &
                                       blki, rbs, cbs, DATA)
     !! Calculates norms of the entire matrix with minimal overhead.
     REAL(kind=sp), DIMENSION(:), INTENT(OUT) :: norms
     INTEGER, INTENT(IN)                      :: nblks
     INTEGER, DIMENSION(3, nblks), INTENT(IN) :: blki
     INTEGER, DIMENSION(:), INTENT(IN)        :: rbs, cbs
     ${type1}$, DIMENSION(:), &
        INTENT(IN)                            :: DATA

     INTEGER, PARAMETER                       :: simd = 64 / ${typesize1}$
     INTEGER                                  :: i, n, blk, bp, bpe, row, col
     REAL(kind=sp)                            :: val

!   ---------------------------------------------------------------------------

!$OMP     parallel default(none) &
!$OMP              private (i, n, row, col, blk, bp, bpe, val) &
!$OMP              shared (nblks, simd) &
!$OMP              shared (rbs, cbs, blki, &
!$OMP                      data, norms)
!$OMP     do
     DO i = 1, nblks, simd
        n = MIN(i + simd, nblks)
        DO blk = i, n
           bp = blki(3, blk)
           IF (bp .NE. 0) THEN
              row = blki(1, blk)
              col = blki(2, blk)
              bpe = bp + rbs(row) * cbs(col) - 1
              val = SQRT(REAL(SUM(DATA(bp:bpe)**2), KIND=sp))
           ELSE
              val = 0.0_sp
           ENDIF
           norms(blk) = val
        ENDDO
     ENDDO
!$OMP     end do
!$OMP     end parallel
  END SUBROUTINE calc_norms_${nametype1}$
#:endfor
