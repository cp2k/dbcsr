!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

#:include '../data/dbcsr.fypp'
#:for n, nametype1, base1, prec1, kind1, type1, typesize1, dkind1 in inst_params_float
  SUBROUTINE calc_norms_${nametype1}$ (norms, nblks, &
                                       blki, rbs, cbs, DATA)
     !! Calculates norms of the entire matrix with minimal overhead.
     REAL(kind=sp), DIMENSION(:), INTENT(OUT) :: norms
     INTEGER, INTENT(IN)                      :: nblks
     INTEGER, DIMENSION(3, nblks), INTENT(IN) :: blki
     INTEGER, DIMENSION(:), INTENT(IN)        :: rbs, cbs
     ${type1}$, DIMENSION(:), &
        INTENT(IN)                            :: DATA

     INTEGER, PARAMETER                       :: nsimd = (2*64)/${typesize1}$
     INTEGER                                  :: i, n, blk, bp, bpe, row, col
     REAL(kind=sp)                            :: vals(0:nsimd - 1)

!   ---------------------------------------------------------------------------

!$OMP     parallel default(none) &
!$OMP              private (i, n, row, col, blk, bp, bpe, vals) &
!$OMP              shared (nblks, rbs, cbs, blki, data, norms)
!$OMP     do
     DO i = 1, nblks, nsimd
        n = MIN(nsimd - 1, nblks - i)
        DO blk = 0, n
           bp = blki(3, blk + i)
           IF (bp .NE. 0) THEN
              row = blki(1, blk + i)
              col = blki(2, blk + i)
              bpe = bp + rbs(row)*cbs(col) - 1
              vals(blk) = REAL(SUM(DATA(bp:bpe)**2), KIND=sp)
           ELSE
              vals(blk) = 0.0_sp
           ENDIF
        ENDDO
        ! SIMD: SQRT is not part of above IF-condition
        IF (n .EQ. (nsimd - 1)) THEN
!$OMP     simd
           DO blk = 0, nsimd - 1
              norms(blk + i) = SQRT(vals(blk))
           ENDDO
!$OMP     end simd
        ELSE ! remainder
           DO blk = 0, n
              norms(blk + i) = SQRT(vals(blk))
           ENDDO
        ENDIF
     ENDDO
!$OMP     end do
!$OMP     end parallel
  END SUBROUTINE calc_norms_${nametype1}$
#:endfor
