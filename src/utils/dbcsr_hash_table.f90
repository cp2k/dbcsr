!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

! -----------------------------------------------------------------------------
! Beginning of hashtable.
! this file can be 'INCLUDE'ed verbatim in various place, where it needs to be
! part of the module to guarantee inlining
! hashes (c,p) pairs, where p is assumed to be >0
! on return (0 is used as a flag for not present)
!
!
  FUNCTION matching_prime(i) RESULT(res)
     !! finds a prime equal or larger than i, needed at creation

     INTEGER, INTENT(IN)                      :: i
     INTEGER                                  :: res

     INTEGER                                  :: j

     res = i
     j = 0
     DO WHILE (j < res)
        DO j = 2, res - 1
           IF (MOD(res, j) == 0) THEN
              res = res + 1
              EXIT
           ENDIF
        ENDDO
     ENDDO
  END FUNCTION

  SUBROUTINE hash_table_create(hash_table, table_size)
     !! create a hash_table of given initial size.
     !! the hash table will expand as needed (but this requires rehashing)

     TYPE(hash_table_type)                    :: hash_table
     INTEGER, INTENT(IN)                      :: table_size

     INTEGER                                  :: j

     ! guarantee a minimal hash table size (8), so that expansion works

     j = 3
     DO WHILE (2**j - 1 < table_size)
        j = j + 1
     ENDDO
     hash_table%nmax = 2**j - 1
     hash_table%prime = matching_prime(hash_table%nmax)
     hash_table%nele = 0
     ALLOCATE (hash_table%table(0:hash_table%nmax))
  END SUBROUTINE hash_table_create

! *****************************************************************************
! **************************************************************************************************
  SUBROUTINE hash_table_release(hash_table)
     TYPE(hash_table_type)                    :: hash_table

     hash_table%nmax = 0
     hash_table%nele = 0
     DEALLOCATE (hash_table%table)

  END SUBROUTINE hash_table_release

  RECURSIVE SUBROUTINE hash_table_add(hash_table, c, p)
     !! add a pair (c,p) to the hash table

     TYPE(hash_table_type), INTENT(INOUT)     :: hash_table
     INTEGER, INTENT(IN)                      :: c, p
        !! this value is being hashed
        !! this is being stored

     REAL(KIND=sp), PARAMETER :: hash_table_expand = 1.5_sp, &
                                 inv_hash_table_fill = 2.5_sp

     INTEGER                                  :: i, j
     TYPE(ele_type), ALLOCATABLE, &
        DIMENSION(:)                           :: tmp_hash

! if too small, make a copy and rehash in a larger table

     IF (hash_table%nele*inv_hash_table_fill > hash_table%nmax) THEN
        ALLOCATE (tmp_hash(LBOUND(hash_table%table, 1):UBOUND(hash_table%table, 1)))
        tmp_hash(:) = hash_table%table
        CALL hash_table_release(hash_table)
        CALL hash_table_create(hash_table, INT((UBOUND(tmp_hash, 1) + 8)*hash_table_expand))
        DO i = LBOUND(tmp_hash, 1), UBOUND(tmp_hash, 1)
           IF (tmp_hash(i)%c .NE. 0) THEN
              CALL hash_table_add(hash_table, tmp_hash(i)%c, tmp_hash(i)%p)
           ENDIF
        ENDDO
        DEALLOCATE (tmp_hash)
     ENDIF

     hash_table%nele = hash_table%nele + 1
     i = IAND(c*hash_table%prime, hash_table%nmax)

     DO j = i, hash_table%nmax
        IF (hash_table%table(j)%c == 0 .OR. hash_table%table(j)%c == c) THEN
           hash_table%table(j)%c = c
           hash_table%table(j)%p = p
           RETURN
        ENDIF
     ENDDO
     DO j = 0, i - 1
        IF (hash_table%table(j)%c == 0 .OR. hash_table%table(j)%c == c) THEN
           hash_table%table(j)%c = c
           hash_table%table(j)%p = p
           RETURN
        ENDIF
     ENDDO

  END SUBROUTINE hash_table_add

! *****************************************************************************
! **************************************************************************************************
  PURE FUNCTION hash_table_get(hash_table, c) RESULT(p)
     TYPE(hash_table_type), INTENT(IN)        :: hash_table
     INTEGER, INTENT(IN)                      :: c
     INTEGER                                  :: p

     INTEGER                                  :: i, j

     i = IAND(c*hash_table%prime, hash_table%nmax)

     ! catch the likely case first
     IF (hash_table%table(i)%c == c) THEN
        p = hash_table%table(i)%p
        RETURN
     ENDIF

     DO j = i, hash_table%nmax
        IF (hash_table%table(j)%c == 0 .OR. hash_table%table(j)%c == c) THEN
           p = hash_table%table(j)%p
           RETURN
        ENDIF
     ENDDO
     DO j = 0, i - 1
        IF (hash_table%table(j)%c == 0 .OR. hash_table%table(j)%c == c) THEN
           p = hash_table%table(j)%p
           RETURN
        ENDIF
     ENDDO

     ! we should never reach this point.
     p = HUGE(p)

  END FUNCTION hash_table_get

! End of hashtable
! -----------------------------------------------------------------------------
