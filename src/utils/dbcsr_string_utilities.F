!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_string_utilities
   !! Utilities for string manipulations

   USE dbcsr_kinds, ONLY: default_blank_character

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: compress, &
             stringify, &
             uppercase, &
             str2int

   INTERFACE stringify
      MODULE PROCEDURE integer_to_string
   END INTERFACE
CONTAINS

   SUBROUTINE compress(string, full)
      !! Eliminate multiple space characters in a string.
      !! If full is .TRUE., then all spaces are eliminated.

      CHARACTER(LEN=*), INTENT(INOUT)                    :: string
      LOGICAL, INTENT(IN), OPTIONAL                      :: full

      CHARACTER                                          :: tmp
      INTEGER                                            :: i, z
      LOGICAL                                            :: remove_all

      IF (PRESENT(full)) THEN
         remove_all = full
      ELSE
         remove_all = .FALSE.
      END IF

      z = 1

      DO i = 1, LEN_TRIM(string)
         IF ((z == 1) .OR. remove_all) THEN
            IF (string(i:i) /= " ") THEN
               tmp = string(i:i)
               string(z:z) = tmp
               z = z + 1
            END IF
         ELSE
            IF ((string(i:i) /= " ") .OR. (string(z - 1:z - 1) /= " ")) THEN
               tmp = string(i:i)
               string(z:z) = tmp
               z = z + 1
            END IF
         END IF
      END DO

      string(z:) = ""

   END SUBROUTINE compress

   FUNCTION integer_to_string(inumber) RESULT(string)
      !! Converts an integer number to a string.
      !! The WRITE statement will return an error message, if the number of
      !! digits of the integer number is larger the than the length of the
      !! supplied string.

      INTEGER, INTENT(IN)                                :: inumber
      CHARACTER(:), ALLOCATABLE                          :: string
      CHARACTER(RANGE(inumber) + 2)                      :: tmp

      WRITE (UNIT=tmp, FMT='(I0)') inumber
      string = TRIM(tmp)
   END FUNCTION integer_to_string

   SUBROUTINE uppercase(string)
      !! Convert all lower case characters in a string to upper case.
      CHARACTER(LEN=*), INTENT(INOUT)                    :: string

      INTEGER                                            :: i, iascii

      DO i = 1, LEN_TRIM(string)
         iascii = ICHAR(string(i:i))
         IF ((iascii >= 97) .AND. (iascii <= 122)) THEN
            string(i:i) = CHAR(iascii - 32)
         END IF
      END DO

   END SUBROUTINE uppercase

   ELEMENTAL SUBROUTINE str2int(str, int, stat)
    !! Convert a string to integer
      CHARACTER(LEN=*), INTENT(in) :: str
      INTEGER, INTENT(OUT)         :: int, stat

      read (str, *, iostat=stat) int
   END SUBROUTINE str2int

END MODULE dbcsr_string_utilities
