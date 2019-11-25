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

   PUBLIC :: ascii_to_string, &
             compress, &
             integer_to_string, &
             is_whitespace, &
             remove_word, &
             str_comp, &
             string_to_ascii, &
             uppercase, &
             xstring

CONTAINS

   SUBROUTINE ascii_to_string(nascii, string)
      !! Convert a sequence of integer numbers (ASCII code) to a string.
      !! Blanks are inserted for invalid ASCII code numbers.

      INTEGER, DIMENSION(:), INTENT(IN)                  :: nascii
      CHARACTER(LEN=*), INTENT(OUT)                      :: string

      INTEGER                                            :: i

      string = ""

      DO i = 1, MIN(LEN(string), SIZE(nascii))
         IF ((nascii(i) >= 0) .AND. (nascii(i) <= 127)) THEN
            string(i:i) = CHAR(nascii(i))
         ELSE
            string(i:i) = " "
         END IF
      END DO

   END SUBROUTINE ascii_to_string

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

   SUBROUTINE integer_to_string(inumber, string)
      !! Converts an integer number to a string.
      !! The WRITE statement will return an error message, if the number of
      !! digits of the integer number is larger the than the length of the
      !! supplied string.

      INTEGER, INTENT(IN)                                :: inumber
      CHARACTER(LEN=*), INTENT(OUT)                      :: string

      WRITE (UNIT=string, FMT='(I0)') inumber
   END SUBROUTINE integer_to_string

   SUBROUTINE string_to_ascii(string, nascii)
      !! Convert a string to sequence of integer numbers.

      CHARACTER(LEN=*), INTENT(IN)                       :: string
      INTEGER, DIMENSION(:), INTENT(OUT)                 :: nascii

      INTEGER                                            :: i

      nascii(:) = 0

      DO i = 1, MIN(LEN(string), SIZE(nascii))
         nascii(i) = ICHAR(string(i:i))
      END DO

   END SUBROUTINE string_to_ascii

   SUBROUTINE remove_word(string)
      !! remove a word from a string (words are separated by white spaces)
      CHARACTER(LEN=*), INTENT(INOUT)                    :: string

      INTEGER                                            :: i

      i = 1
      ! possibly clean white spaces
      DO WHILE (string(i:i) == " ")
         i = i + 1
      END DO
      ! now remove the word
      DO WHILE (string(i:i) /= " ")
         i = i + 1
      END DO
      string = string(i:)

   END SUBROUTINE remove_word

   SUBROUTINE lowercase(string)
      !! Convert all upper case characters in a string to lower case.
      CHARACTER(LEN=*), INTENT(INOUT)                    :: string

      INTEGER                                            :: i, iascii

      DO i = 1, LEN_TRIM(string)
         iascii = ICHAR(string(i:i))
         IF ((iascii >= 65) .AND. (iascii <= 90)) THEN
            string(i:i) = CHAR(iascii + 32)
         END IF
      END DO

   END SUBROUTINE lowercase

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

   SUBROUTINE xstring(string, ia, ib)

      CHARACTER(LEN=*), INTENT(IN)                       :: string
      INTEGER, INTENT(OUT)                               :: ia, ib

      ia = 1
      ib = LEN_TRIM(string)
      IF (ib > 0) THEN
         DO WHILE (string(ia:ia) == ' ')
            ia = ia + 1
         END DO
      END IF

   END SUBROUTINE xstring

   FUNCTION str_comp(str1, str2) RESULT(equal)

      CHARACTER(LEN=*), INTENT(IN)                       :: str1, str2
      LOGICAL                                            :: equal

      INTEGER                                            :: i1, i2, j1, j2

      i1 = 0
      i2 = 0
      j1 = 0
      j2 = 0
      CALL xstring(str1, i1, i2)
      CALL xstring(str2, j1, j2)
      equal = (str1(i1:i2) == str2(j1:j2))
   END FUNCTION str_comp

   FUNCTION is_whitespace(testchar) RESULT(resval)
      !! returns .true. if the character passed is a whitespace char.
      CHARACTER(LEN=1), INTENT(IN)                       :: testchar
      LOGICAL                                            :: resval

      resval = .FALSE.
      IF (ANY(default_blank_character == testchar)) resval = .TRUE.
   END FUNCTION is_whitespace

END MODULE dbcsr_string_utilities
