!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Utilities for string manipulations
!> \par History
!>      Adapted compress and uppercase for use in CP2K (JGH)
!>      string_to_integer and integer_to_string added (06.02.2001, MK)
!>      Cleaned (04.01.2004,MK)
!> \author MK & JGH
! **************************************************************************************************
MODULE dbcsr_string_utilities

   USE dbcsr_kinds,                     ONLY: default_blank_character

   IMPLICIT NONE

   PRIVATE

   CHARACTER(LEN=1), PARAMETER :: backslash = '\\'
   CHARACTER(LEN=1), PARAMETER :: star = '*'
   CHARACTER(LEN=1), PARAMETER :: question = '?'
   CHARACTER(LEN=1), PARAMETER :: newline = ACHAR(10)

   INTEGER, PARAMETER :: maxlen_entity_name = 10
   CHARACTER(LEN=maxlen_entity_name), DIMENSION(252), PARAMETER :: html_entity_table = &
                 (/"&quot;    ", "&#34;     ", "&amp;     ", "&#38;#38; ", "&apos;    ", "&#39;     ", "&lt;      ", "&#38;#60; ", &
                   "&gt;      ", "&#62;     ", "&nbsp;    ", "&#160;    ", "&copy;    ", "&#169;    ", "&deg;     ", "&#176;    ", &
                   "&plusmn;  ", "&#177;    ", "&sup2;    ", "&#178;    ", "&sup3;    ", "&#179;    ", "&middot;  ", "&#183;    ", &
                   "&sup1;    ", "&#185;    ", "&Auml;    ", "&#196;    ", "&Aring;   ", "&#197;    ", "&Ccedil;  ", "&#199;    ", &
                   "&Egrave;  ", "&#200;    ", "&Eacute;  ", "&#201;    ", "&Ecirc;   ", "&#202;    ", "&Ograve;  ", "&#210;    ", &
                   "&Oacute;  ", "&#211;    ", "&Ocirc;   ", "&#212;    ", "&Ouml;    ", "&#214;    ", "&times;   ", "&#215;    ", &
                   "&Uuml;    ", "&#220;    ", "&agrave;  ", "&#224;    ", "&aacute;  ", "&#225;    ", "&acirc;   ", "&#226;    ", &
                   "&auml;    ", "&#228;    ", "&aring;   ", "&#229;    ", "&ccedil;  ", "&#231;    ", "&egrave;  ", "&#232;    ", &
                   "&eacute;  ", "&#233;    ", "&ecirc;   ", "&#234;    ", "&ograve;  ", "&#242;    ", "&oacute;  ", "&#243;    ", &
                   "&ocirc;   ", "&#244;    ", "&ouml;    ", "&#246;    ", "&ugrave;  ", "&#249;    ", "&uacute;  ", "&#250;    ", &
                   "&ucirc;   ", "&#251;    ", "&uuml;    ", "&#252;    ", "&Gamma;   ", "&#915;    ", "&Delta;   ", "&#916;    ", &
                   "&Theta;   ", "&#920;    ", "&Lambda;  ", "&#923;    ", "&Xi;      ", "&#926;    ", "&Pi;      ", "&#928;    ", &
                   "&Sigma;   ", "&#931;    ", "&Phi;     ", "&#934;    ", "&Psi;     ", "&#936;    ", "&Omega;   ", "&#937;    ", &
                   "&alpha;   ", "&#945;    ", "&beta;    ", "&#946;    ", "&gamma;   ", "&#947;    ", "&delta;   ", "&#948;    ", &
                   "&epsilon; ", "&#949;    ", "&zeta;    ", "&#950;    ", "&eta;     ", "&#951;    ", "&theta;   ", "&#952;    ", &
                   "&iota;    ", "&#953;    ", "&kappa;   ", "&#954;    ", "&lambda;  ", "&#955;    ", "&mu;      ", "&#956;    ", &
                   "&nu;      ", "&#957;    ", "&xi;      ", "&#958;    ", "&omicron; ", "&#959;    ", "&pi;      ", "&#960;    ", &
                   "&rho;     ", "&#961;    ", "&sigmaf;  ", "&#962;    ", "&sigma;   ", "&#963;    ", "&tau;     ", "&#964;    ", &
                   "&upsilon; ", "&#965;    ", "&phi;     ", "&#966;    ", "&chi;     ", "&#967;    ", "&psi;     ", "&#968;    ", &
                   "&omega;   ", "&#969;    ", "&thetasym;", "&#977;    ", "&ndash;   ", "&#8211;   ", "&mdash;   ", "&#8212;   ", &
                   "&dagger;  ", "&#8224;   ", "&Dagger;  ", "&#8225;   ", "&bull;    ", "&#8230;   ", "&hellip;  ", "&#8230;   ", &
                   "&permil;  ", "&#8240;   ", "&larr;    ", "&#8592;   ", "&uarr;    ", "&#8593;   ", "&rarr;    ", "&#8594;   ", &
                   "&darr;    ", "&#8595;   ", "&harr;    ", "&#8596;   ", "&lArr;    ", "&#8656;   ", "&uArr;    ", "&#8657;   ", &
                   "&rArr;    ", "&#8658;   ", "&dArr;    ", "&#8659;   ", "&hArr;    ", "&#8660;   ", "&forall;  ", "&#8704;   ", &
                   "&part;    ", "&#8706;   ", "&nabla;   ", "&#8711;   ", "&isin;    ", "&#8712;   ", "&notin;   ", "&#8713;   ", &
                   "&ni;      ", "&#8715;   ", "&prop;    ", "&#8733;   ", "&infin;   ", "&#8734;   ", "&ang;     ", "&#8736;   ", &
                   "&and;     ", "&#8743;   ", "&or;      ", "&#8744;   ", "&cap;     ", "&#8745;   ", "&cup;     ", "&#8746;   ", &
                   "&sim;     ", "&#8764;   ", "&cong;    ", "&#8773;   ", "&asymp;   ", "&#8776;   ", "&ne;      ", "&#8800;   ", &
                   "&equiv;   ", "&#8801;   ", "&le;      ", "&#8804;   ", "&ge;      ", "&#8805;   ", "&sub;     ", "&#8834;   ", &
                   "&sup;     ", "&#8835;   ", "&nsub;    ", "&#8836;   ", "&sube;    ", "&#8838;   ", "&supe;    ", "&#8839;   ", &
                   "&oplus;   ", "&#8853;   ", "&otimes;  ", "&#8855;   ", "&perp;    ", "&#8869;   ", "&sdot;    ", "&#8901;   ", &
                                                                     "&lang;    ", "&#9001;   ", "&rang;    ", "&#9002;   "/)

   INTEGER, PARAMETER :: maxlen_tag_name = 10
   CHARACTER(LEN=maxlen_tag_name), DIMENSION(38), PARAMETER :: html_tag_table = &
                 (/"a         ", "b         ", "big       ", "blockquote", "br/       ", "code      ", "dd        ", "del       ", &
                   "div       ", "dl        ", "dt        ", "em        ", "h1        ", "h2        ", "h3        ", "h4        ", &
                   "h5        ", "h6        ", "hr        ", "i         ", "ins       ", "li        ", "ol        ", "p         ", &
                   "span      ", "sub       ", "sup       ", "table     ", "tbody     ", "td        ", "tfoot     ", "th        ", &
                                                "thead     ", "tr        ", "tt        ", "u         ", "ul        ", "pre       "/)

   PUBLIC :: ascii_to_string, &
             compress, &
             integer_to_string, &
             is_whitespace, &
             remove_word, &
             str_comp, &
             string_to_ascii, &
             uppercase, &
             write_html_tables, &
             xstring

   PUBLIC :: html_entity_table, &
             newline

CONTAINS

! **************************************************************************************************
!> \brief Convert a sequence of integer numbers (ASCII code) to a string.
!>         Blanks are inserted for invalid ASCII code numbers.
!> \param nascii ...
!> \param string ...
!> \date    19.10.2000
!> \author  MK
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE ascii_to_string(nascii, string)

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

! **************************************************************************************************
!> \brief   Eliminate multiple space characters in a string.
!>          If full is .TRUE., then all spaces are eliminated.
!> \param string ...
!> \param full ...
!> \date    23.06.1998
!> \author  MK
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE compress(string, full)

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
               z = z+1
            END IF
         ELSE
            IF ((string(i:i) /= " ") .OR. (string(z-1:z-1) /= " ")) THEN
               tmp = string(i:i)
               string(z:z) = tmp
               z = z+1
            END IF
         END IF
      END DO

      string(z:) = ""

   END SUBROUTINE compress

! **************************************************************************************************
!> \brief   Converts an integer number to a string.
!>          The WRITE statement will return an error message, if the number of
!>          digits of the integer number is larger the than the length of the
!>          supplied string.
!> \param inumber ...
!> \param string ...
!> \date    05.01.2004
!> \author  MK
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE integer_to_string(inumber, string)

      INTEGER, INTENT(IN)                                :: inumber
      CHARACTER(LEN=*), INTENT(OUT)                      :: string

      WRITE (UNIT=string, FMT='(I0)') inumber
   END SUBROUTINE integer_to_string

! **************************************************************************************************
!> \brief   Convert a string to sequence of integer numbers.
!> \param string ...
!> \param nascii ...
!> \date    19.10.2000
!> \author  MK
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE string_to_ascii(string, nascii)

      CHARACTER(LEN=*), INTENT(IN)                       :: string
      INTEGER, DIMENSION(:), INTENT(OUT)                 :: nascii

      INTEGER                                            :: i

      nascii(:) = 0

      DO i = 1, MIN(LEN(string), SIZE(nascii))
         nascii(i) = ICHAR(string(i:i))
      END DO

   END SUBROUTINE string_to_ascii

! **************************************************************************************************
!> \brief   remove a word from a string (words are separated by white spaces)
!> \param string ...
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE remove_word(string)
      CHARACTER(LEN=*), INTENT(INOUT)                    :: string

      INTEGER                                            :: i

      i = 1
      ! possibly clean white spaces
      DO WHILE (string(i:i) == " ")
         i = i+1
      END DO
      ! now remove the word
      DO WHILE (string(i:i) /= " ")
         i = i+1
      END DO
      string = string(i:)

   END SUBROUTINE remove_word

! **************************************************************************************************
!> \brief   Write the HTML entity and HTML tag table to unit output_unit
!> \param output_unit ...
!> \date    14.08.2013
!> \author  Matthias Krack (MK)
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE write_html_tables(output_unit)

      INTEGER, INTENT(IN)                                :: output_unit

      INTEGER                                            :: i

      WRITE (UNIT=output_unit, FMT="(T2,A)") "<CP2K_HTML>"

      DO i = 1, SIZE(html_tag_table)
         WRITE (UNIT=output_unit, FMT="(T3,A)") &
            "<TAG>", &
            " <NAME>&#60;"//TRIM(html_tag_table(i))//"&#62;</NAME>", &
            "</TAG>"
      END DO
      DO i = 1, SIZE(html_entity_table), 2
         WRITE (UNIT=output_unit, FMT="(T3,A)") &
            "<ENTITY>", &
            " <NAME>&#38;"//TRIM(html_entity_table(i) (2:))//"</NAME>", &
            " <CODE>&#38;"//TRIM(html_entity_table(i+1) (2:))//"</CODE>", &
            "</ENTITY>"
      END DO

      WRITE (UNIT=output_unit, FMT="(T2,A)") "</CP2K_HTML>"

   END SUBROUTINE write_html_tables

! **************************************************************************************************
!> \brief   Convert all upper case characters in a string to lower case.
!> \param string ...
!> \date    14.08.2013
!> \author  Matthias Krack (MK)
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE lowercase(string)
      CHARACTER(LEN=*), INTENT(INOUT)                    :: string

      INTEGER                                            :: i, iascii

      DO i = 1, LEN_TRIM(string)
         iascii = ICHAR(string(i:i))
         IF ((iascii >= 65) .AND. (iascii <= 90)) THEN
            string(i:i) = CHAR(iascii+32)
         END IF
      END DO

   END SUBROUTINE lowercase

! **************************************************************************************************
!> \brief   Convert all lower case characters in a string to upper case.
!> \param string ...
!> \date    22.06.1998
!> \author  MK
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE uppercase(string)
      CHARACTER(LEN=*), INTENT(INOUT)                    :: string

      INTEGER                                            :: i, iascii

      DO i = 1, LEN_TRIM(string)
         iascii = ICHAR(string(i:i))
         IF ((iascii >= 97) .AND. (iascii <= 122)) THEN
            string(i:i) = CHAR(iascii-32)
         END IF
      END DO

   END SUBROUTINE uppercase

   ! **************************************************************************************************
! **************************************************************************************************
!> \brief ...
!> \param string ...
!> \param ia ...
!> \param ib ...
! **************************************************************************************************
   SUBROUTINE xstring(string, ia, ib)

      CHARACTER(LEN=*), INTENT(IN)                       :: string
      INTEGER, INTENT(OUT)                               :: ia, ib

      ia = 1
      ib = LEN_TRIM(string)
      IF (ib > 0) THEN
         DO WHILE (string(ia:ia) == ' ')
            ia = ia+1
         END DO
      END IF

   END SUBROUTINE xstring

   ! **************************************************************************************************
! **************************************************************************************************
!> \brief ...
!> \param str1 ...
!> \param str2 ...
!> \return ...
! **************************************************************************************************
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

! **************************************************************************************************
!> \brief returns .true. if the character passed is a whitespace char.
!> \param testchar ...
!> \return ...
!> \par History
!>      02.2008 created, AK
! **************************************************************************************************
   FUNCTION is_whitespace(testchar) RESULT(resval)
      CHARACTER(LEN=1), INTENT(IN)                       :: testchar
      LOGICAL                                            :: resval

      resval = .FALSE.
      IF (ANY(default_blank_character == testchar)) resval = .TRUE.
   END FUNCTION is_whitespace

END MODULE dbcsr_string_utilities
