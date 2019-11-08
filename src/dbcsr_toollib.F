!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_toollib
   !! Tools usually found in a standard library.

   USE dbcsr_array_sort, ONLY: dbcsr_1d_d_sort, &
                               dbcsr_1d_i4_sort, &
                               dbcsr_1d_i8_sort, &
                               dbcsr_1d_s_sort
   USE dbcsr_kinds, ONLY: int_4, &
                          int_8, &
                          real_8
#include "../base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_toollib'

   ! Block sizes and arrays
   PUBLIC :: dbcsr_unpack_i8_2i4, make_coordinate_tuple
   PUBLIC :: swap
   ! math routines
   PUBLIC :: gcd, lcm
   ! utility routines
   PUBLIC :: sort, joaat_hash
   PUBLIC :: ordered_search
   PUBLIC :: atoi, atol, ator

   INTERFACE swap
      MODULE PROCEDURE iswap, bswap
   END INTERFACE

   INTERFACE sort
      MODULE PROCEDURE dbcsr_1d_i4_sort, dbcsr_1d_i8_sort
      MODULE PROCEDURE dbcsr_1d_s_sort, dbcsr_1d_d_sort
   END INTERFACE

CONTAINS

   ELEMENTAL FUNCTION make_coordinate_tuple(most, least) RESULT(tuple)
      INTEGER, INTENT(IN)                                :: most, least
      INTEGER(KIND=int_8)                                :: tuple

!tuple = IOR (ISHFT (most, 32), least)

      tuple = most
      tuple = IOR(ISHFT(tuple, 32), INT(least, int_8))
   END FUNCTION make_coordinate_tuple

   ELEMENTAL SUBROUTINE iswap(a, b)
      !! Swaps two integers

      INTEGER, INTENT(INOUT)                             :: a, b
         !! Integers to swap
         !! Integers to swap

      INTEGER                                            :: tmp

      tmp = a
      a = b
      b = tmp
   END SUBROUTINE iswap

   ELEMENTAL SUBROUTINE bswap(a, b)
      !! Swaps two logicals

      LOGICAL, INTENT(INOUT)                             :: a, b
         !! Logicals to swap
         !! Logicals to swap

      LOGICAL                                            :: tmp

      tmp = a
      a = b
      b = tmp
   END SUBROUTINE bswap

   SUBROUTINE dbcsr_unpack_i8_2i4(merged, array_upper, array_lower)
      !! Splits an array of int8 values into two int4 arrays.

      INTEGER(KIND=int_8), DIMENSION(:), INTENT(IN)      :: merged
         !! array of merged values
      INTEGER(KIND=int_4), DIMENSION(:), INTENT(OUT)     :: array_upper, array_lower
         !! array to fill with the upper bytes of the merged values
         !! array to fill with the lower bytes of the merged values

      INTEGER(KIND=int_8), PARAMETER                     :: lmask8 = 4294967295_int_8

      INTEGER                                            :: i

!
!   ---------------------------------------------------------------------------
! Lmask is used to filter in the lower 4 bytes and so its lower 32 bits are
! set to 1: lmask8 = 2^32-1.
! Umask is used to filter in the higher 4 bytes and so its higher 32 bits
! are set to 1: umask8 = 2^32-1 << 32
!lmask8 = 4294967295 ! 2^32-1
!umask8 = 18446744069414584320 ! (2^32-1) * 2^32 = (2^64-1)-(2^32-1)

      DO i = 1, SIZE(merged)
         array_upper(i) = INT(ISHFT(merged(i), -32), KIND=int_4)
         array_lower(i) = INT(IAND(merged(i), lmask8), KIND=int_4)
      ENDDO
   END SUBROUTINE dbcsr_unpack_i8_2i4

   ELEMENTAL FUNCTION gcd(a, b)
      INTEGER, INTENT(IN)                                :: a, b
      INTEGER                                            :: gcd

      INTEGER                                            :: aa, ab, l, rem, s

      aa = ABS(a)
      ab = ABS(b)
      IF (aa < ab) THEN
         s = aa
         l = ab
      ELSE
         s = ab
         l = aa
      ENDIF
      IF (s .NE. 0) THEN
         DO
            rem = MOD(l, s)
            IF (rem == 0) EXIT
            l = s
            s = rem
         ENDDO
         GCD = s
      ELSE
         GCD = l
      ENDIF
   END FUNCTION gcd

   ELEMENTAL FUNCTION lcm(a, b)
      INTEGER, INTENT(IN)                                :: a, b
      INTEGER                                            :: lcm

      INTEGER                                            :: tmp

      tmp = gcd(a, b)
      IF (tmp == 0) THEN
         lcm = 0
      ELSE
         ! could still overflow if the true lcm is larger than maxint
         lcm = ABS((a/tmp)*b)
      END IF
   END FUNCTION lcm

   FUNCTION joaat_hash(key) RESULT(hash_index)
      !! generates the hash of a string and the index in the table
      !! @note
      !! http://en.wikipedia.org/wiki/Hash_table
      !! http://www.burtleburtle.net/bob/hash/doobs.html
      !! However, since fortran doesn't have an unsigned 4 byte int
      !! we compute it using an integer with the appropriate range
      !! we return already the index in the table as a final result

      ! LIBXSMM: at least v1.9.0-6 is required
#if defined(__LIBXSMM) && TO_VERSION(1, 10, 0) <= TO_VERSION(LIBXSMM_CONFIG_VERSION_MAJOR, LIBXSMM_CONFIG_VERSION_MINOR, LIBXSMM_CONFIG_VERSION_UPDATE)
      USE libxsmm, ONLY: libxsmm_hash
      INTEGER, PARAMETER                                 :: seed = 0
      INTEGER, DIMENSION(:), INTENT(IN)                  :: key
         !! a string of any length
      INTEGER                                            :: hash_index
      hash_index = libxsmm_hash(key, seed)
#else
      INTEGER, DIMENSION(:), INTENT(IN)                  :: key
      INTEGER                                            :: hash_index

      INTEGER(KIND=int_8), PARAMETER                     :: b32 = 2_int_8**32 - 1_int_8

      INTEGER                                            :: i, j
      INTEGER(KIND=int_8)                                :: byte, hash

      hash = 0_int_8
      DO i = 1, SIZE(key)
         DO j = 0, 3
            byte = IAND(ISHFT(key(i), -j*8), 255)
            hash = IAND(hash + byte, b32)
            hash = IAND(hash + IAND(ISHFT(hash, 10), b32), b32)
            hash = IAND(IEOR(hash, IAND(ISHFT(hash, -6), b32)), b32)
         ENDDO
      ENDDO
      hash = IAND(hash + IAND(ISHFT(hash, 3), b32), b32)
      hash = IAND(IEOR(hash, IAND(ISHFT(hash, -11), b32)), b32)
      hash = IAND(hash + IAND(ISHFT(hash, 15), b32), b32)
      ! In fortran 4-byte-integers have only 31 bits because they are signed
      ! In fortran the rightmost (least significant) bit is in position 0
      hash_index = INT(IBCLR(hash, 31))
#endif
   END FUNCTION joaat_hash

   PURE SUBROUTINE ordered_search(array, key, loc, found, lb, ub)
      !! search a value in an ordered array of indices
      INTEGER, DIMENSION(:), INTENT(IN)                  :: array
      INTEGER, INTENT(IN)                                :: key
      INTEGER, INTENT(OUT)                               :: loc
      LOGICAL, INTENT(OUT)                               :: found
      INTEGER, INTENT(IN), OPTIONAL                      :: lb, ub

      INTEGER                                            :: high, low, val

      found = .FALSE.
      IF (PRESENT(lb)) THEN
         low = lb
      ELSE
         low = LBOUND(array, 1)
      ENDIF
      IF (PRESENT(ub)) THEN
         high = ub
      ELSE
         high = UBOUND(array, 1)
      ENDIF
      loc = (low + high)/2
      DO WHILE (loc .GE. low .AND. loc .LE. high)
         val = array(loc)
         IF (val .EQ. key) THEN
            found = .TRUE.
            EXIT
         ELSEIF (val .LT. key) THEN
            low = loc + 1
         ELSE
            high = loc - 1
         ENDIF
         loc = (low + high)/2
      ENDDO
   END SUBROUTINE ordered_search

   FUNCTION atoi(a)
      CHARACTER(len=*), INTENT(in)                       :: a
      INTEGER                                            :: atoi

      READ (a, '(I9)') atoi
   END FUNCTION atoi

   FUNCTION atol(a)
      CHARACTER(len=*), INTENT(in)                       :: a
      LOGICAL                                            :: atol

      READ (a, '(L1)') atol
   END FUNCTION atol

   FUNCTION ator(a)
      CHARACTER(len=*), INTENT(in)                       :: a
      REAL(real_8)                                       :: ator

      READ (a, '(E26.15)') ator
   END FUNCTION ator

END MODULE dbcsr_toollib
