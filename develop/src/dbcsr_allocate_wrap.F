# 1 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F" 1
!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_allocate_wrap
   !! Wrapper for allocating, copying and reshaping arrays.
   !! @todo: with fortran 2008 support, this should be replaced by plain ALLOCATE
   !! @note
   !! in particular ALLOCATE(..., SOURCE=...) does not work in gcc 5.4.0, see also
   !! https://gcc.gnu.org/bugzilla/show_bug.cgi?id=44672
   !! @endnote

# 1 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp" 1
# 9 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp"

# 241 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_tensor.fypp"
# 19 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F" 2
# 20 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

   USE dbcsr_kinds, ONLY: real_4, real_8
# 22 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

#include "base/dbcsr_base_uses.f90"
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: allocate_any

   INTERFACE allocate_any
# 31 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_1d_r_dp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_2d_r_dp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_3d_r_dp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_4d_r_dp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_5d_r_dp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_6d_r_dp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_7d_r_dp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 34 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 31 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_1d_r_sp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_2d_r_sp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_3d_r_sp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_4d_r_sp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_5d_r_sp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_6d_r_sp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_7d_r_sp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 34 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 31 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_1d_c_dp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_2d_c_dp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_3d_c_dp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_4d_c_dp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_5d_c_dp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_6d_c_dp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_7d_c_dp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 34 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 31 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_1d_c_sp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_2d_c_sp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_3d_c_sp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_4d_c_sp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_5d_c_sp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_6d_c_sp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_7d_c_sp
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 34 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 31 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_1d_i
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_2d_i
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_3d_i
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_4d_i
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_5d_i
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_6d_i
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 32 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            MODULE PROCEDURE allocate_7d_i
# 33 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 34 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 35 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
   END INTERFACE

CONTAINS

# 40 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_1d_r_dp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            REAL(kind=real_8), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            REAL(kind=real_8), DIMENSION(:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(1)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_2d_r_dp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            REAL(kind=real_8), DIMENSION(:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            REAL(kind=real_8), DIMENSION(:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(2)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_3d_r_dp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            REAL(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            REAL(kind=real_8), DIMENSION(:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(3)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_4d_r_dp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            REAL(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            REAL(kind=real_8), DIMENSION(:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(4)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_5d_r_dp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            REAL(kind=real_8), DIMENSION(:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            REAL(kind=real_8), DIMENSION(:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(5)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_6d_r_dp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            REAL(kind=real_8), DIMENSION(:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            REAL(kind=real_8), DIMENSION(:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(6)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)),&
# 80 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                      & source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_7d_r_dp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            REAL(kind=real_8), DIMENSION(:,:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            REAL(kind=real_8), DIMENSION(:,:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(7)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6), shape_prv(7)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6),&
# 80 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                      & shape_prv(7)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6), shape_prv(7)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 40 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_1d_r_sp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            REAL(kind=real_4), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            REAL(kind=real_4), DIMENSION(:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(1)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_2d_r_sp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            REAL(kind=real_4), DIMENSION(:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            REAL(kind=real_4), DIMENSION(:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(2)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_3d_r_sp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            REAL(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            REAL(kind=real_4), DIMENSION(:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(3)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_4d_r_sp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            REAL(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            REAL(kind=real_4), DIMENSION(:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(4)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_5d_r_sp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            REAL(kind=real_4), DIMENSION(:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            REAL(kind=real_4), DIMENSION(:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(5)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_6d_r_sp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            REAL(kind=real_4), DIMENSION(:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            REAL(kind=real_4), DIMENSION(:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(6)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)),&
# 80 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                      & source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_7d_r_sp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            REAL(kind=real_4), DIMENSION(:,:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            REAL(kind=real_4), DIMENSION(:,:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(7)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6), shape_prv(7)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6),&
# 80 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                      & shape_prv(7)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6), shape_prv(7)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 40 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_1d_c_dp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            COMPLEX(kind=real_8), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            COMPLEX(kind=real_8), DIMENSION(:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(1)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_2d_c_dp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            COMPLEX(kind=real_8), DIMENSION(:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            COMPLEX(kind=real_8), DIMENSION(:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(2)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_3d_c_dp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            COMPLEX(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            COMPLEX(kind=real_8), DIMENSION(:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(3)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_4d_c_dp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            COMPLEX(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            COMPLEX(kind=real_8), DIMENSION(:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(4)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_5d_c_dp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            COMPLEX(kind=real_8), DIMENSION(:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            COMPLEX(kind=real_8), DIMENSION(:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(5)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_6d_c_dp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            COMPLEX(kind=real_8), DIMENSION(:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            COMPLEX(kind=real_8), DIMENSION(:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(6)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)),&
# 80 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                      & source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_7d_c_dp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            COMPLEX(kind=real_8), DIMENSION(:,:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            COMPLEX(kind=real_8), DIMENSION(:,:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(7)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6), shape_prv(7)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6),&
# 80 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                      & shape_prv(7)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6), shape_prv(7)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 40 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_1d_c_sp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            COMPLEX(kind=real_4), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            COMPLEX(kind=real_4), DIMENSION(:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(1)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_2d_c_sp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            COMPLEX(kind=real_4), DIMENSION(:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            COMPLEX(kind=real_4), DIMENSION(:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(2)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_3d_c_sp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            COMPLEX(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            COMPLEX(kind=real_4), DIMENSION(:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(3)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_4d_c_sp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            COMPLEX(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            COMPLEX(kind=real_4), DIMENSION(:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(4)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_5d_c_sp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            COMPLEX(kind=real_4), DIMENSION(:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            COMPLEX(kind=real_4), DIMENSION(:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(5)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_6d_c_sp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            COMPLEX(kind=real_4), DIMENSION(:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            COMPLEX(kind=real_4), DIMENSION(:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(6)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)),&
# 80 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                      & source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_7d_c_sp (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            COMPLEX(kind=real_4), DIMENSION(:,:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            COMPLEX(kind=real_4), DIMENSION(:,:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(7)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6), shape_prv(7)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6),&
# 80 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                      & shape_prv(7)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6), shape_prv(7)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 40 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_1d_i (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            INTEGER, DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(1)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_2d_i (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            INTEGER, DIMENSION(:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            INTEGER, DIMENSION(:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(2)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_3d_i (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            INTEGER, DIMENSION(:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            INTEGER, DIMENSION(:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(3)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_4d_i (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            INTEGER, DIMENSION(:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            INTEGER, DIMENSION(:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(4)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_5d_i (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            INTEGER, DIMENSION(:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            INTEGER, DIMENSION(:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(5)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_6d_i (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            INTEGER, DIMENSION(:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            INTEGER, DIMENSION(:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(6)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)),&
# 80 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                      & source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 41 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

         SUBROUTINE allocate_7d_i (array, shape_spec, source, order)
# 43 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note
      !! this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.
      !! @endnote

            INTEGER, DIMENSION(:,:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
# 50 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! target array.
            INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: shape_spec
# 52 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
            INTEGER, DIMENSION(:,:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
# 54 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! source array to be copied to target array, must have same rank as target array.
            INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: order
# 56 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
            INTEGER, DIMENSION(7)                                          :: shape_prv
# 58 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"

            IF (PRESENT(shape_spec)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = shape_spec
               ELSE
                  shape_prv = shape_spec
               END IF
            ELSEIF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  shape_prv(order) = SHAPE(source)
               ELSE
                  shape_prv = SHAPE(source)
               END IF
            ELSE
               DBCSR_ABORT("either source or shape_spec must be present")
            END IF

            IF (PRESENT(source)) THEN
               IF (PRESENT(order)) THEN
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6), shape_prv(7)))
# 78 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                  array(:,:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order)
# 79 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               ELSE
                  ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6),&
# 80 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
                      & shape_prv(7)), source=source)
# 81 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
               END IF
            ELSE
               ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6), shape_prv(7)))
# 84 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
            END IF

         END SUBROUTINE
# 88 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
# 89 "/__w/dbcsr/dbcsr/src/tensors/dbcsr_allocate_wrap.F"
END MODULE
