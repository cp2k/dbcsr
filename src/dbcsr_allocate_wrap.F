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
   !! \todo: with fortran 2008 support, this should be replaced by plain ALLOCATE




   USE dbcsr_kinds, ONLY: real_4, real_8

#include "base/dbcsr_base_uses.f90"
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: allocate_any

   INTERFACE allocate_any
      MODULE PROCEDURE allocate_1d_r_dp
      MODULE PROCEDURE allocate_2d_r_dp
      MODULE PROCEDURE allocate_3d_r_dp
      MODULE PROCEDURE allocate_4d_r_dp
      MODULE PROCEDURE allocate_5d_r_dp
      MODULE PROCEDURE allocate_6d_r_dp
      MODULE PROCEDURE allocate_7d_r_dp
      MODULE PROCEDURE allocate_1d_r_sp
      MODULE PROCEDURE allocate_2d_r_sp
      MODULE PROCEDURE allocate_3d_r_sp
      MODULE PROCEDURE allocate_4d_r_sp
      MODULE PROCEDURE allocate_5d_r_sp
      MODULE PROCEDURE allocate_6d_r_sp
      MODULE PROCEDURE allocate_7d_r_sp
      MODULE PROCEDURE allocate_1d_c_dp
      MODULE PROCEDURE allocate_2d_c_dp
      MODULE PROCEDURE allocate_3d_c_dp
      MODULE PROCEDURE allocate_4d_c_dp
      MODULE PROCEDURE allocate_5d_c_dp
      MODULE PROCEDURE allocate_6d_c_dp
      MODULE PROCEDURE allocate_7d_c_dp
      MODULE PROCEDURE allocate_1d_c_sp
      MODULE PROCEDURE allocate_2d_c_sp
      MODULE PROCEDURE allocate_3d_c_sp
      MODULE PROCEDURE allocate_4d_c_sp
      MODULE PROCEDURE allocate_5d_c_sp
      MODULE PROCEDURE allocate_6d_c_sp
      MODULE PROCEDURE allocate_7d_c_sp
      MODULE PROCEDURE allocate_1d_i
      MODULE PROCEDURE allocate_2d_i
      MODULE PROCEDURE allocate_3d_i
      MODULE PROCEDURE allocate_4d_i
      MODULE PROCEDURE allocate_5d_i
      MODULE PROCEDURE allocate_6d_i
      MODULE PROCEDURE allocate_7d_i
   END INTERFACE

CONTAINS

   SUBROUTINE allocate_1d_r_dp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      REAL(kind=real_8), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      REAL(kind=real_8), DIMENSION(:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(1)                                          :: order_prv
      INTEGER, DIMENSION(1)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 1)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1)))

      IF (PRESENT(source)) THEN
         array(:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_2d_r_dp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      REAL(kind=real_8), DIMENSION(:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      REAL(kind=real_8), DIMENSION(:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(2)                                          :: order_prv
      INTEGER, DIMENSION(2)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 2)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2)))

      IF (PRESENT(source)) THEN
         array(:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_3d_r_dp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      REAL(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      REAL(kind=real_8), DIMENSION(:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(3)                                          :: order_prv
      INTEGER, DIMENSION(3)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 3)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)))

      IF (PRESENT(source)) THEN
         array(:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_4d_r_dp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      REAL(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      REAL(kind=real_8), DIMENSION(:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(4)                                          :: order_prv
      INTEGER, DIMENSION(4)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 4)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_5d_r_dp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      REAL(kind=real_8), DIMENSION(:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      REAL(kind=real_8), DIMENSION(:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(5)                                          :: order_prv
      INTEGER, DIMENSION(5)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 5)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_6d_r_dp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      REAL(kind=real_8), DIMENSION(:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      REAL(kind=real_8), DIMENSION(:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(6)                                          :: order_prv
      INTEGER, DIMENSION(6)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 6)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_7d_r_dp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      REAL(kind=real_8), DIMENSION(:,:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      REAL(kind=real_8), DIMENSION(:,:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(7)                                          :: order_prv
      INTEGER, DIMENSION(7)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 7)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6), shape_prv(7)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_1d_r_sp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      REAL(kind=real_4), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      REAL(kind=real_4), DIMENSION(:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(1)                                          :: order_prv
      INTEGER, DIMENSION(1)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 1)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1)))

      IF (PRESENT(source)) THEN
         array(:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_2d_r_sp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      REAL(kind=real_4), DIMENSION(:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      REAL(kind=real_4), DIMENSION(:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(2)                                          :: order_prv
      INTEGER, DIMENSION(2)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 2)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2)))

      IF (PRESENT(source)) THEN
         array(:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_3d_r_sp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      REAL(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      REAL(kind=real_4), DIMENSION(:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(3)                                          :: order_prv
      INTEGER, DIMENSION(3)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 3)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)))

      IF (PRESENT(source)) THEN
         array(:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_4d_r_sp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      REAL(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      REAL(kind=real_4), DIMENSION(:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(4)                                          :: order_prv
      INTEGER, DIMENSION(4)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 4)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_5d_r_sp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      REAL(kind=real_4), DIMENSION(:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      REAL(kind=real_4), DIMENSION(:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(5)                                          :: order_prv
      INTEGER, DIMENSION(5)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 5)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_6d_r_sp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      REAL(kind=real_4), DIMENSION(:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      REAL(kind=real_4), DIMENSION(:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(6)                                          :: order_prv
      INTEGER, DIMENSION(6)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 6)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_7d_r_sp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      REAL(kind=real_4), DIMENSION(:,:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      REAL(kind=real_4), DIMENSION(:,:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(7)                                          :: order_prv
      INTEGER, DIMENSION(7)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 7)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6), shape_prv(7)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_1d_c_dp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      COMPLEX(kind=real_8), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      COMPLEX(kind=real_8), DIMENSION(:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(1)                                          :: order_prv
      INTEGER, DIMENSION(1)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 1)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1)))

      IF (PRESENT(source)) THEN
         array(:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_2d_c_dp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      COMPLEX(kind=real_8), DIMENSION(:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      COMPLEX(kind=real_8), DIMENSION(:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(2)                                          :: order_prv
      INTEGER, DIMENSION(2)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 2)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2)))

      IF (PRESENT(source)) THEN
         array(:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_3d_c_dp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      COMPLEX(kind=real_8), DIMENSION(:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      COMPLEX(kind=real_8), DIMENSION(:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(3)                                          :: order_prv
      INTEGER, DIMENSION(3)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 3)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)))

      IF (PRESENT(source)) THEN
         array(:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_4d_c_dp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      COMPLEX(kind=real_8), DIMENSION(:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      COMPLEX(kind=real_8), DIMENSION(:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(4)                                          :: order_prv
      INTEGER, DIMENSION(4)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 4)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_5d_c_dp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      COMPLEX(kind=real_8), DIMENSION(:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      COMPLEX(kind=real_8), DIMENSION(:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(5)                                          :: order_prv
      INTEGER, DIMENSION(5)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 5)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_6d_c_dp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      COMPLEX(kind=real_8), DIMENSION(:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      COMPLEX(kind=real_8), DIMENSION(:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(6)                                          :: order_prv
      INTEGER, DIMENSION(6)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 6)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_7d_c_dp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      COMPLEX(kind=real_8), DIMENSION(:,:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      COMPLEX(kind=real_8), DIMENSION(:,:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(7)                                          :: order_prv
      INTEGER, DIMENSION(7)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 7)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6), shape_prv(7)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_1d_c_sp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      COMPLEX(kind=real_4), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      COMPLEX(kind=real_4), DIMENSION(:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(1)                                          :: order_prv
      INTEGER, DIMENSION(1)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 1)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1)))

      IF (PRESENT(source)) THEN
         array(:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_2d_c_sp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      COMPLEX(kind=real_4), DIMENSION(:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      COMPLEX(kind=real_4), DIMENSION(:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(2)                                          :: order_prv
      INTEGER, DIMENSION(2)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 2)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2)))

      IF (PRESENT(source)) THEN
         array(:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_3d_c_sp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      COMPLEX(kind=real_4), DIMENSION(:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      COMPLEX(kind=real_4), DIMENSION(:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(3)                                          :: order_prv
      INTEGER, DIMENSION(3)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 3)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)))

      IF (PRESENT(source)) THEN
         array(:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_4d_c_sp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      COMPLEX(kind=real_4), DIMENSION(:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      COMPLEX(kind=real_4), DIMENSION(:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(4)                                          :: order_prv
      INTEGER, DIMENSION(4)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 4)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_5d_c_sp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      COMPLEX(kind=real_4), DIMENSION(:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      COMPLEX(kind=real_4), DIMENSION(:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(5)                                          :: order_prv
      INTEGER, DIMENSION(5)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 5)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_6d_c_sp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      COMPLEX(kind=real_4), DIMENSION(:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      COMPLEX(kind=real_4), DIMENSION(:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(6)                                          :: order_prv
      INTEGER, DIMENSION(6)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 6)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_7d_c_sp (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      COMPLEX(kind=real_4), DIMENSION(:,:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      COMPLEX(kind=real_4), DIMENSION(:,:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(7)                                          :: order_prv
      INTEGER, DIMENSION(7)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 7)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6), shape_prv(7)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_1d_i (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      INTEGER, DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      INTEGER, DIMENSION(:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(1), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(1)                                          :: order_prv
      INTEGER, DIMENSION(1)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 1)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1)))

      IF (PRESENT(source)) THEN
         array(:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_2d_i (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      INTEGER, DIMENSION(:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      INTEGER, DIMENSION(:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(2)                                          :: order_prv
      INTEGER, DIMENSION(2)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 2)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2)))

      IF (PRESENT(source)) THEN
         array(:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_3d_i (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      INTEGER, DIMENSION(:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      INTEGER, DIMENSION(:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(3), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(3)                                          :: order_prv
      INTEGER, DIMENSION(3)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 3)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3)))

      IF (PRESENT(source)) THEN
         array(:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_4d_i (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      INTEGER, DIMENSION(:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      INTEGER, DIMENSION(:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(4), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(4)                                          :: order_prv
      INTEGER, DIMENSION(4)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 4)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_5d_i (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      INTEGER, DIMENSION(:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      INTEGER, DIMENSION(:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(5), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(5)                                          :: order_prv
      INTEGER, DIMENSION(5)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 5)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_6d_i (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      INTEGER, DIMENSION(:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      INTEGER, DIMENSION(:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(6), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(6)                                          :: order_prv
      INTEGER, DIMENSION(6)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 6)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
   SUBROUTINE allocate_7d_i (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      INTEGER, DIMENSION(:,:,:,:,:,:,:), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      INTEGER, DIMENSION(:,:,:,:,:,:,:), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(7), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(7)                                          :: order_prv
      INTEGER, DIMENSION(7)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, 7)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(shape_prv(1), shape_prv(2), shape_prv(3), shape_prv(4), shape_prv(5), shape_prv(6), shape_prv(7)))

      IF (PRESENT(source)) THEN
         array(:,:,:,:,:,:,:) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
END MODULE
