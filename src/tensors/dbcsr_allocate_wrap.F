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

#:include "dbcsr_tensor.fypp"
#:set maxdim = fortran_max_ndim

   USE dbcsr_kinds, ONLY: ${uselist(dtype_float_prec)}$

#include "base/dbcsr_base_uses.f90"
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: allocate_any

   INTERFACE allocate_any
#:for dparam, dtype, dsuffix in dtype_all_list
#:for dim in range(1, maxdim+1)
      MODULE PROCEDURE allocate_${dim}$d_${dsuffix}$
#:endfor
#:endfor
   END INTERFACE

CONTAINS

#:for dparam, dtype, dsuffix in dtype_all_list
#:for dim in range(1, maxdim+1)
   SUBROUTINE allocate_${dim}$d_${dsuffix}$ (array, shape_spec, source, order)
      !! Allocate array according to shape_spec. Possibly assign array from source.
      !! @note  this does not fully replace Fortran RESHAPE intrinsic since source and target array must
      !! have same rank.

      ${dtype}$, DIMENSION(${shape_colon(dim)}$), ALLOCATABLE, INTENT(OUT) :: array
         !! target array.
      INTEGER, DIMENSION(${dim}$), INTENT(IN), OPTIONAL                    :: shape_spec
         !! shape of array to be allocated. If shape is not specified, it is derived from source.
      ${dtype}$, DIMENSION(${shape_colon(dim)}$), INTENT(IN), OPTIONAL     :: source
         !! source array to be copied to target array, must have same rank as target array.
      INTEGER, DIMENSION(${dim}$), INTENT(IN), OPTIONAL                    :: order
         !! in which order to copy source to array (same convention as RESHAPE intrinsic).
      INTEGER, DIMENSION(${dim}$)                                          :: order_prv
      INTEGER, DIMENSION(${dim}$)                                          :: shape_prv
      INTEGER                                                              :: i

      DBCSR_ASSERT(PRESENT(shape_spec) .OR. PRESENT(source))

      IF (PRESENT(order)) THEN
         order_prv(:) = order(:)
      ELSE
         order_prv(:) = (/(i, i=1, ${dim}$)/)
      ENDIF

      IF (PRESENT(source) .AND. .NOT. PRESENT(shape_spec)) THEN
         shape_prv(order_prv) = SHAPE(source)
      ELSE
         shape_prv(order_prv) = shape_spec
      ENDIF

      ALLOCATE (array(${arrlist("shape_prv", nmax=dim)}$))

      IF (PRESENT(source)) THEN
         array(${shape_colon(dim)}$) = RESHAPE(source, shape_prv, order=order_prv)
      ENDIF
   END SUBROUTINE
#:endfor
#:endfor
END MODULE
