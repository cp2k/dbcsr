!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tas_global
   !! Global data (distribution and block sizes) for tall-and-skinny matrices
   !! For very sparse matrices with one very large dimension, storing array data of the same size
   !! as the matrix dimensions may require too much memory and we need to compute them on the fly for a
   !! given row or column. Hence global array data such as distribution and block sizes are specified as
   !! function objects, leaving up to the caller how to efficiently store global data.

   USE dbcsr_kinds, ONLY: int_8
#include "../base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tas_global'

   PUBLIC :: &
      dbcsr_tas_blk_size_arb, &
      dbcsr_tas_blk_size_repl, &
      dbcsr_tas_blk_size_one, &
      dbcsr_tas_dist_arb, &
      dbcsr_tas_dist_arb_default, &
      dbcsr_tas_dist_cyclic, &
      dbcsr_tas_dist_repl, &
      dbcsr_tas_distribution, &
      dbcsr_tas_rowcol_data, &
      cyclic_weighted_dist

   ! abstract type for distribution vectors along one dimension
   TYPE, ABSTRACT :: dbcsr_tas_distribution
      ! number of process rows / columns:
      INTEGER :: nprowcol
      ! number of matrix rows / columns:
      INTEGER(KIND=int_8) :: nmrowcol
   CONTAINS
      ! map matrix rows/cols to distribution rows/cols:
      PROCEDURE(rowcol_dist), deferred :: dist
      ! map distribution rows/cols to matrix rows/cols:
      PROCEDURE(dist_rowcols), deferred :: rowcols
   END TYPE

   ! type for cyclic (round robin) distribution:
   ! - may not be load balanced for arbitrary block sizes
   ! - memory efficient for large dimensions
   TYPE, EXTENDS(dbcsr_tas_distribution) :: dbcsr_tas_dist_cyclic
      INTEGER :: split_size
   CONTAINS
      PROCEDURE :: dist => cyclic_dist
      PROCEDURE :: rowcols => cyclic_rowcols
   END TYPE

   ! type for arbitrary distributions
   ! - stored as an array
   ! - not memory efficient for large dimensions
   TYPE, EXTENDS(dbcsr_tas_distribution) :: dbcsr_tas_dist_arb
      INTEGER, DIMENSION(:), ALLOCATABLE :: dist_vec
   CONTAINS
      PROCEDURE :: dist => arb_dist
      PROCEDURE :: rowcols => arb_rowcols
   END TYPE

   ! type for replicated distribution
   ! - a submatrix distribution replicated on all process groups
   ! - memory efficient for large dimensions
   TYPE, EXTENDS(dbcsr_tas_distribution) :: dbcsr_tas_dist_repl
      INTEGER, DIMENSION(:), ALLOCATABLE :: dist_vec
      INTEGER :: nmrowcol_local
      INTEGER :: n_repl
      INTEGER :: dist_size
   CONTAINS
      PROCEDURE :: dist => repl_dist
      PROCEDURE :: rowcols => repl_rowcols
   END TYPE

   ! abstract type for integer data (e.g. block sizes) along one dimension
   TYPE, ABSTRACT :: dbcsr_tas_rowcol_data
      ! number of matrix rows / columns (blocks):
      INTEGER(KIND=int_8) :: nmrowcol
      ! number of matrix rows / columns (elements):
      INTEGER(KIND=int_8) :: nfullrowcol
   CONTAINS
      ! integer data for each block row / col
      PROCEDURE(rowcol_data), deferred :: DATA
   END TYPE

   ! type for arbitrary block sizes
   ! - stored as an array
   ! - not memory efficient for large dimensions
   TYPE, EXTENDS(dbcsr_tas_rowcol_data) :: dbcsr_tas_blk_size_arb
      INTEGER, DIMENSION(:), ALLOCATABLE :: blk_size_vec
   CONTAINS
      PROCEDURE :: DATA => blk_size_arb
   END TYPE

   ! type for replicated block sizes
   ! - submatrix block sizes replicated on all process groups
   ! - memory efficient for large dimensions
   TYPE, EXTENDS(dbcsr_tas_rowcol_data) :: dbcsr_tas_blk_size_repl
      INTEGER, DIMENSION(:), ALLOCATABLE :: blk_size_vec
      INTEGER :: nmrowcol_local
   CONTAINS
      PROCEDURE :: DATA => blk_size_repl
   END TYPE

   ! type for blocks of size one
   ! - memory efficient for large dimensions
   TYPE, EXTENDS(dbcsr_tas_rowcol_data) :: dbcsr_tas_blk_size_one
   CONTAINS
      PROCEDURE :: DATA => blk_size_one
   END TYPE

   ABSTRACT INTERFACE
      FUNCTION rowcol_dist(t, rowcol)
         !! map matrix rows/cols to distribution rows/cols:
         IMPORT :: dbcsr_tas_distribution, int_8
         CLASS(dbcsr_tas_distribution), INTENT(IN) :: t
         INTEGER(KIND=int_8), INTENT(IN) :: rowcol
         INTEGER :: rowcol_dist
      END FUNCTION

      FUNCTION dist_rowcols(t, dist)
         !! map distribution rows/cols to matrix rows/cols:
         IMPORT :: dbcsr_tas_distribution, int_8
         CLASS(dbcsr_tas_distribution), INTENT(IN) :: t
         INTEGER, INTENT(IN) :: dist
         INTEGER(KIND=int_8), DIMENSION(:), ALLOCATABLE :: dist_rowcols
      END FUNCTION

      FUNCTION rowcol_data(t, rowcol)
         !! integer data for each block row / col
         IMPORT :: dbcsr_tas_rowcol_data, int_8
         CLASS(dbcsr_tas_rowcol_data), INTENT(IN) :: t
         INTEGER(KIND=int_8), INTENT(IN) :: rowcol
         INTEGER :: rowcol_data
      END FUNCTION

   END INTERFACE

   INTERFACE dbcsr_tas_dist_cyclic
      MODULE PROCEDURE new_dbcsr_tas_dist_cyclic
   END INTERFACE

   INTERFACE dbcsr_tas_dist_arb
      MODULE PROCEDURE new_dbcsr_tas_dist_arb
   END INTERFACE

   INTERFACE dbcsr_tas_dist_repl
      MODULE PROCEDURE new_dbcsr_tas_dist_repl
   END INTERFACE

   INTERFACE dbcsr_tas_blk_size_arb
      MODULE PROCEDURE new_dbcsr_tas_blk_size_arb
   END INTERFACE

   INTERFACE dbcsr_tas_blk_size_repl
      MODULE PROCEDURE new_dbcsr_tas_blk_size_repl
   END INTERFACE

   INTERFACE dbcsr_tas_blk_size_one
      MODULE PROCEDURE new_dbcsr_tas_blk_size_one
   END INTERFACE

CONTAINS
   FUNCTION blk_size_arb(t, rowcol)
      CLASS(dbcsr_tas_blk_size_arb), INTENT(IN) :: t
      INTEGER(KIND=int_8), INTENT(IN) :: rowcol
      INTEGER :: blk_size_arb
      blk_size_arb = t%blk_size_vec(rowcol)
   END FUNCTION

   FUNCTION blk_size_repl(t, rowcol)
      CLASS(dbcsr_tas_blk_size_repl), INTENT(IN) :: t
      INTEGER(KIND=int_8), INTENT(IN) :: rowcol
      INTEGER :: blk_size_repl
      INTEGER :: igroup
      INTEGER :: rowcol_local

      igroup = INT((rowcol - 1_int_8)/t%nmrowcol_local)
      rowcol_local = INT(MOD(rowcol - 1_int_8, INT(t%nmrowcol_local, KIND=int_8))) + 1
      blk_size_repl = t%blk_size_vec(rowcol_local)

   END FUNCTION

   FUNCTION blk_size_one(t, rowcol)
      CLASS(dbcsr_tas_blk_size_one), INTENT(IN) :: t
      INTEGER(KIND=int_8), INTENT(IN) :: rowcol
      INTEGER :: blk_size_one

      MARK_USED(t)
      MARK_USED(rowcol)
      blk_size_one = 1
   END FUNCTION

   FUNCTION new_dbcsr_tas_blk_size_arb(blk_size_vec)
      INTEGER, DIMENSION(:), INTENT(IN)                  :: blk_size_vec
      TYPE(dbcsr_tas_blk_size_arb)                         :: new_dbcsr_tas_blk_size_arb

      ALLOCATE (new_dbcsr_tas_blk_size_arb%blk_size_vec(SIZE(blk_size_vec)))
      new_dbcsr_tas_blk_size_arb%blk_size_vec(:) = blk_size_vec(:)
      new_dbcsr_tas_blk_size_arb%nmrowcol = SIZE(blk_size_vec)
      new_dbcsr_tas_blk_size_arb%nfullrowcol = SUM(blk_size_vec)
   END FUNCTION

   FUNCTION new_dbcsr_tas_blk_size_repl(blk_size_vec, n_repl)
      INTEGER, DIMENSION(:), INTENT(IN)                  :: blk_size_vec
      INTEGER, INTENT(IN)                                :: n_repl
      TYPE(dbcsr_tas_blk_size_repl)                        :: new_dbcsr_tas_blk_size_repl

      new_dbcsr_tas_blk_size_repl%nmrowcol_local = SIZE(blk_size_vec)
      ALLOCATE (new_dbcsr_tas_blk_size_repl%blk_size_vec(new_dbcsr_tas_blk_size_repl%nmrowcol_local))
      new_dbcsr_tas_blk_size_repl%blk_size_vec(:) = blk_size_vec(:)
      new_dbcsr_tas_blk_size_repl%nmrowcol = new_dbcsr_tas_blk_size_repl%nmrowcol_local*n_repl
      new_dbcsr_tas_blk_size_repl%nfullrowcol = SUM(blk_size_vec)*n_repl
   END FUNCTION

   FUNCTION new_dbcsr_tas_blk_size_one(nrowcol)
      INTEGER(KIND=int_8), INTENT(IN)      :: nrowcol
      TYPE(dbcsr_tas_blk_size_one)         :: new_dbcsr_tas_blk_size_one

      new_dbcsr_tas_blk_size_one%nmrowcol = nrowcol
      new_dbcsr_tas_blk_size_one%nfullrowcol = nrowcol
   END FUNCTION

   FUNCTION arb_dist(t, rowcol)
      CLASS(dbcsr_tas_dist_arb), INTENT(IN) :: t
      INTEGER(KIND=int_8), INTENT(IN) :: rowcol
      INTEGER :: arb_dist

      arb_dist = t%dist_vec(rowcol)
   END FUNCTION

   FUNCTION repl_dist(t, rowcol)
      CLASS(dbcsr_tas_dist_repl), INTENT(IN) :: t
      INTEGER(KIND=int_8), INTENT(IN) :: rowcol
      INTEGER :: repl_dist
      INTEGER :: rowcol_local
      INTEGER :: igroup

      igroup = INT((rowcol - 1_int_8)/t%nmrowcol_local)
      rowcol_local = INT(MOD(rowcol - 1_int_8, INT(t%nmrowcol_local, KIND=int_8))) + 1

      repl_dist = t%dist_vec(rowcol_local) + igroup*t%dist_size

   END FUNCTION

   FUNCTION repl_rowcols(t, dist)
      CLASS(dbcsr_tas_dist_repl), INTENT(IN) :: t
      INTEGER, INTENT(IN) :: dist
      INTEGER :: nrowcols
      INTEGER(KIND=int_8), DIMENSION(:), ALLOCATABLE :: repl_rowcols, rowcols_tmp
      INTEGER :: igroup
      INTEGER :: rowcol, count
      LOGICAL :: cond

      igroup = dist/t%dist_size

      nrowcols = t%nmrowcol_local
      count = 0
      ALLOCATE (rowcols_tmp(nrowcols))
      rowcols_tmp(:) = 0
      DO rowcol = 1, nrowcols
         cond = t%dist_vec(rowcol) + igroup*t%dist_size == dist

         IF (cond) THEN
            count = count + 1
            rowcols_tmp(count) = rowcol
         ENDIF
      ENDDO

      ALLOCATE (repl_rowcols(count))
      repl_rowcols(:) = rowcols_tmp(1:count) + igroup*t%nmrowcol_local

   END FUNCTION

   FUNCTION arb_rowcols(t, dist)
      CLASS(dbcsr_tas_dist_arb), INTENT(IN) :: t
      INTEGER, INTENT(IN) :: dist
      INTEGER(KIND=int_8) :: rowcol, nrowcols
      INTEGER(KIND=int_8), DIMENSION(:), ALLOCATABLE :: arb_rowcols, rowcols_tmp
      INTEGER :: count

      nrowcols = t%nmrowcol
      count = 0
      ALLOCATE (rowcols_tmp(nrowcols))
      rowcols_tmp(:) = 0
      DO rowcol = 1, nrowcols
         IF (t%dist_vec(rowcol) == dist) THEN
            count = count + 1
            rowcols_tmp(count) = rowcol
         ENDIF
      ENDDO

      ALLOCATE (arb_rowcols(count))
      arb_rowcols(:) = rowcols_tmp(1:count)
   END FUNCTION

   FUNCTION new_dbcsr_tas_dist_cyclic(split_size, nprowcol, nmrowcol)
      INTEGER, INTENT(IN)                                :: split_size, nprowcol
      INTEGER(KIND=int_8), INTENT(IN)                    :: nmrowcol
      TYPE(dbcsr_tas_dist_cyclic)                          :: new_dbcsr_tas_dist_cyclic

      new_dbcsr_tas_dist_cyclic%split_size = split_size
      new_dbcsr_tas_dist_cyclic%nprowcol = nprowcol
      new_dbcsr_tas_dist_cyclic%nmrowcol = nmrowcol
   END FUNCTION

   FUNCTION new_dbcsr_tas_dist_arb(dist_vec, nprowcol, nmrowcol)
      INTEGER, DIMENSION(:), INTENT(IN)                  :: dist_vec
      INTEGER, INTENT(IN)                                :: nprowcol
      INTEGER(KIND=int_8), INTENT(IN)                    :: nmrowcol
      TYPE(dbcsr_tas_dist_arb)                             :: new_dbcsr_tas_dist_arb

      ALLOCATE (new_dbcsr_tas_dist_arb%dist_vec(nmrowcol))
      new_dbcsr_tas_dist_arb%dist_vec(:) = dist_vec(:)
      new_dbcsr_tas_dist_arb%nprowcol = nprowcol
      new_dbcsr_tas_dist_arb%nmrowcol = nmrowcol
   END FUNCTION

   FUNCTION dbcsr_tas_dist_arb_default(nprowcol, nmrowcol, block_sizes)
      !! Distribution that is more or less cyclic (round robin) and load balanced with different
      !! weights for each element.
      !! This is used for creating adhoc distributions whenever matrices are mapped to new grids.
      !! Only for small dimensions since distribution is created as an array

      INTEGER                           :: nprowcol
      INTEGER(KIND=int_8), INTENT(IN)   :: nmrowcol
      CLASS(dbcsr_tas_rowcol_data), INTENT(IN) :: block_sizes
      TYPE(dbcsr_tas_dist_arb)            :: dbcsr_tas_dist_arb_default
      INTEGER, DIMENSION(nmrowcol) :: dist_vec, bsize_vec
      INTEGER(KIND=int_8) :: ind

      DO ind = 1, nmrowcol
         bsize_vec(ind) = block_sizes%data(ind)
      ENDDO

      CALL cyclic_weighted_dist(INT(nmrowcol), nprowcol, bsize_vec, dist_vec)
      dbcsr_tas_dist_arb_default = dbcsr_tas_dist_arb(dist_vec, nprowcol, nmrowcol)

   END FUNCTION

   SUBROUTINE cyclic_weighted_dist(nel, nbin, weights, dist)
      INTEGER, INTENT(IN) :: nel
      INTEGER, INTENT(IN) :: nbin
      INTEGER, DIMENSION(nel), INTENT(IN)            :: weights
      INTEGER, DIMENSION(nel), INTENT(OUT)            :: dist
      INTEGER, DIMENSION(nbin)                        :: occup
      INTEGER                :: iel, ibin
      INTEGER                :: niter

      occup(:) = 0
      ibin = 0
      DO iel = 1, nel
         niter = 0
         ibin = MOD(ibin + 1, nbin)
         DO WHILE (occup(ibin + 1) + weights(iel) .GE. MAXVAL(occup))
            IF (MINLOC(occup, DIM=1) == ibin + 1) EXIT
            ibin = MOD(ibin + 1, nbin)
            niter = niter + 1
         ENDDO
         dist(iel) = ibin
         occup(ibin + 1) = occup(ibin + 1) + weights(iel)
      ENDDO

   END SUBROUTINE

   FUNCTION new_dbcsr_tas_dist_repl(dist_vec, nprowcol, nmrowcol, n_repl, dist_size)
      INTEGER, DIMENSION(:), INTENT(IN)                  :: dist_vec
      INTEGER, INTENT(IN)                                :: nprowcol, nmrowcol, n_repl, dist_size
      TYPE(dbcsr_tas_dist_repl)                            :: new_dbcsr_tas_dist_repl

      new_dbcsr_tas_dist_repl%n_repl = n_repl
      new_dbcsr_tas_dist_repl%dist_size = dist_size
      ALLOCATE (new_dbcsr_tas_dist_repl%dist_vec(nmrowcol))
      new_dbcsr_tas_dist_repl%dist_vec(:) = MOD(dist_vec(:), dist_size)
      new_dbcsr_tas_dist_repl%nprowcol = nprowcol
      new_dbcsr_tas_dist_repl%nmrowcol_local = nmrowcol
      new_dbcsr_tas_dist_repl%nmrowcol = nmrowcol*n_repl
   END FUNCTION

   FUNCTION cyclic_dist(t, rowcol)
      CLASS(dbcsr_tas_dist_cyclic), INTENT(IN) :: t
      INTEGER(KIND=int_8), INTENT(IN) :: rowcol
      INTEGER :: cyclic_dist

      cyclic_dist = INT(MOD((rowcol - 1)/INT(t%split_size, KIND=int_8), INT(t%nprowcol, KIND=int_8)))

   END FUNCTION

   FUNCTION cyclic_rowcols(t, dist)
      CLASS(dbcsr_tas_dist_cyclic), INTENT(IN) :: t
      INTEGER, INTENT(IN) :: dist
      INTEGER(KIND=int_8), DIMENSION(:), ALLOCATABLE :: cyclic_rowcols
      INTEGER :: count, nsplit, isplit, irowcol, max_size
      INTEGER(KIND=int_8) :: rowcol
      INTEGER(KIND=int_8), DIMENSION(:), ALLOCATABLE :: rowcols_tmp

      nsplit = INT((t%nmrowcol - 1)/INT(t%split_size, KIND=int_8) + 1_int_8)
      max_size = nsplit*t%split_size
      ALLOCATE (rowcols_tmp(max_size))
      rowcols_tmp(:) = 0
      count = 0
      loop: DO isplit = 1, nsplit
         DO irowcol = 1, t%split_size
            rowcol = INT((dist + (isplit - 1)*t%nprowcol), KIND=int_8)*INT(t%split_size, KIND=int_8) + &
                     INT(irowcol, KIND=int_8)
            IF (rowcol > t%nmrowcol) THEN
               EXIT loop
            ELSE
               count = count + 1
               rowcols_tmp(count) = rowcol
            ENDIF
         ENDDO
      ENDDO loop

      ALLOCATE (cyclic_rowcols(count))
      cyclic_rowcols(:) = rowcols_tmp(1:count)
   END FUNCTION

END MODULE
