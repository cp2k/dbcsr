!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_tas_types
   !! DBCSR tall-and-skinny base types.
   !! Mostly wrappers around existing DBCSR routines.

   USE dbcsr_tas_global, ONLY: &
      dbcsr_tas_distribution, dbcsr_tas_rowcol_data
   USE dbcsr_types, ONLY: &
      dbcsr_distribution_obj, dbcsr_iterator, dbcsr_type
   USE dbcsr_kinds, ONLY: int_8
   USE dbcsr_data_types, ONLY: dbcsr_scalar_type
   USE dbcsr_mpiwrap, ONLY: mp_comm_type
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_tas_types'

   PUBLIC :: &
      dbcsr_tas_distribution_type, &
      dbcsr_tas_iterator, &
      dbcsr_tas_split_info, &
      dbcsr_tas_type, &
      dbcsr_tas_mm_storage

   ! info on MPI Cartesian grid that is split on MPI subgroups.
   ! info on distribution of matrix rows / columns to different subgroups.
   TYPE dbcsr_tas_split_info
      TYPE(mp_comm_type) :: mp_comm ! global communicator
      INTEGER, DIMENSION(2) :: pdims ! dimensions of process grid
      INTEGER :: igroup ! which subgroup do I belong to
      INTEGER :: ngroup ! how many groups in total
      INTEGER :: split_rowcol ! split row or column?
      INTEGER :: pgrid_split_size ! how many process rows/cols in subgroups
      INTEGER :: group_size ! group size (how many cores) of subgroups
      TYPE(mp_comm_type) :: mp_comm_group ! sub communicator
      INTEGER, ALLOCATABLE :: ngroup_opt ! optimal number of groups (split factor)
      LOGICAL, DIMENSION(2) :: strict_split = [.FALSE., .FALSE.]
      ! if .true., split factor should not be modified (2 parameters for current and general settings)
      INTEGER, POINTER :: refcount => NULL() ! lightweight reference counting for communicators
   END TYPE

   TYPE dbcsr_tas_distribution_type
      TYPE(dbcsr_tas_split_info) :: info
      TYPE(dbcsr_distribution_obj) :: dbcsr_dist
      CLASS(dbcsr_tas_distribution), ALLOCATABLE :: row_dist
      CLASS(dbcsr_tas_distribution), ALLOCATABLE :: col_dist
      INTEGER(KIND=int_8), ALLOCATABLE, DIMENSION(:) :: local_rowcols
   END TYPE

   ! storage for batched matrix multiplication
   TYPE dbcsr_tas_mm_storage
      TYPE(dbcsr_tas_type), POINTER :: store_batched => NULL() ! intermediate replicated matrix
      TYPE(dbcsr_tas_type), POINTER :: store_batched_repl => NULL() ! intermediate replicated matrix
      LOGICAL :: batched_out ! whether replicated matrix has been changed in mm and should be copied to actual matrix
      LOGICAL :: batched_trans
      TYPE(dbcsr_scalar_type) :: batched_beta
   END TYPE

   ! type for tall-and-skinny matrices
   TYPE dbcsr_tas_type
      TYPE(dbcsr_tas_distribution_type)  :: dist
      CLASS(dbcsr_tas_rowcol_data), ALLOCATABLE :: row_blk_size
      CLASS(dbcsr_tas_rowcol_data), ALLOCATABLE :: col_blk_size

      TYPE(dbcsr_type) :: matrix ! matrix on subgroup
      INTEGER(KIND=int_8) :: nblkrows ! total number of rows
      INTEGER(KIND=int_8) :: nblkcols ! total number of columns
      INTEGER(KIND=int_8) :: nblkrowscols_split ! nblkrows or nblkcols depending on which is splitted
      INTEGER(KIND=int_8) :: nfullrows ! total number of full (not blocked) rows
      INTEGER(KIND=int_8) :: nfullcols ! total number of full (not blocked) columns
      LOGICAL :: valid = .FALSE. ! has been created?

      ! storage and flags for batched matrix multiplication
      INTEGER :: do_batched = 0 ! state flag for batched multiplication
      TYPE(dbcsr_tas_mm_storage), ALLOCATABLE :: mm_storage ! storage for batched processing of matrix matrix multiplication.
      LOGICAL :: has_opt_pgrid = .FALSE. ! whether pgrid was automatically optimized
   END TYPE

   TYPE dbcsr_tas_iterator
      TYPE(dbcsr_tas_split_info) :: info
      TYPE(dbcsr_tas_distribution_type) :: dist
      TYPE(dbcsr_iterator) :: iter
   END TYPE dbcsr_tas_iterator

END MODULE
