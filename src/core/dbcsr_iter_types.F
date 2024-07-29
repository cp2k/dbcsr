!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_iter_types
   !! Collection of routines to handle the iteration info
   USE dbcsr_kinds, ONLY: default_path_length, &
                          default_string_length
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   ! iteration_info
   PUBLIC :: dbcsr_iteration_info_type, &
             dbcsr_iteration_info_create, &
             dbcsr_iteration_info_retain, &
             dbcsr_iteration_info_release

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_iter_types'
   LOGICAL, PRIVATE, PARAMETER          :: debug_this_module = .FALSE.
   INTEGER, SAVE, PRIVATE               :: last_it_info_id = 0

   TYPE dbcsr_iteration_info_type
      !! contains the information about the current state of the program
      !! to be able to decide if output is necessary

      INTEGER                              :: ref_count = -1, id_nr = -1
      INTEGER                              :: print_level = -1, n_rlevel = -1
      INTEGER, DIMENSION(:), POINTER       :: iteration => NULL()
      LOGICAL, DIMENSION(:), POINTER       :: last_iter => NULL()
      CHARACTER(len=default_string_length) :: project_name = ""
      CHARACTER(LEN=default_string_length), &
         DIMENSION(:), POINTER           :: level_name => NULL()
   END TYPE dbcsr_iteration_info_type

CONTAINS

   SUBROUTINE dbcsr_iteration_info_create(iteration_info, project_name)
      !! creates an output info object

      TYPE(dbcsr_iteration_info_type), POINTER           :: iteration_info
         !! the object to create
      CHARACTER(len=*), INTENT(in)                       :: project_name
         !! name of the project, used to create the filenames

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_iteration_info_create', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: stat

      ALLOCATE (iteration_info, stat=stat)
      IF (stat /= 0) &
         DBCSR_ABORT(routineP//" could not allocate iteration_info")

      last_it_info_id = last_it_info_id + 1
      iteration_info%id_nr = last_it_info_id
      iteration_info%ref_count = 1
      iteration_info%print_level = 2
      iteration_info%n_rlevel = 1
      iteration_info%project_name = project_name
      NULLIFY (iteration_info%iteration)
      NULLIFY (iteration_info%level_name)
      NULLIFY (iteration_info%last_iter)
      ALLOCATE (iteration_info%iteration(iteration_info%n_rlevel), stat=stat)
      IF (stat /= 0) THEN
         DBCSR_ABORT(routineP//" iteration_info%iteration allocation")
      END IF
      ALLOCATE (iteration_info%level_name(iteration_info%n_rlevel), stat=stat)
      IF (stat /= 0) THEN
         DBCSR_ABORT(routineP//" iteration_info%level_name allocation")
      END IF
      ALLOCATE (iteration_info%last_iter(iteration_info%n_rlevel), stat=stat)
      IF (stat /= 0) THEN
         DBCSR_ABORT(routineP//" iteration_info%last_iter allocation")
      END IF
      iteration_info%iteration(iteration_info%n_rlevel) = 1
      iteration_info%level_name(iteration_info%n_rlevel) = "__ROOT__"
      iteration_info%last_iter(iteration_info%n_rlevel) = .FALSE.

   END SUBROUTINE dbcsr_iteration_info_create

   SUBROUTINE dbcsr_iteration_info_retain(iteration_info)
      !! retains the iteration_info (see doc/ReferenceCounting.html)

      TYPE(dbcsr_iteration_info_type), POINTER           :: iteration_info
         !! the iteration_info to retain

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_iteration_info_retain', &
                                     routineP = moduleN//':'//routineN

      IF (.NOT. ASSOCIATED(iteration_info)) THEN
         DBCSR_ABORT(routineP//" iteration_info not associated")
      END IF
      IF (iteration_info%ref_count <= 0) THEN
         DBCSR_ABORT(routineP//" iteration_info%ref_counf<=0")
      END IF
      iteration_info%ref_count = iteration_info%ref_count + 1
   END SUBROUTINE dbcsr_iteration_info_retain

   SUBROUTINE dbcsr_iteration_info_release(iteration_info)
      !! releases the iteration_info (see doc/ReferenceCounting.html)

      TYPE(dbcsr_iteration_info_type), POINTER           :: iteration_info
         !! the iteration_info to release

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_iteration_info_release', &
                                     routineP = moduleN//':'//routineN

      IF (ASSOCIATED(iteration_info)) THEN
         IF (iteration_info%ref_count <= 0) THEN
            DBCSR_ABORT(routineP//" iteration_info%ref_counf<=0")
         END IF
         iteration_info%ref_count = iteration_info%ref_count - 1
         IF (iteration_info%ref_count == 0) THEN
            IF (ASSOCIATED(iteration_info%iteration)) THEN
               DEALLOCATE (iteration_info%iteration)
            END IF
            IF (ASSOCIATED(iteration_info%last_iter)) THEN
               DEALLOCATE (iteration_info%last_iter)
            END IF
            IF (ASSOCIATED(iteration_info%level_name)) THEN
               DEALLOCATE (iteration_info%level_name)
            END IF
            DEALLOCATE (iteration_info)
         END IF
      END IF
   END SUBROUTINE dbcsr_iteration_info_release

END MODULE dbcsr_iter_types
