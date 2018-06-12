!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright (C) 2000 - 2018  CP2K developers group                                               !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief  set of type/routines to handle the storage of results in force_envs
!> \author fschiff (12.2007)
!> \par    History
!>         - 10.2008 Teodoro Laino [tlaino] - University of Zurich
!>                   major rewriting:
!>                   - information stored in a proper type (not in a character!)
!>                   - module more lean
! **************************************************************************************************
MODULE dbcsr_result_methods
   USE dbcsr_kinds,                     ONLY: default_string_length,&
                                              dp
   USE dbcsr_memory_utilities,          ONLY: reallocate
   USE dbcsr_message_passing,           ONLY: mp_bcast
   USE dbcsr_para_types,                ONLY: dbcsr_para_env_type
   USE dbcsr_result_types,              ONLY: &
        dbcsr_result_clean, dbcsr_result_copy, dbcsr_result_create, dbcsr_result_release, &
        dbcsr_result_type, dbcsr_result_value_copy, dbcsr_result_value_create, &
        dbcsr_result_value_init, dbcsr_result_value_p_reallocate, result_type_integer, &
        result_type_logical, result_type_real
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_result_methods'

   PUBLIC :: put_results, &
             test_for_result, &
             get_results, &
             dbcsr_results_erase, &
             dbcsr_results_mp_bcast

   INTERFACE put_results
      MODULE PROCEDURE put_result_r1, put_result_r2
   END INTERFACE

   INTERFACE get_results
      MODULE PROCEDURE get_result_r1, get_result_r2, get_nreps
   END INTERFACE

CONTAINS

! **************************************************************************************************
!> \brief Store a 1D array of reals in result_list
!> \param results ...
!> \param description ...
!> \param values ...
!> \par History
!>      12.2007 created
!>      10.2008 Teodoro Laino [tlaino] - major rewriting
!> \author fschiff
! **************************************************************************************************
   SUBROUTINE put_result_r1(results, description, values)
      TYPE(dbcsr_result_type), POINTER                   :: results
      CHARACTER(LEN=default_string_length)               :: description
      REAL(KIND=dp), DIMENSION(:)                        :: values

      CHARACTER(len=*), PARAMETER :: routineN = 'put_result_r1', routineP = moduleN//':'//routineN

      INTEGER                                            :: isize, jsize
      LOGICAL                                            :: check

      DBCSR_ASSERT(ASSOCIATED(results))
      DBCSR_ASSERT(description(1:1) == '[')
      check = SIZE(results%result_label) == SIZE(results%result_value)
      DBCSR_ASSERT(check)
      isize = SIZE(results%result_label)
      jsize = SIZE(values)

      CALL reallocate(results%result_label, 1, isize+1)
      CALL dbcsr_result_value_p_reallocate(results%result_value, 1, isize+1)

      results%result_label(isize+1) = description
      CALL dbcsr_result_value_init(results%result_value(isize+1)%value, result_type_real, jsize)
      results%result_value(isize+1)%value%real_type = values

   END SUBROUTINE put_result_r1

! **************************************************************************************************
!> \brief Store a 2D array of reals in result_list
!> \param results ...
!> \param description ...
!> \param values ...
!> \par History
!>      12.2007 created
!>      10.2008 Teodoro Laino [tlaino] - major rewriting
!> \author fschiff
! **************************************************************************************************
   SUBROUTINE put_result_r2(results, description, values)
      TYPE(dbcsr_result_type), POINTER                   :: results
      CHARACTER(LEN=default_string_length)               :: description
      REAL(KIND=dp), DIMENSION(:, :)                     :: values

      CHARACTER(len=*), PARAMETER :: routineN = 'put_result_r2', routineP = moduleN//':'//routineN

      INTEGER                                            :: isize, jsize
      LOGICAL                                            :: check

      DBCSR_ASSERT(ASSOCIATED(results))
      DBCSR_ASSERT(description(1:1) == '[')
      check = SIZE(results%result_label) == SIZE(results%result_value)
      DBCSR_ASSERT(check)
      isize = SIZE(results%result_label)
      jsize = SIZE(values, 1)*SIZE(values, 2)

      CALL reallocate(results%result_label, 1, isize+1)
      CALL dbcsr_result_value_p_reallocate(results%result_value, 1, isize+1)

      results%result_label(isize+1) = description
      CALL dbcsr_result_value_init(results%result_value(isize+1)%value, result_type_real, jsize)
      results%result_value(isize+1)%value%real_type = RESHAPE(values, (/jsize/))

   END SUBROUTINE put_result_r2

! **************************************************************************************************
!> \brief test for a certain result in the result_list
!> \param results ...
!> \param description ...
!> \return ...
!> \par History
!>      10.2013
!> \author Mandes
! **************************************************************************************************
   FUNCTION test_for_result(results, description) RESULT(res_exist)
      TYPE(dbcsr_result_type), POINTER                   :: results
      CHARACTER(LEN=default_string_length)               :: description
      LOGICAL                                            :: res_exist

      CHARACTER(len=*), PARAMETER :: routineN = 'test_for_result', &
         routineP = moduleN//':'//routineN

      INTEGER                                            :: i, nlist

      DBCSR_ASSERT(ASSOCIATED(results))
      nlist = SIZE(results%result_value)
      res_exist = .FALSE.
      DO i = 1, nlist
         IF (TRIM(results%result_label(i)) == TRIM(description)) THEN
            res_exist = .TRUE.
            EXIT
         END IF
      END DO

   END FUNCTION test_for_result

! **************************************************************************************************
!> \brief gets the required part out of the result_list
!> \param results ...
!> \param description ...
!> \param values ...
!> \param nval      : if more than one entry for a given description is given you may choose
!>                    which entry you want
!> \param n_rep     : integer indicating how many times the section exists in result_list
!> \param n_entries : gets the number of lines used for a given description
!> \par History
!>      12.2007 created
!>      10.2008 Teodoro Laino [tlaino] - major rewriting
!> \author fschiff
! **************************************************************************************************
   SUBROUTINE get_result_r1(results, description, values, nval, n_rep, n_entries)
      TYPE(dbcsr_result_type), POINTER                   :: results
      CHARACTER(LEN=default_string_length)               :: description
      REAL(KIND=dp), DIMENSION(:)                        :: values
      INTEGER, OPTIONAL                                  :: nval, n_rep, n_entries

      CHARACTER(len=*), PARAMETER :: routineN = 'get_result_r1', routineP = moduleN//':'//routineN

      INTEGER                                            :: i, k, nlist, nrep, size_res, size_values

      DBCSR_ASSERT(ASSOCIATED(results))
      nlist = SIZE(results%result_value)
      DBCSR_ASSERT(description(1:1) == '[')
      DBCSR_ASSERT(SIZE(results%result_label) == nlist)
      nrep = 0
      DO i = 1, nlist
         IF (TRIM(results%result_label(i)) == TRIM(description)) nrep = nrep+1
      END DO

      IF (PRESENT(n_rep)) THEN
         n_rep = nrep
      END IF

      IF (nrep .LE. 0) &
         CALL dbcsr_abort(__LOCATION__, &
                          " Trying to access result ("//TRIM(description)//") which was never stored!")

      DO i = 1, nlist
         IF (TRIM(results%result_label(i)) == TRIM(description)) THEN
            IF (results%result_value(i)%value%type_in_use /= result_type_real) &
               DBCSR_ABORT("Attempt to retrieve a RESULT which is not a REAL!")

            size_res = SIZE(results%result_value(i)%value%real_type)
            EXIT
         END IF
      END DO
      IF (PRESENT(n_entries)) n_entries = size_res
      size_values = SIZE(values, 1)
      IF (PRESENT(nval)) THEN
         DBCSR_ASSERT(size_res == size_values)
      ELSE
         DBCSR_ASSERT(nrep*size_res == size_values)
      END IF
      k = 0
      DO i = 1, nlist
         IF (TRIM(results%result_label(i)) == TRIM(description)) THEN
            k = k+1
            IF (PRESENT(nval)) THEN
               IF (k == nval) THEN
                  values = results%result_value(i)%value%real_type
                  EXIT
               END IF
            ELSE
               values((k-1)*size_res+1:k*size_res) = results%result_value(i)%value%real_type
            END IF
         END IF
      END DO

   END SUBROUTINE get_result_r1

! **************************************************************************************************
!> \brief gets the required part out of the result_list
!> \param results ...
!> \param description ...
!> \param values ...
!> \param nval      : if more than one entry for a given description is given you may choose
!>                    which entry you want
!> \param n_rep     : integer indicating how many times the section exists in result_list
!> \param n_entries : gets the number of lines used for a given description
!> \par History
!>      12.2007 created
!>      10.2008 Teodoro Laino [tlaino] - major rewriting
!> \author fschiff
! **************************************************************************************************
   SUBROUTINE get_result_r2(results, description, values, nval, n_rep, n_entries)
      TYPE(dbcsr_result_type), POINTER                   :: results
      CHARACTER(LEN=default_string_length)               :: description
      REAL(KIND=dp), DIMENSION(:, :)                     :: values
      INTEGER, OPTIONAL                                  :: nval, n_rep, n_entries

      CHARACTER(len=*), PARAMETER :: routineN = 'get_result_r2', routineP = moduleN//':'//routineN

      INTEGER                                            :: i, k, nlist, nrep, size_res, size_values

      DBCSR_ASSERT(ASSOCIATED(results))
      nlist = SIZE(results%result_value)
      DBCSR_ASSERT(description(1:1) == '[')
      DBCSR_ASSERT(SIZE(results%result_label) == nlist)
      nrep = 0
      DO i = 1, nlist
         IF (TRIM(results%result_label(i)) == TRIM(description)) nrep = nrep+1
      END DO

      IF (PRESENT(n_rep)) THEN
         n_rep = nrep
      END IF

      IF (nrep .LE. 0) &
         CALL dbcsr_abort(__LOCATION__, &
                          " Trying to access result ("//TRIM(description)//") which was never stored!")

      DO i = 1, nlist
         IF (TRIM(results%result_label(i)) == TRIM(description)) THEN
            IF (results%result_value(i)%value%type_in_use /= result_type_real) &
               DBCSR_ABORT("Attempt to retrieve a RESULT which is not a REAL!")

            size_res = SIZE(results%result_value(i)%value%real_type)
            EXIT
         END IF
      END DO
      IF (PRESENT(n_entries)) n_entries = size_res
      size_values = SIZE(values, 1)*SIZE(values, 2)
      IF (PRESENT(nval)) THEN
         DBCSR_ASSERT(size_res == size_values)
      ELSE
         DBCSR_ASSERT(nrep*size_res == size_values)
      END IF
      k = 0
      DO i = 1, nlist
         IF (TRIM(results%result_label(i)) == TRIM(description)) THEN
            k = k+1
            IF (PRESENT(nval)) THEN
               IF (k == nval) THEN
                  values = RESHAPE(results%result_value(i)%value%real_type, (/SIZE(values, 1), SIZE(values, 2)/))
                  EXIT
               END IF
            ELSE
               values((k-1)*size_res+1:k*size_res, :) = RESHAPE(results%result_value(i)%value%real_type, &
                                                                (/SIZE(values, 1), SIZE(values, 2)/))
            END IF
         END IF
      END DO

   END SUBROUTINE get_result_r2

! **************************************************************************************************
!> \brief gets the required part out of the result_list
!> \param results ...
!> \param description ...
!> \param n_rep     : integer indicating how many times the section exists in result_list
!> \param n_entries : gets the number of lines used for a given description
!> \param type_in_use ...
!> \par History
!>      12.2007 created
!>      10.2008 Teodoro Laino [tlaino] - major rewriting
!> \author fschiff
! **************************************************************************************************
   SUBROUTINE get_nreps(results, description, n_rep, n_entries, type_in_use)
      TYPE(dbcsr_result_type), POINTER                   :: results
      CHARACTER(LEN=default_string_length)               :: description
      INTEGER, OPTIONAL                                  :: n_rep, n_entries, type_in_use

      CHARACTER(len=*), PARAMETER :: routineN = 'get_nreps', routineP = moduleN//':'//routineN

      INTEGER                                            :: I, nlist

      DBCSR_ASSERT(ASSOCIATED(results))
      nlist = SIZE(results%result_value)
      DBCSR_ASSERT(description(1:1) == '[')
      DBCSR_ASSERT(SIZE(results%result_label) == nlist)
      IF (PRESENT(n_rep)) THEN
         n_rep = 0
         DO i = 1, nlist
            IF (TRIM(results%result_label(i)) == TRIM(description)) n_rep = n_rep+1
         END DO
      END IF
      IF (PRESENT(n_entries)) THEN
         n_entries = 0
         DO i = 1, nlist
            IF (TRIM(results%result_label(i)) == TRIM(description)) THEN
               SELECT CASE (results%result_value (i)%value%type_in_use)
               CASE (result_type_real)
                  n_entries = n_entries+SIZE(results%result_value(i)%value%real_type)
               CASE (result_type_integer)
                  n_entries = n_entries+SIZE(results%result_value(i)%value%integer_type)
               CASE (result_type_logical)
                  n_entries = n_entries+SIZE(results%result_value(i)%value%logical_type)
               CASE DEFAULT
                  ! Type not implemented in dbcsr_result_type
                  DBCSR_ABORT("")
               END SELECT
               EXIT
            END IF
         END DO
      END IF
      IF (PRESENT(type_in_use)) THEN
         DO i = 1, nlist
            IF (TRIM(results%result_label(i)) == TRIM(description)) THEN
               type_in_use = results%result_value(i)%value%type_in_use
               EXIT
            END IF
         END DO
      END IF
   END SUBROUTINE get_nreps

! **************************************************************************************************
!> \brief erase a part of  result_list
!> \param results ...
!> \param description ...
!> \param nval : if more than one entry for a given description is given you may choose
!>               which entry you want to delete
!> \par History
!>      12.2007 created
!>      10.2008 Teodoro Laino [tlaino] - major rewriting
!> \author fschiff
! **************************************************************************************************
   SUBROUTINE dbcsr_results_erase(results, description, nval)
      TYPE(dbcsr_result_type), POINTER                   :: results
      CHARACTER(LEN=default_string_length), OPTIONAL     :: description
      INTEGER, OPTIONAL                                  :: nval

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_results_erase', &
         routineP = moduleN//':'//routineN

      INTEGER                                            :: entry_deleted, i, k, new_size, nlist, &
                                                            nrep
      TYPE(dbcsr_result_type), POINTER                   :: clean_results

      DBCSR_ASSERT(ASSOCIATED(results))
      new_size = 0
      IF (PRESENT(description)) THEN
         DBCSR_ASSERT(description(1:1) == '[')
         nlist = SIZE(results%result_value)
         nrep = 0
         DO i = 1, nlist
            IF (TRIM(results%result_label(i)) == TRIM(description)) nrep = nrep+1
         END DO
         IF (nrep .NE. 0) THEN
            k = 0
            entry_deleted = 0
            DO i = 1, nlist
               IF (TRIM(results%result_label(i)) == TRIM(description)) THEN
                  k = k+1
                  IF (PRESENT(nval)) THEN
                     IF (nval == k) THEN
                        entry_deleted = entry_deleted+1
                        EXIT
                     END IF
                  ELSE
                     entry_deleted = entry_deleted+1
                  END IF
               END IF
            END DO
            DBCSR_ASSERT(nlist-entry_deleted >= 0)
            new_size = nlist-entry_deleted
            NULLIFY (clean_results)
            CALL dbcsr_result_create(clean_results)
            CALL dbcsr_result_clean(clean_results)
            ALLOCATE (clean_results%result_label(new_size))
            ALLOCATE (clean_results%result_value(new_size))
            DO i = 1, new_size
               NULLIFY (clean_results%result_value(i)%value)
               CALL dbcsr_result_value_create(clean_results%result_value(i)%value)
            END DO
            k = 0
            DO i = 1, nlist
               IF (TRIM(results%result_label(i)) /= TRIM(description)) THEN
                  k = k+1
                  clean_results%result_label(k) = results%result_label(i)
                  CALL dbcsr_result_value_copy(clean_results%result_value(k)%value, &
                                               results%result_value(i)%value)
               END IF
            END DO
            CALL dbcsr_result_copy(clean_results, results)
            CALL dbcsr_result_release(clean_results)
         END IF
      ELSE
         CALL dbcsr_result_clean(results)
         ALLOCATE (results%result_label(new_size))
         ALLOCATE (results%result_value(new_size))
      END IF
   END SUBROUTINE dbcsr_results_erase

! **************************************************************************************************
!> \brief broadcast results type
!> \param results ...
!> \param source ...
!> \param para_env ...
!> \author  10.2008 Teodoro Laino [tlaino] - University of Zurich
! **************************************************************************************************
   SUBROUTINE dbcsr_results_mp_bcast(results, source, para_env)
      TYPE(dbcsr_result_type), POINTER                   :: results
      INTEGER, INTENT(IN)                                :: source
      TYPE(dbcsr_para_env_type), POINTER                 :: para_env

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_results_mp_bcast', &
         routineP = moduleN//':'//routineN

      INTEGER                                            :: i, nlist
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: size_value, type_in_use

      DBCSR_ASSERT(ASSOCIATED(results))
      nlist = 0
      IF (para_env%mepos == source) nlist = SIZE(results%result_value)
      CALL mp_bcast(nlist, source, para_env%group)

      ALLOCATE (size_value(nlist))
      ALLOCATE (type_in_use(nlist))
      IF (para_env%mepos == source) THEN
         DO i = 1, nlist
            CALL get_nreps(results, description=results%result_label(i), &
                           n_entries=size_value(i), type_in_use=type_in_use(i))
         END DO
      END IF
      CALL mp_bcast(size_value, source, para_env%group)
      CALL mp_bcast(type_in_use, source, para_env%group)

      IF (para_env%mepos /= source) THEN
         CALL dbcsr_result_clean(results)
         ALLOCATE (results%result_value(nlist))
         ALLOCATE (results%result_label(nlist))
         DO i = 1, nlist
            results%result_label(i) = ""
            NULLIFY (results%result_value(i)%value)
            CALL dbcsr_result_value_create(results%result_value(i)%value)
            CALL dbcsr_result_value_init(results%result_value(i)%value, &
                                         type_in_use=type_in_use(i), size_value=size_value(i))
         END DO
      END IF
      DO i = 1, nlist
         CALL mp_bcast(results%result_label(i), source, para_env%group)
         SELECT CASE (results%result_value (i)%value%type_in_use)
         CASE (result_type_real)
            CALL mp_bcast(results%result_value(i)%value%real_type, source, para_env%group)
         CASE (result_type_integer)
            CALL mp_bcast(results%result_value(i)%value%integer_type, source, para_env%group)
         CASE (result_type_logical)
            CALL mp_bcast(results%result_value(i)%value%logical_type, source, para_env%group)
         CASE DEFAULT
            DBCSR_ABORT("Type not implemented in dbcsr_result_type")
         END SELECT
      END DO
      DEALLOCATE (type_in_use)
      DEALLOCATE (size_value)
   END SUBROUTINE dbcsr_results_mp_bcast

END MODULE dbcsr_result_methods
