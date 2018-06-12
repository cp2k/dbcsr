!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright (C) 2000 - 2018  CP2K developers group                                               !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Central dispatch for basic hooks
!> \author Ole Schuett
! **************************************************************************************************
MODULE dbcsr_base_hooks
   USE dbcsr_kinds,                     ONLY: default_string_length
   USE dbcsr_machine,                   ONLY: default_output_unit,&
                                              m_abort,&
                                              m_flush

   IMPLICIT NONE
   PRIVATE

   !API
   PUBLIC :: dbcsr_abort, dbcsr_warn, timeset, timestop
   !API
   PUBLIC :: dbcsr_abort_hook, dbcsr_warn_hook, timeset_hook, timestop_hook
   !API
   PUBLIC :: dbcsr__a, dbcsr__b, dbcsr__w, dbcsr__l

   ! this interface (with subroutines in it) musst to be defined right before
   ! the regular subroutines/functions - otherwise prettify.py will screw up.
   INTERFACE
      SUBROUTINE dbcsr_abort_interface(location, message)
      CHARACTER(len=*), INTENT(in)                       :: location, message

      END SUBROUTINE dbcsr_abort_interface

      SUBROUTINE dbcsr_warn_interface(location, message)
      CHARACTER(len=*), INTENT(in)                       :: location, message

      END SUBROUTINE dbcsr_warn_interface

      SUBROUTINE timeset_interface(routineN, handle)
      CHARACTER(LEN=*), INTENT(IN)                       :: routineN
      INTEGER, INTENT(OUT)                               :: handle

      END SUBROUTINE timeset_interface

      SUBROUTINE timestop_interface(handle)
      INTEGER, INTENT(IN)                                :: handle

      END SUBROUTINE timestop_interface
   END INTERFACE

   PROCEDURE(dbcsr_abort_interface), POINTER :: dbcsr_abort_hook => Null()
   PROCEDURE(dbcsr_warn_interface), POINTER :: dbcsr_warn_hook => Null()
   PROCEDURE(timeset_interface), POINTER :: timeset_hook => Null()
   PROCEDURE(timestop_interface), POINTER :: timestop_hook => Null()

CONTAINS

! **************************************************************************************************
!> \brief Terminate the program
!> \param location ...
!> \param message ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE dbcsr_abort(location, message)
      CHARACTER(len=*), INTENT(in)                       :: location, message

      IF (ASSOCIATED(dbcsr_abort_hook)) THEN
         CALL dbcsr_abort_hook(location, message)
      ELSE
         WRITE (default_output_unit, *) "ABORT in "//TRIM(location)//" "//TRIM(message)
         CALL m_flush(default_output_unit)
         CALL m_abort()
      ENDIF
      ! compiler hint
      STOP "Never return from here"
   END SUBROUTINE dbcsr_abort

! **************************************************************************************************
!> \brief Issue a warning
!> \param location ...
!> \param message ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE dbcsr_warn(location, message)
      CHARACTER(len=*), INTENT(in)                       :: location, message

      IF (ASSOCIATED(dbcsr_warn_hook)) THEN
         CALL dbcsr_warn_hook(location, message)
      ELSE
         WRITE (default_output_unit, *) "WARNING in "//TRIM(location)//" "//TRIM(message)
         CALL m_flush(default_output_unit)
      ENDIF
   END SUBROUTINE dbcsr_warn

! **************************************************************************************************
!> \brief Start timer
!> \param routineN ...
!> \param handle ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE timeset(routineN, handle)
      CHARACTER(LEN=*), INTENT(IN)                       :: routineN
      INTEGER, INTENT(OUT)                               :: handle

      IF (ASSOCIATED(timeset_hook)) THEN
         CALL timeset_hook(routineN, handle)
      ELSE
         handle = -1
      ENDIF
   END SUBROUTINE timeset

! **************************************************************************************************
!> \brief Stop timer
!> \param handle ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE timestop(handle)
      INTEGER, INTENT(IN)                                :: handle

      IF (ASSOCIATED(timestop_hook)) THEN
         CALL timestop_hook(handle)
      ELSE
         IF (handle /= -1) &
            CALL dbcsr_abort(dbcsr__l("base_hooks.F", __LINE__), "Got wrong handle")
      ENDIF
   END SUBROUTINE timestop

! **************************************************************************************************
!> \brief DBCSR_ASSERT handler
!> \param filename ...
!> \param lineNr ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE dbcsr__a(filename, lineNr)
      CHARACTER(len=*), INTENT(in)                       :: filename
      INTEGER, INTENT(in)                                :: lineNr

      CALL dbcsr_abort(location=dbcsr__l(filename, lineNr), message="DBCSR_ASSERT failed")
      ! compiler hint
      STOP "Never return from here"
   END SUBROUTINE dbcsr__a

! **************************************************************************************************
!> \brief DBCSR_ABORT handler
!> \param filename ...
!> \param lineNr ...
!> \param message ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE dbcsr__b(filename, lineNr, message)
      CHARACTER(len=*), INTENT(in)                       :: filename
      INTEGER, INTENT(in)                                :: lineNr
      CHARACTER(len=*), INTENT(in)                       :: message

      CALL dbcsr_abort(location=dbcsr__l(filename, lineNr), message=message)
      ! compiler hint
      STOP "Never return from here"
   END SUBROUTINE dbcsr__b

! **************************************************************************************************
!> \brief DBCSR_WARN handler
!> \param filename ...
!> \param lineNr ...
!> \param message ...
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE dbcsr__w(filename, lineNr, message)
      CHARACTER(len=*), INTENT(in)                       :: filename
      INTEGER, INTENT(in)                                :: lineNr
      CHARACTER(len=*), INTENT(in)                       :: message

      CALL dbcsr_warn(location=dbcsr__l(filename, lineNr), message=message)
   END SUBROUTINE dbcsr__w

! **************************************************************************************************
!> \brief Helper routine to assemble __LOCATION__
!> \param filename ...
!> \param lineNr ...
!> \return ...
!> \author Ole Schuett
! **************************************************************************************************
   FUNCTION dbcsr__l(filename, lineNr) RESULT(location)
      CHARACTER(len=*), INTENT(in)                       :: filename
      INTEGER, INTENT(in)                                :: lineNr
      CHARACTER(len=default_string_length)               :: location

      CHARACTER(len=15)                                  :: lineNr_str

      WRITE (lineNr_str, FMT='(I10)') lineNr
      location = TRIM(filename)//":"//TRIM(ADJUSTL(lineNr_str))

   END FUNCTION dbcsr__l

END MODULE dbcsr_base_hooks
