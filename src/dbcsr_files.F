!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_files
   !! Utility routines to open and close files.

   USE dbcsr_kinds, ONLY: default_path_length
   USE dbcsr_machine, ONLY: default_input_unit, &
                            default_output_unit, &
                            m_getcwd
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: close_file, &
             open_file, &
             get_unit_number, &
             file_exists

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_files'

   INTEGER, PARAMETER :: max_preconnections = 10, &
                         max_unit_number = 999

   TYPE preconnection_type
      PRIVATE
      CHARACTER(LEN=default_path_length) :: file_name
      INTEGER                            :: unit_number
   END TYPE preconnection_type

   TYPE(preconnection_type), DIMENSION(max_preconnections) :: preconnected

CONTAINS

   SUBROUTINE assign_preconnection(file_name, unit_number)
      !! Add an entry to the list of preconnected units

      CHARACTER(LEN=*), INTENT(IN)                       :: file_name
      INTEGER, INTENT(IN)                                :: unit_number

      CHARACTER(LEN=*), PARAMETER :: routineN = 'assign_preconnection', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: ic, islot, nc

      IF ((unit_number < 1) .OR. (unit_number > max_unit_number)) THEN
         DBCSR_ABORT("An invalid logical unit number was specified.")
      END IF

      IF (LEN_TRIM(file_name) == 0) THEN
         DBCSR_ABORT("No valid file name was specified")
      END IF

      nc = SIZE(preconnected)

      ! Check if a preconnection already exists
      DO ic = 1, nc
         IF (TRIM(preconnected(ic)%file_name) == TRIM(file_name)) THEN
            ! Return if the entry already exists
            IF (preconnected(ic)%unit_number == unit_number) THEN
               RETURN
            ELSE
               CALL print_preconnection_list()
               CALL dbcsr_abort(__LOCATION__, &
                                "Attempt to connect the already connected file <"// &
                                TRIM(ADJUSTL(file_name))//"> to another unit")
            END IF
         END IF
      END DO

      ! Search for an unused entry
      islot = -1
      DO ic = 1, nc
         IF (preconnected(ic)%unit_number == -1) THEN
            islot = ic
            EXIT
         END IF
      END DO

      IF (islot == -1) THEN
         CALL print_preconnection_list()
         DBCSR_ABORT("No free slot found in the list of preconnected units")
      END IF

      preconnected(islot)%file_name = TRIM(ADJUSTL(file_name))
      preconnected(islot)%unit_number = unit_number

   END SUBROUTINE assign_preconnection

   SUBROUTINE close_file(unit_number, file_status, keep_preconnection)
      !! Close an open file given by its logical unit number.
      !! Optionally, keep the file and unit preconnected.

      INTEGER, INTENT(IN)                                :: unit_number
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL             :: file_status
      LOGICAL, INTENT(IN), OPTIONAL                      :: keep_preconnection

      CHARACTER(LEN=*), PARAMETER :: routineN = 'close_file', routineP = moduleN//':'//routineN

      CHARACTER(LEN=2*default_path_length)               :: message
      CHARACTER(LEN=6)                                   :: status_string
      CHARACTER(LEN=default_path_length)                 :: file_name
      INTEGER                                            :: istat
      LOGICAL                                            :: exists, is_open, keep_file_connection

      keep_file_connection = .FALSE.
      IF (PRESENT(keep_preconnection)) keep_file_connection = keep_preconnection

      INQUIRE (UNIT=unit_number, EXIST=exists, OPENED=is_open, IOSTAT=istat)

      IF (istat /= 0) THEN
         WRITE (UNIT=message, FMT="(A,I0,A,I0,A)") &
            "An error occurred inquiring the unit with the number ", unit_number, &
            " (IOSTAT = ", istat, ")"
         DBCSR_ABORT(TRIM(message))
      ELSE IF (.NOT. exists) THEN
         WRITE (UNIT=message, FMT="(A,I0,A)") &
            "The specified unit number ", unit_number, &
            " cannot be closed, because it does not exist."
         DBCSR_ABORT(TRIM(message))
      END IF

      ! Close the specified file

      IF (is_open) THEN
         ! Refuse to close any preconnected system unit
         IF (unit_number == default_input_unit) THEN
            WRITE (UNIT=message, FMT="(A,I0)") &
               "Attempt to close the default input unit number ", unit_number
            DBCSR_ABORT(TRIM(message))
         END IF
         IF (unit_number == default_output_unit) THEN
            WRITE (UNIT=message, FMT="(A,I0)") &
               "Attempt to close the default output unit number ", unit_number
            DBCSR_ABORT(TRIM(message))
         END IF
         ! Define status after closing the file
         IF (PRESENT(file_status)) THEN
            status_string = TRIM(ADJUSTL(file_status))
         ELSE
            status_string = "KEEP"
         END IF
         ! Optionally, keep this unit preconnected
         INQUIRE (UNIT=unit_number, NAME=file_name, IOSTAT=istat)
         IF (istat /= 0) THEN
            WRITE (UNIT=message, FMT="(A,I0,A,I0,A)") &
               "An error occurred inquiring the unit with the number ", unit_number, &
               " (IOSTAT = ", istat, ")"
            DBCSR_ABORT(TRIM(message))
         END IF
         ! Manage preconnections
         IF (keep_file_connection) THEN
            CALL assign_preconnection(file_name, unit_number)
         ELSE
            CALL delete_preconnection(file_name, unit_number)
            CLOSE (UNIT=unit_number, IOSTAT=istat, STATUS=TRIM(status_string))
            IF (istat /= 0) THEN
               WRITE (UNIT=message, FMT="(A,I0,A,I0,A)") &
                  "An error occurred closing the file with the logical unit number ", &
                  unit_number, " (IOSTAT = ", istat, ")"
               DBCSR_ABORT(TRIM(message))
            END IF
         END IF
      END IF

   END SUBROUTINE close_file

   SUBROUTINE delete_preconnection(file_name, unit_number)
      !! Remove an entry from the list of preconnected units

      CHARACTER(LEN=*), INTENT(IN)                       :: file_name
      INTEGER                                            :: unit_number

      CHARACTER(LEN=*), PARAMETER :: routineN = 'delete_preconnection', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: ic, nc

      nc = SIZE(preconnected)

      ! Search for preconnection entry and delete it when found
      DO ic = 1, nc
         IF (TRIM(preconnected(ic)%file_name) == TRIM(file_name)) THEN
            IF (preconnected(ic)%unit_number == unit_number) THEN
               preconnected(ic)%file_name = ""
               preconnected(ic)%unit_number = -1
               EXIT
            ELSE
               CALL print_preconnection_list()
               CALL dbcsr_abort(__LOCATION__, &
                                "Attempt to disconnect the file <"// &
                                TRIM(ADJUSTL(file_name))// &
                                "> from an unlisted unit")
            END IF
         END IF
      END DO

   END SUBROUTINE delete_preconnection

   FUNCTION get_unit_number(file_name) RESULT(unit_number)
      !! Returns the first logical unit that is not preconnected
      !! @note
      !! -1 if no free unit exists

      CHARACTER(LEN=*), INTENT(IN), OPTIONAL             :: file_name
      INTEGER                                            :: unit_number

      INTEGER                                            :: ic, istat, nc
      LOGICAL                                            :: exists, is_open

      IF (PRESENT(file_name)) THEN
         nc = SIZE(preconnected)
         ! Check for preconnected units
         DO ic = 3, nc ! Exclude the preconnected system units (< 3)
            IF (TRIM(preconnected(ic)%file_name) == TRIM(file_name)) THEN
               unit_number = preconnected(ic)%unit_number
               RETURN
            END IF
         END DO
      END IF

      ! Get a new unit number
      DO unit_number = 1, max_unit_number
         IF (ANY(unit_number == preconnected(:)%unit_number)) CYCLE
         INQUIRE (UNIT=unit_number, EXIST=exists, OPENED=is_open, IOSTAT=istat)
         IF (exists .AND. (.NOT. is_open) .AND. (istat == 0)) RETURN
      END DO

      unit_number = -1

   END FUNCTION get_unit_number

   SUBROUTINE init_preconnection_list()
      !! Allocate and initialise the list of preconnected units

      CHARACTER(LEN=*), PARAMETER :: routineN = 'init_preconnection_list', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: ic, nc

      nc = SIZE(preconnected)

      DO ic = 1, nc
         preconnected(ic)%file_name = ""
         preconnected(ic)%unit_number = -1
      END DO

      ! Define reserved unit numbers
      preconnected(1)%file_name = "stdin"
      preconnected(1)%unit_number = default_input_unit
      preconnected(2)%file_name = "stdout"
      preconnected(2)%unit_number = default_output_unit

   END SUBROUTINE init_preconnection_list

   SUBROUTINE open_file(file_name, file_status, file_form, file_action, &
                        file_position, file_pad, unit_number, debug, &
                        skip_get_unit_number, file_access)
      !! Opens the requested file using a free unit number

      CHARACTER(LEN=*), INTENT(IN)                       :: file_name
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL             :: file_status, file_form, file_action, &
                                                            file_position, file_pad
      INTEGER, INTENT(INOUT)                             :: unit_number
      INTEGER, INTENT(IN), OPTIONAL                      :: debug
      LOGICAL, INTENT(IN), OPTIONAL                      :: skip_get_unit_number
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL             :: file_access
         !! file access mode

      CHARACTER(LEN=*), PARAMETER :: routineN = 'open_file', routineP = moduleN//':'//routineN

      CHARACTER(LEN=11) :: access_string, action_string, current_action, current_form, &
                           form_string, pad_string, position_string, status_string
      CHARACTER(LEN=2*default_path_length)               :: message
      CHARACTER(LEN=default_path_length)                 :: cwd, iomsgstr
      INTEGER                                            :: debug_unit, istat
      LOGICAL                                            :: exists, get_a_new_unit, is_open

      IF (PRESENT(file_access)) THEN
         access_string = TRIM(file_access)
      ELSE
         access_string = "SEQUENTIAL"
      END IF

      IF (PRESENT(file_status)) THEN
         status_string = TRIM(file_status)
      ELSE
         status_string = "OLD"
      END IF

      IF (PRESENT(file_form)) THEN
         form_string = TRIM(file_form)
      ELSE
         form_string = "FORMATTED"
      END IF

      IF (PRESENT(file_pad)) THEN
         pad_string = file_pad
         IF (form_string == "UNFORMATTED") THEN
            WRITE (UNIT=message, FMT="(A)") &
               "The PAD specifier is not allowed for an UNFORMATTED file"
            DBCSR_ABORT(TRIM(message))
         END IF
      ELSE
         pad_string = "YES"
      END IF

      IF (PRESENT(file_action)) THEN
         action_string = TRIM(file_action)
      ELSE
         action_string = "READ"
      END IF

      IF (PRESENT(file_position)) THEN
         position_string = TRIM(file_position)
      ELSE
         position_string = "REWIND"
      END IF

      IF (PRESENT(debug)) THEN
         debug_unit = debug
      ELSE
         debug_unit = 0 ! use default_output_unit for debugging
      END IF

      ! Check the specified input file name
      INQUIRE (FILE=TRIM(file_name), EXIST=exists, OPENED=is_open, IOSTAT=istat)

      IF (istat /= 0) THEN
         WRITE (UNIT=message, FMT="(A,I0,A)") &
            "An error occurred inquiring the file <"//TRIM(file_name)// &
            "> (IOSTAT = ", istat, ")"
         DBCSR_ABORT(TRIM(message))
      ELSE IF (status_string == "OLD") THEN
         IF (.NOT. exists) THEN
            WRITE (UNIT=message, FMT="(A)") &
               "The specified file <"//TRIM(ADJUSTL(file_name))// &
               "> cannot be opened. It does not exist. "
            DBCSR_ABORT(TRIM(message))
         END IF
      END IF

      ! Open the specified input file
      IF (is_open) THEN
         INQUIRE (FILE=TRIM(file_name), NUMBER=unit_number, &
                  ACTION=current_action, FORM=current_form)
         IF (TRIM(position_string) == "REWIND") REWIND (UNIT=unit_number)
         IF (TRIM(status_string) == "NEW") THEN
            CALL dbcsr_abort(__LOCATION__, &
                             "Attempt to re-open the existing OLD file <"// &
                             TRIM(file_name)//"> with status attribute NEW.")
         END IF
         IF (TRIM(current_form) /= TRIM(form_string)) THEN
            CALL dbcsr_abort(__LOCATION__, &
                             "Attempt to re-open the existing "// &
                             TRIM(current_form)//" file <"//TRIM(file_name)// &
                             "> as "//TRIM(form_string)//" file.")
         END IF
         IF (TRIM(current_action) /= TRIM(action_string)) THEN
            CALL dbcsr_abort(__LOCATION__, &
                             "Attempt to re-open the existing file <"// &
                             TRIM(file_name)//"> with the modified ACTION attribute "// &
                             TRIM(action_string)//". The current ACTION attribute is "// &
                             TRIM(current_action)//".")
         END IF
      ELSE
         ! Find an unused unit number
         get_a_new_unit = .TRUE.
         IF (PRESENT(skip_get_unit_number)) THEN
            IF (skip_get_unit_number) get_a_new_unit = .FALSE.
         END IF
         IF (get_a_new_unit) unit_number = get_unit_number(TRIM(file_name))
         IF (unit_number < 1) THEN
            WRITE (UNIT=message, FMT="(A)") &
               "Cannot open the file <"//TRIM(ADJUSTL(file_name))// &
               ">, because no unused logical unit number could be obtained."
            DBCSR_ABORT(TRIM(message))
         END IF
         IF (TRIM(form_string) == "FORMATTED") THEN
            OPEN (UNIT=unit_number, &
                  FILE=TRIM(file_name), &
                  STATUS=TRIM(status_string), &
                  ACCESS=TRIM(access_string), &
                  FORM=TRIM(form_string), &
                  POSITION=TRIM(position_string), &
                  ACTION=TRIM(action_string), &
                  PAD=TRIM(pad_string), &
                  IOMSG=iomsgstr, &
                  IOSTAT=istat)
         ELSE
            OPEN (UNIT=unit_number, &
                  FILE=TRIM(file_name), &
                  STATUS=TRIM(status_string), &
                  ACCESS=TRIM(access_string), &
                  FORM=TRIM(form_string), &
                  POSITION=TRIM(position_string), &
                  ACTION=TRIM(action_string), &
                  IOMSG=iomsgstr, &
                  IOSTAT=istat)
         END IF
         IF (istat /= 0) THEN
            CALL m_getcwd(cwd)
            WRITE (UNIT=message, FMT="(A,I0,A,I0,A)") &
               "An error occurred opening the file '"//TRIM(ADJUSTL(file_name))// &
               "' (UNIT = ", unit_number, ", IOSTAT = ", istat, "). "//TRIM(iomsgstr)//". "// &
               "Current working directory: "//TRIM(cwd)

            DBCSR_ABORT(TRIM(message))
         END IF
      END IF

      IF (debug_unit > 0) THEN
         INQUIRE (FILE=TRIM(file_name), OPENED=is_open, NUMBER=unit_number, &
                  POSITION=position_string, NAME=message, ACCESS=access_string, &
                  FORM=form_string, ACTION=action_string)
         WRITE (UNIT=debug_unit, FMT="(T2,A)") "BEGIN DEBUG "//TRIM(ADJUSTL(routineN))
         WRITE (UNIT=debug_unit, FMT="(T3,A,I0)") "NUMBER  : ", unit_number
         WRITE (UNIT=debug_unit, FMT="(T3,A,L1)") "OPENED  : ", is_open
         WRITE (UNIT=debug_unit, FMT="(T3,A)") "NAME    : "//TRIM(ADJUSTL(message))
         WRITE (UNIT=debug_unit, FMT="(T3,A)") "POSITION: "//TRIM(ADJUSTL(position_string))
         WRITE (UNIT=debug_unit, FMT="(T3,A)") "ACCESS  : "//TRIM(ADJUSTL(access_string))
         WRITE (UNIT=debug_unit, FMT="(T3,A)") "FORM    : "//TRIM(ADJUSTL(form_string))
         WRITE (UNIT=debug_unit, FMT="(T3,A)") "ACTION  : "//TRIM(ADJUSTL(action_string))
         WRITE (UNIT=debug_unit, FMT="(T2,A)") "END DEBUG "//TRIM(ADJUSTL(routineN))
         CALL print_preconnection_list(debug_unit)
      END IF

   END SUBROUTINE open_file

   FUNCTION file_exists(file_name) RESULT(exist)
      !! Checks if file exists, considering also the file discovery mechanism.
      CHARACTER(LEN=*), INTENT(IN)                       :: file_name
      LOGICAL                                            :: exist

      INQUIRE (FILE=TRIM(file_name), exist=exist)
   END FUNCTION file_exists

   SUBROUTINE print_preconnection_list(output_unit)
      !! Print the list of preconnected units

      INTEGER, INTENT(IN), OPTIONAL                      :: output_unit
         !! which unit to print to (optional)

      CHARACTER(LEN=*), PARAMETER :: routineN = 'print_preconnection_list', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: ic, nc, unit

      IF (PRESENT(output_unit)) THEN
         unit = output_unit
      ELSE
         unit = default_output_unit
      ENDIF

      nc = SIZE(preconnected)

      IF (output_unit > 0) THEN

         WRITE (UNIT=output_unit, FMT="(A,/,A)") &
            " LIST OF PRECONNECTED LOGICAL UNITS", &
            "  Slot   Unit number   File name"
         DO ic = 1, nc
            IF (preconnected(ic)%unit_number > 0) THEN
               WRITE (UNIT=output_unit, FMT="(I6,3X,I6,8X,A)") &
                  ic, preconnected(ic)%unit_number, &
                  TRIM(ADJUSTL(preconnected(ic)%file_name))
            ELSE
               WRITE (UNIT=output_unit, FMT="(I6,17X,A)") &
                  ic, "UNUSED"
            END IF
         END DO
      END IF
   END SUBROUTINE print_preconnection_list

END MODULE dbcsr_files
