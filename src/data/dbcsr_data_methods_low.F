!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_data_methods_low
   !! DBCSR data methods

   USE dbcsr_acc_devmem, ONLY: acc_devmem_allocate_bytes, &
                               acc_devmem_allocated, &
                               acc_devmem_deallocate, &
                               acc_devmem_set_cptr
   USE dbcsr_acc_event, ONLY: acc_event_create, &
                              acc_event_destroy
   USE dbcsr_data_types, ONLY: &
      dbcsr_data_area_type, dbcsr_data_obj, dbcsr_datatype_sizeof, dbcsr_memtype_type, &
      dbcsr_scalar_type, dbcsr_type_complex_4, dbcsr_type_complex_4_2d, dbcsr_type_complex_8, &
      dbcsr_type_complex_8_2d, dbcsr_type_int_4, dbcsr_type_int_8, dbcsr_type_real_4, &
      dbcsr_type_real_4_2d, dbcsr_type_real_8, dbcsr_type_real_8_2d
   USE dbcsr_ptr_util, ONLY: memory_allocate, &
                             memory_deallocate, &
                             memory_zero, &
                             pointer_rank_remap2
   USE dbcsr_kinds, ONLY: dp, &
                          real_4, &
                          real_8
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_data_methods_low'

   PUBLIC :: dbcsr_type_is_2d, dbcsr_type_2d_to_1d, dbcsr_type_1d_to_2d
   PUBLIC :: dbcsr_scalar, dbcsr_scalar_one, dbcsr_scalar_zero, &
             dbcsr_scalar_are_equal, dbcsr_scalar_negative, &
             dbcsr_scalar_get_type, dbcsr_scalar_set_type, &
             dbcsr_scalar_fill_all, dbcsr_scalar_get_value, &
             dbcsr_data_valid
   PUBLIC :: dbcsr_data_init, dbcsr_data_hold, &
             dbcsr_data_get_size, dbcsr_data_get_type
   PUBLIC :: dbcsr_get_data, &
             dbcsr_data_set_pointer, &
             dbcsr_data_clear_pointer, &
             dbcsr_data_get_sizes, dbcsr_data_verify_bounds, &
             dbcsr_data_exists, dbcsr_data_get_memory_type
   PUBLIC :: dbcsr_data_set_size_referenced, dbcsr_data_get_size_referenced
   PUBLIC :: dbcsr_get_data_p, dbcsr_get_data_p_s, dbcsr_get_data_p_c, &
             dbcsr_get_data_p_d, dbcsr_get_data_p_z, &
             dbcsr_get_data_p_2d_s, dbcsr_get_data_p_2d_d, &
             dbcsr_get_data_p_2d_c, dbcsr_get_data_p_2d_z
   PUBLIC :: dbcsr_data_zero

   PUBLIC :: internal_data_allocate, internal_data_deallocate

   INTERFACE dbcsr_scalar
      !! Encapsulates a scalar.
      MODULE PROCEDURE dbcsr_scalar_s, dbcsr_scalar_d, &
         dbcsr_scalar_c, dbcsr_scalar_z
   END INTERFACE

   INTERFACE dbcsr_scalar_get_value
      !! Encapsulates a scalar.
      MODULE PROCEDURE dbcsr_scalar_get_value_s, dbcsr_scalar_get_value_d, &
         dbcsr_scalar_get_value_c, dbcsr_scalar_get_value_z
   END INTERFACE

   INTERFACE dbcsr_data_set_pointer
      MODULE PROCEDURE set_data_p_s, set_data_p_d, set_data_p_c, set_data_p_z
      MODULE PROCEDURE set_data_p_2d_s, set_data_p_2d_d, &
         set_data_p_2d_c, set_data_p_2d_z
      MODULE PROCEDURE set_data_area_area
   END INTERFACE

   INTERFACE dbcsr_get_data
      MODULE PROCEDURE get_data_s, get_data_d, get_data_c, get_data_z
      MODULE PROCEDURE get_data_2d_s, get_data_2d_d, get_data_2d_c, get_data_2d_z
   END INTERFACE

   INTERFACE dbcsr_get_data_p
      MODULE PROCEDURE dbcsr_get_data_c_s, dbcsr_get_data_c_c, &
         dbcsr_get_data_c_d, dbcsr_get_data_c_z
   END INTERFACE

   INTERFACE dbcsr_get_data_cptr
      MODULE PROCEDURE dbcsr_get_data_c_s, dbcsr_get_data_c_c, &
         dbcsr_get_data_c_d, dbcsr_get_data_c_z
   END INTERFACE

   INTERFACE dbcsr_data_get_sizes
      MODULE PROCEDURE dbcsr_data_get_sizes_any
      MODULE PROCEDURE dbcsr_data_get_sizes_1, dbcsr_data_get_sizes_2
   END INTERFACE

   LOGICAL, PARAMETER :: careful_mod = .FALSE.
   LOGICAL, PARAMETER :: debug_mod = .FALSE.

CONTAINS

   PURE FUNCTION dbcsr_data_get_type(area) RESULT(data_type)
      !! Returns data type of a data area

      TYPE(dbcsr_data_obj), INTENT(IN)                   :: area
         !! data area
      INTEGER                                            :: data_type
         !! data type of the data area

      data_type = area%d%data_type
   END FUNCTION dbcsr_data_get_type

   PURE FUNCTION dbcsr_data_unique(area) RESULT(unique)
      !! Returns if there is a single reference of the data

      TYPE(dbcsr_data_obj), INTENT(IN)                   :: area
         !! data area
      LOGICAL                                            :: unique
         !! true if reference is 1

      unique = area%d%refcount .EQ. 1
   END FUNCTION dbcsr_data_unique

   FUNCTION dbcsr_data_get_memory_type(area) RESULT(memory_type)
      TYPE(dbcsr_data_obj), INTENT(IN)                   :: area
      TYPE(dbcsr_memtype_type)                           :: memory_type

      memory_type = area%d%memory_type
   END FUNCTION dbcsr_data_get_memory_type

   SUBROUTINE dbcsr_data_init(area)
      !! Initializes a data area

      TYPE(dbcsr_data_obj), INTENT(INOUT)                :: area
         !! data area

      NULLIFY (area%d)
   END SUBROUTINE dbcsr_data_init

   SUBROUTINE internal_data_allocate(area, sizes)
      !! Allocates pointers in the data type

      TYPE(dbcsr_data_area_type), INTENT(INOUT)          :: area
         !! internal structure holding array pointers
      INTEGER, DIMENSION(:), INTENT(IN)                  :: sizes
         !! sizes to allocate to

      CHARACTER(len=*), PARAMETER :: routineN = 'internal_data_allocate', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: error_handle

!   ---------------------------------------------------------------------------

      IF (careful_mod) &
         CALL timeset(routineN, error_handle)
      IF (debug_mod) &
         WRITE (*, *) routineN//" Setting to sizes", sizes
      IF (dbcsr_type_is_2d(area%data_type)) THEN
         IF (SIZE(sizes) /= 2) &
            DBCSR_ABORT("Sizes must have 2 elements for 2-D data")
      ELSE
         IF (SIZE(sizes) /= 1) &
            DBCSR_ABORT("Sizes must have 1 elements for 1-D data")
      ENDIF

      SELECT CASE (area%data_type)
      CASE (dbcsr_type_int_4)
         CALL memory_allocate(area%i4, n=sizes(1), mem_type=area%memory_type)
      CASE (dbcsr_type_int_8)
         CALL memory_allocate(area%i8, n=sizes(1), mem_type=area%memory_type)
      CASE (dbcsr_type_real_4)
         CALL memory_allocate(area%r_sp, n=sizes(1), mem_type=area%memory_type)
      CASE (dbcsr_type_real_8)
         CALL memory_allocate(area%r_dp, n=sizes(1), mem_type=area%memory_type)
      CASE (dbcsr_type_complex_4)
         CALL memory_allocate(area%c_sp, n=sizes(1), mem_type=area%memory_type)
      CASE (dbcsr_type_complex_8)
         CALL memory_allocate(area%c_dp, n=sizes(1), mem_type=area%memory_type)

      CASE (dbcsr_type_real_4_2d)
         CALL memory_allocate(area%r2_sp, sizes=sizes, mem_type=area%memory_type)
      CASE (dbcsr_type_real_8_2d)
         CALL memory_allocate(area%r2_dp, sizes=sizes, mem_type=area%memory_type)
      CASE (dbcsr_type_complex_4_2d)
         CALL memory_allocate(area%c2_sp, sizes=sizes, mem_type=area%memory_type)
      CASE (dbcsr_type_complex_8_2d)
         CALL memory_allocate(area%c2_dp, sizes=sizes, mem_type=area%memory_type)

      CASE default
         DBCSR_ABORT("Invalid data type.")
      END SELECT

      IF (area%memory_type%acc_devalloc) THEN
         IF (sizes(1) >= 0) &
            CALL acc_devmem_allocate_bytes(area%acc_devmem, dbcsr_datatype_sizeof(area%data_type)*sizes(1))
         CALL acc_event_create(area%acc_ready)
      ENDIF

      IF (careful_mod) &
         CALL timestop(error_handle)

   END SUBROUTINE internal_data_allocate

   SUBROUTINE internal_data_deallocate(area)
      !! Allocates pointers in the data type

      TYPE(dbcsr_data_area_type), INTENT(INOUT)          :: area
         !! internal structure holding array pointers

      CHARACTER(len=*), PARAMETER :: routineN = 'internal_data_deallocate', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: handle

!   ---------------------------------------------------------------------------

      IF (careful_mod) &
         CALL timeset(routineN, handle)

      SELECT CASE (area%data_type)
      CASE (dbcsr_type_int_4)
         CALL memory_deallocate(area%i4, mem_type=area%memory_type)
         NULLIFY (area%i4)
      CASE (dbcsr_type_int_8)
         CALL memory_deallocate(area%i8, mem_type=area%memory_type)
         NULLIFY (area%i8)
      CASE (dbcsr_type_real_4)
         CALL memory_deallocate(area%r_sp, mem_type=area%memory_type)
         NULLIFY (area%r_sp)
      CASE (dbcsr_type_real_8)
         CALL memory_deallocate(area%r_dp, mem_type=area%memory_type)
         NULLIFY (area%r_dp)
      CASE (dbcsr_type_complex_4)
         CALL memory_deallocate(area%c_sp, mem_type=area%memory_type)
         NULLIFY (area%c_sp)
      CASE (dbcsr_type_complex_8)
         CALL memory_deallocate(area%c_dp, mem_type=area%memory_type)
         NULLIFY (area%c_dp)

      CASE (dbcsr_type_real_4_2d)
         CALL memory_deallocate(area%r2_sp, mem_type=area%memory_type)
         NULLIFY (area%r2_sp)
      CASE (dbcsr_type_real_8_2d)
         CALL memory_deallocate(area%r2_dp, mem_type=area%memory_type)
         NULLIFY (area%r2_dp)
      CASE (dbcsr_type_complex_4_2d)
         CALL memory_deallocate(area%c2_sp, mem_type=area%memory_type)
         NULLIFY (area%c2_sp)
      CASE (dbcsr_type_complex_8_2d)
         CALL memory_deallocate(area%c2_dp, mem_type=area%memory_type)
         NULLIFY (area%c2_dp)

      CASE default
         DBCSR_ABORT("Invalid data type.")
      END SELECT

      IF (area%memory_type%acc_devalloc) THEN
         IF (acc_devmem_allocated(area%acc_devmem)) &
            CALL acc_devmem_deallocate(area%acc_devmem)
         CALL acc_event_destroy(area%acc_ready)
      ENDIF

      IF (careful_mod) &
         CALL timestop(handle)
   END SUBROUTINE internal_data_deallocate

   SUBROUTINE dbcsr_data_clear_pointer(area)
      !! Clears pointers from the data area.

      TYPE(dbcsr_data_obj), INTENT(INOUT)                :: area
         !! data area

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_data_clear_pointer', &
                                     routineP = moduleN//':'//routineN

!   ---------------------------------------------------------------------------

      IF (.NOT. ASSOCIATED(area%d)) THEN
         RETURN
      ENDIF

      IF (area%d%refcount .LE. 0) &
         DBCSR_WARN("Data seems to be unreferenced.")
      SELECT CASE (area%d%data_type)
      CASE (dbcsr_type_int_4)
         NULLIFY (area%d%i4)
      CASE (dbcsr_type_int_8)
         NULLIFY (area%d%i8)
      CASE (dbcsr_type_real_4)
         NULLIFY (area%d%r_sp)
      CASE (dbcsr_type_real_8)
         NULLIFY (area%d%r_dp)
      CASE (dbcsr_type_complex_4)
         NULLIFY (area%d%c_sp)
      CASE (dbcsr_type_complex_8)
         NULLIFY (area%d%c_dp)
      CASE (dbcsr_type_real_8_2d)
         NULLIFY (area%d%r2_dp)
      CASE (dbcsr_type_real_4_2d)
         NULLIFY (area%d%r2_sp)
      CASE (dbcsr_type_complex_8_2d)
         NULLIFY (area%d%c2_dp)
      CASE (dbcsr_type_complex_4_2d)
         NULLIFY (area%d%c2_sp)
      CASE default
         DBCSR_ABORT("Invalid data type.")
      END SELECT
   END SUBROUTINE dbcsr_data_clear_pointer

   ELEMENTAL FUNCTION dbcsr_data_valid(area) RESULT(valid)
      !! Checks whether a data area is valid

      TYPE(dbcsr_data_obj), INTENT(IN)                   :: area
         !! data area
      LOGICAL                                            :: valid
         !! whether the data area is valid

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_data_valid', &
                                     routineP = moduleN//':'//routineN

!   ---------------------------------------------------------------------------

      valid = ASSOCIATED(area%d)
   END FUNCTION dbcsr_data_valid

   FUNCTION dbcsr_data_exists(area) RESULT(valid)
      !! Checks whether a data pointer exists

      TYPE(dbcsr_data_obj), INTENT(IN)                   :: area
         !! data area
      LOGICAL                                            :: valid
         !! whether the data pointer exists

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_data_exists', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: error_handle

!   ---------------------------------------------------------------------------

      IF (careful_mod) THEN
         CALL timeset(routineN, error_handle)
      ENDIF
      !
      valid = dbcsr_data_valid(area)
      IF (.NOT. valid) &
         DBCSR_ABORT("Data area is invalid.")

      SELECT CASE (area%d%data_type)
      CASE (dbcsr_type_int_4)
         valid = ASSOCIATED(area%d%i4)
      CASE (dbcsr_type_int_8)
         valid = ASSOCIATED(area%d%i8)
      CASE (dbcsr_type_real_4)
         valid = ASSOCIATED(area%d%r_sp)
      CASE (dbcsr_type_real_8)
         valid = ASSOCIATED(area%d%r_dp)
      CASE (dbcsr_type_complex_4)
         valid = ASSOCIATED(area%d%c_sp)
      CASE (dbcsr_type_complex_8)
         valid = ASSOCIATED(area%d%c_dp)
      CASE (dbcsr_type_real_4_2d)
         valid = ASSOCIATED(area%d%r2_sp)
      CASE (dbcsr_type_real_8_2d)
         valid = ASSOCIATED(area%d%r2_dp)
      CASE (dbcsr_type_complex_4_2d)
         valid = ASSOCIATED(area%d%c2_sp)
      CASE (dbcsr_type_complex_8_2d)
         valid = ASSOCIATED(area%d%c2_dp)
      CASE default
         DBCSR_ABORT("Invalid data type.")
      END SELECT

      IF (careful_mod) THEN
         CALL timestop(error_handle)
      ENDIF
   END FUNCTION dbcsr_data_exists

   SUBROUTINE dbcsr_data_hold(area)
      !! Registers another use of the data area

      TYPE(dbcsr_data_obj), INTENT(INOUT)                :: area
         !! data area

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_data_hold', &
                                     routineP = moduleN//':'//routineN

!   ---------------------------------------------------------------------------

      IF (careful_mod) THEN
         IF (.NOT. ASSOCIATED(area%d)) &
            DBCSR_ABORT("Can't hold an empty data area.")
         IF (area%d%refcount <= 0) &
            DBCSR_ABORT("Should not hold an area with zero references.")
      ENDIF
      IF (.NOT. ASSOCIATED(area%d)) THEN
         RETURN
      ENDIF
!$OMP ATOMIC
      area%d%refcount = area%d%refcount + 1
   END SUBROUTINE dbcsr_data_hold

   SUBROUTINE set_data_area_area(area, rsize, csize, pointee, source_lb)
      !! Points data area data pointers to another data area
      !! Assumes that no memory will be lost when repointing the pointer in the data
      !! area and that the area is initialized.

      TYPE(dbcsr_data_obj), INTENT(INOUT)                :: area
         !! data area to repoint
      INTEGER, INTENT(IN)                                :: rsize, csize
         !! size of data area to point to
         !! size of data area to point to
      TYPE(dbcsr_data_obj), INTENT(IN)                   :: pointee
         !! data area to point to
      INTEGER, INTENT(IN), OPTIONAL                      :: source_lb
         !! point to this offset in pointee

      CHARACTER(len=*), PARAMETER :: routineN = 'set_data_area_area', &
                                     routineP = moduleN//':'//routineN

      COMPLEX(KIND=real_4), DIMENSION(:), POINTER        :: c_sp
      COMPLEX(KIND=real_8), DIMENSION(:), POINTER        :: c_dp
      INTEGER                                            :: bp, dt1, dt2, nze
      LOGICAL                                            :: compatible, pointee_is_2d, rmp
      REAL(KIND=real_4), DIMENSION(:), POINTER           :: r_sp
      REAL(KIND=real_8), DIMENSION(:), POINTER           :: r_dp

!   ---------------------------------------------------------------------------

      bp = 1; IF (PRESENT(source_lb)) bp = source_lb
      nze = rsize*csize
      dt1 = area%d%data_type
      dt2 = pointee%d%data_type
      IF (careful_mod) THEN
         compatible = dt1 .EQ. dt2 .OR. dt1 .EQ. dbcsr_type_1d_to_2d(dt2)
         IF (.NOT. compatible) &
            DBCSR_ABORT("Can not point 1-d pointer to 2-d data")

      ENDIF
      pointee_is_2d = dbcsr_type_is_2d(dt2)
      IF (careful_mod) THEN
         IF (PRESENT(source_lb) .AND. pointee_is_2d) &
            DBCSR_ABORT("Lower bound specification not possible with 2-d data")
         ! Check if size is OK.
         IF (bp < 1) &
            DBCSR_ABORT("Attempt to point out of bounds")
         IF (bp + nze - 1 > dbcsr_data_get_size(pointee)) &
            DBCSR_ABORT("Attempt to point out of bounds")
      ENDIF
      ! There's a remap if the ranks are compatible but not equal.
      rmp = dt1 .NE. dt2
      SELECT CASE (dt2)
      CASE (dbcsr_type_int_4)
         area%d%i4 => pointee%d%i4(bp:bp + nze - 1)
      CASE (dbcsr_type_real_4_2d)
         area%d%r2_sp => pointee%d%r2_sp(1:rsize, 1:csize)
      CASE (dbcsr_type_real_4)
         IF (rmp) THEN
            r_sp => dbcsr_get_data_p_s(pointee, bp, bp + nze - 1)
            CALL pointer_rank_remap2(area%d%r2_sp, rsize, csize, &
                                     r_sp)
         ELSE
            area%d%r_sp => dbcsr_get_data_p_s(pointee, bp, bp + nze - 1)
         ENDIF
      CASE (dbcsr_type_real_8_2d)
         area%d%r2_dp => pointee%d%r2_dp(1:rsize, 1:csize)
      CASE (dbcsr_type_real_8)
         IF (rmp) THEN
            r_dp => dbcsr_get_data_p_d(pointee, bp, bp + nze - 1)
            CALL pointer_rank_remap2(area%d%r2_dp, rsize, csize, &
                                     r_dp)
         ELSE
            area%d%r_dp => dbcsr_get_data_p_d(pointee, bp, bp + nze - 1)
         ENDIF
      CASE (dbcsr_type_complex_4_2d)
         area%d%c2_sp => pointee%d%c2_sp(1:rsize, 1:csize)
      CASE (dbcsr_type_complex_4)
         IF (rmp) THEN
            c_sp => dbcsr_get_data_p_c(pointee, bp, bp + nze - 1)
            CALL pointer_rank_remap2(area%d%c2_sp, rsize, csize, &
                                     c_sp)
         ELSE
            area%d%c_sp => dbcsr_get_data_p_c(pointee, bp, bp + nze - 1)
         ENDIF
      CASE (dbcsr_type_complex_8_2d)
         area%d%c2_dp => pointee%d%c2_dp(1:rsize, 1:csize)
      CASE (dbcsr_type_complex_8)
         IF (rmp) THEN
            c_dp => dbcsr_get_data_p_z(pointee, bp, bp + nze - 1)
            CALL pointer_rank_remap2(area%d%c2_dp, rsize, csize, &
                                     c_dp)
         ELSE
            area%d%c_dp => dbcsr_get_data_p_z(pointee, bp, bp + nze - 1)
         ENDIF
      CASE default
         DBCSR_ABORT("Invalid data type")
      END SELECT
      CALL dbcsr_data_set_size_referenced(area, rsize*csize)
      IF (debug_mod) THEN
         IF (dbcsr_data_get_size_referenced(area) /= dbcsr_data_get_size(area)) &
            DBCSR_ABORT("Size mismatch")
      ENDIF
      !
      IF (area%d%memory_type%acc_devalloc .AND. pointee%d%memory_type%acc_devalloc) THEN
         IF (pointee_is_2d) &
            DBCSR_ABORT("Setting GPU pointers for 2D data is not available!")
         CALL acc_devmem_set_cptr(area%d%acc_devmem, &
                                  pointee%d%acc_devmem, &
                                  dbcsr_datatype_sizeof(area%d%data_type)*nze, &
                                  dbcsr_datatype_sizeof(area%d%data_type)*(bp - 1))
      ENDIF
   END SUBROUTINE set_data_area_area

   FUNCTION dbcsr_data_get_size(area) RESULT(data_size)
      !! Returns the allocated data size

      TYPE(dbcsr_data_obj), INTENT(IN)                   :: area
         !! data area
      INTEGER                                            :: data_size
         !! size of data

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_data_get_size', &
                                     routineP = moduleN//':'//routineN

      data_size = 0
      IF (ASSOCIATED(area%d)) THEN
         SELECT CASE (area%d%data_type)
         CASE (dbcsr_type_int_4)
            IF (ASSOCIATED(area%d%i4)) &
               data_size = SIZE(area%d%i4)
         CASE (dbcsr_type_int_8)
            IF (ASSOCIATED(area%d%i8)) &
               data_size = SIZE(area%d%i8)
         CASE (dbcsr_type_real_8)
            IF (ASSOCIATED(area%d%r_dp)) &
               data_size = SIZE(area%d%r_dp)
         CASE (dbcsr_type_real_4)
            IF (ASSOCIATED(area%d%r_sp)) &
               data_size = SIZE(area%d%r_sp)
         CASE (dbcsr_type_complex_8)
            IF (ASSOCIATED(area%d%c_dp)) &
               data_size = SIZE(area%d%c_dp)
         CASE (dbcsr_type_complex_4)
            IF (ASSOCIATED(area%d%c_sp)) &
               data_size = SIZE(area%d%c_sp)
         CASE (dbcsr_type_real_8_2d)
            IF (ASSOCIATED(area%d%r2_dp)) &
               data_size = SIZE(area%d%r2_dp)
         CASE (dbcsr_type_real_4_2d)
            IF (ASSOCIATED(area%d%r2_sp)) &
               data_size = SIZE(area%d%r2_sp)
         CASE (dbcsr_type_complex_8_2d)
            IF (ASSOCIATED(area%d%c2_dp)) &
               data_size = SIZE(area%d%c2_dp)
         CASE (dbcsr_type_complex_4_2d)
            IF (ASSOCIATED(area%d%c2_sp)) &
               data_size = SIZE(area%d%c2_sp)
         CASE default
            DBCSR_ABORT("Incorrect data type")
         END SELECT
      ELSE
         DBCSR_WARN("Uninitialized data area")
         data_size = 0
      ENDIF
   END FUNCTION dbcsr_data_get_size

   SUBROUTINE dbcsr_data_verify_bounds(area, lb, ub)
      !! Verifies bounds of a data area

      TYPE(dbcsr_data_obj), INTENT(IN)                   :: area
         !! Data area
      INTEGER, DIMENSION(:), INTENT(IN)                  :: lb, ub
         !! lower bounds
         !! upper bounds

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_data_verify_bounds', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: data_type, handle

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)
      data_type = dbcsr_data_get_type(area)
      IF (dbcsr_type_is_2d(data_type)) THEN
         IF (SIZE(lb) /= 2) &
            DBCSR_ABORT("size must be 2 for 2-d lb")
         IF (SIZE(ub) /= 2) &
            DBCSR_ABORT("size must be 2 for 2-d ub")
      ELSE
         IF (SIZE(lb) /= 1) &
            DBCSR_ABORT("size must be 1 for 1-d lb")
         IF (SIZE(ub) /= 1) &
            DBCSR_ABORT("size must be 1 for 1-d ub")
      ENDIF
      SELECT CASE (data_type)
      CASE (dbcsr_type_real_4)
         IF (lb(1) < LBOUND(area%d%r_sp, 1)) DBCSR_ABORT("lb r_sp")
         IF (ub(1) > UBOUND(area%d%r_sp, 1)) DBCSR_ABORT("ub r_sp")
      CASE (dbcsr_type_real_4_2d)
         IF (lb(1) < LBOUND(area%d%r2_sp, 1)) DBCSR_ABORT("lb r_sp 2d")
         IF (ub(1) > UBOUND(area%d%r2_sp, 1)) DBCSR_ABORT("ub r_sp 2d")
         IF (lb(2) < LBOUND(area%d%r2_sp, 2)) DBCSR_ABORT("lb r_sp 2d")
         IF (ub(2) > UBOUND(area%d%r2_sp, 2)) DBCSR_ABORT("ub r_sp 2d")
      CASE (dbcsr_type_real_8)
         IF (lb(1) < LBOUND(area%d%r_dp, 1)) DBCSR_ABORT("lb r_dp")
         IF (ub(1) > UBOUND(area%d%r_dp, 1)) DBCSR_ABORT("ub r_dp")
      CASE (dbcsr_type_real_8_2d)
         IF (lb(1) < LBOUND(area%d%r2_dp, 1)) DBCSR_ABORT("lb r_dp 2d")
         IF (ub(1) > UBOUND(area%d%r2_dp, 1)) DBCSR_ABORT("ub r_dp 2d")
         IF (lb(2) < LBOUND(area%d%r2_dp, 2)) DBCSR_ABORT("lb r_dp 2d")
         IF (ub(2) > UBOUND(area%d%r2_dp, 2)) DBCSR_ABORT("ub r_dp 2d")
      CASE (dbcsr_type_complex_4)
         IF (lb(1) < LBOUND(area%d%c_sp, 1)) DBCSR_ABORT("lb c_sp")
         IF (ub(1) > UBOUND(area%d%c_sp, 1)) DBCSR_ABORT("ub c_sp")
      CASE (dbcsr_type_complex_4_2d)
         IF (lb(1) < LBOUND(area%d%c2_sp, 1)) DBCSR_ABORT("lb c_sp 2d")
         IF (ub(1) > UBOUND(area%d%c2_sp, 1)) DBCSR_ABORT("ub c_sp 2d")
         IF (lb(2) < LBOUND(area%d%c2_sp, 2)) DBCSR_ABORT("lb c_sp 2d")
         IF (ub(2) > UBOUND(area%d%c2_sp, 2)) DBCSR_ABORT("ub c_sp 2d")
      CASE (dbcsr_type_complex_8)
         IF (lb(1) < LBOUND(area%d%c_dp, 1)) DBCSR_ABORT("lb c_dp")
         IF (ub(1) > UBOUND(area%d%c_dp, 1)) DBCSR_ABORT("ub c_dp")
      CASE (dbcsr_type_complex_8_2d)
         IF (lb(1) < LBOUND(area%d%c2_dp, 1)) DBCSR_ABORT("lb c_dp 2d")
         IF (ub(1) > UBOUND(area%d%c2_dp, 1)) DBCSR_ABORT("ub c_dp 2d")
         IF (lb(2) < LBOUND(area%d%c2_dp, 2)) DBCSR_ABORT("lb c_dp 2d")
         IF (ub(2) > UBOUND(area%d%c2_dp, 2)) DBCSR_ABORT("ub c_dp 2d")
      CASE default
         DBCSR_ABORT("Invalid data type")
      END SELECT
      CALL timestop(handle)
   END SUBROUTINE dbcsr_data_verify_bounds

   SUBROUTINE dbcsr_data_zero(area, lb, ub)
      !! Clears a part of the data area
      !! @note Optimized for clearing big 1-D data areas from all data types.

      TYPE(dbcsr_data_obj), INTENT(INOUT)                :: area
         !! data area
      INTEGER, DIMENSION(:), INTENT(in)                  :: lb, ub

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_data_zero', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: error_handle

!   ---------------------------------------------------------------------------

      IF (careful_mod) THEN
         CALL timeset(routineN, error_handle)
      ENDIF

      SELECT CASE (area%d%data_type)
      CASE (dbcsr_type_real_4)
         CALL memory_zero(area%d%r_sp(lb(1):), ub(1))
      CASE (dbcsr_type_real_8)
         CALL memory_zero(area%d%r_dp(lb(1):), ub(1))
      CASE (dbcsr_type_complex_4)
         CALL memory_zero(area%d%c_sp(lb(1):), ub(1))
      CASE (dbcsr_type_complex_8)
         CALL memory_zero(area%d%c_dp(lb(1):), ub(1))
      CASE (dbcsr_type_real_4_2d)
         area%d%r2_sp(lb(1):ub(1), lb(2):ub(2)) = 0.0_real_4
      CASE (dbcsr_type_real_8_2d)
         area%d%r2_dp(lb(1):ub(1), lb(2):ub(2)) = 0.0_real_8
      CASE (dbcsr_type_complex_4_2d)
         area%d%c2_sp(lb(1):ub(1), lb(2):ub(2)) = 0.0_real_4
      CASE (dbcsr_type_complex_8_2d)
         area%d%c2_dp(lb(1):ub(1), lb(2):ub(2)) = 0.0_real_8
      CASE default
         DBCSR_ABORT("Invalid data type.")
      END SELECT

      IF (area%d%memory_type%acc_devalloc) &
         DBCSR_ABORT("not yet supported for acc devmem")

      IF (careful_mod) THEN
         CALL timestop(error_handle)
      ENDIF
   END SUBROUTINE dbcsr_data_zero

   SUBROUTINE dbcsr_data_get_sizes_any(area, sizes, valid)
      !! Returns the allocated data size

      TYPE(dbcsr_data_obj), INTENT(IN)                   :: area
         !! data area to query for size
      INTEGER, DIMENSION(:), INTENT(OUT)                 :: sizes
         !! array with the data sizes
      LOGICAL, INTENT(OUT)                               :: valid
         !! whether the data is actually allocated

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_data_get_sizes_any', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: handle

!   ---------------------------------------------------------------------------

      IF (careful_mod) &
         CALL timeset(routineN, handle)

      valid = .FALSE.
      sizes(:) = 0
      IF (ASSOCIATED(area%d)) THEN
         IF (careful_mod) THEN
            IF (dbcsr_type_is_2d(area%d%data_type)) THEN
               IF (SIZE(sizes) /= 2) &
                  DBCSR_ABORT("Sizes must have 2 elements for 2-D data")
            ELSE
               IF (SIZE(sizes) /= 1) &
                  DBCSR_ABORT("Sizes must have 1 elements for 1-D data")
            ENDIF
         ENDIF
         valid = dbcsr_data_exists(area)
         IF (valid) THEN
            SELECT CASE (area%d%data_type)
            CASE (dbcsr_type_real_8)
               sizes(1) = SIZE(area%d%r_dp)
            CASE (dbcsr_type_real_4)
               sizes(1) = SIZE(area%d%r_sp)
            CASE (dbcsr_type_complex_8)
               sizes(1) = SIZE(area%d%c_dp)
            CASE (dbcsr_type_complex_4)
               sizes(1) = SIZE(area%d%c_sp)
            CASE (dbcsr_type_real_8_2d)
               sizes(1) = SIZE(area%d%r2_dp, 1)
               sizes(2) = SIZE(area%d%r2_dp, 2)
            CASE (dbcsr_type_real_4_2d)
               sizes(1) = SIZE(area%d%r2_sp, 1)
               sizes(2) = SIZE(area%d%r2_sp, 2)
            CASE (dbcsr_type_complex_8_2d)
               sizes(1) = SIZE(area%d%c2_dp, 1)
               sizes(2) = SIZE(area%d%c2_dp, 2)
            CASE (dbcsr_type_complex_4_2d)
               sizes(1) = SIZE(area%d%c2_sp, 1)
               sizes(2) = SIZE(area%d%c2_sp, 2)
            CASE default
               DBCSR_ABORT("Incorrect data type")
            END SELECT
         ENDIF
      ENDIF
      IF (careful_mod) &
         CALL timestop(handle)
   END SUBROUTINE dbcsr_data_get_sizes_any

   SUBROUTINE dbcsr_data_get_sizes_2(area, row_size, col_size, valid)
      !! Returns the allocated data size

      TYPE(dbcsr_data_obj), INTENT(IN)                   :: area
         !! data area to query for size, should be 2-D
      INTEGER, INTENT(OUT)                               :: row_size, col_size
         !! row size
         !! column size
      LOGICAL, INTENT(OUT)                               :: valid
         !! whether the data is actually allocated

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_data_get_sizes_2', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: handle
      INTEGER, DIMENSION(2)                              :: s

!   ---------------------------------------------------------------------------

      IF (careful_mod) &
         CALL timeset(routineN, handle)
      IF (ASSOCIATED(area%d)) THEN
         IF (careful_mod .AND. .NOT. dbcsr_type_is_2d(area%d%data_type)) &
            DBCSR_ABORT("1-D data can not have column size")
         CALL dbcsr_data_get_sizes_any(area, s, valid)
         row_size = s(1)
         col_size = s(2)
      ELSE
         valid = .FALSE.
         row_size = 0
         col_size = 0
      ENDIF
      IF (careful_mod) &
         CALL timestop(handle)
   END SUBROUTINE dbcsr_data_get_sizes_2

   SUBROUTINE dbcsr_data_get_sizes_1(area, total_size, valid)
      !! Returns the allocated data size

      TYPE(dbcsr_data_obj), INTENT(IN)                   :: area
         !! data area to query for size
      INTEGER, INTENT(OUT)                               :: total_size
         !! size of array
      LOGICAL, INTENT(OUT)                               :: valid
         !! whether the data is actually allocated

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_data_get_sizes_1', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: handle
      INTEGER, DIMENSION(1)                              :: s

!   ---------------------------------------------------------------------------

      CALL timeset(routineN, handle)

      IF (ASSOCIATED(area%d)) THEN
         IF (careful_mod .AND. dbcsr_type_is_2d(area%d%data_type)) &
            DBCSR_ABORT("Should not use 2-D data")
         CALL dbcsr_data_get_sizes_any(area, s, valid)
         total_size = s(1)
      ELSE
         valid = .FALSE.
         total_size = 0
      ENDIF
      CALL timestop(handle)
   END SUBROUTINE dbcsr_data_get_sizes_1

   ELEMENTAL FUNCTION dbcsr_scalar_one(data_type) RESULT(one)
      !! Returns an encapsulated scalar "1"

      INTEGER, INTENT(IN)                                :: data_type
         !! use the data type
      TYPE(dbcsr_scalar_type)                            :: one
         !! encapsulated value of one

      one = dbcsr_scalar_zero(data_type)
      SELECT CASE (data_type)
      CASE (dbcsr_type_real_4)
         one%r_sp = 1.0_real_4
      CASE (dbcsr_type_real_8)
         one%r_dp = 1.0_real_8
      CASE (dbcsr_type_complex_4)
         one%c_sp = CMPLX(1.0, 0.0, real_4)
      CASE (dbcsr_type_complex_8)
         one%c_dp = CMPLX(1.0, 0.0, real_8)
      END SELECT
   END FUNCTION dbcsr_scalar_one

   ELEMENTAL FUNCTION dbcsr_scalar_i(data_type) RESULT(i)
      !! Returns an encapsulated scalar "sqrt(-1)"

      INTEGER, INTENT(IN)                                :: data_type
         !! use the data type
      TYPE(dbcsr_scalar_type)                            :: i
         !! encapsulated value of i

      i = dbcsr_scalar_zero(data_type)
      SELECT CASE (data_type)
      CASE (dbcsr_type_real_4)
         i%r_sp = 0.0_real_4
      CASE (dbcsr_type_real_8)
         i%r_dp = 0.0_real_8
      CASE (dbcsr_type_complex_4)
         i%c_sp = CMPLX(0.0, 1.0, real_4)
      CASE (dbcsr_type_complex_8)
         i%c_dp = CMPLX(0.0, 1.0, real_8)
      END SELECT
   END FUNCTION dbcsr_scalar_i

   ELEMENTAL FUNCTION dbcsr_scalar_zero(data_type) RESULT(zero)
      !! Returns an encapsulated scalar "0"

      INTEGER, INTENT(IN)                                :: data_type
         !! use the data type
      TYPE(dbcsr_scalar_type)                            :: zero
         !! encapsulated value of zero

      zero%data_type = data_type
      zero%r_sp = 0.0_real_4
      zero%r_dp = 0.0_real_8
      zero%c_sp = CMPLX(0.0, 0.0, real_4)
      zero%c_dp = CMPLX(0.0, 0.0, real_8)
   END FUNCTION dbcsr_scalar_zero

   ELEMENTAL FUNCTION dbcsr_scalar_are_equal(s1, s2) RESULT(are_equal)
      !! Returns whether an encapsulated scalar is equal to another value

      TYPE(dbcsr_scalar_type), INTENT(IN)                :: s1, s2
         !! one value
         !! another value
      LOGICAL                                            :: are_equal
         !! whether values are equal

      IF (s1%data_type .NE. s2%data_type) THEN
         are_equal = .FALSE.
      ELSE
         SELECT CASE (s1%data_type)
         CASE (dbcsr_type_real_4)
            are_equal = s1%r_sp .EQ. s2%r_sp
         CASE (dbcsr_type_real_8)
            are_equal = s1%r_dp .EQ. s2%r_dp
         CASE (dbcsr_type_complex_4)
            are_equal = s1%c_sp .EQ. s2%c_sp
         CASE (dbcsr_type_complex_8)
            are_equal = s1%c_dp .EQ. s2%c_dp
         CASE default
            are_equal = .FALSE.
         END SELECT
      ENDIF
   END FUNCTION dbcsr_scalar_are_equal

   ELEMENTAL FUNCTION dbcsr_scalar_negative(s) RESULT(negated)
      !! Returns an encapsulated scalar as a negation of the given

      TYPE(dbcsr_scalar_type), INTENT(IN)                :: s
         !! given value
      TYPE(dbcsr_scalar_type)                            :: negated
         !! negated value

      negated = dbcsr_scalar_zero(s%data_type)
      SELECT CASE (s%data_type)
      CASE (dbcsr_type_real_4)
         negated%r_sp = -s%r_sp
      CASE (dbcsr_type_real_8)
         negated%r_dp = -s%r_dp
      CASE (dbcsr_type_complex_4)
         negated%c_sp = -s%c_sp
      CASE (dbcsr_type_complex_8)
         negated%c_dp = -s%c_dp
      CASE default
         negated = dbcsr_scalar_zero(s%data_type)
      END SELECT
   END FUNCTION dbcsr_scalar_negative

   ELEMENTAL FUNCTION dbcsr_scalar_add(s1, s2) RESULT(s_sum)
      !! Returns an encapsulated scalar as a negation of the given
      TYPE(dbcsr_scalar_type), INTENT(IN)                :: s1, s2
      TYPE(dbcsr_scalar_type)                            :: s_sum

      s_sum = dbcsr_scalar_zero(s1%data_type)
      SELECT CASE (s1%data_type)
      CASE (dbcsr_type_real_4)
         s_sum%r_sp = s1%r_sp + s2%r_sp
      CASE (dbcsr_type_real_8)
         s_sum%r_dp = s1%r_dp + s2%r_dp
      CASE (dbcsr_type_complex_4)
         s_sum%c_sp = s1%c_sp + s2%c_sp
      CASE (dbcsr_type_complex_8)
         s_sum%c_dp = s1%c_dp + s2%c_dp
      CASE default
         s_sum = dbcsr_scalar_zero(s1%data_type)
      END SELECT
   END FUNCTION dbcsr_scalar_add

   ELEMENTAL FUNCTION dbcsr_scalar_multiply(s1, s2) RESULT(s_product)
      TYPE(dbcsr_scalar_type), INTENT(IN)                :: s1, s2
      TYPE(dbcsr_scalar_type)                            :: s_product

      s_product = dbcsr_scalar_zero(s1%data_type)
      SELECT CASE (s1%data_type)
      CASE (dbcsr_type_real_4)
         s_product%r_sp = s1%r_sp*s2%r_sp
      CASE (dbcsr_type_real_8)
         s_product%r_dp = s1%r_dp*s2%r_dp
      CASE (dbcsr_type_complex_4)
         s_product%c_sp = s1%c_sp*s2%c_sp
      CASE (dbcsr_type_complex_8)
         s_product%c_dp = s1%c_dp*s2%c_dp
      CASE default
         s_product = dbcsr_scalar_zero(s1%data_type)
      END SELECT
   END FUNCTION dbcsr_scalar_multiply

   ELEMENTAL FUNCTION dbcsr_scalar_get_type(scalar) RESULT(data_type)
      !! Returns data type of a scalar

      TYPE(dbcsr_scalar_type), INTENT(IN)                :: scalar
         !! scalar
      INTEGER                                            :: data_type
         !! data type of the scalar

      data_type = scalar%data_type
   END FUNCTION dbcsr_scalar_get_type

   ELEMENTAL SUBROUTINE dbcsr_scalar_set_type(scalar, data_type)
      !! Sets data type of a scalar

      TYPE(dbcsr_scalar_type), INTENT(INOUT)             :: scalar
         !! scalar
      INTEGER, INTENT(IN)                                :: data_type

      scalar%data_type = data_type
   END SUBROUTINE dbcsr_scalar_set_type

   ELEMENTAL SUBROUTINE dbcsr_scalar_fill_all(scalar)
      !! Fills all data and precision types from the set one

      TYPE(dbcsr_scalar_type), INTENT(INOUT)             :: scalar
         !! data area

      SELECT CASE (scalar%data_type)
      CASE (dbcsr_type_real_4)
         !scalar%r_sp = 0
         scalar%r_dp = REAL(scalar%r_sp, KIND=real_8)
         scalar%c_sp = CMPLX(scalar%r_sp, 0, KIND=real_4)
         scalar%c_dp = CMPLX(scalar%r_sp, 0, KIND=real_8)
      CASE (dbcsr_type_real_8)
         scalar%r_sp = REAL(scalar%r_dp, KIND=real_4)
         !scalar%r_dp = REAL(scalar%r_dp, KIND=real_8)
         scalar%c_sp = CMPLX(scalar%r_dp, 0, KIND=real_4)
         scalar%c_dp = CMPLX(scalar%r_dp, 0, KIND=real_8)
      CASE (dbcsr_type_complex_4)
         scalar%r_sp = REAL(scalar%c_sp, KIND=real_4)
         scalar%r_dp = REAL(scalar%c_sp, KIND=real_8)
         !scalar%c_sp = CMPLX(scalar%c_sp, KIND=real_4)
         scalar%c_dp = CMPLX(scalar%c_sp, KIND=real_8)
      CASE (dbcsr_type_complex_8)
         scalar%r_sp = REAL(scalar%c_dp, KIND=real_4)
         scalar%r_dp = REAL(scalar%c_dp, KIND=real_8)
         scalar%c_sp = CMPLX(scalar%c_dp, KIND=real_4)
         !scalar%c_dp = CMPLX(scalar%r_dp, KIND=real_8)
      END SELECT
   END SUBROUTINE dbcsr_scalar_fill_all

   PURE FUNCTION dbcsr_type_is_2d(data_type)
      !! Checks whether the data type is 2-D.
      !! \return Data type is 2-D.

      INTEGER, INTENT(IN)                                :: data_type
      LOGICAL                                            :: dbcsr_type_is_2d

      dbcsr_type_is_2d = data_type .EQ. dbcsr_type_real_4_2d .OR. &
                         data_type .EQ. dbcsr_type_real_8_2d .OR. &
                         data_type .EQ. dbcsr_type_complex_4_2d .OR. &
                         data_type .EQ. dbcsr_type_complex_8_2d
   END FUNCTION dbcsr_type_is_2d

   PURE FUNCTION dbcsr_type_2d_to_1d(data_type)
      !! Returns 1-d data type corresponding to the given 2-D one.
      !! \return 1-D data type

      INTEGER, INTENT(IN)                                :: data_type
      INTEGER                                            :: dbcsr_type_2d_to_1d

      SELECT CASE (data_type)
      CASE (dbcsr_type_real_4_2d)
         dbcsr_type_2d_to_1d = dbcsr_type_real_4
      CASE (dbcsr_type_real_8_2d)
         dbcsr_type_2d_to_1d = dbcsr_type_real_8
      CASE (dbcsr_type_complex_4_2d)
         dbcsr_type_2d_to_1d = dbcsr_type_complex_4
      CASE (dbcsr_type_complex_8_2d)
         dbcsr_type_2d_to_1d = dbcsr_type_complex_8
      CASE (dbcsr_type_real_4)
         dbcsr_type_2d_to_1d = dbcsr_type_real_4
      CASE (dbcsr_type_real_8)
         dbcsr_type_2d_to_1d = dbcsr_type_real_8
      CASE (dbcsr_type_complex_4)
         dbcsr_type_2d_to_1d = dbcsr_type_complex_4
      CASE (dbcsr_type_complex_8)
         dbcsr_type_2d_to_1d = dbcsr_type_complex_8
      CASE default
         dbcsr_type_2d_to_1d = -1
      END SELECT
   END FUNCTION dbcsr_type_2d_to_1d

   PURE FUNCTION dbcsr_type_1d_to_2d(data_type)
      !! Returns 2-D data type corresponding to the given 1-D one.
      !! \return 2-D data type

      INTEGER, INTENT(IN)                                :: data_type
      INTEGER                                            :: dbcsr_type_1d_to_2d

      SELECT CASE (data_type)
      CASE (dbcsr_type_real_4)
         dbcsr_type_1d_to_2d = dbcsr_type_real_4_2d
      CASE (dbcsr_type_real_8)
         dbcsr_type_1d_to_2d = dbcsr_type_real_8_2d
      CASE (dbcsr_type_complex_4)
         dbcsr_type_1d_to_2d = dbcsr_type_complex_4_2d
      CASE (dbcsr_type_complex_8)
         dbcsr_type_1d_to_2d = dbcsr_type_complex_8_2d
      CASE (dbcsr_type_real_4_2d)
         dbcsr_type_1d_to_2d = dbcsr_type_real_4_2d
      CASE (dbcsr_type_real_8_2d)
         dbcsr_type_1d_to_2d = dbcsr_type_real_8_2d
      CASE (dbcsr_type_complex_4_2d)
         dbcsr_type_1d_to_2d = dbcsr_type_complex_4_2d
      CASE (dbcsr_type_complex_8_2d)
         dbcsr_type_1d_to_2d = dbcsr_type_complex_8_2d
      CASE default
         dbcsr_type_1d_to_2d = -1
      END SELECT
   END FUNCTION dbcsr_type_1d_to_2d

   PURE FUNCTION dbcsr_data_get_size_referenced(area) RESULT(data_size_referenced)
      !! Get actual data storage used for matrix

      TYPE(dbcsr_data_obj), INTENT(IN)                   :: area
         !! Count data of this matrix
      INTEGER                                            :: data_size_referenced
         !! Data size used by matrix

      IF (ASSOCIATED(area%d)) THEN
         data_size_referenced = area%d%ref_size
      ELSE
         data_size_referenced = 0
      ENDIF
   END FUNCTION dbcsr_data_get_size_referenced

   PURE SUBROUTINE dbcsr_data_set_size_referenced(data_area, referenced_size)
      !! Sets the referenced size of the data area

      TYPE(dbcsr_data_obj), INTENT(INOUT)                :: data_area
         !! area for which to set referenced data size
      INTEGER, INTENT(IN)                                :: referenced_size
         !! set referenced data size to this value

      data_area%d%ref_size = referenced_size
   END SUBROUTINE dbcsr_data_set_size_referenced

! **************************************************************************************************

#:include "dbcsr_data_methods_low.f90"

END MODULE dbcsr_data_methods_low
