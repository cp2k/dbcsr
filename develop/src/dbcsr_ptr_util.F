# 1 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F" 1
!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_ptr_util
   !! DBCSR pointer and unmanaged array utilities
   USE dbcsr_acc_hostmem, ONLY: acc_hostmem_allocate, &
                                acc_hostmem_deallocate
   USE dbcsr_config, ONLY: dbcsr_cfg
   USE dbcsr_data_types, ONLY: dbcsr_data_obj, &
                               dbcsr_memtype_default, &
                               dbcsr_memtype_type, &
                               dbcsr_type_complex_4, &
                               dbcsr_type_complex_8, &
                               dbcsr_type_real_4, &
                               dbcsr_type_real_8
   USE dbcsr_kinds, ONLY: dp, &
                          int_4, &
                          int_8, &
                          real_4, &
                          real_8
   USE dbcsr_mpiwrap, ONLY: mp_allocate, &
                            mp_deallocate
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_ptr_util'

   LOGICAL, PARAMETER :: careful_mod = .FALSE.

   PUBLIC :: ensure_array_size
   PUBLIC :: memory_allocate, memory_deallocate
   PUBLIC :: memory_zero
   PUBLIC :: pointer_view
   PUBLIC :: pointer_rank_remap2
   PUBLIC :: memory_copy

   INTERFACE ensure_array_size
      MODULE PROCEDURE ensure_array_size_i, ensure_array_size_l
      MODULE PROCEDURE ensure_array_size_s, ensure_array_size_d, &
         ensure_array_size_c, ensure_array_size_z
   END INTERFACE

   INTERFACE pointer_view
      MODULE PROCEDURE pointer_view_s, pointer_view_d, &
         pointer_view_c, pointer_view_z
      MODULE PROCEDURE pointer_view_i, pointer_view_l
      MODULE PROCEDURE pointer_view_a
   END INTERFACE

   INTERFACE pointer_rank_remap2
      MODULE PROCEDURE pointer_s_rank_remap2, pointer_d_rank_remap2, &
         pointer_c_rank_remap2, pointer_z_rank_remap2, &
         pointer_l_rank_remap2, pointer_i_rank_remap2
   END INTERFACE

   INTERFACE memory_copy
      MODULE PROCEDURE mem_copy_i, mem_copy_l, &
         mem_copy_s, mem_copy_d, &
         mem_copy_c, mem_copy_z
   END INTERFACE

   INTERFACE memory_zero
      MODULE PROCEDURE mem_zero_i, mem_zero_l
      MODULE PROCEDURE mem_zero_s, mem_zero_d, mem_zero_c, mem_zero_z
   END INTERFACE

   INTERFACE memory_allocate
      MODULE PROCEDURE mem_alloc_i, mem_alloc_l, mem_alloc_s, mem_alloc_d, mem_alloc_c, mem_alloc_z
      MODULE PROCEDURE mem_alloc_i_2d, mem_alloc_l_2d, mem_alloc_s_2d, mem_alloc_d_2d, mem_alloc_c_2d, mem_alloc_z_2d
   END INTERFACE

   INTERFACE memory_deallocate
      MODULE PROCEDURE mem_dealloc_i, mem_dealloc_l, mem_dealloc_s, mem_dealloc_d, mem_dealloc_c, mem_dealloc_z
      MODULE PROCEDURE mem_dealloc_i_2d, mem_dealloc_l_2d, mem_dealloc_s_2d, mem_dealloc_d_2d, mem_dealloc_c_2d, mem_dealloc_z_2d
   END INTERFACE

CONTAINS

   FUNCTION pointer_view_a(new_area, area, offset, len) RESULT(narea2)
      !! Repoints a pointer into a part of a data area

      TYPE(dbcsr_data_obj), INTENT(INOUT)                :: new_area
         !! repoints this encapsulated pointer
      TYPE(dbcsr_data_obj), INTENT(IN)                   :: area
         !! area to point into
      INTEGER, INTENT(IN)                                :: offset
         !! point to this offset in area
      INTEGER, INTENT(IN), OPTIONAL                      :: len
         !! length of data area to point to
      TYPE(dbcsr_data_obj)                               :: narea2
         !! copy of new_area

      IF (area%d%data_type /= new_area%d%data_type) &
         DBCSR_ABORT("Incompatible data types.")
      IF (PRESENT(len)) THEN
         SELECT CASE (area%d%data_type)
         CASE (dbcsr_type_real_4)
            new_area%d%r_sp => area%d%r_sp(offset:offset + len - 1)
         CASE (dbcsr_type_real_8)
            new_area%d%r_dp => area%d%r_dp(offset:offset + len - 1)
         CASE (dbcsr_type_complex_4)
            new_area%d%c_sp => area%d%c_sp(offset:offset + len - 1)
         CASE (dbcsr_type_complex_8)
            new_area%d%c_dp => area%d%c_dp(offset:offset + len - 1)
         CASE default
            DBCSR_ABORT("Invalid data type.")
         END SELECT
      ELSE
         SELECT CASE (area%d%data_type)
         CASE (dbcsr_type_real_4)
            new_area%d%r_sp => area%d%r_sp(offset:)
         CASE (dbcsr_type_real_8)
            new_area%d%r_dp => area%d%r_dp(offset:)
         CASE (dbcsr_type_complex_4)
            new_area%d%c_sp => area%d%c_sp(offset:)
         CASE (dbcsr_type_complex_8)
            new_area%d%c_dp => area%d%c_dp(offset:)
         CASE default
            DBCSR_ABORT("Invalid data type.")
         END SELECT
      END IF
      narea2 = new_area
   END FUNCTION pointer_view_a

# 1 "/__w/dbcsr/dbcsr/src/data/dbcsr.fypp" 1
# 9 "/__w/dbcsr/dbcsr/src/data/dbcsr.fypp"

# 11 "/__w/dbcsr/dbcsr/src/data/dbcsr.fypp"

# 169 "/__w/dbcsr/dbcsr/src/data/dbcsr.fypp"
# 135 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F" 2
# 136 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
      FUNCTION pointer_view_d (original, lb, ub) RESULT(view)
# 137 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Returns a pointer with different bounds.

         REAL(kind=real_8), DIMENSION(:), POINTER :: original, view
# 140 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! original data pointer
        !! new pointer
         INTEGER, INTENT(IN)                  :: lb, ub
        !! lower and upper bound for the new pointer view
        !! lower and upper bound for the new pointer view
         view => original(lb:ub)
      END FUNCTION pointer_view_d
# 147 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE ensure_array_size_d (array, array_resize, lb, ub, factor, &
# 149 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                                  nocopy, memory_type, zero_pad)
     !! Ensures that an array is appropriately large.

         REAL(kind=real_8), DIMENSION(:), POINTER, CONTIGUOUS     :: array
# 153 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! array to verify and possibly resize
         REAL(kind=real_8), DIMENSION(:), POINTER, OPTIONAL       :: array_resize
# 155 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER, INTENT(IN), OPTIONAL                  :: lb
        !! desired array lower bound
         INTEGER, INTENT(IN)                            :: ub
        !! desired array upper bound
         REAL(KIND=dp), INTENT(IN), OPTIONAL            :: factor
        !! factor by which to exaggerate enlargements
         LOGICAL, INTENT(IN), OPTIONAL                  :: nocopy, zero_pad
        !! copy array on enlargement; default is to copy
        !! zero new allocations; default is to write nothing
         TYPE(dbcsr_memtype_type), INTENT(IN), OPTIONAL :: memory_type
        !! use special memory

         CHARACTER(len=*), PARAMETER :: routineN = 'ensure_array_size_d', &
# 168 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                        routineP = moduleN//':'//routineN

         INTEGER                                  :: lb_new, lb_orig, &
                                                     ub_new, ub_orig, old_size, &
                                                     size_increase
         TYPE(dbcsr_memtype_type)                 :: mem_type
         LOGICAL                                  :: dbg, docopy, &
                                                     pad
         REAL(kind=real_8), DIMENSION(:), POINTER, CONTIGUOUS    :: newarray
# 177 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

!   ---------------------------------------------------------------------------
         !CALL timeset(routineN, error_handler)
         dbg = .FALSE.

         IF (PRESENT(array_resize)) NULLIFY (array_resize)

         IF (PRESENT(nocopy)) THEN
            docopy = .NOT. nocopy
         ELSE
            docopy = .TRUE.
         END IF
         IF (PRESENT(memory_type)) THEN
            mem_type = memory_type
         ELSE
            mem_type = dbcsr_memtype_default
         END IF
         lb_new = 1
         IF (PRESENT(lb)) lb_new = lb
         pad = .FALSE.
         IF (PRESENT(zero_pad)) pad = zero_pad
         ! Creates a new array if it doesn't yet exist.
         IF (.NOT. ASSOCIATED(array)) THEN
            IF (lb_new /= 1) &
               DBCSR_ABORT("Arrays must start at 1")
            CALL mem_alloc_d (array, ub, mem_type=mem_type)
# 203 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            IF (pad .AND. ub .GT. 0) CALL mem_zero_d (array, ub)
# 204 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            !CALL timestop(error_handler)
            RETURN
         END IF
         lb_orig = LBOUND(array, 1)
         ub_orig = UBOUND(array, 1)
         old_size = ub_orig - lb_orig + 1
         ! The existing array is big enough.
         IF (lb_orig .LE. lb_new .AND. ub_orig .GE. ub) THEN
            !CALL timestop(error_handler)
            RETURN
         END IF
         ! A reallocation must be performed
         IF (dbg) WRITE (*, *) routineP//' Current bounds are', lb_orig, ':', ub_orig, &
            '; special?' !,mem_type
         !CALL timeset(routineN,timing_handle)
         IF (lb_orig .GT. lb_new) THEN
            IF (PRESENT(factor)) THEN
               size_increase = lb_orig - lb_new
               size_increase = MAX(NINT(size_increase*factor), &
                                   NINT(old_size*(factor - 1)), 0)
               lb_new = MIN(lb_orig, lb_new - size_increase)
            ELSE
               lb_new = lb_orig
            END IF
         END IF
         IF (ub_orig .LT. ub) THEN
            IF (PRESENT(factor)) THEN
               size_increase = ub - ub_orig
               size_increase = MAX(NINT(size_increase*factor), &
                                   NINT(old_size*(factor - 1)), 0)
               ub_new = MAX(ub_orig, ub + size_increase)
            ELSE
               ub_new = ub
            END IF
         ELSE
            ub_new = ub
         END IF
         IF (dbg) WRITE (*, *) routineP//' Resizing to bounds', lb_new, ':', ub_new
         !
         ! Deallocates the old array if it's not needed to copy the old data.
         IF (.NOT. docopy) THEN
            IF (PRESENT(array_resize)) THEN
               array_resize => array
               NULLIFY (array)
            ELSE
               CALL mem_dealloc_d (array, mem_type=mem_type)
# 250 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
         END IF
         !
         ! Allocates the new array
         IF (lb_new /= 1) &
            DBCSR_ABORT("Arrays must start at 1")
         CALL mem_alloc_d (newarray, ub_new - lb_new + 1, mem_type)
# 257 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         !
         ! Now copy and/or zero pad.
         IF (docopy) THEN
            IF (dbg .AND. (lb_new .GT. lb_orig .OR. ub_new .LT. ub_orig)) &
               DBCSR_ABORT("Old extent exceeds the new one.")
            IF (ub_orig - lb_orig + 1 .gt. 0) THEN
               !newarray(lb_orig:ub_orig) = array(lb_orig:ub_orig)
               CALL mem_copy_d (newarray(lb_orig:ub_orig), &
# 265 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                            array(lb_orig:ub_orig), ub_orig - lb_orig + 1)
            END IF
            IF (pad) THEN
               !newarray(lb_new:lb_orig-1) = 0
               CALL mem_zero_d (newarray(lb_new:lb_orig - 1), (lb_orig - 1) - lb_new + 1)
# 270 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
               !newarray(ub_orig+1:ub_new) = 0
               CALL mem_zero_d (newarray(ub_orig + 1:ub_new), ub_new - (ub_orig + 1) + 1)
# 272 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
            IF (PRESENT(array_resize)) THEN
               array_resize => array
               NULLIFY (array)
            ELSE
               CALL mem_dealloc_d (array, mem_type=mem_type)
# 278 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
         ELSEIF (pad) THEN
            !newarray(:) = 0.0_real_8
# 281 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            CALL mem_zero_d (newarray, SIZE(newarray))
# 282 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         END IF
         array => newarray
         IF (dbg) WRITE (*, *) routineP//' New array size', SIZE(array)
         !CALL timestop(error_handler)
      END SUBROUTINE ensure_array_size_d
# 287 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_copy_d (dst, src, n)
# 289 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Copies memory area

         INTEGER, INTENT(IN) :: n
        !! length of copy
         REAL(kind=real_8), DIMENSION(1:n), INTENT(OUT) :: dst
# 294 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! destination memory
         REAL(kind=real_8), DIMENSION(1:n), INTENT(IN)  :: src
# 296 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! source memory
         dst(:) = src(:)
      END SUBROUTINE mem_copy_d
# 299 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_zero_d (dst, n)
# 301 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Zeros memory area

         INTEGER, INTENT(IN) :: n
        !! length of elements to zero
         REAL(kind=real_8), DIMENSION(1:n), INTENT(OUT) :: dst
# 306 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! destination memory
         dst(:) = 0.0_real_8
# 308 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
      END SUBROUTINE mem_zero_d
# 309 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_alloc_d (mem, n, mem_type)
# 311 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Allocates memory

         REAL(kind=real_8), DIMENSION(:), POINTER, CONTIGUOUS :: mem
# 314 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         INTEGER, INTENT(IN)                   :: n
        !! length of elements to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_alloc_d'
# 320 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc .AND. n > 1) THEN
            CALL acc_hostmem_allocate(mem, n, mem_type%acc_stream)
         ELSE IF (mem_type%mpi .AND. dbcsr_cfg%use_mpi_allocator%val) THEN
            CALL mp_allocate(mem, n)
         ELSE
            ALLOCATE (mem(n))
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_alloc_d
# 337 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_alloc_d_2d(mem, sizes, mem_type)
# 339 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Allocates memory

         REAL(kind=real_8), DIMENSION(:, :), POINTER      :: mem
# 342 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         INTEGER, DIMENSION(2), INTENT(IN)     :: sizes
        !! length of elements to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_alloc_d_2d'
# 348 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc) THEN
            DBCSR_ABORT("Accelerator hostalloc not supported for 2D arrays.")
            !CALL acc_hostmem_allocate(mem, n, mem_type%acc_stream)
         ELSE IF (mem_type%mpi) THEN
            DBCSR_ABORT("MPI allocate not supported for 2D arrays.")
            !CALL mp_allocate(mem, n)
         ELSE
            ALLOCATE (mem(sizes(1), sizes(2)))
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_alloc_d_2d
# 367 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_dealloc_d (mem, mem_type)
# 369 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Deallocates memory

         REAL(kind=real_8), DIMENSION(:), POINTER, CONTIGUOUS :: mem
# 372 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_dealloc_d'
# 376 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc .AND. SIZE(mem) > 1) THEN
            CALL acc_hostmem_deallocate(mem, mem_type%acc_stream)
         ELSE IF (mem_type%mpi .AND. dbcsr_cfg%use_mpi_allocator%val) THEN
            CALL mp_deallocate(mem)
         ELSE
            DEALLOCATE (mem)
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_dealloc_d
# 393 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_dealloc_d_2d(mem, mem_type)
# 395 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Deallocates memory

         REAL(kind=real_8), DIMENSION(:, :), POINTER      :: mem
# 398 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_dealloc_d'
# 402 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc) THEN
            DBCSR_ABORT("Accelerator host deallocate not supported for 2D arrays.")
            !CALL acc_hostmem_deallocate(mem, mem_type%acc_stream)
         ELSE IF (mem_type%mpi) THEN
            DBCSR_ABORT("MPI deallocate not supported for 2D arrays.")
            !CALL mp_deallocate(mem)
         ELSE
            DEALLOCATE (mem)
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_dealloc_d_2d
# 421 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE pointer_d_rank_remap2(r2p, d1, d2, r1p)
# 423 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Sets a rank-2 pointer to rank-1 data using Fortran 2003 pointer
     !! rank remapping.

         INTEGER, INTENT(IN)                      :: d1, d2
         REAL(kind=real_8), DIMENSION(:, :), &
# 428 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            POINTER                                :: r2p
         REAL(kind=real_8), DIMENSION(:), &
# 430 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            POINTER                                :: r1p

         r2p(1:d1, 1:d2) => r1p(1:d1*d2)
      END SUBROUTINE pointer_d_rank_remap2
# 434 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
# 136 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
      FUNCTION pointer_view_s (original, lb, ub) RESULT(view)
# 137 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Returns a pointer with different bounds.

         REAL(kind=real_4), DIMENSION(:), POINTER :: original, view
# 140 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! original data pointer
        !! new pointer
         INTEGER, INTENT(IN)                  :: lb, ub
        !! lower and upper bound for the new pointer view
        !! lower and upper bound for the new pointer view
         view => original(lb:ub)
      END FUNCTION pointer_view_s
# 147 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE ensure_array_size_s (array, array_resize, lb, ub, factor, &
# 149 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                                  nocopy, memory_type, zero_pad)
     !! Ensures that an array is appropriately large.

         REAL(kind=real_4), DIMENSION(:), POINTER, CONTIGUOUS     :: array
# 153 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! array to verify and possibly resize
         REAL(kind=real_4), DIMENSION(:), POINTER, OPTIONAL       :: array_resize
# 155 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER, INTENT(IN), OPTIONAL                  :: lb
        !! desired array lower bound
         INTEGER, INTENT(IN)                            :: ub
        !! desired array upper bound
         REAL(KIND=dp), INTENT(IN), OPTIONAL            :: factor
        !! factor by which to exaggerate enlargements
         LOGICAL, INTENT(IN), OPTIONAL                  :: nocopy, zero_pad
        !! copy array on enlargement; default is to copy
        !! zero new allocations; default is to write nothing
         TYPE(dbcsr_memtype_type), INTENT(IN), OPTIONAL :: memory_type
        !! use special memory

         CHARACTER(len=*), PARAMETER :: routineN = 'ensure_array_size_s', &
# 168 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                        routineP = moduleN//':'//routineN

         INTEGER                                  :: lb_new, lb_orig, &
                                                     ub_new, ub_orig, old_size, &
                                                     size_increase
         TYPE(dbcsr_memtype_type)                 :: mem_type
         LOGICAL                                  :: dbg, docopy, &
                                                     pad
         REAL(kind=real_4), DIMENSION(:), POINTER, CONTIGUOUS    :: newarray
# 177 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

!   ---------------------------------------------------------------------------
         !CALL timeset(routineN, error_handler)
         dbg = .FALSE.

         IF (PRESENT(array_resize)) NULLIFY (array_resize)

         IF (PRESENT(nocopy)) THEN
            docopy = .NOT. nocopy
         ELSE
            docopy = .TRUE.
         END IF
         IF (PRESENT(memory_type)) THEN
            mem_type = memory_type
         ELSE
            mem_type = dbcsr_memtype_default
         END IF
         lb_new = 1
         IF (PRESENT(lb)) lb_new = lb
         pad = .FALSE.
         IF (PRESENT(zero_pad)) pad = zero_pad
         ! Creates a new array if it doesn't yet exist.
         IF (.NOT. ASSOCIATED(array)) THEN
            IF (lb_new /= 1) &
               DBCSR_ABORT("Arrays must start at 1")
            CALL mem_alloc_s (array, ub, mem_type=mem_type)
# 203 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            IF (pad .AND. ub .GT. 0) CALL mem_zero_s (array, ub)
# 204 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            !CALL timestop(error_handler)
            RETURN
         END IF
         lb_orig = LBOUND(array, 1)
         ub_orig = UBOUND(array, 1)
         old_size = ub_orig - lb_orig + 1
         ! The existing array is big enough.
         IF (lb_orig .LE. lb_new .AND. ub_orig .GE. ub) THEN
            !CALL timestop(error_handler)
            RETURN
         END IF
         ! A reallocation must be performed
         IF (dbg) WRITE (*, *) routineP//' Current bounds are', lb_orig, ':', ub_orig, &
            '; special?' !,mem_type
         !CALL timeset(routineN,timing_handle)
         IF (lb_orig .GT. lb_new) THEN
            IF (PRESENT(factor)) THEN
               size_increase = lb_orig - lb_new
               size_increase = MAX(NINT(size_increase*factor), &
                                   NINT(old_size*(factor - 1)), 0)
               lb_new = MIN(lb_orig, lb_new - size_increase)
            ELSE
               lb_new = lb_orig
            END IF
         END IF
         IF (ub_orig .LT. ub) THEN
            IF (PRESENT(factor)) THEN
               size_increase = ub - ub_orig
               size_increase = MAX(NINT(size_increase*factor), &
                                   NINT(old_size*(factor - 1)), 0)
               ub_new = MAX(ub_orig, ub + size_increase)
            ELSE
               ub_new = ub
            END IF
         ELSE
            ub_new = ub
         END IF
         IF (dbg) WRITE (*, *) routineP//' Resizing to bounds', lb_new, ':', ub_new
         !
         ! Deallocates the old array if it's not needed to copy the old data.
         IF (.NOT. docopy) THEN
            IF (PRESENT(array_resize)) THEN
               array_resize => array
               NULLIFY (array)
            ELSE
               CALL mem_dealloc_s (array, mem_type=mem_type)
# 250 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
         END IF
         !
         ! Allocates the new array
         IF (lb_new /= 1) &
            DBCSR_ABORT("Arrays must start at 1")
         CALL mem_alloc_s (newarray, ub_new - lb_new + 1, mem_type)
# 257 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         !
         ! Now copy and/or zero pad.
         IF (docopy) THEN
            IF (dbg .AND. (lb_new .GT. lb_orig .OR. ub_new .LT. ub_orig)) &
               DBCSR_ABORT("Old extent exceeds the new one.")
            IF (ub_orig - lb_orig + 1 .gt. 0) THEN
               !newarray(lb_orig:ub_orig) = array(lb_orig:ub_orig)
               CALL mem_copy_s (newarray(lb_orig:ub_orig), &
# 265 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                            array(lb_orig:ub_orig), ub_orig - lb_orig + 1)
            END IF
            IF (pad) THEN
               !newarray(lb_new:lb_orig-1) = 0
               CALL mem_zero_s (newarray(lb_new:lb_orig - 1), (lb_orig - 1) - lb_new + 1)
# 270 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
               !newarray(ub_orig+1:ub_new) = 0
               CALL mem_zero_s (newarray(ub_orig + 1:ub_new), ub_new - (ub_orig + 1) + 1)
# 272 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
            IF (PRESENT(array_resize)) THEN
               array_resize => array
               NULLIFY (array)
            ELSE
               CALL mem_dealloc_s (array, mem_type=mem_type)
# 278 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
         ELSEIF (pad) THEN
            !newarray(:) = 0.0_real_4
# 281 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            CALL mem_zero_s (newarray, SIZE(newarray))
# 282 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         END IF
         array => newarray
         IF (dbg) WRITE (*, *) routineP//' New array size', SIZE(array)
         !CALL timestop(error_handler)
      END SUBROUTINE ensure_array_size_s
# 287 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_copy_s (dst, src, n)
# 289 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Copies memory area

         INTEGER, INTENT(IN) :: n
        !! length of copy
         REAL(kind=real_4), DIMENSION(1:n), INTENT(OUT) :: dst
# 294 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! destination memory
         REAL(kind=real_4), DIMENSION(1:n), INTENT(IN)  :: src
# 296 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! source memory
         dst(:) = src(:)
      END SUBROUTINE mem_copy_s
# 299 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_zero_s (dst, n)
# 301 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Zeros memory area

         INTEGER, INTENT(IN) :: n
        !! length of elements to zero
         REAL(kind=real_4), DIMENSION(1:n), INTENT(OUT) :: dst
# 306 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! destination memory
         dst(:) = 0.0_real_4
# 308 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
      END SUBROUTINE mem_zero_s
# 309 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_alloc_s (mem, n, mem_type)
# 311 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Allocates memory

         REAL(kind=real_4), DIMENSION(:), POINTER, CONTIGUOUS :: mem
# 314 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         INTEGER, INTENT(IN)                   :: n
        !! length of elements to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_alloc_s'
# 320 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc .AND. n > 1) THEN
            CALL acc_hostmem_allocate(mem, n, mem_type%acc_stream)
         ELSE IF (mem_type%mpi .AND. dbcsr_cfg%use_mpi_allocator%val) THEN
            CALL mp_allocate(mem, n)
         ELSE
            ALLOCATE (mem(n))
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_alloc_s
# 337 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_alloc_s_2d(mem, sizes, mem_type)
# 339 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Allocates memory

         REAL(kind=real_4), DIMENSION(:, :), POINTER      :: mem
# 342 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         INTEGER, DIMENSION(2), INTENT(IN)     :: sizes
        !! length of elements to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_alloc_s_2d'
# 348 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc) THEN
            DBCSR_ABORT("Accelerator hostalloc not supported for 2D arrays.")
            !CALL acc_hostmem_allocate(mem, n, mem_type%acc_stream)
         ELSE IF (mem_type%mpi) THEN
            DBCSR_ABORT("MPI allocate not supported for 2D arrays.")
            !CALL mp_allocate(mem, n)
         ELSE
            ALLOCATE (mem(sizes(1), sizes(2)))
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_alloc_s_2d
# 367 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_dealloc_s (mem, mem_type)
# 369 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Deallocates memory

         REAL(kind=real_4), DIMENSION(:), POINTER, CONTIGUOUS :: mem
# 372 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_dealloc_s'
# 376 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc .AND. SIZE(mem) > 1) THEN
            CALL acc_hostmem_deallocate(mem, mem_type%acc_stream)
         ELSE IF (mem_type%mpi .AND. dbcsr_cfg%use_mpi_allocator%val) THEN
            CALL mp_deallocate(mem)
         ELSE
            DEALLOCATE (mem)
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_dealloc_s
# 393 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_dealloc_s_2d(mem, mem_type)
# 395 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Deallocates memory

         REAL(kind=real_4), DIMENSION(:, :), POINTER      :: mem
# 398 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_dealloc_s'
# 402 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc) THEN
            DBCSR_ABORT("Accelerator host deallocate not supported for 2D arrays.")
            !CALL acc_hostmem_deallocate(mem, mem_type%acc_stream)
         ELSE IF (mem_type%mpi) THEN
            DBCSR_ABORT("MPI deallocate not supported for 2D arrays.")
            !CALL mp_deallocate(mem)
         ELSE
            DEALLOCATE (mem)
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_dealloc_s_2d
# 421 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE pointer_s_rank_remap2(r2p, d1, d2, r1p)
# 423 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Sets a rank-2 pointer to rank-1 data using Fortran 2003 pointer
     !! rank remapping.

         INTEGER, INTENT(IN)                      :: d1, d2
         REAL(kind=real_4), DIMENSION(:, :), &
# 428 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            POINTER                                :: r2p
         REAL(kind=real_4), DIMENSION(:), &
# 430 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            POINTER                                :: r1p

         r2p(1:d1, 1:d2) => r1p(1:d1*d2)
      END SUBROUTINE pointer_s_rank_remap2
# 434 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
# 136 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
      FUNCTION pointer_view_z (original, lb, ub) RESULT(view)
# 137 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Returns a pointer with different bounds.

         COMPLEX(kind=real_8), DIMENSION(:), POINTER :: original, view
# 140 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! original data pointer
        !! new pointer
         INTEGER, INTENT(IN)                  :: lb, ub
        !! lower and upper bound for the new pointer view
        !! lower and upper bound for the new pointer view
         view => original(lb:ub)
      END FUNCTION pointer_view_z
# 147 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE ensure_array_size_z (array, array_resize, lb, ub, factor, &
# 149 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                                  nocopy, memory_type, zero_pad)
     !! Ensures that an array is appropriately large.

         COMPLEX(kind=real_8), DIMENSION(:), POINTER, CONTIGUOUS     :: array
# 153 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! array to verify and possibly resize
         COMPLEX(kind=real_8), DIMENSION(:), POINTER, OPTIONAL       :: array_resize
# 155 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER, INTENT(IN), OPTIONAL                  :: lb
        !! desired array lower bound
         INTEGER, INTENT(IN)                            :: ub
        !! desired array upper bound
         REAL(KIND=dp), INTENT(IN), OPTIONAL            :: factor
        !! factor by which to exaggerate enlargements
         LOGICAL, INTENT(IN), OPTIONAL                  :: nocopy, zero_pad
        !! copy array on enlargement; default is to copy
        !! zero new allocations; default is to write nothing
         TYPE(dbcsr_memtype_type), INTENT(IN), OPTIONAL :: memory_type
        !! use special memory

         CHARACTER(len=*), PARAMETER :: routineN = 'ensure_array_size_z', &
# 168 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                        routineP = moduleN//':'//routineN

         INTEGER                                  :: lb_new, lb_orig, &
                                                     ub_new, ub_orig, old_size, &
                                                     size_increase
         TYPE(dbcsr_memtype_type)                 :: mem_type
         LOGICAL                                  :: dbg, docopy, &
                                                     pad
         COMPLEX(kind=real_8), DIMENSION(:), POINTER, CONTIGUOUS    :: newarray
# 177 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

!   ---------------------------------------------------------------------------
         !CALL timeset(routineN, error_handler)
         dbg = .FALSE.

         IF (PRESENT(array_resize)) NULLIFY (array_resize)

         IF (PRESENT(nocopy)) THEN
            docopy = .NOT. nocopy
         ELSE
            docopy = .TRUE.
         END IF
         IF (PRESENT(memory_type)) THEN
            mem_type = memory_type
         ELSE
            mem_type = dbcsr_memtype_default
         END IF
         lb_new = 1
         IF (PRESENT(lb)) lb_new = lb
         pad = .FALSE.
         IF (PRESENT(zero_pad)) pad = zero_pad
         ! Creates a new array if it doesn't yet exist.
         IF (.NOT. ASSOCIATED(array)) THEN
            IF (lb_new /= 1) &
               DBCSR_ABORT("Arrays must start at 1")
            CALL mem_alloc_z (array, ub, mem_type=mem_type)
# 203 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            IF (pad .AND. ub .GT. 0) CALL mem_zero_z (array, ub)
# 204 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            !CALL timestop(error_handler)
            RETURN
         END IF
         lb_orig = LBOUND(array, 1)
         ub_orig = UBOUND(array, 1)
         old_size = ub_orig - lb_orig + 1
         ! The existing array is big enough.
         IF (lb_orig .LE. lb_new .AND. ub_orig .GE. ub) THEN
            !CALL timestop(error_handler)
            RETURN
         END IF
         ! A reallocation must be performed
         IF (dbg) WRITE (*, *) routineP//' Current bounds are', lb_orig, ':', ub_orig, &
            '; special?' !,mem_type
         !CALL timeset(routineN,timing_handle)
         IF (lb_orig .GT. lb_new) THEN
            IF (PRESENT(factor)) THEN
               size_increase = lb_orig - lb_new
               size_increase = MAX(NINT(size_increase*factor), &
                                   NINT(old_size*(factor - 1)), 0)
               lb_new = MIN(lb_orig, lb_new - size_increase)
            ELSE
               lb_new = lb_orig
            END IF
         END IF
         IF (ub_orig .LT. ub) THEN
            IF (PRESENT(factor)) THEN
               size_increase = ub - ub_orig
               size_increase = MAX(NINT(size_increase*factor), &
                                   NINT(old_size*(factor - 1)), 0)
               ub_new = MAX(ub_orig, ub + size_increase)
            ELSE
               ub_new = ub
            END IF
         ELSE
            ub_new = ub
         END IF
         IF (dbg) WRITE (*, *) routineP//' Resizing to bounds', lb_new, ':', ub_new
         !
         ! Deallocates the old array if it's not needed to copy the old data.
         IF (.NOT. docopy) THEN
            IF (PRESENT(array_resize)) THEN
               array_resize => array
               NULLIFY (array)
            ELSE
               CALL mem_dealloc_z (array, mem_type=mem_type)
# 250 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
         END IF
         !
         ! Allocates the new array
         IF (lb_new /= 1) &
            DBCSR_ABORT("Arrays must start at 1")
         CALL mem_alloc_z (newarray, ub_new - lb_new + 1, mem_type)
# 257 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         !
         ! Now copy and/or zero pad.
         IF (docopy) THEN
            IF (dbg .AND. (lb_new .GT. lb_orig .OR. ub_new .LT. ub_orig)) &
               DBCSR_ABORT("Old extent exceeds the new one.")
            IF (ub_orig - lb_orig + 1 .gt. 0) THEN
               !newarray(lb_orig:ub_orig) = array(lb_orig:ub_orig)
               CALL mem_copy_z (newarray(lb_orig:ub_orig), &
# 265 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                            array(lb_orig:ub_orig), ub_orig - lb_orig + 1)
            END IF
            IF (pad) THEN
               !newarray(lb_new:lb_orig-1) = 0
               CALL mem_zero_z (newarray(lb_new:lb_orig - 1), (lb_orig - 1) - lb_new + 1)
# 270 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
               !newarray(ub_orig+1:ub_new) = 0
               CALL mem_zero_z (newarray(ub_orig + 1:ub_new), ub_new - (ub_orig + 1) + 1)
# 272 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
            IF (PRESENT(array_resize)) THEN
               array_resize => array
               NULLIFY (array)
            ELSE
               CALL mem_dealloc_z (array, mem_type=mem_type)
# 278 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
         ELSEIF (pad) THEN
            !newarray(:) = CMPLX(0.0, 0.0, real_8)
# 281 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            CALL mem_zero_z (newarray, SIZE(newarray))
# 282 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         END IF
         array => newarray
         IF (dbg) WRITE (*, *) routineP//' New array size', SIZE(array)
         !CALL timestop(error_handler)
      END SUBROUTINE ensure_array_size_z
# 287 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_copy_z (dst, src, n)
# 289 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Copies memory area

         INTEGER, INTENT(IN) :: n
        !! length of copy
         COMPLEX(kind=real_8), DIMENSION(1:n), INTENT(OUT) :: dst
# 294 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! destination memory
         COMPLEX(kind=real_8), DIMENSION(1:n), INTENT(IN)  :: src
# 296 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! source memory
         dst(:) = src(:)
      END SUBROUTINE mem_copy_z
# 299 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_zero_z (dst, n)
# 301 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Zeros memory area

         INTEGER, INTENT(IN) :: n
        !! length of elements to zero
         COMPLEX(kind=real_8), DIMENSION(1:n), INTENT(OUT) :: dst
# 306 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! destination memory
         dst(:) = CMPLX(0.0, 0.0, real_8)
# 308 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
      END SUBROUTINE mem_zero_z
# 309 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_alloc_z (mem, n, mem_type)
# 311 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Allocates memory

         COMPLEX(kind=real_8), DIMENSION(:), POINTER, CONTIGUOUS :: mem
# 314 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         INTEGER, INTENT(IN)                   :: n
        !! length of elements to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_alloc_z'
# 320 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc .AND. n > 1) THEN
            CALL acc_hostmem_allocate(mem, n, mem_type%acc_stream)
         ELSE IF (mem_type%mpi .AND. dbcsr_cfg%use_mpi_allocator%val) THEN
            CALL mp_allocate(mem, n)
         ELSE
            ALLOCATE (mem(n))
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_alloc_z
# 337 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_alloc_z_2d(mem, sizes, mem_type)
# 339 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Allocates memory

         COMPLEX(kind=real_8), DIMENSION(:, :), POINTER      :: mem
# 342 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         INTEGER, DIMENSION(2), INTENT(IN)     :: sizes
        !! length of elements to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_alloc_z_2d'
# 348 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc) THEN
            DBCSR_ABORT("Accelerator hostalloc not supported for 2D arrays.")
            !CALL acc_hostmem_allocate(mem, n, mem_type%acc_stream)
         ELSE IF (mem_type%mpi) THEN
            DBCSR_ABORT("MPI allocate not supported for 2D arrays.")
            !CALL mp_allocate(mem, n)
         ELSE
            ALLOCATE (mem(sizes(1), sizes(2)))
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_alloc_z_2d
# 367 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_dealloc_z (mem, mem_type)
# 369 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Deallocates memory

         COMPLEX(kind=real_8), DIMENSION(:), POINTER, CONTIGUOUS :: mem
# 372 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_dealloc_z'
# 376 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc .AND. SIZE(mem) > 1) THEN
            CALL acc_hostmem_deallocate(mem, mem_type%acc_stream)
         ELSE IF (mem_type%mpi .AND. dbcsr_cfg%use_mpi_allocator%val) THEN
            CALL mp_deallocate(mem)
         ELSE
            DEALLOCATE (mem)
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_dealloc_z
# 393 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_dealloc_z_2d(mem, mem_type)
# 395 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Deallocates memory

         COMPLEX(kind=real_8), DIMENSION(:, :), POINTER      :: mem
# 398 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_dealloc_z'
# 402 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc) THEN
            DBCSR_ABORT("Accelerator host deallocate not supported for 2D arrays.")
            !CALL acc_hostmem_deallocate(mem, mem_type%acc_stream)
         ELSE IF (mem_type%mpi) THEN
            DBCSR_ABORT("MPI deallocate not supported for 2D arrays.")
            !CALL mp_deallocate(mem)
         ELSE
            DEALLOCATE (mem)
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_dealloc_z_2d
# 421 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE pointer_z_rank_remap2(r2p, d1, d2, r1p)
# 423 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Sets a rank-2 pointer to rank-1 data using Fortran 2003 pointer
     !! rank remapping.

         INTEGER, INTENT(IN)                      :: d1, d2
         COMPLEX(kind=real_8), DIMENSION(:, :), &
# 428 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            POINTER                                :: r2p
         COMPLEX(kind=real_8), DIMENSION(:), &
# 430 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            POINTER                                :: r1p

         r2p(1:d1, 1:d2) => r1p(1:d1*d2)
      END SUBROUTINE pointer_z_rank_remap2
# 434 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
# 136 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
      FUNCTION pointer_view_c (original, lb, ub) RESULT(view)
# 137 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Returns a pointer with different bounds.

         COMPLEX(kind=real_4), DIMENSION(:), POINTER :: original, view
# 140 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! original data pointer
        !! new pointer
         INTEGER, INTENT(IN)                  :: lb, ub
        !! lower and upper bound for the new pointer view
        !! lower and upper bound for the new pointer view
         view => original(lb:ub)
      END FUNCTION pointer_view_c
# 147 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE ensure_array_size_c (array, array_resize, lb, ub, factor, &
# 149 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                                  nocopy, memory_type, zero_pad)
     !! Ensures that an array is appropriately large.

         COMPLEX(kind=real_4), DIMENSION(:), POINTER, CONTIGUOUS     :: array
# 153 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! array to verify and possibly resize
         COMPLEX(kind=real_4), DIMENSION(:), POINTER, OPTIONAL       :: array_resize
# 155 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER, INTENT(IN), OPTIONAL                  :: lb
        !! desired array lower bound
         INTEGER, INTENT(IN)                            :: ub
        !! desired array upper bound
         REAL(KIND=dp), INTENT(IN), OPTIONAL            :: factor
        !! factor by which to exaggerate enlargements
         LOGICAL, INTENT(IN), OPTIONAL                  :: nocopy, zero_pad
        !! copy array on enlargement; default is to copy
        !! zero new allocations; default is to write nothing
         TYPE(dbcsr_memtype_type), INTENT(IN), OPTIONAL :: memory_type
        !! use special memory

         CHARACTER(len=*), PARAMETER :: routineN = 'ensure_array_size_c', &
# 168 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                        routineP = moduleN//':'//routineN

         INTEGER                                  :: lb_new, lb_orig, &
                                                     ub_new, ub_orig, old_size, &
                                                     size_increase
         TYPE(dbcsr_memtype_type)                 :: mem_type
         LOGICAL                                  :: dbg, docopy, &
                                                     pad
         COMPLEX(kind=real_4), DIMENSION(:), POINTER, CONTIGUOUS    :: newarray
# 177 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

!   ---------------------------------------------------------------------------
         !CALL timeset(routineN, error_handler)
         dbg = .FALSE.

         IF (PRESENT(array_resize)) NULLIFY (array_resize)

         IF (PRESENT(nocopy)) THEN
            docopy = .NOT. nocopy
         ELSE
            docopy = .TRUE.
         END IF
         IF (PRESENT(memory_type)) THEN
            mem_type = memory_type
         ELSE
            mem_type = dbcsr_memtype_default
         END IF
         lb_new = 1
         IF (PRESENT(lb)) lb_new = lb
         pad = .FALSE.
         IF (PRESENT(zero_pad)) pad = zero_pad
         ! Creates a new array if it doesn't yet exist.
         IF (.NOT. ASSOCIATED(array)) THEN
            IF (lb_new /= 1) &
               DBCSR_ABORT("Arrays must start at 1")
            CALL mem_alloc_c (array, ub, mem_type=mem_type)
# 203 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            IF (pad .AND. ub .GT. 0) CALL mem_zero_c (array, ub)
# 204 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            !CALL timestop(error_handler)
            RETURN
         END IF
         lb_orig = LBOUND(array, 1)
         ub_orig = UBOUND(array, 1)
         old_size = ub_orig - lb_orig + 1
         ! The existing array is big enough.
         IF (lb_orig .LE. lb_new .AND. ub_orig .GE. ub) THEN
            !CALL timestop(error_handler)
            RETURN
         END IF
         ! A reallocation must be performed
         IF (dbg) WRITE (*, *) routineP//' Current bounds are', lb_orig, ':', ub_orig, &
            '; special?' !,mem_type
         !CALL timeset(routineN,timing_handle)
         IF (lb_orig .GT. lb_new) THEN
            IF (PRESENT(factor)) THEN
               size_increase = lb_orig - lb_new
               size_increase = MAX(NINT(size_increase*factor), &
                                   NINT(old_size*(factor - 1)), 0)
               lb_new = MIN(lb_orig, lb_new - size_increase)
            ELSE
               lb_new = lb_orig
            END IF
         END IF
         IF (ub_orig .LT. ub) THEN
            IF (PRESENT(factor)) THEN
               size_increase = ub - ub_orig
               size_increase = MAX(NINT(size_increase*factor), &
                                   NINT(old_size*(factor - 1)), 0)
               ub_new = MAX(ub_orig, ub + size_increase)
            ELSE
               ub_new = ub
            END IF
         ELSE
            ub_new = ub
         END IF
         IF (dbg) WRITE (*, *) routineP//' Resizing to bounds', lb_new, ':', ub_new
         !
         ! Deallocates the old array if it's not needed to copy the old data.
         IF (.NOT. docopy) THEN
            IF (PRESENT(array_resize)) THEN
               array_resize => array
               NULLIFY (array)
            ELSE
               CALL mem_dealloc_c (array, mem_type=mem_type)
# 250 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
         END IF
         !
         ! Allocates the new array
         IF (lb_new /= 1) &
            DBCSR_ABORT("Arrays must start at 1")
         CALL mem_alloc_c (newarray, ub_new - lb_new + 1, mem_type)
# 257 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         !
         ! Now copy and/or zero pad.
         IF (docopy) THEN
            IF (dbg .AND. (lb_new .GT. lb_orig .OR. ub_new .LT. ub_orig)) &
               DBCSR_ABORT("Old extent exceeds the new one.")
            IF (ub_orig - lb_orig + 1 .gt. 0) THEN
               !newarray(lb_orig:ub_orig) = array(lb_orig:ub_orig)
               CALL mem_copy_c (newarray(lb_orig:ub_orig), &
# 265 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                            array(lb_orig:ub_orig), ub_orig - lb_orig + 1)
            END IF
            IF (pad) THEN
               !newarray(lb_new:lb_orig-1) = 0
               CALL mem_zero_c (newarray(lb_new:lb_orig - 1), (lb_orig - 1) - lb_new + 1)
# 270 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
               !newarray(ub_orig+1:ub_new) = 0
               CALL mem_zero_c (newarray(ub_orig + 1:ub_new), ub_new - (ub_orig + 1) + 1)
# 272 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
            IF (PRESENT(array_resize)) THEN
               array_resize => array
               NULLIFY (array)
            ELSE
               CALL mem_dealloc_c (array, mem_type=mem_type)
# 278 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
         ELSEIF (pad) THEN
            !newarray(:) = CMPLX(0.0, 0.0, real_4)
# 281 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            CALL mem_zero_c (newarray, SIZE(newarray))
# 282 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         END IF
         array => newarray
         IF (dbg) WRITE (*, *) routineP//' New array size', SIZE(array)
         !CALL timestop(error_handler)
      END SUBROUTINE ensure_array_size_c
# 287 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_copy_c (dst, src, n)
# 289 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Copies memory area

         INTEGER, INTENT(IN) :: n
        !! length of copy
         COMPLEX(kind=real_4), DIMENSION(1:n), INTENT(OUT) :: dst
# 294 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! destination memory
         COMPLEX(kind=real_4), DIMENSION(1:n), INTENT(IN)  :: src
# 296 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! source memory
         dst(:) = src(:)
      END SUBROUTINE mem_copy_c
# 299 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_zero_c (dst, n)
# 301 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Zeros memory area

         INTEGER, INTENT(IN) :: n
        !! length of elements to zero
         COMPLEX(kind=real_4), DIMENSION(1:n), INTENT(OUT) :: dst
# 306 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! destination memory
         dst(:) = CMPLX(0.0, 0.0, real_4)
# 308 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
      END SUBROUTINE mem_zero_c
# 309 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_alloc_c (mem, n, mem_type)
# 311 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Allocates memory

         COMPLEX(kind=real_4), DIMENSION(:), POINTER, CONTIGUOUS :: mem
# 314 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         INTEGER, INTENT(IN)                   :: n
        !! length of elements to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_alloc_c'
# 320 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc .AND. n > 1) THEN
            CALL acc_hostmem_allocate(mem, n, mem_type%acc_stream)
         ELSE IF (mem_type%mpi .AND. dbcsr_cfg%use_mpi_allocator%val) THEN
            CALL mp_allocate(mem, n)
         ELSE
            ALLOCATE (mem(n))
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_alloc_c
# 337 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_alloc_c_2d(mem, sizes, mem_type)
# 339 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Allocates memory

         COMPLEX(kind=real_4), DIMENSION(:, :), POINTER      :: mem
# 342 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         INTEGER, DIMENSION(2), INTENT(IN)     :: sizes
        !! length of elements to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_alloc_c_2d'
# 348 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc) THEN
            DBCSR_ABORT("Accelerator hostalloc not supported for 2D arrays.")
            !CALL acc_hostmem_allocate(mem, n, mem_type%acc_stream)
         ELSE IF (mem_type%mpi) THEN
            DBCSR_ABORT("MPI allocate not supported for 2D arrays.")
            !CALL mp_allocate(mem, n)
         ELSE
            ALLOCATE (mem(sizes(1), sizes(2)))
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_alloc_c_2d
# 367 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_dealloc_c (mem, mem_type)
# 369 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Deallocates memory

         COMPLEX(kind=real_4), DIMENSION(:), POINTER, CONTIGUOUS :: mem
# 372 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_dealloc_c'
# 376 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc .AND. SIZE(mem) > 1) THEN
            CALL acc_hostmem_deallocate(mem, mem_type%acc_stream)
         ELSE IF (mem_type%mpi .AND. dbcsr_cfg%use_mpi_allocator%val) THEN
            CALL mp_deallocate(mem)
         ELSE
            DEALLOCATE (mem)
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_dealloc_c
# 393 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_dealloc_c_2d(mem, mem_type)
# 395 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Deallocates memory

         COMPLEX(kind=real_4), DIMENSION(:, :), POINTER      :: mem
# 398 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_dealloc_c'
# 402 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc) THEN
            DBCSR_ABORT("Accelerator host deallocate not supported for 2D arrays.")
            !CALL acc_hostmem_deallocate(mem, mem_type%acc_stream)
         ELSE IF (mem_type%mpi) THEN
            DBCSR_ABORT("MPI deallocate not supported for 2D arrays.")
            !CALL mp_deallocate(mem)
         ELSE
            DEALLOCATE (mem)
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_dealloc_c_2d
# 421 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE pointer_c_rank_remap2(r2p, d1, d2, r1p)
# 423 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Sets a rank-2 pointer to rank-1 data using Fortran 2003 pointer
     !! rank remapping.

         INTEGER, INTENT(IN)                      :: d1, d2
         COMPLEX(kind=real_4), DIMENSION(:, :), &
# 428 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            POINTER                                :: r2p
         COMPLEX(kind=real_4), DIMENSION(:), &
# 430 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            POINTER                                :: r1p

         r2p(1:d1, 1:d2) => r1p(1:d1*d2)
      END SUBROUTINE pointer_c_rank_remap2
# 434 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
# 136 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
      FUNCTION pointer_view_i (original, lb, ub) RESULT(view)
# 137 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Returns a pointer with different bounds.

         INTEGER(kind=int_4), DIMENSION(:), POINTER :: original, view
# 140 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! original data pointer
        !! new pointer
         INTEGER, INTENT(IN)                  :: lb, ub
        !! lower and upper bound for the new pointer view
        !! lower and upper bound for the new pointer view
         view => original(lb:ub)
      END FUNCTION pointer_view_i
# 147 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE ensure_array_size_i (array, array_resize, lb, ub, factor, &
# 149 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                                  nocopy, memory_type, zero_pad)
     !! Ensures that an array is appropriately large.

         INTEGER(kind=int_4), DIMENSION(:), POINTER, CONTIGUOUS     :: array
# 153 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! array to verify and possibly resize
         INTEGER(kind=int_4), DIMENSION(:), POINTER, OPTIONAL       :: array_resize
# 155 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER, INTENT(IN), OPTIONAL                  :: lb
        !! desired array lower bound
         INTEGER, INTENT(IN)                            :: ub
        !! desired array upper bound
         REAL(KIND=dp), INTENT(IN), OPTIONAL            :: factor
        !! factor by which to exaggerate enlargements
         LOGICAL, INTENT(IN), OPTIONAL                  :: nocopy, zero_pad
        !! copy array on enlargement; default is to copy
        !! zero new allocations; default is to write nothing
         TYPE(dbcsr_memtype_type), INTENT(IN), OPTIONAL :: memory_type
        !! use special memory

         CHARACTER(len=*), PARAMETER :: routineN = 'ensure_array_size_i', &
# 168 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                        routineP = moduleN//':'//routineN

         INTEGER                                  :: lb_new, lb_orig, &
                                                     ub_new, ub_orig, old_size, &
                                                     size_increase
         TYPE(dbcsr_memtype_type)                 :: mem_type
         LOGICAL                                  :: dbg, docopy, &
                                                     pad
         INTEGER(kind=int_4), DIMENSION(:), POINTER, CONTIGUOUS    :: newarray
# 177 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

!   ---------------------------------------------------------------------------
         !CALL timeset(routineN, error_handler)
         dbg = .FALSE.

         IF (PRESENT(array_resize)) NULLIFY (array_resize)

         IF (PRESENT(nocopy)) THEN
            docopy = .NOT. nocopy
         ELSE
            docopy = .TRUE.
         END IF
         IF (PRESENT(memory_type)) THEN
            mem_type = memory_type
         ELSE
            mem_type = dbcsr_memtype_default
         END IF
         lb_new = 1
         IF (PRESENT(lb)) lb_new = lb
         pad = .FALSE.
         IF (PRESENT(zero_pad)) pad = zero_pad
         ! Creates a new array if it doesn't yet exist.
         IF (.NOT. ASSOCIATED(array)) THEN
            IF (lb_new /= 1) &
               DBCSR_ABORT("Arrays must start at 1")
            CALL mem_alloc_i (array, ub, mem_type=mem_type)
# 203 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            IF (pad .AND. ub .GT. 0) CALL mem_zero_i (array, ub)
# 204 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            !CALL timestop(error_handler)
            RETURN
         END IF
         lb_orig = LBOUND(array, 1)
         ub_orig = UBOUND(array, 1)
         old_size = ub_orig - lb_orig + 1
         ! The existing array is big enough.
         IF (lb_orig .LE. lb_new .AND. ub_orig .GE. ub) THEN
            !CALL timestop(error_handler)
            RETURN
         END IF
         ! A reallocation must be performed
         IF (dbg) WRITE (*, *) routineP//' Current bounds are', lb_orig, ':', ub_orig, &
            '; special?' !,mem_type
         !CALL timeset(routineN,timing_handle)
         IF (lb_orig .GT. lb_new) THEN
            IF (PRESENT(factor)) THEN
               size_increase = lb_orig - lb_new
               size_increase = MAX(NINT(size_increase*factor), &
                                   NINT(old_size*(factor - 1)), 0)
               lb_new = MIN(lb_orig, lb_new - size_increase)
            ELSE
               lb_new = lb_orig
            END IF
         END IF
         IF (ub_orig .LT. ub) THEN
            IF (PRESENT(factor)) THEN
               size_increase = ub - ub_orig
               size_increase = MAX(NINT(size_increase*factor), &
                                   NINT(old_size*(factor - 1)), 0)
               ub_new = MAX(ub_orig, ub + size_increase)
            ELSE
               ub_new = ub
            END IF
         ELSE
            ub_new = ub
         END IF
         IF (dbg) WRITE (*, *) routineP//' Resizing to bounds', lb_new, ':', ub_new
         !
         ! Deallocates the old array if it's not needed to copy the old data.
         IF (.NOT. docopy) THEN
            IF (PRESENT(array_resize)) THEN
               array_resize => array
               NULLIFY (array)
            ELSE
               CALL mem_dealloc_i (array, mem_type=mem_type)
# 250 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
         END IF
         !
         ! Allocates the new array
         IF (lb_new /= 1) &
            DBCSR_ABORT("Arrays must start at 1")
         CALL mem_alloc_i (newarray, ub_new - lb_new + 1, mem_type)
# 257 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         !
         ! Now copy and/or zero pad.
         IF (docopy) THEN
            IF (dbg .AND. (lb_new .GT. lb_orig .OR. ub_new .LT. ub_orig)) &
               DBCSR_ABORT("Old extent exceeds the new one.")
            IF (ub_orig - lb_orig + 1 .gt. 0) THEN
               !newarray(lb_orig:ub_orig) = array(lb_orig:ub_orig)
               CALL mem_copy_i (newarray(lb_orig:ub_orig), &
# 265 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                            array(lb_orig:ub_orig), ub_orig - lb_orig + 1)
            END IF
            IF (pad) THEN
               !newarray(lb_new:lb_orig-1) = 0
               CALL mem_zero_i (newarray(lb_new:lb_orig - 1), (lb_orig - 1) - lb_new + 1)
# 270 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
               !newarray(ub_orig+1:ub_new) = 0
               CALL mem_zero_i (newarray(ub_orig + 1:ub_new), ub_new - (ub_orig + 1) + 1)
# 272 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
            IF (PRESENT(array_resize)) THEN
               array_resize => array
               NULLIFY (array)
            ELSE
               CALL mem_dealloc_i (array, mem_type=mem_type)
# 278 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
         ELSEIF (pad) THEN
            !newarray(:) = 0
# 281 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            CALL mem_zero_i (newarray, SIZE(newarray))
# 282 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         END IF
         array => newarray
         IF (dbg) WRITE (*, *) routineP//' New array size', SIZE(array)
         !CALL timestop(error_handler)
      END SUBROUTINE ensure_array_size_i
# 287 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_copy_i (dst, src, n)
# 289 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Copies memory area

         INTEGER, INTENT(IN) :: n
        !! length of copy
         INTEGER(kind=int_4), DIMENSION(1:n), INTENT(OUT) :: dst
# 294 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! destination memory
         INTEGER(kind=int_4), DIMENSION(1:n), INTENT(IN)  :: src
# 296 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! source memory
         dst(:) = src(:)
      END SUBROUTINE mem_copy_i
# 299 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_zero_i (dst, n)
# 301 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Zeros memory area

         INTEGER, INTENT(IN) :: n
        !! length of elements to zero
         INTEGER(kind=int_4), DIMENSION(1:n), INTENT(OUT) :: dst
# 306 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! destination memory
         dst(:) = 0
# 308 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
      END SUBROUTINE mem_zero_i
# 309 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_alloc_i (mem, n, mem_type)
# 311 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Allocates memory

         INTEGER(kind=int_4), DIMENSION(:), POINTER, CONTIGUOUS :: mem
# 314 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         INTEGER, INTENT(IN)                   :: n
        !! length of elements to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_alloc_i'
# 320 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc .AND. n > 1) THEN
            CALL acc_hostmem_allocate(mem, n, mem_type%acc_stream)
         ELSE IF (mem_type%mpi .AND. dbcsr_cfg%use_mpi_allocator%val) THEN
            CALL mp_allocate(mem, n)
         ELSE
            ALLOCATE (mem(n))
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_alloc_i
# 337 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_alloc_i_2d(mem, sizes, mem_type)
# 339 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Allocates memory

         INTEGER(kind=int_4), DIMENSION(:, :), POINTER      :: mem
# 342 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         INTEGER, DIMENSION(2), INTENT(IN)     :: sizes
        !! length of elements to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_alloc_i_2d'
# 348 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc) THEN
            DBCSR_ABORT("Accelerator hostalloc not supported for 2D arrays.")
            !CALL acc_hostmem_allocate(mem, n, mem_type%acc_stream)
         ELSE IF (mem_type%mpi) THEN
            DBCSR_ABORT("MPI allocate not supported for 2D arrays.")
            !CALL mp_allocate(mem, n)
         ELSE
            ALLOCATE (mem(sizes(1), sizes(2)))
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_alloc_i_2d
# 367 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_dealloc_i (mem, mem_type)
# 369 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Deallocates memory

         INTEGER(kind=int_4), DIMENSION(:), POINTER, CONTIGUOUS :: mem
# 372 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_dealloc_i'
# 376 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc .AND. SIZE(mem) > 1) THEN
            CALL acc_hostmem_deallocate(mem, mem_type%acc_stream)
         ELSE IF (mem_type%mpi .AND. dbcsr_cfg%use_mpi_allocator%val) THEN
            CALL mp_deallocate(mem)
         ELSE
            DEALLOCATE (mem)
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_dealloc_i
# 393 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_dealloc_i_2d(mem, mem_type)
# 395 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Deallocates memory

         INTEGER(kind=int_4), DIMENSION(:, :), POINTER      :: mem
# 398 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_dealloc_i'
# 402 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc) THEN
            DBCSR_ABORT("Accelerator host deallocate not supported for 2D arrays.")
            !CALL acc_hostmem_deallocate(mem, mem_type%acc_stream)
         ELSE IF (mem_type%mpi) THEN
            DBCSR_ABORT("MPI deallocate not supported for 2D arrays.")
            !CALL mp_deallocate(mem)
         ELSE
            DEALLOCATE (mem)
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_dealloc_i_2d
# 421 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE pointer_i_rank_remap2(r2p, d1, d2, r1p)
# 423 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Sets a rank-2 pointer to rank-1 data using Fortran 2003 pointer
     !! rank remapping.

         INTEGER, INTENT(IN)                      :: d1, d2
         INTEGER(kind=int_4), DIMENSION(:, :), &
# 428 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            POINTER                                :: r2p
         INTEGER(kind=int_4), DIMENSION(:), &
# 430 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            POINTER                                :: r1p

         r2p(1:d1, 1:d2) => r1p(1:d1*d2)
      END SUBROUTINE pointer_i_rank_remap2
# 434 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
# 136 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
      FUNCTION pointer_view_l (original, lb, ub) RESULT(view)
# 137 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Returns a pointer with different bounds.

         INTEGER(kind=int_8), DIMENSION(:), POINTER :: original, view
# 140 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! original data pointer
        !! new pointer
         INTEGER, INTENT(IN)                  :: lb, ub
        !! lower and upper bound for the new pointer view
        !! lower and upper bound for the new pointer view
         view => original(lb:ub)
      END FUNCTION pointer_view_l
# 147 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE ensure_array_size_l (array, array_resize, lb, ub, factor, &
# 149 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                                  nocopy, memory_type, zero_pad)
     !! Ensures that an array is appropriately large.

         INTEGER(kind=int_8), DIMENSION(:), POINTER, CONTIGUOUS     :: array
# 153 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! array to verify and possibly resize
         INTEGER(kind=int_8), DIMENSION(:), POINTER, OPTIONAL       :: array_resize
# 155 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER, INTENT(IN), OPTIONAL                  :: lb
        !! desired array lower bound
         INTEGER, INTENT(IN)                            :: ub
        !! desired array upper bound
         REAL(KIND=dp), INTENT(IN), OPTIONAL            :: factor
        !! factor by which to exaggerate enlargements
         LOGICAL, INTENT(IN), OPTIONAL                  :: nocopy, zero_pad
        !! copy array on enlargement; default is to copy
        !! zero new allocations; default is to write nothing
         TYPE(dbcsr_memtype_type), INTENT(IN), OPTIONAL :: memory_type
        !! use special memory

         CHARACTER(len=*), PARAMETER :: routineN = 'ensure_array_size_l', &
# 168 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                        routineP = moduleN//':'//routineN

         INTEGER                                  :: lb_new, lb_orig, &
                                                     ub_new, ub_orig, old_size, &
                                                     size_increase
         TYPE(dbcsr_memtype_type)                 :: mem_type
         LOGICAL                                  :: dbg, docopy, &
                                                     pad
         INTEGER(kind=int_8), DIMENSION(:), POINTER, CONTIGUOUS    :: newarray
# 177 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

!   ---------------------------------------------------------------------------
         !CALL timeset(routineN, error_handler)
         dbg = .FALSE.

         IF (PRESENT(array_resize)) NULLIFY (array_resize)

         IF (PRESENT(nocopy)) THEN
            docopy = .NOT. nocopy
         ELSE
            docopy = .TRUE.
         END IF
         IF (PRESENT(memory_type)) THEN
            mem_type = memory_type
         ELSE
            mem_type = dbcsr_memtype_default
         END IF
         lb_new = 1
         IF (PRESENT(lb)) lb_new = lb
         pad = .FALSE.
         IF (PRESENT(zero_pad)) pad = zero_pad
         ! Creates a new array if it doesn't yet exist.
         IF (.NOT. ASSOCIATED(array)) THEN
            IF (lb_new /= 1) &
               DBCSR_ABORT("Arrays must start at 1")
            CALL mem_alloc_l (array, ub, mem_type=mem_type)
# 203 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            IF (pad .AND. ub .GT. 0) CALL mem_zero_l (array, ub)
# 204 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            !CALL timestop(error_handler)
            RETURN
         END IF
         lb_orig = LBOUND(array, 1)
         ub_orig = UBOUND(array, 1)
         old_size = ub_orig - lb_orig + 1
         ! The existing array is big enough.
         IF (lb_orig .LE. lb_new .AND. ub_orig .GE. ub) THEN
            !CALL timestop(error_handler)
            RETURN
         END IF
         ! A reallocation must be performed
         IF (dbg) WRITE (*, *) routineP//' Current bounds are', lb_orig, ':', ub_orig, &
            '; special?' !,mem_type
         !CALL timeset(routineN,timing_handle)
         IF (lb_orig .GT. lb_new) THEN
            IF (PRESENT(factor)) THEN
               size_increase = lb_orig - lb_new
               size_increase = MAX(NINT(size_increase*factor), &
                                   NINT(old_size*(factor - 1)), 0)
               lb_new = MIN(lb_orig, lb_new - size_increase)
            ELSE
               lb_new = lb_orig
            END IF
         END IF
         IF (ub_orig .LT. ub) THEN
            IF (PRESENT(factor)) THEN
               size_increase = ub - ub_orig
               size_increase = MAX(NINT(size_increase*factor), &
                                   NINT(old_size*(factor - 1)), 0)
               ub_new = MAX(ub_orig, ub + size_increase)
            ELSE
               ub_new = ub
            END IF
         ELSE
            ub_new = ub
         END IF
         IF (dbg) WRITE (*, *) routineP//' Resizing to bounds', lb_new, ':', ub_new
         !
         ! Deallocates the old array if it's not needed to copy the old data.
         IF (.NOT. docopy) THEN
            IF (PRESENT(array_resize)) THEN
               array_resize => array
               NULLIFY (array)
            ELSE
               CALL mem_dealloc_l (array, mem_type=mem_type)
# 250 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
         END IF
         !
         ! Allocates the new array
         IF (lb_new /= 1) &
            DBCSR_ABORT("Arrays must start at 1")
         CALL mem_alloc_l (newarray, ub_new - lb_new + 1, mem_type)
# 257 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         !
         ! Now copy and/or zero pad.
         IF (docopy) THEN
            IF (dbg .AND. (lb_new .GT. lb_orig .OR. ub_new .LT. ub_orig)) &
               DBCSR_ABORT("Old extent exceeds the new one.")
            IF (ub_orig - lb_orig + 1 .gt. 0) THEN
               !newarray(lb_orig:ub_orig) = array(lb_orig:ub_orig)
               CALL mem_copy_l (newarray(lb_orig:ub_orig), &
# 265 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
                                            array(lb_orig:ub_orig), ub_orig - lb_orig + 1)
            END IF
            IF (pad) THEN
               !newarray(lb_new:lb_orig-1) = 0
               CALL mem_zero_l (newarray(lb_new:lb_orig - 1), (lb_orig - 1) - lb_new + 1)
# 270 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
               !newarray(ub_orig+1:ub_new) = 0
               CALL mem_zero_l (newarray(ub_orig + 1:ub_new), ub_new - (ub_orig + 1) + 1)
# 272 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
            IF (PRESENT(array_resize)) THEN
               array_resize => array
               NULLIFY (array)
            ELSE
               CALL mem_dealloc_l (array, mem_type=mem_type)
# 278 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            END IF
         ELSEIF (pad) THEN
            !newarray(:) = 0
# 281 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            CALL mem_zero_l (newarray, SIZE(newarray))
# 282 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         END IF
         array => newarray
         IF (dbg) WRITE (*, *) routineP//' New array size', SIZE(array)
         !CALL timestop(error_handler)
      END SUBROUTINE ensure_array_size_l
# 287 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_copy_l (dst, src, n)
# 289 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Copies memory area

         INTEGER, INTENT(IN) :: n
        !! length of copy
         INTEGER(kind=int_8), DIMENSION(1:n), INTENT(OUT) :: dst
# 294 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! destination memory
         INTEGER(kind=int_8), DIMENSION(1:n), INTENT(IN)  :: src
# 296 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! source memory
         dst(:) = src(:)
      END SUBROUTINE mem_copy_l
# 299 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_zero_l (dst, n)
# 301 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Zeros memory area

         INTEGER, INTENT(IN) :: n
        !! length of elements to zero
         INTEGER(kind=int_8), DIMENSION(1:n), INTENT(OUT) :: dst
# 306 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! destination memory
         dst(:) = 0
# 308 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
      END SUBROUTINE mem_zero_l
# 309 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_alloc_l (mem, n, mem_type)
# 311 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Allocates memory

         INTEGER(kind=int_8), DIMENSION(:), POINTER, CONTIGUOUS :: mem
# 314 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         INTEGER, INTENT(IN)                   :: n
        !! length of elements to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_alloc_l'
# 320 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc .AND. n > 1) THEN
            CALL acc_hostmem_allocate(mem, n, mem_type%acc_stream)
         ELSE IF (mem_type%mpi .AND. dbcsr_cfg%use_mpi_allocator%val) THEN
            CALL mp_allocate(mem, n)
         ELSE
            ALLOCATE (mem(n))
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_alloc_l
# 337 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_alloc_l_2d(mem, sizes, mem_type)
# 339 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Allocates memory

         INTEGER(kind=int_8), DIMENSION(:, :), POINTER      :: mem
# 342 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         INTEGER, DIMENSION(2), INTENT(IN)     :: sizes
        !! length of elements to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_alloc_l_2d'
# 348 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc) THEN
            DBCSR_ABORT("Accelerator hostalloc not supported for 2D arrays.")
            !CALL acc_hostmem_allocate(mem, n, mem_type%acc_stream)
         ELSE IF (mem_type%mpi) THEN
            DBCSR_ABORT("MPI allocate not supported for 2D arrays.")
            !CALL mp_allocate(mem, n)
         ELSE
            ALLOCATE (mem(sizes(1), sizes(2)))
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_alloc_l_2d
# 367 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_dealloc_l (mem, mem_type)
# 369 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Deallocates memory

         INTEGER(kind=int_8), DIMENSION(:), POINTER, CONTIGUOUS :: mem
# 372 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_dealloc_l'
# 376 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc .AND. SIZE(mem) > 1) THEN
            CALL acc_hostmem_deallocate(mem, mem_type%acc_stream)
         ELSE IF (mem_type%mpi .AND. dbcsr_cfg%use_mpi_allocator%val) THEN
            CALL mp_deallocate(mem)
         ELSE
            DEALLOCATE (mem)
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_dealloc_l
# 393 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE mem_dealloc_l_2d(mem, mem_type)
# 395 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Deallocates memory

         INTEGER(kind=int_8), DIMENSION(:, :), POINTER      :: mem
# 398 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
        !! memory to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_dealloc_l'
# 402 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
         INTEGER                               :: error_handle
!   ---------------------------------------------------------------------------

         IF (careful_mod) &
            CALL timeset(routineN, error_handle)

         IF (mem_type%acc_hostalloc) THEN
            DBCSR_ABORT("Accelerator host deallocate not supported for 2D arrays.")
            !CALL acc_hostmem_deallocate(mem, mem_type%acc_stream)
         ELSE IF (mem_type%mpi) THEN
            DBCSR_ABORT("MPI deallocate not supported for 2D arrays.")
            !CALL mp_deallocate(mem)
         ELSE
            DEALLOCATE (mem)
         END IF

         IF (careful_mod) &
            CALL timestop(error_handle)
      END SUBROUTINE mem_dealloc_l_2d
# 421 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

      SUBROUTINE pointer_l_rank_remap2(r2p, d1, d2, r1p)
# 423 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
     !! Sets a rank-2 pointer to rank-1 data using Fortran 2003 pointer
     !! rank remapping.

         INTEGER, INTENT(IN)                      :: d1, d2
         INTEGER(kind=int_8), DIMENSION(:, :), &
# 428 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            POINTER                                :: r2p
         INTEGER(kind=int_8), DIMENSION(:), &
# 430 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
            POINTER                                :: r1p

         r2p(1:d1, 1:d2) => r1p(1:d1*d2)
      END SUBROUTINE pointer_l_rank_remap2
# 434 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"
# 435 "/__w/dbcsr/dbcsr/src/data/dbcsr_ptr_util.F"

END MODULE dbcsr_ptr_util
