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

!$ USE OMP_LIB, ONLY: omp_in_parallel

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

   #:include 'dbcsr.fypp'
   #:for nametype1, type1, zero1 in inst_params_all
      FUNCTION pointer_view_${nametype1}$ (original, lb, ub) RESULT(view)
     !! Returns a pointer with different bounds.

         ${type1}$, DIMENSION(:), POINTER :: original, view
        !! original data pointer
        !! new pointer
         INTEGER, INTENT(IN)                  :: lb, ub
        !! lower and upper bound for the new pointer view
        !! lower and upper bound for the new pointer view
         view => original(lb:ub)
      END FUNCTION pointer_view_${nametype1}$

      SUBROUTINE ensure_array_size_${nametype1}$ (array, array_resize, lb, ub, factor, &
                                                  nocopy, memory_type, zero_pad)
     !! Ensures that an array is appropriately large.

         ${type1}$, DIMENSION(:), POINTER, CONTIGUOUS     :: array
        !! array to verify and possibly resize
         ${type1}$, DIMENSION(:), POINTER, OPTIONAL       :: array_resize
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

         CHARACTER(len=*), PARAMETER :: routineN = 'ensure_array_size_${nametype1}$', &
                                        routineP = moduleN//':'//routineN

         INTEGER                                  :: lb_new, lb_orig, &
                                                     ub_new, ub_orig, old_size, &
                                                     size_increase
         TYPE(dbcsr_memtype_type)                 :: mem_type
         LOGICAL                                  :: dbg, docopy, &
                                                     pad
         ${type1}$, DIMENSION(:), POINTER, CONTIGUOUS    :: newarray

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
            CALL mem_alloc_${nametype1}$ (array, ub, mem_type=mem_type)
            IF (pad .AND. ub .GT. 0) CALL mem_zero_${nametype1}$ (array, ub)
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
               CALL mem_dealloc_${nametype1}$ (array, mem_type=mem_type)
            END IF
         END IF
         !
         ! Allocates the new array
         IF (lb_new /= 1) &
            DBCSR_ABORT("Arrays must start at 1")
         CALL mem_alloc_${nametype1}$ (newarray, ub_new - lb_new + 1, mem_type)
         !
         ! Now copy and/or zero pad.
         IF (docopy) THEN
            IF (dbg .AND. (lb_new .GT. lb_orig .OR. ub_new .LT. ub_orig)) &
               DBCSR_ABORT("Old extent exceeds the new one.")
            IF (ub_orig - lb_orig + 1 .gt. 0) THEN
               !newarray(lb_orig:ub_orig) = array(lb_orig:ub_orig)
               CALL mem_copy_${nametype1}$ (newarray(lb_orig:ub_orig), &
                                            array(lb_orig:ub_orig), ub_orig - lb_orig + 1)
            END IF
            IF (pad) THEN
               !newarray(lb_new:lb_orig-1) = 0
               CALL mem_zero_${nametype1}$ (newarray(lb_new:lb_orig - 1), (lb_orig - 1) - lb_new + 1)
               !newarray(ub_orig+1:ub_new) = 0
               CALL mem_zero_${nametype1}$ (newarray(ub_orig + 1:ub_new), ub_new - (ub_orig + 1) + 1)
            END IF
            IF (PRESENT(array_resize)) THEN
               array_resize => array
               NULLIFY (array)
            ELSE
               CALL mem_dealloc_${nametype1}$ (array, mem_type=mem_type)
            END IF
         ELSEIF (pad) THEN
            !newarray(:) = ${zero1}$
            CALL mem_zero_${nametype1}$ (newarray, SIZE(newarray))
         END IF
         array => newarray
         IF (dbg) WRITE (*, *) routineP//' New array size', SIZE(array)
         !CALL timestop(error_handler)
      END SUBROUTINE ensure_array_size_${nametype1}$

      SUBROUTINE mem_copy_${nametype1}$ (dst, src, n)
     !! Copies memory area

         INTEGER, INTENT(IN) :: n
        !! length of copy
         ${type1}$, DIMENSION(1:n), INTENT(OUT) :: dst
        !! destination memory
         ${type1}$, DIMENSION(1:n), INTENT(IN)  :: src
        !! source memory
#if !defined(__DBCSR_DISABLE_WORKSHARE)
!$       IF (.NOT. omp_in_parallel()) THEN
!$OMP       PARALLEL WORKSHARE DEFAULT(none) SHARED(dst,src)
            dst(:) = src(:)
!$OMP       END PARALLEL WORKSHARE
!$       ELSE
#endif
            dst(:) = src(:)
#if !defined(__DBCSR_DISABLE_WORKSHARE)
!$       END IF
#endif
      END SUBROUTINE mem_copy_${nametype1}$

      SUBROUTINE mem_zero_${nametype1}$ (dst, n)
     !! Zeros memory area

         INTEGER, INTENT(IN) :: n
        !! length of elements to zero
         ${type1}$, DIMENSION(1:n), INTENT(OUT) :: dst
        !! destination memory
#if !defined(__DBCSR_DISABLE_WORKSHARE)
!$       IF (.NOT. omp_in_parallel()) THEN
!$OMP       PARALLEL WORKSHARE DEFAULT(none) SHARED(dst)
            dst(:) = ${zero1}$
!$OMP       END PARALLEL WORKSHARE
!$       ELSE
#endif
            dst(:) = ${zero1}$
#if !defined(__DBCSR_DISABLE_WORKSHARE)
!$       END IF
#endif
      END SUBROUTINE mem_zero_${nametype1}$

      SUBROUTINE mem_alloc_${nametype1}$ (mem, n, mem_type)
     !! Allocates memory

         ${type1}$, DIMENSION(:), POINTER, CONTIGUOUS :: mem
        !! memory to allocate
         INTEGER, INTENT(IN)                   :: n
        !! length of elements to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_alloc_${nametype1}$'
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
      END SUBROUTINE mem_alloc_${nametype1}$

      SUBROUTINE mem_alloc_${nametype1}$_2d(mem, sizes, mem_type)
     !! Allocates memory

         ${type1}$, DIMENSION(:, :), POINTER      :: mem
        !! memory to allocate
         INTEGER, DIMENSION(2), INTENT(IN)     :: sizes
        !! length of elements to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_alloc_${nametype1}$_2d'
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
      END SUBROUTINE mem_alloc_${nametype1}$_2d

      SUBROUTINE mem_dealloc_${nametype1}$ (mem, mem_type)
     !! Deallocates memory

         ${type1}$, DIMENSION(:), POINTER, CONTIGUOUS :: mem
        !! memory to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_dealloc_${nametype1}$'
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
      END SUBROUTINE mem_dealloc_${nametype1}$

      SUBROUTINE mem_dealloc_${nametype1}$_2d(mem, mem_type)
     !! Deallocates memory

         ${type1}$, DIMENSION(:, :), POINTER      :: mem
        !! memory to allocate
         TYPE(dbcsr_memtype_type), INTENT(IN)  :: mem_type
        !! memory type
         CHARACTER(len=*), PARAMETER :: routineN = 'mem_dealloc_${nametype1}$'
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
      END SUBROUTINE mem_dealloc_${nametype1}$_2d

      SUBROUTINE pointer_${nametype1}$_rank_remap2(r2p, d1, d2, r1p)
     !! Sets a rank-2 pointer to rank-1 data using Fortran 2003 pointer
     !! rank remapping.

         INTEGER, INTENT(IN)                      :: d1, d2
         ${type1}$, DIMENSION(:, :), &
            POINTER                                :: r2p
         ${type1}$, DIMENSION(:), &
            POINTER                                :: r1p

         r2p(1:d1, 1:d2) => r1p(1:d1*d2)
      END SUBROUTINE pointer_${nametype1}$_rank_remap2
   #:endfor

END MODULE dbcsr_ptr_util
