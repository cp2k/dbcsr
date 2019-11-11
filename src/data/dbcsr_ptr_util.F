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

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads

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

   TYPE dbcsr_data_allocation_type
      LOGICAL :: use_mpi_allocator = .TRUE.
   END TYPE dbcsr_data_allocation_type

   PUBLIC :: dbcsr_data_allocation_type
   TYPE(dbcsr_data_allocation_type), PUBLIC, SAVE :: dbcsr_data_allocation = dbcsr_data_allocation_type() ! defaults

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

      CHARACTER(len=*), PARAMETER :: routineN = 'pointer_view_a', routineP = moduleN//':'//routineN

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
      ENDIF
      narea2 = new_area
   END FUNCTION pointer_view_a

#:include "dbcsr_ptr_util.f90"

END MODULE dbcsr_ptr_util
