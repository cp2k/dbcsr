!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_data_types
   !! Types related to DBCSR data area
   USE dbcsr_acc_devmem, ONLY: acc_devmem_type
   USE dbcsr_acc_event, ONLY: acc_event_type
   USE dbcsr_acc_stream, ONLY: acc_stream_type
   USE dbcsr_kinds, ONLY: &
      dp, int_4, int_4_size, int_8, int_8_size, real_4, real_4_size, real_8, real_8_size

!$ USE OMP_LIB, ONLY: omp_get_max_threads, omp_get_thread_num, omp_get_num_threads, omp_lock_kind

#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_data_types'

   PUBLIC :: dbcsr_data_obj, dbcsr_data_area_type, dbcsr_scalar_type
   PUBLIC :: dbcsr_datatype_sizeof
   PUBLIC :: dbcsr_mempool_type, &
             dbcsr_mempool_entry_type, &
             dbcsr_memtype_type, &
             dbcsr_memtype_default
   PUBLIC :: dbcsr_type_real_4, dbcsr_type_real_8, &
             dbcsr_type_complex_4, dbcsr_type_complex_8, &
             dbcsr_type_real_default, dbcsr_type_complex_default, &
             dbcsr_type_real_4_2d, dbcsr_type_real_8_2d, &
             dbcsr_type_complex_4_2d, dbcsr_type_complex_8_2d, &
             dbcsr_type_int_4, dbcsr_type_int_8

   TYPE dbcsr_scalar_type
      !! Stores a scalar in any of the supported data types.
      !!
      !! Reasoning
      !! Easier routine interfaces

      REAL(KIND=real_4)    :: r_sp
         !! stores real values in single precision
      REAL(KIND=real_8)    :: r_dp
         !! stores real values in double precision
      COMPLEX(KIND=real_4) :: c_sp
         !! stores complex values in single precision
      COMPLEX(KIND=real_8) :: c_dp
         !! stores complex values in double precision
      INTEGER              :: data_type
         !! which of the data types is actually used
   END TYPE dbcsr_scalar_type

   TYPE dbcsr_data_obj
      TYPE(dbcsr_data_area_type), POINTER           :: d => Null()
   END TYPE dbcsr_data_obj

   TYPE dbcsr_mempool_type
      !! Memory related types
      TYPE(dbcsr_mempool_entry_type), POINTER :: root => Null()
      INTEGER                                 :: capacity = 1
!$    INTEGER(KIND=omp_lock_kind)          :: lock
   END TYPE dbcsr_mempool_type

   TYPE dbcsr_mempool_entry_type
      TYPE(dbcsr_data_obj)                    :: area
      TYPE(dbcsr_mempool_entry_type), POINTER :: next => Null()
   END TYPE dbcsr_mempool_entry_type

   TYPE dbcsr_memtype_type
      LOGICAL                           :: mpi = .FALSE.
      LOGICAL                           :: acc_hostalloc = .FALSE.
      LOGICAL                           :: acc_devalloc = .FALSE.
      TYPE(acc_stream_type)             :: acc_stream = acc_stream_type()
      TYPE(dbcsr_mempool_type), POINTER :: pool => Null()
      REAL(KIND=dp)                     :: oversize_factor = 1.0
   END TYPE dbcsr_memtype_type

   !providing pool=Null() explicitly to circumvent bug in ifort 12.1
   TYPE(dbcsr_memtype_type), PARAMETER :: dbcsr_memtype_default = dbcsr_memtype_type(pool=Null())

   TYPE dbcsr_data_area_type
      !! Stores actual data

      INTEGER(KIND=int_4), DIMENSION(:), POINTER    :: i4 => Null()
      INTEGER(KIND=int_8), DIMENSION(:), POINTER    :: i8 => Null()
      REAL(KIND=real_4), DIMENSION(:), POINTER      :: r_sp => Null()
         !! stores real values in single precision
      REAL(KIND=real_8), DIMENSION(:), POINTER      :: r_dp => Null()
         !! stores real values in double precision
      COMPLEX(KIND=real_4), DIMENSION(:), POINTER   :: c_sp => Null()
         !! stores complex values in single precision
      COMPLEX(KIND=real_8), DIMENSION(:), POINTER   :: c_dp => Null()
         !! stores complex values in double precision
      REAL(KIND=real_4), DIMENSION(:, :), POINTER    :: r2_sp => Null()
      REAL(KIND=real_8), DIMENSION(:, :), POINTER    :: r2_dp => Null()
      COMPLEX(KIND=real_4), DIMENSION(:, :), POINTER :: c2_sp => Null()
      COMPLEX(KIND=real_8), DIMENSION(:, :), POINTER :: c2_dp => Null()
      INTEGER                                  :: ref_size = -1
         !! last data element in array actually referenced
      INTEGER                                  :: refcount = -1
         !! reference counter for current structure
      INTEGER                                  :: id = -1
      TYPE(dbcsr_memtype_type)                 :: memory_type = dbcsr_memtype_default
         !! type of memory where data lives
      INTEGER                                  :: data_type = -1
         !! which of the data types is actually used
      TYPE(acc_devmem_type)                    :: acc_devmem
      TYPE(acc_event_type)                     :: acc_ready
   END TYPE dbcsr_data_area_type

   ! Type definitions:
   ! * bit 0: always 1
   ! * bit 1: single (0: 4) vs. double (1: 1)
   ! * bit 2: real (0) vs. complex (1)
   ! * bit 3: dimension (0: 1, 1: 2)
   ! * bit 4: floating point (0) or integer type (1)
   INTEGER, PARAMETER          :: dbcsr_type_real_4 = 1 !001
   INTEGER, PARAMETER          :: dbcsr_type_real_8 = 3 !011
   INTEGER, PARAMETER          :: dbcsr_type_complex_4 = 5 !101
   INTEGER, PARAMETER          :: dbcsr_type_complex_8 = 7 !111

   INTEGER, PARAMETER          :: dbcsr_type_real_4_2d = 9 !1001
   INTEGER, PARAMETER          :: dbcsr_type_real_8_2d = 11 !1011
   INTEGER, PARAMETER          :: dbcsr_type_complex_4_2d = 13 !1101
   INTEGER, PARAMETER          :: dbcsr_type_complex_8_2d = 15 !1111

   INTEGER, PARAMETER          :: dbcsr_type_int_4 = 17 !10001
   INTEGER, PARAMETER          :: dbcsr_type_int_8 = 19 !10011

   INTEGER, PARAMETER          :: dbcsr_type_real_default = dbcsr_type_real_8
   INTEGER, PARAMETER          :: dbcsr_type_complex_default = dbcsr_type_complex_8

CONTAINS

   FUNCTION dbcsr_datatype_sizeof(datatype) RESULT(size)
      !! Helper-routine, returns size of given datatype in terms of bytes.
      INTEGER, INTENT(IN)                                :: datatype
      INTEGER                                            :: size

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbcsr_datatype_sizeof', &
                                     routineP = moduleN//':'//routineN

      SELECT CASE (datatype)
      CASE (dbcsr_type_int_4)
         size = int_4_size
      CASE (dbcsr_type_int_8)
         size = int_8_size
      CASE (dbcsr_type_real_4)
         size = real_4_size
      CASE (dbcsr_type_real_8)
         size = real_8_size
      CASE (dbcsr_type_complex_4)
         size = (2*real_4_size)
      CASE (dbcsr_type_complex_8)
         size = (2*real_8_size)
      CASE default
         DBCSR_ABORT("Invalid data type")
      END SELECT
   END FUNCTION dbcsr_datatype_sizeof

END MODULE dbcsr_data_types
