!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

MODULE dbcsr_cuda_profiling
   !! routines for profiling cuda
   USE ISO_C_BINDING, ONLY: C_CHAR, &
                            C_INT, &
                            C_NULL_CHAR, &
                            C_SIZE_T
   USE dbcsr_kinds, ONLY: default_string_length, &
                          int_8
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_thread_num

   IMPLICIT NONE

   PRIVATE

   PUBLIC  :: cuda_nvtx_init, cuda_nvtx_range_push, cuda_nvtx_range_pop

#if defined( __CUDA_PROFILING )

   INTERFACE
      FUNCTION cuda_nvtx_range_push_dc(message) RESULT(level) &
         BIND(C, name="cuda_nvtx_range_push_cu")
         IMPORT
         CHARACTER(kind=C_CHAR), DIMENSION(*), &
            INTENT(IN)                             :: message
         INTEGER(KIND=C_INT)                      :: level

      END FUNCTION cuda_nvtx_range_push_dc
   END INTERFACE

   INTERFACE
      FUNCTION cuda_nvtx_range_pop_dc() RESULT(level) &
         BIND(C, name="cuda_nvtx_range_pop_cu")
         IMPORT
         INTEGER(KIND=C_INT)                      :: level

      END FUNCTION cuda_nvtx_range_pop_dc
   END INTERFACE

   INTERFACE
      SUBROUTINE cuda_nvtx_name_osthread_cu(name) &
         BIND(C, name="cuda_nvtx_name_osthread_cu")
         IMPORT
         CHARACTER(KIND=C_CHAR), DIMENSION(*)     :: name

      END SUBROUTINE cuda_nvtx_name_osthread_cu
   END INTERFACE

#endif

CONTAINS

   SUBROUTINE cuda_nvtx_init()
#if defined( __CUDA_PROFILING )
      CHARACTER(len=default_string_length)     :: threadname
      INTEGER                                  :: ithread

!$OMP     PARALLEL DEFAULT (NONE), PRIVATE (ithread,threadname)
      ithread = 0
!$    ithread = OMP_GET_THREAD_NUM()
      WRITE (threadname, "(I3,A,I2,A)") ithread
      CALL cuda_nvtx_name_osthread_cu(TRIM(threadname)//c_null_char)
!$OMP     END PARALLEL
#endif
   END SUBROUTINE cuda_nvtx_init

   SUBROUTINE cuda_nvtx_range_push(routineN)
      CHARACTER(LEN=*), INTENT(IN)             :: routineN
#if defined( __CUDA_PROFILING )
      INTEGER                                  :: level
      level = cuda_nvtx_range_push_dc(TRIM(routineN)//CHAR(0))
#else
      CALL dbcsr_abort(__LOCATION__, "cuda_nvtx_range_push: "// &
                       "__CUDA_PROFILING not compiled in, but called with:"//TRIM(routineN))
#endif
   END SUBROUTINE cuda_nvtx_range_push

   SUBROUTINE cuda_nvtx_range_pop()
#if defined( __CUDA_PROFILING )
      INTEGER                                  :: level
      level = cuda_nvtx_range_pop_dc()
#else
      DBCSR_ABORT("cuda_nvtx_range_push: __CUDA_PROFILING not compiled in.")
#endif
   END SUBROUTINE cuda_nvtx_range_pop

END MODULE dbcsr_cuda_profiling
